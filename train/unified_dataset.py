"""
Unified dataset conversion (roadmap step 4)
============================================
Builds TRAINING graphs with the SAME Stage 2 -> consolidate -> Stage 3
pipeline that inference uses, killing the train/inference domain gap that
collapsed the SW edge head (legacy training used a separate skeleton+Hough
path with different segment statistics).

Per StructGAN pair:
  A-image -> binary wall mask (color-derived; stands in for Stage 1 output)
          -> VectorizationPipeline (Stage 2) -> consolidate_wall_rects
          -> stage_3.build_graph (EDGE_PSW_DW, auto gap detection,
             openings=None -- exactly like inference)
  B-image -> SW mask -> edge ratio labels sampled along the REAL graph edges
  structural_grid rules on the resulting segments -> column pseudo-labels

Sample schema is identical to train_gnn_multi.convert_pair_multi, so the
training / evaluation code is unchanged.  Conversion is slow (~1 s/image),
so results are pickled to a cache file.
"""

from __future__ import annotations
import logging, os, pickle, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE,
           os.path.join(_ROOT, "stages", "stage_2"),
           os.path.join(_ROOT, "stages", "stage_3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.gnn_ep import add_virtual_candidates, ET_PSW      # noqa: E402
import structural_grid as sg                                   # noqa: E402
import stage_3 as s3                                           # noqa: E402
from stage_2 import VectorizationPipeline, consolidate_wall_rects  # noqa: E402

logger = logging.getLogger(__name__)

SCALE_M_PER_PX = 0.01      # StructGAN: 10 mm / pixel
COL_SNAP_M = 0.50
CACHE_VERSION = 1

# StructGAN semantic colors (BGR)
_COLOR_PSW = np.array([152, 152, 152], dtype=np.int16)
_COLOR_PW = np.array([128, 128, 128], dtype=np.int16)
_COLOR_SW = np.array([0, 0, 255], dtype=np.int16)
_TOL = 15


def _match(img_bgr, color):
    return (np.abs(img_bgr.astype(np.int16) - color) < _TOL).all(axis=2)


def wall_mask_from_semantic(img_bgr) -> np.ndarray:
    """Binary wall mask (PSW + partition walls) — the Stage 1 stand-in."""
    m = _match(img_bgr, _COLOR_PSW) | _match(img_bgr, _COLOR_PW)
    return (m * 255).astype(np.uint8)


def _floor_to_dict(floor) -> dict:
    """Serialize a stage_2 Floor object (mirrors export_json, no file I/O)."""
    data = {
        "floor_id": floor.floor_id,
        "scale_m_per_px": floor.scale,
        "floor_height_m": floor.height,
        "offset": {"x": floor.offset_x, "y": floor.offset_y},
        "walls": [],
    }
    for wall in floor.walls:
        data["walls"].append({
            "wall_id": wall.id,
            "rejected": wall.rejected,
            "rejection_reason": wall.rejection_reason,
            "points": [{"id": p.id, "x": round(p.x, 4), "y": round(p.y, 4)}
                       for p in wall.points.values()],
            "rects": [{"id": r.id, "p1_id": r.p1_id, "p2_id": r.p2_id,
                       "thickness_m": r.thickness, "length_m": r.length,
                       "angle_rad": r.angle}
                      for r in wall.rects.values()],
        })
    # NOTE: no auto_calibrate_floor here -- training inputs (StructGAN,
    # synthetic renders) are known-scale by construction, and rescaling
    # would desynchronize the geometry from the attached labels.
    return consolidate_wall_rects(data)


def _sw_ratios_for_edge(p1_m, p2_m, sw_mask, *, scale=SCALE_M_PER_PX,
                        n_samples=50):
    """[ratio_left, ratio_right] of SW coverage along a metre-space edge."""
    h, w = sw_mask.shape
    p1 = np.asarray(p1_m) / scale
    p2 = np.asarray(p2_m) / scale
    hits = []
    for t in np.linspace(0.0, 1.0, n_samples):
        pt = p1 + t * (p2 - p1)
        ix, iy = int(round(pt[0])), int(round(pt[1]))
        hit = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sy, sx = iy + dy, ix + dx
                if 0 <= sy < h and 0 <= sx < w and sw_mask[sy, sx] > 127:
                    hit = True
                    break
            if hit:
                break
        hits.append(hit)

    left = 0.0
    for i, hval in enumerate(hits):
        if hval:
            left = (i + 1) / n_samples
        else:
            break
    right = 0.0
    for i in range(len(hits) - 1, -1, -1):
        if hits[i]:
            right = (n_samples - i) / n_samples
        else:
            break
    if left + right > 1.0:
        left = right = 0.5
    return left, right


def convert_pair_pipeline(img_a_path: str, img_b_path: str):
    """One StructGAN pair -> multi-task sample via the REAL pipeline."""
    import cv2
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    if img_a is None or img_b is None:
        return None
    h, w = img_a.shape[:2]
    if w > 1.5 * h:
        img_a = img_a[:, : w // 2]
        img_b = img_b[:, : w // 2]

    # --- Stage 1 stand-in + Stage 2 + consolidate ---------------------------
    mask = wall_mask_from_semantic(img_a)
    if int((mask > 0).sum()) < 200:
        return None
    try:
        floor_obj = VectorizationPipeline(scale=SCALE_M_PER_PX).run(mask)
        floor = _floor_to_dict(floor_obj)
    except Exception as e:
        logger.warning("Stage 2 failed on %s: %s", os.path.basename(img_a_path), e)
        return None

    # --- Stage 3 (EXACTLY like inference: no openings, auto gaps) -----------
    try:
        G = s3.build_graph(floor, openings=None,
                           variant=s3.GraphVariant.EDGE_PSW_DW,
                           proximity_tol_m=0.10, auto_detect_gaps=True,
                           classify_mode="fixed", psw_min_thickness=0.06)
    except Exception as e:
        logger.warning("Stage 3 failed on %s: %s", os.path.basename(img_a_path), e)
        return None
    if G.number_of_edges() < 4:
        return None

    node_ids = sorted(G.nodes())
    n2i = {n: i for i, n in enumerate(node_ids)}
    xy_m = np.array([[G.nodes[n].get("x_m", 0.0), G.nodes[n].get("y_m", 0.0)]
                     for n in node_ids], dtype=np.float64)

    edges = list(G.edges(data=True))
    E = len(edges)
    ei = np.zeros((2, E), dtype=np.int64)
    ecl = np.zeros(E, dtype=int)
    eth = np.zeros(E)
    for k, (u, v, d) in enumerate(edges):
        ei[0, k], ei[1, k] = n2i[u], n2i[v]
        ecl[k] = min(int(d.get("edge_type", 0)), 2)
        eth[k] = float(d.get("thickness_m", 0.0))
    psw_mask = ecl == ET_PSW
    if not psw_mask.any():
        return None

    # --- edge labels from the B-image SW mask --------------------------------
    sw_mask = (_match(img_b, _COLOR_SW) * 255).astype(np.uint8)
    sw_labels = np.zeros((E, 2), dtype=np.float32)
    for k in range(E):
        if not psw_mask[k]:
            continue
        u, v = int(ei[0, k]), int(ei[1, k])
        sw_labels[k] = _sw_ratios_for_edge(xy_m[u], xy_m[v], sw_mask)

    # --- column pseudo-labels (rule engine on the pipeline segments) ---------
    segs = []
    edge_is_sw = sw_labels.sum(axis=1) > 0
    for k in range(E):
        if not psw_mask[k]:
            continue
        u, v = int(ei[0, k]), int(ei[1, k])
        segs.append(sg.Segment(xy_m[u].copy(), xy_m[v].copy(),
                               float(eth[k]), bool(edge_is_sw[k]), str(k)))
    if not segs:
        return None
    try:
        grid = sg.place_columns(segs)
    except Exception:
        return None

    xy2, ei2, ecl2, eth2, vmask = add_virtual_candidates(
        xy_m, ei, ecl, eth, grid.candidates)
    E2 = ei2.shape[1]
    sw2 = np.zeros((E2, 2), dtype=np.float32)
    sw2[:E] = sw_labels
    psw2 = np.zeros(E2, dtype=bool)
    psw2[:E] = psw_mask

    col_labels = np.zeros(xy2.shape[0], dtype=np.float32)
    for c in grid.columns:
        d = np.linalg.norm(xy2 - np.array([c.x, c.y]), axis=1)
        j = int(np.argmin(d))
        if d[j] <= COL_SNAP_M:
            col_labels[j] = 1.0

    return dict(xy_m=xy2, edge_index=ei2, edge_class=ecl2, edge_thk=eth2,
                sw_labels=sw2, psw_mask=psw2, col_labels=col_labels,
                virtual_mask=vmask)


def convert_dataset_pipeline(data_dir: str, split: str,
                             groups=("L1_7", "L2_7", "L1L2_8"),
                             cache_path: str | None = None,
                             rebuild: bool = False):
    """Convert (with pickle cache) all pairs of a split via the real pipeline."""
    if cache_path and not rebuild and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("version") == CACHE_VERSION:
            logger.info("Loaded %d cached samples (%s) from %s",
                        len(blob["samples"]), split, cache_path)
            return blob["samples"]
        logger.info("Cache version mismatch -- rebuilding %s", cache_path)

    # Quiet the very chatty stage 2/3 loggers during bulk conversion
    for name in ("stage_2", "stage_3", "structural_grid"):
        logging.getLogger(name).setLevel(logging.WARNING)

    samples = []
    for grp in groups:
        dir_a = os.path.join(data_dir, grp, f"{split}_A")
        dir_b = os.path.join(data_dir, grp, f"{split}_B")
        if not os.path.isdir(dir_a):
            logger.warning("Missing: %s", dir_a)
            continue
        files = sorted(f for f in os.listdir(dir_a) if f.endswith(".png"))
        for i, fname in enumerate(files):
            pa, pb = os.path.join(dir_a, fname), os.path.join(dir_b, fname)
            if not os.path.exists(pb):
                continue
            s = convert_pair_pipeline(pa, pb)
            if s is None:
                logger.warning("Failed: %s/%s", grp, fname)
                continue
            s["name"], s["group"] = fname, grp
            samples.append(s)
            logger.info("  [%s %d/%d] %s: %d nodes (%d virt), %d edges "
                        "(%d PSW, %d SW), %d cols",
                        grp, i + 1, len(files), fname,
                        s["xy_m"].shape[0], int(s["virtual_mask"].sum()),
                        s["edge_index"].shape[1], int(s["psw_mask"].sum()),
                        int((s["sw_labels"].sum(1) > 0).sum()),
                        int(s["col_labels"].sum()))
    logger.info("Pipeline-converted %d samples (%s)", len(samples), split)

    if cache_path and samples:
        with open(cache_path, "wb") as f:
            pickle.dump(dict(version=CACHE_VERSION, samples=samples), f)
        logger.info("Cached -> %s", cache_path)
    return samples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Unified dataset smoke test")
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "data"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    d = os.path.join(args.data_dir, "L1_7")
    files = sorted(os.listdir(os.path.join(d, f"{args.split}_A")))[: args.limit]
    for f in files:
        s = convert_pair_pipeline(os.path.join(d, f"{args.split}_A", f),
                                  os.path.join(d, f"{args.split}_B", f))
        if s is None:
            print(f"{f}: FAILED")
            continue
        print(f"{f}: N={s['xy_m'].shape[0]} (virt={int(s['virtual_mask'].sum())}) "
              f"E={s['edge_index'].shape[1]} PSW={int(s['psw_mask'].sum())} "
              f"SW={int((s['sw_labels'].sum(1) > 0).sum())} "
              f"cols={int(s['col_labels'].sum())}")
