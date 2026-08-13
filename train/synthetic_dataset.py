"""
Synthetic design-loop dataset (roadmap step 6, track C)
========================================================
Generates parametric buildings whose column layout is engineering-valid BY
CONSTRUCTION, so the GNN can learn column patterns the StructGAN shear-wall
corpus cannot teach:

  - a TRUE structural grid (known, not detected) with spans in 3.5-7 m
  - columns at every grid intersection not already supported by a wall
    -> interior OPEN-SPACE columns = positive labels on virtual candidate
       nodes (the StructGAN corpus has virtually none of these)
  - optional shear-wall core (stair/elevator box) + braced perimeter piers
  - door/window gaps so Stage 3's opening detection sees realistic graphs

Distribution match: the synthetic plan is RENDERED to a binary wall mask at
10 mm/px and pushed through the SAME Stage 2 -> consolidate -> Stage 3
pipeline as training/inference (unified_dataset).  Labels are then attached
geometrically from the known design.

Usage (standalone smoke test):
  python train/synthetic_dataset.py --n 3
In training:
  python train/train_gnn_multi.py --synthetic 150
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
from unified_dataset import _floor_to_dict                     # noqa: E402

logger = logging.getLogger(__name__)

SCALE = 0.01          # m per px (same as StructGAN conversion)
COL_SNAP_M = 0.50
CACHE_VERSION = 1


# =============================================================================
# 1. Parametric plan generation
# =============================================================================

def _jitter_axes(total, span_lo, span_hi, rng):
    """Grid axis positions 0..total with spans in [span_lo, span_hi]."""
    axes = [0.0]
    while total - axes[-1] > span_hi:
        axes.append(axes[-1] + rng.uniform(span_lo, span_hi))
    axes.append(total)
    if axes[-1] - axes[-2] < span_lo and len(axes) > 2:
        axes.pop(-2)
    return axes


def gen_plan(rng: np.random.Generator) -> dict:
    """
    One synthetic building.

    Returns dict:
      walls   : [(p1, p2, thickness_m, is_sw)]
      columns : [(x, y)]  ground-truth column positions
      grid_x, grid_y : true axis positions
      W, H    : plan dimensions
    """
    W = rng.uniform(12.0, 28.0)
    H = rng.uniform(9.0, 18.0)
    gx = _jitter_axes(W, 3.5, 7.0, rng)
    gy = _jitter_axes(H, 3.5, 7.0, rng)

    walls = []   # (p1, p2, t, is_sw)

    def _add_wall_run(a, b, t, is_sw, rng, gap_prob=0.5):
        """Wall from a to b with 0-2 door/window gaps."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 0.6:
            return
        cuts = []
        if L > 3.0 and rng.random() < gap_prob:
            n_gaps = 1 + int(rng.random() < 0.4)
            for _ in range(n_gaps):
                g0 = rng.uniform(0.15, 0.75)
                gw = rng.uniform(0.9, 1.6) / L
                cuts.append((g0, min(g0 + gw, 0.92)))
        cuts.sort()
        t0 = 0.0
        for c0, c1 in cuts:
            if c0 - t0 > 0.4 / L:
                walls.append((a + t0 * (b - a), a + c0 * (b - a), t, is_sw))
            t0 = max(t0, c1)
        if 1.0 - t0 > 0.4 / L:
            walls.append((a + t0 * (b - a), b, t, is_sw))

    # -- perimeter (thick, occasionally shear) --------------------------------
    per_t = rng.uniform(0.20, 0.30)
    corners = [(0, 0), (W, 0), (W, H), (0, H)]
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        _add_wall_run(a, b, per_t, False, rng, gap_prob=0.8)

    # -- shear-wall core (stair/elevator box), most plans ---------------------
    sw_segs = []
    if rng.random() < 0.75:
        cw = rng.uniform(2.4, 4.0)
        ch = rng.uniform(2.4, 4.0)
        cx = rng.uniform(1.0, max(1.01, W - cw - 1.0))
        cy = rng.uniform(1.0, max(1.01, H - ch - 1.0))
        core_t = rng.uniform(0.20, 0.30)
        box = [((cx, cy), (cx + cw, cy)), ((cx + cw, cy), (cx + cw, cy + ch)),
               ((cx + cw, cy + ch), (cx, cy + ch)), ((cx, cy + ch), (cx, cy))]
        for k, (a, b) in enumerate(box):
            has_gap = (k == int(rng.integers(4)))   # one door into the core
            _add_wall_run(a, b, core_t, True, rng,
                          gap_prob=1.0 if has_gap else 0.0)
        sw_segs += [(np.array(a, float), np.array(b, float)) for a, b in box]

    # -- a few perimeter SW piers for bidirectional resistance ----------------
    for orient in ("H", "V"):
        for _ in range(int(rng.integers(1, 3))):
            L = rng.uniform(1.8, 3.5)
            if orient == "H":
                x0 = rng.uniform(0, W - L)
                y = float(rng.choice([0.0, H]))
                a, b = np.array([x0, y]), np.array([x0 + L, y])
            else:
                y0 = rng.uniform(0, H - L)
                x = float(rng.choice([0.0, W]))
                a, b = np.array([x, y0]), np.array([x, y0 + L])
            walls.append((a, b, rng.uniform(0.20, 0.30), True))
            sw_segs.append((a, b))

    # -- interior partition walls along some grid lines -----------------------
    for x in gx[1:-1]:
        if rng.random() < 0.55:
            y0 = rng.uniform(0, H * 0.4)
            y1 = y0 + rng.uniform(0.3, 0.7) * (H - y0)
            _add_wall_run((x, y0), (x, y1), rng.uniform(0.10, 0.15),
                          False, rng)
    for y in gy[1:-1]:
        if rng.random() < 0.55:
            x0 = rng.uniform(0, W * 0.4)
            x1 = x0 + rng.uniform(0.3, 0.7) * (W - x0)
            _add_wall_run((x0, y), (x1, y), rng.uniform(0.10, 0.15),
                          False, rng)

    # -- ground-truth columns: unsupported grid intersections -----------------
    def _near_sw(pt, tol=0.35):
        return any(_pt_seg(pt, a, b) <= tol for a, b in sw_segs)

    columns = []
    for x in gx:
        for y in gy:
            pt = np.array([x, y], float)
            if _near_sw(pt):
                continue
            columns.append((float(x), float(y)))

    return dict(walls=walls, columns=columns, grid_x=gx, grid_y=gy,
                W=W, H=H, sw_segs=sw_segs)


def _pt_seg(pt, a, b):
    ab = np.asarray(b, float) - np.asarray(a, float)
    dn = float(np.dot(ab, ab))
    if dn < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = np.clip(np.dot(pt - np.asarray(a, float), ab) / dn, 0.0, 1.0)
    return float(np.linalg.norm(pt - (a + t * ab)))


# =============================================================================
# 2. Rasterize -> real pipeline -> labelled sample
# =============================================================================

def render_mask(plan, *, scale=SCALE, margin_m=0.5):
    import cv2
    W, H = plan["W"], plan["H"]
    w_px = int((W + 2 * margin_m) / scale)
    h_px = int((H + 2 * margin_m) / scale)
    img = np.zeros((h_px, w_px), np.uint8)
    off = margin_m
    for a, b, t, _sw in plan["walls"]:
        p1 = (int((a[0] + off) / scale), int((a[1] + off) / scale))
        p2 = (int((b[0] + off) / scale), int((b[1] + off) / scale))
        cv2.line(img, p1, p2, 255, max(2, int(round(t / scale))))
    return img, off


def convert_synthetic(plan) -> dict | None:
    """plan -> multi-task sample (same schema as unified_dataset)."""
    mask, off = render_mask(plan)
    try:
        floor = _floor_to_dict(VectorizationPipeline(scale=SCALE).run(mask))
        G = s3.build_graph(floor, openings=None,
                           variant=s3.GraphVariant.EDGE_PSW_DW,
                           proximity_tol_m=0.10, auto_detect_gaps=True,
                           classify_mode="fixed", psw_min_thickness=0.06)
    except Exception as e:
        logger.warning("synthetic pipeline failed: %s", e)
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

    # -- SW edge labels from the DESIGNED shear walls (offset to mask coords) --
    sw_segs = [(np.asarray(a) + off, np.asarray(b) + off)
               for a, b in plan["sw_segs"]]
    # ratio labels via left/right coverage runs (same construction as the
    # StructGAN B-image sampling) so piers merged into longer edges by the
    # vectorizer still get correct PARTIAL shear-wall ratios.
    def _sw_hit(pt):
        return bool(sw_segs) and min(_pt_seg(pt, a, b)
                                     for a, b in sw_segs) < 0.25

    sw_labels = np.zeros((E, 2), dtype=np.float32)
    for k in range(E):
        if not psw_mask[k]:
            continue
        u, v = int(ei[0, k]), int(ei[1, k])
        hits = [_sw_hit(xy_m[u] + t * (xy_m[v] - xy_m[u]))
                for t in np.linspace(0.0, 1.0, 25)]
        left = 0.0
        for i, hv in enumerate(hits):
            if hv:
                left = (i + 1) / len(hits)
            else:
                break
        right = 0.0
        for i in range(len(hits) - 1, -1, -1):
            if hits[i]:
                right = (len(hits) - i) / len(hits)
            else:
                break
        if left + right > 1.0:
            left = right = 0.5
        # end-anchored runs miss piers that sit mid-edge or overhang by a
        # sample step: fall back to total coverage split over both ends.
        if left + right == 0.0:
            cov = sum(hits) / len(hits)
            if cov > 0.30:
                left = right = min(cov / 2, 0.5)
        sw_labels[k] = (left, right)

    # -- virtual candidates: rule candidates + the TRUE grid ------------------
    segs = []
    for k in range(E):
        if not psw_mask[k]:
            continue
        u, v = int(ei[0, k]), int(ei[1, k])
        segs.append(sg.Segment(xy_m[u].copy(), xy_m[v].copy(),
                               float(eth[k]),
                               bool(sw_labels[k].sum() > 0), str(k)))
    candidates = []
    try:
        candidates = sg.place_columns(segs, pier_end_columns=False).candidates
    except Exception:
        pass
    candidates += [(x + off, y + off)
                   for x in plan["grid_x"] for y in plan["grid_y"]]

    xy2, ei2, ecl2, eth2, vmask = add_virtual_candidates(
        xy_m, ei, ecl, eth, candidates)
    E2 = ei2.shape[1]
    sw2 = np.zeros((E2, 2), dtype=np.float32)
    sw2[:E] = sw_labels
    psw2 = np.zeros(E2, dtype=bool)
    psw2[:E] = psw_mask

    # -- column labels from the DESIGN (not from rules) ------------------------
    col_labels = np.zeros(xy2.shape[0], dtype=np.float32)
    n_hit = 0
    for (x, y) in plan["columns"]:
        pt = np.array([x + off, y + off])
        d = np.linalg.norm(xy2 - pt, axis=1)
        j = int(np.argmin(d))
        if d[j] <= COL_SNAP_M:
            col_labels[j] = 1.0
            n_hit += 1
    if n_hit < max(2, len(plan["columns"]) // 3):
        return None     # graph too degenerate to carry the design

    return dict(xy_m=xy2, edge_index=ei2, edge_class=ecl2, edge_thk=eth2,
                sw_labels=sw2, psw_mask=psw2, col_labels=col_labels,
                virtual_mask=vmask)


# =============================================================================
# 3. Dataset generation (cached)
# =============================================================================

def generate_dataset(n: int, *, seed=0, cache_path=None, rebuild=False):
    if cache_path and not rebuild and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("version") == CACHE_VERSION and len(blob["samples"]) >= n:
            logger.info("Loaded %d cached synthetic samples from %s",
                        len(blob["samples"][:n]), cache_path)
            return blob["samples"][:n]

    for name in ("stage_2", "stage_3", "structural_grid"):
        logging.getLogger(name).setLevel(logging.WARNING)

    rng = np.random.default_rng(seed)
    samples, tries = [], 0
    while len(samples) < n and tries < n * 3:
        tries += 1
        s = convert_synthetic(gen_plan(rng))
        if s is None:
            continue
        s["name"] = f"synth_{seed}_{len(samples):04d}"
        s["group"] = "synthetic"
        samples.append(s)
        if len(samples) % 25 == 0:
            logger.info("  synthetic: %d/%d generated", len(samples), n)
    logger.info("Generated %d synthetic samples (%d attempts)",
                len(samples), tries)

    if cache_path and samples:
        with open(cache_path, "wb") as f:
            pickle.dump(dict(version=CACHE_VERSION, samples=samples), f)
        logger.info("Cached -> %s", cache_path)
    return samples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Synthetic dataset smoke test")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        plan = gen_plan(rng)
        s = convert_synthetic(plan)
        if s is None:
            print(f"plan {i}: conversion FAILED")
            continue
        n_vpos = int((s["col_labels"] * s["virtual_mask"]).sum())
        print(f"plan {i}: {plan['W']:.1f}x{plan['H']:.1f}m "
              f"grid {len(plan['grid_x'])}x{len(plan['grid_y'])} "
              f"design_cols={len(plan['columns'])} | graph "
              f"N={s['xy_m'].shape[0]} (virt={int(s['virtual_mask'].sum())}) "
              f"E={s['edge_index'].shape[1]} SW={int((s['sw_labels'].sum(1)>0).sum())} "
              f"col_labels={int(s['col_labels'].sum())} "
              f"(on virtual: {n_vpos})")
