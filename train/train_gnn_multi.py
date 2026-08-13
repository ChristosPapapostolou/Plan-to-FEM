"""
Multi-task GNN training: shear walls (edges) + columns (nodes)
===============================================================
Rule-distillation stage (roadmap §3.2-A):

  edge labels : ground-truth SW ratios from StructGAN B-images (as before)
  node labels : column pseudo-labels from the structural_grid rule engine
                (grid + span + pier-end rules run on the ground-truth SW
                 layout), snapped to graph nodes

The model (models/gnn_ep.GNNEPMulti) therefore learns to predict BOTH the
shear-wall layout and the column layout end-to-end.  The rules remain only
as teacher (here) and runtime validator (Stage 4/5).

Usage:
  python train/train_gnn_multi.py --epochs 150
  python train/train_gnn_multi.py --data-dir train/data --device cuda
"""

from __future__ import annotations
import argparse, logging, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "stages", "stage_3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.gnn_ep import (                                   # noqa: E402
    HAS_TORCH, D_NODE, D_EDGE, ET_PSW, ET_VIRTUAL,
    build_node_features, build_edge_features, add_virtual_candidates,
    normalize_graph, nms_points, match_points,
)
import stages.stage_3.structural_grid as sg                                   # noqa: E402
import train_gnn as tg                                         # noqa: E402

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from models.gnn_ep import GNNEPMulti, save_checkpoint

logger = logging.getLogger(__name__)

MM_PER_PX = 10.0
COL_SNAP_M = 0.50      # column pseudo-label -> node snap tolerance


# =============================================================================
# 1. Sample preparation
# =============================================================================

def convert_pair_multi(img_a_path: str, img_b_path: str):
    """
    One StructGAN pair -> multi-task sample dict:
      xy_m (N,2)  edge_index (2,E)  edge_class (E,)  edge_thk (E,)
      sw_labels (E,2)  psw_mask (E,)  col_labels (N,)  virtual_mask (N,)
    """
    import cv2
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    if img_a is None or img_b is None:
        return None
    h, w = img_a.shape[:2]
    if w > 1.5 * h:
        img_a = img_a[:, : w // 2]
        img_b = img_b[:, : w // 2]

    g = tg.image_to_graph(img_a, scale_mm_per_px=MM_PER_PX)
    if g is None:
        return None
    sw_mask = tg.extract_masks(img_b)["sw"]
    sw_labels = tg.compute_sw_ratios(g, sw_mask)          # (E, 2)

    # --- metric geometry -----------------------------------------------------
    xy_m = g["node_coords_px"] * MM_PER_PX / 1000.0        # (N, 2) metres
    ei = g["edge_index"]                                     # (2, E)
    E = ei.shape[1]

    # edge class + thickness re-derived from the raw GraphEdge list.  The
    # k-th surviving edge in edge_index corresponds to the k-th GraphEdge
    # that was not degenerate; rebuild that mapping here.
    edge_class = np.zeros(E, dtype=int)
    edge_thk = np.zeros(E)
    kept = 0
    canonical, node_map = g["canonical"], g["node_map"]
    for k, ge in enumerate(g["edges"]):
        ni1 = node_map[canonical[2 * k]]
        ni2 = node_map[canonical[2 * k + 1]]
        if ni1 == ni2:
            continue
        edge_class[kept] = ge.edge_type
        edge_thk[kept] = ge.thickness_px * MM_PER_PX / 1000.0
        kept += 1
    assert kept == E, f"edge bookkeeping mismatch ({kept} != {E})"

    psw_mask = edge_class == ET_PSW
    if not psw_mask.any():
        return None

    # --- column pseudo-labels from the rule engine ---------------------------
    segs = []
    edge_is_sw = sw_labels.sum(axis=1) > 0
    for k in range(E):
        if edge_class[k] != ET_PSW:
            continue
        u, v = int(ei[0, k]), int(ei[1, k])
        segs.append(sg.Segment(xy_m[u].copy(), xy_m[v].copy(),
                               float(edge_thk[k]), bool(edge_is_sw[k]), str(k)))
    if not segs:
        return None
    try:
        grid = sg.place_columns(segs)
    except Exception:
        return None

    # --- virtual candidate nodes (grid intersections in open space) ---------
    xy2, ei2, ecl2, eth2, vmask = add_virtual_candidates(
        xy_m, ei, edge_class, edge_thk, grid.candidates)

    # sw labels for appended virtual edges are zero / non-PSW
    E2 = ei2.shape[1]
    sw2 = np.zeros((E2, 2), dtype=np.float32)
    sw2[:E] = sw_labels
    psw2 = np.zeros(E2, dtype=bool)
    psw2[:E] = psw_mask

    # --- node labels: 1 where a rule-placed column snaps to a node ----------
    col_labels = np.zeros(xy2.shape[0], dtype=np.float32)
    for c in grid.columns:
        d = np.linalg.norm(xy2 - np.array([c.x, c.y]), axis=1)
        j = int(np.argmin(d))
        if d[j] <= COL_SNAP_M:
            col_labels[j] = 1.0

    return dict(xy_m=xy2, edge_index=ei2, edge_class=ecl2, edge_thk=eth2,
                sw_labels=sw2, psw_mask=psw2, col_labels=col_labels,
                virtual_mask=vmask)


def convert_dataset(data_dir: str, split: str,
                    groups=("L1_7", "L2_7", "L1L2_8")):
    samples = []
    for grp in groups:
        dir_a = os.path.join(data_dir, grp, f"{split}_A")
        dir_b = os.path.join(data_dir, grp, f"{split}_B")
        if not os.path.isdir(dir_a):
            logger.warning("Missing: %s", dir_a)
            continue
        for fname in sorted(f for f in os.listdir(dir_a) if f.endswith(".png")):
            pa, pb = os.path.join(dir_a, fname), os.path.join(dir_b, fname)
            if not os.path.exists(pb):
                continue
            s = convert_pair_multi(pa, pb)
            if s is None:
                logger.warning("Failed: %s/%s", grp, fname)
                continue
            s["name"], s["group"] = fname, grp
            samples.append(s)
            logger.info("  %s/%s: %d nodes (%d virtual), %d edges, "
                        "%d SW edges, %d column nodes",
                        grp, fname, s["xy_m"].shape[0],
                        int(s["virtual_mask"].sum()),
                        s["edge_index"].shape[1],
                        int((s["sw_labels"].sum(1) > 0).sum()),
                        int(s["col_labels"].sum()))
    logger.info("Converted %d samples (%s)", len(samples), split)
    return samples


# =============================================================================
# 2. Augmentation + featurization (features rebuilt per transform)
# =============================================================================

def featurize(sample, *, flip=False, rot=0, jitter=0.0, rng=None):
    xy = sample["xy_m"].copy()
    if flip:
        xy[:, 1] *= -1
    if rot:
        th = np.radians(rot)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        xy = xy @ R.T
    if jitter > 0 and rng is not None:
        xy += rng.normal(0, jitter, xy.shape)

    nf, center, scale = build_node_features(
        xy, sample["edge_index"], sample["edge_thk"],
        sample["edge_class"], sample["virtual_mask"])
    ef = build_edge_features(xy, sample["edge_index"],
                             sample["edge_class"], center, scale)
    return nf, ef


# =============================================================================
# 3. Training
# =============================================================================

def train(train_samples, val_samples, *, epochs=150, lr=1e-3,
          weight_decay=1e-4, augment=4, device="cpu",
          save_path="gnn_multi.pt", node_loss_w=1.0, pos_weight_cap=8.0):
    model = GNNEPMulti().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(42)

    # global column pos_weight
    n_pos = sum(float(s["col_labels"].sum()) for s in train_samples)
    n_all = sum(s["col_labels"].size for s in train_samples)
    # Cap the class weight: the raw imbalance (~19x) drives the node head
    # into recall-at-all-costs (~2.3 predictions per label column).
    pw = min((n_all - n_pos) / max(n_pos, 1.0), pos_weight_cap)
    pos_w = torch.tensor([pw], device=device)
    logger.info("Column nodes: %d / %d (pos_weight=%.1f, cap=%.1f)",
                int(n_pos), n_all, pos_w.item(), pos_weight_cap)

    best_score, best_thresh = -1.0, 0.5
    flips = [False, True]
    rots = [0, 90, 180, 270]

    for ep in range(1, epochs + 1):
        model.train()
        tot, n_seen = 0.0, 0
        order = rng.permutation(len(train_samples))
        for si in order:
            s = train_samples[si]
            for _ in range(augment):
                nf, ef = featurize(s, flip=bool(rng.integers(2)),
                                   rot=int(rng.choice(rots)),
                                   jitter=0.02, rng=rng)
                x = torch.tensor(nf, device=device)
                ei = torch.tensor(s["edge_index"], device=device)
                ea = torch.tensor(ef, device=device)
                y_sw = torch.tensor(s["sw_labels"], device=device)
                y_col = torch.tensor(s["col_labels"], device=device)
                psw = torch.tensor(s["psw_mask"], device=device)

                e_out, n_logit = model(x, ei, ea)

                loss = torch.zeros((), device=device)
                if psw.any():
                    p, t = e_out[psw], y_sw[psw]
                    is_sw = (t.sum(-1) > 0).float()
                    l1 = (p[:, :2] - t).abs().mean()
                    # dedicated SW logit -- no conflict with the ratio targets
                    bce_e = F.binary_cross_entropy_with_logits(p[:, 2], is_sw)
                    loss = loss + l1 + bce_e
                bce_n = F.binary_cross_entropy_with_logits(
                    n_logit.squeeze(-1), y_col, pos_weight=pos_w)
                loss = loss + node_loss_w * bce_n

                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                n_seen += 1
        sched.step()

        # ---- validation ----------------------------------------------------
        if val_samples:
            m = evaluate(model, val_samples, device=device)
            score = m["edge_iou"] + m["node_f1"]
            if score > best_score:
                best_score = score
                best_thresh = m["node_thresh"]
                save_checkpoint(model, save_path,
                                node_threshold=best_thresh,
                                edge_threshold=m["edge_thresh"],
                                meta=dict(epoch=ep, **m))
            if ep % 5 == 0 or ep == 1:
                msg = (f"Ep {ep:>3}/{epochs} loss={tot/max(n_seen,1):.4f} "
                       f"edge_IoU={m['edge_iou']:.3f}@{m['edge_thresh']:.2f} "
                       f"col P={m['node_p']:.3f} R={m['node_r']:.3f} "
                       f"F1={m['node_f1']:.3f}@{m['node_thresh']:.2f} "
                       f"best={best_score:.3f}")
                print(msg)
                logger.info(msg)
    print(f"\nDone. Best (edge_IoU + node_F1) = {best_score:.3f}, "
          f"node threshold = {best_thresh:.2f}\nSaved: {save_path}")
    return model


def evaluate(model, samples, device="cpu"):
    model.eval()
    with torch.no_grad():
        return _evaluate_inner(model, samples, device)


EDGE_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
NODE_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
NMS_DIST_M = 0.60      # spatial NMS radius for predicted columns
MATCH_TOL_M = 0.50     # layout-level matching tolerance


def _evaluate_inner(model, samples, device):
    """
    Layout-level evaluation:
      edge head : SW IoU with threshold sweep (scores are not calibrated
                  to 0.5 -- fixed-threshold IoU flaps between runs)
      node head : spatial NMS + distance-tolerant point matching.  Graph
                  nodes sit centimetres apart, so exact-node matching
                  massively understates precision.
    """
    e_stats = {t: [0, 0] for t in EDGE_THRESHOLDS}       # inter, union
    n_stats = {t: [0, 0, 0] for t in NODE_THRESHOLDS}    # tp, fp, fn
    for s in samples:
        nf, ef = featurize(s)
        x = torch.tensor(nf, device=device)
        ei = torch.tensor(s["edge_index"], device=device)
        ea = torch.tensor(ef, device=device)
        e_out, n_logit = model(x, ei, ea)

        psw = torch.tensor(s["psw_mask"], device=device)
        if psw.any():
            p = e_out[psw]
            t_lab = torch.tensor(s["sw_labels"], device=device)[psw]
            gt = (t_lab.sum(-1) > 0)
            sw_score = torch.sigmoid(p[:, 2])
            for t in EDGE_THRESHOLDS:
                pr = sw_score > t
                e_stats[t][0] += (pr & gt).sum().item()
                e_stats[t][1] += (pr | gt).sum().item()

        probs = torch.sigmoid(n_logit.squeeze(-1)).cpu().numpy()
        xy = s["xy_m"]
        label_xy = xy[s["col_labels"] > 0.5]
        for t in n_stats:
            sel = np.where(probs > t)[0]
            if sel.size:
                keep = nms_points(xy[sel], probs[sel], min_dist=NMS_DIST_M)
                pred_xy = xy[sel][keep]
            else:
                pred_xy = np.zeros((0, 2))
            tp, fp, fn = match_points(pred_xy, label_xy, tol=MATCH_TOL_M)
            n_stats[t][0] += tp
            n_stats[t][1] += fp
            n_stats[t][2] += fn

    edge_iou, edge_thresh = 0.0, 0.5
    for t, (i, u) in e_stats.items():
        iou = i / max(u, 1)
        if iou > edge_iou:
            edge_iou, edge_thresh = iou, t

    best = (0.0, 0.5, 0.0, 0.0)   # f1, thresh, p, r
    for t, (tp, fp, fn) in n_stats.items():
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[0]:
            best = (f1, t, prec, rec)
    return dict(edge_iou=edge_iou, edge_thresh=edge_thresh,
                node_f1=best[0], node_thresh=best[1],
                node_p=best[2], node_r=best[3])


# =============================================================================
# 4. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Train multi-task GNN (SW edges + column nodes)")
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "data"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--node-loss-w", type=float, default=1.0)
    ap.add_argument("--pos-weight-cap", type=float, default=8.0,
                    help="Cap on the column-class BCE weight (raw ~19 "
                         "over-predicts; lower = higher precision)")
    ap.add_argument("--save", default=None,
                    help="Checkpoint path (default: gnn_multi.pt for "
                         "pipeline source, gnn_multi_legacy.pt for legacy)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Use only the first N train samples (smoke test)")
    ap.add_argument("--graph-source", default="pipeline",
                    choices=["pipeline", "legacy"],
                    help="pipeline = graphs via the real Stage 2->3 pipeline "
                         "(matches inference; default). legacy = old "
                         "skeleton+Hough converter.")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="Force re-conversion of the pipeline dataset cache")
    ap.add_argument("--synthetic", type=int, default=150,
                    help="Synthetic design-loop samples to mix into training "
                         "(step 6: engineering-valid column labels incl. "
                         "open-space grid columns; 0 = disable)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not HAS_TORCH:
        logger.error("PyTorch is required for training.")
        return

    if args.graph_source == "pipeline":
        import unified_dataset as ud
        train_data = ud.convert_dataset_pipeline(
            args.data_dir, "train",
            cache_path=os.path.join(_HERE, "cache_unified_train.pkl"),
            rebuild=args.rebuild_cache)
        val_data = ud.convert_dataset_pipeline(
            args.data_dir, "test",
            cache_path=os.path.join(_HERE, "cache_unified_test.pkl"),
            rebuild=args.rebuild_cache)
    else:
        train_data = convert_dataset(args.data_dir, "train")
        val_data = convert_dataset(args.data_dir, "test")

    if args.synthetic > 0:
        import synthetic_dataset as sdset
        syn_tr = sdset.generate_dataset(
            args.synthetic, seed=1,
            cache_path=os.path.join(_HERE, "cache_synth_train.pkl"),
            rebuild=args.rebuild_cache)
        syn_va = sdset.generate_dataset(
            max(10, args.synthetic // 5), seed=2,
            cache_path=os.path.join(_HERE, "cache_synth_val.pkl"),
            rebuild=args.rebuild_cache)
        train_data = train_data + syn_tr
        val_data = val_data + syn_va
        logger.info("Mixed in %d synthetic train / %d synthetic val samples",
                    len(syn_tr), len(syn_va))
    if args.limit:
        train_data = train_data[: args.limit]
        val_data = val_data[: max(2, args.limit // 4)]
    if not train_data:
        logger.error("No training samples found under %s", args.data_dir)
        return

    save_path = args.save or os.path.join(
        _HERE, "gnn_multi.pt" if args.graph_source == "pipeline"
        else "gnn_multi_legacy.pt")

    print(f"\nTrain: {len(train_data)} graphs   Val: {len(val_data)} graphs "
          f"(graphs: {args.graph_source})")
    train(train_data, val_data, epochs=args.epochs, lr=args.lr,
          augment=args.augment, device=args.device, save_path=save_path,
          node_loss_w=args.node_loss_w, pos_weight_cap=args.pos_weight_cap)


if __name__ == "__main__":
    main()
