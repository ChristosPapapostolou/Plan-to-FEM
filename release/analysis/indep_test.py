# -*- coding: utf-8 -*-
"""Comment 26: evaluate the deployed GNN on an untouched test set at the
thresholds that were fixed beforehand.

Training used synthetic seeds 1 (train) and 2 (validation); seed 101 has never
been seen by the model or by any selection decision. We report:

  (a) validation, best threshold  -- the protocol behind the reported numbers
  (b) validation, fixed threshold -- the same data, thresholds frozen
  (c) test (seed 101), fixed      -- untouched data, thresholds frozen

The gap between (a) and (c) is the optimism introduced by choosing the
checkpoint and the operating thresholds on the same split that is reported.
"""
import os, sys, json
ROOT = r"C:\Dev\Plan_2_FEM_2026"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "models"))
import numpy as np, torch

import synthetic_dataset as sdset
import train_gnn_multi as T
from gnn_ep import nms_points, match_points, load_checkpoint

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
CKPT = os.path.join(ROOT, "train", "gnn_multi.pt")
TEST_SEED = 101
N_TEST = 60


def score(model, samples, edge_t, node_t):
    """Fixed-threshold layout metrics, plus the all-positive edge base rate."""
    ei_i = ei_u = 0
    bi = bu = 0
    tp = fp = fn = 0
    for s in samples:
        nf, ef = T.featurize(s)
        x = torch.tensor(nf); e_idx = torch.tensor(s["edge_index"]); ea = torch.tensor(ef)
        with torch.no_grad():
            e_out, n_logit = model(x, e_idx, ea)
        psw = torch.tensor(s["psw_mask"])
        if psw.any():
            p = e_out[psw]
            gt = (torch.tensor(s["sw_labels"])[psw].sum(-1) > 0)
            pr = torch.sigmoid(p[:, 2]) > edge_t
            ei_i += (pr & gt).sum().item(); ei_u += (pr | gt).sum().item()
            allpos = torch.ones_like(gt, dtype=torch.bool)
            bi += (allpos & gt).sum().item(); bu += (allpos | gt).sum().item()
        probs = torch.sigmoid(n_logit.squeeze(-1)).numpy()
        xy = s["xy_m"]; label_xy = xy[s["col_labels"] > 0.5]
        sel = np.where(probs > node_t)[0]
        if sel.size:
            keep = nms_points(xy[sel], probs[sel], min_dist=T.NMS_DIST_M)
            pred_xy = xy[sel][keep]
        else:
            pred_xy = np.zeros((0, 2))
        a, b, c = match_points(pred_xy, label_xy, tol=T.MATCH_TOL_M)
        tp += a; fp += b; fn += c
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(edge_iou=round(ei_i / max(ei_u, 1), 4),
                edge_base_rate=round(bi / max(bu, 1), 4),
                node_p=round(prec, 4), node_r=round(rec, 4),
                node_f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4))


def main():
    model, node_t, meta = load_checkpoint(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    edge_t = float(ck.get("edge_threshold", 0.5))
    node_t = float(ck.get("node_threshold", 0.5))
    model.eval()
    print(f"checkpoint thresholds: node={node_t} edge={edge_t}", flush=True)

    val = sdset.generate_dataset(30, seed=2,
                                 cache_path=os.path.join(ROOT, "train",
                                                         "cache_synth_val.pkl"))
    print(f"validation samples (seed 2): {len(val)}", flush=True)
    test = sdset.generate_dataset(N_TEST, seed=TEST_SEED,
                                  cache_path=os.path.join(SCR, "cache_synth_test101.pkl"))
    print(f"test samples (seed {TEST_SEED}, untouched): {len(test)}", flush=True)

    val_best = T.evaluate(model, val)          # threshold sweep, as reported
    val_fix = score(model, val, edge_t, node_t)
    test_fix = score(model, test, edge_t, node_t)
    test_best = T.evaluate(model, test)        # sweep on test, for reference

    out = {
        "checkpoint": os.path.basename(CKPT),
        "fixed_thresholds": {"node": node_t, "edge": edge_t},
        "n_val": len(val), "n_test": len(test), "test_seed": TEST_SEED,
        "val_best_threshold": {k: round(float(v), 4) for k, v in val_best.items()},
        "val_fixed_threshold": val_fix,
        "test_fixed_threshold": test_fix,
        "test_best_threshold": {k: round(float(v), 4) for k, v in test_best.items()},
    }
    print(json.dumps(out, indent=2), flush=True)
    open(os.path.join(SCR, "indep_test.json"), "w").write(json.dumps(out, indent=2))


main()
