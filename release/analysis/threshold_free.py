# -*- coding: utf-8 -*-
"""Comment 36: threshold-free and uncertainty-aware evaluation of the deployed
multi-task GNN, reported separately for real and synthetic graphs.

  - edge head : ROC AUC and average precision over potential shear-wall edges
  - node head : node-level ROC AUC / AP, plus a layout-level precision-recall
                curve obtained by sweeping the operating threshold
  - calibration: reliability curves and expected calibration error
  - F1 versus matching tolerance
  - bootstrap confidence intervals, resampling graphs rather than elements
"""
import os, sys, json, pickle
ROOT = r"C:\Dev\Plan_2_FEM_2026"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "models"))
import numpy as np, torch
import train_gnn_multi as T
import synthetic_dataset as sdset
from gnn_ep import nms_points, match_points, load_checkpoint

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
sys.path.insert(0, SCR)
CKPT = os.path.join(ROOT, "train", "gnn_multi1.pt")
NODE_THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
TOLS = [0.25, 0.50, 0.75, 1.00, 1.50]
NBOOT = 500
RNG = np.random.default_rng(7)


def auc_roc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or not y.any(): return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(~y)
    tpr = tp / tp[-1]; fpr = fp / fp[-1]
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def avg_precision(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if not y.any(): return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    prec = tp / k; rec = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def ece(y, s, bins=10):
    y = np.asarray(y, float); s = np.asarray(s, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0; n = len(s)
    curve = []
    for i in range(bins):
        m = (s >= edges[i]) & (s < edges[i + 1] if i < bins - 1 else s <= 1.0)
        if not m.any():
            curve.append((float((edges[i] + edges[i + 1]) / 2), None, 0)); continue
        conf = float(s[m].mean()); acc = float(y[m].mean())
        e += m.sum() / n * abs(acc - conf)
        curve.append((conf, acc, int(m.sum())))
    return float(e), curve


def forward(model, samples):
    """Per-graph edge and node scores/labels, kept grouped for bootstrapping."""
    out = []
    for s in samples:
        nf, ef = T.featurize(s)
        with torch.no_grad():
            e_out, n_logit = model(torch.tensor(nf), torch.tensor(s["edge_index"]),
                                   torch.tensor(ef))
        psw = np.asarray(s["psw_mask"], bool)
        es = torch.sigmoid(e_out[:, 2]).numpy()[psw] if psw.any() else np.zeros(0)
        el = (np.asarray(s["sw_labels"])[psw].sum(-1) > 0) if psw.any() else np.zeros(0, bool)
        ns = torch.sigmoid(n_logit.squeeze(-1)).numpy()
        nl = np.asarray(s["col_labels"]) > 0.5
        out.append(dict(edge_s=es, edge_y=el, node_s=ns, node_y=nl,
                        xy=np.asarray(s["xy_m"], float)))
    return out


def layout_pr(G, thr, tol):
    tp = fp = fn = 0
    for g in G:
        sel = np.where(g["node_s"] > thr)[0]
        pred = g["xy"][sel][nms_points(g["xy"][sel], g["node_s"][sel],
                                       min_dist=T.NMS_DIST_M)] if sel.size else np.zeros((0, 2))
        a, b, c = match_points(pred, g["xy"][g["node_y"]], tol=tol)
        tp += a; fp += b; fn += c
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return P, R, (2 * P * R / max(P + R, 1e-9))


def summarise(G, label):
    es = np.concatenate([g["edge_s"] for g in G]) if G else np.zeros(0)
    ey = np.concatenate([g["edge_y"] for g in G]) if G else np.zeros(0, bool)
    ns = np.concatenate([g["node_s"] for g in G])
    ny = np.concatenate([g["node_y"] for g in G])
    e_ece, e_curve = ece(ey, es); n_ece, n_curve = ece(ny, ns)
    res = dict(
        label=label, n_graphs=len(G), n_psw_edges=int(len(es)), n_nodes=int(len(ns)),
        edge_base_rate=round(float(ey.mean()), 4) if len(ey) else None,
        edge_roc_auc=round(auc_roc(ey, es), 4), edge_ap=round(avg_precision(ey, es), 4),
        edge_ece=round(e_ece, 4),
        node_base_rate=round(float(ny.mean()), 4),
        node_roc_auc=round(auc_roc(ny, ns), 4), node_ap=round(avg_precision(ny, ns), 4),
        node_ece=round(n_ece, 4),
        edge_reliability=e_curve, node_reliability=n_curve,
    )
    # layout-level PR curve and F1 vs tolerance
    res["layout_pr"] = [dict(thr=float(t), **dict(zip(("P", "R", "F1"),
                        [round(v, 4) for v in layout_pr(G, float(t), 0.5)])))
                        for t in NODE_THRS]
    res["f1_vs_tol"] = [dict(tol=t, F1=round(layout_pr(G, 0.4, t)[2], 4)) for t in TOLS]
    # bootstrap over graphs
    boots = {k: [] for k in ("edge_roc_auc", "edge_ap", "node_roc_auc", "node_ap", "f1_at_op")}
    idx = np.arange(len(G))
    for _ in range(NBOOT):
        take = RNG.choice(idx, size=len(idx), replace=True)
        B = [G[i] for i in take]
        be = np.concatenate([g["edge_s"] for g in B]); by = np.concatenate([g["edge_y"] for g in B])
        bn = np.concatenate([g["node_s"] for g in B]); bny = np.concatenate([g["node_y"] for g in B])
        boots["edge_roc_auc"].append(auc_roc(by, be)); boots["edge_ap"].append(avg_precision(by, be))
        boots["node_roc_auc"].append(auc_roc(bny, bn)); boots["node_ap"].append(avg_precision(bny, bn))
        boots["f1_at_op"].append(layout_pr(B, 0.4, 0.5)[2])
    for k, v in boots.items():
        v = np.array([x for x in v if np.isfinite(x)])
        res[k + "_ci95"] = [round(float(np.percentile(v, 2.5)), 4),
                            round(float(np.percentile(v, 97.5)), 4)] if len(v) else None
    return res


def main():
    model, _nt, _m = load_checkpoint(CKPT); model.eval()
    print("checkpoint:", os.path.basename(CKPT), flush=True)

    import clean_split as CSmod  # reuse leak detection
    names, leaks = CSmod.find_leaks()
    S = pickle.load(open(os.path.join(ROOT, "train", "cache_unified_test.pkl"), "rb"))["samples"]
    seen, dedup = set(), []
    for s in S:
        if s["name"] in seen: continue
        seen.add(s["name"]); dedup.append(s)
    real = [s for s in dedup if s["name"] not in leaks]
    syn_val = sdset.generate_dataset(30, seed=2,
                                     cache_path=os.path.join(ROOT, "train", "cache_synth_val.pkl"))
    syn_test = sdset.generate_dataset(60, seed=101,
                                      cache_path=os.path.join(SCR, "cache_synth_test101.pkl"))
    print(f"real {len(real)} | synthetic val {len(syn_val)} | synthetic test {len(syn_test)}", flush=True)

    out = {}
    for label, S2 in (("StructGAN real (clean split)", real),
                      ("synthetic validation", syn_val),
                      ("synthetic test (unseen)", syn_test)):
        G = forward(model, S2)
        out[label] = summarise(G, label)
        r = out[label]
        print(f"\n{label}  ({r['n_graphs']} graphs)", flush=True)
        print(f"  edge  ROC AUC {r['edge_roc_auc']:.3f} {r['edge_roc_auc_ci95']} | "
              f"AP {r['edge_ap']:.3f} {r['edge_ap_ci95']} | base {r['edge_base_rate']:.3f} | "
              f"ECE {r['edge_ece']:.3f}", flush=True)
        print(f"  node  ROC AUC {r['node_roc_auc']:.3f} {r['node_roc_auc_ci95']} | "
              f"AP {r['node_ap']:.3f} {r['node_ap_ci95']} | base {r['node_base_rate']:.3f} | "
              f"ECE {r['node_ece']:.3f}", flush=True)
        print(f"  layout F1 at op threshold {r['f1_at_op_ci95']}", flush=True)

    json.dump(out, open(os.path.join(SCR, "threshold_free.json"), "w"), indent=2)
    print("\nsaved threshold_free.json", flush=True)


main()
