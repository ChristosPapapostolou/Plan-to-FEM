# -*- coding: utf-8 -*-
"""Comment 32 (fix): de-duplicate the StructGAN validation split and remove the
drawings that leak from training, then re-evaluate the deployed checkpoint.

  original : 72 entries  = 24 unique drawings x 3 identical group folders
  dedup    : 24 unique drawings
  clean    : 24 minus the drawings near-identical to a training drawing
"""
import os, sys, glob, json, pickle
ROOT = r"C:\Dev\Plan_2_FEM_2026"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "models"))
import numpy as np, cv2, torch
import train_gnn_multi as T
from gnn_ep import nms_points, match_points

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
DATA = os.path.join(ROOT, "train", "data")
CKPT = os.path.join(ROOT, "train", "gnn_multi.pt")
GROUPS = ["L1_7", "L2_7", "L1L2_8"]
CORR_T, HAM_T = 0.99, 8


def small(p, n=64):
    im = cv2.imread(p)
    if im is None: return None
    h, w = im.shape[:2]
    if w > 1.5 * h: im = im[:, : w // 2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean(); s = g.std()
    return g / s if s > 1e-6 else g


def dhash(p, n=16):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    h, w = im.shape[:2]
    if w > 1.5 * h: im = im[:, : w // 2]
    g = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def find_leaks():
    tr = []
    for g in GROUPS:
        tr += sorted(glob.glob(os.path.join(DATA, g, "train_A", "*.png")))
    te = {}
    for g in GROUPS:
        for p in sorted(glob.glob(os.path.join(DATA, g, "test_A", "*.png"))):
            te.setdefault(os.path.basename(p), p)
    TRs = [(os.path.basename(p), small(p), dhash(p)) for p in tr]
    TRs = [t for t in TRs if t[1] is not None]
    A = np.stack([t[1].ravel() for t in TRs]); HA = np.stack([t[2] for t in TRs])
    leaks = {}
    for name, p in sorted(te.items()):
        v = small(p); hb = dhash(p)
        if v is None: continue
        c = (A @ v.ravel()) / A.shape[1]
        ham = (HA != hb).sum(axis=1)
        i = int(np.argmax(c))
        if c[i] >= CORR_T and ham[i] <= HAM_T:
            leaks[name] = dict(train=TRs[i][0], corr=round(float(c[i]), 4),
                               hamming=int(ham[i]))
    return sorted(te), leaks


def score(model, samples, edge_t, node_t, sweep=False):
    if sweep:
        m = T.evaluate(model, samples)
        return {k: (round(float(v), 4)) for k, v in m.items()}
    ei_i = ei_u = bi = bu = 0; tp = fp = fn = 0
    for s in samples:
        nf, ef = T.featurize(s)
        x = torch.tensor(nf); idx = torch.tensor(s["edge_index"]); ea = torch.tensor(ef)
        with torch.no_grad():
            e_out, n_logit = model(x, idx, ea)
        psw = torch.tensor(s["psw_mask"])
        if psw.any():
            p = e_out[psw]
            gt = (torch.tensor(s["sw_labels"])[psw].sum(-1) > 0)
            pr = torch.sigmoid(p[:, 2]) > edge_t
            ei_i += (pr & gt).sum().item(); ei_u += (pr | gt).sum().item()
            ap = torch.ones_like(gt, dtype=torch.bool)
            bi += (ap & gt).sum().item(); bu += (ap | gt).sum().item()
        probs = torch.sigmoid(n_logit.squeeze(-1)).numpy()
        xy = s["xy_m"]; lab = xy[s["col_labels"] > 0.5]
        sel = np.where(probs > node_t)[0]
        pred = xy[sel][nms_points(xy[sel], probs[sel], min_dist=T.NMS_DIST_M)] \
            if sel.size else np.zeros((0, 2))
        a, b, c = match_points(pred, lab, tol=T.MATCH_TOL_M)
        tp += a; fp += b; fn += c
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(edge_iou=round(ei_i / max(ei_u, 1), 4),
                edge_base_rate=round(bi / max(bu, 1), 4),
                node_p=round(P, 4), node_r=round(R, 4),
                node_f1=round(2 * P * R / max(P + R, 1e-9), 4))


def main():
    names, leaks = find_leaks()
    print(f"unique test drawings: {len(names)} | leaking: {len(leaks)}", flush=True)
    for k, v in leaks.items():
        print(f"   {k}  ~  {v['train']}  corr={v['corr']} hamming={v['hamming']}", flush=True)

    with open(os.path.join(ROOT, "train", "cache_unified_test.pkl"), "rb") as f:
        d = pickle.load(f)
    S = d["samples"] if isinstance(d, dict) else d
    orig = list(S)
    seen, dedup = set(), []
    for s in S:
        n = s.get("name")
        if n in seen: continue
        seen.add(n); dedup.append(s)
    clean = [s for s in dedup if s.get("name") not in leaks]
    print(f"graphs: original {len(orig)} | dedup {len(dedup)} | clean {len(clean)}", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    from gnn_ep import load_checkpoint
    model, _nt, _meta = load_checkpoint(CKPT)
    model.eval()
    et = float(ck.get("edge_threshold", 0.5)); nt = float(ck.get("node_threshold", 0.5))

    out = {"thresholds": {"edge": et, "node": nt},
           "n": {"original": len(orig), "dedup": len(dedup), "clean": len(clean)},
           "leaks": leaks}
    for label, S2 in (("original_72", orig), ("dedup_24", dedup), ("clean_21", clean)):
        out[label + "_fixed"] = score(model, S2, et, nt)
        out[label + "_swept"] = score(model, S2, et, nt, sweep=True)
        print(f"{label:12s} fixed {out[label+'_fixed']}", flush=True)

    json.dump(out, open(os.path.join(SCR, "clean_split.json"), "w"), indent=2)
    print("\nsaved clean_split.json", flush=True)


main()
