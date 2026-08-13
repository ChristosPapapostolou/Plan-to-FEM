# -*- coding: utf-8 -*-
"""Comment 32: audit the StructGAN 167/72 split.

Test images are anonymised ('test (n).png') while train images carry project and
building identity, so separation cannot be checked from names. This compares
image content across the split for exact and near duplicates, and reports the
class distributions of the resulting graphs.
"""
import os, glob, json, pickle
import numpy as np, cv2

ROOT = r"C:\Dev\Plan_2_FEM_2026"
DATA = os.path.join(ROOT, "train", "data")
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
GROUPS = ["L1_7", "L2_7", "L1L2_8"]


def load_small(path, n=64):
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    if w > 1.5 * h:                     # StructGAN double panels: left half
        im = im[:, : w // 2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    s = g.std()
    return g / s if s > 1e-6 else g


def dhash(path, n=16):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape[:2]
    if w > 1.5 * h:
        im = im[:, : w // 2]
    g = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def collect(split):
    out = []
    for g in GROUPS:
        for p in sorted(glob.glob(os.path.join(DATA, g, f"{split}_A", "*.png"))):
            out.append((g, os.path.basename(p), p))
    return out


def main():
    tr, te = collect("train"), collect("test")
    print(f"train {len(tr)} | test {len(te)}", flush=True)

    TR = [(m, load_small(p), dhash(p)) for (g, m, p) in tr]
    TE = [(m, load_small(p), dhash(p)) for (g, m, p) in te]
    TR = [t for t in TR if t[1] is not None]
    TE = [t for t in TE if t[1] is not None]

    A = np.stack([t[1].ravel() for t in TR])
    B = np.stack([t[1].ravel() for t in TE])
    C = (B @ A.T) / A.shape[1]                       # normalised correlation

    ham = np.zeros((len(TE), len(TR)), dtype=int)
    HA = np.stack([t[2] for t in TR]); HB = np.stack([t[2] for t in TE])
    for i in range(len(TE)):
        ham[i] = (HB[i] != HA).sum(axis=1)

    best_c = C.max(axis=1); best_i = C.argmax(axis=1)
    best_h = ham.min(axis=1)

    res = {"n_train": len(TR), "n_test": len(TE),
           "corr_max_mean": round(float(best_c.mean()), 4),
           "corr_max_p95": round(float(np.percentile(best_c, 95)), 4),
           "n_corr_ge_099": int((best_c >= 0.99).sum()),
           "n_corr_ge_095": int((best_c >= 0.95).sum()),
           "n_corr_ge_090": int((best_c >= 0.90).sum()),
           "hamming_min_mean": float(best_h.mean()),
           "n_hamming_le_8": int((best_h <= 8).sum()),
           "n_hamming_le_16": int((best_h <= 16).sum()),
           "closest_pairs": []}
    order = np.argsort(-best_c)[:8]
    for i in order:
        res["closest_pairs"].append(dict(
            test=TE[i][0], train=TR[best_i[i]][0],
            corr=round(float(best_c[i]), 4), hamming=int(ham[i, best_i[i]])))

    # within-train relatedness, for context
    CA = (A @ A.T) / A.shape[1]
    np.fill_diagonal(CA, -1)
    res["within_train_corr_max_mean"] = round(float(CA.max(axis=1).mean()), 4)
    res["within_train_n_ge_095"] = int((CA.max(axis=1) >= 0.95).sum())

    # class distributions from the cached unified graphs
    for split, cache in (("train", "cache_unified_train.pkl"),
                         ("test", "cache_unified_test.pkl")):
        p = os.path.join(ROOT, "train", cache)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                data = data.get("samples", [])
        except Exception as e:
            res[f"{split}_cache_error"] = str(e); continue
        n_e = n_psw = 0; sw_pos = 0; col_pos = 0; n_nodes = 0
        for s in data:
            ec = np.asarray(s["edge_class"])
            n_e += ec.size; n_psw += int((ec == 0).sum())
            m = np.asarray(s["psw_mask"], bool)
            lab = np.asarray(s["sw_labels"])
            if m.any():
                sw_pos += int((lab[m].sum(-1) > 0).sum())
            col = np.asarray(s["col_labels"])
            col_pos += int((col > 0.5).sum()); n_nodes += col.size
        res[f"{split}_graphs"] = len(data)
        res[f"{split}_edges"] = n_e
        res[f"{split}_psw_edges"] = n_psw
        res[f"{split}_sw_positive_rate"] = round(sw_pos / max(n_psw, 1), 4)
        res[f"{split}_nodes"] = n_nodes
        res[f"{split}_column_positive_rate"] = round(col_pos / max(n_nodes, 1), 4)

    print(json.dumps(res, indent=2), flush=True)
    open(os.path.join(SCR, "split_audit.json"), "w").write(json.dumps(res, indent=2))


main()
