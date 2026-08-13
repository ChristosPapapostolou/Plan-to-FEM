# -*- coding: utf-8 -*-
"""Comment 30: sensitivity of the coherence score to the weighting scheme and
to the warn credit.

The score is a weighted mean over six checks of a status credit
(pass = 1, warn = 0.5, fail = 0). Both the weights and the warn credit are
choices; this recomputes the score under alternatives from the stored per-check
outcomes, so no re-run of the pipeline is needed.
"""
import json, glob, os, itertools
import numpy as np

ROOT = r"C:\Dev\Plan_2_FEM_2026"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"

CHECKS = ["lateral_lines", "eccentricity", "beam_support", "slab_span",
          "column_continuity", "orphan_nodes"]

SCHEMES = {
    "equal (deployed)":        {c: 1.0 for c in CHECKS},
    "lateral system emphasis": {"lateral_lines": 2.0, "eccentricity": 2.0,
                                "beam_support": 1.0, "slab_span": 1.0,
                                "column_continuity": 1.0, "orphan_nodes": 1.0},
    "gravity system emphasis": {"lateral_lines": 1.0, "eccentricity": 1.0,
                                "beam_support": 2.0, "slab_span": 2.0,
                                "column_continuity": 1.0, "orphan_nodes": 1.0},
    "topology down-weighted":  {"lateral_lines": 1.0, "eccentricity": 1.0,
                                "beam_support": 1.0, "slab_span": 1.0,
                                "column_continuity": 0.25, "orphan_nodes": 0.25},
    "four informative checks": {"lateral_lines": 1.0, "eccentricity": 1.0,
                                "beam_support": 1.0, "slab_span": 1.0,
                                "column_continuity": 0.0, "orphan_nodes": 0.0},
}
WARN_CREDITS = [0.25, 0.5, 0.75]


def load(paths):
    out = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        st = {c["name"]: c["status"] for c in d.get("checks", [])}
        if len(st) >= 5:
            out.append(st)
    return out


def score(st, w, warn):
    credit = {"pass": 1.0, "warn": warn, "fail": 0.0}
    num = sum(w[c] * credit.get(st.get(c, "fail"), 0.0) for c in CHECKS)
    den = sum(w[c] for c in CHECKS)
    return num / den if den else 0.0


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def analyse(name, samples):
    print(f"\n===== {name}  (n = {len(samples)}) =====", flush=True)
    base = [score(s, SCHEMES["equal (deployed)"], 0.5) for s in samples]
    rows = []
    for sname, w in SCHEMES.items():
        sc = [score(s, w, 0.5) for s in samples]
        rows.append(dict(scheme=sname, warn=0.5,
                         mean=round(float(np.mean(sc)), 4),
                         median=round(float(np.median(sc)), 4),
                         frac_ge_080=round(float(np.mean(np.array(sc) >= 0.80)), 4),
                         rank_corr=round(spearman(base, sc), 4)))
    for wc in WARN_CREDITS:
        if wc == 0.5:
            continue
        sc = [score(s, SCHEMES["equal (deployed)"], wc) for s in samples]
        rows.append(dict(scheme=f"equal, warn credit {wc}", warn=wc,
                         mean=round(float(np.mean(sc)), 4),
                         median=round(float(np.median(sc)), 4),
                         frac_ge_080=round(float(np.mean(np.array(sc) >= 0.80)), 4),
                         rank_corr=round(spearman(base, sc), 4)))
    print(f"{'scheme':28s} {'mean':>7s} {'median':>7s} {'>=0.80':>7s} {'rank rho':>9s}")
    for r in rows:
        print(f"{r['scheme']:28s} {r['mean']:7.3f} {r['median']:7.3f} "
              f"{r['frac_ge_080']:7.2%} {r['rank_corr']:9.3f}")
    return rows


def main():
    ten = load([p for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "sanity_report.json")))
               if os.sep + "_work" + os.sep not in p])
    msd_paths = sorted(glob.glob(os.path.join(ROOT, "results", "msd_validation",
                                              "*", "sanity_report.json")))
    msd = load(msd_paths)
    out = {}
    if ten:
        out["ten_plan"] = analyse("ten evaluation plans", ten)
    if msd:
        out["msd"] = analyse("MSD corpus", msd)
    # per-check failure incidence, for the justification text
    for label, S in (("ten_plan", ten), ("msd", msd)):
        if not S:
            continue
        inc = {c: {k: round(float(np.mean([s.get(c) == k for s in S])), 4)
                   for k in ("pass", "warn", "fail")} for c in CHECKS}
        out[label + "_incidence"] = inc
    json.dump(out, open(os.path.join(SCR, "weight_sens.json"), "w"), indent=2)
    print("\nsaved weight_sens.json", flush=True)


main()
