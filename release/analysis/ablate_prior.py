# -*- coding: utf-8 -*-
"""Comment 49 (and part of 27): how much of the final shear-wall layout is
attributable to the learned prediction rather than to the rules.

Compares the deployed selector against one with the learned prior disabled,
on the same plans and the same geometry.
"""
import os, sys, json, glob, subprocess, shutil, time
import numpy as np

ROOT = r"C:\Dev\Plan_2_FEM_2026"
PY = r"C:\Dev\Plan_2_FEM_2026\venv\Scripts\python.exe"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
WRAP = os.path.join(SCR, "stage4_prior.py")
CACHE = os.path.join(SCR, "psw_cache")
WORK = os.path.join(SCR, "prior_work")
PRIORS = [("rules only", 2.00), ("deployed", 0.50), ("GNN permissive", 0.00)]


def run(cmd, env=None, timeout=900):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout, env=e)
    return p.returncode, p.stderr


def selected(path):
    d = json.load(open(path))
    ids, length = set(), 0.0
    for w in d.get("walls", []):
        for r in w.get("rects", []):
            if r.get("is_shear_wall"):
                ids.add(r["id"]); length += float(r.get("length_m", 0.0))
    return ids, length


def main():
    os.makedirs(WORK, exist_ok=True)
    plans = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    out = {}
    t0 = time.time()
    for label, P in PRIORS:
        rows = []
        for pj in plans:
            stem = os.path.splitext(os.path.basename(pj))[0]
            w = os.path.join(WORK, f"{stem}_{P}"); os.makedirs(w, exist_ok=True)
            e = os.path.join(w, "e.json"); m = os.path.join(w, "m.json")
            s = os.path.join(w, "s.json")
            rc, se = run([WRAP, "-i", pj, "-o", e, "-v", os.path.join(w, "a.png"),
                          "--pred-vis", os.path.join(w, "b.png"),
                          "--graph-vis", os.path.join(w, "c.png")],
                         env={"GNN_PRIOR": str(P)})
            if rc: print("s4 fail", stem, P, se[-140:], flush=True); continue
            rc, se = run(["etabs/export.py", "-i", e, "-o", m,
                          "--stories", "3", "--floor-height", "3.0"])
            if rc: print("s5 fail", stem, P, se[-140:], flush=True); continue
            rc, se = run(["etabs/sanity.py", "-i", m, "-o", s])
            if rc: print("sanity fail", stem, P, se[-140:], flush=True); continue
            ids, L = selected(e)
            S = json.load(open(s))
            rows.append(dict(plan=stem, ids=sorted(ids), n=len(ids),
                             length=round(L, 2), score=S["score"],
                             n_fail=S["n_fail"]))
            shutil.rmtree(w, ignore_errors=True)
        out[label] = rows
        print(f"  {label} done ({time.time()-t0:.0f}s)", flush=True)

    json.dump(out, open(os.path.join(SCR, "prior_ablation.json"), "w"), indent=2)

    base = {r["plan"]: set(r["ids"]) for r in out["deployed"]}
    print(f"\n{'configuration':16s} {'SW rects':>9s} {'length m':>9s} "
          f"{'score':>6s} {'Jaccard vs deployed':>20s} {'added by GNN':>13s}")
    for label, _P in PRIORS:
        rows = out[label]
        if not rows: continue
        n = np.mean([r["n"] for r in rows]); L = np.mean([r["length"] for r in rows])
        sc = np.mean([r["score"] for r in rows])
        js, add = [], []
        for r in rows:
            b = base.get(r["plan"], set()); a = set(r["ids"])
            u = len(a | b)
            js.append(len(a & b) / u if u else 1.0)
            add.append(len(b - a))
        print(f"{label:16s} {n:9.1f} {L:9.1f} {sc:6.3f} {np.mean(js):20.3f} "
              f"{np.mean(add):13.1f}")

    ro = {r["plan"]: set(r["ids"]) for r in out["rules only"]}
    tot_dep = sum(len(v) for v in base.values())
    tot_only_gnn = sum(len(base[p] - ro.get(p, set())) for p in base)
    tot_only_rule = sum(len(ro.get(p, set()) - base[p]) for p in base)
    print(f"\nacross the batch: {tot_dep} selected piers in the deployed system; "
          f"{tot_only_gnn} ({100*tot_only_gnn/max(tot_dep,1):.0f}%) are admitted only "
          f"because of the learned prior; {tot_only_rule} would be selected by the "
          f"rules alone but are not in the deployed set")


main()
