# -*- coding: utf-8 -*-
"""Choose the column thinning radius from data.

Sweeps the non-maximum-suppression radius applied to GNN column predictions and
measures both the resulting member density and the coherence outcome, so the
value is picked where density reaches practice without breaking the checks.
"""
import os, sys, json, glob, subprocess, shutil, time
import numpy as np

ROOT = r"C:\Dev\Plan_2_FEM_2026"
PY = r"C:\Dev\Plan_2_FEM_2026\venv\Scripts\python.exe"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
WRAP = os.path.join(SCR, "stage4_nms.py")
CACHE = os.path.join(SCR, "psw_cache")
WORK = os.path.join(SCR, "nms_work")
RADII = [0.60, 1.50, 2.00, 2.50, 3.00, 4.00]


def run(cmd, env=None, timeout=900):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout, env=e)
    return p.returncode, p.stdout, p.stderr


def poly_area(pts):
    p = np.asarray(pts, float)
    if len(p) < 3: return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def main():
    os.makedirs(WORK, exist_ok=True)
    plans = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    rows = []
    t0 = time.time()
    for R in RADII:
        for pj in plans:
            stem = os.path.splitext(os.path.basename(pj))[0]
            w = os.path.join(WORK, f"{stem}_{R}"); os.makedirs(w, exist_ok=True)
            e = os.path.join(w, "e.json"); m = os.path.join(w, "m.json")
            s = os.path.join(w, "s.json")
            rc, so, se = run([WRAP, "-i", pj, "-o", e, "-v", os.path.join(w, "a.png"),
                              "--pred-vis", os.path.join(w, "b.png"),
                              "--graph-vis", os.path.join(w, "c.png")],
                             env={"NMS_R": str(R)})
            if rc != 0:
                print("s4 fail", stem, R, se[-160:], flush=True); continue
            rc, so, se = run(["etabs/export.py", "-i", e, "-o", m,
                              "--stories", "3", "--floor-height", "3.0"])
            if rc != 0:
                print("s5 fail", stem, R, se[-160:], flush=True); continue
            rc, so, se = run(["etabs/sanity.py", "-i", m, "-o", s])
            if rc != 0:
                print("sanity fail", stem, R, se[-160:], flush=True); continue

            M = json.load(open(m)); S = json.load(open(s))
            nid = {n["id"]: (n["x"], n["y"], n["z"]) for n in M["nodes"]}
            slab = next((x for x in M.get("slabs", []) if x.get("story") == 0), None)
            area = poly_area([nid[i][:2] for i in slab["nodes"]]) if slab else 0.0
            cols = np.array([nid[f["start_node"]][:2] for f in M["frame_members"]
                             if f.get("type") == "column" and f.get("story") == 0])
            sp = []
            if len(cols) > 1:
                Dm = np.linalg.norm(cols[:, None, :] - cols[None, :, :], axis=2)
                np.fill_diagonal(Dm, np.inf); sp = Dm.min(axis=1)
            chk = {c["name"]: c for c in S["checks"]}
            be = chk["beam_support"]["value"]
            rows.append(dict(
                radius=R, plan=stem, area=round(area, 1), n_cols=len(cols),
                per100=round(100 * len(cols) / max(area, 1e-6), 2),
                trib=round(area / max(len(cols), 1), 1),
                spacing_med=round(float(np.median(sp)), 2) if len(sp) else None,
                n_beams0=sum(1 for f in M["frame_members"]
                             if f.get("type") == "beam" and f.get("story") == 0),
                score=S["score"], n_warn=S["n_warn"], n_fail=S["n_fail"],
                span=chk["slab_span"]["value"],
                span_status=chk["slab_span"]["status"],
                beam_unsup=be.get("unsupported", 0),
                beam_status=chk["beam_support"]["status"],
            ))
            shutil.rmtree(w, ignore_errors=True)
        print(f"  radius {R} done ({time.time()-t0:.0f}s)", flush=True)

    json.dump(rows, open(os.path.join(SCR, "nms_sweep.json"), "w"), indent=2)
    print(f"\n{'R (m)':>6s} {'cols':>6s} {'/100m2':>7s} {'trib m2':>8s} {'space':>6s} "
          f"{'beams0':>7s} {'span m':>7s} {'unsup':>6s} {'score':>6s} {'nofail':>7s}")
    for R in RADII:
        rr = [r for r in rows if r["radius"] == R]
        if not rr: continue
        f = lambda k: float(np.mean([r[k] for r in rr]))
        print(f"{R:6.2f} {f('n_cols'):6.1f} {f('per100'):7.2f} {f('trib'):8.1f} "
              f"{f('spacing_med'):6.2f} {f('n_beams0'):7.1f} {f('span'):7.2f} "
              f"{f('beam_unsup'):6.2f} {f('score'):6.3f} "
              f"{sum(1 for r in rr if r['n_fail'] == 0):7d}")


main()
