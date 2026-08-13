# -*- coding: utf-8 -*-
"""Comment 23: sensitivity of the pipeline to the PSW thickness threshold.

Stage 1+2 are run once per plan and cached; the threshold is then swept through
Stage 4 (which builds the Stage 3 graph internally) -> FE export -> coherence
checks, exactly as in production apart from the swept value.
"""
import os, sys, json, glob, shutil, subprocess, time

ROOT = r"C:\Dev\Plan_2_FEM_2026"
PY = r"C:\Dev\Plan_2_FEM_2026\venv\Scripts\python.exe"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
WRAP = os.path.join(SCR, "stage4_psw.py")
CACHE = os.path.join(SCR, "psw_cache")
WORK = os.path.join(SCR, "psw_work")
THRESHOLDS = [0.03, 0.045, 0.06, 0.08, 0.10, 0.12, 0.15]
TIMEOUT = 900


def run(cmd, env=None, timeout=TIMEOUT):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout, env=e)
    return p.returncode, p.stdout, p.stderr


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    for img in sorted(glob.glob(os.path.join(ROOT, "images", "*.png"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        out = os.path.join(CACHE, stem + ".json")
        if os.path.exists(out):
            continue
        w = os.path.join(WORK, "_c_" + stem); os.makedirs(w, exist_ok=True)
        rc, so, se = run(["stages/stage_1/stage_1.py", "-i", img, "--out-dir", w])
        if rc != 0:
            print("stage1 fail", stem, se[-200:], flush=True); continue
        rc, so, se = run(["stages/stage_2/stage_2.py",
                          "-i", os.path.join(w, "output_mask.png"),
                          "-o", os.path.join(w, "floor_output.json"),
                          "-v", os.path.join(w, "floor_vis.png")])
        if rc != 0:
            print("stage2 fail", stem, se[-200:], flush=True); continue
        rc, so, se = run(["stages/stage_2/consolidate.py",
                          "-i", os.path.join(w, "floor_output.json"), "-o", out])
        if rc != 0:
            print("consolidate fail", stem, se[-200:], flush=True); continue
        shutil.rmtree(w, ignore_errors=True)
        print("cached", stem, flush=True)


def sweep():
    plans = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    print(f"{len(plans)} cached plans", flush=True)
    rows = []; t0 = time.time()
    for th in THRESHOLDS:
        for pj in plans:
            stem = os.path.splitext(os.path.basename(pj))[0]
            w = os.path.join(WORK, f"{stem}_{th}"); os.makedirs(w, exist_ok=True)
            e = os.path.join(w, "enriched.json")
            gs = os.path.join(w, "graph_stats.json")
            rc, so, se = run([WRAP, "-i", pj, "-o", e,
                              "-v", os.path.join(w, "e.png"),
                              "--pred-vis", os.path.join(w, "p.png"),
                              "--graph-vis", os.path.join(w, "gg.png")],
                             env={"PSW_T": str(th), "PSW_STATS": gs})
            if rc != 0:
                print("s4 fail", stem, th, se[-200:], flush=True); continue
            m = os.path.join(w, "model.json")
            rc, so, se = run(["etabs/export.py", "-i", e, "-o", m,
                              "--stories", "3", "--floor-height", "3.0"])
            if rc != 0:
                print("s5 fail", stem, th, se[-200:], flush=True); continue
            s = os.path.join(w, "sanity.json")
            rc, so, se = run(["etabs/sanity.py", "-i", m, "-o", s])
            if rc != 0:
                print("sanity fail", stem, th, se[-200:], flush=True); continue

            ej = json.load(open(e)); mj = json.load(open(m)); sj = json.load(open(s))
            gj = json.load(open(gs)) if os.path.exists(gs) else {}
            bc = gj.get("by_class", {})
            n_psw = sum(v for k, v in bc.items() if "PSW" in k)
            n_open = sum(v for k, v in bc.items() if ("DW" in k or "OD" in k))
            n_sw = 0; sw_len = 0.0
            for wl in ej.get("walls", []):
                for r in wl.get("rects", []):
                    if r.get("is_shear_wall"):
                        n_sw += 1; sw_len += float(r.get("length_m", 0.0))
            rows.append(dict(
                plan=stem, threshold=th,
                n_nodes=gj.get("n_nodes"), n_edges=gj.get("n_edges"),
                n_psw=n_psw, n_open=n_open, by_class=bc,
                n_sw_selected=n_sw, sw_len_m=round(sw_len, 2),
                n_cols=sum(1 for f in mj.get("frame_members", [])
                           if f.get("type") == "column"),
                n_panels=len(mj.get("wall_panels", [])),
                score=sj.get("score", sj.get("aggregate_score")),
            ))
            shutil.rmtree(w, ignore_errors=True)
        print(f"  threshold {th} done ({time.time()-t0:.0f}s)", flush=True)

    json.dump(rows, open(os.path.join(SCR, "psw_sweep.json"), "w"), indent=2)
    print(f"\n{'thr':>6s} {'edges':>7s} {'PSW':>7s} {'SW sel':>7s} "
          f"{'SW len':>8s} {'cols':>7s} {'panels':>7s} {'score':>6s}")
    for th in THRESHOLDS:
        rr = [r for r in rows if r["threshold"] == th]
        if not rr: continue
        n = len(rr); f = lambda k: sum((r[k] or 0) for r in rr) / n
        print(f"{th:6.3f} {f('n_edges'):7.1f} {f('n_psw'):7.1f} {f('n_open'):6.1f} "
              f"{f('n_sw_selected'):7.1f} {f('sw_len_m'):8.1f} {f('n_cols'):7.1f} "
              f"{f('n_panels'):7.1f} {f('score'):6.3f}  (n={n})")


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    build_cache()
    sweep()
