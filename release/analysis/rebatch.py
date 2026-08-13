# -*- coding: utf-8 -*-
"""Re-derive the per-plan batch table under the new default GNN checkpoint.

Stage 1-2 are taken from the cache built earlier (same code, same deadband);
Stage 4, FE export and the coherence checks are re-run so that element counts
and scores correspond to the deployed model.
"""
import os, sys, json, glob, subprocess, shutil, time

ROOT = r"C:\Dev\Plan_2_FEM_2026"
PY = r"C:\Dev\Plan_2_FEM_2026\venv\Scripts\python.exe"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
CACHE = os.path.join(SCR, "psw_cache")
WORK = os.path.join(SCR, "rebatch_work")


def run(cmd, timeout=900):
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def main():
    os.makedirs(WORK, exist_ok=True)
    rows = []
    t0 = time.time()
    for pj in sorted(glob.glob(os.path.join(CACHE, "*.json"))):
        stem = os.path.splitext(os.path.basename(pj))[0]
        w = os.path.join(WORK, stem); os.makedirs(w, exist_ok=True)
        e = os.path.join(w, "enriched.json")
        rc, so, se = run(["stages/stage_4/stage_4.py", "-i", pj, "-o", e,
                          "-v", os.path.join(w, "e.png"),
                          "--pred-vis", os.path.join(w, "p.png"),
                          "--graph-vis", os.path.join(w, "g.png")])
        if rc != 0:
            print("s4 fail", stem, se[-200:], flush=True); continue
        m = os.path.join(w, "model.json")
        rc, so, se = run(["etabs/export.py", "-i", e, "-o", m,
                          "--stories", "3", "--floor-height", "3.0"])
        if rc != 0:
            print("s5 fail", stem, se[-200:], flush=True); continue
        s = os.path.join(w, "sanity.json")
        rc, so, se = run(["etabs/sanity.py", "-i", m, "-o", s])
        if rc != 0:
            print("sanity fail", stem, se[-200:], flush=True); continue

        M = json.load(open(m)); S = json.load(open(s))
        chk = {c["name"]: c for c in S["checks"]}
        ecc = chk["eccentricity"]["value"]
        rows.append(dict(
            plan=stem,
            sw_panels=len(M.get("wall_panels", [])),
            beams=sum(1 for f in M["frame_members"] if f.get("type") == "beam"),
            columns=sum(1 for f in M["frame_members"] if f.get("type") == "column"),
            cols_per_floor=chk["column_continuity"]["value"]["locations"],
            ecc_x=round(100 * float(ecc["x"]), 1), ecc_y=round(100 * float(ecc["y"]), 1),
            ecc_worst=round(100 * max(float(ecc["x"]), float(ecc["y"])), 1),
            span=chk["slab_span"]["value"],
            lines_x=chk["lateral_lines"]["value"]["x_dir"],
            lines_y=chk["lateral_lines"]["value"]["y_dir"],
            beam_ends=chk["beam_support"]["value"],
            n_warn=S["n_warn"], n_fail=S["n_fail"], score=S["score"],
        ))
        print(f"  {stem:15s} score={S['score']:.3f} cols/floor="
              f"{chk['column_continuity']['value']['locations']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        shutil.rmtree(w, ignore_errors=True)

    json.dump(rows, open(os.path.join(SCR, "rebatch.json"), "w"), indent=2)
    n = len(rows)
    print(f"\n{n} plans | mean score "
          f"{sum(r['score'] for r in rows)/max(n,1):.4f}", flush=True)
    print(f"SW panels {min(r['sw_panels'] for r in rows)}-{max(r['sw_panels'] for r in rows)} | "
          f"beams {min(r['beams'] for r in rows)}-{max(r['beams'] for r in rows)} | "
          f"columns {min(r['columns'] for r in rows)}-{max(r['columns'] for r in rows)} | "
          f"cols/floor {min(r['cols_per_floor'] for r in rows)}-{max(r['cols_per_floor'] for r in rows)}")
    print(f"no failing check: {sum(1 for r in rows if r['n_fail']==0)} | "
          f"all six pass: {sum(1 for r in rows if r['n_fail']==0 and r['n_warn']==0)}")


main()
