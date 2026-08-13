# -*- coding: utf-8 -*-
"""Comment 41: normalise member counts by floor area and report spacing and
section distributions for the ten evaluation plans."""
import os, sys, json, glob, subprocess, shutil, time
import numpy as np

ROOT = r"C:\Dev\Plan_2_FEM_2026"
PY = r"C:\Dev\Plan_2_FEM_2026\venv\Scripts\python.exe"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
CACHE = os.path.join(SCR, "psw_cache")
WORK = os.path.join(SCR, "density_work")


def run(cmd, timeout=900):
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def poly_area(pts):
    p = np.asarray(pts, float)
    if len(p) < 3: return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def main():
    os.makedirs(WORK, exist_ok=True)
    rows = []
    t0 = time.time()
    for pj in sorted(glob.glob(os.path.join(CACHE, "*.json"))):
        stem = os.path.splitext(os.path.basename(pj))[0]
        w = os.path.join(WORK, stem); os.makedirs(w, exist_ok=True)
        e = os.path.join(w, "e.json"); m = os.path.join(w, "m.json")
        rc, so, se = run(["stages/stage_4/stage_4.py", "-i", pj, "-o", e,
                          "-v", os.path.join(w, "a.png"),
                          "--pred-vis", os.path.join(w, "b.png"),
                          "--graph-vis", os.path.join(w, "c.png")])
        if rc != 0:
            print("s4 fail", stem, se[-160:], flush=True); continue
        rc, so, se = run(["etabs/export.py", "-i", e, "-o", m,
                          "--stories", "3", "--floor-height", "3.0"])
        if rc != 0:
            print("s5 fail", stem, se[-160:], flush=True); continue

        M = json.load(open(m))
        nid = {n["id"]: (n["x"], n["y"], n["z"]) for n in M["nodes"]}
        # storey-0 slab polygon gives the floor footprint
        slab = next((s for s in M.get("slabs", []) if s.get("story") == 0), None)
        area = poly_area([nid[i][:2] for i in slab["nodes"]]) if slab else 0.0

        cols = np.array([nid[f["start_node"]][:2] for f in M["frame_members"]
                         if f.get("type") == "column" and f.get("story") == 0])
        beams0 = sum(1 for f in M["frame_members"]
                     if f.get("type") == "beam" and f.get("story") == 0)
        # nearest-neighbour spacing
        sp = []
        if len(cols) > 1:
            D = np.linalg.norm(cols[:, None, :] - cols[None, :, :], axis=2)
            np.fill_diagonal(D, np.inf)
            sp = D.min(axis=1)
        # section distribution
        secs = {}
        for f in M["frame_members"]:
            if f.get("type") == "column" and f.get("story") == 0:
                secs[f.get("section", "?")] = secs.get(f.get("section", "?"), 0) + 1
        # shear-wall area ratio per storey
        wl = 0.0
        for wp in M.get("wall_panels", []):
            if wp.get("story") != 0: continue
            ns = [nid[i][:2] for i in wp["nodes"]]
            L = max(np.linalg.norm(np.array(ns[0]) - np.array(ns[1])),
                    np.linalg.norm(np.array(ns[1]) - np.array(ns[2])))
            wl += L * float(wp.get("thickness_m", 0.0))

        rows.append(dict(
            plan=stem, area_m2=round(area, 1), n_cols=len(cols), n_beams0=beams0,
            cols_per_100m2=round(100 * len(cols) / max(area, 1e-6), 2),
            trib_area_m2=round(area / max(len(cols), 1), 1),
            spacing_med=round(float(np.median(sp)), 2) if len(sp) else None,
            spacing_p10=round(float(np.percentile(sp, 10)), 2) if len(sp) else None,
            spacing_p90=round(float(np.percentile(sp, 90)), 2) if len(sp) else None,
            sections=secs,
            sw_area_ratio=round(100 * wl / max(area, 1e-6), 2),
        ))
        print(f"  {stem:15s} area={area:7.1f} cols={len(cols):3d} "
              f"({100*len(cols)/max(area,1e-6):5.2f}/100m2) trib={area/max(len(cols),1):5.1f} m2 "
              f"({time.time()-t0:.0f}s)", flush=True)
        shutil.rmtree(w, ignore_errors=True)

    json.dump(rows, open(os.path.join(SCR, "density.json"), "w"), indent=2)
    a = [r["cols_per_100m2"] for r in rows]; t = [r["trib_area_m2"] for r in rows]
    ar = [r["area_m2"] for r in rows]
    allsec = {}
    for r in rows:
        for k, v in r["sections"].items(): allsec[k] = allsec.get(k, 0) + v
    sp = [r["spacing_med"] for r in rows if r["spacing_med"]]
    print(f"\nfloor area {min(ar):.0f}-{max(ar):.0f} m2")
    print(f"columns per 100 m2: {min(a):.2f}-{max(a):.2f} (median {np.median(a):.2f})")
    print(f"tributary area per column: {min(t):.1f}-{max(t):.1f} m2 (median {np.median(t):.1f})")
    print(f"median nearest-neighbour spacing per plan: {min(sp):.2f}-{max(sp):.2f} m")
    print(f"sections: {dict(sorted(allsec.items()))}")
    print(f"SW area ratio per storey: {min(r['sw_area_ratio'] for r in rows):.2f}"
          f"-{max(r['sw_area_ratio'] for r in rows):.2f} %")


main()
