# -*- coding: utf-8 -*-
"""Comment 42: material quantities and wall density for the ten batch models.

Concrete volumes are computed from the exported geometry and sections, so that
over-provision shows up as a quantity rather than only as a member count.
"""
import os, sys, json, glob, subprocess, shutil, time, re
import numpy as np

ROOT = r"C:\Dev\Plan_2_FEM_2026"
PY = r"C:\Dev\Plan_2_FEM_2026\venv\Scripts\python.exe"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
CACHE = os.path.join(SCR, "psw_cache")
WORK = os.path.join(SCR, "qty_work")
STOREY_H = 3.0
N_STOREYS = 3


def run(cmd, timeout=900):
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stderr


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
        rc, se = run(["stages/stage_4/stage_4.py", "-i", pj, "-o", e,
                      "-v", os.path.join(w, "a.png"),
                      "--pred-vis", os.path.join(w, "b.png"),
                      "--graph-vis", os.path.join(w, "c.png")])
        if rc: print("s4 fail", stem, se[-140:], flush=True); continue
        rc, se = run(["etabs/export.py", "-i", e, "-o", m,
                      "--stories", str(N_STOREYS), "--floor-height", str(STOREY_H)])
        if rc: print("s5 fail", stem, se[-140:], flush=True); continue

        M = json.load(open(m))
        nid = {n["id"]: np.array([n["x"], n["y"], n["z"]], float) for n in M["nodes"]}
        slab0 = next((s for s in M.get("slabs", []) if s.get("story") == 0), None)
        area = poly_area([nid[i][:2] for i in slab0["nodes"]]) if slab0 else 0.0
        floor_area_total = area * N_STOREYS

        # sections
        colsec = {s["name"]: s for s in M.get("column_sections", [])} \
            if "column_sections" in M else {}
        beamsec = {s["name"]: s for s in M.get("beam_sections", [])}
        slabsec = {s["name"]: s for s in M.get("slab_sections", [])}

        def col_b(name):
            if name in colsec:
                return float(colsec[name].get("width_m", 0.4))
            mm = re.match(r"C(\d+)X(\d+)", str(name))
            return int(mm.group(1)) / 1000.0 if mm else 0.4

        v_col = v_beam = 0.0
        for f in M["frame_members"]:
            a, b = nid.get(f["start_node"]), nid.get(f["end_node"])
            if a is None or b is None: continue
            L = float(np.linalg.norm(b - a))
            if f.get("type") == "column":
                bw = col_b(f.get("section"))
                v_col += bw * bw * L
            else:
                s = beamsec.get(f.get("section"))
                bw = float(s["width_m"]) if s else 0.30
                dp = float(s["depth_m"]) if s else 0.50
                v_beam += bw * dp * L

        v_wall = 0.0
        wl_storey = 0.0
        for wp in M.get("wall_panels", []):
            ns = [nid[i] for i in wp["nodes"]]
            L = max(float(np.linalg.norm(ns[1] - ns[0])),
                    float(np.linalg.norm(ns[2] - ns[1])))
            h = abs(float(ns[2][2] - ns[0][2])) or STOREY_H
            t = float(wp.get("thickness_m", 0.15))
            v_wall += t * L * h
            if wp.get("story") == 0:
                wl_storey += L

        v_slab = 0.0
        for s in M.get("slabs", []):
            t = float(s.get("thickness_m", 0.15))
            v_slab += t * poly_area([nid[i][:2] for i in s["nodes"]])

        v_tot = v_col + v_beam + v_wall + v_slab
        rows.append(dict(
            plan=stem, area_m2=round(area, 1),
            floor_area_total=round(floor_area_total, 1),
            v_col=round(v_col, 2), v_beam=round(v_beam, 2),
            v_wall=round(v_wall, 2), v_slab=round(v_slab, 2), v_tot=round(v_tot, 2),
            m3_per_m2=round(v_tot / max(floor_area_total, 1e-6), 3),
            vertical_m3_per_m2=round((v_col + v_wall) / max(floor_area_total, 1e-6), 3),
            wall_len_per_m2=round(wl_storey / max(area, 1e-6), 3),
        ))
        print(f"  {stem:15s} {area:6.1f} m2 | conc {v_tot:7.1f} m3 "
              f"= {v_tot/max(floor_area_total,1e-6):5.3f} m3/m2 "
              f"(col {v_col:5.1f} beam {v_beam:5.1f} wall {v_wall:5.1f} slab {v_slab:5.1f}) "
              f"({time.time()-t0:.0f}s)", flush=True)
        shutil.rmtree(w, ignore_errors=True)

    json.dump(rows, open(os.path.join(SCR, "quantities.json"), "w"), indent=2)
    q = [r["m3_per_m2"] for r in rows]; vq = [r["vertical_m3_per_m2"] for r in rows]
    wl = [r["wall_len_per_m2"] for r in rows]
    print(f"\nconcrete {min(q):.3f}-{max(q):.3f} m3/m2 (median {np.median(q):.3f})")
    print(f"vertical structure only {min(vq):.3f}-{max(vq):.3f} m3/m2 (median {np.median(vq):.3f})")
    print(f"storey wall length {min(wl):.3f}-{max(wl):.3f} m per m2")
    tot = {k: sum(r[k] for r in rows) for k in ("v_col", "v_beam", "v_wall", "v_slab")}
    s = sum(tot.values())
    print("share: " + ", ".join(f"{k[2:]} {100*v/s:.0f}%" for k, v in tot.items()))


main()
