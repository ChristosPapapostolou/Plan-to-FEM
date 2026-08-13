# -*- coding: utf-8 -*-
"""Is CubiCasa5K's F1_scaled.png at a constant metres-per-pixel?

Each Space carries a DimensionMeasureLabel with its real size ("3.72 m x 1.86 m").
Pairing that with the Space polygon's pixel extents gives m/px per room.
"""
import sys, re, json
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
import numpy as np, cv2
from xml.dom import minidom

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
exec(open(SCR + r"\eval_cubicasa.py", encoding="utf-8").read().split("def main()")[0])

ROOT = r"C:\Dev\Plan_2_FEM_2026\cubicasa5k\cubicasa5k"
MET = re.compile(r"([\d.]+)\s*m\s*[x\u00d7]\s*([\d.]+)\s*m")


def spaces_with_dims(svg_path):
    doc = minidom.parse(svg_path)
    out = []
    for g in doc.getElementsByTagName("g"):
        cls = g.getAttribute("class")
        if not (cls == "Space" or cls.startswith("Space ")):
            continue
        poly = next((ch for ch in g.childNodes
                     if getattr(ch, "tagName", None) == "polygon"), None)
        if poly is None:
            continue
        lab = None
        for sub in g.getElementsByTagName("g"):
            if "DimensionMeasureLabel" in sub.getAttribute("class"):
                for tx in sub.getElementsByTagName("text"):
                    txt = "".join(n.data for n in tx.childNodes
                                  if n.nodeType == n.TEXT_NODE)
                    m = MET.search(txt)
                    if m:
                        lab = (float(m.group(1)), float(m.group(2))); break
            if lab: break
        if lab is None:
            continue
        pts = [tuple(map(float, p.split(",")))
               for p in poly.getAttribute("points").strip().split() if "," in p]
        if len(pts) < 3:
            continue
        M = elem_matrix(g)
        P = np.array([[x, y, 1.0] for x, y in pts]).T
        Q = (M @ P)[:2].T.astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(Q)
        if min(w, h) < 5:
            continue
        px = sorted([w, h], reverse=True)
        me = sorted(lab, reverse=True)
        if me[1] < 0.3:
            continue
        out.append((me[0] / px[0], me[1] / px[1], me, px))
    return out


def main(n_plans=40, stride=7):
    folders = [l.strip().strip("/") for l in open(f"{ROOT}\\test.txt") if l.strip()]
    folders = folders[::stride][:n_plans]
    per_plan, ratios = [], []
    for rel in folders:
        d = f"{ROOT}\\{rel.replace('/', chr(92))}"
        try:
            rows = spaces_with_dims(d + r"\model.svg")
        except Exception:
            continue
        if not rows:
            continue
        r = [x for pair in rows for x in pair[:2]]
        ratios.extend(r)
        per_plan.append(float(np.median(r)))
    a = np.array(ratios)
    res = {
        "plans_with_labels": len(per_plan),
        "room_measurements": len(a),
        "m_per_px_median": round(float(np.median(a)), 6),
        "m_per_px_mean": round(float(np.mean(a)), 6),
        "m_per_px_p05": round(float(np.percentile(a, 5)), 6),
        "m_per_px_p95": round(float(np.percentile(a, 95)), 6),
        "per_plan_median_min": round(float(np.min(per_plan)), 6),
        "per_plan_median_max": round(float(np.max(per_plan)), 6),
        "per_plan_median_std": round(float(np.std(per_plan)), 6),
        "frac_within_2pct_of_0.01": round(float(np.mean(np.abs(a - 0.01) <= 0.0002)), 4),
    }
    print(json.dumps(res, indent=2))
    open(SCR + r"\cubicasa_scale.json", "w").write(json.dumps(res, indent=2))


main()
