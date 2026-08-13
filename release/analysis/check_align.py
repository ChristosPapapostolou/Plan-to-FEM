# -*- coding: utf-8 -*-
"""Verify how CubiCasa SVG wall coordinates map onto F1_scaled.png pixels."""
import re, glob, cv2, numpy as np
from xml.dom import minidom

ROOT = r"C:\Dev\Plan_2_FEM_2026\cubicasa5k\cubicasa5k"

def wall_polys(svg_path):
    """First polygon of every <g class='Wall'>, with ancestor transforms applied."""
    doc = minidom.parse(svg_path)
    svg = doc.getElementsByTagName("svg")[0]
    W = float(svg.getAttribute("width")); H = float(svg.getAttribute("height"))
    polys = []
    for g in doc.getElementsByTagName("g"):
        if g.getAttribute("class") != "Wall":
            continue
        poly = None
        for ch in g.childNodes:
            if getattr(ch, "tagName", None) == "polygon":
                poly = ch; break
        if poly is None:
            continue
        # accumulate ancestor translate/matrix transforms
        tx = ty = 0.0
        node = g
        while node is not None and getattr(node, "getAttribute", None):
            t = node.getAttribute("transform") if node.nodeType == 1 else ""
            if t:
                m = re.match(r"matrix\(([-\d.eE,\s]+)\)", t)
                if m:
                    v = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip())]
                    if len(v) == 6: tx += v[4]; ty += v[5]
                m = re.match(r"translate\(([-\d.eE,\s]+)\)", t)
                if m:
                    v = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip())]
                    tx += v[0]; ty += v[1] if len(v) > 1 else 0
            node = node.parentNode
        pts = [tuple(map(float, p.split(",")))
               for p in poly.getAttribute("points").strip().split() if "," in p]
        polys.append(np.array([[x + tx, y + ty] for x, y in pts], np.float32))
    return polys, W, H

def render(polys, w, h, sx=1.0, sy=1.0, ox=0.0, oy=0.0):
    m = np.zeros((int(h), int(w)), np.uint8)
    for p in polys:
        q = p.copy(); q[:, 0] = q[:, 0] * sx + ox; q[:, 1] = q[:, 1] * sy + oy
        cv2.fillPoly(m, [q.astype(np.int32)], 255)
    return m

for d in sorted(glob.glob(ROOT + r"\high_quality_architectural\*"))[:5]:
    try:
        polys, SW, SH = wall_polys(d + r"\model.svg")
        img = cv2.imread(d + r"\F1_scaled.png")
        if img is None or not polys:
            print(d[-6:], "skip"); continue
        H, W = img.shape[:2]
        ink = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 200)

        # hypothesis A: SVG units == F1_scaled pixels, no scaling
        a = render(polys, W, H) > 0
        # hypothesis B: stretch SVG box onto the full image
        b = render(polys, W, H, sx=W / SW, sy=H / SH) > 0
        fa = (a & ink).sum() / max(a.sum(), 1)
        fb = (b & ink).sum() / max(b.sum(), 1)
        print(f"{d[-6:]:>7s} img={W}x{H} svg={SW:.0f}x{SH:.0f} "
              f"| A(direct) wall-on-ink={fa:.3f}  B(stretch)={fb:.3f}")
    except Exception as e:
        print(d[-6:], "ERR", e)
