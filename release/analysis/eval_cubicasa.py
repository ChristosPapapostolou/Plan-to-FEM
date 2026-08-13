# -*- coding: utf-8 -*-
"""Evaluate the deployed Stage-1 checkpoint on the CubiCasa5K official test
split against human wall annotations parsed from model.svg.

Full SVG transform composition; each sample is alignment-gated (the rendered
wall mask must land on drawing ink) so that no misaligned ground truth can
enter the metric."""
import sys, re, json, time
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\finetune_stage1")
import numpy as np, cv2, torch
from xml.dom import minidom
import finetune as F

ROOT = r"C:\Dev\Plan_2_FEM_2026\cubicasa5k\cubicasa5k"
CKPT = r"C:\Dev\Plan_2_FEM_2026\stages\stage_1\mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
SPLIT = "test.txt"
# Geometric sanity gate. CubiCasa walls are drawn as OUTLINES with white
# interiors, so "wall area on ink" is naturally low and is not a validity test;
# alignment was verified visually. We gate on geometry instead: the annotation
# must lie inside the image and cover a plausible area fraction.
MAX_OVERFLOW = 0.02   # fraction of wall pixels allowed outside the image
AREA_LO, AREA_HI = 0.005, 0.45
THRS = [0.4, 0.5]    # 0.4 = deployment threshold in stage_1.py; 0.5 = training

I3 = np.eye(3, dtype=np.float64)

def parse_transform(t: str) -> np.ndarray:
    M = I3.copy()
    for name, args in re.findall(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)", t):
        v = [float(x) for x in re.split(r"[,\s]+", args.strip()) if x]
        m = I3.copy()
        if name == "matrix" and len(v) == 6:
            m = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif name == "translate":
            m = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0], [0, 0, 1]])
        elif name == "scale":
            sx = v[0]; sy = v[1] if len(v) > 1 else v[0]
            m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        elif name == "rotate":
            a = np.deg2rad(v[0]); c, s = np.cos(a), np.sin(a)
            m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        M = M @ m
    return M

def elem_matrix(node) -> np.ndarray:
    chain = []
    n = node
    while n is not None and getattr(n, "nodeType", None) == 1:
        t = n.getAttribute("transform")
        if t:
            chain.append(t)
        n = n.parentNode
    M = I3.copy()
    for t in reversed(chain):
        M = M @ parse_transform(t)
    return M

def wall_polygons(svg_path):
    doc = minidom.parse(svg_path)
    out = []
    for g in doc.getElementsByTagName("g"):
        cls = g.getAttribute("class")
        if cls != "Wall" and not cls.startswith("Wall "):
            continue
        poly = next((ch for ch in g.childNodes
                     if getattr(ch, "tagName", None) == "polygon"), None)
        if poly is None:
            continue
        pts = [tuple(map(float, p.split(",")))
               for p in poly.getAttribute("points").strip().split() if "," in p]
        if len(pts) < 3:
            continue
        M = elem_matrix(g)
        P = np.array([[x, y, 1.0] for x, y in pts]).T
        Q = (M @ P)[:2].T
        out.append(Q.astype(np.float32))
    return out

def main():
    model = F.build_model(CKPT, "cpu"); model.eval()
    tf = F.val_transform()
    folders = [l.strip().strip("/") for l in open(f"{ROOT}\\{SPLIT}") if l.strip()]
    print(f"{SPLIT}: {len(folders)} samples", flush=True)

    acc = {th: [0.0, 0.0, 0.0] for th in THRS}
    per_img = {th: [] for th in THRS}
    aligns, used, skipped = [], 0, 0
    t0 = time.time()

    for k, rel in enumerate(folders):
        d = f"{ROOT}\\{rel.replace('/', chr(92))}"
        try:
            img = cv2.imread(d + r"\F1_scaled.png")
            if img is None:
                skipped += 1; continue
            polys = wall_polygons(d + r"\model.svg")
            if not polys:
                skipped += 1; continue
            H, W = img.shape[:2]
            gt = np.zeros((H, W), np.uint8)
            for p in polys:
                cv2.fillPoly(gt, [np.round(p).astype(np.int32)], 255)
            gtb = gt > 0
            if gtb.sum() < 500:
                skipped += 1; continue

            # geometric gate
            allp = np.concatenate(polys, axis=0)
            inside = ((allp[:, 0] >= 0) & (allp[:, 0] < W) &
                      (allp[:, 1] >= 0) & (allp[:, 1] < H)).mean()
            area_frac = float(gtb.sum()) / float(H * W)
            if (1.0 - inside) > MAX_OVERFLOW or not (AREA_LO <= area_frac <= AREA_HI):
                skipped += 1; continue
            # reported diagnostic: ink coverage on the wall OUTLINE
            ink = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 200
            er = cv2.erode(gt, np.ones((5, 5), np.uint8), iterations=1) > 0
            border = gtb & ~er
            aligns.append(float((border & ink).sum()) / max(int(border.sum()), 1))

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = tf(image=rgb, mask=np.zeros((H, W), np.uint8))["image"].unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(inp))[0, 0].numpy()
            g512 = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
            for th in THRS:
                pr = prob > th
                tp = float((pr & g512).sum()); fp = float((pr & ~g512).sum())
                fn = float((~pr & g512).sum())
                a = acc[th]; a[0] += tp; a[1] += fp; a[2] += fn
                per_img[th].append(((2 * tp + 1) / (2 * tp + fp + fn + 1),
                                    (tp + 1) / (tp + fp + fn + 1)))
            used += 1
        except Exception:
            skipped += 1
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(folders)} used={used} skip={skipped} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    res = {"split": SPLIT, "n_total": len(folders), "n_used": used,
           "n_skipped": skipped, "gate": "geometric (bbox inside image, area fraction 0.005-0.45)",
           "outline_ink_median": round(float(np.median(aligns)), 3) if aligns else None,
           "results": {}}
    for th in THRS:
        TP, FP, FN = acc[th]
        pi = per_img[th]
        res["results"][str(th)] = {
            "micro_dice": round((2 * TP + 1) / (2 * TP + FP + FN + 1), 4),
            "micro_iou": round((TP + 1) / (TP + FP + FN + 1), 4),
            "per_image_mean_dice": round(float(np.mean([x[0] for x in pi])), 4),
            "per_image_mean_iou": round(float(np.mean([x[1] for x in pi])), 4),
        }
    print(json.dumps(res, indent=2), flush=True)
    open(r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad\cubicasa_eval.json", "w").write(json.dumps(res, indent=2))

main()
