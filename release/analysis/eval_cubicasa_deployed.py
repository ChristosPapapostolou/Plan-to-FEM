# -*- coding: utf-8 -*-
"""CubiCasa5K test split, replicating the Stage-1 DEPLOYMENT path exactly
(stages/stage_1/stage_1.py): predict at 512, threshold 0.40, upsample to the
original resolution with nearest neighbour, and score there.

Three variants:
  A raw            -> written as output_mask.png and consumed by Stage 2
  B raw+filter     -> output_mask_raw_filtered.png (saved, not consumed)
  C thin+filter    -> output_mask_enhanced_filtered.png (saved, not consumed)
"""
import sys, re, json, time
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\finetune_stage1")
import numpy as np, cv2, torch
from xml.dom import minidom
import finetune as F

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
sys.path.insert(0, SCR)
exec(open(SCR + r"\eval_cubicasa.py", encoding="utf-8").read().split("def main()")[0])

ROOT = r"C:\Dev\Plan_2_FEM_2026\cubicasa5k\cubicasa5k"
CKPT = r"C:\Dev\Plan_2_FEM_2026\stages\stage_1\mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
PROB_THRESHOLD = 0.4        # stage_1.py
MIN_COMPONENT_AREA = 2000   # stage_1.py, applied at ORIGINAL resolution

def thin_wall_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, all_walls = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(all_walls, cv2.DIST_L2, 5)
    thick_seed = (dist >= 2).astype(np.uint8) * 255
    if cv2.countNonZero(thick_seed) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thick_restored = cv2.dilate(thick_seed, k)
    else:
        thick_restored = np.zeros_like(all_walls)
    return cv2.bitwise_and(all_walls, cv2.bitwise_not(thick_restored))

def suppress_thin(image_bgr):
    out = image_bgr.copy()
    out[thin_wall_mask(image_bgr) > 0] = [255, 255, 255]
    return out

def postprocess(mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    clean = np.zeros_like(mask_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255
    return clean

model = F.build_model(CKPT, "cpu"); model.eval()
tf = F.val_transform()

def run_model(img_bgr, W, H):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = tf(image=rgb, mask=np.zeros(img_bgr.shape[:2], np.uint8))["image"].unsqueeze(0)
    with torch.no_grad():
        p = torch.sigmoid(model(t))
    m = (p > PROB_THRESHOLD).float().squeeze().numpy()
    return (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)

VARIANTS = ["A_raw_consumed", "B_raw_filtered", "C_thin_filtered"]
acc = {v: [0.0, 0.0, 0.0] for v in VARIANTS}
per_img = {v: [] for v in VARIANTS}
folders = [l.strip().strip("/") for l in open(f"{ROOT}\\test.txt") if l.strip()]
used = skipped = 0
t0 = time.time()

for k, rel in enumerate(folders):
    d = f"{ROOT}\\{rel.replace('/', chr(92))}"
    try:
        img = cv2.imread(d + r"\F1_scaled.png")
        if img is None: skipped += 1; continue
        polys = wall_polygons(d + r"\model.svg")
        if not polys: skipped += 1; continue
        H, W = img.shape[:2]
        gt = np.zeros((H, W), np.uint8)
        for p in polys:
            cv2.fillPoly(gt, [np.round(p).astype(np.int32)], 255)
        if (gt > 0).sum() < 500: skipped += 1; continue
        allp = np.concatenate(polys, axis=0)
        inside = ((allp[:, 0] >= 0) & (allp[:, 0] < W) &
                  (allp[:, 1] >= 0) & (allp[:, 1] < H)).mean()
        af = float((gt > 0).sum()) / float(H * W)
        if (1.0 - inside) > 0.02 or not (0.005 <= af <= 0.45):
            skipped += 1; continue

        raw = run_model(img, W, H)
        enh = run_model(suppress_thin(img), W, H)
        preds = {"A_raw_consumed": raw,
                 "B_raw_filtered": postprocess(raw),
                 "C_thin_filtered": postprocess(enh)}
        g = gt > 0
        for v, m in preds.items():
            pr = m > 0
            tp = float((pr & g).sum()); fp = float((pr & ~g).sum()); fn = float((~pr & g).sum())
            a = acc[v]; a[0] += tp; a[1] += fp; a[2] += fn
            per_img[v].append(((2*tp+1)/(2*tp+fp+fn+1), (tp+1)/(tp+fp+fn+1)))
        used += 1
    except Exception:
        skipped += 1
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{len(folders)} used={used} ({time.time()-t0:.0f}s)", flush=True)

res = {"split": "test.txt", "n_used": used, "n_skipped": skipped,
       "resolution": "original (deployment)", "threshold": PROB_THRESHOLD,
       "results": {}}
for v in VARIANTS:
    TP, FP, FN = acc[v]; pi = per_img[v]
    res["results"][v] = {
        "micro_dice": round((2*TP+1)/(2*TP+FP+FN+1), 4),
        "micro_iou": round((TP+1)/(TP+FP+FN+1), 4),
        "per_image_mean_dice": round(float(np.mean([x[0] for x in pi])), 4),
        "per_image_mean_iou": round(float(np.mean([x[1] for x in pi])), 4),
    }
print(json.dumps(res, indent=2), flush=True)
open(SCR + r"\cubicasa_deployed.json", "w").write(json.dumps(res, indent=2))
