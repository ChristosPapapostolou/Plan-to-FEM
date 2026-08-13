# -*- coding: utf-8 -*-
"""Test the hypothesis that the headline 88.64 figure came from the 2-pair
train/data/val 'reference subset'."""
import sys, json, glob
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\finetune_stage1")
import torch, cv2, numpy as np
import finetune as F

CKPT = r"C:\Dev\Plan_2_FEM_2026\stages\stage_1\mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
VA = r"C:\Dev\Plan_2_FEM_2026\train\data\val\val_A"
VB = r"C:\Dev\Plan_2_FEM_2026\train\data\val\val_B"

model = F.build_model(CKPT, "cpu"); model.eval()
tf = F.val_transform()

rows = []
TP = FP = FN = 0.0
for ap in sorted(glob.glob(VA + r"\*.png")):
    name = ap.split("\\")[-1]
    bp = VB + "\\" + name
    ia, ib = cv2.imread(ap), cv2.imread(bp)
    h, w = ia.shape[:2]
    if w > 1.5 * h:
        ia, ib = ia[:, :w // 2], ib[:, :w // 2]
    image = cv2.cvtColor(ia, cv2.COLOR_BGR2RGB)
    mask = F.extract_sw_mask(ib)
    out = tf(image=image, mask=mask)
    img = out["image"].unsqueeze(0)
    m = (out["mask"].float() / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
    d, i = F.dice_iou(logits, m)
    rows.append({"image": name, "dice": round(d, 4), "iou": round(i, 4)})
    pred = (torch.sigmoid(logits) > 0.5).float(); t = (m > 0.5).float()
    TP += (pred * t).sum().item(); FP += (pred * (1 - t)).sum().item()
    FN += ((1 - pred) * t).sum().item()

res = {
    "reference_subset": "train/data/val", "n_images": len(rows), "per_image": rows,
    "mean_dice": round(sum(r["dice"] for r in rows) / len(rows), 4),
    "mean_iou": round(sum(r["iou"] for r in rows) / len(rows), 4),
    "micro_dice": round((2 * TP + 1) / (2 * TP + FP + FN + 1), 4),
    "micro_iou": round((TP + 1) / (TP + FP + FN + 1), 4),
}
print(json.dumps(res, indent=2))
open(r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad\stage1_refsubset.json", "w").write(json.dumps(res, indent=2))
