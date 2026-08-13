# -*- coding: utf-8 -*-
"""Comment 20: evaluate the DEPLOYED Stage-1 checkpoint on the documented
72-pair held-out split, using the repository's own dataset, transform and
metric definitions (finetune_stage1/finetune.py)."""
import sys, json, time
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\finetune_stage1")

import torch
from torch.utils.data import DataLoader
import finetune as F

DATA = r"C:\Dev\Plan_2_FEM_2026\train\data"
CKPT = r"C:\Dev\Plan_2_FEM_2026\stages\stage_1\mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
DEV = "cpu"

ds = F.ShearWallDataset(DATA, "test", F.val_transform())
print(f"test pairs: {len(ds)}", flush=True)

model = F.build_model(CKPT, DEV)
model.eval()

# ---- per-image metrics + dataset-level (micro) accumulation
per_img_dice, per_img_iou = [], []
TP = FP = FN = 0.0
t0 = time.time()
loader = DataLoader(ds, batch_size=1, shuffle=False)
with torch.no_grad():
    for k, (img, mask) in enumerate(loader):
        logits = model(img.to(DEV))
        d, i = F.dice_iou(logits, mask.to(DEV))
        per_img_dice.append(d); per_img_iou.append(i)
        pred = (torch.sigmoid(logits) > 0.5).float()
        t = (mask > 0.5).float()
        TP += (pred * t).sum().item()
        FP += (pred * (1 - t)).sum().item()
        FN += ((1 - pred) * t).sum().item()
        if (k + 1) % 12 == 0:
            print(f"  {k+1}/{len(ds)}  ({time.time()-t0:.0f}s)", flush=True)

n = len(per_img_dice)
mean_dice = sum(per_img_dice) / n
mean_iou = sum(per_img_iou) / n
micro_dice = (2 * TP + 1) / (2 * TP + FP + FN + 1)
micro_iou = (TP + 1) / (TP + FP + FN + 1)

# ---- batch-size-2 convention, matching the training log's reported numbers
loader2 = DataLoader(ds, batch_size=2, shuffle=False)
tot_d = tot_i = 0.0
with torch.no_grad():
    for img, mask in loader2:
        logits = model(img.to(DEV))
        d, i = F.dice_iou(logits, mask.to(DEV))
        tot_d += d * img.size(0); tot_i += i * img.size(0)
b2_dice, b2_iou = tot_d / n, tot_i / n

res = {
    "checkpoint": CKPT,
    "split": "test (held out)", "n_images": n,
    "per_image_mean_dice": round(mean_dice, 4),
    "per_image_mean_iou": round(mean_iou, 4),
    "dataset_micro_dice": round(micro_dice, 4),
    "dataset_micro_iou": round(micro_iou, 4),
    "batch2_convention_dice": round(b2_dice, 4),
    "batch2_convention_iou": round(b2_iou, 4),
    "elapsed_s": round(time.time() - t0, 1),
}
print(json.dumps(res, indent=2), flush=True)
open(r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad\stage1_eval.json", "w").write(json.dumps(res, indent=2))
