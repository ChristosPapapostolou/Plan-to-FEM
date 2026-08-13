# -*- coding: utf-8 -*-
"""Which target/threshold reproduces 0.8864?  One forward pass per image,
scored against three candidate targets at two thresholds."""
import sys, json
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\finetune_stage1")
import numpy as np, cv2, torch
import finetune as F

CKPT = r"C:\Dev\Plan_2_FEM_2026\stages\stage_1\mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
model = F.build_model(CKPT, "cpu"); model.eval()
tf = F.val_transform()

ds = F.ShearWallDataset(r"C:\Dev\Plan_2_FEM_2026\train\data", "test")
print("pairs:", len(ds.pairs), flush=True)

TARGETS = ["red_shear_wall_from_B", "all_dark_pixels_in_A", "all_non_white_in_B"]
THRS = [0.4, 0.5]
acc = {(t, th): [0.0, 0.0, 0.0] for t in TARGETS for th in THRS}   # TP, FP, FN

for k, (ap, bp) in enumerate(ds.pairs):
    ia, ib = cv2.imread(ap), cv2.imread(bp)
    h, w = ia.shape[:2]
    if w > 1.5 * h:
        ia, ib = ia[:, :w // 2], ib[:, :w // 2]

    image = cv2.cvtColor(ia, cv2.COLOR_BGR2RGB)
    inp = tf(image=image, mask=np.zeros(ia.shape[:2], np.uint8))["image"].unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp))[0, 0].numpy()

    gray = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
    masks = {
        "red_shear_wall_from_B": F.extract_sw_mask(ib),
        "all_dark_pixels_in_A": ((gray < 200) * 255).astype(np.uint8),
        "all_non_white_in_B": ((cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY) < 200) * 255).astype(np.uint8),
    }
    for name, m in masks.items():
        gt = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
        for th in THRS:
            pr = prob > th
            a = acc[(name, th)]
            a[0] += float((pr & gt).sum())
            a[1] += float((pr & ~gt).sum())
            a[2] += float((~pr & gt).sum())
    if (k + 1) % 24 == 0:
        print(f"  {k+1}/{len(ds.pairs)}", flush=True)

print(f"\n{'target':26s} {'thr':>5s} {'Dice':>8s} {'IoU':>8s}")
out = []
for t in TARGETS:
    for th in THRS:
        TP, FP, FN = acc[(t, th)]
        dice = (2 * TP + 1) / (2 * TP + FP + FN + 1)
        iou = (TP + 1) / (TP + FP + FN + 1)
        print(f"{t:26s} {th:5.2f} {dice:8.4f} {iou:8.4f}")
        out.append({"target": t, "threshold": th,
                    "dice": round(dice, 4), "iou": round(iou, 4)})
open(r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad\target_probe.json", "w").write(json.dumps(out, indent=2))
