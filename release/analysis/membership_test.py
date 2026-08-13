# -*- coding: utf-8 -*-
"""Was the deployed checkpoint trained on CubiCasa5K?

Membership signal: a model trained on the train split almost always scores
noticeably higher there than on the held-out test split. If train ~= test, the
model most likely never saw this corpus.

Same protocol as the earlier test-split run (512 space, threshold 0.4/0.5).
"""
import sys, json, time
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\finetune_stage1")
import numpy as np, cv2, torch
import finetune as F

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
exec(open(SCR + r"\eval_cubicasa.py", encoding="utf-8").read().split("def main()")[0])

ROOT = r"C:\Dev\Plan_2_FEM_2026\cubicasa5k\cubicasa5k"
CKPT = r"C:\Dev\Plan_2_FEM_2026\stages\stage_1\mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
STRIDE = 17          # deterministic spread over the whole train split
THRS = [0.4, 0.5]

model = F.build_model(CKPT, "cpu"); model.eval()
tf = F.val_transform()

def evaluate(split, stride=1, cap=None):
    folders = [l.strip().strip("/") for l in open(f"{ROOT}\\{split}") if l.strip()]
    folders = folders[::stride]
    if cap: folders = folders[:cap]
    acc = {th: [0.0, 0.0, 0.0] for th in THRS}
    per = {th: [] for th in THRS}
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
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = tf(image=rgb, mask=np.zeros((H, W), np.uint8))["image"].unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(inp))[0, 0].numpy()
            g = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST) > 127
            for th in THRS:
                pr = prob > th
                tp = float((pr & g).sum()); fp = float((pr & ~g).sum()); fn = float((~pr & g).sum())
                a = acc[th]; a[0] += tp; a[1] += fp; a[2] += fn
                per[th].append((tp + 1) / (tp + fp + fn + 1))
            used += 1
        except Exception:
            skipped += 1
        if (k + 1) % 50 == 0:
            print(f"  [{split}] {k+1}/{len(folders)} used={used} ({time.time()-t0:.0f}s)", flush=True)
    out = {"split": split, "n_used": used, "n_skipped": skipped}
    for th in THRS:
        TP, FP, FN = acc[th]
        out[f"micro_iou@{th}"] = round((TP + 1) / (TP + FP + FN + 1), 4)
        out[f"mean_iou@{th}"] = round(float(np.mean(per[th])), 4)
    return out

res = {"train_sample": evaluate("train.txt", stride=STRIDE),
       "val_full": evaluate("val.txt", stride=1)}
print(json.dumps(res, indent=2), flush=True)
open(SCR + r"\membership.json", "w").write(json.dumps(res, indent=2))

t = res["train_sample"]["micro_iou@0.4"]
v = res["val_full"]["micro_iou@0.4"]
print(f"\ntrain={t}  val={v}  test=0.5919  (test measured earlier, same protocol)")
print(f"train-minus-test gap: {t - 0.5919:+.4f}")
