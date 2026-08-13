# -*- coding: utf-8 -*-
"""Comment 36: threshold-free evaluation figure."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
D = json.load(open(os.path.join(SCR, "threshold_free.json")))
OUT = os.path.join(SCR, "fig_threshold_free")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

KEYS = ["StructGAN real (clean split)", "synthetic validation", "synthetic test (unseen)"]
SHORT = {"StructGAN real (clean split)": "StructGAN real",
         "synthetic validation": "synthetic val.",
         "synthetic test (unseen)": "synthetic test"}
COL = {"StructGAN real (clean split)": "#C0181C",
       "synthetic validation": "#1F4E79",
       "synthetic test (unseen)": "#2E8B3D"}

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.45))

# (a) layout precision-recall
a = ax[0]
for k in KEYS:
    pr = D[k]["layout_pr"]
    R = [p["R"] for p in pr]; P = [p["P"] for p in pr]
    o = np.argsort(R)
    a.plot(np.array(R)[o], np.array(P)[o], "-", color=COL[k], lw=1.4, label=SHORT[k])
    j = min(range(len(pr)), key=lambda i: abs(pr[i]["thr"] - 0.4))
    a.plot(pr[j]["R"], pr[j]["P"], "o", color=COL[k], ms=4, mec="white", mew=0.6)
a.set_xlabel("recall", fontsize=7); a.set_ylabel("precision", fontsize=7)
a.set_title("(a) column layout, threshold swept", fontsize=7.2)
a.set_xlim(0, 1); a.set_ylim(0, 1)
a.legend(fontsize=6, frameon=False, loc="lower left")

# (b) calibration of the column head
b = ax[1]
b.plot([0, 1], [0, 1], "--", color="#999999", lw=0.9)
for k in KEYS:
    pts = [(c[0], c[1]) for c in D[k]["node_reliability"] if c[1] is not None]
    if pts:
        x, y = zip(*pts)
        b.plot(x, y, "o-", color=COL[k], lw=1.2, ms=3,
               label=f"{SHORT[k]} (ECE {D[k]['node_ece']:.2f})")
b.set_xlabel("predicted probability", fontsize=7)
b.set_ylabel("observed frequency", fontsize=7)
b.set_title("(b) column head calibration", fontsize=7.2)
b.set_xlim(0, 1); b.set_ylim(0, 1)
b.legend(fontsize=5.8, frameon=False, loc="upper left")

# (c) F1 versus matching tolerance
c = ax[2]
for k in KEYS:
    ft = D[k]["f1_vs_tol"]
    c.plot([f["tol"] for f in ft], [f["F1"] for f in ft], "o-",
           color=COL[k], lw=1.3, ms=3.4, label=SHORT[k])
c.axvline(0.5, color="#999999", ls="--", lw=0.9)
c.set_xlabel("matching tolerance (m)", fontsize=7)
c.set_ylabel("layout F1", fontsize=7)
c.set_title("(c) F1 versus matching tolerance", fontsize=7.2)
c.set_ylim(0, 1)
c.legend(fontsize=6, frameon=False, loc="lower right")

for x in ax:
    x.tick_params(labelsize=6.2)
    for s in x.spines.values():
        s.set_linewidth(0.6); s.set_color("#999999")
    x.grid(alpha=0.25, lw=0.5)

fig.tight_layout(pad=0.45)
fig.savefig(OUT + ".pdf", dpi=600)
fig.savefig(OUT + ".svg")
fig.savefig(OUT + ".png", dpi=600, facecolor="white")
from PIL import Image
im = Image.open(OUT + ".png")
print("px", im.size, "aspect", round(im.size[0] / im.size[1], 3))
im.resize((1300, int(1300 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_preview.png")
