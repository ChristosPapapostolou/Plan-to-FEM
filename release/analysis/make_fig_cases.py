# -*- coding: utf-8 -*-
"""Comment 45: one success and one failure at readable size, replacing the
two multi-panel galleries in the main text."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, cv2

ROOT = r"C:\Dev\Plan_2_FEM_2026"
SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
OUT = os.path.join(SCR, "fig_cases")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

CASES = [("floor_plan_9", "highest-scoring model"),
         ("floor_plan_1", "lowest-scoring model")]
PANELS = [("source drawing", None),
          ("Stage 1 wall mask", "output_mask.png"),
          ("Stage 4 shear walls and columns", "floor_predictions_vis.png"),
          ("Stage 5 three-storey model", "model_fem_vis.png")]


def load(path):
    im = cv2.imread(path)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def trim(im, pad=6):
    """Crop uniform white margins so each panel fills its axes."""
    if im is None: return None
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    m = g < 245
    if not m.any(): return im
    ys, xs = np.where(m)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, im.shape[0] - 1)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, im.shape[1] - 1)
    return im[y0:y1 + 1, x0:x1 + 1]


fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.25))
for r, (plan, role) in enumerate(CASES):
    d = os.path.join(ROOT, "results", plan)
    rep = json.load(open(os.path.join(d, "sanity_report.json")))
    score = rep["score"]; nf = rep["n_fail"]; nw = rep["n_warn"]
    fails = [c["name"].replace("_", " ") for c in rep["checks"] if c["status"] == "fail"]
    for c, (title, fname) in enumerate(PANELS):
        ax = axes[r, c]
        if fname is None:
            src = os.path.join(d, plan + ".png")
            if not os.path.exists(src):
                src = os.path.join(ROOT, "images", plan + ".png")
            im = trim(load(src))
        else:
            im = trim(load(os.path.join(d, fname)))
        if im is not None:
            ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_color("#BBBBBB")
        if r == 0:
            ax.set_title(title, fontsize=6.6, pad=3.5)
    lab = f"{plan}\n{role}\ncoherence {score:.2f}"
    if nf:
        lab += f"\n{nf} failing: {', '.join(fails)}"
    else:
        lab += f"\n{nw} warning{'s' if nw != 1 else ''}, none failing"
    axes[r, 0].set_ylabel(lab, fontsize=6.0, rotation=0, ha="right", va="center",
                          labelpad=5, linespacing=1.6)

fig.subplots_adjust(left=0.215, right=0.995, top=0.935, bottom=0.012,
                    wspace=0.04, hspace=0.05)
fig.savefig(OUT + ".pdf", dpi=600)
fig.savefig(OUT + ".svg")
fig.savefig(OUT + ".png", dpi=600, facecolor="white")
from PIL import Image
im = Image.open(OUT + ".png")
print("px", im.size, "aspect", round(im.size[0] / im.size[1], 3))
im.resize((1300, int(1300 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_preview.png")
