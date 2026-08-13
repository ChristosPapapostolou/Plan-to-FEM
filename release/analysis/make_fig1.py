# -*- coding: utf-8 -*-
"""Comment 19: redraw Fig. 1 as a vector pipeline figure that distinguishes
learned outputs, rule-based outputs, user-prescribed assumptions, and the
boundary beyond which no FE-solver verification is performed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad\fig1_pipeline"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

LEARN_F, LEARN_E = "#D6E4F2", "#1F4E79"
RULE_F,  RULE_E  = "#FCE6C9", "#A55A00"
ASSUM_F, ASSUM_E = "#DDEBDF", "#3D6B4A"
OUT_F,   OUT_E   = "#F2F2F2", "#8A8A8A"
RED = "#B00000"

fig, ax = plt.subplots(figsize=(6.9, 3.85))
ax.set_xlim(0, 100); ax.set_ylim(1.5, 59.5); ax.axis("off")

def box(x, y, w, h, title, sub, fill, edge, ls="solid", tcol=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=fill, edgecolor=edge, linewidth=1.0, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center",
            fontsize=6.9, fontweight="bold", color=tcol or edge, zorder=3,
            linespacing=1.25)
    ax.text(x + w / 2, y + h / 2 - 2.7, sub, ha="center", va="center",
            fontsize=6.2, color="#222222", zorder=3, linespacing=1.35)

def arrow(x1, y1, x2, y2, color="#444444", ls="solid", lw=1.1, ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))

# ---------------------------------------------------------------- main chain
BW, GAP, X0 = 10.2, 1.25, 1.0
BY, BH = 33.8, 13.4
TOP = BY + BH
stages = [
    ("Stage 1",  "Wall\nsegmentation", "MiT-B4 U-Net\n(SCSE decoder)",        "L", "binary wall mask"),
    ("Stage 2",  "Vectorisation",      "skeleton + Hough;\nscale estimation", "R", "vectorised floor"),
    ("Stage 3",  "Graph\nconstruction","Edge-PSW-DW;\ngrid detection",        "R", "typed graph"),
    ("Stage 4a", "GNN\nenrichment",    "shear-wall ratios;\ncolumn probabilities", "L", "enriched graph"),
    ("Stage 4b", "Rule\nvalidation",   "selector, budget,\nNMS, fallback",    "R", "validated layout"),
    ("Stage 5",  "FE export",          "extrusion, sizing,\nETABS COM / JSON","R", "FE model (JSON)"),
    ("",         "Coherence\nchecks",  "six heuristic\nstructural checks",    "R", "coherence report"),
]
cx = []
for i, (stg, title, sub, kind, artifact) in enumerate(stages):
    x = X0 + i * (BW + GAP)
    cx.append(x + BW / 2)
    f, e = (LEARN_F, LEARN_E) if kind == "L" else (RULE_F, RULE_E)
    box(x, BY, BW, BH, title, sub, f, e)
    if stg:
        ax.text(x + BW / 2, TOP + 2.1, stg, ha="center", va="center",
                fontsize=6.2, color="#555555", fontweight="bold")
    ax.text(x + BW / 2, BY - 3.4, artifact, ha="center", va="center",
            fontsize=5.9, color="#333333", style="italic")
    if i:
        arrow(x - GAP - 0.15, BY + BH / 2, x - 0.25, BY + BH / 2)

# input
ax.text(X0, 55.4, "raster floor-plan image", ha="left", va="center",
        fontsize=6.6, color=LEARN_E, fontweight="bold")
arrow(cx[0], 53.9, cx[0], TOP + 4.4, color=LEARN_E)

# artifact rule
ax.plot([X0, X0 + 7 * BW + 6 * GAP], [BY - 6.1, BY - 6.1], color="#BBBBBB",
        lw=0.7, zorder=1)
ax.text(X0, BY - 8.4, "every stage emits one inspectable artifact",
        ha="left", va="center", fontsize=5.9, color="#777777", style="italic")

# ---------------------------------------------------------------- verification boundary
XB = X0 + 7 * BW + 6 * GAP + 1.7
ax.plot([XB, XB], [14.0, 52.0], color=RED, lw=1.2, ls=(0, (4, 3)), zorder=3)
ax.text(XB, 53.6, "automated pipeline ends", ha="center", va="center",
        fontsize=6.0, color=RED, fontweight="bold")

box(XB + 2.4, BY, 14.0, BH, "Engineer review\n& FE analysis",
    "loads, mass source,\nload cases; solver run",
    OUT_F, OUT_E, ls=(0, (3, 2)), tcol="#5A5A5A")
ax.text(XB + 2.4 + 7.0, BY - 3.4, "not performed in this study",
        ha="center", va="center", fontsize=5.9, color=RED, fontweight="bold")
arrow(cx[-1] + BW / 2 + 0.2, BY + BH / 2, XB + 2.1, BY + BH / 2,
      color="#909090", ls=(0, (3, 2)))

# ---------------------------------------------------------------- assumptions band
AX0, AY, AH = X0, 11.5, 11.6
AW = 7 * BW + 6 * GAP
ax.add_patch(FancyBboxPatch(
    (AX0, AY), AW, AH, boxstyle="round,pad=0,rounding_size=1.0",
    facecolor=ASSUM_F, edgecolor=ASSUM_E, linewidth=1.0, zorder=2))
ax.text(AX0 + AW / 2, AY + AH - 3.0,
        "User-prescribed assumptions and fixed thresholds",
        ha="center", va="center", fontsize=6.9, fontweight="bold", color=ASSUM_E)
ax.text(AX0 + AW / 2, AY + 4.2,
        "0.20 m wall-thickness prior   |   0.06 m PSW threshold, grid tolerances   |   "
        "2% shear-wall budget, 1.5 m minimum pier   |   thresholds 0.40 / 0.50\n"
        "number of storeys and storey height   |   concrete grade   |   "
        "12 kPa column-sizing intensity   |   wall and column modes",
        ha="center", va="center", fontsize=5.9, color="#222222", linespacing=1.5)

for i in (1, 2, 4, 5):
    arrow(cx[i], AY + AH + 0.2, cx[i], BY - 9.6, color=ASSUM_E, lw=0.9, ms=6)

# ---------------------------------------------------------------- legend
LY = 5.0
def swatch(x, fill, edge, label, ls="solid"):
    ax.add_patch(Rectangle((x, LY - 1.3), 2.9, 2.6, facecolor=fill,
                           edgecolor=edge, linewidth=1.0, linestyle=ls, zorder=3))
    ax.text(x + 3.7, LY, label, ha="left", va="center", fontsize=6.3,
            color="#222222")

swatch(1.0,  LEARN_F, LEARN_E, "learned output")
swatch(21.0, RULE_F,  RULE_E,  "rule-based / deterministic output")
swatch(52.0, ASSUM_F, ASSUM_E, "user-prescribed assumption")
swatch(78.0, OUT_F,   OUT_E,   "outside the pipeline", ls=(0, (3, 2)))

fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
fig.savefig(OUT + ".pdf", dpi=600)
fig.savefig(OUT + ".svg")
fig.savefig(OUT + ".png", dpi=600, facecolor="white")

from PIL import Image
im = Image.open(OUT + ".png")
print("png px:", im.size, "aspect:", round(im.size[0] / im.size[1], 4))
im.resize((1380, int(1380 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_preview.png")
