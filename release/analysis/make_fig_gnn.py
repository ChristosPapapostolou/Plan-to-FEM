# -*- coding: utf-8 -*-
"""Comment 25: compact architecture diagram of GNN-EP-Multi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad\fig_gnn_arch"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

BB_F, BB_E = "#D6E4F2", "#1F4E79"     # message passing
NM_F, NM_E = "#EDE3F5", "#5B3E86"     # normalisation / dropout
HD_F, HD_E = "#FCE6C9", "#A55A00"     # heads
IN_F, IN_E = "#E8E8E8", "#666666"     # inputs

fig, ax = plt.subplots(figsize=(6.9, 3.15))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

def box(x, y, w, h, title, sub, fill, edge, fs=6.0, fst=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor=fill, edgecolor=edge, lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
            fontsize=fst, fontweight="bold", color=edge, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center",
                fontsize=fs, color="#222222", zorder=3, linespacing=1.3)

def arrow(x1, y1, x2, y2, color="#444444", ls="solid", lw=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))

# inputs
box(1.0, 27.0, 13.5, 9.5, "Node features", "11-d\nper node", IN_F, IN_E)
box(1.0, 14.0, 13.5, 9.5, "Edge features", "9-d\nper edge", IN_F, IN_E)

# message-passing stack
xs = [17.5, 28.0, 38.5, 49.0, 59.5, 70.0]
widths = ["11>16", "16>16", "16>32", "32>32", "32>32", "32>32"]
for i, (x, wd) in enumerate(zip(xs, widths)):
    box(x, 21.0, 9.0, 11.0, f"SF {i+1}", wd, BB_F, BB_E, fs=5.8, fst=6.4)
    if i:
        arrow(x - 1.5, 26.5, x - 0.3, 26.5)
arrow(14.7, 31.5, 17.3, 27.5, color=IN_E)
arrow(14.7, 18.5, 17.3, 25.0, color=IN_E, ls=(0, (3, 2)))

# norm/dropout markers after layers 2 and 4
for x in (37.2, 58.2):
    ax.add_patch(FancyBboxPatch((x, 33.4), 9.8, 6.0,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=NM_F, edgecolor=NM_E, lw=0.9, zorder=2))
    ax.text(x + 4.9, 36.4, "LayerNorm\n+ Dropout 0.1", ha="center", va="center",
            fontsize=5.6, color=NM_E, linespacing=1.25, zorder=3)
    arrow(x + 4.9, 33.2, x + 4.9, 32.2, color=NM_E, lw=0.9)

# heads
box(81.0, 30.5, 18.0, 11.5, "Edge head (MLP)",
    "[h_u | e_uv | h_v] -> 73\n32-32-16-8 -> 3\nsigmoid(r_l, r_r), SW logit",
    HD_F, HD_E, fs=5.5)
box(81.0, 12.5, 18.0, 11.5, "Node head (MLP)",
    "[h_v | x_v] -> 43\n32-16-8 -> 1\ncolumn logit",
    HD_F, HD_E, fs=5.5)
arrow(79.2, 26.5, 80.7, 34.0)
arrow(79.2, 26.5, 80.7, 20.0)

# aggregation formula
ax.text(50.0, 8.6,
        "SF layer:   h'$_v$ = ReLU( W$_s$ h$_v$  +  W$_n$ · mean$_{u \\in N(v)}$ h$_u$"
        "  +  mean$_{u \\in N(v)}$ W$_e$ e$_{uv}$ )",
        ha="center", va="center", fontsize=6.6, color="#222222")
ax.text(50.0, 4.4,
        "aggregation is bidirectional (each stored edge is traversed both ways); "
        "no residual connections; 16,340 parameters",
        ha="center", va="center", fontsize=5.9, color="#666666", style="italic")

fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
fig.savefig(OUT + ".pdf", dpi=600)
fig.savefig(OUT + ".svg")
fig.savefig(OUT + ".png", dpi=600, facecolor="white")
from PIL import Image
im = Image.open(OUT + ".png")
print("px", im.size, "aspect", round(im.size[0] / im.size[1], 4))
im.resize((1300, int(1300 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_preview.png")
