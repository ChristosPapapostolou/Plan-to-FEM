# -*- coding: utf-8 -*-
"""Comment 28: replace the illegible enrichment figure with the actual Stage 4
output for the worked example, distinguishing GNN probability from final
rule-validated selection."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
OUT = os.path.join(SCR, "fig5_enrichment")
E = json.load(open(os.path.join(SCR, "fig5", "enriched.json")))
M = json.load(open(os.path.join(SCR, "fig5", "model.json")))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

pts = {}
for w in E["walls"]:
    for p in w.get("points", []):
        pts[p["id"]] = (p["x"], p["y"])

ep = E.get("edge_predictions", {})
other, psw, sel = [], [], []
scores = []
for w in E["walls"]:
    if w.get("rejected"):
        continue
    for r in w.get("rects", []):
        a, b = pts.get(r["p1_id"]), pts.get(r["p2_id"])
        if a is None or b is None:
            continue
        seg = [a, b]
        pred = ep.get(r["id"], {})
        if r.get("is_shear_wall"):
            sel.append(seg)
        if pred.get("is_psw"):
            psw.append(seg); scores.append(float(pred.get("gnn_score", 0.0)))
        else:
            other.append(seg)

# storey-0 columns from the exported model
nid = {n["id"]: (n["x"], n["y"], n["z"]) for n in M["nodes"]}
cols = []
for f in M["frame_members"]:
    if f.get("type") != "column" or f.get("story") != 0:
        continue
    x, y, _z = nid[f["start_node"]]
    cols.append((x, y))
cols = np.array(cols) if cols else np.zeros((0, 2))

allpts = np.array([p for s in (other + psw) for p in s]) if (other or psw) else np.zeros((1, 2))
w_m = allpts[:, 0].max() - allpts[:, 0].min()
h_m = allpts[:, 1].max() - allpts[:, 1].min()
agg = max(w_m / max(h_m, 1e-6), 0.1)

fig_w = 6.9
fig_h = min(max(fig_w / agg * 1.02 + 0.78, 2.4), 5.0)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

if other:
    ax.add_collection(LineCollection(other, colors="#C9C9C9", linewidths=1.0, zorder=1))

norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
cmap = matplotlib.colormaps["viridis"]
if psw:
    lc = LineCollection(psw, cmap=cmap, norm=norm, linewidths=2.0, zorder=2)
    lc.set_array(np.array(scores))
    ax.add_collection(lc)
if sel:
    ax.add_collection(LineCollection(sel, colors="#C0181C", linewidths=4.2,
                                     zorder=3, alpha=0.9))
if len(cols):
    ax.plot(cols[:, 0], cols[:, 1], "s", ms=3.6, mfc="#2E8B3D",
            mec="#14521f", mew=0.5, lw=0, zorder=4)

ax.set_aspect("equal")
ax.autoscale_view()
ax.set_xlabel("x (m)", fontsize=7)
ax.set_ylabel("y (m)", fontsize=7)
ax.tick_params(labelsize=6.5)
for s in ax.spines.values():
    s.set_linewidth(0.6); s.set_color("#999999")

cb = fig.colorbar(lc, ax=ax, fraction=0.030, pad=0.015)
cb.set_label("GNN shear-wall probability", fontsize=6.6)
cb.ax.tick_params(labelsize=6.2)
cb.outline.set_linewidth(0.5)

handles = [
    Line2D([], [], color="#C0181C", lw=3.6,
           label=f"selected shear wall after rule validation ({len(sel)})"),
    Line2D([], [], color=cmap(0.75), lw=2.0,
           label=f"potential shear wall, colour = GNN probability ({len(psw)})"),
    Line2D([], [], marker="s", color="none", mfc="#2E8B3D", mec="#14521f",
           ms=5.2, lw=0, label=f"placed column, storey 1 ({len(cols)})"),
]
if other:
    handles.insert(2, Line2D([], [], color="#C9C9C9", lw=1.4,
                             label=f"non-PSW segment ({len(other)})"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=3, fontsize=6.4, frameon=False, handlelength=2.4,
          columnspacing=1.6, borderaxespad=0.0)

fig.subplots_adjust(left=0.055, right=0.90, top=0.98, bottom=0.185)
fig.savefig(OUT + ".pdf", dpi=600)
fig.savefig(OUT + ".svg")
fig.savefig(OUT + ".png", dpi=600, facecolor="white")

from PIL import Image
im = Image.open(OUT + ".png")
print("px", im.size, "| psw", len(psw), "selected", len(sel), "other", len(other),
      "cols", len(cols))
print("plan extent m: %.1f x %.1f" % (w_m, h_m))
im.resize((1300, int(1300 * im.size[1] / im.size[0])), Image.LANCZOS).save(OUT + "_preview.png")
