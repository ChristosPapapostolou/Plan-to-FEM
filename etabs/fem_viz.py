"""
Stage 5 - FEM Model Viewer
==========================
Renders a `model_fem.json` (the Stage 5 export) to a validation image so the
3D structural model can be checked WITHOUT opening ETABS.

Produces a two-panel figure:
  - left  : plan view of one representative story (wall piers, slab outline,
            beams, nodes)
  - right : 3D isometric of the whole building (extruded walls, floor slabs,
            beams)

Usage:
  .\\venv\\Scripts\\python.exe etabs\\fem_viz.py -i model_fem.json -o model_fem_vis.png
"""

from __future__ import annotations
import argparse
import json
import logging

logger = logging.getLogger(__name__)

# Colours
SW_COLOR = "#d62728"     # shear wall (red)
NW_COLOR = "#8c8c8c"     # non-structural wall (grey)
BEAM_COLOR = "#1f77b4"   # beam (blue)
COL_COLOR = "#2ca02c"    # column (green)
SLAB_COLOR = "#5fa8d3"   # slab (light blue)
NODE_COLOR = "#333333"


def _wall_lw(thickness_m: float) -> float:
    """Plan-view line width scaled to wall thickness."""
    return max(1.5, thickness_m * 1000.0 / 30.0)


def visualize_fem(model: dict, out_path: str, plan_story: int = 0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    except ImportError:
        logger.warning("matplotlib not available - cannot render")
        return False

    nodes = {n["id"]: (n["x"], n["y"], n["z"]) for n in model.get("nodes", [])}
    panels = model.get("wall_panels", [])
    slabs = model.get("slabs", [])
    beams = [m for m in model.get("frame_members", []) if m.get("type") == "beam"]
    columns = [m for m in model.get("frame_members", []) if m.get("type") == "column"]
    meta = model.get("metadata", {})
    summ = model.get("summary", {})

    if not nodes:
        logger.warning("model has no nodes - nothing to draw")
        return False

    fig = plt.figure(figsize=(18, 9))
    ax_plan = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    # ------------------------------------------------------------------
    # Plan view: one representative story
    # ------------------------------------------------------------------
    n_sw_plan = 0
    for wp in panels:
        if wp.get("story", 0) != plan_story:
            continue
        nd = wp["nodes"]
        if len(nd) < 4:
            continue
        bl, br = nodes[nd[0]], nodes[nd[1]]  # bottom edge of the wall
        is_sw = wp.get("is_shear_wall", False)
        n_sw_plan += int(is_sw)
        ax_plan.plot([bl[0], br[0]], [bl[1], br[1]],
                     color=SW_COLOR if is_sw else NW_COLOR,
                     lw=_wall_lw(wp.get("thickness_m", 0.15)),
                     solid_capstyle="round", zorder=4)

    for sl in slabs:
        if sl.get("story", 0) != plan_story:
            continue
        xy = [(nodes[nid][0], nodes[nid][1]) for nid in sl["nodes"]]
        ax_plan.fill([p[0] for p in xy], [p[1] for p in xy],
                     color=SLAB_COLOR, alpha=0.18, zorder=1)
        ax_plan.plot([p[0] for p in xy] + [xy[0][0]],
                     [p[1] for p in xy] + [xy[0][1]],
                     color=SLAB_COLOR, lw=1.2, ls="--", zorder=2)

    for b in beams:
        if b.get("story", 0) != plan_story:
            continue
        s, e = nodes[b["start_node"]], nodes[b["end_node"]]
        ax_plan.plot([s[0], e[0]], [s[1], e[1]],
                     color=BEAM_COLOR, lw=1.6, alpha=0.9, zorder=3)

    # columns of this story: square markers at their plan location
    cxs, cys = [], []
    for c in columns:
        if c.get("story", 0) != plan_story:
            continue
        s0 = nodes[c["start_node"]]
        cxs.append(s0[0]); cys.append(s0[1])
    if cxs:
        ax_plan.scatter(cxs, cys, s=46, marker="s", color=COL_COLOR,
                        edgecolors="k", linewidths=0.5, zorder=6)

    # nodes at this story's top (slab/beam level)
    nx = [v[0] for v in nodes.values()]
    ny = [v[1] for v in nodes.values()]
    ax_plan.scatter(nx, ny, s=6, color=NODE_COLOR, zorder=5)

    ax_plan.set_aspect("equal", adjustable="datalim")
    ax_plan.set_title(f"Plan - Story {plan_story + 1}", fontsize=12)
    ax_plan.set_xlabel("X (m)")
    ax_plan.set_ylabel("Y (m)")
    ax_plan.grid(True, ls=":", alpha=0.4)
    ax_plan.legend(handles=[
        Line2D([0], [0], color=SW_COLOR, lw=4, label="Shear wall (pier)"),
        Line2D([0], [0], color=NW_COLOR, lw=3, label="Non-structural wall"),
        Line2D([0], [0], color=BEAM_COLOR, lw=2, label="Beam"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_COLOR,
               markeredgecolor="k", markersize=8, label="Column"),
        Patch(facecolor=SLAB_COLOR, alpha=0.3, label="Slab"),
    ], loc="upper right", fontsize=8)

    # ------------------------------------------------------------------
    # 3D view: whole building
    # ------------------------------------------------------------------
    wall_quads, wall_colors = [], []
    for wp in panels:
        nd = wp["nodes"]
        if len(nd) < 4:
            continue
        wall_quads.append([nodes[i] for i in nd[:4]])
        wall_colors.append(SW_COLOR if wp.get("is_shear_wall") else NW_COLOR)
    if wall_quads:
        pc = Poly3DCollection(wall_quads, facecolors=wall_colors,
                              edgecolors="k", linewidths=0.2, alpha=0.85)
        ax_3d.add_collection3d(pc)

    slab_polys = [[nodes[nid] for nid in sl["nodes"]] for sl in slabs]
    if slab_polys:
        sc = Poly3DCollection(slab_polys, facecolors=SLAB_COLOR,
                              edgecolors=BEAM_COLOR, linewidths=0.4, alpha=0.30)
        ax_3d.add_collection3d(sc)

    beam_segs = [[nodes[b["start_node"]], nodes[b["end_node"]]] for b in beams]
    if beam_segs:
        ax_3d.add_collection3d(Line3DCollection(
            beam_segs, colors=BEAM_COLOR, linewidths=1.3))

    col_segs = [[nodes[c["start_node"]], nodes[c["end_node"]]] for c in columns]
    if col_segs:
        ax_3d.add_collection3d(Line3DCollection(
            col_segs, colors=COL_COLOR, linewidths=2.2))

    xs = [v[0] for v in nodes.values()]
    ys = [v[1] for v in nodes.values()]
    zs = [v[2] for v in nodes.values()]
    ax_3d.set_xlim(min(xs), max(xs))
    ax_3d.set_ylim(min(ys), max(ys))
    ax_3d.set_zlim(min(zs), max(zs))
    dx, dy, dz = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1, (max(zs) - min(zs)) or 1
    ax_3d.set_box_aspect((dx, dy, dz))
    ax_3d.view_init(elev=22, azim=-58)
    ax_3d.set_title("3D model", fontsize=12)
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")

    title = (f"FEM model - {meta.get('n_stories', '?')} stories x "
             f"{meta.get('floor_height_m', '?')}m  |  "
             f"{summ.get('n_shear_walls', 0)} SW panels, "
             f"{summ.get('n_beams', 0)} beams, "
             f"{summ.get('n_columns', 0)} columns, "
             f"{summ.get('n_slabs', 0)} slabs, "
             f"{summ.get('n_diaphragms', 0)} diaphragms  |  "
             f"wall_mode={meta.get('wall_mode', '?')}")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved FEM visualization to %s", out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description="Render a Stage 5 FEM model JSON to an image")
    ap.add_argument("-i", "--input", default="model_fem.json",
                    help="FEM model JSON (Stage 5 export)")
    ap.add_argument("-o", "--output", default="model_fem_vis.png",
                    help="Output image path")
    ap.add_argument("--plan-story", type=int, default=0,
                    help="Story index (0-based) to draw in the plan view")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    with open(args.input) as f:
        model = json.load(f)
    visualize_fem(model, args.output, plan_story=args.plan_story)


if __name__ == "__main__":
    main()
