"""
Stage 5 - ETABS Export
=======================
Converts the Stage 4 enriched floor plan into a 3D ETABS structural model.

Takes the 2D wall layout with shear wall predictions and:
  1. Extrudes walls vertically using the floor height
  2. Defines concrete material and wall section properties
  3. Creates shear wall panels as ETABS Area objects (shell elements)
  4. Creates non-shear-wall segments as frame elements
  5. Handles partial shear walls via sw_ratio_left / sw_ratio_right
  6. Supports multi-story stacking

Data flow:
  Stage 4 enriched JSON  ->  ETABSHandler  ->  ETABS .edb model
                         ->  or standalone JSON for other FEM software

Usage:
  # With live ETABS connection
  from pipeline.stage5_etabs_export import ETABSExporter
  exporter = ETABSExporter(enriched_json, n_stories=10, floor_height=2.8)
  exporter.build_model()       # builds node/element lists
  exporter.push_to_etabs()     # sends to ETABS via COM

  # Without ETABS (export JSON for other FEM tools)
  exporter.export_json("model_fem.json")

References:
  [1] CSI ETABS API Documentation
  [2] Zhao et al., Adv. Eng. Informatics 55 (2023) 101886
"""

from __future__ import annotations
import json, logging, math, os, uuid
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Data classes for the 3D structural model
# ======================================================================

@dataclass
class Node3D:
    """A point in 3D space."""
    id: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self):
        return {"id": self.id, "x": round(self.x, 4),
                "y": round(self.y, 4), "z": round(self.z, 4)}


@dataclass
class WallPanel:
    """
    A rectangular wall panel (ETABS Area/Shell element).
    Defined by 4 corner node IDs (bottom-left, bottom-right, top-right, top-left).
    """
    id: str = ""
    nodes: list = field(default_factory=list)  # 4 node IDs
    section: str = ""
    thickness_m: float = 0.20
    is_shear_wall: bool = False
    story: int = 0
    wall_id: str = ""      # parent wall from Stage 2
    rect_id: str = ""      # parent rect from Stage 2

    def to_dict(self):
        return {
            "id": self.id, "type": "wall",
            "nodes": self.nodes, "section": self.section,
            "thickness_m": self.thickness_m,
            "is_shear_wall": self.is_shear_wall,
            "story": self.story,
        }


@dataclass
class FrameMember:
    """A beam/column frame element between two nodes."""
    id: str = ""
    start_node: str = ""
    end_node: str = ""
    section: str = ""
    member_type: str = "beam"  # beam, column, brace
    story: int = 0

    def to_dict(self):
        return {
            "id": self.id, "type": self.member_type,
            "start_node": self.start_node, "end_node": self.end_node,
            "section": self.section, "story": self.story,
        }


@dataclass
class Slab:
    """
    A floor slab panel (ETABS Area/Shell element) covering a story footprint.
    Defined by N boundary node IDs (CCW) and carries a rigid diaphragm.
    """
    id: str = ""
    nodes: list = field(default_factory=list)  # boundary node IDs (CCW)
    section: str = ""
    thickness_m: float = 0.15
    story: int = 0
    diaphragm: str = ""

    def to_dict(self):
        return {
            "id": self.id, "type": "slab",
            "nodes": self.nodes, "section": self.section,
            "thickness_m": self.thickness_m,
            "story": self.story, "diaphragm": self.diaphragm,
        }


@dataclass
class WallSection:
    """A wall section property (concrete shell)."""
    name: str = ""
    material: str = "CONC"
    thickness_m: float = 0.20
    is_shear_wall: bool = True

    def to_dict(self):
        return {"name": self.name, "material": self.material,
                "thickness_m": self.thickness_m}


@dataclass
class BeamSection:
    """A concrete beam frame section (depth x width)."""
    name: str = ""
    material: str = "CONC"
    width_m: float = 0.30
    depth_m: float = 0.50

    def to_dict(self):
        return {"name": self.name, "material": self.material,
                "width_m": self.width_m, "depth_m": self.depth_m}


@dataclass
class SlabSection:
    """A concrete floor slab section (shell)."""
    name: str = ""
    material: str = "CONC"
    thickness_m: float = 0.15

    def to_dict(self):
        return {"name": self.name, "material": self.material,
                "thickness_m": self.thickness_m}


@dataclass
class ColumnSection:
    """A concrete column frame section (square b x b)."""
    name: str = ""
    material: str = "CONC"
    width_m: float = 0.40
    depth_m: float = 0.40

    def to_dict(self):
        return {"name": self.name, "material": self.material,
                "width_m": self.width_m, "depth_m": self.depth_m}


# ======================================================================
# ETABS Exporter
# ======================================================================

class ETABSExporter:
    """
    Converts Stage 4 enriched output to a 3D structural model and
    optionally pushes it to ETABS via COM.

    Parameters
    ----------
    enriched_json : dict
        Stage 4 enriched floor JSON (with edge_predictions and rene_features).
    n_stories : int
        Number of stories to extrude.
    floor_height_m : float
        Floor-to-floor height (overrides JSON value if set).
    base_z : float
        Z coordinate of the ground floor base.
    concrete_fc_mpa : float
        Concrete compressive strength for material definition.
    sw_thickness_default : float
        Default shear wall thickness if not predicted.
    min_sw_ratio : float
        Minimum sw_ratio to create a shear wall panel.
    """

    def __init__(
        self,
        enriched_json: dict,
        *,
        n_stories: int = 1,
        floor_height_m: Optional[float] = None,
        base_z: float = 0.0,
        concrete_fc_mpa: float = 30.0,
        sw_thickness_default: float = 0.20,
        min_sw_ratio: float = 0.05,
        split_sw_by_ratio: bool = False,
        add_beams: bool = True,
        add_slabs: bool = True,
        add_diaphragm: bool = True,
        beam_mode: str = "perimeter_sw",
        beam_width_m: float = 0.30,
        beam_depth_m: float = 0.50,
        slab_thickness_m: float = 0.15,
        wall_mode: str = "structural",
        add_columns: bool = True,
        max_pier_len_m: float = 3.0,
        col_max_len_m: float = 1.2,
        col_size_m: float = 0.40,
        boundary_columns: bool = True,
        column_mode: str = "grid",
        max_span_m: float = 6.0,
        connect_joints: bool = True,
        max_beam_run_m: Optional[float] = None,
        principal_frame: bool = True,
        axis_tol_deg: float = 5.0,
        snap_tol_m: float = 0.30,
    ):
        self.data = enriched_json
        # Orientation: the pier, grid, beam and column rules work along two orthogonal plan axes.  With
        # principal_frame the plan is rotated into its dominant wall direction before the model is built
        # and the nodes are rotated back afterwards, so a rotated building is treated like an
        # axis-aligned one.  Piers that still deviate from the frame axes by more than axis_tol_deg keep
        # their own direction instead of being projected onto an axis.
        self.principal_frame = principal_frame
        self.axis_tol_deg = axis_tol_deg
        self.snap_tol_m = snap_tol_m       # largest end-point move allowed when a pier is snapped to an axis
        self.frame_angle_deg = 0.0
        self._frame = None
        self.n_stories = n_stories
        self.floor_h = floor_height_m or enriched_json.get("floor_height_m", 2.8)
        self.base_z = base_z
        self.fc = concrete_fc_mpa
        self.sw_t_default = sw_thickness_default
        self.min_sw = min_sw_ratio
        # Rules-mode (default): a selected shear wall is modelled as ONE
        # continuous pier.  The old behaviour split each wall by the GNN
        # left/right ratios, shattering walls into ~0.2 m stubs.
        self.split_sw_by_ratio = split_sw_by_ratio

        # Gravity / floor system options.
        #   wall_mode: which walls are extruded as concrete shell panels --
        #     "structural" : only the predicted shear-wall piers (default).
        #                    Non-structural partitions are dropped so they
        #                    don't over-stiffen the model or steal lateral
        #                    load from the real shear walls.
        #     "all"        : every architectural wall (legacy behaviour).
        #   beam_mode: which wall tops receive a tie/spandrel beam --
        #     "perimeter"    : slab boundary ring only
        #     "perimeter_sw" : ring + a beam over every shear-wall pier (default)
        #     "all_walls"    : ring + a beam over every wall top
        self.wall_mode = wall_mode
        self.add_beams = add_beams
        self.add_slabs = add_slabs
        self.add_diaphragm = add_diaphragm
        self.beam_mode = beam_mode
        self.beam_width_m = beam_width_m
        self.beam_depth_m = beam_depth_m
        self.slab_thickness_m = slab_thickness_m
        # Columns: short shear-wall piers (< col_max_len_m) are modelled as
        # columns rather than stubby walls; long piers get boundary columns at
        # their ends; the building perimeter gets corner columns.  This gives a
        # real gravity/lateral frame instead of a wall-only model.
        self.add_columns = add_columns
        self.max_pier_len = max_pier_len_m
        self.col_max_len = col_max_len_m
        self.col_size = col_size_m
        self.boundary_columns = boundary_columns
        # column_mode:
        #   "grid"   : structural-grid placement (axes from wall centerlines,
        #              columns at unsupported grid points, span enforcement,
        #              tributary-area sizing).  If the enriched JSON already
        #              carries a "columns" list (GNN-predicted), those points
        #              are used instead of the rules.
        #   "corner" : legacy behaviour (column at every footprint corner).
        self.column_mode = column_mode
        self.max_span = max_span_m
        # Joint connectivity (etabs/connect.py): snap beam ends onto supports, merge near-support joints,
        # posts under unsupported corner joints and on unsupported beam runs longer than max_beam_run.
        self.connect_joints = connect_joints
        self.max_beam_run = max_span_m if max_beam_run_m is None else max_beam_run_m
        self.connect_stats = {}
        self._in_repair = False
        self._grid = None            # GridResult from structural_grid
        self._grid_beam_lines = []   # interior beam spans [((x1,y1),(x2,y2))]

        # Built model
        self.nodes: dict[str, Node3D] = {}
        self.wall_panels: list[WallPanel] = []
        self.frame_members: list[FrameMember] = []
        self.slabs: list[Slab] = []
        self.wall_sections: dict[str, WallSection] = {}
        self.beam_sections: dict[str, BeamSection] = {}
        self.slab_sections: dict[str, SlabSection] = {}
        self.column_sections: dict[str, ColumnSection] = {}
        self._col_keys: set = set()
        self.diaphragms: list[str] = []
        self.stories: list[dict] = []

    # ------------------------------------------------------------------
    # Build the 3D model
    # ------------------------------------------------------------------

    def build_model(self):
        """Convert 2D enriched floor plan to 3D structural model."""
        logger.info("Building 3D model: %d stories x %.1fm",
                    self.n_stories, self.floor_h)
        if self.principal_frame:
            self._enter_principal_frame()

        edge_preds = self.data.get("edge_predictions", {})

        # Collect all 2D wall rects with their properties
        rects_2d = []
        for w in self.data.get("walls", []):
            if w.get("rejected"):
                continue
            pts = {p["id"]: (p["x"], p["y"]) for p in w.get("points", [])}
            for r in w.get("rects", []):
                p1 = pts.get(r["p1_id"])
                p2 = pts.get(r["p2_id"])
                if p1 is None or p2 is None:
                    continue

                # Determine if this rect is a shear wall
                is_sw = r.get("is_shear_wall", False)
                sw_ratio_l = 0.0
                sw_ratio_r = 0.0

                # Check edge predictions (from GNN-EP)
                ep = edge_preds.get(r["id"])
                if ep:
                    sw_ratio_l = ep.get("sw_ratio_left", 0.0)
                    sw_ratio_r = ep.get("sw_ratio_right", 0.0)
                    is_sw = ep.get("is_shear_wall", is_sw)

                # Use engineering thickness if available, else architectural
                thickness = r.get("eng_thickness_m", 0.0)
                if thickness < 0.05:
                    thickness = r.get("thickness_m", self.sw_t_default)

                rects_2d.append({
                    "rect_id": r["id"],
                    "wall_id": w["wall_id"],
                    "p1": p1, "p2": p2,
                    "thickness": thickness,
                    "length": r.get("length_m", 0),
                    "is_sw": is_sw,
                    "sw_ratio_l": sw_ratio_l,
                    "sw_ratio_r": sw_ratio_r,
                    "pier_id": r.get("pier_id"),
                    "pier_len": r.get("pier_len_m"),
                })

        logger.info("  %d wall rects to extrude", len(rects_2d))

        # Building footprint outline, computed from ALL wall rects up front so
        # the slab + perimeter beams cover the whole plan even when the
        # non-structural walls are dropped (wall_mode="structural").
        footprint_xy = []
        for rect in rects_2d:
            footprint_xy.append(rect["p1"])
            footprint_xy.append(rect["p2"])
        self._footprint_corners = self._hull_xy(footprint_xy)

        if self.wall_mode == "structural":
            n_struct = sum(1 for r in rects_2d if r["is_sw"])
            logger.info("  wall_mode=structural: keeping %d shear-wall rects, "
                        "dropping %d non-structural",
                        n_struct, len(rects_2d) - n_struct)

        # In structural mode, consolidate the selected shear-wall rects into
        # continuous piers so each pier is ONE clean panel (not a string of
        # short stubs), and so short piers can be demoted to columns.
        piers = None
        if self.wall_mode == "structural" and self.add_columns:
            piers = self._build_piers(rects_2d)
            n_col_piers = sum(1 for pr in piers if pr["length"] < self.col_max_len)
            logger.info("  piers: %d total, %d wall piers, %d short->columns",
                        len(piers), len(piers) - n_col_piers, n_col_piers)

        # Structural grid: axes, column points (rule- or GNN-provided) and
        # interior beam spans.  Computed once in plan, reused per story.
        if self.add_columns and self.column_mode == "grid":
            self._compute_grid(rects_2d)

        # Extrude for each story
        for story in range(self.n_stories):
            z_bot = self.base_z + story * self.floor_h
            z_top = z_bot + self.floor_h
            story_label = f"Story{story + 1}"

            self.stories.append({
                "label": story_label,
                "z_bottom": z_bot,
                "z_top": z_top,
                "height": self.floor_h,
            })

            if piers is not None:
                self._build_story_piers(story, z_bot, z_top, piers)
            else:
                for rect in rects_2d:
                    if self.wall_mode == "structural" and not rect["is_sw"]:
                        continue
                    self._extrude_rect(rect, story, z_bot, z_top)

            # Columns for the lateral/gravity frame.
            if self.add_columns:
                if self._grid is not None:
                    self._build_grid_columns(story, z_bot, z_top)
                else:
                    self._build_corner_columns(story, z_bot, z_top)

            # Floor system at the top of this story: beams, slab, diaphragm.
            # Boundary nodes come from the precomputed footprint outline (not
            # from generated panels), so the slab covers the whole plan and the
            # perimeter beam spans between piers even in structural mode.
            if self.add_beams or self.add_slabs:
                hull_ids = [self._get_or_create_node(x, y, z_top).id
                            for (x, y) in self._footprint_corners]
                if self.add_beams:
                    self._build_story_beams(story, hull_ids)
                if self.add_slabs:
                    self._build_story_slab(story, hull_ids)

        # Generate wall sections
        self._build_sections()

        # Guaranteed load path: any beam end not landing on a column, a
        # wall panel edge, or another beam gets a gravity post underneath
        # (mirrors the step-5 sanity check exactly, so it cannot miss).
        if self.add_columns:
            if self.connect_joints:
                self._connect_joints()
            else:
                self._ensure_beam_supports()

        if self._frame is not None:
            self._leave_principal_frame()

        n_sw = sum(1 for wp in self.wall_panels if wp.is_shear_wall)
        n_nw = len(self.wall_panels) - n_sw
        logger.info("  3D model: %d nodes, %d wall panels (%d SW, %d non-SW), "
                    "%d beams, %d slabs",
                    len(self.nodes), len(self.wall_panels), n_sw, n_nw,
                    len(self.frame_members), len(self.slabs))

    def _connect_joints(self, tol: float = 0.20):
        """Make the exported topology connected through shared joints (see etabs/connect.py).

        An FE package joins elements only where they share a node.  Beam ends that merely lie within `tol`
        of a support, slab-outline corners a few millimetres off a wall corner, and long beam-on-beam runs
        otherwise leave sub-assemblies without a load path to the base.  Order: snap dangling beam ends ->
        post under any end with nothing within tol -> snap again -> merge near-support joints and post
        unsupported corner joints -> limit unsupported beam runs to max_beam_run -> snap again -> drop
        orphan joints.  Counts are kept in self.connect_stats and written next to the JSON export.
        """
        try:
            from . import connect as _cn
        except ImportError:
            import sys as _sys
            _here = os.path.dirname(os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            import connect as _cn
        s1 = _cn.connect_model(self, tol=tol)
        self._ensure_beam_supports(tol=tol)
        s2 = _cn.connect_model(self, tol=tol)
        s3 = _cn.support_joints(self, tol=tol)
        s5 = _cn.limit_spans(self, max_span=self.max_beam_run) if self.max_beam_run and self.max_beam_run > 0 else {}
        s4 = _cn.connect_model(self, tol=tol)
        n_orph = _cn.drop_orphan_nodes(self)
        self.connect_stats = {"pass1": s1, "pass2": s2, "joints": s3, "spans": s5, "pass3": s4,
                              "orphan_nodes_removed": n_orph}

    def _ensure_beam_supports(self, tol: float = 0.20):
        """Add a post under every otherwise-unsupported beam end.

        The tolerance must match check_beam_support in etabs/sanity.py, which
        audits this repair. It previously stood at 0.25 m against the audit's
        0.20 m, so every end falling in that 5 cm band was left unrepaired and
        then reported as unsupported: over the 4,572-plan MSD corpus all 2,944
        reported-unsupported ends lay in it, none beyond, the largest at exactly
        0.250 m. Keep the two values equal.
        """
        nd = {n.id: np.array([n.x, n.y, n.z]) for n in self.nodes.values()}
        n_posts = 0
        for story in range(self.n_stories):
            z_bot = self.base_z + story * self.floor_h
            z_top = z_bot + self.floor_h

            cols_xy = [nd[fm.start_node][:2]
                       for fm in self.frame_members
                       if fm.member_type == "column" and fm.story == story
                       and fm.start_node in nd]
            wall_edges = []
            for wp in self.wall_panels:
                if wp.story != story or len(wp.nodes) < 2:
                    continue
                a, b = nd.get(wp.nodes[0]), nd.get(wp.nodes[1])
                if a is not None and b is not None:
                    wall_edges.append((a[:2], b[:2]))
            beams = [fm for fm in self.frame_members
                     if fm.member_type == "beam" and fm.story == story]
            beam_segs = []
            for bm in beams:
                a, b = nd.get(bm.start_node), nd.get(bm.end_node)
                if a is not None and b is not None:
                    beam_segs.append((a[:2], b[:2]))

            def _seg_d(pt, a, b):
                ab = b - a
                dn = float(np.dot(ab, ab))
                if dn < 1e-12:
                    return float(np.linalg.norm(pt - a))
                t = np.clip(np.dot(pt - a, ab) / dn, 0.0, 1.0)
                return float(np.linalg.norm(pt - (a + t * ab)))

            for bi, (a, b) in enumerate(beam_segs):
                for pt in (a, b):
                    if cols_xy and min(np.linalg.norm(pt - c)
                                       for c in cols_xy) <= tol:
                        continue
                    if wall_edges and min(_seg_d(pt, w1, w2)
                                          for w1, w2 in wall_edges) <= tol:
                        continue
                    if any(_seg_d(pt, u, v) <= tol
                           for bj, (u, v) in enumerate(beam_segs)
                           if bj != bi):
                        continue
                    self._add_column(float(pt[0]), float(pt[1]),
                                     z_bot, z_top, story)
                    cols_xy.append(np.array(pt))
                    n_posts += 1
        if n_posts:
            logger.info("  added %d gravity post(s) under unsupported "
                        "beam ends", n_posts)

    def _extrude_rect(self, rect, story, z_bot, z_top):
        """
        Extrude one 2D rect into 3D wall panel(s).

        If the rect has partial shear wall coverage (sw_ratio_l, sw_ratio_r),
        it gets split into up to 3 panels:
          - Left SW portion  (if sw_ratio_l > min_sw)
          - Middle non-SW gap
          - Right SW portion (if sw_ratio_r > min_sw)
        """
        x1, y1 = rect["p1"]
        x2, y2 = rect["p2"]
        t = rect["thickness"]
        total_len = rect["length"]

        if total_len < 0.01:
            return

        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        L = math.hypot(dx, dy)
        if L < 0.01:
            return

        rl = rect["sw_ratio_l"]
        rr = rect["sw_ratio_r"]

        if self.split_sw_by_ratio and rect["is_sw"] and (rl + rr) > self.min_sw:
            # Legacy: split into SW / non-SW portions by GNN ratio.
            segments = self._split_by_ratios(x1, y1, x2, y2, rl, rr)
            for seg_x1, seg_y1, seg_x2, seg_y2, is_sw_seg in segments:
                self._add_wall_panel(
                    seg_x1, seg_y1, seg_x2, seg_y2,
                    z_bot, z_top, t, is_sw_seg, story,
                    rect["wall_id"], rect["rect_id"])
        else:
            # Rules mode: whole rect is one continuous panel (SW or non-SW).
            self._add_wall_panel(
                x1, y1, x2, y2, z_bot, z_top, t,
                rect["is_sw"], story,
                rect["wall_id"], rect["rect_id"])

    def _split_by_ratios(self, x1, y1, x2, y2, rl, rr):
        """
        Split an edge into up to 3 segments based on shear wall ratios.

        Returns list of (x1, y1, x2, y2, is_shear_wall).
        """
        segments = []
        dx, dy = x2 - x1, y2 - y1

        # Clamp ratios
        rl = max(0.0, min(rl, 1.0))
        rr = max(0.0, min(rr, 1.0))
        if rl + rr > 1.0:
            rl = rr = 0.5

        # Left SW portion
        if rl > self.min_sw:
            mx = x1 + dx * rl
            my = y1 + dy * rl
            segments.append((x1, y1, mx, my, True))

        # Middle non-SW gap
        gap_start = rl if rl > self.min_sw else 0.0
        gap_end = 1.0 - rr if rr > self.min_sw else 1.0
        if gap_end - gap_start > self.min_sw:
            gx1 = x1 + dx * gap_start
            gy1 = y1 + dy * gap_start
            gx2 = x1 + dx * gap_end
            gy2 = y1 + dy * gap_end
            segments.append((gx1, gy1, gx2, gy2, False))

        # Right SW portion
        if rr > self.min_sw:
            mx = x2 - dx * rr
            my = y2 - dy * rr
            segments.append((mx, my, x2, y2, True))

        return segments if segments else [(x1, y1, x2, y2, False)]

    def _add_wall_panel(self, x1, y1, x2, y2, z_bot, z_top,
                        thickness, is_sw, story, wall_id, rect_id):
        """Create a wall panel from 2 bottom points + extrusion height."""
        # Create 4 corner nodes: BL, BR, TR, TL
        n_bl = self._get_or_create_node(x1, y1, z_bot)
        n_br = self._get_or_create_node(x2, y2, z_bot)
        n_tr = self._get_or_create_node(x2, y2, z_top)
        n_tl = self._get_or_create_node(x1, y1, z_top)

        # Section name based on thickness, snapped to standard 25 mm increments
        # so sections are clean/physical and de-duplicate (e.g. no "SW206").
        t_mm = max(25, int(round(thickness * 1000 / 25.0)) * 25)
        thickness = t_mm / 1000.0
        sec_name = f"SW{t_mm}" if is_sw else f"W{t_mm}"

        panel = WallPanel(
            id=f"WP_{uuid.uuid4().hex[:6]}",
            nodes=[n_bl.id, n_br.id, n_tr.id, n_tl.id],
            section=sec_name,
            thickness_m=thickness,
            is_shear_wall=is_sw,
            story=story,
            wall_id=wall_id,
            rect_id=rect_id,
        )
        self.wall_panels.append(panel)

        # Register the section
        if sec_name not in self.wall_sections:
            self.wall_sections[sec_name] = WallSection(
                name=sec_name,
                material="CONC",
                thickness_m=thickness,
                is_shear_wall=is_sw,
            )

    def _get_or_create_node(self, x, y, z, tol=0.005):
        """Find or create a 3D node, merging if within tolerance."""
        for nid, n in self.nodes.items():
            if (abs(n.x - x) < tol and abs(n.y - y) < tol
                    and abs(n.z - z) < tol):
                return n
        nid = f"N_{len(self.nodes)+1}"
        n = Node3D(id=nid, x=round(x, 4), y=round(y, 4), z=round(z, 4))
        self.nodes[nid] = n
        return n

    def _build_sections(self):
        """Ensure we have sections for all referenced thicknesses."""
        # Already built during _add_wall_panel
        pass

    # ------------------------------------------------------------------
    # Floor system: beams, slab, rigid diaphragm
    # ------------------------------------------------------------------

    @staticmethod
    def _hull_xy(pts_xy):
        """
        Andrew's monotone-chain convex hull over (x, y) points.

        Returns the boundary corners in CCW order.  Computed from the full
        footprint cloud and used as the slab outline / perimeter-beam ring.
        It is an outer boundary, so strongly concave (L/U-shaped) plans get an
        over-covering slab -- acceptable for the near-rectangular footprints
        this pipeline targets.
        """
        pts = sorted(set((float(x), float(y)) for x, y in pts_xy))
        if len(pts) < 3:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        # Drop the last point of each chain (it is the first of the other).
        return lower[:-1] + upper[:-1]

    def _beam_section_name(self):
        """Register (once) and return the default beam section name."""
        w_mm = int(round(self.beam_width_m * 1000))
        d_mm = int(round(self.beam_depth_m * 1000))
        name = f"B{d_mm}X{w_mm}"
        if name not in self.beam_sections:
            self.beam_sections[name] = BeamSection(
                name=name, material="CONC",
                width_m=self.beam_width_m, depth_m=self.beam_depth_m)
        return name

    def _slab_section_name(self):
        """Register (once) and return the slab section name (t snapped to 25mm)."""
        t_mm = max(25, int(round(self.slab_thickness_m * 1000 / 25.0)) * 25)
        name = f"SLAB{t_mm}"
        if name not in self.slab_sections:
            self.slab_sections[name] = SlabSection(
                name=name, material="CONC", thickness_m=t_mm / 1000.0)
        return name

    def _build_story_beams(self, story, hull_ids):
        """
        Floor-level beams at the top of `story`:
          - a perimeter ring beam along the slab boundary (always), plus
          - tie/spandrel beams over wall tops per `beam_mode`.
        Edges are de-duplicated so a wall top that lies on the ring is not
        doubled.
        """
        sec = self._beam_section_name()
        edges = {}  # frozenset(a, b) -> (a, b)

        # 1. Perimeter ring along the slab boundary.
        n = len(hull_ids)
        for i in range(n):
            a, b = hull_ids[i], hull_ids[(i + 1) % n]
            if a != b:
                edges[frozenset((a, b))] = (a, b)

        # 2. Interior tie/spandrel beams over wall tops.
        if self.beam_mode != "perimeter":
            for wp in self.wall_panels:
                if wp.story != story or len(wp.nodes) < 4:
                    continue
                if self.beam_mode == "perimeter_sw" and not wp.is_shear_wall:
                    continue
                a, b = wp.nodes[3], wp.nodes[2]  # TL, TR -> wall top edge
                if a != b:
                    edges.setdefault(frozenset((a, b)), (a, b))

        # 3. Interior grid beams: load paths along structural axes between
        #    vertical supports (columns / walls).  Without these, interior
        #    slab regions have no beam to span to.
        if self._grid_beam_lines and self.beam_mode != "perimeter":
            z_top = self.stories[story]["z_top"]
            for (p1, p2) in self._grid_beam_lines:
                a = self._get_or_create_node(p1[0], p1[1], z_top).id
                b = self._get_or_create_node(p2[0], p2[1], z_top).id
                if a != b:
                    edges.setdefault(frozenset((a, b)), (a, b))

        for a, b in edges.values():
            self.frame_members.append(FrameMember(
                id=f"B_{uuid.uuid4().hex[:6]}",
                start_node=a, end_node=b,
                section=sec, member_type="beam", story=story))

    def _build_story_slab(self, story, hull_ids):
        """One floor slab over the story footprint, carrying a rigid diaphragm."""
        if len(hull_ids) < 3:
            logger.warning("  story %d: <3 boundary nodes, slab skipped",
                           story + 1)
            return
        sec = self._slab_section_name()
        diaph = ""
        if self.add_diaphragm:
            diaph = f"D{story + 1}"
            if diaph not in self.diaphragms:
                self.diaphragms.append(diaph)
        self.slabs.append(Slab(
            id=f"SL_{uuid.uuid4().hex[:6]}",
            nodes=list(hull_ids),
            section=sec,
            thickness_m=self.slab_sections[sec].thickness_m,
            story=story,
            diaphragm=diaph))

    # ------------------------------------------------------------------
    # Columns: pier consolidation, short-pier demotion, boundary + corner
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Principal frame
    # ------------------------------------------------------------------
    @staticmethod
    def dominant_angle(segments):
        """Length-weighted dominant direction (degrees, in (-45, 45]) of a set of 2-D segments,
        from the circular mean of 4*phi, so walls at phi and phi + 90 deg reinforce each other."""
        sx = sy = 0.0
        for (x1, y1), (x2, y2) in segments:
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 1e-9:
                continue
            phi = math.atan2(y2 - y1, x2 - x1)
            sx += L * math.cos(4 * phi); sy += L * math.sin(4 * phi)
        if sx == 0.0 and sy == 0.0:
            return 0.0
        return math.degrees(math.atan2(sy, sx) / 4.0)

    def _enter_principal_frame(self):
        """Rotate the enriched plan (wall points, predicted columns) by -theta about its centroid when
        its dominant wall direction deviates from the x axis by more than axis_tol_deg."""
        import copy
        segs, allp = [], []
        for w in self.data.get("walls", []):
            if w.get("rejected"):
                continue
            pts = {p["id"]: (p["x"], p["y"]) for p in w.get("points", [])}
            for r in w.get("rects", []):
                a, b = pts.get(r["p1_id"]), pts.get(r["p2_id"])
                if a is not None and b is not None:
                    segs.append((a, b)); allp += [a, b]
        theta = self.dominant_angle(segs)
        if abs(theta) <= self.axis_tol_deg or not allp:
            return
        cx = sum(p[0] for p in allp) / len(allp); cy = sum(p[1] for p in allp) / len(allp)
        c, s = math.cos(math.radians(-theta)), math.sin(math.radians(-theta))
        rot = lambda x, y: (cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy))
        data = copy.deepcopy(self.data)
        for w in data.get("walls", []):
            for p in w.get("points", []):
                p["x"], p["y"] = rot(p["x"], p["y"])
        for col in data.get("columns", []) or []:
            if "x" in col and "y" in col:
                col["x"], col["y"] = rot(col["x"], col["y"])
        self.data = data
        self._frame = (theta, cx, cy)
        self.frame_angle_deg = round(theta, 3)
        logger.info("  principal frame: plan rotated by %.2f deg for the build", -theta)

    def _leave_principal_frame(self):
        """Rotate every node back by +theta (the inverse of _enter_principal_frame)."""
        theta, cx, cy = self._frame
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        for n in self.nodes.values():
            x, y = n.x - cx, n.y - cy
            n.x = round(cx + c * x - s * y, 4); n.y = round(cy + s * x + c * y, 4)
        self._frame = None

    def _split_bent_groups(self, groups, tol_m: float = 0.50):
        """A Stage-4 pier group is meant to be one collinear run.  Groups whose member end points lie
        more than tol_m off their common line (parallel offset walls, bends) are split into straight
        clusters, each emitted as its own pier, instead of being collapsed onto one averaged line."""
        out = {}
        for key, members in groups.items():
            P = np.array([p for r in members for p in (r["p1"], r["p2"])], float)
            c = P.mean(axis=0)
            vt = np.linalg.svd(P - c)[2]
            if len(members) < 2 or np.abs((P - c) @ vt[1]).max() <= tol_m:
                out[key] = members
                continue
            clusters = []          # [origin, unit direction, members]
            for r in sorted(members, key=lambda r: -math.dist(r["p1"], r["p2"])):
                a, b = np.array(r["p1"], float), np.array(r["p2"], float)
                d = b - a; L = float(np.hypot(*d)); d = d / L if L > 1e-9 else np.array([1.0, 0.0])
                for cl in clusters:
                    o, u, mem = cl
                    nrm = np.array([-u[1], u[0]])
                    if (abs(float(np.dot(d, u))) >= math.cos(math.radians(self.axis_tol_deg))
                            and abs(float(np.dot(a - o, nrm))) <= tol_m and abs(float(np.dot(b - o, nrm))) <= tol_m):
                        mem.append(r); break
                else:
                    clusters.append([a, d, [r]])
            for k, (_o, _u, mem) in enumerate(clusters):
                # the Stage-4 pier length refers to the whole group, not to this cluster
                out[key if k == 0 else f"{key}.{k}"] = [dict(r, pier_len=None) for r in mem]
        return out

    def _build_piers(self, rects_2d):
        """Group selected shear-wall rects into continuous piers.

        Rects are grouped by the pier id assigned in Stage 4 (collinear runs).
        Each pier returns its straight end-to-end endpoints, length and
        thickness, so it can be emitted as a single clean panel.
        """
        groups: dict = {}
        for r in rects_2d:
            if not r["is_sw"]:
                continue
            key = r["pier_id"] if r["pier_id"] is not None else f"rect:{r['rect_id']}"
            groups.setdefault(key, []).append(r)

        if self.principal_frame:
            groups = self._split_bent_groups(groups, tol_m=self.snap_tol_m)

        piers = []
        for key, members in groups.items():
            xs, ys = [], []
            for r in members:
                xs += [r["p1"][0], r["p2"][0]]
                ys += [r["p1"][1], r["p2"][1]]
            dx, dy = max(xs) - min(xs), max(ys) - min(ys)
            thickness = max(r["thickness"] for r in members)
            P = np.column_stack([xs, ys]).astype(float)
            c = P.mean(axis=0)
            d = np.linalg.svd(P - c)[2][0]
            ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
            perp = np.array(ys if dx >= dy else xs, float)
            snap_move = float(np.abs(perp - perp.mean()).max())
            if self.principal_frame and (min(ang, 90.0 - ang) > self.axis_tol_deg or snap_move > self.snap_tol_m):
                # off-axis pier (or one that snapping would move by more than snap_tol_m): straight
                # end-to-end panel along its own principal line
                t = (P - c) @ d
                a, b = c + d * t.min(), c + d * t.max()
                ends = (float(a[0]), float(a[1]), float(b[0]), float(b[1]))
                length = float(t.max() - t.min())
            elif dx >= dy:                     # horizontal pier
                ymean = sum(ys) / len(ys)
                ends = (min(xs), ymean, max(xs), ymean)
                length = dx
            else:                              # vertical pier
                xmean = sum(xs) / len(xs)
                ends = (xmean, min(ys), xmean, max(ys))
                length = dy
            plen = members[0].get("pier_len") or 0.0
            piers.append(dict(
                key=key, ends=ends, length=max(length, plen),
                thickness=thickness,
                wall_id=members[0]["wall_id"], rect_id=members[0]["rect_id"],
                center=((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)))
        return piers

    def _build_story_piers(self, story, z_bot, z_top, piers):
        """Emit wall panels per pier and the column frame.

        - A pier shorter than col_max_len is modelled as a single column.
        - A pier longer than max_pier_len is split into N equal sub-piers
          (each <= max_pier_len) with a column at every joint, since real walls
          that long are interrupted by openings/columns rather than monolithic.
        - Otherwise the pier is one wall panel with boundary columns at its ends.
        """
        import math
        for pr in piers:
            x1, y1, x2, y2 = pr["ends"]
            L = pr["length"]
            if L < self.col_max_len:
                cx, cy = pr["center"]
                self._add_column(cx, cy, z_bot, z_top, story)
                continue

            # split long piers (20% tolerance so a 3.1 m wall stays whole)
            n = 1
            if L > self.max_pier_len * 1.2:
                n = int(math.ceil(L / self.max_pier_len))
            for k in range(n):
                ax1 = x1 + (x2 - x1) * k / n
                ay1 = y1 + (y2 - y1) * k / n
                ax2 = x1 + (x2 - x1) * (k + 1) / n
                ay2 = y1 + (y2 - y1) * (k + 1) / n
                self._add_wall_panel(ax1, ay1, ax2, ay2, z_bot, z_top,
                                     pr["thickness"], True, story,
                                     pr["wall_id"], pr["rect_id"])
            if self.boundary_columns:
                for k in range(n + 1):
                    jx = x1 + (x2 - x1) * k / n
                    jy = y1 + (y2 - y1) * k / n
                    self._add_column(jx, jy, z_bot, z_top, story)

    def _build_corner_columns(self, story, z_bot, z_top):
        """Legacy: place a column at every footprint corner of this story."""
        for (x, y) in getattr(self, "_footprint_corners", []):
            self._add_column(x, y, z_bot, z_top, story)

    # -- structural grid (new default) ---------------------------------

    @staticmethod
    def _import_structural_grid():
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        _s3 = os.path.join(_here, "..", "stages", "stage_3")
        if _s3 not in _sys.path:
            _sys.path.insert(0, _s3)
        import structural_grid as sg
        return sg

    def _compute_grid(self, rects_2d):
        """
        Detect the structural grid and decide column points.

        Priority for column points:
          1. "columns" list in the enriched JSON (GNN node predictions,
             written by Stage 4) -- the model decides, rules only validate.
          2. structural_grid rule engine (grid + span; pier ends are handled
             by _build_story_piers so they are not duplicated here).
        Interior beam spans along grid axes are computed in both cases.
        """
        try:
            sg = self._import_structural_grid()
        except ImportError:
            logger.warning("structural_grid module not found -- "
                           "falling back to corner columns")
            self._grid = None
            return

        segs = [sg.Segment(np.array(r["p1"], float), np.array(r["p2"], float),
                           r["thickness"], bool(r["is_sw"]), r["rect_id"])
                for r in rects_2d]

        gnn_cols = self.data.get("columns") or []
        if gnn_cols:
            # GNN-predicted columns: keep axes/beams from the rules, override
            # the column points themselves.
            res = sg.place_columns(segs, max_span=self.max_span,
                                   pier_end_columns=False)
            res.columns = [sg.ColumnPoint(float(c["x"]), float(c["y"]),
                                          c.get("source", "gnn"),
                                          float(c.get("size_m", self.col_size)))
                           for c in gnn_cols]
            logger.info("  grid columns: %d GNN-predicted points", len(res.columns))
        else:
            res = sg.place_columns(segs, max_span=self.max_span,
                                   pier_end_columns=False)
        sg.size_all_columns(res, self.n_stories, fc_mpa=self.fc)
        self._grid = res
        self._grid_beam_lines = sg.beam_lines(res, segs)
        logger.info("  grid: %d x-axes, %d y-axes, %d columns, "
                    "%d interior beam spans",
                    len(res.axes_x), len(res.axes_y), len(res.columns),
                    len(self._grid_beam_lines))

    def _build_grid_columns(self, story, z_bot, z_top):
        for c in self._grid.columns:
            self._add_column(c.x, c.y, z_bot, z_top, story, size=c.size_m)

    def _column_section_name(self, size=None):
        b = size or self.col_size
        b_mm = int(round(b * 1000))
        name = f"C{b_mm}X{b_mm}"
        if name not in self.column_sections:
            self.column_sections[name] = ColumnSection(
                name=name, material="CONC", width_m=b, depth_m=b)
        return name

    def _add_column(self, x, y, z_bot, z_top, story, size=None):
        """Create one vertical column member (de-duplicated per story/location)."""
        key = (story, round(x, 3), round(y, 3))
        if key in self._col_keys:
            return
        self._col_keys.add(key)
        sec = self._column_section_name(size)
        n_b = self._get_or_create_node(x, y, z_bot)
        n_t = self._get_or_create_node(x, y, z_top)
        self.frame_members.append(FrameMember(
            id=f"C_{uuid.uuid4().hex[:6]}",
            start_node=n_b.id, end_node=n_t.id,
            section=sec, member_type="column", story=story))

    # ------------------------------------------------------------------
    # Export to ETABS via COM (uses the ETABSHandler)
    # ------------------------------------------------------------------

    def push_to_etabs(
        self,
        handler=None,
        *,
        attach: bool = False,
        program_path: Optional[str] = None,
        visible: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Push the built 3D model into ETABS.

        Parameters
        ----------
        handler : ETABSHandler, optional
            Pre-connected handler. If None, creates a new connection.
        save_path : str, optional
            Path to save the .edb file.
        """
        try:
            import comtypes.client
        except ImportError:
            logger.error("comtypes not available — ETABS push requires Windows + comtypes")
            return False

        close_handler = False
        if handler is None:
            try:
                from .etabs_handler import ETABSHandler
            except ImportError:
                from etabs.etabs_handler import ETABSHandler
            handler = ETABSHandler(
                attach=attach,
                specify_path=program_path is not None,
                program_path=program_path,
                visible=visible,
            )
            handler.connect_to_etabs()
            close_handler = True

        sap = handler._sap_model

        if sap is None:
            logger.error("ETABS push failed: _sap_model is None — connection not established")
            return False

        try:
            # 0. Initialize a blank model with kN/m units (unit code 6)
            ret = sap.InitializeNewModel(6)
            if ret != 0:
                logger.warning("ETABS: InitializeNewModel returned %s", ret)
            ret = sap.File.NewBlank()
            if ret != 0:
                logger.warning("ETABS: File.NewBlank returned %s", ret)
            logger.info("ETABS: new blank model initialized (kN/m)")

            # 1. Define material (use SetMaterial_1 so the concrete is design-aware)
            MATERIAL_CONCRETE = 2
            try:
                ret = sap.PropMaterial.SetMaterial_1(
                    "CONC", MATERIAL_CONCRETE, "ACI 318-19", "Normalweight", "4000Psi")
                if ret != 0:
                    raise RuntimeError(f"SetMaterial_1 returned {ret}")
            except Exception as e:
                logger.warning("ETABS: SetMaterial_1 failed (%s); falling back to SetMaterial", e)
                ret = sap.PropMaterial.SetMaterial("CONC", MATERIAL_CONCRETE)
                if ret != 0:
                    logger.warning("ETABS: SetMaterial returned %s", ret)

            E_mpa = 4700 * math.sqrt(self.fc)  # ACI 318 approximation
            ret = sap.PropMaterial.SetMPIsotropic(
                "CONC", E_mpa * 1000,  # kN/m²
                0.2, 0.0000055)
            if ret != 0:
                logger.warning("ETABS: SetMPIsotropic returned %s", ret)
            logger.info("ETABS: defined material CONC (fc=%.0f MPa, E=%.0f MPa)",
                        self.fc, E_mpa)

            # 2. Define wall sections (correct signature: Name, WallPropType, ShellType, MatProp, Thickness)
            for sec in self.wall_sections.values():
                # eWallPropType: 1=Specified
                # eShellType: 1=ShellThin, 2=ShellThick, 3=Membrane, 4=Plate, 5=Layered
                try:
                    ret = sap.PropArea.SetWall(
                        sec.name,
                        1,                  # eWallPropType.Specified
                        1,                  # eShellType.ShellThin
                        "CONC",
                        sec.thickness_m,
                    )
                    if ret != 0:
                        raise RuntimeError(f"SetWall returned {ret}")
                    logger.info("ETABS: defined section %s (t=%.0fmm)",
                                sec.name, sec.thickness_m * 1000)
                except Exception as e:
                    logger.warning("ETABS: could not define section %s: %s",
                                   sec.name, e)

            # 2b. Define beam frame sections (PropFrame.SetRectangle takes
            #     depth, width in the present units = metres).
            for bsec in self.beam_sections.values():
                try:
                    ret = sap.PropFrame.SetRectangle(
                        bsec.name, "CONC", bsec.depth_m, bsec.width_m)
                    if ret != 0:
                        raise RuntimeError(f"SetRectangle returned {ret}")
                    logger.info("ETABS: defined beam section %s (%.0fx%.0fmm)",
                                bsec.name, bsec.depth_m * 1000, bsec.width_m * 1000)
                except Exception as e:
                    logger.warning("ETABS: could not define beam section %s: %s",
                                   bsec.name, e)

            # 2c. Define floor slab sections (concrete shell).
            for ssec in self.slab_sections.values():
                try:
                    # eSlabType: 0=Slab ; eShellType: 1=ShellThin
                    ret = sap.PropArea.SetSlab(
                        ssec.name, 0, 1, "CONC", ssec.thickness_m)
                    if ret != 0:
                        raise RuntimeError(f"SetSlab returned {ret}")
                    logger.info("ETABS: defined slab section %s (t=%.0fmm)",
                                ssec.name, ssec.thickness_m * 1000)
                except Exception as e:
                    logger.warning("ETABS: could not define slab section %s: %s",
                                   ssec.name, e)

            # 3. Define stories FIRST, before any geometry, so areas auto-assign by Z.
            # ETABS expects stories ordered TOP-DOWN (highest first).
            n_stories = len(self.stories)
            if n_stories > 0:
                stories_top_down = list(reversed(self.stories))
                story_names = [s["label"] for s in stories_top_down]
                story_heights = [s["height"] for s in stories_top_down]
                try:
                    ret = sap.Story.SetStories_2(
                        self.base_z,          # base elevation
                        n_stories,            # number of stories
                        story_names,          # names (top -> bottom)
                        story_heights,        # heights
                        [False] * n_stories,  # is master story
                        [""] * n_stories,     # similar to story
                        [False] * n_stories,  # splice above
                        [0.0] * n_stories,    # splice height
                        [1] * n_stories,      # color
                    )
                    if ret != 0:
                        logger.warning("ETABS: SetStories_2 returned %s", ret)
                    else:
                        logger.info("ETABS: defined %d stories", n_stories)
                except Exception as e:
                    logger.warning("ETABS: story definition error: %s", e)

            # 3b. Define rigid diaphragms (one per story). SemiRigid=False -> rigid.
            for dname in self.diaphragms:
                try:
                    ret = sap.Diaphragm.SetDiaphragm(dname, False)
                    if ret != 0:
                        raise RuntimeError(f"SetDiaphragm returned {ret}")
                    logger.info("ETABS: defined rigid diaphragm %s", dname)
                except Exception as e:
                    logger.warning("ETABS: could not define diaphragm %s: %s",
                                   dname, e)

            # 4. Add nodes
            node_id_to_etabs = {}
            n_nodes_ok = 0
            for nid, n in self.nodes.items():
                try:
                    ret = sap.PointObj.AddCartesian(
                        float(n.x), float(n.y), float(n.z), "")
                    # comtypes returns (point_name, ret_code) for ByRef + return-value calls
                    if isinstance(ret, (tuple, list)):
                        point_name = ret[0] if ret else nid
                        ret_code = ret[-1] if len(ret) > 1 else 0
                    else:
                        point_name, ret_code = nid, ret
                    node_id_to_etabs[nid] = point_name or nid
                    if ret_code == 0:
                        n_nodes_ok += 1
                    else:
                        logger.warning("ETABS: AddCartesian for %s returned %s",
                                       nid, ret_code)
                except Exception as e:
                    logger.warning("ETABS: AddCartesian for %s raised %s", nid, e)
                    node_id_to_etabs[nid] = nid

            logger.info("ETABS: added %d / %d nodes", n_nodes_ok, len(self.nodes))

            # 4b. Restrain all nodes at the base elevation
            n_restraints = 0
            base_tol = 1e-3
            restraint = [True, True, True, True, True, True]  # Ux,Uy,Uz,Rx,Ry,Rz
            for nid, n in self.nodes.items():
                if abs(n.z - self.base_z) < base_tol:
                    try:
                        ret = sap.PointObj.SetRestraint(
                            node_id_to_etabs.get(nid, nid), restraint)
                        if not (isinstance(ret, (tuple, list)) and ret[-1] != 0):
                            n_restraints += 1
                    except Exception as e:
                        logger.debug("ETABS: SetRestraint for %s: %s", nid, e)
            logger.info("ETABS: applied %d base restraints", n_restraints)

            # 5. Add wall panels as Area objects (pass Python lists, not ctypes).
            # Use minimal AddByCoord args, then explicitly SetProperty and SetPier
            # so the section assignment and shear-wall tagging are unambiguous.
            n_walls = 0
            n_walls_fail = 0
            n_section_set = 0
            n_section_fail = 0
            n_piers = 0
            pier_seq = 0
            for wp in self.wall_panels:
                try:
                    n_pts = len(wp.nodes)
                    xs = [float(self.nodes[nid].x) for nid in wp.nodes]
                    ys = [float(self.nodes[nid].y) for nid in wp.nodes]
                    zs = [float(self.nodes[nid].z) for nid in wp.nodes]

                    ret = sap.AreaObj.AddByCoord(n_pts, xs, ys, zs, "")
                    # comtypes returns ByRef args + return code as a tuple.
                    # Locate the area name (a string) and the int return code.
                    area_name = ""
                    ret_code = 0
                    if isinstance(ret, (tuple, list)):
                        for v in ret:
                            if isinstance(v, str) and v:
                                area_name = v
                            elif isinstance(v, int):
                                ret_code = v
                    else:
                        ret_code = ret

                    if ret_code != 0 or not area_name:
                        n_walls_fail += 1
                        if n_walls_fail <= 3:
                            logger.warning(
                                "ETABS: AddByCoord for wall %s (sec=%s) returned %s name=%r",
                                wp.id, wp.section, ret_code, area_name)
                        continue

                    n_walls += 1

                    # Explicitly assign the wall section property (the PropName arg
                    # in AddByCoord is unreliable across ETABS versions).
                    try:
                        ret2 = sap.AreaObj.SetProperty(area_name, wp.section)
                        ret2_code = ret2[-1] if isinstance(ret2, (tuple, list)) else ret2
                        if ret2_code == 0:
                            n_section_set += 1
                        else:
                            n_section_fail += 1
                            if n_section_fail <= 3:
                                logger.warning(
                                    "ETABS: SetProperty(%s, %s) returned %s",
                                    area_name, wp.section, ret2_code)
                    except Exception as e:
                        n_section_fail += 1
                        if n_section_fail <= 3:
                            logger.warning(
                                "ETABS: SetProperty(%s, %s) raised: %s",
                                area_name, wp.section, e)

                    # Tag shear walls with a pier label so ETABS treats them as
                    # shear walls for design and results.
                    if wp.is_shear_wall:
                        pier_seq += 1
                        pier_label = f"P{pier_seq}"
                        try:
                            # Ensure the pier label exists before assignment
                            sap.PierLabel.SetPier(pier_label)
                        except Exception:
                            pass
                        try:
                            ret3 = sap.AreaObj.SetPier(area_name, pier_label)
                            ret3_code = ret3[-1] if isinstance(ret3, (tuple, list)) else ret3
                            if ret3_code == 0:
                                n_piers += 1
                        except Exception as e:
                            if n_piers < 3:
                                logger.debug(
                                    "ETABS: SetPier(%s, %s) raised: %s",
                                    area_name, pier_label, e)
                except Exception as e:
                    n_walls_fail += 1
                    if n_walls_fail <= 3:
                        logger.warning("ETABS: wall panel %s raised: %s", wp.id, e)

            logger.info("ETABS: added %d / %d wall panels (%d failed)",
                        n_walls, len(self.wall_panels), n_walls_fail)
            logger.info("ETABS: assigned sections to %d panels (%d failed)",
                        n_section_set, n_section_fail)
            logger.info("ETABS: tagged %d panels with pier labels (shear walls)",
                        n_piers)

            # 5b. Add floor slabs as Area objects and assign the rigid diaphragm.
            n_slabs = 0
            n_slabs_fail = 0
            for sl in self.slabs:
                try:
                    n_pts = len(sl.nodes)
                    xs = [float(self.nodes[nid].x) for nid in sl.nodes]
                    ys = [float(self.nodes[nid].y) for nid in sl.nodes]
                    zs = [float(self.nodes[nid].z) for nid in sl.nodes]

                    ret = sap.AreaObj.AddByCoord(n_pts, xs, ys, zs, "")
                    area_name = ""
                    ret_code = 0
                    if isinstance(ret, (tuple, list)):
                        for v in ret:
                            if isinstance(v, str) and v:
                                area_name = v
                            elif isinstance(v, int):
                                ret_code = v
                    else:
                        ret_code = ret

                    if ret_code != 0 or not area_name:
                        n_slabs_fail += 1
                        continue
                    n_slabs += 1

                    try:
                        sap.AreaObj.SetProperty(area_name, sl.section)
                    except Exception as e:
                        logger.warning("ETABS: slab SetProperty(%s, %s) raised: %s",
                                       area_name, sl.section, e)

                    if sl.diaphragm:
                        try:
                            sap.AreaObj.SetDiaphragm(area_name, sl.diaphragm)
                        except Exception as e:
                            logger.debug("ETABS: slab SetDiaphragm(%s, %s): %s",
                                         area_name, sl.diaphragm, e)
                except Exception as e:
                    n_slabs_fail += 1
                    if n_slabs_fail <= 3:
                        logger.warning("ETABS: slab %s raised: %s", sl.id, e)

            if self.slabs:
                logger.info("ETABS: added %d / %d floor slabs (%d failed)",
                            n_slabs, len(self.slabs), n_slabs_fail)

            # 5c. Assign each story's rigid diaphragm to ALL joints at that level,
            #     so the shear walls are tied together and share lateral load
            #     (the fix for the previous walls-only model). DiaphragmOption
            #     3 = use a defined diaphragm.
            if self.add_diaphragm and self.slabs:
                story_diaph = {sl.story: sl.diaphragm
                               for sl in self.slabs if sl.diaphragm}
                n_pt_diaph = 0
                for story_idx, dname in story_diaph.items():
                    z_top = self.stories[story_idx]["z_top"]
                    for nid, n in self.nodes.items():
                        if abs(n.z - z_top) >= 1e-3:
                            continue
                        try:
                            pname = node_id_to_etabs.get(nid, nid)
                            ret = sap.PointObj.SetDiaphragm(pname, 3, dname)
                            rc = ret[-1] if isinstance(ret, (tuple, list)) else ret
                            if rc == 0:
                                n_pt_diaph += 1
                        except Exception as e:
                            logger.debug("ETABS: SetDiaphragm joint %s: %s", nid, e)
                logger.info("ETABS: assigned rigid diaphragm to %d joints",
                            n_pt_diaph)

            # 6. Add frame members
            n_frames = 0
            n_frames_fail = 0
            for fm in self.frame_members:
                try:
                    n1 = self.nodes[fm.start_node]
                    n2 = self.nodes[fm.end_node]
                    ret = sap.FrameObj.AddByCoord(
                        float(n1.x), float(n1.y), float(n1.z),
                        float(n2.x), float(n2.y), float(n2.z),
                        "", fm.section)
                    ret_code = ret[-1] if isinstance(ret, (tuple, list)) else ret
                    if ret_code == 0:
                        n_frames += 1
                    else:
                        n_frames_fail += 1
                except Exception as e:
                    n_frames_fail += 1
                    if n_frames_fail <= 3:
                        logger.warning("ETABS: frame %s raised: %s", fm.id, e)

            if self.frame_members:
                logger.info("ETABS: added %d / %d frame members",
                            n_frames, len(self.frame_members))

            # 7. Refresh the view so the model actually redraws
            try:
                sap.View.RefreshView(0, False)
            except Exception as e:
                logger.debug("ETABS: RefreshView failed: %s", e)

            # 8. Save
            if save_path:
                ret = sap.File.Save(save_path)
                if ret != 0:
                    logger.warning("ETABS: File.Save returned %s", ret)
                else:
                    logger.info("ETABS: saved to %s", save_path)

            return True

        except Exception as e:
            logger.error("ETABS push failed: %s", e)
            return False

        finally:
            if close_handler:
                handler.disconnect_from_etabs()

    # ------------------------------------------------------------------
    # Export to standalone JSON (for OpenSees / Abaqus / ANSYS / other)
    # ------------------------------------------------------------------

    def export_json(self, path: str):
        """Export the 3D model as a FEM-ready JSON file."""
        model = {
            "metadata": {
                "source": "shear_wall_pipeline",
                "floor_id": self.data.get("floor_id", ""),
                "n_stories": self.n_stories,
                "floor_height_m": self.floor_h,
                "concrete_fc_mpa": self.fc,
                "wall_mode": self.wall_mode,
                "frame_angle_deg": self.frame_angle_deg,
            },
            "stories": self.stories,
            "materials": [{
                "name": "CONC",
                "type": "concrete",
                "fc_mpa": self.fc,
                "E_mpa": round(4700 * math.sqrt(self.fc), 1),
                "poisson": 0.2,
                "thermal_coeff": 0.0000055,
            }],
            "wall_sections": [s.to_dict() for s in self.wall_sections.values()],
            "beam_sections": [b.to_dict() for b in self.beam_sections.values()],
            "column_sections": [c.to_dict() for c in self.column_sections.values()],
            "slab_sections": [s.to_dict() for s in self.slab_sections.values()],
            "diaphragms": list(self.diaphragms),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "wall_panels": [wp.to_dict() for wp in self.wall_panels],
            "slabs": [sl.to_dict() for sl in self.slabs],
            "frame_members": [fm.to_dict() for fm in self.frame_members],
            "summary": {
                "n_nodes": len(self.nodes),
                "n_wall_panels": len(self.wall_panels),
                "n_shear_walls": sum(1 for wp in self.wall_panels
                                     if wp.is_shear_wall),
                "n_frame_members": len(self.frame_members),
                "n_beams": sum(1 for fm in self.frame_members
                               if fm.member_type == "beam"),
                "n_columns": sum(1 for fm in self.frame_members
                                 if fm.member_type == "column"),
                "n_slabs": len(self.slabs),
                "n_diaphragms": len(self.diaphragms),
            },
        }

        with open(path, "w") as f:
            json.dump(model, f, indent=2)
        logger.info("Exported FEM model to %s", path)

    # ------------------------------------------------------------------
    # Export to Structure schema (compatible with existing ETABS handler)
    # ------------------------------------------------------------------

    def to_structure_dict(self) -> dict:
        """
        Convert to the Structure schema format used by the existing
        app.schemas.structure / builder.py / etabs_handler.py.

        Returns a dict compatible with Structure.model_dump().
        """
        nodes_list = []
        for n in self.nodes.values():
            nodes_list.append({
                "id": n.id, "x": n.x, "y": n.y, "z": n.z,
            })

        members_list = []
        # Wall panels as "members" with 4 nodes (for the existing handler,
        # we represent each wall panel as 4 frame edges)
        for wp in self.wall_panels:
            if len(wp.nodes) >= 4:
                for i in range(4):
                    j = (i + 1) % 4
                    members_list.append({
                        "id": f"{wp.id}_e{i}",
                        "start_node": wp.nodes[i],
                        "end_node": wp.nodes[j],
                        "section": wp.section,
                        "type": "wall_edge",
                    })

        for fm in self.frame_members:
            members_list.append({
                "id": fm.id,
                "start_node": fm.start_node,
                "end_node": fm.end_node,
                "section": fm.section,
                "type": fm.member_type,
            })

        return {
            "units": "kN_m",
            "nodes": nodes_list,
            "members": members_list,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        n_sw = sum(1 for wp in self.wall_panels if wp.is_shear_wall)
        n_nw = len(self.wall_panels) - n_sw

        print(f"\n{'='*60}")
        print(f"  STAGE 5 - ETABS EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"  Stories:        {self.n_stories} x {self.floor_h:.1f}m "
              f"= {self.n_stories * self.floor_h:.1f}m total")
        n_beams = sum(1 for fm in self.frame_members if fm.member_type == "beam")
        print(f"  Nodes:          {len(self.nodes)}")
        print(f"  Wall mode:      {self.wall_mode}")
        print(f"  Wall panels:    {len(self.wall_panels)} "
              f"({n_sw} shear wall, {n_nw} non-structural)")
        n_cols = sum(1 for fm in self.frame_members if fm.member_type == "column")
        print(f"  Beams:          {n_beams}  (mode='{self.beam_mode}')")
        print(f"  Columns:        {n_cols}  (short-pier + boundary + corner)")
        print(f"  Slabs:          {len(self.slabs)}")
        print(f"  Diaphragms:     {len(self.diaphragms)} "
              f"({'rigid' if self.add_diaphragm else 'none'})")
        print(f"  Frame members:  {len(self.frame_members)}")
        print(f"  Wall sections:  {len(self.wall_sections)}")
        for sec in self.wall_sections.values():
            tag = "SW" if sec.is_shear_wall else "W"
            n = sum(1 for wp in self.wall_panels if wp.section == sec.name)
            print(f"    {sec.name:10s}  t={sec.thickness_m*1000:.0f}mm  "
                  f"[{tag}]  {n} panels")
        for bsec in self.beam_sections.values():
            n = sum(1 for fm in self.frame_members if fm.section == bsec.name)
            print(f"    {bsec.name:10s}  {bsec.depth_m*1000:.0f}x"
                  f"{bsec.width_m*1000:.0f}mm  [beam]  {n} members")
        for ssec in self.slab_sections.values():
            n = sum(1 for sl in self.slabs if sl.section == ssec.name)
            print(f"    {ssec.name:10s}  t={ssec.thickness_m*1000:.0f}mm  "
                  f"[slab]  {n} panels")
        print(f"{'='*60}")


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 5: Export enriched floor plan to ETABS / FEM JSON")
    parser.add_argument("-i", "--input", required=True,
                        help="Enriched floor JSON (Stage 4 output)")
    parser.add_argument("-o", "--output", default="model_fem.json",
                        help="Output FEM JSON path")
    parser.add_argument("--stories", type=int, default=1,
                        help="Number of stories to extrude")
    parser.add_argument("--floor-height", type=float, default=3.0,
                        help="Floor-to-floor height in meters")
    parser.add_argument("--fc", type=float, default=30.0,
                        help="Concrete strength in MPa")
    parser.add_argument("--walls", default="structural",
                        choices=["structural", "all"],
                        help="Which walls to model: 'structural' = predicted "
                             "shear-wall piers only (default); 'all' = every "
                             "architectural wall (legacy)")
    parser.add_argument("--no-beams", action="store_true",
                        help="Skip floor beam generation")
    parser.add_argument("--no-slabs", action="store_true",
                        help="Skip floor slab generation")
    parser.add_argument("--no-diaphragm", action="store_true",
                        help="Skip rigid diaphragm assignment")
    parser.add_argument("--beam-mode", default="perimeter_sw",
                        choices=["perimeter", "perimeter_sw", "all_walls"],
                        help="Which wall tops get a beam (default: perimeter_sw)")
    parser.add_argument("--beam-width", type=float, default=0.30,
                        help="Beam width in meters")
    parser.add_argument("--beam-depth", type=float, default=0.50,
                        help="Beam depth in meters")
    parser.add_argument("--slab-thickness", type=float, default=0.15,
                        help="Floor slab thickness in meters")
    parser.add_argument("--no-columns", action="store_true",
                        help="Skip column generation (wall-only model)")
    parser.add_argument("--max-pier-len", type=float, default=3.0,
                        help="Split shear-wall piers longer than this (m) into "
                             "sub-piers with columns at the joints")
    parser.add_argument("--col-max-len", type=float, default=1.2,
                        help="Shear-wall piers shorter than this (m) become columns")
    parser.add_argument("--col-size", type=float, default=0.40,
                        help="Square column size in meters")
    parser.add_argument("--no-boundary-columns", action="store_true",
                        help="Do not add boundary columns at shear-wall pier ends")
    parser.add_argument("--column-mode", default="grid",
                        choices=["grid", "corner"],
                        help="Column placement: 'grid' = structural grid + span "
                             "rules / GNN points (default); 'corner' = legacy "
                             "footprint corners")
    parser.add_argument("--max-span", type=float, default=6.0,
                        help="Max slab/beam span (m) before intermediate "
                             "columns are inserted")
    parser.add_argument("--no-connect", action="store_true",
                        help="Skip the joint-connectivity step (etabs/connect.py); "
                             "reproduces the exporter used before October 2026")
    parser.add_argument("--max-beam-run", type=float, default=None,
                        help="Longest unsupported beam run (m) before a gravity post is "
                             "inserted (default: --max-span; 0 disables)")
    parser.add_argument("--no-principal-frame", action="store_true",
                        help="Build in the drawing axes and project every pier onto x or y; "
                             "reproduces the exporter used before the orientation fix")
    parser.add_argument("--push-etabs", action="store_true",
                        help="Push to live ETABS instance")
    parser.add_argument("--etabs-save", default=None,
                        help="Path to save ETABS .edb file")
    parser.add_argument("--etabs-path", default=None,
                        help=r"Full path to ETABS.exe (e.g. 'C:\Program Files\Computers and Structures\ETABS 23\ETABS.exe'). "
                             "Use this when multiple ETABS versions are installed.")
    parser.add_argument("--attach-etabs", action="store_true",
                        help="Attach to a running ETABS instance instead of starting a new one")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    with open(args.input) as f:
        enriched = json.load(f)

    exporter = ETABSExporter(
        enriched,
        n_stories=args.stories,
        floor_height_m=args.floor_height,
        concrete_fc_mpa=args.fc,
        add_beams=not args.no_beams,
        add_slabs=not args.no_slabs,
        add_diaphragm=not args.no_diaphragm,
        beam_mode=args.beam_mode,
        beam_width_m=args.beam_width,
        beam_depth_m=args.beam_depth,
        slab_thickness_m=args.slab_thickness,
        wall_mode=args.walls,
        add_columns=not args.no_columns,
        max_pier_len_m=args.max_pier_len,
        col_max_len_m=args.col_max_len,
        col_size_m=args.col_size,
        boundary_columns=not args.no_boundary_columns,
        column_mode=args.column_mode,
        max_span_m=args.max_span,
        connect_joints=not args.no_connect,
        max_beam_run_m=args.max_beam_run,
        principal_frame=not args.no_principal_frame,
    )
    exporter.build_model()
    exporter.print_summary()
    exporter.export_json(args.output)
    if exporter.connect_stats:
        with open(os.path.splitext(args.output)[0] + ".connect.json", "w") as f:
            json.dump(exporter.connect_stats, f, indent=1)

    if args.push_etabs:
        exporter.push_to_etabs(
            save_path=args.etabs_save,
            program_path=args.etabs_path,
            attach=args.attach_etabs,
        )


if __name__ == "__main__":
    main()
