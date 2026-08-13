"""
Stage 3 - Graph Construction
==============================
Converts the vectorized floor plan from Stage 2 (points + rectangles) into a
structural graph suitable for GNN-based shear wall layout prediction (Stage 4).

Graph representation follows Zhao et al. (2023):
  - Nodes  = component intersection points (merged by proximity)
  - Edges  = linear components: walls, doors, windows
  - Best variant: Edge-PSW-DW  (recommended)
    Edge of PSW:                [1, 0, 0, x_l, y_l, x_r, y_r, length]
    Edge of indoor door/window: [0, 1, 0, x_l, y_l, x_r, y_r, length]
    Edge of outdoor door:       [0, 0, 1, x_l, y_l, x_r, y_r, length]

Stage 2 -> 3 data flow:
  Stage 2 JSON (floor_output.json)
    walls[].points[]  {id, x, y}
    walls[].rects[]   {id, p1_id, p2_id, thickness_m, length_m, angle_rad}
  + optional openings (door/window annotations from Stage 1 or manual)
    openings[]  {type, x1, y1, x2, y2, wall_id?}

  --> Stage 3 produces: networkx.Graph
    Nodes: {x, y, x_m, y_m}            (normalized + original coords)
    Edges: {id, edge_type, x_left, y_left, x_right, y_right, length,
            thickness_m, length_m, angle_rad, wall_id, rect_id}

Usage:
  python graph_construction.py                          # demo on Stage 2 output
  python graph_construction.py -i floor.json -o graph.json
  python graph_construction.py -i floor.json --openings openings.json

References:
  [1] Zhao et al., Adv. Eng. Informatics 55 (2023) 101886
"""

from __future__ import annotations
import argparse, copy, json, logging, math, os, uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Constants & data types
# =============================================================================

L_GRAPH_SCALE = 20_480.0   # mm -- Zhao et al. Eq(1): normalise coords to [-1,1]


class EdgeType(IntEnum):
    """Edge classes for the Edge-PSW-DW scheme (Table 1 in Zhao et al.)."""
    PSW = 0             # Potential shear wall
    INDOOR_DW = 1       # Indoor door / window
    OUTDOOR_D = 2       # Outdoor door / gate
    PW = 3              # Partition wall (used only in Edge-PSW-PW-DW)


class GraphVariant(IntEnum):
    """Which graph representation variant to build."""
    EDGE_PSW = 0           # walls only
    EDGE_PSW_DW = 1        # walls + doors/windows  (recommended)
    EDGE_PSW_PW = 2        # walls + partition walls
    EDGE_PSW_PW_DW = 3     # walls + partition walls + doors/windows


# Mapping: variant -> which edge types are included
_VARIANT_TYPES = {
    GraphVariant.EDGE_PSW:        {EdgeType.PSW},
    GraphVariant.EDGE_PSW_DW:     {EdgeType.PSW, EdgeType.INDOOR_DW, EdgeType.OUTDOOR_D},
    GraphVariant.EDGE_PSW_PW:     {EdgeType.PSW, EdgeType.PW},
    GraphVariant.EDGE_PSW_PW_DW:  {EdgeType.PSW, EdgeType.PW, EdgeType.INDOOR_DW, EdgeType.OUTDOOR_D},
}


@dataclass
class Opening:
    """A door or window opening annotation."""
    opening_id: str = ""
    kind: str = "door"            # door | window | outdoor_door
    p1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    length_m: float = 0.0
    wall_id: str = ""             # which wall it belongs to (optional)
    is_exterior: bool = False     # exterior => outdoor_door class


@dataclass
class RawSegment:
    """A linear segment (wall rect or opening) before graph insertion."""
    seg_id: str = ""
    p1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p1_id: str = ""
    p2_id: str = ""
    edge_type: EdgeType = EdgeType.PSW
    thickness_m: float = 0.0
    length_m: float = 0.0
    angle_rad: float = 0.0
    wall_id: str = ""


# =============================================================================
# 2. Stage 2 -> 3 Adapter
#    Parses Stage 2 JSON + optional openings -> list of RawSegments
# =============================================================================

class Stage2To3Adapter:
    """
    Reads Stage 2 vectorised output and optional opening annotations,
    classifies every linear element, and produces a list of RawSegments
    ready for graph construction.

    Wall classification modes:
      "auto" (default):
        Scans all wall thicknesses and sets the PSW/PW boundary
        adaptively.  If the thickness distribution is unimodal (all
        walls are similar), everything is classified as PSW.  If
        bimodal, the thinner cluster becomes PW.
      "fixed":
        Uses the explicit psw_min_thickness / pw thresholds passed
        at construction time.

    Openings:
      - If opening annotations are provided, each opening inserts an
        INDOOR_DW or OUTDOOR_D edge and the parent wall edge is kept.
      - If no annotations, all valid wall rects become PSW edges.
    """

    def __init__(
        self,
        *,
        classify_mode: str = "auto",       # "auto" or "fixed"
        psw_min_thickness: float = 0.06,   # m — only used in "fixed" mode
        pw_min_thickness: float = 0.03,    # m
        pw_max_thickness: float = 0.06,    # m
        max_wall_thickness: float = 0.60,  # m — thicker => rejected
        exterior_detect: bool = True,      # auto-detect exterior walls
        auto_detect_gaps: bool = True,     # infer window/door edges from wall gaps
        min_opening_m: float = 0.40,       # shortest opening to detect (m)
        max_opening_m: float = 2.60,       # longest INTERIOR opening (m) — wide room
                                           # mouths / sliding doors at TRUE scale
        max_opening_ext_m: float = 3.20,   # longest EXTERIOR opening (m) — facade
                                           # windows and balcony sliders; the old
                                           # 1.30 m cap was tuned before scale
                                           # auto-calibration and left window
                                           # gaps as holes in the graph
        door_opening_m: float = 0.90,      # exterior gap in [door, door_max] -> OUTDOOR_D
        door_max_m: float = 1.40,          # wider exterior gaps are WINDOWS (INDOOR_DW)
        collinear_tol_m: float = 0.08,     # lateral tolerance for "same wall axis" (m)
    ):
        self.mode = classify_mode
        self.psw_min_t = psw_min_thickness
        self.pw_min_t = pw_min_thickness
        self.pw_max_t = pw_max_thickness
        self.max_t = max_wall_thickness
        self.exterior_detect = exterior_detect
        self.auto_detect_gaps = auto_detect_gaps
        self.min_opening_m = min_opening_m
        self.max_opening_m = max_opening_m
        self.max_opening_ext_m = max_opening_ext_m
        self.door_opening_m = door_opening_m
        self.door_max_m = door_max_m
        self.collinear_tol_m = collinear_tol_m

    def parse(
        self,
        stage2_floor: dict,
        openings: Optional[list[dict]] = None,
    ) -> list[RawSegment]:
        """
        Returns
        -------
        list[RawSegment]
            All classified linear segments (PSW, PW, DW, outdoor door).
        """
        # --- Collect all valid rects first (for auto-calibration) ---
        raw_rects = []
        for w in stage2_floor.get("walls", []):
            if w.get("rejected"):
                continue
            pts = {p["id"]: np.array([p["x"], p["y"]])
                   for p in w.get("points", [])}
            for r in w.get("rects", []):
                p1 = pts.get(r["p1_id"])
                p2 = pts.get(r["p2_id"])
                if p1 is None or p2 is None:
                    continue
                t = r["thickness_m"]
                if t > self.max_t or t < 0.01:
                    continue
                raw_rects.append((w, r, p1, p2, t))

        # --- Determine PSW/PW boundary ---
        if self.mode == "auto" and raw_rects:
            psw_thr, pw_thr = self._auto_thresholds(
                [t for _, _, _, _, t in raw_rects])
        else:
            psw_thr = self.psw_min_t
            pw_thr = self.pw_max_t

        # --- Classify ---
        segments: list[RawSegment] = []
        for w, r, p1, p2, t in raw_rects:
            if t >= psw_thr:
                etype = EdgeType.PSW
            elif t >= self.pw_min_t:
                etype = EdgeType.PW
            else:
                continue

            segments.append(RawSegment(
                seg_id=r["id"],
                p1=p1, p2=p2,
                p1_id=r["p1_id"], p2_id=r["p2_id"],
                edge_type=etype,
                thickness_m=t,
                length_m=r["length_m"],
                angle_rad=r["angle_rad"],
                wall_id=w["wall_id"],
            ))

        n_psw = sum(1 for s in segments if s.edge_type == EdgeType.PSW)
        n_pw = sum(1 for s in segments if s.edge_type == EdgeType.PW)
        logger.info("Adapter [%s]: %d segments (PSW=%d, PW=%d)  "
                    "psw_thr=%.3fm",
                    self.mode, len(segments), n_psw, n_pw, psw_thr)

        # --- Mark exterior walls ---
        if self.exterior_detect and segments:
            self._mark_exterior(segments)

        # --- Auto-detect opening gaps (wall breaks = windows/doors) ---
        if self.auto_detect_gaps:
            gap_segs = self._detect_opening_gaps(segments)
            if gap_segs:
                segments.extend(gap_segs)

        # --- Process openings (doors/windows) ---
        if openings:
            opening_segs = self._parse_openings(openings, segments)
            segments.extend(opening_segs)
            logger.info("Adapter: +%d opening segments (DW=%d, OutD=%d)",
                        len(opening_segs),
                        sum(1 for s in opening_segs
                            if s.edge_type == EdgeType.INDOOR_DW),
                        sum(1 for s in opening_segs
                            if s.edge_type == EdgeType.OUTDOOR_D))

        return segments

    # -- auto-calibration ------------------------------------------------------

    def _auto_thresholds(self, thicknesses: list[float]):
        """
        Determine PSW vs PW thickness boundary from the data.

        Strategy:
          1. If all walls are within 35% of the max thickness,
             treat everything as PSW (unimodal distribution).
          2. Otherwise, split at the first significant gap in the sorted
             unique thickness values. This correctly separates the thin
             partition-wall cluster (e.g. 4-5cm) from structural walls
             (6cm+) regardless of the upper end of the distribution.

        Returns (psw_threshold, pw_threshold).
        """
        t = np.array(sorted(thicknesses))
        t_min, t_max = t.min(), t.max()

        if t_max < 1e-6:
            return (0.01, 0.01)

        spread = (t_max - t_min) / t_max
        if spread < 0.35:
            psw_thr = max(t_min * 0.8, 0.02)
            logger.info("Auto-classify: unimodal (spread=%.0f%%), "
                        "all walls -> PSW (thr=%.3fm)",
                        spread * 100, psw_thr)
            return (psw_thr, psw_thr)

        # Bimodal: split at the *largest* gap between consecutive unique
        # thickness values (the true cluster boundary).  The old code split at
        # the FIRST gap >= 5 mm, which for near-continuous (1 cm-quantised)
        # thicknesses always landed just above the minimum -> nearly everything
        # became PSW.  Otsu, conversely, pushes the split too high.  Largest-gap
        # is the robust middle ground.
        unique = np.unique(t)
        min_gap_m = 0.005  # 5 mm — minimum gap to count as a cluster boundary
        best_thr = float(np.median(t))  # fallback if no clear gap exists
        if len(unique) >= 2:
            diffs = np.diff(unique)
            gi = int(np.argmax(diffs))
            if diffs[gi] >= min_gap_m:
                best_thr = float(unique[gi] + unique[gi + 1]) / 2.0

        logger.info("Auto-classify: bimodal (spread=%.0f%%), largest-gap "
                    "PSW threshold=%.3fm", spread * 100, best_thr)
        return (best_thr, best_thr)

    # -- exterior wall detection -----------------------------------------------

    def _mark_exterior(self, segments: list[RawSegment]):
        """
        Heuristic: compute the convex hull of all wall midpoints.
        Segments whose midpoint is on or near the hull are exterior.
        """
        from scipy.spatial import ConvexHull
        mids = np.array([(s.p1 + s.p2) / 2 for s in segments])
        if len(mids) < 3:
            return
        try:
            hull = ConvexHull(np.vstack([s.p1 for s in segments] +
                                        [s.p2 for s in segments]))
        except Exception:
            return
        hull_pts = hull.points[hull.vertices]

        for s in segments:
            mid = (s.p1 + s.p2) / 2
            # Distance from midpoint to nearest hull edge
            d = self._dist_to_hull(mid, hull_pts)
            if d < 0.3:  # within 30cm of hull boundary
                s._is_exterior = True
            else:
                s._is_exterior = False

    @staticmethod
    def _dist_to_hull(pt, hull_pts):
        """Minimum distance from pt to the hull polygon edges."""
        n = len(hull_pts)
        best = float("inf")
        for i in range(n):
            a = hull_pts[i]
            b = hull_pts[(i + 1) % n]
            ab = b - a
            ap = pt - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
            proj = a + t * ab
            best = min(best, np.linalg.norm(pt - proj))
        return best

    # -- opening parsing -------------------------------------------------------

    def _parse_openings(
        self,
        openings: list[dict],
        wall_segs: list[RawSegment],
    ) -> list[RawSegment]:
        """
        Convert opening annotations into graph segments.

        Opening dict format:
          {
            "type": "door" | "window" | "outdoor_door",
            "x1": float, "y1": float,       # endpoint 1 in metres
            "x2": float, "y2": float,       # endpoint 2 in metres
            "wall_id": str (optional)        # parent wall
          }

        Each opening becomes an edge of type INDOOR_DW or OUTDOOR_D.
        """
        out = []
        for o in openings:
            p1 = np.array([o["x1"], o["y1"]])
            p2 = np.array([o["x2"], o["y2"]])
            L = float(np.linalg.norm(p2 - p1))
            if L < 0.01:
                continue

            kind = o.get("type", "door").lower()
            if kind in ("outdoor_door", "outdoor", "gate"):
                etype = EdgeType.OUTDOOR_D
            else:
                # door or window => indoor DW
                etype = EdgeType.INDOOR_DW

            # Override: if parent wall is exterior and kind is "door"
            wid = o.get("wall_id", "")
            if wid:
                parent = [s for s in wall_segs if s.wall_id == wid]
                if parent and getattr(parent[0], "_is_exterior", False):
                    if kind == "door":
                        etype = EdgeType.OUTDOOR_D

            angle = float(np.arctan2(abs(p2[0] - p1[0]),
                                     abs(p2[1] - p1[1])))

            out.append(RawSegment(
                seg_id=o.get("id", uuid.uuid4().hex[:8]),
                p1=p1, p2=p2,
                p1_id=uuid.uuid4().hex[:8],
                p2_id=uuid.uuid4().hex[:8],
                edge_type=etype,
                thickness_m=0.0,
                length_m=L,
                angle_rad=angle,
                wall_id=wid,
            ))
        return out

    # -- gap-based opening detection -------------------------------------------

    def _detect_opening_gaps(self, segments: list[RawSegment]) -> list[RawSegment]:
        """
        Scan collinear PSW wall pairs for gaps in [min_opening, max_opening] metres
        and emit a synthetic DW edge for each gap found.

        A gap between two collinear wall segments on the same axis line (within
        collinear_tol_m laterally) represents a window or door opening that was
        lost when Stage 1 produced a binary wall mask.  Re-inserting these edges
        closes the train/inference distribution mismatch for the GNN.

        Classification:
          - Both bounding walls exterior AND gap >= door_opening_m -> OUTDOOR_D
          - Otherwise -> INDOOR_DW
        """
        psw = [s for s in segments if s.edge_type == EdgeType.PSW]
        if len(psw) < 2:
            return []

        def _seg_info(s):
            """Return (orient, lateral, lo, hi, hi_pt, hi_pid, lo_pt, lo_pid)."""
            dx = abs(s.p2[0] - s.p1[0])
            dy = abs(s.p2[1] - s.p1[1])
            if dx >= dy:  # horizontal-dominant
                lat = (s.p1[1] + s.p2[1]) / 2
                if s.p1[0] <= s.p2[0]:
                    return "H", lat, s.p1[0], s.p2[0], s.p2, s.p2_id, s.p1, s.p1_id
                else:
                    return "H", lat, s.p2[0], s.p1[0], s.p1, s.p1_id, s.p2, s.p2_id
            else:          # vertical-dominant
                lat = (s.p1[0] + s.p2[0]) / 2
                if s.p1[1] <= s.p2[1]:
                    return "V", lat, s.p1[1], s.p2[1], s.p2, s.p2_id, s.p1, s.p1_id
                else:
                    return "V", lat, s.p2[1], s.p1[1], s.p1, s.p1_id, s.p2, s.p2_id

        info = [_seg_info(s) for s in psw]
        gap_segs: list[RawSegment] = []
        n = len(psw)

        for i in range(n):
            oi, lati, loi, hii, hi_pt_i, hi_pid_i, lo_pt_i, lo_pid_i = info[i]
            ext_i = getattr(psw[i], "_is_exterior", False)

            for j in range(i + 1, n):
                oj, latj, loj, hij, hi_pt_j, hi_pid_j, lo_pt_j, lo_pid_j = info[j]

                if oi != oj:
                    continue
                if abs(latj - lati) > self.collinear_tol_m:
                    continue

                # Determine which segment comes first along the main axis
                if hii <= loj:
                    gap = loj - hii
                    g_p1, g_p1_id = hi_pt_i, hi_pid_i
                    g_p2, g_p2_id = lo_pt_j, lo_pid_j
                elif hij <= loi:
                    gap = loi - hij
                    g_p1, g_p1_id = hi_pt_j, hi_pid_j
                    g_p2, g_p2_id = lo_pt_i, lo_pid_i
                else:
                    continue  # overlapping — not a gap

                ext_j = getattr(psw[j], "_is_exterior", False)
                is_exterior = ext_i and ext_j
                limit = (self.max_opening_ext_m if is_exterior
                         else self.max_opening_m)
                if gap < self.min_opening_m or gap > limit:
                    continue

                # exterior DOOR-sized gaps are outdoor doors; wider exterior
                # gaps are facade windows / balcony sliders (DW)
                etype = (EdgeType.OUTDOOR_D
                         if is_exterior
                         and self.door_opening_m <= gap <= self.door_max_m
                         else EdgeType.INDOOR_DW)

                length_m = float(np.linalg.norm(g_p2 - g_p1))
                angle = float(np.arctan2(abs(g_p2[0] - g_p1[0]),
                                         abs(g_p2[1] - g_p1[1])))
                gap_segs.append(RawSegment(
                    seg_id=uuid.uuid4().hex[:8],
                    p1=g_p1.copy(), p2=g_p2.copy(),
                    p1_id=g_p1_id,
                    p2_id=g_p2_id,
                    edge_type=etype,
                    thickness_m=0.0,
                    length_m=length_m,
                    angle_rad=angle,
                    wall_id="",
                ))

        logger.info(
            "Gap detection: %d openings found (INDOOR_DW=%d, OUTDOOR_D=%d)",
            len(gap_segs),
            sum(1 for s in gap_segs if s.edge_type == EdgeType.INDOOR_DW),
            sum(1 for s in gap_segs if s.edge_type == EdgeType.OUTDOOR_D),
        )
        return gap_segs


# =============================================================================
# 3. Graph Builder
#    Takes classified RawSegments -> networkx.Graph
# =============================================================================

class GraphBuilder:
    """
    Builds the structural graph from classified segments.

    Steps:
      1. Collect all segment endpoints
      2. Merge nearby points (proximity tolerance)
      3. Create graph nodes with normalised coordinates
      4. Create graph edges with typed features
      5. Optionally filter by GraphVariant
    """

    def __init__(
        self,
        *,
        variant: GraphVariant = GraphVariant.EDGE_PSW_DW,
        proximity_tol_m: float = 0.10,
        l_graph_scale: float = L_GRAPH_SCALE,
    ):
        self.variant = variant
        self.prox_tol = proximity_tol_m
        self.l_scale = l_graph_scale

    def build(self, segments: list[RawSegment]) -> nx.Graph:
        """
        Build the graph from classified segments.

        Returns
        -------
        nx.Graph with node/edge attributes.
        """
        allowed = _VARIANT_TYPES[self.variant]
        segs = [s for s in segments if s.edge_type in allowed]

        if not segs:
            logger.warning("GraphBuilder: no segments after filtering for %s",
                           self.variant.name)
            return nx.Graph()

        # --- Step 1-2: Collect & merge points ---
        canonical, merged_pts = self._merge_points(segs)

        # --- Step 3: Compute center & add nodes ---
        coords = np.vstack(list(merged_pts.values()))
        center_m = coords.mean(axis=0)

        # OOD check: the GNN checkpoint was trained on plans normalised to
        # l_graph_scale (~20 m).  A much smaller plan maps to a tiny cluster near
        # the origin -> the GNN is out-of-distribution and its output should be
        # treated as a soft prior only (Stage 4 applies engineering rules).
        extent_m = float(np.max(coords.max(axis=0) - coords.min(axis=0)))
        if extent_m * 1000.0 < self.l_scale:
            logger.warning(
                "Plan extent %.1fm < graph scale %.1fm: plan does not fill the "
                "trained coordinate range, so GNN inference is out-of-distribution; "
                "treat predictions as a soft prior only.",
                extent_m, self.l_scale / 1000.0)

        G = nx.Graph()
        for cid, xy in merged_pts.items():
            xn, yn = self._normalise(xy, center_m)
            G.add_node(cid,
                       x=float(xn), y=float(yn),           # normalised
                       x_m=float(xy[0]), y_m=float(xy[1]),  # original metres
                       )

        # --- Step 4: Add edges ---
        for s in segs:
            n1 = canonical.get(s.p1_id, s.p1_id)
            n2 = canonical.get(s.p2_id, s.p2_id)
            if n1 == n2:
                continue   # degenerate after merge

            p1 = merged_pts.get(n1, s.p1)
            p2 = merged_pts.get(n2, s.p2)
            norm1 = self._normalise(p1, center_m)
            norm2 = self._normalise(p2, center_m)
            ln = s.length_m * 1000.0 / self.l_scale

            # If edge already exists (e.g. opening on same wall segment),
            # networkx overwrites - that's fine, the later segment wins.
            G.add_edge(n1, n2,
                       id=s.seg_id,
                       edge_type=int(s.edge_type),
                       edge_type_name=s.edge_type.name,
                       x_left=float(norm1[0]),
                       y_left=float(norm1[1]),
                       x_right=float(norm2[0]),
                       y_right=float(norm2[1]),
                       length=float(ln),
                       # keep raw properties for Stage 4 / visualisation
                       thickness_m=s.thickness_m,
                       length_m=s.length_m,
                       angle_rad=s.angle_rad,
                       wall_id=s.wall_id,
                       rect_id=s.seg_id,
                       )

        # Store metadata on the graph object
        G.graph["variant"] = self.variant.name
        G.graph["l_graph_scale"] = self.l_scale
        G.graph["center_m"] = center_m.tolist()
        G.graph["proximity_tol_m"] = self.prox_tol
        G.graph["n_edge_classes"] = len(allowed)

        logger.info("GraphBuilder [%s]: %d nodes (%d merged), %d edges",
                    self.variant.name,
                    G.number_of_nodes(),
                    self._n_merged,
                    G.number_of_edges())
        return G

    # -- point merging ---------------------------------------------------------

    def _merge_points(self, segs):
        """
        Merge segment endpoints that are within proximity_tol of each other.
        Uses union-find for efficient clustering.

        Returns:
          canonical: dict[original_id -> canonical_id]
          merged_pts: dict[canonical_id -> averaged np.array position]
        """
        # Collect all unique point IDs and positions
        pts: dict[str, np.ndarray] = {}
        for s in segs:
            if s.p1_id not in pts:
                pts[s.p1_id] = s.p1.copy()
            if s.p2_id not in pts:
                pts[s.p2_id] = s.p2.copy()

        ids = list(pts.keys())
        n = len(ids)

        # Union-Find
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        # Merge points within tolerance
        # For small datasets, O(n^2) is fine. For large, use a KD-tree.
        if n > 500:
            from scipy.spatial import cKDTree
            coords = np.array([pts[pid] for pid in ids])
            tree = cKDTree(coords)
            pairs = tree.query_pairs(self.prox_tol)
            for i, j in pairs:
                union(i, j)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(pts[ids[i]] - pts[ids[j]]) < self.prox_tol:
                        union(i, j)

        # Build canonical mapping
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            r = find(i)
            clusters.setdefault(r, []).append(i)

        canonical: dict[str, str] = {}
        merged_pts: dict[str, np.ndarray] = {}
        for rep, members in clusters.items():
            cid = ids[rep]  # canonical = representative's original ID
            avg = np.mean([pts[ids[m]] for m in members], axis=0)
            merged_pts[cid] = avg
            for m in members:
                canonical[ids[m]] = cid

        self._n_merged = n - len(merged_pts)
        return canonical, merged_pts

    # -- coordinate normalization -----------------------------------------------

    def _normalise(self, xy_m, center_m):
        """Zhao et al. Eq(1): c_n = (c - c_center) * 1000 / L_GraphScale"""
        return (xy_m - center_m) * 1000.0 / self.l_scale


# =============================================================================
# 4. Data Augmentation (for training, Zhao et al. Section 4.3)
#    Flip (2), Rotate (4 x 90deg), Translate (21x21 grid of 2m steps)
# =============================================================================

class GraphAugmenter:
    """
    Augments a structural graph for GNN training.

    Methods from Zhao et al.:
      - Upside-down flip (2 cases: original + flipped)
      - 90-degree rotation (4 cases: 0, 90, 180, 270)
      - Translation grid (modulus 2m, range 0-20m in x and y => 21x21)
    Total multiplier: 2 x 4 x 21 x 21 = 3528

    Augmentation is applied to the *normalised* coordinates, so the center
    shifts accordingly (equivalent to different c_center choices).
    """

    def __init__(
        self,
        *,
        flips: bool = True,
        rotations: bool = True,
        translations: bool = True,
        translate_step_m: float = 2.0,
        translate_range_m: float = 20.0,
        l_graph_scale: float = L_GRAPH_SCALE,
    ):
        self.flips = flips
        self.rotations = rotations
        self.translations = translations
        self.t_step = translate_step_m
        self.t_range = translate_range_m
        self.l_scale = l_graph_scale

    def augment_all(self, G: nx.Graph) -> list[nx.Graph]:
        """Generate all augmented copies.  WARNING: up to 3528 copies."""
        results = []
        flip_cases = [False, True] if self.flips else [False]
        rot_cases = [0, 90, 180, 270] if self.rotations else [0]
        if self.translations:
            steps = np.arange(0, self.t_range + 0.01, self.t_step)
            trans_cases = [(dx, dy) for dx in steps for dy in steps]
        else:
            trans_cases = [(0.0, 0.0)]

        for flip in flip_cases:
            for rot in rot_cases:
                for tx, ty in trans_cases:
                    Ga = self._transform(G, flip, rot, tx, ty)
                    results.append(Ga)
        logger.info("Augmenter: %d copies from 1 graph", len(results))
        return results

    def augment_sample(self, G: nx.Graph, n: int = 16, rng=None) -> list[nx.Graph]:
        """Random sample of n augmentations (for mini-batch training)."""
        if rng is None:
            rng = np.random.default_rng()
        results = []
        for _ in range(n):
            flip = bool(rng.integers(2)) if self.flips else False
            rot = int(rng.choice([0, 90, 180, 270])) if self.rotations else 0
            tx = float(rng.uniform(0, self.t_range)) if self.translations else 0
            ty = float(rng.uniform(0, self.t_range)) if self.translations else 0
            results.append(self._transform(G, flip, rot, tx, ty))
        return results

    def _transform(self, G, flip, rot_deg, tx_m, ty_m):
        """Apply flip + rotation + translation to a copy of G."""
        Ga = G.copy()

        # Translation in normalised space: offset = metres * 1000 / L
        dx = tx_m * 1000.0 / self.l_scale
        dy = ty_m * 1000.0 / self.l_scale

        # Rotation matrix
        theta = math.radians(rot_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        for n in Ga.nodes():
            d = Ga.nodes[n]
            x, y = d["x"], d["y"]
            # Flip
            if flip:
                y = -y
            # Rotate
            xr = cos_t * x - sin_t * y
            yr = sin_t * x + cos_t * y
            # Translate
            d["x"] = float(xr + dx)
            d["y"] = float(yr + dy)

        for u, v in Ga.edges():
            d = Ga.edges[u, v]
            for prefix in [("x_left", "y_left"), ("x_right", "y_right")]:
                x, y = d[prefix[0]], d[prefix[1]]
                if flip:
                    y = -y
                xr = cos_t * x - sin_t * y
                yr = sin_t * x + cos_t * y
                d[prefix[0]] = float(xr + dx)
                d[prefix[1]] = float(yr + dy)

        Ga.graph["augmentation"] = dict(flip=flip, rotation=rot_deg,
                                         translate_m=(tx_m, ty_m))
        return Ga


# =============================================================================
# 5. Graph Export / Import (JSON)
# =============================================================================

def graph_to_json(G: nx.Graph) -> dict:
    """Serialise a networkx graph to a JSON-friendly dict."""
    nodes = []
    for n, d in G.nodes(data=True):
        nd = {"id": str(n)}
        nd.update({k: v for k, v in d.items()})
        nodes.append(nd)

    edges = []
    for u, v, d in G.edges(data=True):
        ed = {"source": str(u), "target": str(v)}
        ed.update({k: (v2 if not isinstance(v2, np.integer) else int(v2))
                   for k, v2 in d.items()})
        edges.append(ed)

    meta = {}
    for k, v in G.graph.items():
        if isinstance(v, np.ndarray):
            meta[k] = v.tolist()
        else:
            meta[k] = v

    return dict(metadata=meta, nodes=nodes, edges=edges)


def json_to_graph(data: dict) -> nx.Graph:
    """Reconstruct a networkx graph from JSON dict."""
    G = nx.Graph()
    G.graph.update(data.get("metadata", {}))
    for nd in data.get("nodes", []):
        nid = nd.pop("id")
        G.add_node(nid, **nd)
    for ed in data.get("edges", []):
        u = ed.pop("source")
        v = ed.pop("target")
        G.add_edge(u, v, **ed)
    return G


# =============================================================================
# 6. Graph Statistics & Validation
# =============================================================================

def graph_stats(G: nx.Graph) -> dict:
    """Compute summary statistics for a structural graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Edge type counts
    type_counts = {}
    lengths = []
    for u, v, d in G.edges(data=True):
        et = d.get("edge_type_name", "unknown")
        type_counts[et] = type_counts.get(et, 0) + 1
        lengths.append(d.get("length_m", 0))

    # Degree distribution
    degrees = [d for _, d in G.degree()]

    # Coordinate extent (normalised)
    if n_nodes > 0:
        xs = [G.nodes[n]["x"] for n in G.nodes()]
        ys = [G.nodes[n]["y"] for n in G.nodes()]
        extent_norm = dict(x_min=min(xs), x_max=max(xs),
                           y_min=min(ys), y_max=max(ys))
    else:
        extent_norm = {}

    # Connected components
    n_components = nx.number_connected_components(G)

    return dict(
        n_nodes=n_nodes,
        n_edges=n_edges,
        edge_type_counts=type_counts,
        mean_degree=round(np.mean(degrees), 2) if degrees else 0,
        max_degree=max(degrees) if degrees else 0,
        mean_length_m=round(np.mean(lengths), 3) if lengths else 0,
        n_connected_components=n_components,
        extent_normalised=extent_norm,
        variant=G.graph.get("variant", "unknown"),
    )


# =============================================================================
# 7. Visualization
# =============================================================================

def visualize_graph(
    G: nx.Graph,
    out_path: str,
    *,
    title: str = "Stage 3 - Graph Construction",
    show_labels: bool = True,
    use_metres: bool = True,
):
    """Draw the structural graph with colour-coded edge types."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not available -- skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Node positions
    if use_metres:
        pos = {n: (d.get("x_m", d["x"]), d.get("y_m", d["y"]))
               for n, d in G.nodes(data=True)}
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    else:
        pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
        ax.set_xlabel("x (normalised)"); ax.set_ylabel("y (normalised)")

    # Edge colours by type
    _COLORS = {
        EdgeType.PSW:       "#1a1a1a",   # black
        EdgeType.INDOOR_DW: "#43A047",   # green
        EdgeType.OUTDOOR_D: "#1565C0",   # blue
        EdgeType.PW:        "#AB47BC",   # purple
    }
    _STYLES = {
        EdgeType.PSW:       "solid",
        EdgeType.INDOOR_DW: "dashed",
        EdgeType.OUTDOOR_D: "dashdot",
        EdgeType.PW:        "dotted",
    }
    _WIDTHS = {
        EdgeType.PSW: 3.0, EdgeType.INDOOR_DW: 2.0,
        EdgeType.OUTDOOR_D: 2.5, EdgeType.PW: 1.5,
    }

    # Draw edges by type
    for etype in EdgeType:
        elist = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get("edge_type", 0) == int(etype)]
        if not elist:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=elist, ax=ax,
            edge_color=_COLORS.get(etype, "#999"),
            style=_STYLES.get(etype, "solid"),
            width=_WIDTHS.get(etype, 2),
        )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color="#333",
                           edgecolors="white", linewidths=0.5)

    # Labels
    if show_labels:
        for n, (x, y) in pos.items():
            ax.annotate(str(n)[:6], (x, y), fontsize=5, ha="center",
                        va="bottom", color="#555",
                        xytext=(0, 5), textcoords="offset points")
        # Edge labels: length + thickness
        for u, v, d in G.edges(data=True):
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            lbl = f"L={d.get('length_m', 0):.2f}"
            if d.get("thickness_m", 0) > 0:
                lbl += f"\nt={d['thickness_m']:.3f}"
            ax.annotate(lbl, (mx, my), fontsize=5, ha="center",
                        color="#1565C0",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#ccc", alpha=0.85))

    # Legend
    legend_items = []
    for etype in EdgeType:
        if any(d.get("edge_type", 0) == int(etype)
               for _, _, d in G.edges(data=True)):
            legend_items.append(
                Patch(facecolor=_COLORS[etype], label=etype.name))
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    # Stats annotation
    stats = graph_stats(G)
    info = (f"Nodes: {stats['n_nodes']}  Edges: {stats['n_edges']}  "
            f"Components: {stats['n_connected_components']}\n"
            f"Variant: {stats['variant']}  "
            f"Mean degree: {stats['mean_degree']}")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7,
            va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Visualization -> %s", out_path)


# =============================================================================
# 8. End-to-end: Stage 2 JSON -> Stage 3 Graph
# =============================================================================

def build_graph(
    stage2_floor: dict,
    *,
    openings: Optional[list[dict]] = None,
    variant: GraphVariant = GraphVariant.EDGE_PSW_DW,
    proximity_tol_m: float = 0.10,
    classify_mode: str = "auto",
    psw_min_thickness: float = 0.06,
    pw_min_thickness: float = 0.03,
    pw_max_thickness: float = 0.06,
    auto_detect_gaps: bool = True,
    min_opening_m: float = 0.40,
    max_opening_m: float = 2.60,
    max_opening_ext_m: float = 3.20,
    door_opening_m: float = 0.90,
    collinear_tol_m: float = 0.08,
) -> nx.Graph:
    """
    Convenience function: Stage 2 JSON -> Stage 3 Graph in one call.

    Parameters
    ----------
    stage2_floor : dict
        Stage 2 output JSON.
    openings : list[dict], optional
        Door/window annotations.
    variant : GraphVariant
        Which graph representation to use.
    proximity_tol_m : float
        Point merge tolerance in metres.
    classify_mode : str
        "auto" (default) — adapt PSW/PW threshold to data.
        "fixed" — use explicit psw_min_thickness / pw thresholds.
    psw_min_thickness : float
        Minimum wall thickness (m) for PSW (only used in "fixed" mode).
    pw_min_thickness : float
        Minimum thickness for partition wall classification.
    pw_max_thickness : float
        Maximum thickness for partition wall classification.
    auto_detect_gaps : bool
        Infer window/door edges from gaps between collinear wall segments.
    min_opening_m, max_opening_m : float
        Gap size range (m) to classify as an opening.
    door_opening_m : float
        Gap >= this on an exterior wall is classified as OUTDOOR_D.
    collinear_tol_m : float
        Lateral tolerance for treating two segments as on the same axis.

    Returns
    -------
    nx.Graph
    """
    adapter = Stage2To3Adapter(
        classify_mode=classify_mode,
        psw_min_thickness=psw_min_thickness,
        pw_min_thickness=pw_min_thickness,
        pw_max_thickness=pw_max_thickness,
        auto_detect_gaps=auto_detect_gaps,
        min_opening_m=min_opening_m,
        max_opening_m=max_opening_m,
        max_opening_ext_m=max_opening_ext_m,
        door_opening_m=door_opening_m,
        collinear_tol_m=collinear_tol_m,
    )
    segments = adapter.parse(stage2_floor, openings)

    builder = GraphBuilder(
        variant=variant,
        proximity_tol_m=proximity_tol_m,
    )
    return builder.build(segments)


# =============================================================================
# 9. CLI entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Stage 3 - Graph Construction")
    ap.add_argument("-i", "--input", default=None,
                    help="Stage 2 floor JSON (default: floor_output.json)")
    ap.add_argument("-o", "--output", default="graph_output.json",
                    help="Stage 3 graph JSON output")
    ap.add_argument("-v", "--vis", default="graph_construction_vis.png")
    ap.add_argument("--openings", default=None,
                    help="Optional openings JSON file")
    ap.add_argument("--variant", default="EDGE_PSW_DW",
                    choices=[v.name for v in GraphVariant],
                    help="Graph representation variant")
    ap.add_argument("--proximity-tol", type=float, default=0.10,
                    help="Point merge tolerance in metres")
    ap.add_argument("--classify-mode", default="auto",
                    choices=["auto", "fixed"],
                    help="Wall classification mode (default: auto)")
    ap.add_argument("--psw-min-t", type=float, default=0.06,
                    help="Min thickness (m) for PSW (fixed mode only)")
    ap.add_argument("--pw-min-t", type=float, default=0.03,
                    help="Min thickness (m) for PW classification")
    ap.add_argument("--pw-max-t", type=float, default=0.06,
                    help="Max thickness (m) for PW classification")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    inp = args.input or os.path.join(project_root, "floor_output.json")
    # -- Load Stage 2 --
    #inp = args.input or os.path.join(
    #    os.path.dirname(os.path.abspath(__file__)), "floor_output.json")
    if not os.path.exists(inp):
        logger.error("Not found: %s  -- run vectorization.py first", inp)
        return
    with open(inp) as f:
        floor = json.load(f)
    logger.info("Loaded Stage 2: %d walls", len(floor.get("walls", [])))

    # -- Load openings (optional) --
    openings = None
    if args.openings and os.path.exists(args.openings):
        with open(args.openings) as f:
            openings = json.load(f)
        logger.info("Loaded %d openings", len(openings))

    # -- Build graph --
    variant = GraphVariant[args.variant]
    G = build_graph(
        floor,
        openings=openings,
        variant=variant,
        proximity_tol_m=args.proximity_tol,
        classify_mode=args.classify_mode,
        psw_min_thickness=args.psw_min_t,
        pw_min_thickness=args.pw_min_t,
        pw_max_thickness=args.pw_max_t,
    )

    # -- Statistics --
    stats = graph_stats(G)
    print("\n" + "=" * 64)
    print("  STAGE 3 - GRAPH CONSTRUCTION SUMMARY")
    print("=" * 64)
    print(f"\n  Variant:     {stats['variant']}")
    print(f"  Nodes:       {stats['n_nodes']}")
    print(f"  Edges:       {stats['n_edges']}")
    print(f"  Components:  {stats['n_connected_components']}")
    print(f"  Mean degree: {stats['mean_degree']}")
    print(f"  Max degree:  {stats['max_degree']}")
    print(f"  Mean length: {stats['mean_length_m']:.3f} m")
    print(f"\n  Edge type breakdown:")
    for et, cnt in stats["edge_type_counts"].items():
        print(f"    {et:15s}  {cnt}")

    if stats["extent_normalised"]:
        ex = stats["extent_normalised"]
        print(f"\n  Normalised extent:")
        print(f"    x: [{ex['x_min']:+.4f}, {ex['x_max']:+.4f}]")
        print(f"    y: [{ex['y_min']:+.4f}, {ex['y_max']:+.4f}]")

    print(f"\n  Graph metadata:")
    print(f"    L_GraphScale:   {G.graph.get('l_graph_scale', L_GRAPH_SCALE):.0f} mm")
    print(f"    Center (m):     {G.graph.get('center_m', [])}")
    print(f"    Proximity tol:  {G.graph.get('proximity_tol_m', 0.10):.2f} m")

    # -- Node & edge detail --
    print(f"\n  Nodes:")
    for n, d in G.nodes(data=True):
        print(f"    {str(n)[:8]:8s}  ({d['x_m']:.3f}, {d['y_m']:.3f})m  "
              f"norm=({d['x']:+.4f}, {d['y']:+.4f})  "
              f"deg={G.degree(n)}")

    print(f"\n  Edges:")
    for u, v, d in G.edges(data=True):
        print(f"    {str(u)[:6]}-{str(v)[:6]}  "
              f"{d.get('edge_type_name','?'):12s}  "
              f"L={d.get('length_m',0):.2f}m  "
              f"t={d.get('thickness_m',0):.3f}m  "
              f"wall={d.get('wall_id','')[:8]}")

    # -- Export --
    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.output)
    gj = graph_to_json(G)
    with open(out_path, "w") as f:
        json.dump(gj, f, indent=2)
    logger.info("Exported graph -> %s", out_path)

    vis_path = os.path.join(base, args.vis)
    visualize_graph(G, vis_path)

    print(f"\n  Exported: {out_path}")
    print(f"  Visual:   {vis_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()