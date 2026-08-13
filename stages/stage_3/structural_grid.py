"""
Structural Grid Detection & Column Placement
=============================================
Shared module used by:
  - Stage 3   : candidate column nodes for the GNN graph
  - Stage 5   : grid-based column + interior-beam placement in the FEM export
  - Training  : column pseudo-labels for GNN rule-distillation

Replaces the old "column at every footprint corner" heuristic with an
engineering-style workflow:

  1. Cluster wall centerlines into structural axes (X- and Y-grid lines).
  2. Grid points = axis intersections inside the footprint.
  3. A grid point receives a column iff no (shear) wall already supports it.
  4. Span enforcement: supports along any axis may not be further apart than
     max_span; intermediate columns are inserted at subdivided positions.
  5. Tributary-area column sizing.

All coordinates are metres. Only numpy + scipy required.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Data types
# =============================================================================

@dataclass
class Segment:
    """A wall centerline segment in metres."""
    p1: np.ndarray
    p2: np.ndarray
    thickness: float = 0.0
    is_sw: bool = False
    rect_id: str = ""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def orient(self) -> str:
        d = np.abs(self.p2 - self.p1)
        return "H" if d[0] >= d[1] else "V"

    @property
    def offset(self) -> float:
        """Constant coordinate: y for H segments, x for V segments."""
        ax = 1 if self.orient == "H" else 0
        return float((self.p1[ax] + self.p2[ax]) / 2.0)


@dataclass
class Axis:
    """One structural grid line."""
    orient: str            # "X" = vertical line (const x), "Y" = horizontal (const y)
    offset: float          # the constant coordinate
    support_len: float     # total wall length backing this axis
    lo: float              # extent along the axis
    hi: float
    virtual: bool = False  # inserted by span enforcement, no wall backing


@dataclass
class ColumnPoint:
    x: float
    y: float
    reason: str            # "grid" | "span" | "pier_end"
    size_m: float = 0.30

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class GridResult:
    axes_x: list = field(default_factory=list)   # vertical lines (const x)
    axes_y: list = field(default_factory=list)   # horizontal lines (const y)
    columns: list = field(default_factory=list)  # list[ColumnPoint]
    candidates: list = field(default_factory=list)  # all grid pts inside footprint
    hull: np.ndarray | None = None


# =============================================================================
# 2. Converters
# =============================================================================

def segments_from_floor(floor: dict, *, sw_ids: set | None = None) -> list[Segment]:
    """Extract wall centerline segments from a Stage 2 floor JSON."""
    segs = []
    for w in floor.get("walls", []):
        if w.get("rejected"):
            continue
        pts = {p["id"]: np.array([p["x"], p["y"]], float)
               for p in w.get("points", [])}
        for r in w.get("rects", []):
            p1, p2 = pts.get(r["p1_id"]), pts.get(r["p2_id"])
            if p1 is None or p2 is None:
                continue
            rid = r["id"]
            is_sw = bool(r.get("is_shear_wall", False))
            if sw_ids is not None:
                is_sw = rid in sw_ids
            segs.append(Segment(p1, p2, float(r.get("thickness_m", 0.0)),
                                is_sw, rid))
    return segs


def segments_from_graph(G) -> list[Segment]:
    """Extract segments from a Stage 3 networkx graph (uses metre coords)."""
    segs = []
    for u, v, d in G.edges(data=True):
        nu, nv = G.nodes[u], G.nodes[v]
        if "x_m" in nu and "x_m" in nv:
            p1 = np.array([nu["x_m"], nu["y_m"]], float)
            p2 = np.array([nv["x_m"], nv["y_m"]], float)
        else:
            p1 = np.array([nu.get("x", 0.0), nu.get("y", 0.0)], float)
            p2 = np.array([nv.get("x", 0.0), nv.get("y", 0.0)], float)
        segs.append(Segment(p1, p2, float(d.get("thickness_m", 0.0)),
                            bool(d.get("is_shear_wall", d.get("edge_type", 0) == 0)),
                            str(d.get("id", f"{u}_{v}"))))
    return segs


# =============================================================================
# 3. Axis detection
# =============================================================================

def detect_axes(segments: list[Segment], *, axis_tol: float = 0.30,
                min_axis_len: float = 1.5,
                min_axis_spacing: float = 1.8) -> tuple[list[Axis], list[Axis]]:
    """
    Cluster wall centerlines into structural grid axes.

    Returns (axes_x, axes_y):
      axes_x : vertical grid lines  (constant x), from V segments
      axes_y : horizontal grid lines (constant y), from H segments

    Clustering is 1-D greedy on the segment offset, weighted by length;
    clusters whose total backing wall length < min_axis_len are dropped.
    Axes closer than min_axis_spacing are pruned keeping the dominant one
    (most backing wall length) — real structural grids run at metre spacing,
    not at every partition wall.  The two extreme axes (building edges) are
    always kept.
    """
    ax_x = _cluster_axes([s for s in segments if s.orient == "V"],
                         "X", axis_tol, min_axis_len)
    ax_y = _cluster_axes([s for s in segments if s.orient == "H"],
                         "Y", axis_tol, min_axis_len)
    ax_x = _prune_axes(ax_x, min_axis_spacing)
    ax_y = _prune_axes(ax_y, min_axis_spacing)
    return ax_x, ax_y


def _prune_axes(axes: list[Axis], min_spacing: float) -> list[Axis]:
    """Keep dominant axes at least min_spacing apart (extremes always kept)."""
    if len(axes) <= 2:
        return axes
    ends = {id(axes[0]), id(axes[-1])}
    order = sorted(axes, key=lambda a: (id(a) not in ends, -a.support_len))
    kept: list[Axis] = []
    for a in order:
        if all(abs(a.offset - k.offset) >= min_spacing for k in kept):
            kept.append(a)
    kept.sort(key=lambda a: a.offset)
    return kept


def _cluster_axes(segs: list[Segment], orient: str,
                  tol: float, min_len: float) -> list[Axis]:
    if not segs:
        return []
    run_ax = 1 if orient == "X" else 0   # extent axis (along the grid line)
    items = sorted(segs, key=lambda s: s.offset)

    clusters: list[list[Segment]] = []
    for s in items:
        if clusters:
            cur = clusters[-1]
            w = sum(m.length for m in cur)
            mean = sum(m.offset * m.length for m in cur) / max(w, 1e-9)
            if abs(s.offset - mean) <= tol:
                cur.append(s)
                continue
        clusters.append([s])

    axes = []
    for cl in clusters:
        w = sum(m.length for m in cl)
        if w < min_len:
            continue
        mean = sum(m.offset * m.length for m in cl) / max(w, 1e-9)
        lo = min(min(m.p1[run_ax], m.p2[run_ax]) for m in cl)
        hi = max(max(m.p1[run_ax], m.p2[run_ax]) for m in cl)
        axes.append(Axis(orient, float(mean), float(w), float(lo), float(hi)))
    axes.sort(key=lambda a: a.offset)
    return axes


def enforce_axis_spacing(axes: list[Axis], *, max_span: float = 6.0) -> list[Axis]:
    """
    Insert virtual axes so that no two adjacent grid lines are further apart
    than max_span (slab one-way span limit).
    """
    if len(axes) < 2:
        return list(axes)
    out = [axes[0]]
    for nxt in axes[1:]:
        prev = out[-1]
        gap = nxt.offset - prev.offset
        if gap > max_span:
            n_new = int(math.ceil(gap / max_span)) - 1
            step = gap / (n_new + 1)
            lo = min(prev.lo, nxt.lo)
            hi = max(prev.hi, nxt.hi)
            for k in range(1, n_new + 1):
                out.append(Axis(prev.orient, prev.offset + k * step,
                                0.0, lo, hi, virtual=True))
        out.append(nxt)
    return out


# =============================================================================
# 4. Footprint / geometry helpers
# =============================================================================

def footprint_hull(segments: list[Segment]) -> np.ndarray:
    """Convex hull vertices (CCW) of all segment endpoints."""
    from scipy.spatial import ConvexHull
    pts = np.vstack([np.vstack([s.p1, s.p2]) for s in segments])
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def point_in_hull(pt, hull_pts: np.ndarray, *, tol: float = 0.05) -> bool:
    """True if pt is inside (or within tol of) the convex hull polygon."""
    n = len(hull_pts)
    if n < 3:
        return False
    for i in range(n):
        a, b = hull_pts[i], hull_pts[(i + 1) % n]
        edge = b - a
        # CCW hull: interior is to the left; allow tol outside
        cross = edge[0] * (pt[1] - a[1]) - edge[1] * (pt[0] - a[0])
        if cross < -tol * (np.linalg.norm(edge) + 1e-9):
            return False
    return True


def dist_point_to_segment(pt, s: Segment) -> float:
    ab = s.p2 - s.p1
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(pt - s.p1))
    t = np.clip(np.dot(pt - s.p1, ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(pt - (s.p1 + t * ab)))


def is_supported(pt, segments: list[Segment], *, tol: float = 0.35,
                 sw_only: bool = True) -> bool:
    """A point is vertically supported if a (shear) wall passes within tol."""
    for s in segments:
        if sw_only and not s.is_sw:
            continue
        if dist_point_to_segment(np.asarray(pt, float), s) <= tol:
            return True
    return False


# =============================================================================
# 5. Column placement
# =============================================================================

def place_columns(segments: list[Segment], *,
                  axis_tol: float = 0.30,
                  min_axis_len: float = 1.5,
                  min_axis_spacing: float = 1.8,
                  max_span: float = 6.0,
                  support_tol: float = 0.35,
                  min_col_spacing: float = 0.60,
                  sw_only_support: bool = True,
                  pier_end_columns: bool = True) -> GridResult:
    """
    Engineering-rule column placement on the structural grid.

    Steps:
      1. detect axes from ALL walls (architecture defines the grid)
      2. insert virtual axes so slab spans stay <= max_span
      3. columns at grid intersections not supported by a shear wall
      4. boundary columns at shear-wall pier ends
      5. along-axis span check: consecutive supports further apart than
         max_span get intermediate columns
      6. de-duplicate within min_col_spacing

    Returns GridResult (axes, columns with reasons, candidate points, hull).
    """
    res = GridResult()
    if not segments:
        return res

    res.hull = footprint_hull(segments)
    ax_x, ax_y = detect_axes(segments, axis_tol=axis_tol,
                             min_axis_len=min_axis_len,
                             min_axis_spacing=min_axis_spacing)
    ax_x = enforce_axis_spacing(ax_x, max_span=max_span)
    ax_y = enforce_axis_spacing(ax_y, max_span=max_span)
    res.axes_x, res.axes_y = ax_x, ax_y

    sw_segs = [s for s in segments if s.is_sw] if sw_only_support else segments
    support_segs = sw_segs if sw_segs else segments  # fallback: any wall

    cols: list[ColumnPoint] = []

    # -- 3. grid intersections ------------------------------------------------
    for ax in ax_x:
        for ay in ax_y:
            pt = np.array([ax.offset, ay.offset])
            # Only where the two axes actually overlap in extent (with slack)
            slack = 0.5
            if not (ax.lo - slack <= pt[1] <= ax.hi + slack and
                    ay.lo - slack <= pt[0] <= ay.hi + slack):
                continue
            if not point_in_hull(pt, res.hull, tol=0.10):
                continue
            res.candidates.append((float(pt[0]), float(pt[1])))
            if not is_supported(pt, support_segs, tol=support_tol,
                                sw_only=False):
                cols.append(ColumnPoint(float(pt[0]), float(pt[1]), "grid"))

    # -- 4. shear-wall pier FREE ends (not junctions with other SW walls) -----
    if pier_end_columns:
        for s in sw_segs:
            for pt in (s.p1, s.p2):
                joined = any(o is not s and
                             dist_point_to_segment(pt, o) <= support_tol
                             for o in sw_segs)
                if not joined:
                    cols.append(ColumnPoint(float(pt[0]), float(pt[1]),
                                            "pier_end"))

    # -- 5. along-axis support gaps ----------------------------------------------
    support_pts = [c.xy() for c in cols]
    for s in support_segs:
        support_pts.extend(_sample_segment(s, step=0.5))
    span_cols = _fill_span_gaps(ax_x, ax_y, support_pts, res.hull,
                                max_span=max_span)
    cols.extend(span_cols)

    # -- 6. de-duplicate ------------------------------------------------------------
    res.columns = _dedup_columns(cols, min_col_spacing)
    logger.info("Grid: %d x-axes, %d y-axes, %d candidates, %d columns "
                "(%d grid, %d pier_end, %d span)",
                len(ax_x), len(ax_y), len(res.candidates), len(res.columns),
                sum(1 for c in res.columns if c.reason == "grid"),
                sum(1 for c in res.columns if c.reason == "pier_end"),
                sum(1 for c in res.columns if c.reason == "span"))
    return res


def _sample_segment(s: Segment, step: float = 0.5) -> list[np.ndarray]:
    n = max(2, int(s.length / step) + 1)
    return [s.p1 + t * (s.p2 - s.p1) for t in np.linspace(0, 1, n)]


def _fill_span_gaps(ax_x, ax_y, support_pts, hull, *,
                    max_span: float, band: float = 0.40) -> list[ColumnPoint]:
    """
    For every axis, project nearby supports onto the axis line; where two
    consecutive supports are further apart than max_span, insert columns.
    """
    out = []
    if not support_pts:
        return out
    sp = np.vstack(support_pts)

    for axes, const_i, run_i in ((ax_x, 0, 1), (ax_y, 1, 0)):
        for ax in axes:
            near = sp[np.abs(sp[:, const_i] - ax.offset) <= band]
            if near.shape[0] == 0:
                continue
            ts = np.sort(near[:, run_i])
            ts = ts[(ts >= ax.lo - band) & (ts <= ax.hi + band)]
            if ts.size < 2:
                continue
            for a, b in zip(ts[:-1], ts[1:]):
                gap = b - a
                if gap <= max_span:
                    continue
                n_new = int(math.ceil(gap / max_span)) - 1
                for k in range(1, n_new + 1):
                    v = a + k * gap / (n_new + 1)
                    pt = np.zeros(2)
                    pt[const_i] = ax.offset
                    pt[run_i] = v
                    if point_in_hull(pt, hull, tol=0.10):
                        out.append(ColumnPoint(float(pt[0]), float(pt[1]),
                                               "span"))
    return out


def _dedup_columns(cols: list[ColumnPoint], min_spacing: float) -> list[ColumnPoint]:
    # priority: grid > span > pier_end (grid columns are the "real" ones)
    prio = {"grid": 0, "span": 1, "pier_end": 2}
    cols = sorted(cols, key=lambda c: prio.get(c.reason, 9))
    kept: list[ColumnPoint] = []
    for c in cols:
        if all(np.linalg.norm(c.xy() - k.xy()) >= min_spacing for k in kept):
            kept.append(c)
    return kept


# =============================================================================
# 6. Column sizing (tributary area)
# =============================================================================

def size_column(spacing_x: float, spacing_y: float, n_stories: int, *,
                fc_mpa: float = 30.0, w_kpa: float = 12.0,
                min_b: float = 0.25, snap: float = 0.05) -> float:
    """
    Square column width b from tributary axial load.

      N_Ed = A_trib * w * n_stories        (w ~ 12 kPa: DL+LL, service->ULS-ish)
      b    = sqrt(N_Ed / (0.35 * fc))      (crude axial capacity check)

    Snapped up to `snap` increments, floored at min_b.
    """
    a_trib = max(spacing_x, 0.5) * max(spacing_y, 0.5)
    n_ed_n = a_trib * w_kpa * 1000.0 * max(n_stories, 1)     # N
    b = math.sqrt(n_ed_n / (0.35 * fc_mpa * 1e6))            # m
    b = max(min_b, b)
    return math.ceil(b / snap) * snap


def size_all_columns(result: GridResult, n_stories: int, *,
                     fc_mpa: float = 30.0) -> None:
    """Assign per-column size from local axis spacing (in-place)."""
    def _spacing(axes, v):
        offs = [a.offset for a in axes]
        if len(offs) < 2:
            return 4.0
        below = [v - o for o in offs if o < v - 1e-6]
        above = [o - v for o in offs if o > v + 1e-6]
        lo = min(below) if below else max(above) if above else 4.0
        hi = min(above) if above else lo
        return (lo + hi) / 2.0

    for c in result.columns:
        sx = _spacing(result.axes_x, c.x)
        sy = _spacing(result.axes_y, c.y)
        c.size_m = size_column(sx, sy, n_stories, fc_mpa=fc_mpa)


# =============================================================================
# 7. Interior beam lines  (grid segments between vertical supports)
# =============================================================================

def beam_lines(result: GridResult, segments: list[Segment], *,
               support_tol: float = 0.40,
               min_beam_len: float = 0.80) -> list[tuple]:
    """
    Beam spans along grid axes between consecutive vertical supports
    (columns or walls crossing the axis).  Returns [((x1,y1),(x2,y2)), ...].

    Perimeter ring beams are handled by the exporter; this generates the
    INTERIOR load paths that were previously missing.
    """
    supports = [c.xy() for c in result.columns]
    for s in segments:
        if s.is_sw:
            supports.extend(_sample_segment(s, step=0.5))
    if not supports:
        return []
    sp = np.vstack(supports)

    out = []
    for axes, const_i, run_i in ((result.axes_x, 0, 1),
                                 (result.axes_y, 1, 0)):
        for ax in axes:
            near = sp[np.abs(sp[:, const_i] - ax.offset) <= support_tol]
            if near.shape[0] < 2:
                continue
            ts = np.unique(np.round(np.sort(near[:, run_i]), 3))
            ts = ts[(ts >= ax.lo - support_tol) & (ts <= ax.hi + support_tol)]
            # consecutive supports -> beam span (merge sub-0.5m gaps)
            for a, b in zip(ts[:-1], ts[1:]):
                if b - a < min_beam_len:
                    continue
                p1, p2 = np.zeros(2), np.zeros(2)
                p1[const_i] = p2[const_i] = ax.offset
                p1[run_i], p2[run_i] = a, b
                out.append(((float(p1[0]), float(p1[1])),
                            (float(p2[0]), float(p2[1]))))
    return out


# =============================================================================
# 8. CLI smoke test
# =============================================================================

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Structural grid smoke test")
    ap.add_argument("-i", "--input", default="floor_output.json")
    ap.add_argument("--sw-all", action="store_true",
                    help="Treat every wall as shear wall (no enrichment yet)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    with open(args.input) as f:
        floor = json.load(f)
    segs = segments_from_floor(floor)
    if args.sw_all:
        for s in segs:
            s.is_sw = True
    res = place_columns(segs)
    size_all_columns(res, n_stories=3)
    beams = beam_lines(res, segs)

    print(f"axes_x: {[round(a.offset,2) for a in res.axes_x]}")
    print(f"axes_y: {[round(a.offset,2) for a in res.axes_y]}")
    print(f"columns ({len(res.columns)}):")
    for c in res.columns:
        print(f"  ({c.x:6.2f},{c.y:6.2f})  {c.reason:9s} b={c.size_m:.2f}")
    print(f"interior beam lines: {len(beams)}")
