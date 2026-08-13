"""
Stage 2 - Vectorization Pipeline (v2)
=======================================
Converts a binary wall mask (output of Stage 1 segmentation) into a structured
collection of Wall objects, each decomposed into rectangles connected by points.

Key improvement over v1: Uses skeleton-based decomposition instead of
polygon-validate-split.  This naturally handles L, T, C, U, and + shaped
walls by tracing the medial axis into individual straight segments.

Data model follows Pizarro & Massone (2021):
  - Point: unique (x, y) vertex where rectangles connect
  - Rect:  defined by two Point IDs + thickness, representing a wall segment
  - Wall:  a collection of connected Points and Rects

Pipeline sub-steps:
  2.1  Mask cleanup & connected component extraction
  2.2  Skeleton extraction & line segment tracing
  2.3  Thickness measurement via distance transform
  2.4  Point merging & Wall assembly
  2.5  Scale calibration & coordinate assignment

References:
  [1] Pizarro & Massone, Eng. Structures 241 (2021) 112377
  [2] Zhao et al., Adv. Eng. Informatics 55 (2023) 101886
"""

from __future__ import annotations

import uuid
import math
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Geometry helpers
# ======================================================================

def polygon_area(pts: np.ndarray) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def minimum_area_bounding_rect(pts: np.ndarray):
    if len(pts) < 3:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        corners = np.array([mn, [mx[0], mn[1]], mx, [mn[0], mx[1]]])
        w, h = mx[0] - mn[0], mx[1] - mn[1]
        return (corners, max(w, h), min(w, h), 0.0)
    pts_f32 = pts.astype(np.float32)
    rect = cv2.minAreaRect(pts_f32)
    box_pts = cv2.boxPoints(rect)
    _, (w, h), angle_deg = rect
    angle_rad = np.radians(angle_deg)
    if w >= h:
        length, width = w, h
    else:
        length, width = h, w
        angle_rad += np.pi / 2
    return box_pts, float(length), float(width), float(angle_rad)


# ======================================================================
# Data Classes
# ======================================================================

@dataclass
class Point:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Rect:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    p1_id: str = ""
    p2_id: str = ""
    thickness: float = 0.0
    length: float = 0.0
    angle: float = 0.0  # w.r.t. y-axis, [0, pi]


@dataclass
class Wall:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    points: dict = field(default_factory=dict)
    rects: dict = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: str = ""


@dataclass
class Floor:
    floor_id: str = "floor_01"
    walls: list = field(default_factory=list)
    scale: float = 1.0
    height: float = 2.80
    offset_x: float = 0.0
    offset_y: float = 0.0


# ======================================================================
# 2.1  Mask Cleanup & Connected Component Extraction
# ======================================================================

class MaskCleaner:

    def __init__(self, close_kernel_size=5, open_kernel_size=3,
                 min_component_area=50):
        self.close_ks = close_kernel_size
        self.open_ks = open_kernel_size
        self.min_area = min_component_area

    def clean(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        ck = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (self.close_ks, self.close_ks))
        ok = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (self.open_ks, self.open_ks))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, ok)

        # Remove small components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)
        removed = 0
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                cleaned[labels == i] = 0
                removed += 1
        logger.info("Mask cleanup: %d components, %d removed (<%d px)",
                    n_labels - 1 - removed, removed, self.min_area)
        return cleaned

    def extract_components(self, mask: np.ndarray):
        """
        Extract each connected component as a separate binary sub-mask.
        Returns list of (sub_mask, bbox, area).
        """
        n_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            sub = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
            components.append((sub, (x, y, w, h), area))
        logger.info("Extracted %d connected components", len(components))
        return components


# ======================================================================
# 2.2  Skeleton Extraction & Segment Tracing
# ======================================================================

class SkeletonTracer:
    """
    Extracts the morphological skeleton of a wall mask region and traces
    it into individual straight line segments.

    Improvements over naive approach:
      - Prunes short spurs (< prune_length pixels)
      - Deduplicates branches (same path traced from both ends)
      - Uses higher RDP epsilon for cleaner straight segments
      - Snaps near-orthogonal to exact 0/90 degrees
    """

    def __init__(self, min_segment_length_px=8, rdp_epsilon=3.0,
                 angle_snap_deg=8.0, prune_length_px=6):
        self.min_seg_len = min_segment_length_px
        self.rdp_eps = rdp_epsilon
        self.snap_deg = angle_snap_deg
        self.prune_len = prune_length_px

    def skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Morphological thinning to 1-pixel-wide skeleton."""
        # Pad with 1px zero border so erosion converges at component edges
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1,
                                   cv2.BORDER_CONSTANT, value=0)
        try:
            skel = cv2.ximgproc.thinning(mask, thinningType=0)
            return skel[1:-1, 1:-1]
        except AttributeError:
            pass
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros_like(mask)
        temp = mask.copy()
        while True:
            eroded = cv2.erode(temp, kernel)
            opened = cv2.dilate(eroded, kernel)
            skel = cv2.bitwise_or(skel, temp - opened)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skel[1:-1, 1:-1]

    def _get_degree_map(self, skel):
        """Compute the connectivity degree of each skeleton pixel."""
        h, w = skel.shape
        degree = np.zeros((h, w), dtype=np.int32)
        s = (skel > 0).astype(np.int32)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = np.zeros_like(s)
                y1 = max(0, dy); y2 = min(h, h + dy)
                x1 = max(0, dx); x2 = min(w, w + dx)
                sy1 = max(0, -dy); sy2 = min(h, h - dy)
                sx1 = max(0, -dx); sx2 = min(w, w - dx)
                shifted[y1:y2, x1:x2] = s[sy1:sy2, sx1:sx2]
                degree += shifted
        degree[skel == 0] = 0
        return degree

    def _prune_skeleton(self, skel):
        """Remove short spurs (branches shorter than prune_length)."""
        pruned = skel.copy()
        for _ in range(self.prune_len):
            degree = self._get_degree_map(pruned)
            # Endpoints = degree 1
            tips = (degree == 1) & (pruned > 0)
            if not np.any(tips):
                break
            pruned[tips] = 0
        # Re-add the original skeleton where it overlaps with the pruned
        # This preserves the core structure but removes short branches
        # Actually, we need a smarter approach: only prune branches shorter
        # than the threshold
        return self._smart_prune(skel)

    def _smart_prune(self, skel):
        """Prune branches shorter than prune_length, keep long ones."""
        h, w = skel.shape
        degree = self._get_degree_map(skel)
        endpoints = set(map(tuple, np.argwhere(
            (degree == 1) & (skel > 0))))
        junctions = set(map(tuple, np.argwhere(
            (degree >= 3) & (skel > 0))))

        if not endpoints:
            return skel

        pruned = skel.copy()
        for ey, ex in endpoints:
            # Trace from this endpoint until we hit a junction or another endpoint
            path = [(ey, ex)]
            cy, cx = ey, ex
            py, px = -1, -1
            while True:
                nbrs = self._neighbours(pruned, cy, cx)
                nbrs = [(ny, nx) for ny, nx in nbrs if (ny, nx) != (py, px)]
                if not nbrs:
                    break
                ny, nx = nbrs[0]
                if (ny, nx) in junctions:
                    break
                if (ny, nx) in endpoints and (ny, nx) != (ey, ex):
                    # End-to-end branch - keep if long enough
                    path.append((ny, nx))
                    break
                path.append((ny, nx))
                py, px = cy, cx
                cy, cx = ny, nx

            # If short branch ending at a junction, prune it
            if len(path) < self.prune_len and path[-1] not in endpoints:
                for py_, px_ in path:
                    if (py_, px_) not in junctions:
                        pruned[py_, px_] = 0

        return pruned

    def _neighbours(self, skel, y, x):
        h, w = skel.shape
        out = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                    out.append((ny, nx))
        return out

    def trace_segments(self, skel: np.ndarray):
        """
        Trace the skeleton into straight line segments using Hough lines.

        Approach: Apply probabilistic Hough transform on the skeleton to
        detect straight line segments. This naturally handles corners and
        junctions without needing to trace branches.

        Returns list of segments: [(p1_px, p2_px, []), ...]
        where p1_px, p2_px are (x, y) float arrays.
        """
        if cv2.countNonZero(skel) == 0:
            return []

        # Step 1: Prune short spurs
        skel = self._prune_skeleton(skel)
        if cv2.countNonZero(skel) == 0:
            return []

        # Step 2: Dilate skeleton slightly for better Hough detection
        # (1-pixel skeleton can miss detections)
        skel_d = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=1)

        # Step 3: Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            skel_d,
            rho=1,
            theta=np.pi / 180,
            threshold=max(6, self.min_seg_len // 2),
            minLineLength=self.min_seg_len,
            maxLineGap=6,
        )

        if lines is None:
            # Fallback: fit a single line to all skeleton pixels
            return self._pca_fallback(skel)

        # Step 4: Collect and deduplicate segments
        raw_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            p1 = np.array([float(x1), float(y1)])
            p2 = np.array([float(x2), float(y2)])
            seg_len = np.linalg.norm(p2 - p1)
            if seg_len < self.min_seg_len:
                continue
            # Snap to orthogonal
            p1, p2 = self._snap_segment(p1, p2)
            raw_segments.append((p1, p2, seg_len))

        # Step 5: Merge overlapping collinear segments
        segments = self._merge_collinear(raw_segments)

        # Step 6: Deduplicate
        final = []
        seen = set()
        for p1, p2 in segments:
            key = (round(min(p1[0],p2[0]),0), round(min(p1[1],p2[1]),0),
                   round(max(p1[0],p2[0]),0), round(max(p1[1],p2[1]),0))
            if key not in seen:
                seen.add(key)
                final.append((p1, p2, []))

        logger.info("Skeleton: Hough found %d raw lines -> %d segments",
                    len(lines) if lines is not None else 0, len(final))
        return final

    def _pca_fallback(self, skel):
        """Fit a single line to all skeleton pixels using PCA."""
        pts = np.argwhere(skel > 0)
        if len(pts) < 2:
            return []
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        if cov.ndim < 2:
            return []
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_dir = eigvecs[:, -1]
        proj = centered @ main_dir
        i_min, i_max = np.argmin(proj), np.argmax(proj)
        p1 = np.array([float(pts[i_min][1]), float(pts[i_min][0])])
        p2 = np.array([float(pts[i_max][1]), float(pts[i_max][0])])
        p1, p2 = self._snap_segment(p1, p2)
        return [(p1, p2, [])]

    def _merge_collinear(self, raw_segments):
        """
        Merge overlapping collinear segments.

        Two segments are merged only if:
          1. They have the same orientation (horiz / vert / same diagonal angle)
          2. Their perpendicular distance is < perp_tol pixels
          3. They overlap or are within gap_tol pixels along the main axis
        """
        if not raw_segments:
            return []

        perp_tol = 3.0   # max perpendicular distance to merge (pixels)
        gap_tol = 6.0     # max gap along main axis to bridge

        # Classify orientation
        horiz, vert, diag = [], [], []
        for p1, p2, L in raw_segments:
            dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            angle = math.degrees(math.atan2(dy, max(dx, 0.01)))
            if angle < 20:
                horiz.append((p1, p2))
            elif angle > 70:
                vert.append((p1, p2))
            else:
                diag.append((p1, p2))

        result = []
        # For horizontal segments: perp axis = y, main axis = x
        result.extend(self._merge_axis_group(horiz, perp_axis=1, main_axis=0,
                                              perp_tol=perp_tol, gap_tol=gap_tol))
        # For vertical segments: perp axis = x, main axis = y
        result.extend(self._merge_axis_group(vert, perp_axis=0, main_axis=1,
                                              perp_tol=perp_tol, gap_tol=gap_tol))
        # Keep diagonals as-is (no merging)
        result.extend(diag)
        return result

    def _merge_axis_group(self, segs, perp_axis, main_axis, perp_tol, gap_tol):
        """Merge collinear segments that are close perpendicularly and overlap."""
        if not segs:
            return []

        # Sort by perpendicular position
        indexed = []
        for p1, p2 in segs:
            perp = (p1[perp_axis] + p2[perp_axis]) / 2.0
            lo = min(p1[main_axis], p2[main_axis])
            hi = max(p1[main_axis], p2[main_axis])
            indexed.append((perp, lo, hi, p1, p2))
        indexed.sort(key=lambda x: x[0])

        # Greedy merge: walk through sorted list, merge compatible segments
        merged = []
        used = [False] * len(indexed)

        for i in range(len(indexed)):
            if used[i]:
                continue
            perp_i, lo_i, hi_i, _, _ = indexed[i]
            cluster_perps = [perp_i]
            cluster_lo = lo_i
            cluster_hi = hi_i
            used[i] = True

            # Find all segments that can merge with this cluster
            changed = True
            while changed:
                changed = False
                for j in range(i + 1, len(indexed)):
                    if used[j]:
                        continue
                    perp_j, lo_j, hi_j, _, _ = indexed[j]
                    # Check perpendicular distance
                    if abs(perp_j - np.mean(cluster_perps)) > perp_tol:
                        continue
                    # Check overlap/proximity along main axis
                    if lo_j > cluster_hi + gap_tol or hi_j < cluster_lo - gap_tol:
                        continue
                    # Merge
                    cluster_perps.append(perp_j)
                    cluster_lo = min(cluster_lo, lo_j)
                    cluster_hi = max(cluster_hi, hi_j)
                    used[j] = True
                    changed = True

            # Create merged segment
            avg_perp = float(np.mean(cluster_perps))
            p1_new = np.zeros(2)
            p2_new = np.zeros(2)
            p1_new[perp_axis] = avg_perp
            p2_new[perp_axis] = avg_perp
            p1_new[main_axis] = cluster_lo
            p2_new[main_axis] = cluster_hi
            merged.append((p1_new, p2_new))

        return merged

    def _snap_segment(self, p1, p2):
        """Snap near-horizontal/vertical segments to exact axis alignment."""
        dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        tol = self.snap_deg
        p1_s, p2_s = p1.copy(), p2.copy()
        if angle < tol:
            mid_y = (p1[1] + p2[1]) / 2
            p1_s[1] = mid_y
            p2_s[1] = mid_y
        elif angle > 90 - tol:
            mid_x = (p1[0] + p2[0]) / 2
            p1_s[0] = mid_x
            p2_s[0] = mid_x
        return p1_s, p2_s


# ======================================================================
# 2.3  Thickness Measurement via Distance Transform
# ======================================================================

class ThicknessMeasurer:
    """
    Measures wall thickness and refines centerline position by finding
    the distance-transform ridge (true medial axis) perpendicular to
    each skeleton segment.
    """

    def __init__(self, min_thickness_px=3, max_thickness_px=80):
        self.min_t = min_thickness_px
        self.max_t = max_thickness_px

    def measure_and_refine(self, dist_map, p1_px, p2_px, n_samples=20):
        """
        Measure thickness AND refine the segment to sit on the true
        medial axis (peak of distance transform).

        Returns
        -------
        thickness_px : float
        p1_refined : np.ndarray  (x, y) refined endpoint 1
        p2_refined : np.ndarray  (x, y) refined endpoint 2
        """
        h, w = dist_map.shape
        seg_dir = p2_px - p1_px
        seg_len = np.linalg.norm(seg_dir)
        if seg_len < 1e-6:
            return 0.0, p1_px, p2_px

        seg_unit = seg_dir / seg_len
        # Perpendicular direction
        perp = np.array([-seg_unit[1], seg_unit[0]])

        # Sample along the segment
        ts = np.linspace(0, 1, n_samples)
        ridge_offsets = []
        ridge_dists = []

        for t in ts:
            pt = p1_px + t * seg_dir
            # Search perpendicular for the ridge (max dist_map value)
            best_offset = 0.0
            best_val = 0.0
            for offset in np.linspace(-15, 15, 31):
                sx = int(round(pt[0] + perp[0] * offset))
                sy = int(round(pt[1] + perp[1] * offset))
                if 0 <= sy < h and 0 <= sx < w:
                    val = dist_map[sy, sx]
                    if val > best_val:
                        best_val = val
                        best_offset = offset
            ridge_offsets.append(best_offset)
            ridge_dists.append(best_val)

        if not ridge_dists or max(ridge_dists) < 0.5:
            return 0.0, p1_px, p2_px

        # Median offset = how far to shift the line to the ridge
        median_offset = float(np.median(ridge_offsets))

        # Refine endpoints
        p1_ref = p1_px + perp * median_offset
        p2_ref = p2_px + perp * median_offset

        # Thickness = 2 * median distance at ridge
        thickness = 2.0 * float(np.median(ridge_dists))
        thickness = np.clip(thickness, self.min_t, self.max_t)

        return thickness, p1_ref, p2_ref

    def measure(self, dist_map, p1_px, p2_px, n_samples=20):
        """Legacy interface: just return thickness."""
        t, _, _ = self.measure_and_refine(dist_map, p1_px, p2_px, n_samples)
        return t


# ======================================================================
# 2.4  Wall Assembly
# ======================================================================

class WallAssembler:
    """
    Assembles segments into Wall objects:
      - Converts pixel segments to metre coordinates
      - Merges nearby endpoints into shared Points
      - Groups connected segments into Walls
      - Applies validation filters
    """

    def __init__(self, scale: float,
                 merge_point_tol_m: float = 0.05,
                 min_rect_length_m: float = 0.08,
                 max_wall_thickness_m: float = 0.50,
                 min_wall_thickness_m: float = 0.04):
        self.scale = scale
        self.merge_tol = merge_point_tol_m
        self.min_rect_len = min_rect_length_m
        self.max_thick = max_wall_thickness_m
        self.min_thick = min_wall_thickness_m

    def assemble(self, segments, thicknesses, bbox_offset):
        """
        Parameters
        ----------
        segments : list of (p1_px, p2_px, path) from skeleton tracer
        thicknesses : list of float (pixel thickness per segment)
        bbox_offset : (x_off, y_off) pixel offset of the component bbox

        Returns
        -------
        Wall object with Points and Rects in metre coordinates.
        """
        wall = Wall()
        ox, oy = bbox_offset
        all_pts: list[Point] = []

        # First pass: create candidate rects
        candidates = []
        for (p1_px, p2_px, _), thick_px in zip(segments, thicknesses):
            # Convert to global pixel coords, then to metres
            gp1 = np.array([p1_px[0] + ox, p1_px[1] + oy])
            gp2 = np.array([p2_px[0] + ox, p2_px[1] + oy])
            m1 = gp1 * self.scale
            m2 = gp2 * self.scale
            thick_m = thick_px * self.scale

            # Filter by thickness
            if thick_m < self.min_thick:
                continue
            if thick_m > self.max_thick:
                continue

            # Create/merge points
            pt1 = self._get_or_create(all_pts, float(m1[0]), float(m1[1]))
            pt2 = self._get_or_create(all_pts, float(m2[0]), float(m2[1]))

            if pt1.id == pt2.id:
                continue

            length_m = pt1.distance_to(pt2)
            if length_m < self.min_rect_len:
                continue

            dx = pt2.x - pt1.x
            dy = pt2.y - pt1.y
            angle = float(math.atan2(abs(dx), abs(dy)))

            candidates.append((pt1.id, pt2.id, round(thick_m, 4),
                              round(length_m, 4), round(angle, 4)))

        # Second pass: deduplicate — keep thickest rect per endpoint pair
        # Also filter short diagonal segments (corner artifacts)
        best_rects: dict[tuple, tuple] = {}
        for p1_id, p2_id, thick, length, angle in candidates:
            # Short diagonals (not aligned to H or V) are junction artifacts
            angle_deg = math.degrees(angle)
            is_diagonal = 15 < angle_deg < 75
            if is_diagonal and length < 0.15:
                continue

            key = (min(p1_id, p2_id), max(p1_id, p2_id))
            if key not in best_rects or thick > best_rects[key][2]:
                best_rects[key] = (p1_id, p2_id, thick, length, angle)

        for p1_id, p2_id, thick, length, angle in best_rects.values():
            r = Rect(
                p1_id=p1_id, p2_id=p2_id,
                thickness=thick,
                length=length,
                angle=angle,
            )
            wall.rects[r.id] = r

        for p in all_pts:
            if any(p.id in (r.p1_id, r.p2_id) for r in wall.rects.values()):
                wall.points[p.id] = p

        if not wall.rects:
            wall.rejected = True
            wall.rejection_reason = "no valid segments"

        return wall

    def _get_or_create(self, pts, x, y):
        for p in pts:
            if math.hypot(p.x - x, p.y - y) < self.merge_tol:
                return p
        p = Point(x=round(x, 4), y=round(y, 4))
        pts.append(p)
        return p


# ======================================================================
# 2.45  Wall-rect consolidation (de-fragmentation)
# ======================================================================

def _merge_axis_segments(segs, orient, lateral_tol_m, gap_tol_m):
    """
    Merge a set of axis-aligned (H or V) wall segments that lie on the same line
    and overlap or nearly touch, into continuous spans.

    Each seg is a dict with x1,y1,x2,y2,t,L.  For H, the run is along x at roughly
    constant y (lateral); for V the run is along y at constant x.  Segments are
    grouped when their lateral coords agree within `lateral_tol_m` and their main
    intervals overlap or sit within `gap_tol_m` of each other.  Returns a list of
    (x1, y1, x2, y2, thickness) merged spans (length-weighted thickness).
    """
    iv = []  # [lateral, lo, hi, t, L]
    for s in segs:
        if orient == "H":
            lat = (s["y1"] + s["y2"]) / 2.0
            lo, hi = min(s["x1"], s["x2"]), max(s["x1"], s["x2"])
        else:
            lat = (s["x1"] + s["x2"]) / 2.0
            lo, hi = min(s["y1"], s["y2"]), max(s["y1"], s["y2"])
        iv.append([lat, lo, hi, s["t"], hi - lo])

    order = sorted(range(len(iv)), key=lambda k: (iv[k][0], iv[k][1]))
    used = [False] * len(iv)
    out = []
    for oi in order:
        if used[oi]:
            continue
        group = [oi]
        used[oi] = True
        changed = True
        while changed:
            changed = False
            lat_w = sum(iv[k][0] * iv[k][4] for k in group)
            w_tot = sum(iv[k][4] for k in group) or 1e-9
            g_lat = lat_w / w_tot
            g_lo = min(iv[k][1] for k in group)
            g_hi = max(iv[k][2] for k in group)
            for k in order:
                if used[k]:
                    continue
                if (abs(iv[k][0] - g_lat) <= lateral_tol_m and
                        iv[k][1] <= g_hi + gap_tol_m and
                        iv[k][2] >= g_lo - gap_tol_m):
                    group.append(k)
                    used[k] = True
                    changed = True
        w_tot = sum(iv[k][4] for k in group) or 1e-9
        lat = sum(iv[k][0] * iv[k][4] for k in group) / w_tot
        lo = min(iv[k][1] for k in group)
        hi = max(iv[k][2] for k in group)
        thick = sum(iv[k][3] * iv[k][4] for k in group) / w_tot
        if orient == "H":
            out.append((lo, lat, hi, lat, thick))
        else:
            out.append((lat, lo, lat, hi, thick))
    return out


def auto_calibrate_floor(data: dict, *, target_thickness_m: float = 0.20,
                         deadband: float = 0.50,
                         clamp: tuple = (0.2, 6.0)):
    """
    Auto-calibrate the drawing scale from the length-weighted MEDIAN wall
    thickness (residential walls are ~0.20 m).

    The pipeline assumes 10 mm/px, but real plan images come at arbitrary
    scales.  A wrong scale silently destroys opening handling: a true 0.9 m
    door shrinks below the 0.25 m consolidation gap tolerance (bridged!) and
    below the 0.40 m opening-detection band (never classified) -- and every
    downstream length (piers, spans, grid) is distorted by the same factor.

    A +-deadband no-op keeps correctly-scaled inputs (StructGAN semantic
    images, synthetic renders) untouched.  Returns (data, factor).

    Deadband width: real residential wall thickness is dispersed far more
    widely than the target implies -- measured over 130 annotated CubiCasa5K
    plans the length-weighted median is 0.21 m with a 5th-95th percentile
    range of 0.14-0.30 m, i.e. factors 0.67-1.43 (|k-1| up to 0.43) arise from
    genuine thin- or thick-walled stock rather than from a wrong scale.  The
    old +-0.20 deadband was narrower than that natural spread, so 44% of
    correctly-scaled drawings were rescaled spuriously.  +-0.50 brackets the
    dispersion with margin and lifts protection of correctly-scaled inputs to
    ~98%, while leaving recovery of genuinely mis-scaled drawings unchanged
    (a 2x-4x scale error gives k >= 2, far outside the band).  Measured to be
    flat for 0.45 <= deadband <= 0.60 and to degrade at 0.70.
    """
    ths, wts = [], []
    for w in data.get("walls", []):
        if w.get("rejected"):
            continue
        for r in w.get("rects", []):
            ths.append(float(r.get("thickness_m", 0.0)))
            wts.append(max(float(r.get("length_m", 0.0)), 1e-3))
    if not ths:
        return data, 1.0
    order = np.argsort(ths)
    ths_s = np.asarray(ths)[order]
    cum = np.cumsum(np.asarray(wts)[order])
    t_med = float(ths_s[int(np.searchsorted(cum, cum[-1] / 2.0))])
    if t_med <= 1e-6:
        return data, 1.0
    k = target_thickness_m / t_med
    k = float(min(max(k, clamp[0]), clamp[1]))
    if abs(k - 1.0) <= deadband:
        return data, 1.0

    for w in data.get("walls", []):
        for pt in w.get("points", []):
            pt["x"] = round(pt["x"] * k, 4)
            pt["y"] = round(pt["y"] * k, 4)
        for r in w.get("rects", []):
            r["thickness_m"] = round(r["thickness_m"] * k, 4)
            r["length_m"] = round(r["length_m"] * k, 4)
    data["scale_m_per_px"] = data.get("scale_m_per_px", 0.01) * k
    data["scale_autocalibrated_factor"] = round(k, 4)
    logger.info("Auto-scale calibration: x%.2f (median wall thickness "
                "%.3f m -> %.2f m target)", k, t_med, target_thickness_m)
    return data, k


def consolidate_wall_rects(floor: dict, *,
                           lateral_tol_m: float = 0.10,
                           gap_tol_m: float = 0.25,
                           min_seg_len_m: float = 0.10,
                           diag_min_len_m: float = 0.80,
                           point_merge_m: float = 0.10) -> dict:
    """
    De-fragment the Stage 2 wall set into continuous walls.

    Stage 2's skeleton+Hough tracing over-segments: a physical wall ends up as
    many short rects (median ~0.23 m), often *overlapping duplicate* detections
    of the same line (slightly different lateral offset / thickness) plus 45-deg
    corner artifacts.  Endpoint chaining cannot fix this because the fragments
    rarely share endpoints.  Instead we re-merge at the floor level by axis:

      1. classify every rect as H, V, or diagonal (drop short diagonal artifacts,
         keep long diagonals as real angled walls);
      2. for H and V separately, cluster rects that lie on the same line (lateral
         coord within `lateral_tol_m`) and overlap or sit within `gap_tol_m`,
         merging each cluster into one continuous span (length-weighted thickness);
      3. rebuild points (coincident endpoints merged within `point_merge_m`),
         rects (fresh ids), and walls (grouped by shared-point connectivity).

    `gap_tol_m` (0.25 m) stays below the opening-detection band (>= 0.40 m) so real
    door/window gaps are preserved.  Operates on / returns the Stage 2 floor JSON
    dict; the input is not mutated.
    """
    import copy as _copy
    floor = _copy.deepcopy(floor)

    # 1. collect + classify
    segs = []
    for w in floor.get("walls", []):
        if w.get("rejected"):
            continue
        pts = {p["id"]: (p["x"], p["y"]) for p in w.get("points", [])}
        for r in w.get("rects", []):
            p1 = pts.get(r["p1_id"])
            p2 = pts.get(r["p2_id"])
            if p1 is None or p2 is None:
                continue
            segs.append(dict(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1],
                             t=r["thickness_m"], L=r["length_m"]))

    diag_ratio = math.tan(math.radians(25.0))
    H, V, merged_spans = [], [], []
    for s in segs:
        dx, dy = abs(s["x2"] - s["x1"]), abs(s["y2"] - s["y1"])
        if max(dx, dy) < 1e-9:
            continue
        if min(dx, dy) / max(dx, dy) > diag_ratio:          # diagonal
            if math.hypot(dx, dy) >= diag_min_len_m:        # keep long angled wall
                merged_spans.append((s["x1"], s["y1"], s["x2"], s["y2"], s["t"]))
            continue
        (H if dx >= dy else V).append(s)

    merged_spans += _merge_axis_segments(H, "H", lateral_tol_m, gap_tol_m)
    merged_spans += _merge_axis_segments(V, "V", lateral_tol_m, gap_tol_m)
    merged_spans = [m for m in merged_spans
                    if math.hypot(m[2] - m[0], m[3] - m[1]) >= min_seg_len_m]

    # 2. rebuild points (merge coincident endpoints)
    node_pts = []   # list of [x, y]
    node_ids = []

    def _get_node(x, y):
        for i, q in enumerate(node_pts):
            if math.hypot(q[0] - x, q[1] - y) < point_merge_m:
                return node_ids[i]
        nid = uuid.uuid4().hex[:8]
        node_pts.append([x, y])
        node_ids.append(nid)
        return nid

    rect_recs = []  # (id, p1_id, p2_id, thickness, length, angle)
    for (x1, y1, x2, y2, t) in merged_spans:
        a = _get_node(x1, y1)
        b = _get_node(x2, y2)
        if a == b:
            continue
        L = math.hypot(x2 - x1, y2 - y1)
        ang = math.atan2(abs(x2 - x1), abs(y2 - y1))
        rect_recs.append((uuid.uuid4().hex[:8], a, b,
                          round(float(t), 4), round(L, 4), round(ang, 4)))

    # 3. group rects into walls by shared-point connectivity (union-find)
    parent = {nid: nid for nid in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for (_rid, a, b, *_rest) in rect_recs:
        union(a, b)

    id_to_xy = {nid: node_pts[i] for i, nid in enumerate(node_ids)}
    walls_by_root: dict[str, dict] = {}
    for (rid, a, b, t, L, ang) in rect_recs:
        root = find(a)
        w = walls_by_root.setdefault(root, {"point_ids": set(), "rects": []})
        w["point_ids"].update((a, b))
        w["rects"].append({"id": rid, "p1_id": a, "p2_id": b,
                           "thickness_m": t, "length_m": L, "angle_rad": ang})

    new_walls = []
    for w in walls_by_root.values():
        new_walls.append({
            "wall_id": uuid.uuid4().hex[:8],
            "rejected": False,
            "rejection_reason": "",
            "points": [{"id": nid,
                        "x": round(float(id_to_xy[nid][0]), 4),
                        "y": round(float(id_to_xy[nid][1]), 4)}
                       for nid in w["point_ids"]],
            "rects": w["rects"],
        })

    floor["walls"] = new_walls
    return floor


# ======================================================================
# 2.5  Scale Calibration
# ======================================================================

class ScaleCalibrator:
    @staticmethod
    def from_known_factor(meters_per_pixel: float) -> float:
        return meters_per_pixel

    @staticmethod
    def from_reference_length(p1_px, p2_px, real_length_m):
        pixel_dist = math.hypot(p2_px[0]-p1_px[0], p2_px[1]-p1_px[1])
        if pixel_dist < 1e-9:
            raise ValueError("Reference points are coincident")
        return real_length_m / pixel_dist


# ======================================================================
# Main Pipeline
# ======================================================================

class VectorizationPipeline:
    """
    End-to-end Stage 2: binary wall mask -> Points + Rectangles (JSON).

    v2 approach:
      1. Clean mask & extract connected components
      2. For each component:
         a. Compute distance transform (-> thickness)
         b. Extract skeleton
         c. Trace skeleton into line segments
         d. Measure thickness per segment
         e. Assemble into Wall with Points + Rects
    """

    def __init__(self, scale=0.01, floor_height=2.80,
                 close_kernel_size=5, open_kernel_size=3,
                 min_component_area=50,
                 min_segment_length_px=8, rdp_epsilon=2.0,
                 angle_snap_deg=5.0,
                 min_wall_thickness_m=0.04, max_wall_thickness_m=0.50,
                 min_rect_length_m=0.08, merge_point_tol_m=0.05):
        self.scale = scale
        self.floor_height = floor_height
        self.cleaner = MaskCleaner(close_kernel_size, open_kernel_size,
                                    min_component_area)
        self.tracer = SkeletonTracer(min_segment_length_px, rdp_epsilon,
                                      angle_snap_deg, prune_length_px=6)
        self.measurer = ThicknessMeasurer()
        self.assembler_kwargs = dict(
            scale=scale,
            merge_point_tol_m=merge_point_tol_m,
            min_rect_length_m=min_rect_length_m,
            max_wall_thickness_m=max_wall_thickness_m,
            min_wall_thickness_m=min_wall_thickness_m,
        )

    def run(self, mask, floor_id="floor_01"):
        floor = Floor(floor_id=floor_id, scale=self.scale,
                       height=self.floor_height)

        logger.info("=" * 60)
        logger.info("Stage 2.1: Mask cleanup")
        cleaned = self.cleaner.clean(mask)

        # Global distance transform (for thickness measurement)
        dist_map = distance_transform_edt(cleaned > 0).astype(np.float32)

        logger.info("Stage 2.2: Extract connected components")
        components = self.cleaner.extract_components(cleaned)

        logger.info("Stage 2.3-2.4: Skeleton tracing & wall assembly")
        assembler = WallAssembler(**self.assembler_kwargs)
        accepted, rejected = 0, 0

        for ci, (sub_mask, bbox, area) in enumerate(components):
            bx, by, bw, bh = bbox

            # Skeleton of this component
            skel = self.tracer.skeletonize(sub_mask)

            # Trace into segments (in sub-mask local coords)
            segments = self.tracer.trace_segments(skel)
            if not segments:
                # Fallback: treat as single rectangle via bounding box
                segments = self._bbox_fallback(sub_mask, bw, bh)
                if not segments:
                    rejected += 1
                    wall = Wall(rejected=True,
                                rejection_reason="no skeleton segments")
                    floor.walls.append(wall)
                    continue

            # Measure thickness for each segment using the GLOBAL dist map
            # Also refine the centerline position to the true medial axis
            refined_segments = []
            thicknesses = []
            for p1_px, p2_px, path in segments:
                # Convert local coords to global for dist_map sampling
                gp1 = np.array([p1_px[0] + bx, p1_px[1] + by])
                gp2 = np.array([p2_px[0] + bx, p2_px[1] + by])
                t, gp1_ref, gp2_ref = self.measurer.measure_and_refine(
                    dist_map, gp1, gp2)
                # Convert refined global back to local for assembler
                p1_ref = gp1_ref - np.array([bx, by])
                p2_ref = gp2_ref - np.array([bx, by])
                refined_segments.append((p1_ref, p2_ref, path))
                thicknesses.append(t)

            # Assemble wall
            wall = assembler.assemble(refined_segments, thicknesses, (bx, by))
            if wall.rejected:
                rejected += 1
            else:
                accepted += 1
            floor.walls.append(wall)

        logger.info("Done: %d accepted, %d rejected (%.0f%%)",
                    accepted, rejected,
                    100 * rejected / max(1, accepted + rejected))
        tp = sum(len(w.points) for w in floor.walls if not w.rejected)
        tr = sum(len(w.rects) for w in floor.walls if not w.rejected)
        logger.info("Totals: %d points, %d rects in %d valid walls",
                    tp, tr, accepted)
        return floor

    def _bbox_fallback(self, sub_mask, w, h):
        """Fallback: fit a single rectangle to a small component."""
        pts = np.argwhere(sub_mask > 0)  # (y, x)
        if len(pts) < 4:
            return []
        pts_xy = pts[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(pts_xy)
        (cx, cy), (rw, rh), angle = rect
        if rw < 3 or rh < 3:
            return []
        if rw >= rh:
            length_dir = np.array([math.cos(math.radians(angle)),
                                   math.sin(math.radians(angle))])
        else:
            length_dir = np.array([math.cos(math.radians(angle + 90)),
                                   math.sin(math.radians(angle + 90))])
        half_len = max(rw, rh) / 2
        center = np.array([cx, cy])
        p1 = center - length_dir * half_len
        p2 = center + length_dir * half_len
        return [(p1, p2, [(int(p1[1]), int(p1[0])),
                          (int(p2[1]), int(p2[0]))])]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, floor, path):
        data = {
            "floor_id": floor.floor_id,
            "scale_m_per_px": floor.scale,
            "floor_height_m": floor.height,
            "offset": {"x": floor.offset_x, "y": floor.offset_y},
            "walls": [],
        }
        for wall in floor.walls:
            data["walls"].append({
                "wall_id": wall.id,
                "rejected": wall.rejected,
                "rejection_reason": wall.rejection_reason,
                "points": [{"id": p.id, "x": round(p.x, 4),
                            "y": round(p.y, 4)}
                           for p in wall.points.values()],
                "rects": [{"id": r.id, "p1_id": r.p1_id, "p2_id": r.p2_id,
                           "thickness_m": r.thickness,
                           "length_m": r.length,
                           "angle_rad": r.angle}
                          for r in wall.rects.values()],
            })

        # Auto-scale calibration MUST run before consolidation so the metric
        # gap tolerances act on true dimensions (else real doors get bridged).
        data, _k = auto_calibrate_floor(data)

        # De-fragment: merge collinear same-wall rects into continuous walls so
        # downstream stages (graph, shear-wall logic, FEM) see real walls.
        n_before = sum(len(w["rects"]) for w in data["walls"]
                       if not w["rejected"])
        data = consolidate_wall_rects(data)
        n_after = sum(len(w["rects"]) for w in data["walls"]
                      if not w["rejected"])
        logger.info("Rect consolidation: %d -> %d rects", n_before, n_after)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported to %s", path)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, mask, floor, output_path=None, show_skeleton=False):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 \
              else mask.copy()
        vis = (vis * 0.3).astype(np.uint8)
        colors = [(0, 255, 0), (255, 100, 0), (0, 200, 255),
                  (200, 0, 255), (255, 200, 0), (100, 255, 200),
                  (255, 150, 150)]

        for wi, wall in enumerate(floor.walls):
            color = (0, 0, 255) if wall.rejected else colors[wi % len(colors)]
            for rect in wall.rects.values():
                p1 = wall.points.get(rect.p1_id)
                p2 = wall.points.get(rect.p2_id)
                if not p1 or not p2:
                    continue
                s = floor.scale
                cx1, cy1 = p1.x / s, p1.y / s
                cx2, cy2 = p2.x / s, p2.y / s
                # Draw centerline
                cv2.line(vis, (int(cx1), int(cy1)), (int(cx2), int(cy2)),
                         color, 2)
                # Draw rectangle outline
                dx, dy = cx2 - cx1, cy2 - cy1
                lp = math.hypot(dx, dy)
                if lp < 1e-6:
                    continue
                px = -dy / lp * (rect.thickness / s / 2)
                py = dx / lp * (rect.thickness / s / 2)
                corners = np.array([
                    [cx1+px, cy1+py], [cx2+px, cy2+py],
                    [cx2-px, cy2-py], [cx1-px, cy1-py]
                ], dtype=np.int32)
                cv2.polylines(vis, [corners], True, color, 1)

            # Draw points
            for pt in wall.points.values():
                px_x = int(pt.x / floor.scale)
                px_y = int(pt.y / floor.scale)
                cv2.circle(vis, (px_x, px_y), 4, (255, 255, 255), -1)
                cv2.circle(vis, (px_x, px_y), 4, color, 1)

        if output_path:
            cv2.imwrite(output_path, vis)
            logger.info("Visualization: %s", output_path)
        return vis


# ======================================================================
# Demo
# ======================================================================

def create_demo_mask(w=512, h=512):
    """Synthetic mask with L, T, straight, and room-enclosure walls."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (70, 250), 255, -1)    # L vertical
    cv2.rectangle(mask, (50, 230), (200, 250), 255, -1)   # L horizontal
    cv2.rectangle(mask, (50, 400), (450, 415), 255, -1)   # straight
    cv2.rectangle(mask, (350, 80), (370, 300), 255, -1)   # T stem
    cv2.rectangle(mask, (300, 80), (420, 100), 255, -1)   # T top
    cv2.rectangle(mask, (150, 120), (280, 135), 255, -1)  # room top
    cv2.rectangle(mask, (150, 120), (165, 250), 255, -1)  # room left
    cv2.rectangle(mask, (265, 120), (280, 250), 255, -1)  # room right
    cv2.rectangle(mask, (150, 235), (280, 250), 255, -1)  # room bottom
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Wall Vectorization (skeleton-based)")
    parser.add_argument("--input", "-i", type=str, default="C:\\Dev\\Plan_2_FEM_2026\\output_mask.png")
    parser.add_argument("--scale", "-s", type=float, default=0.01)
    parser.add_argument("--output-json", "-o", type=str,
                        default="floor_output.json")
    parser.add_argument("--output-vis", "-v", type=str,
                        default="floor_visualization.png")
    parser.add_argument("--floor-height", type=float, default=2.80)
    args = parser.parse_args()

    if args.input:
        mask = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read: {args.input}")
    else:
        logger.info("Using synthetic demo mask")
        mask = create_demo_mask()

    pipe = VectorizationPipeline(
        scale=args.scale,
        floor_height=args.floor_height,
    )

    floor = pipe.run(mask)
    pipe.export_json(floor, args.output_json)
    pipe.visualize(mask, floor, args.output_vis)

    valid = [w for w in floor.walls if not w.rejected]
    rejected_w = [w for w in floor.walls if w.rejected]
    print(f"\n{'='*60}")
    print("VECTORIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Valid walls:    {len(valid)}")
    print(f"  Rejected walls: {len(rejected_w)}")
    print(f"  Total points:   {sum(len(w.points) for w in valid)}")
    print(f"  Total rects:    {sum(len(w.rects) for w in valid)}")
    for w in valid:
        print(f"\n  Wall {w.id}:")
        for r in w.rects.values():
            p1, p2 = w.points[r.p1_id], w.points[r.p2_id]
            print(f"    Rect: ({p1.x:.3f},{p1.y:.3f})->({p2.x:.3f},{p2.y:.3f})"
                  f"  L={r.length:.3f}m  t={r.thickness:.3f}m  "
                  f"a={math.degrees(r.angle):.1f}deg")
    for w in rejected_w:
        print(f"  [REJECTED] Wall {w.id}: {w.rejection_reason}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()