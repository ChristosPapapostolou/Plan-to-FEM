"""
Stage 2 - Vectorization Pipeline
=================================
Converts a binary wall mask (output of Stage 1 segmentation) into a structured
collection of Wall objects, each decomposed into rectangles connected by points.

Data model follows Pizarro & Massone (2021):
  - Point: unique (x, y) vertex where rectangles connect
  - Rect:  defined by two Point IDs + thickness, representing a wall segment
  - Wall:  a collection of connected Points and Rects (possibly L/T/U-shaped)

Pipeline sub-steps:
  2.1  Mask cleanup & contour extraction
  2.2  Polygon simplification & orthogonal snapping
  2.3  Rectangle discretization (convex hull, vertex projection, planar graph)
  2.4  Scale calibration & coordinate assignment

References:
  [1] Pizarro & Massone, Eng. Structures 241 (2021) 112377
  [2] Zhao et al., Adv. Eng. Informatics 55 (2023) 101886

Requirements:
  numpy, opencv-python, scipy  (no Shapely dependency)
"""

from __future__ import annotations

import uuid
import math
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry Helpers (pure numpy/cv2, no Shapely)
# ---------------------------------------------------------------------------

def polygon_area(pts: np.ndarray) -> float:
    """Signed area of a 2D polygon via the shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_signed_area(pts: np.ndarray) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_polygon_ccw(pts: np.ndarray) -> bool:
    return polygon_signed_area(pts) > 0


def ensure_ccw(pts: np.ndarray) -> np.ndarray:
    if not is_polygon_ccw(pts):
        return pts[::-1].copy()
    return pts


def minimum_area_bounding_rect(pts: np.ndarray):
    """
    Compute the oriented minimum-area bounding rectangle via cv2.minAreaRect.

    Returns: (box_corners(4,2), length, width, angle_rad)
    """
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


def line_segment_intersection(p1, p2, p3, p4):
    """Intersection of segments (p1,p2) and (p3,p4). Returns point or None."""
    p1, p2, p3, p4 = map(np.asarray, [p1, p2, p3, p4])
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * d1
    return None


def ray_polygon_intersection(origin, direction, polygon):
    """Cast a ray and find nearest polygon boundary intersection."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    n = len(polygon)
    extent = 1e4
    ray_end = origin + direction * extent
    best_pt, best_dist = None, float("inf")
    for i in range(n):
        ipt = line_segment_intersection(origin, ray_end, polygon[i], polygon[(i+1) % n])
        if ipt is not None:
            dist = np.linalg.norm(ipt - origin)
            if dist > 1e-3 and dist < best_dist:
                best_dist = dist
                best_pt = ipt
    return best_pt


# ---------------------------------------------------------------------------
# Data Classes - Pizarro & Massone data model
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """A unique vertex where wall rectangles connect."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    x: float = 0.0
    y: float = 0.0

    @property
    def xy(self):
        return (self.x, self.y)

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Rect:
    """A rectangular wall segment defined by two endpoint Points + thickness."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    p1_id: str = ""
    p2_id: str = ""
    thickness: float = 0.0
    length: float = 0.0
    angle: float = 0.0  # w.r.t. y-axis, [0, pi]


@dataclass
class Wall:
    """A complex-cross-section wall decomposed into Points and Rects."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    points: dict = field(default_factory=dict)
    rects: dict = field(default_factory=dict)
    contour_px: object = None
    polygon_px: object = None
    rejected: bool = False
    rejection_reason: str = ""


@dataclass
class Floor:
    """Container for a single floor's vectorization results."""
    floor_id: str = "floor_01"
    walls: list = field(default_factory=list)
    scale: float = 1.0
    height: float = 2.80
    offset_x: float = 0.0
    offset_y: float = 0.0


# ---------------------------------------------------------------------------
# 2.1  Mask Cleanup & Contour Extraction
# ---------------------------------------------------------------------------

class MaskCleaner:

    def __init__(self, close_kernel_size=5, open_kernel_size=3, min_component_area=50):
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_size, open_kernel_size))
        self.min_area = min_component_area

    def clean(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.open_kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        removed = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                cleaned[labels == i] = 0
                removed += 1
        logger.info("Mask cleanup: %d components retained, %d removed", num_labels - 1 - removed, removed)
        return cleaned

    def extract_contours(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info("Extracted %d contours", len(contours))
        return list(contours)


# ---------------------------------------------------------------------------
# 2.2  Polygon Simplification & Orthogonal Snapping
# ---------------------------------------------------------------------------

class PolygonSimplifier:

    def __init__(self, dp_epsilon=3.0, angle_tolerance_deg=10.0, min_edge_length_px=5.0):
        self.dp_epsilon = dp_epsilon
        self.angle_tolerance = angle_tolerance_deg
        self.min_edge_length = min_edge_length_px

    def simplify(self, contour: np.ndarray) -> np.ndarray:
        approx = cv2.approxPolyDP(contour, self.dp_epsilon, closed=True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        pts = self._remove_short_edges(pts)
        pts = self._snap_to_orthogonal(pts)
        pts = self._remove_collinear(pts)
        return pts

    def _remove_short_edges(self, pts):
        if len(pts) < 4:
            return pts
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) >= self.min_edge_length:
                keep.append(i)
        if len(keep) > 2 and np.linalg.norm(pts[keep[-1]] - pts[keep[0]]) < self.min_edge_length:
            keep.pop()
        return pts[keep]

    def _snap_to_orthogonal(self, pts):
        if len(pts) < 3:
            return pts
        n = len(pts)
        snapped = pts.copy()
        tol_rad = np.radians(self.angle_tolerance)
        for _ in range(3):
            new_pts = snapped.copy()
            for i in range(n):
                a, b, c = snapped[(i-1) % n], snapped[i], snapped[(i+1) % n]
                v1, v2 = a - b, c - b
                len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if len1 < 1e-9 or len2 < 1e-9:
                    continue
                cos_a = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
                angle = np.arccos(cos_a)
                if abs(angle - np.pi/2) < tol_rad:
                    v1u = v1 / len1
                    perp = np.array([-v1u[1], v1u[0]])
                    new_pts[(i+1) % n] = b + perp * np.dot(v2, perp)
            snapped = new_pts
        return snapped

    def _remove_collinear(self, pts, tol_deg=5.0):
        if len(pts) < 4:
            return pts
        n = len(pts)
        keep = []
        for i in range(n):
            a, b, c = pts[(i-1) % n], pts[i], pts[(i+1) % n]
            v1, v2 = b - a, c - b
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 < 1e-9 or l2 < 1e-9:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
            if abs(np.degrees(np.arccos(cos_a)) - 180.0) > tol_deg:
                keep.append(i)
        return pts[keep] if len(keep) >= 3 else pts

    def validate_polygon(self, pts):
        if len(pts) < 4:
            return False, "fewer than 4 vertices"
        n = len(pts)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                ipt = line_segment_intersection(pts[i], pts[(i+1) % n], pts[j], pts[(j+1) % n])
                if ipt is not None:
                    return False, f"self-intersection edges {i},{j}"
        tol = self.angle_tolerance
        for i in range(n):
            a, b, c = pts[(i-1)%n], pts[i], pts[(i+1)%n]
            v1, v2 = a - b, c - b
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 < 1e-9 or l2 < 1e-9:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1, 1)
            angle_deg = np.degrees(np.arccos(cos_a))
            rem = angle_deg % 90
            if min(rem, 90 - rem) > tol:
                return False, f"non-orthogonal {angle_deg:.1f} deg at vertex {i}"
        return True, ""


# ---------------------------------------------------------------------------
# 2.3  Rectangle Discretization
# ---------------------------------------------------------------------------

class RectangleDiscretizer:

    def __init__(self, max_thickness_m=0.50, min_rect_length_m=0.10, merge_point_tolerance_m=0.05):
        self.max_thickness = max_thickness_m
        self.min_rect_length = min_rect_length_m
        self.merge_tol = merge_point_tolerance_m

    def discretize(self, polygon_px: np.ndarray, scale: float) -> Wall:
        wall = Wall()
        wall.polygon_px = polygon_px
        poly_m = polygon_px * scale
        area = polygon_area(poly_m)
        if area < 1e-6:
            wall.rejected = True
            wall.rejection_reason = "near-zero area"
            return wall
        poly_m = ensure_ccw(poly_m)
        _, _, _, dominant_angle = minimum_area_bounding_rect(poly_m)
        rects_data = self._recursive_split(poly_m, dominant_angle, depth=0)
        if not rects_data:
            rects_data = [self._ombr_to_rect_data(poly_m)]
        self._build_wall(wall, rects_data)
        for rect in wall.rects.values():
            if rect.thickness > self.max_thickness:
                wall.rejected = True
                wall.rejection_reason = f"thickness {rect.thickness:.3f}m > {self.max_thickness}m"
                break
        return wall

    def _recursive_split(self, poly_m, dominant_angle, depth):
        if depth > 15:
            return [self._ombr_to_rect_data(poly_m)]
        area = polygon_area(poly_m)
        if area < 1e-6 or len(poly_m) < 4:
            return [self._ombr_to_rect_data(poly_m)] if area > 1e-6 else []

        _, ombr_len, ombr_wid, _ = minimum_area_bounding_rect(poly_m)
        ombr_area = ombr_len * ombr_wid
        if ombr_area > 1e-9 and area / ombr_area > 0.92:
            return [self._ombr_to_rect_data(poly_m)]

        concave = self._find_concave_vertices(poly_m)
        if not concave:
            return [self._ombr_to_rect_data(poly_m)]

        best_split, best_score = None, -1
        for ci in concave:
            parts = self._try_split(poly_m, ci, dominant_angle)
            if parts and len(parts) == 2:
                score = 0
                for p in parts:
                    pa = polygon_area(p)
                    _, pl, pw, _ = minimum_area_bounding_rect(p)
                    oa = pl * pw
                    score += pa / oa if oa > 1e-9 else 0
                if score > best_score:
                    best_score = score
                    best_split = parts

        if best_split is None:
            return [self._ombr_to_rect_data(poly_m)]

        result = []
        for part in best_split:
            result.extend(self._recursive_split(part, dominant_angle, depth + 1))
        return result

    def _find_concave_vertices(self, pts):
        n = len(pts)
        ccw = is_polygon_ccw(pts)
        concave = []
        for i in range(n):
            a, b, c = pts[(i-1)%n], pts[i], pts[(i+1)%n]
            cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if (ccw and cross < -1e-9) or (not ccw and cross > 1e-9):
                concave.append(i)
        return concave

    def _try_split(self, poly, vertex_idx, dominant_angle):
        n = len(poly)
        v = poly[vertex_idx]
        prev_v, next_v = poly[(vertex_idx-1)%n], poly[(vertex_idx+1)%n]

        # Try adjacent-edge perpendiculars, then dominant directions
        candidates = []
        for ref in [prev_v, next_v]:
            ed = ref - v
            el = np.linalg.norm(ed)
            if el < 1e-9:
                continue
            eu = ed / el
            candidates.append(np.array([-eu[1], eu[0]]))
            candidates.append(np.array([eu[1], -eu[0]]))

        for off in [0, np.pi/2]:
            a = dominant_angle + off
            candidates.append(np.array([math.cos(a), math.sin(a)]))
            candidates.append(np.array([-math.cos(a), -math.sin(a)]))

        for direction in candidates:
            hit = ray_polygon_intersection(v, direction, poly)
            if hit is None:
                continue
            parts = self._split_polygon(poly, v, hit)
            if parts and len(parts) == 2:
                return parts
        return None

    def _split_polygon(self, poly, p_start, p_end):
        n = len(poly)
        se = self._find_edge(poly, p_start)
        ee = self._find_edge(poly, p_end)
        if se is None or ee is None or se == ee:
            return None

        # Build vertex list with split points inserted
        verts = list(poly)
        inserts = []
        inserts.append((se, p_start))
        inserts.append((ee, p_end))
        inserts.sort(key=lambda x: x[0], reverse=True)

        for edge_idx, pt in inserts:
            verts.insert(edge_idx + 1, pt.copy())

        # Recalculate indices after insertions
        idx_s, idx_e = None, None
        for i, vt in enumerate(verts):
            if np.linalg.norm(np.asarray(vt) - p_start) < 1e-6:
                idx_s = i
            if np.linalg.norm(np.asarray(vt) - p_end) < 1e-6:
                idx_e = i

        if idx_s is None or idx_e is None:
            return None

        m = len(verts)
        # Walk idx_s -> idx_e
        poly1 = []
        i = idx_s
        for _ in range(m + 1):
            poly1.append(np.array(verts[i % m]))
            if i % m == idx_e % m:
                break
            i += 1
        # Walk idx_e -> idx_s
        poly2 = []
        i = idx_e
        for _ in range(m + 1):
            poly2.append(np.array(verts[i % m]))
            if i % m == idx_s % m:
                break
            i += 1

        poly1, poly2 = np.array(poly1), np.array(poly2)
        a1, a2 = polygon_area(poly1), polygon_area(poly2)
        if a1 < 1e-6 or a2 < 1e-6:
            return None
        return [ensure_ccw(poly1), ensure_ccw(poly2)]

    def _find_edge(self, poly, pt, tol=0.02):
        n = len(poly)
        best_idx, best_dist = None, float("inf")
        pt = np.asarray(pt)
        for i in range(n):
            a, b = np.asarray(poly[i]), np.asarray(poly[(i+1)%n])
            ab = b - a
            ab_len = np.linalg.norm(ab)
            if ab_len < 1e-12:
                continue
            t = np.clip(np.dot(pt - a, ab) / (ab_len**2), 0, 1)
            proj = a + t * ab
            dist = np.linalg.norm(pt - proj)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx if best_dist < tol else None

    def _ombr_to_rect_data(self, poly_m):
        box_pts, length, width, angle = minimum_area_bounding_rect(poly_m)
        edges = [np.linalg.norm(box_pts[(i+1)%4] - box_pts[i]) for i in range(4)]
        long_idx = 0 if edges[0] >= edges[1] else 1
        short_idx = 1 - long_idx
        # Centerline connects midpoints of the two SHORT edges
        # Edge i goes from box_pts[i] to box_pts[(i+1)%4]
        mid_s = (box_pts[short_idx] + box_pts[(short_idx+1)%4]) / 2
        mid_e = (box_pts[(short_idx+2)%4] + box_pts[(short_idx+3)%4]) / 2
        thick = min(edges[short_idx], edges[(short_idx+2)%4])
        dx, dy = mid_e[0]-mid_s[0], mid_e[1]-mid_s[1]
        return {
            "center_start": mid_s,
            "center_end": mid_e,
            "thickness": float(thick),
            "length": float(np.linalg.norm(mid_e - mid_s)),
            "angle": float(math.atan2(abs(dx), abs(dy))),
        }

    def _build_wall(self, wall, rects_data):
        all_pts = []
        for rd in rects_data:
            cs, ce = rd["center_start"], rd["center_end"]
            p1 = self._get_or_create_pt(all_pts, float(cs[0]), float(cs[1]))
            p2 = self._get_or_create_pt(all_pts, float(ce[0]), float(ce[1]))
            length = p1.distance_to(p2)
            if length < self.min_rect_length:
                continue
            r = Rect(p1_id=p1.id, p2_id=p2.id, thickness=rd["thickness"],
                     length=length, angle=rd["angle"])
            wall.rects[r.id] = r
        for p in all_pts:
            wall.points[p.id] = p

    def _get_or_create_pt(self, pts, x, y):
        for p in pts:
            if math.hypot(p.x - x, p.y - y) < self.merge_tol:
                return p
        p = Point(x=x, y=y)
        pts.append(p)
        return p


# ---------------------------------------------------------------------------
# 2.4  Scale Calibration
# ---------------------------------------------------------------------------

class ScaleCalibrator:

    @staticmethod
    def from_known_factor(meters_per_pixel: float) -> float:
        return meters_per_pixel

    @staticmethod
    def from_reference_length(p1_px, p2_px, real_length_m):
        pixel_dist = math.hypot(p2_px[0]-p1_px[0], p2_px[1]-p1_px[1])
        if pixel_dist < 1e-9:
            raise ValueError("Reference points are coincident")
        scale = real_length_m / pixel_dist
        logger.info("Scale: %.6f m/px (%.1f px -> %.2f m)", scale, pixel_dist, real_length_m)
        return scale

    @staticmethod
    def from_image_dpi(dpi, drawing_scale=100.0):
        scale = (1.0 / dpi) * 0.0254 * drawing_scale
        logger.info("Scale from DPI: %.6f m/px (DPI=%d, 1:%d)", scale, int(dpi), int(drawing_scale))
        return scale


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class VectorizationPipeline:
    """
    End-to-end Stage 2: binary wall mask -> Points + Rectangles (JSON).

    Usage:
        pipe = VectorizationPipeline(scale=0.005)
        floor = pipe.run(mask_image)
        pipe.export_json(floor, "floor_01.json")
    """

    def __init__(self, scale=0.005, floor_height=2.80, dp_epsilon=3.0,
                 angle_tolerance_deg=10.0, max_wall_thickness_m=0.50,
                 close_kernel_size=5, open_kernel_size=3,
                 min_component_area=50, min_edge_length_px=5.0):
        self.scale = scale
        self.floor_height = floor_height
        self.cleaner = MaskCleaner(close_kernel_size, open_kernel_size, min_component_area)
        self.simplifier = PolygonSimplifier(dp_epsilon, angle_tolerance_deg, min_edge_length_px)
        self.discretizer = RectangleDiscretizer(max_wall_thickness_m)

    def run(self, mask, floor_id="floor_01"):
        floor = Floor(floor_id=floor_id, scale=self.scale, height=self.floor_height)

        logger.info("=" * 60)
        logger.info("Stage 2.1: Mask cleanup & contour extraction")
        cleaned = self.cleaner.clean(mask)
        contours = self.cleaner.extract_contours(cleaned)

        logger.info("Stage 2.2-2.3: Simplification -> Discretization")
        accepted, rejected = 0, 0
        for i, contour in enumerate(contours):
            poly_pts = self.simplifier.simplify(contour)
            if len(poly_pts) < 4:
                rejected += 1
                continue
            is_valid, reason = self.simplifier.validate_polygon(poly_pts)
            wall = self.discretizer.discretize(poly_pts, self.scale)
            wall.contour_px = contour
            if not is_valid:
                wall.rejected = True
                wall.rejection_reason = reason
            if wall.rejected:
                rejected += 1
            else:
                accepted += 1
            floor.walls.append(wall)

        logger.info("Done: %d accepted, %d rejected (%.1f%%)",
                     accepted, rejected, 100*rejected/max(1, accepted+rejected))
        tp = sum(len(w.points) for w in floor.walls if not w.rejected)
        tr = sum(len(w.rects) for w in floor.walls if not w.rejected)
        logger.info("Totals: %d points, %d rectangles, %d valid walls", tp, tr, accepted)
        return floor

    def export_json(self, floor, path):
        data = {
            "floor_id": floor.floor_id,
            "scale_m_per_px": floor.scale,
            "floor_height_m": floor.height,
            "offset": {"x": floor.offset_x, "y": floor.offset_y},
            "walls": []
        }
        for wall in floor.walls:
            data["walls"].append({
                "wall_id": wall.id,
                "rejected": wall.rejected,
                "rejection_reason": wall.rejection_reason,
                "points": [{"id": p.id, "x": round(p.x, 4), "y": round(p.y, 4)}
                           for p in wall.points.values()],
                "rects": [{"id": r.id, "p1_id": r.p1_id, "p2_id": r.p2_id,
                           "thickness_m": round(r.thickness, 4),
                           "length_m": round(r.length, 4),
                           "angle_rad": round(r.angle, 4)}
                          for r in wall.rects.values()]
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported to %s", path)

    def visualize(self, mask, floor, output_path=None):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask.copy()
        vis = (vis * 0.3).astype(np.uint8)
        colors = [(0,255,0),(255,100,0),(0,200,255),(200,0,255),(255,200,0)]

        for wi, wall in enumerate(floor.walls):
            color = (0,0,255) if wall.rejected else colors[wi % len(colors)]
            for rect in wall.rects.values():
                p1, p2 = wall.points.get(rect.p1_id), wall.points.get(rect.p2_id)
                if not p1 or not p2:
                    continue
                s = floor.scale
                cx1, cy1, cx2, cy2 = p1.x/s, p1.y/s, p2.x/s, p2.y/s
                cv2.line(vis, (int(cx1),int(cy1)), (int(cx2),int(cy2)), color, 2)
                dx, dy = cx2-cx1, cy2-cy1
                lp = math.hypot(dx, dy)
                if lp < 1e-6:
                    continue
                px, py = -dy/lp*(rect.thickness/s/2), dx/lp*(rect.thickness/s/2)
                corners = np.array([
                    [cx1+px,cy1+py],[cx2+px,cy2+py],
                    [cx2-px,cy2-py],[cx1-px,cy1-py]], dtype=np.int32)
                cv2.polylines(vis, [corners], True, color, 1)
            for pt in wall.points.values():
                cv2.circle(vis, (int(pt.x/floor.scale), int(pt.y/floor.scale)), 4, (255,255,255), -1)
                cv2.circle(vis, (int(pt.x/floor.scale), int(pt.y/floor.scale)), 4, color, 1)
        if output_path:
            cv2.imwrite(output_path, vis)
            logger.info("Visualization: %s", output_path)
        return vis


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def create_demo_mask(w=512, h=512):
    """Synthetic mask: L-wall, straight wall, T-wall, room enclosure."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (50,50), (70,250), 255, -1)      # L vertical
    cv2.rectangle(mask, (50,230), (200,250), 255, -1)     # L horizontal
    cv2.rectangle(mask, (50,400), (450,415), 255, -1)     # straight
    cv2.rectangle(mask, (350,80), (370,300), 255, -1)     # T stem
    cv2.rectangle(mask, (300,80), (420,100), 255, -1)     # T top
    cv2.rectangle(mask, (150,120), (280,135), 255, -1)    # room top
    cv2.rectangle(mask, (150,120), (165,250), 255, -1)    # room left
    cv2.rectangle(mask, (265,120), (280,250), 255, -1)    # room right
    cv2.rectangle(mask, (150,235), (280,250), 255, -1)    # room bottom
    return mask


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Wall Vectorization")
    #parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--input", "-i", type=str, default="C:\\Dev\\Plan_2_FEM_2026\\output_mask.png")
    parser.add_argument("--scale", "-s", type=float, default=0.01)
    parser.add_argument("--output-json", "-o", type=str, default="floor_output.json")
    parser.add_argument("--output-vis", "-v", type=str, default="floor_visualization.png")
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
        scale=args.scale, floor_height=args.floor_height,
        dp_epsilon=3.0, angle_tolerance_deg=10.0, max_wall_thickness_m=0.50)

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
            print(f"    Rect: ({p1.x:.2f},{p1.y:.2f})->({p2.x:.2f},{p2.y:.2f})"
                  f"  L={r.length:.2f}m  t={r.thickness:.3f}m  a={math.degrees(r.angle):.1f}deg")
    for w in rejected_w:
        print(f"  [REJECTED] Wall {w.id}: {w.rejection_reason}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()