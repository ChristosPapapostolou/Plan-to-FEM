"""
GNN-EP Training Pipeline
=========================
Converts StructGAN semantic image pairs into graph data and trains the
GNN-EP-4 model for shear wall layout prediction.

Data flow:
  StructGAN image A (architectural) -> skeleton -> Stage 3 graph (input)
  StructGAN image B (structural)    -> shear wall mask -> ground truth ratios

Usage:
  python train_gnn_ep.py --data-dir StructGAN_v1/0_datasets --epochs 200
"""

from __future__ import annotations
import argparse, json, logging, math, os, time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

# ======================================================================
# 1. Semantic image parser
# ======================================================================

# BGR color map for StructGAN semantic images
COLOR_PSW       = np.array([152, 152, 152], dtype=np.uint8)  # potential shear wall
COLOR_PW        = np.array([128, 128, 128], dtype=np.uint8)  # partition wall
COLOR_DW_INDOOR = np.array([0, 255, 0], dtype=np.uint8)      # indoor door/window (green)
COLOR_DW_OUTDOOR= np.array([255, 0, 0], dtype=np.uint8)      # outdoor door (blue in RGB -> BGR)
COLOR_SW        = np.array([0, 0, 255], dtype=np.uint8)       # shear wall (red in RGB -> BGR)
COLOR_BG        = np.array([255, 255, 255], dtype=np.uint8)   # background

TOLERANCE = 15  # color matching tolerance for anti-aliased edges


def extract_masks(img_bgr, tol=TOLERANCE):
    """Extract semantic masks from a StructGAN semantic image."""
    def _match(color):
        return np.all(np.abs(img_bgr.astype(int) - color.astype(int)) < tol, axis=2).astype(np.uint8) * 255

    return {
        "psw":       _match(COLOR_PSW),
        "pw":        _match(COLOR_PW),
        "dw_indoor": _match(COLOR_DW_INDOOR),
        "dw_outdoor":_match(COLOR_DW_OUTDOOR),
        "sw":        _match(COLOR_SW),
    }


# ======================================================================
# 2. Skeleton-based segment extraction
# ======================================================================

def skeletonize(mask):
    """Morphological thinning with border padding."""
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
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


def extract_segments(mask, min_length=8, snap_deg=8.0):
    """
    Extract straight line segments from a binary mask using
    skeleton + Hough transform.

    Returns list of (p1, p2, thickness_px) where p1,p2 are (x,y) arrays.
    """
    if cv2.countNonZero(mask) < 20:
        return []

    dist_map = distance_transform_edt(mask > 0).astype(np.float32)
    skel = skeletonize(mask)
    if cv2.countNonZero(skel) == 0:
        return []

    # Dilate skeleton for better Hough detection
    skel_d = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(skel_d, 1, np.pi/180, 6,
                             minLineLength=min_length, maxLineGap=6)
    if lines is None:
        return []

    # Collect raw segments
    raw = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = np.array([float(x1), float(y1)])
        p2 = np.array([float(x2), float(y2)])
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < min_length:
            continue
        # Snap to orthogonal
        p1, p2 = _snap(p1, p2, snap_deg)
        raw.append((p1, p2, seg_len))

    # Merge collinear overlapping segments
    merged = _merge_collinear(raw)

    # Measure thickness and refine centerline
    result = []
    for p1, p2 in merged:
        t, p1r, p2r = _measure_and_refine(dist_map, p1, p2)
        if t >= 3.0 and np.linalg.norm(p2r - p1r) >= min_length:
            result.append((p1r, p2r, t))

    return result


def _snap(p1, p2, tol_deg):
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    angle = math.degrees(math.atan2(abs(dy), max(abs(dx), 0.01)))
    p1s, p2s = p1.copy(), p2.copy()
    if angle < tol_deg:
        mid_y = (p1[1] + p2[1]) / 2
        p1s[1] = mid_y; p2s[1] = mid_y
    elif angle > 90 - tol_deg:
        mid_x = (p1[0] + p2[0]) / 2
        p1s[0] = mid_x; p2s[0] = mid_x
    return p1s, p2s


def _merge_collinear(raw_segments, perp_tol=3.0, gap_tol=6.0):
    if not raw_segments:
        return []
    horiz, vert, diag = [], [], []
    for p1, p2, L in raw_segments:
        dx, dy = abs(p2[0]-p1[0]), abs(p2[1]-p1[1])
        angle = math.degrees(math.atan2(dy, max(dx, 0.01)))
        if angle < 20: horiz.append((p1, p2))
        elif angle > 70: vert.append((p1, p2))
        else: diag.append((p1, p2))

    result = []
    result.extend(_merge_axis(horiz, 1, 0, perp_tol, gap_tol))
    result.extend(_merge_axis(vert, 0, 1, perp_tol, gap_tol))
    result.extend(diag)
    return result


def _merge_axis(segs, perp_axis, main_axis, perp_tol, gap_tol):
    if not segs: return []
    indexed = []
    for p1, p2 in segs:
        perp = (p1[perp_axis] + p2[perp_axis]) / 2.0
        lo = min(p1[main_axis], p2[main_axis])
        hi = max(p1[main_axis], p2[main_axis])
        indexed.append((perp, lo, hi))
    indexed.sort(key=lambda x: x[0])

    merged = []
    used = [False] * len(indexed)
    for i in range(len(indexed)):
        if used[i]: continue
        perps = [indexed[i][0]]
        lo, hi = indexed[i][1], indexed[i][2]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(i+1, len(indexed)):
                if used[j]: continue
                if abs(indexed[j][0] - np.mean(perps)) > perp_tol: continue
                if indexed[j][1] > hi + gap_tol or indexed[j][2] < lo - gap_tol: continue
                perps.append(indexed[j][0])
                lo = min(lo, indexed[j][1])
                hi = max(hi, indexed[j][2])
                used[j] = True
                changed = True
        p1, p2 = np.zeros(2), np.zeros(2)
        p1[perp_axis] = np.mean(perps); p2[perp_axis] = np.mean(perps)
        p1[main_axis] = lo; p2[main_axis] = hi
        merged.append((p1, p2))
    return merged


def _measure_and_refine(dist_map, p1, p2, n_samples=20):
    h, w = dist_map.shape
    seg_dir = p2 - p1
    seg_len = np.linalg.norm(seg_dir)
    if seg_len < 1e-6:
        return 0.0, p1, p2
    seg_unit = seg_dir / seg_len
    perp = np.array([-seg_unit[1], seg_unit[0]])

    offsets, dists = [], []
    for t in np.linspace(0, 1, n_samples):
        pt = p1 + t * seg_dir
        best_off, best_val = 0.0, 0.0
        for off in np.linspace(-15, 15, 31):
            sx = int(round(pt[0] + perp[0] * off))
            sy = int(round(pt[1] + perp[1] * off))
            if 0 <= sy < h and 0 <= sx < w:
                val = dist_map[sy, sx]
                if val > best_val:
                    best_val = val
                    best_off = off
        offsets.append(best_off)
        dists.append(best_val)

    if not dists or max(dists) < 0.5:
        return 0.0, p1, p2
    med_off = float(np.median(offsets))
    p1r = p1 + perp * med_off
    p2r = p2 + perp * med_off
    thickness = 2.0 * float(np.median(dists))
    return thickness, p1r, p2r


# ======================================================================
# 3. Graph construction from semantic image
# ======================================================================

L_GRAPH_SCALE = 20480.0  # mm normalization constant

# Edge type indices (Edge-PSW-DW scheme)
ET_PSW = 0
ET_INDOOR_DW = 1
ET_OUTDOOR_D = 2


@dataclass
class GraphEdge:
    edge_id: int
    p1: np.ndarray      # (x, y) in pixels
    p2: np.ndarray
    edge_type: int       # 0=PSW, 1=indoor DW, 2=outdoor door
    thickness_px: float
    length_px: float


def image_to_graph(img_a_bgr, scale_mm_per_px=10.0, min_seg_px=8):
    """
    Convert a StructGAN architectural semantic image to a graph.

    Parameters
    ----------
    img_a_bgr : the architectural image (BGR)
    scale_mm_per_px : spatial resolution (StructGAN uses 10mm/pixel)

    Returns
    -------
    nodes : (N, 2) normalised coordinates [x_n, y_n]
    edge_index : (2, E) adjacency
    edge_features : (E, 8) [one_hot(3), x_l, y_l, x_r, y_r, length]
    edges : list[GraphEdge] with pixel coords
    center_px : (2,) center used for normalisation
    """
    masks = extract_masks(img_a_bgr)

    all_edges: list[GraphEdge] = []
    eid = 0

    # Extract PSW segments
    for p1, p2, t in extract_segments(masks["psw"], min_length=min_seg_px):
        all_edges.append(GraphEdge(eid, p1, p2, ET_PSW, t, np.linalg.norm(p2-p1)))
        eid += 1

    # Extract indoor door/window segments
    for p1, p2, t in extract_segments(masks["dw_indoor"], min_length=min_seg_px):
        all_edges.append(GraphEdge(eid, p1, p2, ET_INDOOR_DW, t, np.linalg.norm(p2-p1)))
        eid += 1

    # Extract outdoor door segments
    for p1, p2, t in extract_segments(masks["dw_outdoor"], min_length=min_seg_px):
        all_edges.append(GraphEdge(eid, p1, p2, ET_OUTDOOR_D, t, np.linalg.norm(p2-p1)))
        eid += 1

    if not all_edges:
        return None

    # Collect & merge endpoints
    pts = []
    for e in all_edges:
        pts.append(e.p1.copy())
        pts.append(e.p2.copy())
    pts = np.array(pts)

    # Merge nearby points (tolerance = half min wall thickness)
    merge_tol = 5.0  # pixels
    canonical = list(range(len(pts)))

    for i in range(len(pts)):
        if canonical[i] != i:
            continue
        for j in range(i+1, len(pts)):
            if canonical[j] != j:
                continue
            if np.linalg.norm(pts[i] - pts[j]) < merge_tol:
                canonical[j] = i

    # Build unique node list
    node_map = {}  # canonical_idx -> node_idx
    node_coords = []
    for i in range(len(pts)):
        ci = canonical[i]
        if ci not in node_map:
            node_map[ci] = len(node_coords)
            # Average position of merged points
            members = [j for j in range(len(pts)) if canonical[j] == ci]
            avg = np.mean(pts[members], axis=0)
            node_coords.append(avg)
    node_coords = np.array(node_coords)

    # Center for normalisation
    center_px = node_coords.mean(axis=0)

    # Normalise: c_n = (c_px * scale_mm - center_mm) / L_GRAPH_SCALE
    def norm(xy_px):
        return (xy_px * scale_mm_per_px - center_px * scale_mm_per_px) / L_GRAPH_SCALE

    N = len(node_coords)
    nodes_norm = np.array([norm(c) for c in node_coords], dtype=np.float32)

    # Build edges
    edge_src, edge_dst = [], []
    edge_feats = []
    psw_edge_indices = []  # which edges are PSW (for label computation)

    for k, e in enumerate(all_edges):
        ci1 = canonical[2*k]
        ci2 = canonical[2*k+1]
        ni1 = node_map[ci1]
        ni2 = node_map[ci2]
        if ni1 == ni2:
            continue

        n1 = norm(node_coords[ni1])
        n2 = norm(node_coords[ni2])
        ln = e.length_px * scale_mm_per_px / L_GRAPH_SCALE

        # One-hot class
        oh = [0.0, 0.0, 0.0]
        oh[e.edge_type] = 1.0

        feat = oh + [float(n1[0]), float(n1[1]),
                     float(n2[0]), float(n2[1]), float(ln)]
        edge_feats.append(feat)
        edge_src.append(ni1)
        edge_dst.append(ni2)

        if e.edge_type == ET_PSW:
            psw_edge_indices.append(len(edge_src) - 1)

    if not edge_src:
        return None

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_features = np.array(edge_feats, dtype=np.float32)

    return {
        "nodes": nodes_norm,             # (N, 2)
        "edge_index": edge_index,         # (2, E)
        "edge_features": edge_features,   # (E, 8)
        "edges": all_edges,               # raw GraphEdge list
        "psw_edge_indices": psw_edge_indices,
        "center_px": center_px,
        "node_coords_px": node_coords,
        "canonical": canonical,
        "node_map": node_map,
    }


# ======================================================================
# 4. Ground truth label computation
# ======================================================================

def compute_sw_ratios(graph_data, sw_mask, n_samples=50):
    """
    For each PSW edge, compute [ratio_left, ratio_right] by sampling
    the shear wall mask along the edge centerline.

    ratio_left  = fraction of edge covered by SW from the left endpoint
    ratio_right = fraction of edge covered by SW from the right endpoint
    """
    all_edges = graph_data["edges"]
    canonical = graph_data["canonical"]
    node_map = graph_data["node_map"]
    node_coords = graph_data["node_coords_px"]

    E = graph_data["edge_index"].shape[1]
    labels = np.zeros((E, 2), dtype=np.float32)

    h, w = sw_mask.shape
    edge_count = 0

    for ei in range(E):
        ni1 = graph_data["edge_index"][0, ei]
        ni2 = graph_data["edge_index"][1, ei]

        # Find the corresponding GraphEdge
        ge = all_edges[ei] if ei < len(all_edges) else None
        if ge is None or ge.edge_type != ET_PSW:
            continue

        p1 = node_coords[ni1]
        p2 = node_coords[ni2]
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < 1:
            continue

        # Sample SW mask along the segment
        ts = np.linspace(0, 1, n_samples)
        is_sw = []
        for t in ts:
            pt = p1 + t * (p2 - p1)
            ix, iy = int(round(pt[0])), int(round(pt[1]))
            # Sample a small neighbourhood for robustness
            hit = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    sy, sx = iy + dy, ix + dx
                    if 0 <= sy < h and 0 <= sx < w:
                        if sw_mask[sy, sx] > 127:
                            hit = True
                            break
                if hit:
                    break
            is_sw.append(hit)

        is_sw = np.array(is_sw)

        # Compute ratio_left: longest continuous SW run from t=0
        ratio_left = 0.0
        for i in range(len(is_sw)):
            if is_sw[i]:
                ratio_left = (i + 1) / n_samples
            else:
                break

        # Compute ratio_right: longest continuous SW run from t=1
        ratio_right = 0.0
        for i in range(len(is_sw) - 1, -1, -1):
            if is_sw[i]:
                ratio_right = (n_samples - i) / n_samples
            else:
                break

        # Cap at 0.5 each (if overlapping, both are 0.5)
        total = ratio_left + ratio_right
        if total > 1.0:
            ratio_left = 0.5
            ratio_right = 0.5

        labels[ei, 0] = ratio_left
        labels[ei, 1] = ratio_right
        edge_count += 1

    return labels


# ======================================================================
# 5. Dataset conversion
# ======================================================================

def convert_image_pair(img_a_path, img_b_path, scale_mm_per_px=10.0):
    """
    Convert one StructGAN image pair to a training sample.

    Returns dict with graph tensors + ground truth labels, or None on failure.
    """
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)
    if img_a is None or img_b is None:
        return None

    # Handle the 2048-wide format (some are paired left-right)
    h, w = img_a.shape[:2]
    if w > 1.5 * h:
        # Use left half only (both halves are similar floor plans)
        img_a = img_a[:, :w//2]
        img_b = img_b[:, :w//2]

    # Build graph from architectural image
    graph_data = image_to_graph(img_a, scale_mm_per_px)
    if graph_data is None:
        return None

    # Extract shear wall mask from structural image
    masks_b = extract_masks(img_b)
    sw_mask = masks_b["sw"]

    # Compute ground truth ratios
    labels = compute_sw_ratios(graph_data, sw_mask)

    n_psw = len(graph_data["psw_edge_indices"])
    n_sw = np.sum(labels.sum(axis=1) > 0)

    return {
        "nodes": graph_data["nodes"],
        "edge_index": graph_data["edge_index"],
        "edge_features": graph_data["edge_features"],
        "labels": labels,  # (E, 2) - only PSW edges have non-zero labels
        "n_nodes": graph_data["nodes"].shape[0],
        "n_edges": graph_data["edge_index"].shape[1],
        "n_psw": n_psw,
        "n_sw_edges": int(n_sw),
        "psw_edge_indices": graph_data["psw_edge_indices"],
    }


def convert_dataset(data_dir, split="train", groups=None):
    """
    Convert all image pairs in the StructGAN dataset.

    Returns list of sample dicts.
    """
    if groups is None:
        groups = ["L1_7", "L2_7", "L1L2_8"]

    prefix_a = f"{split}_A"
    prefix_b = f"{split}_B"

    samples = []
    for group in groups:
        dir_a = os.path.join(data_dir, group, prefix_a)
        dir_b = os.path.join(data_dir, group, prefix_b)
        if not os.path.isdir(dir_a):
            logger.warning("Missing: %s", dir_a)
            continue

        files = sorted([f for f in os.listdir(dir_a) if f.endswith(".png")])
        for fname in files:
            path_a = os.path.join(dir_a, fname)
            path_b = os.path.join(dir_b, fname)
            if not os.path.exists(path_b):
                continue

            sample = convert_image_pair(path_a, path_b)
            if sample is None:
                logger.warning("Failed to convert: %s", fname)
                continue

            sample["name"] = fname
            sample["group"] = group
            samples.append(sample)
            logger.info("  %s/%s: %d nodes, %d edges (%d PSW, %d SW)",
                       group, fname, sample["n_nodes"], sample["n_edges"],
                       sample["n_psw"], sample["n_sw_edges"])

    logger.info("Converted %d samples from %d groups", len(samples), len(groups))
    return samples


# ======================================================================
# 6. Data augmentation (per Zhao et al. Section 4.3)
# ======================================================================

def augment_graph(sample, rng=None):
    """
    Random augmentation: flip + rotate + translate.
    Operates on normalised coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = sample["nodes"].copy()
    ef = sample["edge_features"].copy()

    # Random flip (vertical axis)
    if rng.random() > 0.5:
        nodes[:, 1] *= -1
        # Flip y coordinates in edge features (indices 4, 6)
        ef[:, 4] *= -1
        ef[:, 6] *= -1

    # Random rotation: 0, 90, 180, 270
    rot = rng.choice([0, 90, 180, 270])
    if rot > 0:
        theta = np.radians(rot)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        nodes = (R @ nodes.T).T
        for i in range(ef.shape[0]):
            xy1 = R @ ef[i, 3:5]
            xy2 = R @ ef[i, 5:7]
            ef[i, 3:5] = xy1
            ef[i, 5:7] = xy2

    # Random translation (0-20m in 2m steps -> normalised)
    tx = rng.uniform(0, 20) * 1000 / L_GRAPH_SCALE
    ty = rng.uniform(0, 20) * 1000 / L_GRAPH_SCALE
    nodes[:, 0] += tx
    nodes[:, 1] += ty
    ef[:, 3] += tx; ef[:, 5] += tx
    ef[:, 4] += ty; ef[:, 6] += ty

    return {
        **sample,
        "nodes": nodes.astype(np.float32),
        "edge_features": ef.astype(np.float32),
    }


# ======================================================================
# 7. GNN-EP-4 Model (PyTorch)
# ======================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:

    class SFLayer(nn.Module):
        """GraphSAGE-Frame layer with edge feature enhancement."""
        def __init__(self, d_in, d_out, d_e=8):
            super().__init__()
            self.w_self = nn.Linear(d_in, d_out)
            self.w_neigh = nn.Linear(d_in, d_out)
            self.w_edge = nn.Linear(d_e, d_out)

        def forward(self, x, ei, ea):
            src, dst = ei
            N = x.size(0)
            # Mean aggregate neighbours
            na = torch.zeros(N, x.size(1), device=x.device)
            ee = torch.zeros(N, self.w_edge.out_features, device=x.device)
            cnt = torch.zeros(N, 1, device=x.device)
            na.index_add_(0, dst, x[src])
            cnt.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))
            cnt = cnt.clamp(min=1)
            na /= cnt
            ee.index_add_(0, dst, self.w_edge(ea))
            ee /= cnt
            return F.relu(self.w_self(x) + self.w_neigh(na) + ee)

    class GNNEP4(nn.Module):
        """
        GNN-EP-4: 6 SF layers + MLP head.
        Best model from Zhao et al. (2023).
        Output: (E, 2) shear wall ratios per edge.
        """
        def __init__(self, d_in=2, d_e=8, d_h=32, dropout=0.25):
            super().__init__()
            self.sf = nn.ModuleList([
                SFLayer(d_in, 16, d_e),
                SFLayer(16, 16, d_e),
                SFLayer(16, 32, d_e),
                SFLayer(32, 32, d_e),
                SFLayer(32, 32, d_e),
                SFLayer(32, d_h, d_e),
            ])
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(32)
            self.drop = nn.Dropout(dropout)

            mlp_in = 2 * d_h + d_e
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.BatchNorm1d(32), nn.Dropout(dropout),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Dropout(dropout),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 2), nn.Sigmoid(),
            )

        def forward(self, x, ei, ea):
            h = self.sf[0](x, ei, ea)
            h = self.sf[1](h, ei, ea)
            h = self.drop(self.bn1(h))
            h = self.sf[2](h, ei, ea)
            h = self.sf[3](h, ei, ea)
            h = self.drop(self.bn2(h))
            h = self.sf[4](h, ei, ea)
            h = self.sf[5](h, ei, ea)
            s, d = ei
            edge_rep = torch.cat([h[s], ea, h[d]], dim=-1)
            return self.mlp(edge_rep)

    # ================================================================
    # 8. Training loop
    # ================================================================

    def train_gnn_ep(
        train_samples,
        test_samples=None,
        *,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        augment_per_epoch=8,
        device="cpu",
        save_path="gnn_ep4.pt",
        sw_pos_weight=5.0,
        bce_alpha=1.0,
    ):
        """
        Train GNN-EP-4 on converted StructGAN data.

        Uses L1 + class-balanced BCE loss on PSW edges only.
        BCE provides direct binary SW/non-SW gradient signal.
        sw_pos_weight scales the L1 contribution from SW edges.
        """
        model = GNNEP4().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        rng = np.random.default_rng(42)
        best_val_loss = float("inf")
        best_val_iou = 0.0
        best_thresh = 0.10
        history = {"train_loss": [], "val_loss": [], "val_iou": []}

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_edges = 0

            # Shuffle + augment
            rng.shuffle(train_samples)

            for sample in train_samples:
                for _ in range(augment_per_epoch):
                    aug = augment_graph(sample, rng)
                    x = torch.tensor(aug["nodes"], device=device)
                    ei = torch.tensor(aug["edge_index"], device=device)
                    ea = torch.tensor(aug["edge_features"], device=device)
                    y = torch.tensor(aug["labels"], device=device)

                    pred = model(x, ei, ea)

                    # L1 loss on PSW edges only
                    psw_mask = torch.zeros(ei.shape[1], dtype=torch.bool,
                                           device=device)
                    for idx in aug["psw_edge_indices"]:
                        if idx < ei.shape[1]:
                            psw_mask[idx] = True

                    if psw_mask.any():
                        p, t = pred[psw_mask], y[psw_mask]
                        is_sw = (t.sum(-1) > 0).float()

                        # L1 regression (weighted by sw_pos_weight)
                        w = (1.0 + (sw_pos_weight - 1.0) * is_sw).unsqueeze(1)
                        l1_loss = (w * (p - t).abs()).mean()

                        # BCE for binary SW/non-SW discrimination
                        sw_prob = ((p[:, 0] + p[:, 1]) / 2).clamp(1e-7, 1 - 1e-7)
                        bce = F.binary_cross_entropy(sw_prob, is_sw)

                        loss = l1_loss + bce_alpha * bce
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * psw_mask.sum().item()
                        n_edges += psw_mask.sum().item()

            scheduler.step()
            avg_loss = epoch_loss / max(n_edges, 1)
            history["train_loss"].append(avg_loss)

            # Validation
            val_loss = 0.0
            val_iou = 0.0
            val_sw_recall = 0.0
            epoch_best_thresh = best_thresh
            if test_samples:
                model.eval()
                vl, ve = 0.0, 0
                vsw_hit, vsw_total = 0, 0
                vthr_iou: dict = {}
                with torch.no_grad():
                    for sample in test_samples:
                        x = torch.tensor(sample["nodes"], device=device)
                        ei = torch.tensor(sample["edge_index"], device=device)
                        ea = torch.tensor(sample["edge_features"], device=device)
                        y = torch.tensor(sample["labels"], device=device)

                        pred = model(x, ei, ea)

                        psw_mask = torch.zeros(ei.shape[1], dtype=torch.bool,
                                               device=device)
                        for idx in sample["psw_edge_indices"]:
                            if idx < ei.shape[1]:
                                psw_mask[idx] = True

                        if psw_mask.any():
                            vl += F.l1_loss(pred[psw_mask], y[psw_mask],
                                            reduction="sum").item()
                            ve += psw_mask.sum().item()

                            p_psw = pred[psw_mask]
                            t_psw = y[psw_mask]
                            gt_sw = (t_psw.sum(-1) > 0)
                            sw_score = (p_psw[:, 0] + p_psw[:, 1]) / 2

                            # IoU at best threshold (sweep across candidates)
                            for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                                pr = sw_score > thr
                                tp = (pr & gt_sw).sum().item()
                                union = (pr | gt_sw).sum().item()
                                vthr_iou[thr] = vthr_iou.get(thr, 0.0) + (tp / max(union, 1))

                            # Primary recall metric (threshold 0.10)
                            pr_sw = sw_score > 0.10
                            vsw_hit += (pr_sw & gt_sw).sum().item()
                            vsw_total += gt_sw.sum().item()

                val_loss = vl / max(ve, 1)
                # Pick threshold with best average IoU across val graphs
                val_iou = 0.0
                for thr, iou_sum in vthr_iou.items():
                    avg = iou_sum / max(len(test_samples), 1)
                    if avg > val_iou:
                        val_iou = avg
                        epoch_best_thresh = thr
                val_sw_recall = vsw_hit / max(vsw_total, 1)
                history["val_loss"].append(val_loss)
                history["val_iou"].append(val_iou)

                if val_loss < best_val_loss and val_sw_recall > 0.40:
                    best_val_loss = val_loss
                    best_val_iou = val_iou
                    best_thresh = epoch_best_thresh
                    torch.save(model.state_dict(), save_path)
                    logger.info("  Best val L1=%.4f recall=%.3f  Saved: %s",
                                best_val_loss, val_sw_recall, save_path)

            if epoch % 10 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                msg = (f"Epoch {epoch:>3d}/{epochs}  "
                       f"train_L1={avg_loss:.4f}  "
                       f"val_L1={val_loss:.4f}  "
                       f"val_IoU={val_iou:.4f}@{epoch_best_thresh:.2f}  "
                       f"val_SW_recall={val_sw_recall:.3f}  "
                       f"lr={lr_now:.6f}")
                logger.info(msg)
                print(msg)

        logger.info("Best val L1: %.4f  IoU: %.4f @ thresh=%.2f  Saved: %s",
                    best_val_loss, best_val_iou, best_thresh, save_path)
        return model, history, best_thresh

    def _approx_iou(pred, target):
        """
        Approximate IoU from edge ratios.
        For each PSW edge, the SW coverage = ratio_left + ratio_right.
        IoU = intersection / union of coverage.
        """
        pred_cov = np.clip(pred[:, 0] + pred[:, 1], 0, 1)
        tgt_cov = np.clip(target[:, 0] + target[:, 1], 0, 1)
        inter = np.minimum(pred_cov, tgt_cov).sum()
        union = np.maximum(pred_cov, tgt_cov).sum()
        return inter / max(union, 1e-6)


# ======================================================================
# 9. CLI entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GNN-EP-4 on StructGAN dataset")
    parser.add_argument("--data-dir", default="C:\\Dev\\Plan_2_FEM_2026\\train\\data",
                        help="Path to StructGAN_v1/0_datasets")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", type=int, default=8,
                        help="Augmentations per sample per epoch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default="gnn_ep4.pt")
    parser.add_argument("--sw-pos-weight", type=float, default=1.0,
                        help="L1 loss multiplier for SW edges (1.0=equal weight)")
    parser.add_argument("--bce-alpha", type=float, default=1.0,
                        help="Weight for BCE classification term (0=pure L1)")
    parser.add_argument("--convert-only", default=False, action="store_true",
                        help="Only convert dataset, don't train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    if not HAS_TORCH and not args.convert_only:
        logger.error("PyTorch required for training. "
                     "Use --convert-only to just convert data.")
        return

    # Convert dataset
    logger.info("=" * 60)
    logger.info("Converting training data...")
    train_data = convert_dataset(args.data_dir, split="train")

    logger.info("Converting test data...")
    test_data = convert_dataset(args.data_dir, split="test")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Train: {len(train_data)} graphs")
    print(f"  Test:  {len(test_data)} graphs")
    total_psw = sum(s["n_psw"] for s in train_data)
    total_sw = sum(s["n_sw_edges"] for s in train_data)
    print(f"  Total PSW edges (train): {total_psw}")
    print(f"  Total SW edges (train):  {total_sw} "
          f"({100*total_sw/max(total_psw,1):.0f}%)")
    avg_nodes = np.mean([s["n_nodes"] for s in train_data])
    avg_edges = np.mean([s["n_edges"] for s in train_data])
    print(f"  Avg nodes/graph: {avg_nodes:.1f}")
    print(f"  Avg edges/graph: {avg_edges:.1f}")
    print(f"{'='*60}")

    if args.convert_only:
        # Save converted data
        out_path = os.path.join(os.path.dirname(args.data_dir),
                                "converted_dataset.json")
        serializable = []
        for s in train_data + test_data:
            serializable.append({
                "name": s["name"], "group": s["group"],
                "n_nodes": s["n_nodes"], "n_edges": s["n_edges"],
                "n_psw": s["n_psw"], "n_sw_edges": s["n_sw_edges"],
                "nodes": s["nodes"].tolist(),
                "edge_index": s["edge_index"].tolist(),
                "edge_features": s["edge_features"].tolist(),
                "labels": s["labels"].tolist(),
                "psw_edge_indices": s["psw_edge_indices"],
            })
        with open(out_path, "w") as f:
            json.dump(serializable, f)
        logger.info("Saved converted data to %s", out_path)
        return

    # Train
    logger.info("Training GNN-EP-4...")
    model, history, best_thresh = train_gnn_ep(
        train_data, test_data,
        epochs=args.epochs,
        lr=args.lr,
        augment_per_epoch=args.augment,
        device=args.device,
        save_path=args.save,
        sw_pos_weight=args.sw_pos_weight,
        bce_alpha=args.bce_alpha,
    )

    print(f"\nTraining complete. Best model saved to {args.save}")
    print(f"Optimal SW threshold: {best_thresh:.2f}")


if __name__ == "__main__":
    main()