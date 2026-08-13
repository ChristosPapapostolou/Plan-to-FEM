"""
Shared GNN model & graph featurization  (single source of truth)
================================================================
Used by BOTH training (train/train_gnn_multi.py) and inference
(stages/stage_4) so the two sides can never drift apart again.

Contents:
  - build_node_features()   : rich per-node features (junction type, degree,
                              thickness, tributary proxy, exterior flag)
  - normalize_graph()       : per-graph bbox normalization (replaces the fixed
                              L_GRAPH_SCALE=20480mm which put small plans OOD)
  - add_virtual_candidates(): grid-intersection candidate nodes so the GNN can
                              place columns in open space, not only on walls
  - GNNEPMulti              : multi-task model
                              edge head -> shear-wall ratios  (E, 2)
                              node head -> column probability (N, 1)
                              LayerNorm everywhere (BatchNorm collapses with
                              batch = 1 graph)

Edge feature layout (d_e = 9):
  [onehot_PSW, onehot_INDOOR_DW, onehot_OUTDOOR_D, onehot_VIRTUAL,
   x_l, y_l, x_r, y_r, length]        (coords per-graph normalized)

Node feature layout (d_in = 11):
  [x, y, degree/4, end, L, T, X, is_corner, max_thk, sum_len/10, is_ext]
"""

from __future__ import annotations
import math
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:   # ImportError, or broken CUDA install (ValueError/OSError)
    HAS_TORCH = False

# Edge classes
ET_PSW = 0
ET_INDOOR_DW = 1
ET_OUTDOOR_D = 2
ET_VIRTUAL = 3
N_EDGE_CLASSES = 4

D_EDGE = N_EDGE_CLASSES + 5   # 9
D_NODE = 11


# =============================================================================
# 1. Per-graph normalization
# =============================================================================

def normalize_graph(xy_m: np.ndarray):
    """
    Per-graph normalization: center at bbox centroid, scale by half the bbox
    diagonal (floored at 1 m).  Returns (xy_norm, center, scale).
    """
    xy_m = np.asarray(xy_m, dtype=np.float64)
    lo, hi = xy_m.min(axis=0), xy_m.max(axis=0)
    center = (lo + hi) / 2.0
    scale = max(float(np.linalg.norm(hi - lo)) / 2.0, 1.0)
    return ((xy_m - center) / scale).astype(np.float32), center, scale


# =============================================================================
# 2. Node featurization
# =============================================================================

def build_node_features(xy_m: np.ndarray, edge_index: np.ndarray,
                        edge_thickness: np.ndarray | None = None,
                        edge_class: np.ndarray | None = None,
                        virtual_mask: np.ndarray | None = None):
    """
    Compute D_NODE features per node.

    Parameters
    ----------
    xy_m           : (N, 2) node coordinates in METRES
    edge_index     : (2, E)
    edge_thickness : (E,) wall thickness per edge in metres (0 for openings)
    edge_class     : (E,) integer edge class (virtual edges excluded from
                     junction statistics)
    virtual_mask   : (N,) bool, True for virtual candidate nodes

    Returns (N, D_NODE) float32. Coordinates are per-graph normalized.
    """
    xy_m = np.asarray(xy_m, dtype=np.float64)
    N = xy_m.shape[0]
    E = edge_index.shape[1] if edge_index.size else 0
    if edge_thickness is None:
        edge_thickness = np.zeros(E)
    if edge_class is None:
        edge_class = np.zeros(E, dtype=int)
    if virtual_mask is None:
        virtual_mask = np.zeros(N, dtype=bool)

    xy_n, center, scale = normalize_graph(xy_m)

    # Incident REAL (non-virtual) edges per node
    inc: list[list[int]] = [[] for _ in range(N)]
    for k in range(E):
        if edge_class[k] == ET_VIRTUAL:
            continue
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        inc[u].append(k)
        inc[v].append(k)

    # Exterior flag: distance to convex hull of REAL nodes
    is_ext = np.zeros(N)
    real_pts = xy_m[~virtual_mask]
    hull_pts = None
    if real_pts.shape[0] >= 3:
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(real_pts)
            hull_pts = real_pts[h.vertices]
        except Exception:
            hull_pts = None
    if hull_pts is not None:
        for i in range(N):
            if _dist_to_poly(xy_m[i], hull_pts) < 0.35:
                is_ext[i] = 1.0

    feats = np.zeros((N, D_NODE), dtype=np.float32)
    for i in range(N):
        deg = len(inc[i])
        # incident edge directions (unit vectors away from node i)
        dirs = []
        max_t, sum_len = 0.0, 0.0
        for k in inc[i]:
            u, v = int(edge_index[0, k]), int(edge_index[1, k])
            j = v if u == i else u
            d = xy_m[j] - xy_m[i]
            ln = np.linalg.norm(d)
            if ln > 1e-9:
                dirs.append(d / ln)
            max_t = max(max_t, float(edge_thickness[k]))
            sum_len += ln / 2.0
        # corner: exactly 2 edges meeting at ~90 deg
        is_corner = 0.0
        if len(dirs) == 2:
            cosang = float(np.clip(np.dot(dirs[0], dirs[1]), -1, 1))
            ang = math.degrees(math.acos(cosang))
            if 45.0 <= ang <= 135.0:
                is_corner = 1.0

        feats[i, 0] = xy_n[i, 0]
        feats[i, 1] = xy_n[i, 1]
        feats[i, 2] = min(deg, 8) / 4.0
        feats[i, 3] = 1.0 if deg <= 1 else 0.0                 # end / isolated
        feats[i, 4] = 1.0 if deg == 2 else 0.0                 # L / through
        feats[i, 5] = 1.0 if deg == 3 else 0.0                 # T
        feats[i, 6] = 1.0 if deg >= 4 else 0.0                 # X
        feats[i, 7] = is_corner
        feats[i, 8] = max_t
        feats[i, 9] = min(sum_len, 20.0) / 10.0
        feats[i, 10] = is_ext[i]
    return feats, center, scale


def _dist_to_poly(pt, poly):
    best = float("inf")
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab, ap = b - a, pt - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(pt - (a + t * ab))))
    return best


def build_edge_features(xy_m: np.ndarray, edge_index: np.ndarray,
                        edge_class: np.ndarray, center, scale):
    """(E, D_EDGE) edge features using the SAME per-graph normalization."""
    E = edge_index.shape[1]
    ef = np.zeros((E, D_EDGE), dtype=np.float32)
    for k in range(E):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        c = int(edge_class[k])
        ef[k, min(c, N_EDGE_CLASSES - 1)] = 1.0
        n1 = (xy_m[u] - center) / scale
        n2 = (xy_m[v] - center) / scale
        ef[k, N_EDGE_CLASSES + 0] = n1[0]
        ef[k, N_EDGE_CLASSES + 1] = n1[1]
        ef[k, N_EDGE_CLASSES + 2] = n2[0]
        ef[k, N_EDGE_CLASSES + 3] = n2[1]
        ef[k, N_EDGE_CLASSES + 4] = float(np.linalg.norm(xy_m[v] - xy_m[u])) / scale
    return ef


# =============================================================================
# 3. Virtual candidate nodes (grid intersections in open space)
# =============================================================================

def add_virtual_candidates(xy_m: np.ndarray, edge_index: np.ndarray,
                           edge_class: np.ndarray,
                           edge_thickness: np.ndarray,
                           candidates_m: list, *,
                           merge_tol: float = 0.40,
                           k_connect: int = 3,
                           max_link_m: float = 8.0):
    """
    Append grid-intersection candidate points as VIRTUAL nodes, each linked to
    its k nearest existing nodes by VIRTUAL edges (message-passing only).

    Candidates within merge_tol of an existing node are skipped (that node is
    already a candidate by construction).

    Returns (xy_m2, edge_index2, edge_class2, edge_thickness2, virtual_mask).
    """
    xy = np.asarray(xy_m, dtype=np.float64)
    N0 = xy.shape[0]
    new_pts = []
    for c in candidates_m:
        c = np.asarray(c, dtype=np.float64)
        d = np.linalg.norm(xy - c, axis=1)
        if d.size and d.min() < merge_tol:
            continue
        if new_pts and min(np.linalg.norm(np.asarray(p) - c)
                           for p in new_pts) < merge_tol:
            continue
        new_pts.append(tuple(c))

    if not new_pts:
        vm = np.zeros(N0, dtype=bool)
        return xy, edge_index, edge_class, edge_thickness, vm

    xy2 = np.vstack([xy, np.array(new_pts)])
    src = list(edge_index[0]) if edge_index.size else []
    dst = list(edge_index[1]) if edge_index.size else []
    ecl = list(edge_class)
    eth = list(edge_thickness)

    all_pts = xy2
    for j, p in enumerate(new_pts):
        ni = N0 + j
        d = np.linalg.norm(all_pts - np.asarray(p), axis=1)
        d[ni] = np.inf
        order = np.argsort(d)
        linked = 0
        for t in order:
            if linked >= k_connect or d[t] > max_link_m:
                break
            src.append(ni)
            dst.append(int(t))
            ecl.append(ET_VIRTUAL)
            eth.append(0.0)
            linked += 1

    vm = np.zeros(xy2.shape[0], dtype=bool)
    vm[N0:] = True
    ei2 = np.array([src, dst], dtype=np.int64)
    return xy2, ei2, np.array(ecl, dtype=int), np.array(eth), vm


# =============================================================================
# 4. Model
# =============================================================================

if HAS_TORCH:

    class SFLayer(nn.Module):
        """GraphSAGE-Frame layer, bidirectional aggregation + edge features."""

        def __init__(self, d_in, d_out, d_e=D_EDGE):
            super().__init__()
            self.w_self = nn.Linear(d_in, d_out)
            self.w_neigh = nn.Linear(d_in, d_out)
            self.w_edge = nn.Linear(d_e, d_out)

        def forward(self, x, ei, ea):
            src, dst = ei
            # aggregate BOTH directions (nx graphs store one direction only)
            s2 = torch.cat([src, dst])
            d2 = torch.cat([dst, src])
            ea2 = torch.cat([ea, ea], dim=0)

            N = x.size(0)
            na = torch.zeros(N, x.size(1), device=x.device)
            ee = torch.zeros(N, self.w_edge.out_features, device=x.device)
            cnt = torch.zeros(N, 1, device=x.device)
            na.index_add_(0, d2, x[s2])
            cnt.index_add_(0, d2, torch.ones(s2.size(0), 1, device=x.device))
            cnt = cnt.clamp(min=1)
            na /= cnt
            ee.index_add_(0, d2, self.w_edge(ea2))
            ee /= cnt
            return F.relu(self.w_self(x) + self.w_neigh(na) + ee)

    class GNNEPMulti(nn.Module):
        """
        Multi-task GNN:
          edge head : shear-wall length ratios per edge  (E, 2)  in [0,1]
          node head : column probability per node        (N, 1)  in [0,1]

        LayerNorm (not BatchNorm) so single-graph batches and small
        out-of-distribution plans do not collapse the statistics.
        """

        def __init__(self, d_in=D_NODE, d_e=D_EDGE, d_h=32, p=0.1,
                     edge_out=3):
            # edge_out=3 (v2): [ratio_l, ratio_r, sw_logit].  The dedicated
            # classification logit removes the objective conflict of v1,
            # where BCE on the MEAN RATIO pulled SW edges toward 1.0 while
            # the L1 ratio targets (capped at 0.5+0.5) pulled them to 0.5 --
            # the head could only learn the all-positive base rate.
            # edge_out=2 keeps v1 checkpoint compatibility.
            super().__init__()
            self.edge_out = edge_out
            self.sf = nn.ModuleList([
                SFLayer(d_in, 16, d_e),
                SFLayer(16, 16, d_e),
                SFLayer(16, 32, d_e),
                SFLayer(32, 32, d_e),
                SFLayer(32, 32, d_e),
                SFLayer(32, d_h, d_e),
            ])
            self.ln1 = nn.LayerNorm(16)
            self.ln2 = nn.LayerNorm(32)
            self.drop = nn.Dropout(p)

            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * d_h + d_e, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.LayerNorm(32), nn.Dropout(p),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, edge_out),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(d_h + d_in, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.LayerNorm(16), nn.Dropout(p),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 1),          # logits; use BCEWithLogits
            )

        def forward(self, x, ei, ea):
            h = self.sf[0](x, ei, ea)
            h = self.sf[1](h, ei, ea)
            h = self.drop(self.ln1(h))
            h = self.sf[2](h, ei, ea)
            h = self.sf[3](h, ei, ea)
            h = self.drop(self.ln2(h))
            h = self.sf[4](h, ei, ea)
            h = self.sf[5](h, ei, ea)
            s, d = ei
            e_raw = self.edge_mlp(torch.cat([h[s], ea, h[d]], dim=-1))
            if self.edge_out >= 3:
                # sigmoid ratios + RAW sw logit (train with BCEWithLogits)
                edge_out = torch.cat(
                    [torch.sigmoid(e_raw[:, :2]), e_raw[:, 2:]], dim=-1)
            else:                       # v1 behaviour
                edge_out = torch.sigmoid(e_raw)
            node_logit = self.node_mlp(torch.cat([h, x], dim=-1))
            return edge_out, node_logit

    def save_checkpoint(model, path, *, node_threshold=0.5,
                        edge_threshold=0.5, meta=None):
        torch.save({
            "format": "gnn_ep_multi_v2",
            "state_dict": model.state_dict(),
            "d_in": D_NODE, "d_e": D_EDGE,
            "edge_out": getattr(model, "edge_out", 3),
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
            "meta": meta or {},
        }, path)

    MULTI_FORMATS = ("gnn_ep_multi_v1", "gnn_ep_multi_v2")

    def load_checkpoint(path, map_location="cpu"):
        """Returns (model, node_threshold, meta). meta includes
        edge_threshold. Raises ValueError for non-multi checkpoints."""
        ck = torch.load(path, map_location=map_location, weights_only=False)
        if not (isinstance(ck, dict) and ck.get("format") in MULTI_FORMATS):
            raise ValueError(f"{os.path.basename(path)} is not a "
                             "multi-task checkpoint")
        edge_out = int(ck.get("edge_out",
                              2 if ck["format"] == "gnn_ep_multi_v1" else 3))
        model = GNNEPMulti(d_in=ck["d_in"], d_e=ck["d_e"], edge_out=edge_out)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        meta = dict(ck.get("meta", {}))
        meta["edge_threshold"] = float(ck.get("edge_threshold", 0.5))
        meta["edge_out"] = edge_out
        return model, float(ck.get("node_threshold", 0.5)), meta


# =============================================================================
# 5. Spatial NMS for point predictions (numpy-only, no torch needed)
# =============================================================================

def nms_points(xy: np.ndarray, probs: np.ndarray, *, min_dist: float = 0.60):
    """
    Greedy non-maximum suppression for point predictions.

    Wall-junction nodes sit centimetres apart, so the node head fires on
    several nodes around each true column.  Keep only the highest-probability
    point within min_dist of each cluster.

    Returns indices (into xy) of the kept points, sorted by prob desc.
    """
    if len(xy) == 0:
        return []
    order = np.argsort(-np.asarray(probs))
    kept: list[int] = []
    for i in order:
        p = xy[i]
        if all(np.linalg.norm(p - xy[j]) >= min_dist for j in kept):
            kept.append(int(i))
    return kept


def match_points(pred_xy, label_xy, *, tol: float = 0.50):
    """
    Greedy distance-tolerant matching between predicted and label points.
    Returns (tp, fp, fn): a prediction within tol of an unmatched label is a
    true positive.  Layout-level metric -- exact node identity is irrelevant.
    """
    pred_xy = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    label_xy = np.asarray(label_xy, dtype=float).reshape(-1, 2)
    used = np.zeros(len(label_xy), dtype=bool)
    tp = 0
    for p in pred_xy:
        if len(label_xy) == 0:
            break
        d = np.linalg.norm(label_xy - p, axis=1)
        d[used] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= tol:
            used[j] = True
            tp += 1
    fp = len(pred_xy) - tp
    fn = len(label_xy) - tp
    return tp, fp, fn
