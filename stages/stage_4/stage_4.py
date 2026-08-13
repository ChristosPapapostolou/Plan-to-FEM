"""
Stage 4 - Structural Enrichment Pipeline
==========================================
Predicts engineering-grade structural properties from the architectural graph
produced by Stage 3 (graph construction) and the vectorized geometry from Stage 2.

Two complementary approaches (combined as per workflow.md):
  A) GNN-EP  (Zhao et al., 2023) - Edge classification: predicts shear wall
     layout as length-ratios per edge on the architectural graph.
     Architecture: graphSAGE-Frame layers + MLP head  (GNN-EP-4 recommended).
  B) RENE DNN (Pizarro & Massone, 2021) - Rectangle-level regression:
     30-feature vector per wall rectangle -> predicts eng. thickness & length.

Stage 3 -> 4 connection:
  Stage 3 produces a networkx Graph with wall/opening edges and intersection
  nodes.  The Stage4Adapter converts this into:
    (A) GNN-EP input tensors  (node feats, edge_index, edge feats)
    (B) RENE feature vectors  (30 floats per rectangle, from Stage 2 geometry)

Usage:
  python structural_enrichment.py                         # demo on Stage 2 output
  python structural_enrichment.py -i floor.json -o enriched.json

References:
  [1] Zhao et al., Adv. Eng. Informatics 55 (2023) 101886
  [2] Pizarro & Massone, Eng. Structures 241 (2021) 112377
  [3] Zhao et al., J. Build. Eng. 63 (2023) 105499
"""

from __future__ import annotations
import argparse, json, logging, math, os, uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)

# -- Optional deep-learning imports -------------------------------------------
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except Exception:   # ImportError, or broken CUDA install (ValueError/OSError)
    HAS_TORCH = False

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except Exception:
    HAS_PYG = False

# -- Shared model / featurization module (models/gnn_ep.py) -------------------
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "stages", "stage_3")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
try:
    from models import gnn_ep as GEP
    try:
        import structural_grid as SG                  # stages/stage_3 on path
    except ImportError:
        from stages.stage_3 import structural_grid as SG   # namespace package
    HAS_MULTI = True
except Exception as _imp_err:
    logger.warning("Multi-task GNN support unavailable (%s)", _imp_err)
    HAS_MULTI = False

# =============================================================================
# 1. Data types
# =============================================================================

class EdgeClass(IntEnum):
    """One-hot class indices for the Edge-PSW-DW scheme (Zhao et al.)."""
    PSW = 0           # Potential shear wall
    INDOOR_DW = 1     # Indoor door / window opening
    OUTDOOR_D = 2     # Outdoor door / gate

N_CLASSES = len(EdgeClass)         # 3
EDGE_FEAT_DIM = N_CLASSES + 5     # 3 one-hot + x_l, y_l, x_r, y_r, length

L_GRAPH_SCALE = 20_480.0  # mm - per Zhao et al. Eq (1)


@dataclass
class EnrichedRect:
    """A wall rectangle enriched with RENE features and (optionally) predictions."""
    rect_id: str = ""
    wall_id: str = ""
    # from Stage 2
    arch_thickness: float = 0.0
    arch_length: float = 0.0
    angle: float = 0.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p1_id: str = ""
    p2_id: str = ""
    # RENE
    rene_features: Optional[np.ndarray] = None
    # Predictions (filled by model inference)
    eng_thickness: float = 0.0
    eng_length: float = 0.0
    is_shear_wall: bool = False
    sw_ratio: float = 0.0


# =============================================================================
# 2. Stage 3 -> Stage 4 Adapter
# =============================================================================

class Stage4Adapter:
    """
    The glue between Stage 3 (graph construction) and Stage 4 (enrichment).

    Two responsibilities:
      A) to_gnn_input()  - pack the graph into GNN-EP arrays / PyG Data
      B) to_rene_input() - compute 30-feature vectors from Stage 2 geometry
    """

    def __init__(self, graph, stage2_floor, *, proximity_tol_m=0.10):
        self.graph = graph
        self.stage2 = stage2_floor
        self.proximity_tol = proximity_tol_m

    # -- A) GNN-EP input -------------------------------------------------------

    def to_gnn_input(self):
        nodes = sorted(self.graph.nodes())
        n2i = {n: i for i, n in enumerate(nodes)}

        nf = np.zeros((len(nodes), 2), dtype=np.float32)
        for n in nodes:
            d = self.graph.nodes[n]
            nf[n2i[n]] = [d.get("x", 0.0), d.get("y", 0.0)]

        edges = list(self.graph.edges(data=True))
        E = len(edges)
        ef = np.zeros((E, EDGE_FEAT_DIM), dtype=np.float32)
        ei = np.zeros((2, E), dtype=np.int64)
        eids = []

        for k, (u, v, d) in enumerate(edges):
            ei[0, k] = n2i[u]
            ei[1, k] = n2i[v]
            cls = int(d.get("edge_type", EdgeClass.PSW))
            ef[k, cls] = 1.0
            ef[k, N_CLASSES + 0] = d.get("x_left", 0.0)
            ef[k, N_CLASSES + 1] = d.get("y_left", 0.0)
            ef[k, N_CLASSES + 2] = d.get("x_right", 0.0)
            ef[k, N_CLASSES + 3] = d.get("y_right", 0.0)
            ef[k, N_CLASSES + 4] = d.get("length", 0.0)
            eids.append(d.get("id", f"e_{u}_{v}"))

        pyg = None
        if HAS_TORCH and HAS_PYG:
            pyg = PyGData(
                x=torch.tensor(nf),
                edge_index=torch.tensor(ei),
                edge_attr=torch.tensor(ef),
            )

        return dict(node_features=nf, edge_index=ei, edge_features=ef,
                     edge_ids=eids, pyg_data=pyg)

    # -- A2) Multi-task GNN input (shared featurization, virtual candidates) --

    def to_multi_gnn_input(self):
        """
        Build inputs for GNNEPMulti (models/gnn_ep.py):
        rich node features, per-graph normalization, and virtual candidate
        nodes at structural-grid intersections so the model can place columns
        in open space.
        """
        if not HAS_MULTI:
            return None
        G = self.graph
        node_ids = sorted(G.nodes())
        n2i = {n: i for i, n in enumerate(node_ids)}
        xy_m = np.array([[G.nodes[n].get("x_m", G.nodes[n].get("x", 0.0)),
                          G.nodes[n].get("y_m", G.nodes[n].get("y", 0.0))]
                         for n in node_ids], dtype=np.float64)

        edges = list(G.edges(data=True))
        E = len(edges)
        ei = np.zeros((2, E), dtype=np.int64)
        ecl = np.zeros(E, dtype=int)
        eth = np.zeros(E)
        eids = []
        for k, (u, v, d) in enumerate(edges):
            ei[0, k], ei[1, k] = n2i[u], n2i[v]
            ecl[k] = min(int(d.get("edge_type", 0)), GEP.ET_OUTDOOR_D)
            eth[k] = float(d.get("thickness_m", 0.0))
            eids.append(d.get("id", f"e_{u}_{v}"))

        # grid candidates from the wall geometry
        segs = SG.segments_from_floor(self.stage2)
        candidates = []
        if segs:
            try:
                res = SG.place_columns(segs, pier_end_columns=False)
                candidates = res.candidates
            except Exception:
                candidates = []

        xy2, ei2, ecl2, eth2, vmask = GEP.add_virtual_candidates(
            xy_m, ei, ecl, eth, candidates)
        nf, center, scale = GEP.build_node_features(xy2, ei2, eth2, ecl2, vmask)
        ef = GEP.build_edge_features(xy2, ei2, ecl2, center, scale)
        return dict(node_features=nf, edge_index=ei2, edge_features=ef,
                    xy_m=xy2, virtual_mask=vmask, edge_ids=eids,
                    n_real_edges=E, edge_class=ecl2)

    # -- B) RENE 30-feature input -----------------------------------------------

    def to_rene_input(self, seismic_zone=2, soil_type=2, floors_above_below=None):
        calc = RENEFeatureCalculator(seismic_zone, soil_type,
                                     proximity_tol=self.proximity_tol)
        return calc.compute(self.stage2, floors_above_below)


# =============================================================================
# 3. RENE Feature Calculator  (30 features, Table 2 in Pizarro & Massone 2021)
# =============================================================================

class RENEFeatureCalculator:
    """Compute the 30-feature RENE vector for every valid wall rectangle."""

    def __init__(self, seismic_zone, soil_type, *, proximity_tol=0.10):
        self.seismic_zone = seismic_zone
        self.soil_type = soil_type
        self.prox_tol = proximity_tol

    def compute(self, floor, floors_ab=None):
        recs = self._collect_rects(floor)
        if not recs:
            return []
        fp = self._floor_props(recs, floor)
        out = []
        for r in recs:
            v = self._features(r, recs, fp, floor, floors_ab)
            er = EnrichedRect(
                rect_id=r["rid"], wall_id=r["wid"],
                arch_thickness=r["t"], arch_length=r["L"], angle=r["a"],
                center=r["c"], p1=r["p1"], p2=r["p2"],
                p1_id=r["p1_id"], p2_id=r["p2_id"],
                rene_features=v,
            )
            out.append(er)
        logger.info("RENE: %d rects -> 30-feature vectors", len(out))
        return out

    def _collect_rects(self, floor):
        recs = []
        for w in floor.get("walls", []):
            if w.get("rejected"):
                continue
            pts = {p["id"]: np.array([p["x"], p["y"]]) for p in w.get("points", [])}
            for r in w.get("rects", []):
                p1 = pts.get(r["p1_id"])
                p2 = pts.get(r["p2_id"])
                if p1 is None or p2 is None:
                    continue
                recs.append(dict(
                    rid=r["id"], wid=w["wall_id"],
                    p1=p1, p2=p2, c=(p1 + p2) / 2,
                    t=r["thickness_m"], L=r["length_m"], a=r["angle_rad"],
                    area=r["thickness_m"] * r["length_m"],
                    p1_id=r["p1_id"], p2_id=r["p2_id"],
                ))
        return recs

    def _floor_props(self, recs, floor):
        areas = np.array([r["area"] for r in recs])
        centers = np.vstack([r["c"] for r in recs])
        total = areas.sum()
        mc = np.average(centers, axis=0, weights=areas) if total > 1e-9 else centers.mean(0)
        all_pts = np.vstack([np.vstack([r["p1"], r["p2"]]) for r in recs])
        gc = all_pts.mean(0)
        lo, hi = all_pts.min(0), all_pts.max(0)
        bw = hi - lo
        bw[bw < 1e-6] = 1e-6
        ar = bw[0] / bw[1]
        slab = bw[0] * bw[1]

        angles = np.array([r["a"] for r in recs])
        mx = np.abs(angles - np.pi / 2) < np.pi / 4
        my = ~mx
        anx = areas[mx].sum() / slab if mx.any() else 0.0
        any_ = areas[my].sum() / slab if my.any() else 0.0

        ipx = bw[0] * bw[1]**3 / 12
        ipy = bw[0]**3 * bw[1] / 12
        inx, iny = 0.0, 0.0
        for r in recs:
            L, t, a = r["L"], r["t"], r["a"]
            ix, iy = L * t**3 / 12, L**3 * t / 12
            inx += (ix + iy) / 2 + (ix - iy) / 2 * np.cos(2 * a)
            iny += (ix + iy) / 2 - (ix - iy) / 2 * np.cos(2 * a)

        return dict(
            mc=mc, gc=gc, lo=lo, hi=hi, bw=bw, ar=ar, slab=slab,
            anx=anx, any=any_,
            inx=inx / ipx if ipx > 1e-12 else 0,
            iny=iny / ipy if ipy > 1e-12 else 0,
            fh=floor.get("floor_height_m", 2.80),
            st=floor.get("slab_thickness_m", 0.15),
            mc_gc=np.abs(mc - gc),
        )

    def _features(self, r, recs, fp, floor, floors_ab):
        f = np.zeros(30)
        # Geometric (16 values, indices 0-15)
        f[0] = r["t"]                                    # RectThickness
        f[1] = r["L"]                                    # RectLength
        f[2] = r["a"]                                    # RectAngle [0,pi]
        d = np.abs(r["c"] - fp["mc"])
        f[3], f[4] = d[0], d[1]                          # RectFloorMCDist x,y
        sw = [x for x in recs if x["wid"] == r["wid"]]
        wa = np.array([x["area"] for x in sw])
        wc = np.vstack([x["c"] for x in sw])
        wmc = np.average(wc, axis=0, weights=wa) if wa.sum() > 1e-9 else wc.mean(0)
        dw = np.abs(wmc - fp["mc"])
        f[5], f[6] = dw[0], dw[1]                        # RectWallMCDist x,y
        f[7] = fp["st"]                                   # SlabThickness
        f[8], f[9] = fp["mc_gc"][0], fp["mc_gc"][1]       # FloorMCDist x,y
        f[10], f[11] = fp["anx"], fp["any"]               # FloorAreaNorm x,y
        f[12] = max(fp["bw"][0], fp["bw"][1])             # FloorBoundWidth
        f[13], f[14] = fp["inx"], fp["iny"]               # FloorInertiaNorm x,y
        f[15] = fp["ar"]                                   # FloorAspectRatio

        # Topological (12 values, indices 16-27)
        f[16] = self._triangulation_area(r, recs)
        proj = self._projection_intersection(r, floors_ab)
        f[17], f[18] = proj
        tp = self._thickness_projection(r, floors_ab)
        f[19], f[20] = tp
        ad = self._axis_displacement(r, recs)
        f[21], f[22] = ad
        rd = self._relative_distance(r, recs)
        f[23], f[24] = rd
        pc = self._point_connectivity(r, recs)
        f[25], f[26] = pc

        # FloorHeight + Other (indices 27-29)
        f[27] = fp["fh"]
        f[28] = self.seismic_zone
        f[29] = self.soil_type
        return f

    # -- topological helpers ---------------------------------------------------

    def _triangulation_area(self, r, recs):
        if len(recs) < 3:
            return 0.0
        pts = np.vstack([x["c"] for x in recs])
        try:
            tri = Delaunay(pts)
        except Exception:
            return 0.0
        idx = next((i for i, x in enumerate(recs) if x["rid"] == r["rid"]), None)
        if idx is None:
            return 0.0
        total = 0.0
        for s in tri.simplices:
            if idx in s:
                v1 = pts[s[1]] - pts[s[0]]
                v2 = pts[s[2]] - pts[s[0]]
                total += 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        return total

    def _projection_intersection(self, r, floors_ab):
        if not floors_ab:
            return (0.0, 0.0)
        res = []
        for key in ("below", "above"):
            adj = floors_ab.get(key)
            if adj is None:
                res.append(0.0); continue
            arecs = self._collect_rects(adj)
            if not arecs:
                res.append(0.0); continue
            best = max(self._overlap(r, ar) for ar in arecs)
            res.append(best)
        return tuple(res)

    def _overlap(self, r1, r2):
        for x in (r1, r2):
            if "_aabb" not in x:
                hl, ht = x["L"] / 2, x["t"] / 2
                c = x["c"]
                x["_aabb"] = (c[0] - hl, c[1] - ht, c[0] + hl, c[1] + ht)
        a, b = r1["_aabb"], r2["_aabb"]
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = r1["t"] * r1["L"]
        return (ix * iy) / area if area > 1e-9 else 0.0

    def _thickness_projection(self, r, floors_ab):
        if not floors_ab:
            return (1.0, 1.0)
        res = []
        for key in ("below", "above"):
            adj = floors_ab.get(key)
            if adj is None:
                res.append(1.0); continue
            arecs = self._collect_rects(adj)
            if not arecs:
                res.append(0.0); continue
            dists = [np.linalg.norm(r["c"] - ar["c"]) for ar in arecs]
            nearest = arecs[int(np.argmin(dists))]
            res.append(nearest["t"] / r["t"] if r["t"] > 1e-6 else 0)
        return tuple(res)

    def _axis_displacement(self, r, recs):
        same = [x for x in recs if x["wid"] == r["wid"] and x["rid"] != r["rid"]]
        if not same:
            return (0.0, 0.0)
        connected = []
        for s in same:
            if self._pts_connected(r, s):
                axis = r["p2"] - r["p1"]
                al = np.linalg.norm(axis)
                if al < 1e-9:
                    continue
                au = axis / al
                v = s["c"] - r["p1"]
                d = abs(v[0] * au[1] - v[1] * au[0])
                connected.append(d)
        if len(connected) >= 2:
            return (connected[0], connected[1])
        elif len(connected) == 1:
            return (connected[0], 0.0)
        return (0.0, 0.0)

    def _relative_distance(self, r, recs):
        others = [x for x in recs if x["wid"] != r["wid"]]
        if not others:
            return (1.0, 1.0)
        same_ax, diff_ax = [], []
        for o in others:
            d = np.linalg.norm(r["c"] - o["c"])
            da = abs(r["a"] - o["a"])
            (same_ax if da < np.pi / 4 or da > 3 * np.pi / 4 else diff_ax).append(d)
        return (min(same_ax) if same_ax else 1.0,
                min(diff_ax) if diff_ax else 1.0)

    def _point_connectivity(self, r, recs):
        """
        Number of other rects connected at each endpoint.
        Uses BOTH exact point-ID match AND proximity (< prox_tol).
        Key fix: Stage 2 T-junction points may be close but not ID-merged.
        """
        def _count(pt, pid):
            c = 0
            for o in recs:
                if o["rid"] == r["rid"]:
                    continue
                if o["p1_id"] == pid or o["p2_id"] == pid:
                    c += 1; continue
                if (np.linalg.norm(pt - o["p1"]) < self.prox_tol or
                    np.linalg.norm(pt - o["p2"]) < self.prox_tol):
                    c += 1
            return c
        return (_count(r["p1"], r["p1_id"]), _count(r["p2"], r["p2_id"]))

    def _pts_connected(self, r1, r2):
        for pid1, pt1 in [(r1["p1_id"], r1["p1"]), (r1["p2_id"], r1["p2"])]:
            for pid2, pt2 in [(r2["p1_id"], r2["p1"]), (r2["p2_id"], r2["p2"])]:
                if pid1 == pid2:
                    return True
                if np.linalg.norm(pt1 - pt2) < self.prox_tol:
                    return True
        return False


# =============================================================================
# 4. Stage 3 Graph Builder (delegates to graph_construction module)
# =============================================================================

def build_graph_from_stage2(floor, proximity_tol_m=0.10, openings=None):
    """
    Build a Stage 3 graph from Stage 2 output.

    Delegates to stage_3.build_graph() (with gap detection) if available,
    otherwise falls back to a minimal inline builder (walls-only PSW).
    """
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _s3_dir = os.path.join(_here, "..", "stage_3")
    if _s3_dir not in _sys.path:
        _sys.path.insert(0, _s3_dir)
    try:
        from stage_3 import build_graph, GraphVariant as GV
        return build_graph(
            floor,
            openings=openings,
            variant=GV.EDGE_PSW_DW,
            proximity_tol_m=proximity_tol_m,
            auto_detect_gaps=True,
            # FIXED classification (matches training / unified_dataset.py):
            # "auto" adapts the PSW threshold per plan and silently drops most
            # walls as partition walls, starving both heads of graph context.
            classify_mode="fixed",
            psw_min_thickness=0.06,
        )
    except ImportError:
        pass
    try:
        from graph_construction import build_graph, GraphVariant
        return build_graph(
            floor,
            openings=openings,
            variant=GraphVariant.EDGE_PSW_DW,
            proximity_tol_m=proximity_tol_m,
        )
    except ImportError:
        logger.info("graph_construction module not found, using inline builder")
        return _build_graph_inline(floor, proximity_tol_m)


def _build_graph_inline(floor, proximity_tol_m=0.10):
    """Minimal inline graph builder (walls-only Edge-PSW) as fallback."""
    G = nx.Graph()
    walls = [w for w in floor.get("walls", []) if not w.get("rejected")]

    raw_pts = {}
    for w in walls:
        for p in w.get("points", []):
            raw_pts[p["id"]] = np.array([p["x"], p["y"]])

    ids = list(raw_pts.keys())
    canonical = {pid: pid for pid in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if canonical[ids[j]] != ids[j]:
                continue
            if np.linalg.norm(raw_pts[ids[i]] - raw_pts[ids[j]]) < proximity_tol_m:
                canonical[ids[j]] = canonical[ids[i]]

    merged = {}
    for pid in ids:
        cid = canonical[pid]
        merged.setdefault(cid, []).append(raw_pts[pid])
    merged_pts = {cid: np.mean(vs, axis=0) for cid, vs in merged.items()}

    coords = np.vstack(list(merged_pts.values()))
    center_m = coords.mean(axis=0)

    def _norm(xy_m):
        return ((xy_m - center_m) * 1000.0) / L_GRAPH_SCALE

    for cid, xy in merged_pts.items():
        xn, yn = _norm(xy)
        G.add_node(cid, x=float(xn), y=float(yn),
                   x_m=float(xy[0]), y_m=float(xy[1]))

    for w in walls:
        pts_map = {p["id"]: np.array([p["x"], p["y"]]) for p in w.get("points", [])}
        for r in w.get("rects", []):
            p1_id = canonical.get(r["p1_id"], r["p1_id"])
            p2_id = canonical.get(r["p2_id"], r["p2_id"])
            if p1_id == p2_id:
                continue
            p1 = merged_pts.get(p1_id, pts_map.get(r["p1_id"], np.zeros(2)))
            p2 = merged_pts.get(p2_id, pts_map.get(r["p2_id"], np.zeros(2)))
            n1, n2 = _norm(p1), _norm(p2)
            ln = r["length_m"] * 1000.0 / L_GRAPH_SCALE
            G.add_edge(p1_id, p2_id,
                       id=r["id"],
                       edge_type=int(EdgeClass.PSW),
                       x_left=float(n1[0]), y_left=float(n1[1]),
                       x_right=float(n2[0]), y_right=float(n2[1]),
                       length=float(ln),
                       thickness_m=r["thickness_m"],
                       length_m=r["length_m"],
                       angle_rad=r["angle_rad"])

    logger.info("Graph (inline): %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())
    return G


# =============================================================================
# 5. GNN-EP-4 Model  (Zhao et al., 2023, Fig. 5b)
# =============================================================================

if HAS_TORCH:

    class SFLayer(nn.Module):
        """graphSAGE-Frame layer: mean-aggregate neighbours + edge features."""
        def __init__(self, d_in, d_out, d_e=EDGE_FEAT_DIM):
            super().__init__()
            self.w_self = nn.Linear(d_in, d_out)
            self.w_neigh = nn.Linear(d_in, d_out)
            self.w_edge = nn.Linear(d_e, d_out)

        def forward(self, x, ei, ea):
            src, dst = ei
            N = x.size(0)
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
        GNN-EP-4 (best model from Zhao et al., 2023).
        6 SF layers with BatchNorm + Dropout, then MLP head.
        Output: 2 shear-wall length ratios per edge in [0, 1].
        """
        def __init__(self, d_in=2, d_e=EDGE_FEAT_DIM, d_h=32, p=0.1):
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
            self.drop = nn.Dropout(p)
            mlp_in = 2 * d_h + d_e
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.BatchNorm1d(32), nn.Dropout(p),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Dropout(p),
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
            return self.mlp(torch.cat([h[s], ea, h[d]], dim=-1))

    class RENERegressor(nn.Module):
        """DNN: 30 RENE features -> (eng_thickness, eng_length)."""
        def __init__(self, d_in=30, p=0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(p),
                nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(p),
                nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 2),
            )

        def forward(self, x):
            return self.net(x)


# =============================================================================
# 5b. Engineering rules  (deterministic thickness + shear-wall selection)
# =============================================================================
# These replace two broken ML mechanisms:
#   - the RENE thickness regressor, which is never trained (constant ~0.20 m
#     output, negatively correlated with the real wall thickness), and
#   - the GNN shear-wall gate, which is out-of-distribution on small plans and
#     collapses to near-uniform ratios.
# The GNN score is retained only as a soft prior to admit borderline interior
# walls; the GNN is no longer the decision-maker.

# Standard concrete thickness increments (m) and minimums.
_THK_SNAP_M = 0.025
_SW_MIN_THK_M = 0.15      # code-style minimum for a structural shear wall
_WALL_MIN_THK_M = 0.08    # minimum for a non-structural wall panel


def engineering_thickness(arch_t: float, is_sw: bool) -> float:
    """
    Deterministic engineering thickness from the architectural thickness.

    A selected shear wall is thickened to at least the structural minimum; every
    thickness is snapped to a standard 25 mm increment so sections are clean and
    physical (and de-duplicate).  Monotonic in arch_t, so it tracks the drawing
    instead of the untrained network's constant output.
    """
    base = max(float(arch_t), _SW_MIN_THK_M if is_sw else _WALL_MIN_THK_M)
    return max(_THK_SNAP_M, round(base / _THK_SNAP_M) * _THK_SNAP_M)


class ShearWallSelector:
    """
    Engineering-rule shear-wall selection (replaces the collapsed/OOD GNN gate).

    A wall rect becomes a shear wall when it is long enough to act as a lateral
    pier and is either on the building perimeter (exterior) or a long interior
    (core) wall.  Both plan directions (X and Y) must contain shear walls so the
    structure resists lateral load bi-directionally.  The GNN score is used only
    as a soft prior to admit borderline interior walls.
    """

    def __init__(self, *, min_len_m: float = 1.0, long_frac: float = 0.4,
                 min_per_dir: int = 2, ext_tol_m: float = 0.35,
                 gnn_prior: float = 0.50, max_ecc_frac: float = 0.10,
                 target_sw_ratio: float = 0.02, sw_min_t: float = 0.15):
        # gnn_prior 0.50: since the v2 checkpoint the gnn_score is a
        # CALIBRATED SW probability (dedicated logit), not the old inflated
        # mean-ratio score -- 0.25 would admit nearly everything.
        self.min_len = min_len_m
        self.long_frac = long_frac      # keep walls >= frac of the longest in-dir
        self.min_per_dir = min_per_dir  # min shear walls per direction
        self.ext_tol = ext_tol_m
        self.gnn_prior = gnn_prior
        # eccentricity balancing (roadmap step 5): keep the length-weighted
        # SW centroid within max_ecc_frac of the plan dimension by promoting
        # additional piers on the weak side.
        self.max_ecc_frac = max_ecc_frac
        # SW quantity budget: select piers per direction only until the SW
        # area ratio (sum t*L / plan area) reaches target_sw_ratio (~2% is
        # standard low-rise practice).  Without this the length threshold
        # admits EVERY wall >= min_len -- e.g. 24.9 m of SW on an 11x4.8 m
        # plan (~7% ratio, wildly over-designed).
        self.target_sw_ratio = target_sw_ratio
        self.sw_min_t = sw_min_t

    def select(self, floor: dict, epreds: dict) -> dict:
        rects = self._collect(floor)
        if not rects:
            return dict(sw_rect_ids=set(), thickness={}, n_candidates=0,
                        n_x=0, n_y=0, n_exterior=0, eccentricity_m=0.0)
        self._mark_exterior(rects)
        for r in rects:
            r["gnn"] = float(epreds.get(r["id"], {}).get("gnn_score", 0.0))
        plan_c = np.mean([r["c"] for r in rects], axis=0)

        # Group fragmented collinear rects into continuous PIERS so a wall the
        # vectorizer chopped into several short segments is evaluated as the
        # single long pier it physically is.  Selection happens at pier level;
        # selected piers expand back to all their member rect ids.
        runs = self._group_runs(rects)
        cand = [g for g in runs if g["len"] >= self.min_len]
        selected_runs: dict[int, dict] = {}

        # Per direction keep only piers that are genuinely long (absolute min_len
        # AND a share of the longest pier in that direction) plus any the GNN
        # prior strongly favours.  Exterior position is NO LONGER an automatic
        # qualifier (facade walls are usually infill); it is only a tie-breaker
        # when balancing both sides.  A minimum count per direction is still
        # guaranteed for bi-directional resistance.
        # plan area for the SW quantity budget
        _pts = np.vstack([np.vstack([r["p1"], r["p2"]]) for r in rects])
        _sz = np.maximum(_pts.max(axis=0) - _pts.min(axis=0), 1e-6)
        plan_area = float(_sz[0] * _sz[1])

        for orient in ("H", "V"):
            dc = [g for g in cand if g["orient"] == orient]
            if not dc:
                continue
            l_max = max(g["len"] for g in dc)
            thr = max(self.min_len, self.long_frac * l_max)
            eligible = [g for g in dc
                        if (g["len"] >= thr or g["gnn"] >= self.gnn_prior)]
            # budgeted greedy: longest piers first, stop once the per-
            # direction SW area ratio target is met (stability rules below
            # may still add more).
            budget = self.target_sw_ratio * plan_area
            pick, used = [], 0.0
            for g in sorted(eligible, key=lambda g: -g["len"]):
                if used >= budget and len(pick) >= self.min_per_dir:
                    break
                pick.append(g)
                used += max(g["t"], self.sw_min_t) * g["len"]
            if len(pick) < self.min_per_dir:
                pick = sorted(dc, key=lambda g: -g["len"])[:self.min_per_dir]
            logger.info("  SW budget [%s]: %d/%d piers, sum t*L=%.2f m^2 "
                        "(target %.2f, ratio %.1f%%)", orient, len(pick),
                        len(eligible), used, budget,
                        100 * used / max(plan_area, 1e-6))

            axis = 1 if orient == "H" else 0

            # torsional stability needs >= 2 DISTINCT lateral lines, not just
            # 2 piers (which may be collinear).  If all picks share one band,
            # add the longest pier from a different band.
            bands = {round(g["c"][axis] / 0.4) for g in pick}
            if len(bands) < 2:
                others = [g for g in dc
                          if round(g["c"][axis] / 0.4) not in bands]
                if others:
                    pick.append(max(others, key=lambda g: g["len"]))

            ref = plan_c[axis]
            picked_ids = {g["run"] for g in pick}
            present_sides = {int(np.sign(g["c"][axis] - ref)) for g in pick}
            for side in (-1, 1):
                if side not in present_sides:
                    opp = [g for g in dc
                           if int(np.sign(g["c"][axis] - ref)) == side
                           and g["run"] not in picked_ids]
                    if opp:
                        pick.append(max(opp, key=lambda g: g["len"]))
            for g in pick:
                selected_runs[g["run"]] = g

        # -- eccentricity balancing (ACT on it, not just report it) ---------
        # Direction-aware stiffness centroid (k ~ t*L^3), same measure as the
        # step-5 sanity checker: H piers resist X (their y-position matters),
        # V piers resist Y (x-position).  While either axis is off by more
        # than max_ecc_frac of the plan dimension, promote the longest
        # unselected pier of the RESISTING orientation on the weak side.
        all_pts = np.vstack([np.vstack([r["p1"], r["p2"]]) for r in rects])
        plan_lo, plan_hi = all_pts.min(axis=0), all_pts.max(axis=0)
        plan_size = np.maximum(plan_hi - plan_lo, 1e-6)
        plan_mid = (plan_lo + plan_hi) / 2.0

        def _stiff_ecc():
            # returns dict axis_index -> (frac, orient_that_resists)
            out = {}
            for orient, ax in (("H", 1), ("V", 0)):
                sv = [g for g in selected_runs.values()
                      if g["orient"] == orient]
                if not sv:
                    continue
                k = np.array([max(g["t"], 0.05) * g["len"] ** 3 for g in sv])
                pos = np.array([g["c"][ax] for g in sv])
                cr = float(np.average(pos, weights=k))
                out[ax] = ((cr - plan_mid[ax]) / plan_size[ax], orient)
            return out

        n_balanced = 0
        residual = None
        for _ in range(6):
            ecc = _stiff_ecc()
            if not ecc:
                break
            ax, (frac, orient) = max(ecc.items(), key=lambda kv: abs(kv[1][0]))
            residual = frac
            if abs(frac) <= self.max_ecc_frac:
                break
            weak = -np.sign(frac)

            def _cands(lmin):
                return [g for g in runs
                        if g["run"] not in selected_runs
                        and g["orient"] == orient
                        and g["len"] >= lmin
                        and np.sign(g["c"][ax] - plan_mid[ax]) == weak]

            # balancing piers may be shorter than regular selection allows --
            # a 1.2 m wall on the weak side beats an unbalanced building.
            cands = _cands(self.min_len) or _cands(1.0)
            if not cands:
                break
            g = max(cands, key=lambda g: g["len"])
            selected_runs[g["run"]] = g
            n_balanced += 1
        if n_balanced or (residual is not None
                          and abs(residual) > self.max_ecc_frac):
            logger.info("Eccentricity balancing: promoted %d pier(s); "
                        "residual stiffness ecc = %.1f%% of plan dim%s",
                        n_balanced, 100 * abs(residual or 0.0),
                        "" if residual is None
                        or abs(residual) <= self.max_ecc_frac
                        else "  (still above %.0f%% -- no more weak-side "
                             "piers; consider columns / thicker walls)"
                             % (100 * self.max_ecc_frac))

        selected: dict[str, dict] = {}
        for g in selected_runs.values():
            for r in g["members"]:
                selected[r["id"]] = r

        sw_ids = set(selected)
        thickness = {rid: engineering_thickness(r["t"], True)
                     for rid, r in selected.items()}
        rect_pier = {}
        for g in selected_runs.values():
            for r in g["members"]:
                rect_pier[r["id"]] = (g["run"], round(g["len"], 4))

        if selected_runs:
            sv = list(selected_runs.values())
            w = np.array([g["len"] for g in sv])
            cs = np.vstack([g["c"] for g in sv])
            ecc = float(np.linalg.norm(np.average(cs, axis=0, weights=w) - plan_c))
        else:
            ecc = 0.0

        return dict(
            sw_rect_ids=sw_ids, thickness=thickness,
            n_candidates=len(cand),
            n_x=sum(1 for g in selected_runs.values() if g["orient"] == "H"),
            n_y=sum(1 for g in selected_runs.values() if g["orient"] == "V"),
            n_exterior=sum(1 for g in selected_runs.values() if g["ext"]),
            eccentricity_m=round(ecc, 3), rect_pier=rect_pier)

    def _group_runs(self, rects, *, align_tol_m: float = 0.15,
                    gap_tol_m: float = 0.10):
        """
        Merge collinear, near-adjacent rects of the same orientation into piers.

        Two rects join the same pier when they share orientation, lie on the
        same line (perpendicular offset < align_tol_m) and are end-to-end within
        gap_tol_m.  The pier length is the extent spanned along its axis, so a
        wall split into N short segments is restored to one long pier.
        """
        runs = []
        for orient in ("H", "V"):
            axis = 0 if orient == "H" else 1
            perp = 1 - axis
            items = [r for r in rects if r["orient"] == orient]
            items.sort(key=lambda r: (round(r["c"][perp] / align_tol_m),
                                      r["c"][axis]))
            used = [False] * len(items)
            for i, r in enumerate(items):
                if used[i]:
                    continue
                members = [r]
                used[i] = True
                lo = min(r["p1"][axis], r["p2"][axis])
                hi = max(r["p1"][axis], r["p2"][axis])
                band = r["c"][perp]
                changed = True
                while changed:
                    changed = False
                    for j, s in enumerate(items):
                        if used[j]:
                            continue
                        if abs(s["c"][perp] - band) > align_tol_m:
                            continue
                        s_lo = min(s["p1"][axis], s["p2"][axis])
                        s_hi = max(s["p1"][axis], s["p2"][axis])
                        if s_lo <= hi + gap_tol_m and s_hi >= lo - gap_tol_m:
                            members.append(s)
                            used[j] = True
                            lo = min(lo, s_lo)
                            hi = max(hi, s_hi)
                            changed = True
                c = np.mean([m["c"] for m in members], axis=0)
                runs.append(dict(
                    run=len(runs), orient=orient, members=members,
                    len=float(hi - lo), c=c,
                    t=max(m["t"] for m in members),
                    ext=any(m["ext"] for m in members),
                    gnn=max((m.get("gnn", 0.0) for m in members), default=0.0)))
        return runs

    def _collect(self, floor):
        out = []
        for w in floor.get("walls", []):
            if w.get("rejected"):
                continue
            pts = {p["id"]: np.array([p["x"], p["y"]], float)
                   for p in w.get("points", [])}
            for r in w.get("rects", []):
                p1, p2 = pts.get(r["p1_id"]), pts.get(r["p2_id"])
                if p1 is None or p2 is None:
                    continue
                dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
                out.append(dict(
                    id=r["id"], wall_id=w.get("wall_id", ""),
                    p1=p1, p2=p2, c=(p1 + p2) / 2,
                    t=r.get("thickness_m", 0.0),
                    len=r.get("length_m", float(np.hypot(dx, dy))),
                    orient="H" if dx >= dy else "V"))
        return out

    def _mark_exterior(self, rects):
        from scipy.spatial import ConvexHull
        endpts = np.vstack([np.vstack([r["p1"], r["p2"]]) for r in rects])
        try:
            hull = ConvexHull(endpts)
            hull_pts = hull.points[hull.vertices]
        except Exception:
            for r in rects:
                r["ext"] = False
            return
        for r in rects:
            r["ext"] = self._dist_to_hull(r["c"], hull_pts) < self.ext_tol

    @staticmethod
    def _dist_to_hull(pt, hull_pts):
        n = len(hull_pts)
        best = float("inf")
        for i in range(n):
            a, b = hull_pts[i], hull_pts[(i + 1) % n]
            ab, ap = b - a, pt - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
            best = min(best, float(np.linalg.norm(pt - (a + t * ab))))
        return best


# =============================================================================
# 6. Structural Enrichment Pipeline  (orchestrator)
# =============================================================================

class StructuralEnrichmentPipeline:
    """
    Orchestrator that:
      1. Takes Stage 3 graph + Stage 2 geometry
      2. Runs GNN-EP  -> shear wall layout (edge classification)
      3. Runs RENE DNN -> engineering dimensions (rect regression)
      4. Merges everything into an enriched JSON for Stage 5
    """

    def __init__(self, *, sw_threshold=0.05,
                 seismic_zone=2, soil_type=2, proximity_tol=0.10,
                 sw_min_len_m=1.5, sw_long_len_m=2.0, sw_gnn_prior=0.50,
                 sw_target_ratio=0.02):
        self.sw_thresh = sw_threshold
        self.sz = seismic_zone
        self.st = soil_type
        self.prox = proximity_tol
        self.sw_min_len = sw_min_len_m
        self.sw_long_len = sw_long_len_m
        self.sw_gnn_prior = sw_gnn_prior
        self.sw_target_ratio = sw_target_ratio
        self.gnn = GNNEP4() if HAS_TORCH else None
        self.rene = RENERegressor() if HAS_TORCH else None
        self.rene_loaded = False   # the RENE regressor is garbage until trained
        self.multi = None          # GNNEPMulti (edge SW + node columns)
        self.multi_thresh = 0.5

    def load_gnn(self, path):
        """
        Load model weights.  Prefers the multi-task checkpoint format
        (gnn_ep_multi_v1: SW edges + column nodes); falls back to the legacy
        GNN-EP-4 state dict.
        """
        assert HAS_TORCH
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # any multi-task format (v1, v2, ...) -- the shared loader picks the
        # right architecture; never feed these dicts to the legacy model
        if isinstance(ck, dict) and str(ck.get("format", "")).startswith(
                "gnn_ep_multi"):
            if not HAS_MULTI:
                logger.error(
                    "%s is a MULTI-TASK checkpoint but models/gnn_ep.py "
                    "could not be imported -- GNN disabled, engineering "
                    "rules will handle SW + columns.", os.path.basename(path))
                self.gnn = None
                return
            self.multi, self.multi_thresh, meta = GEP.load_checkpoint(path)
            logger.info("Loaded MULTI-TASK GNN (thresh=%.2f, epoch=%s)",
                        self.multi_thresh, meta.get("epoch"))
            return
        # legacy GNN-EP-4 plain state dict
        self.gnn.load_state_dict(ck)
        self.gnn.eval()

    def load_rene(self, path):
        assert HAS_TORCH
        self.rene.load_state_dict(torch.load(path, map_location="cpu"))
        self.rene.eval()
        self.rene_loaded = True

    def run(self, graph, floor, *, floors_ab=None, predict=True):
        adapter = Stage4Adapter(graph, floor, proximity_tol_m=self.prox)

        # -- A: GNN edge ratios (SOFT PRIOR ONLY — not the SW decision) --
        gi = adapter.to_gnn_input()
        epreds = {}
        gnn_columns = None   # node-level column predictions (multi model)

        # Build the multi-task GNN input regardless of model availability so
        # the graph the GNN actually sees can be visualized (torch-free).
        mi = adapter.to_multi_gnn_input() if HAS_MULTI else None

        if predict and HAS_TORCH and self.multi is not None and HAS_MULTI:
            if mi is not None:
                with torch.no_grad():
                    e_out, n_logit = self.multi(
                        torch.tensor(mi["node_features"]),
                        torch.tensor(mi["edge_index"]),
                        torch.tensor(mi["edge_features"]))
                    e_out = e_out.numpy()
                    n_prob = torch.sigmoid(n_logit.squeeze(-1)).numpy()
                # edge predictions (real edges only, in edge_ids order).
                # v2 checkpoints carry a dedicated SW logit in column 2;
                # v1 falls back to the mean-ratio score.
                has_logit = e_out.shape[1] >= 3
                for i, eid in enumerate(mi["edge_ids"]):
                    rl, rr = float(e_out[i, 0]), float(e_out[i, 1])
                    if has_logit:
                        score = float(1.0 / (1.0 + np.exp(-e_out[i, 2])))
                    else:
                        score = (rl + rr) / 2
                    epreds[eid] = dict(
                        sw_ratio_left=round(rl, 4), sw_ratio_right=round(rr, 4),
                        is_psw=int(mi["edge_class"][i]) == int(EdgeClass.PSW),
                        gnn_score=round(score, 4))
                # node predictions -> column points (spatial NMS: nodes
                # sit centimetres apart, keep one point per 0.6 m cluster)
                gnn_columns = []
                sel = np.where(n_prob >= self.multi_thresh)[0]
                if sel.size:
                    keep = GEP.nms_points(mi["xy_m"][sel], n_prob[sel],
                                          min_dist=0.60)
                    for k in keep:
                        i = int(sel[k])
                        gnn_columns.append(dict(
                            x=round(float(mi["xy_m"][i, 0]), 4),
                            y=round(float(mi["xy_m"][i, 1]), 4),
                            p=round(float(n_prob[i]), 4), source="gnn",
                            virtual=bool(mi["virtual_mask"][i])))
                logger.info("Multi-GNN: %d edges scored, %d/%d nodes -> "
                            "columns (thresh=%.2f)",
                            len(mi["edge_ids"]), len(gnn_columns),
                            mi["xy_m"].shape[0], self.multi_thresh)

        elif predict and HAS_TORCH and self.gnn is not None:
            self.gnn.eval()
            with torch.no_grad():
                out = self.gnn(
                    torch.tensor(gi["node_features"]),
                    torch.tensor(gi["edge_index"]),
                    torch.tensor(gi["edge_features"]),
                ).numpy()
            ef = gi["edge_features"]
            for i, eid in enumerate(gi["edge_ids"]):
                rl, rr = float(out[i, 0]), float(out[i, 1])
                is_psw = int(ef[i, :N_CLASSES].argmax()) == int(EdgeClass.PSW)
                epreds[eid] = dict(sw_ratio_left=round(rl, 4),
                                   sw_ratio_right=round(rr, 4),
                                   is_psw=is_psw,
                                   gnn_score=round((rl + rr) / 2, 4))
        else:
            for eid in gi["edge_ids"]:
                epreds[eid] = dict(sw_ratio_left=0.0, sw_ratio_right=0.0,
                                   is_psw=True, gnn_score=0.0, note="no model")

        # -- B: RENE features (informational; thickness only used if TRAINED) --
        rene_rects = adapter.to_rene_input(self.sz, self.st, floors_ab)

        if predict and HAS_TORCH and self.rene_loaded and rene_rects:
            self.rene.eval()
            fm = np.vstack([r.rene_features for r in rene_rects])
            with torch.no_grad():
                pred = self.rene(torch.tensor(fm, dtype=torch.float)).numpy()
            for i, er in enumerate(rene_rects):
                er.eng_thickness = float(max(0, pred[i, 0]))
                er.eng_length = float(max(0, pred[i, 1]))

        # -- C: ENGINEERING-RULE shear-wall selection (GNN = soft prior) --
        selector = ShearWallSelector(min_len_m=self.sw_min_len,
                                     gnn_prior=self.sw_gnn_prior,
                                     target_sw_ratio=self.sw_target_ratio)
        sel = selector.select(floor, epreds)
        sw_ids = sel["sw_rect_ids"]
        rect_pier = sel.get("rect_pier", {})
        for eid in epreds:
            epreds[eid]["is_shear_wall"] = eid in sw_ids
            if eid in rect_pier:
                epreds[eid]["pier_id"], epreds[eid]["pier_len_m"] = rect_pier[eid]
        logger.info("Shear-wall selection: %d / %d candidate walls "
                    "(X=%d, Y=%d, exterior=%d), eccentricity=%.2fm",
                    len(sw_ids), sel["n_candidates"], sel["n_x"], sel["n_y"],
                    sel["n_exterior"], sel["eccentricity_m"])

        enriched = self._merge(floor, epreds, rene_rects, sw_ids,
                               sel["thickness"], rect_pier)

        # -- D: column layout -----------------------------------------------
        # GNN-predicted columns (multi model) are written to the enriched
        # JSON; the Stage 5 exporter consumes them directly.  Without a multi
        # model no "columns" key is written and Stage 5 falls back to the
        # structural-grid rule engine (single source of rules, no drift).
        if gnn_columns is not None:
            enriched["columns"] = gnn_columns
            logger.info("Column layout: %d GNN-predicted columns", len(gnn_columns))

        logger.info("Stage 4 done: %d edges, %d rects, %d shear walls",
                    len(epreds), len(rene_rects), len(sw_ids))
        return dict(gnn_input=gi, rene_rects=rene_rects,
                    edge_predictions=epreds, enriched_json=enriched,
                    sw_selection=sel, gnn_columns=gnn_columns,
                    multi_input=mi)

    def _merge(self, floor, epreds, rene_rects, sw_ids=None, sw_thickness=None,
               rect_pier=None):
        sw_ids = sw_ids or set()
        sw_thickness = sw_thickness or {}
        rect_pier = rect_pier or {}
        out = json.loads(json.dumps(floor))
        rm = {r.rect_id: r for r in rene_rects}
        for w in out.get("walls", []):
            for r in w.get("rects", []):
                rid = r["id"]
                ep = epreds.get(rid, {})
                is_sw = rid in sw_ids
                r["is_shear_wall"] = is_sw
                if rid in rect_pier:
                    r["pier_id"], r["pier_len_m"] = rect_pier[rid]
                r["sw_ratio"] = round((ep.get("sw_ratio_left", 0.0)
                                       + ep.get("sw_ratio_right", 0.0)) / 2, 4)
                # Engineering thickness: trained RENE if available, else rules.
                er = rm.get(rid)
                if self.rene_loaded and er is not None:
                    eng_t = er.eng_thickness
                else:
                    eng_t = engineering_thickness(r.get("thickness_m", 0.0), is_sw)
                r["eng_thickness_m"] = round(eng_t, 4)
                r["eng_length_m"] = round(r.get("length_m", 0.0), 4)
                if er is not None:
                    r["rene_features"] = er.rene_features.tolist()
        out["edge_predictions"] = epreds
        return out

    def export(self, result, path):
        with open(path, "w") as f:
            json.dump(result["enriched_json"], f, indent=2)
        logger.info("Exported -> %s", path)


# =============================================================================
# 7. Visualization
# =============================================================================

def visualize(graph, rene_rects, out_path,
              title="Stage 4 - Structural Enrichment"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # -- Left: graph topology --
    ax = axes[0]
    ax.set_title("Graph Topology (Stage 3)", fontsize=10)
    pos = {}
    for n, d in graph.nodes(data=True):
        pos[n] = (d.get("x_m", d.get("x", 0)), d.get("y_m", d.get("y", 0)))

    colors = []
    for u, v, d in graph.edges(data=True):
        et = d.get("edge_type", 0)
        colors.append({0: "#2196F3", 1: "#FF9800", 2: "#F44336"}.get(et, "#999"))

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=colors, width=3)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=50, node_color="#333")
    for n, (x, y) in pos.items():
        ax.annotate(n[:6], (x, y), fontsize=5, ha="center", va="bottom",
                    color="#555", xytext=(0, 5), textcoords="offset points")

    # Edge labels: length + thickness
    for u, v, d in graph.edges(data=True):
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        lbl = f"L={d.get('length_m',0):.2f}\nt={d.get('thickness_m',0):.3f}"
        ax.annotate(lbl, (mx, my), fontsize=5, ha="center", color="#1565C0",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.8))

    ax.set_aspect("equal")
    ax.invert_yaxis()  # image-space Y is top-down; invert to match
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

    # -- Right: RENE feature heatmap --
    ax2 = axes[1]
    ax2.set_title("RENE 30-Feature Vectors", fontsize=10)
    if rene_rects:
        mat = np.vstack([r.rene_features for r in rene_rects])
        col_max = np.abs(mat).max(axis=0)
        col_max[col_max < 1e-9] = 1.0
        mat_n = mat / col_max
        im = ax2.imshow(mat_n, aspect="auto", cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax2.set_yticks(range(len(rene_rects)))
        ax2.set_yticklabels([f"{r.rect_id[:8]}" for r in rene_rects], fontsize=7)
        ax2.set_xlabel("Feature index (0-29)")
        ax2.set_ylabel("Rectangle")
        plt.colorbar(im, ax=ax2, shrink=0.6, label="Normalized value")
        # Feature group separators
        for xp in [15.5, 27.5]:
            ax2.axvline(x=xp, color="k", lw=0.5, ls="--", alpha=0.4)
        ax2.text(7, -0.8, "Geometric", ha="center", fontsize=6, color="#666")
        ax2.text(21, -0.8, "Topological", ha="center", fontsize=6, color="#666")
        ax2.text(29, -0.8, "Oth.", ha="center", fontsize=6, color="#666")
    else:
        ax2.text(0.5, 0.5, "No rectangles", transform=ax2.transAxes, ha="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Visualization -> %s", out_path)


def visualize_predictions(enriched_json, out_path,
                          title="Stage 4 - Shear Wall Layout (engineering rules)"):
    """
    Floor-plan visualization of GNN-EP-4 predictions.

    Draws the actual wall geometry (real metre coordinates) and overlays:
      - All wall rects as thin gray lines
      - Each rect colored by average SW ratio (YlOrRd colormap)
      - For shear walls: thick red/orange segments showing left/right coverage
      - Node dots at intersection points
      - Colorbar + legend
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not available")
        return

    ep = enriched_json.get("edge_predictions", {})
    walls = [w for w in enriched_json.get("walls", []) if not w.get("rejected")]

    # Build point lookup per wall
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    all_xs, all_ys = [], []

    for w in walls:
        pts = {p["id"]: np.array([p["x"], p["y"]]) for p in w.get("points", [])}
        for r in w.get("rects", []):
            p1 = pts.get(r["p1_id"])
            p2 = pts.get(r["p2_id"])
            if p1 is None or p2 is None:
                continue

            all_xs += [p1[0], p2[0]]
            all_ys += [p1[1], p2[1]]

            pred = ep.get(r["id"], {})
            # Rule-based decision lives on the rect; fall back to GNN edge pred.
            is_sw = r.get("is_shear_wall", pred.get("is_shear_wall", False))
            eng_t = r.get("eng_thickness_m", r.get("thickness_m", 0.0))

            # Shear walls drawn as thick red piers; other walls thin grey.
            if is_sw:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#C62828", lw=6, solid_capstyle="round", zorder=3)
                mid = (p1 + p2) / 2
                ax.annotate(f"SW t{int(round(eng_t*1000))}",
                            (mid[0], mid[1]), fontsize=6, ha="center", va="bottom",
                            color="#B71C1C", fontweight="bold",
                            xytext=(0, 4), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="#EF9A9A", alpha=0.85), zorder=5)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#BDBDBD", lw=2.0, solid_capstyle="round", zorder=2)

    # Node dots at all unique points
    node_pts = {}
    for w in walls:
        for p in w.get("points", []):
            node_pts[p["id"]] = (p["x"], p["y"])
    if node_pts:
        xs = [v[0] for v in node_pts.values()]
        ys = [v[1] for v in node_pts.values()]
        ax.scatter(xs, ys, s=18, color="#37474F", zorder=6, linewidths=0)

    # GNN-predicted columns (green squares)
    cols = enriched_json.get("columns", [])
    if cols:
        ax.scatter([c["x"] for c in cols], [c["y"] for c in cols],
                   s=90, marker="s", color="#2E7D32", edgecolors="k",
                   linewidths=0.8, zorder=7)
        for c in cols:
            ax.annotate(f"{c.get('p', 0):.2f}", (c["x"], c["y"]),
                        fontsize=5, ha="center", va="top", color="#1B5E20",
                        xytext=(0, -7), textcoords="offset points")

    # Legend (rule-based shear-wall selection)
    legend_items = [
        Line2D([0], [0], color="#BDBDBD", lw=2.0, label="Non-structural wall"),
        Line2D([0], [0], color="#C62828", lw=6.0, label="Shear wall (rule-selected)"),
    ]
    if cols:
        legend_items.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#2E7D32",
                   markeredgecolor="k", markersize=8,
                   label=f"Column (GNN, {len(cols)})"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#ccc")

    # Axes formatting
    if all_xs:
        margin = 0.3
        ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
        ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.set_facecolor("#FAFAFA")

    # Count summary in corner (read the rule-based decision from the rects)
    n_total = sum(len(w.get("rects", [])) for w in walls)
    n_sw = sum(1 for w in walls for r in w.get("rects", [])
               if r.get("is_shear_wall"))
    ax.text(0.01, 0.01,
            f"Wall panels: {n_total}  |  Shear walls: {n_sw}  |  "
            f"SW rate: {100*n_sw/max(n_total,1):.0f}%",
            transform=ax.transAxes, fontsize=8, color="#555",
            va="bottom", ha="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction visualization -> %s", out_path)


def visualize_gnn_graph(mi, out_path,
                        title="GNN input graph (Stage 3 + virtual candidates)"):
    """
    Draw exactly what GNNEPMulti consumes (Stage4Adapter.to_multi_gnn_input):
      - PSW wall edges (blue), indoor door/window edges (orange),
        outdoor door edges (red), virtual candidate links (dashed grey)
      - real nodes (dark dots) vs virtual candidate nodes (green squares)
    Coordinates in metres.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        logger.warning("matplotlib not available")
        return

    xy = mi["xy_m"]
    ei = mi["edge_index"]
    ecl = mi["edge_class"]
    vmask = mi["virtual_mask"]

    styles = {
        0: dict(color="#1565C0", lw=2.2, ls="-", zorder=3),   # PSW
        1: dict(color="#FB8C00", lw=1.8, ls="-", zorder=3),   # indoor DW
        2: dict(color="#C62828", lw=1.8, ls="-", zorder=3),   # outdoor door
        3: dict(color="#9E9E9E", lw=0.8, ls="--", zorder=2),  # virtual link
    }
    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        c = int(ecl[k]) if int(ecl[k]) in styles else 3
        counts[c] += 1
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]], **styles[c])

    real = xy[~vmask]
    virt = xy[vmask]
    if len(real):
        ax.scatter(real[:, 0], real[:, 1], s=16, color="#263238",
                   zorder=5, linewidths=0)
    if len(virt):
        ax.scatter(virt[:, 0], virt[:, 1], s=70, marker="s",
                   facecolor="#A5D6A7", edgecolor="#1B5E20",
                   linewidths=1.0, zorder=6)

    legend = [
        Line2D([0], [0], color="#1565C0", lw=2.2,
               label=f"PSW wall edge ({counts[0]})"),
        Line2D([0], [0], color="#FB8C00", lw=1.8,
               label=f"indoor door/window ({counts[1]})"),
        Line2D([0], [0], color="#C62828", lw=1.8,
               label=f"outdoor door ({counts[2]})"),
        Line2D([0], [0], color="#9E9E9E", lw=0.8, ls="--",
               label=f"virtual link ({counts[3]})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#263238",
               markersize=6, label=f"node ({int((~vmask).sum())})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#A5D6A7",
               markeredgecolor="#1B5E20", markersize=8,
               label=f"virtual candidate ({int(vmask.sum())})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#ccc")
    ax.text(0.01, 0.01,
            f"N={xy.shape[0]} nodes  E={ei.shape[1]} edges  |  "
            f"node feats: {mi['node_features'].shape[1]}  "
            f"edge feats: {mi['edge_features'].shape[1]}",
            transform=ax.transAxes, fontsize=8, color="#555",
            va="bottom", ha="left")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("GNN graph visualization -> %s", out_path)


# =============================================================================
# 8. CLI entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Stage 4 - Structural Enrichment")
    ap.add_argument("-i", "--input", default="C:\\Dev\\Plan_2_FEM_2026\\floor_output.json",
                    help="Stage 2 floor JSON (default: floor_output.json)")
    ap.add_argument("-o", "--output", default="floor_enriched.json")
    ap.add_argument("-v", "--vis", default="floor_enrichment_vis.png")
    ap.add_argument("--pred-vis", default="floor_predictions_vis.png",
                    help="Output path for GNN-EP prediction floor-plan visualization")
    ap.add_argument("--graph-vis", default="gnn_graph_vis.png",
                    help="Output path for the GNN input-graph visualization")
    ap.add_argument("--seismic-zone", type=int, default=2)
    ap.add_argument("--soil-type", type=int, default=2)
    ap.add_argument("--proximity-tol", type=float, default=0.10,
                    help="Point proximity merge tolerance in metres")
    _train_dir = os.path.join(_ROOT, "train")
    # gnn_multi1.pt is preferred over gnn_multi.pt: on the de-duplicated,
    # leak-free StructGAN validation split (21 drawings) its edge head reaches
    # IoU 0.637 against an all-positive base rate of 0.620 at a genuine
    # operating threshold, whereas gnn_multi.pt sits at the base rate (0.619)
    # and its best threshold degenerates to selecting every candidate.  Column
    # F1 is also slightly better (0.587 vs 0.580).
    for _cand in ("gnn_multi1.pt", "gnn_multi.pt", "gnn_ep4.pt"):
        _default_model = os.path.join(_train_dir, _cand)
        if os.path.exists(_default_model):
            break
    ap.add_argument("--gnn-model", default=_default_model,
                    help="Path to trained weights (multi-task gnn_multi1.pt "
                         "preferred if present, then gnn_multi.pt, else legacy "
                         "gnn_ep4.pt)")
    ap.add_argument("--sw-target-ratio", type=float, default=0.02,
                    help="Target SW area ratio per direction (sum t*L / plan "
                         "area). 0.02 = standard low-rise practice; raise "
                         "for taller buildings")
    ap.add_argument("--sw-threshold", type=float, default=0.10,
                    help="Score threshold for shear-wall classification (default: 0.10)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    inp = args.input or os.path.join(os.path.dirname(os.path.abspath(__file__)), "floor_output.json")
    if not os.path.exists(inp):
        logger.error("Not found: %s  - run vectorization.py first", inp)
        return
    with open(inp) as f:
        floor = json.load(f)
    logger.info("Loaded Stage 2: %d walls", len(floor.get("walls", [])))

    # Stage 3: build graph
    G = build_graph_from_stage2(floor, proximity_tol_m=args.proximity_tol)

    # Stage 4: enrichment
    pipe = StructuralEnrichmentPipeline(
        sw_threshold=args.sw_threshold,
        seismic_zone=args.seismic_zone,
        soil_type=args.soil_type,
        proximity_tol=args.proximity_tol,
        sw_target_ratio=args.sw_target_ratio,
    )
    if args.gnn_model and HAS_TORCH and os.path.exists(args.gnn_model):
        pipe.load_gnn(args.gnn_model)
        logger.info("Loaded GNN weights: %s", args.gnn_model)

    result = pipe.run(G, floor, predict=bool(args.gnn_model and HAS_TORCH))

    # Print summary
    gi = result["gnn_input"]
    rr = result["rene_rects"]
    ep = result["edge_predictions"]

    print("\n" + "=" * 64)
    print("  STAGE 4 - STRUCTURAL ENRICHMENT SUMMARY")
    print("=" * 64)

    print(f"\n  GNN-EP Input (Edge-PSW)")
    print(f"    Nodes:  {gi['node_features'].shape[0]}  features: [x_n, y_n]")
    print(f"    Edges:  {gi['edge_features'].shape[0]}  features: [class_3, x_l, y_l, x_r, y_r, len]")
    print(f"    PyG:    {'yes' if gi['pyg_data'] is not None else 'no (install torch_geometric)'}")
    print(f"    L_GraphScale = {L_GRAPH_SCALE:.0f} mm")

    print(f"\n  RENE Features ({len(rr)} rectangles x 30 features)")
    for er in rr:
        f = er.rene_features
        pc = f"conn=({int(f[25])},{int(f[26])})"
        print(f"    {er.rect_id[:8]}  t={er.arch_thickness:.3f}m  "
              f"L={er.arch_length:.2f}m  angle={math.degrees(er.angle):5.1f}deg  {pc}")
    print(f"    Feature groups: Geom[0:16] Topo[16:28] Other[28:30]")

    print(f"\n  Edge Predictions")
    for eid, p in ep.items():
        sw = "SW" if p["is_shear_wall"] else "--"
        print(f"    {eid[:8]}  ratio_L={p['sw_ratio_left']:.3f}  "
              f"ratio_R={p['sw_ratio_right']:.3f}  [{sw}]  "
              f"{p.get('note', '')}")
    print(f"    threshold = {pipe.sw_thresh}")

    if result.get("gnn_columns") is not None:
        print(f"\n  GNN Column Predictions: {len(result['gnn_columns'])}")
        for c in result["gnn_columns"][:20]:
            v = " (virtual)" if c.get("virtual") else ""
            print(f"    ({c['x']:6.2f},{c['y']:6.2f})  p={c['p']:.3f}{v}")

    # Export
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, args.output)
    pipe.export(result, out)

    vis = os.path.join(base, args.vis)
    visualize(G, rr, vis)

    pred_vis = os.path.join(base, args.pred_vis)
    visualize_predictions(result["enriched_json"], pred_vis)

    # exact multi-task GNN input (works with or without torch/weights)
    if HAS_MULTI:
        adapter = Stage4Adapter(G, floor, proximity_tol_m=args.proximity_tol)
        mi = adapter.to_multi_gnn_input()
        if mi is not None:
            visualize_gnn_graph(mi, os.path.join(base, args.graph_vis))

    print(f"\n  Exported: {out}")
    print(f"  Visual:   {vis}")
    print(f"  Pred vis: {pred_vis}")

    if not HAS_TORCH:
        print("\n  NOTE: PyTorch not installed. Feature computation works.")
        print("  For inference: pip install torch torch_geometric")
        print("  Then load pre-trained GNN-EP-4 and RENE weights.")

    print("=" * 64)


if __name__ == "__main__":
       main()
