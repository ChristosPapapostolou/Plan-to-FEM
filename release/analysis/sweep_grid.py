# -*- coding: utf-8 -*-
"""Comment 24: sensitivity of the structural-grid detector to its hand-set
tolerances.

One-at-a-time sweep around the deployed defaults, run in-process on the cached
Stage 2 geometry of the ten evaluation plans:

    axis_tol    0.30 m  clustering tolerance for wall-centreline axes
    max_span    6.00 m  span limit that inserts virtual axes
    support_tol 0.35 m  radius within which a wall counts as vertical support
    merge_tol   0.40 m  candidate merged into an existing graph node
    max_link_m  8.00 m  longest virtual message-passing link
"""
import os, sys, json, glob
ROOT = r"C:\Dev\Plan_2_FEM_2026"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stages", "stage_3"))
sys.path.insert(0, os.path.join(ROOT, "models"))
import numpy as np
import structural_grid as SG
import gnn_ep as GE

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
CACHE = os.path.join(SCR, "psw_cache")

DEFAULTS = dict(axis_tol=0.30, max_span=6.0, support_tol=0.35)
GRIDS = {
    "axis_tol":    [0.15, 0.20, 0.30, 0.45, 0.60],
    "max_span":    [4.0, 5.0, 6.0, 7.0, 8.0],
    "support_tol": [0.20, 0.30, 0.35, 0.50, 0.70],
}
VIRT = {"merge_tol": [0.20, 0.30, 0.40, 0.60, 0.80],
        "max_link_m": [4.0, 6.0, 8.0, 12.0, 20.0]}


def load_segments(pj):
    floor = json.load(open(pj))
    return SG.segments_from_floor(floor), floor


def grid_metrics(segs, **kw):
    p = dict(DEFAULTS); p.update(kw)
    r = SG.place_columns(segs, **p)
    return dict(axes_x=len(r.axes_x), axes_y=len(r.axes_y),
                candidates=len(r.candidates), columns=len(r.columns))


def virt_metrics(segs, cands, merge_tol=0.40, max_link_m=8.0):
    """Virtual-node injection on the real node set of this plan."""
    pts = []
    for s in segs:
        pts.append(list(np.asarray(s.p1, float)[:2]))
        pts.append(list(np.asarray(s.p2, float)[:2]))
    xy = np.array(pts, float) if pts else np.zeros((0, 2))
    # dedupe real nodes at 1 cm
    if len(xy):
        keep = []
        for p in xy:
            if not keep or min(np.linalg.norm(np.array(k) - p) for k in keep) > 0.01:
                keep.append(tuple(p))
        xy = np.array(keep, float)
    ei = np.zeros((2, 0), dtype=int)
    xy2, ei2, ec2, et2, vm = GE.add_virtual_candidates(
        xy, ei, np.zeros(0, dtype=int), np.zeros(0),
        [tuple(c) for c in cands],
        merge_tol=merge_tol, k_connect=3, max_link_m=max_link_m)
    return dict(real_nodes=int(len(xy)), virtual_nodes=int(vm.sum()),
                virtual_links=int(ei2.shape[1]))


def main():
    plans = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    print(f"{len(plans)} plans", flush=True)
    data = {p: {} for p in ["axis_tol", "max_span", "support_tol",
                            "merge_tol", "max_link_m"]}
    cache = {}
    for pj in plans:
        segs, floor = load_segments(pj)
        cache[pj] = segs

    for param, vals in GRIDS.items():
        for v in vals:
            acc = []
            for pj in plans:
                try:
                    acc.append(grid_metrics(cache[pj], **{param: v}))
                except Exception as e:
                    print("  fail", param, v, os.path.basename(pj), e, flush=True)
            if acc:
                data[param][str(v)] = {k: round(float(np.mean([a[k] for a in acc])), 2)
                                       for k in acc[0]}
        print(f"  {param} done", flush=True)

    # candidate points at default settings, reused for the virtual-node sweep
    base_c = {}
    for pj in plans:
        r = SG.place_columns(cache[pj], **DEFAULTS)
        base_c[pj] = [tuple(np.asarray(c, float).ravel()[:2]) for c in r.candidates]
    for param, vals in VIRT.items():
        for v in vals:
            acc = []
            for pj in plans:
                try:
                    acc.append(virt_metrics(cache[pj], base_c[pj], **{param: v}))
                except Exception as e:
                    print("  fail", param, v, os.path.basename(pj), e, flush=True)
            if acc:
                data[param][str(v)] = {k: round(float(np.mean([a[k] for a in acc])), 2)
                                       for k in acc[0]}
        print(f"  {param} done", flush=True)

    json.dump(data, open(os.path.join(SCR, "grid_sweep.json"), "w"), indent=2)
    for param in data:
        if not data[param]:
            continue
        keys = list(next(iter(data[param].values())).keys())
        print(f"\n{param}  (default {DEFAULTS.get(param, {'merge_tol':0.40,'max_link_m':8.0}.get(param))})")
        print("  " + "value".ljust(8) + "".join(k.rjust(14) for k in keys))
        for v, m in data[param].items():
            print("  " + v.ljust(8) + "".join(f"{m[k]:14.2f}" for k in keys))


main()
