# -*- coding: utf-8 -*-
"""Comment 21: quantitative evaluation of Stage 2 vectorisation against
ground-truth geometry (CubiCasa5K wall annotations).

Stage 2 is driven exactly as deployed (VectorizationPipeline -> export_json,
which applies prior-based scale estimation then consolidation). Ground truth
is the annotated wall polygons; the input mask is rasterised from those same
polygons, so segmentation error is excluded and geometric accuracy isolated.

All geometry is compared in pixel space: predicted metres are divided by the
pipeline's own effective scale, so scale-estimation error cancels and this
experiment measures vectorisation fidelity alone.
"""
import sys, os, json, math, tempfile, time, logging
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\stages\stage_2")
import numpy as np, cv2
from xml.dom import minidom

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
exec(open(SCR + r"\eval_cubicasa.py", encoding="utf-8").read().split("def main()")[0])

import stage_2 as S2
logging.getLogger().setLevel(logging.ERROR)
for n in list(logging.root.manager.loggerDict):
    logging.getLogger(n).setLevel(logging.ERROR)

ROOT = r"C:\Dev\Plan_2_FEM_2026\cubicasa5k\cubicasa5k"
ANG_TOL = 15.0     # deg, max orientation difference for a match
OVERLAP = 0.50     # min fraction of the shorter segment that must overlap
TAU = 10.0         # px, max mean perpendicular offset for a match
JUNC_TAU = 15.0    # px, junction matching radius


# ---------------------------------------------------------------- geometry
def polys_of_class(svg_path, classes):
    """First polygon of every <g> whose class is in `classes`, transformed."""
    doc = minidom.parse(svg_path)
    out = []
    for g in doc.getElementsByTagName("g"):
        cls = g.getAttribute("class")
        if not any(cls == c or cls.startswith(c + " ") for c in classes):
            continue
        poly = next((ch for ch in g.childNodes
                     if getattr(ch, "tagName", None) == "polygon"), None)
        if poly is None:
            continue
        pts = [tuple(map(float, p.split(",")))
               for p in poly.getAttribute("points").strip().split() if "," in p]
        if len(pts) < 3:
            continue
        M = elem_matrix(g)
        P = np.array([[x, y, 1.0] for x, y in pts]).T
        out.append((M @ P)[:2].T.astype(np.float32))
    return out


def seg_from_poly(poly):
    """Centreline endpoints, length and thickness of a wall quad."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(poly)
    L, t = max(w, h), min(w, h)
    if L < 5.0 or t < 1.0:
        return None
    a = math.radians(ang if w >= h else ang + 90.0)
    d = np.array([math.cos(a), math.sin(a)])
    c = np.array([cx, cy])
    return dict(p1=c - d * L / 2, p2=c + d * L / 2, L=L, t=t, d=d)


def seg_angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0


def ang_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def perp_and_overlap(pred, gt):
    """Mean perpendicular offset of pred endpoints from the gt line, and the
    fraction of the shorter segment whose projection overlaps."""
    g1, g2 = gt["p1"], gt["p2"]
    v = g2 - g1
    Lg = float(np.linalg.norm(v))
    if Lg < 1e-6:
        return 1e9, 0.0
    u = v / Lg
    n = np.array([-u[1], u[0]])
    d1 = abs(float(np.dot(pred["p1"] - g1, n)))
    d2 = abs(float(np.dot(pred["p2"] - g1, n)))
    t1 = float(np.dot(pred["p1"] - g1, u))
    t2 = float(np.dot(pred["p2"] - g1, u))
    lo, hi = min(t1, t2), max(t1, t2)
    ov = max(0.0, min(hi, Lg) - max(lo, 0.0))
    denom = max(min(Lg, abs(hi - lo)), 1e-6)
    return (d1 + d2) / 2.0, ov / denom


# ---------------------------------------------------------------- Stage 2
def run_stage2(mask):
    pipe = S2.VectorizationPipeline(scale=0.01, floor_height=2.8)
    floor = pipe.run(mask, floor_id="bench")
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        pipe.export_json(floor, path)
        with open(path) as f:
            data = json.load(f)
    finally:
        try: os.remove(path)
        except OSError: pass
    return data


def pred_segments(data):
    """Predicted centreline segments converted back to pixel space."""
    s = float(data.get("scale_m_per_px", 0.01)) or 0.01
    segs = []
    for w in data.get("walls", []):
        if w.get("rejected"):
            continue
        pts = {p["id"]: (p["x"], p["y"]) for p in w.get("points", [])}
        for r in w.get("rects", []):
            a, b = pts.get(r["p1_id"]), pts.get(r["p2_id"])
            if a is None or b is None:
                continue
            p1 = np.array(a, np.float64) / s
            p2 = np.array(b, np.float64) / s
            L = float(np.linalg.norm(p2 - p1))
            if L < 1e-6:
                continue
            segs.append(dict(p1=p1, p2=p2, L=L,
                             t=float(r.get("thickness_m", 0.0)) / s))
    return segs, s


# ---------------------------------------------------------------- metrics
def evaluate_plan(gt_segs, pr_segs):
    cand = []
    for i, p in enumerate(pr_segs):
        ap = seg_angle(p["p1"], p["p2"])
        for j, g in enumerate(gt_segs):
            ag = seg_angle(g["p1"], g["p2"])
            if ang_diff(ap, ag) > ANG_TOL:
                continue
            d, ov = perp_and_overlap(p, g)
            if d <= TAU and ov >= OVERLAP:
                cand.append((ov, -d, i, j))
    cand.sort(reverse=True)
    used_p, matched_g, pairs = set(), {}, []
    for ov, negd, i, j in cand:
        if i in used_p:
            continue
        used_p.add(i)
        matched_g.setdefault(j, []).append(i)
        pairs.append((i, j, -negd))
    return dict(n_gt=len(gt_segs), n_pred=len(pr_segs), tp=len(used_p),
                matched_gt=len(matched_g), pairs=pairs, matched_map=matched_g)


def junctions_from(segs, tau):
    """Cluster segment endpoints; a junction is a cluster of >= 2 segments."""
    pts, owner = [], []
    for k, s in enumerate(segs):
        pts.append(s["p1"]); owner.append(k)
        pts.append(s["p2"]); owner.append(k)
    if not pts:
        return []
    P = np.array(pts); used = np.zeros(len(P), bool); out = []
    for i in range(len(P)):
        if used[i]:
            continue
        d = np.linalg.norm(P - P[i], axis=1)
        idx = np.where((d <= tau) & (~used))[0]
        used[idx] = True
        if len(set(owner[k] for k in idx)) >= 2:
            out.append(P[idx].mean(axis=0))
    return out


def main(limit=None, stride=1, split="test.txt"):
    folders = [l.strip().strip("/") for l in open(f"{ROOT}\\{split}") if l.strip()]
    folders = folders[::stride]
    if limit:
        folders = folders[:limit]

    agg = dict(tp=0, n_pred=0, n_gt=0, matched_gt=0)
    perp_all, thick_rel, endp_all, frag = [], [], [], []
    jP = jR = 0.0; jn = 0
    dimx, dimy = [], []
    open_tot = open_gap = 0
    used = skipped = 0
    t0 = time.time()

    for k, rel in enumerate(folders):
        d = f"{ROOT}\\{rel.replace('/', chr(92))}"
        try:
            img = cv2.imread(d + r"\F1_scaled.png")
            if img is None:
                skipped += 1; continue
            wp = polys_of_class(d + r"\model.svg", ["Wall"])
            if not wp:
                skipped += 1; continue
            H, W = img.shape[:2]
            allp = np.concatenate(wp, axis=0)
            inside = ((allp[:, 0] >= 0) & (allp[:, 0] < W) &
                      (allp[:, 1] >= 0) & (allp[:, 1] < H)).mean()
            if (1.0 - inside) > 0.02:
                skipped += 1; continue

            gt_segs = [s for s in (seg_from_poly(p) for p in wp) if s]
            if len(gt_segs) < 4:
                skipped += 1; continue

            mask = np.zeros((H, W), np.uint8)
            for p in wp:
                cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 255)
            af = float((mask > 0).sum()) / float(H * W)
            if not (0.005 <= af <= 0.45):
                skipped += 1; continue

            data = run_stage2(mask)
            pr_segs, _s = pred_segments(data)
            if not pr_segs:
                skipped += 1; continue

            r = evaluate_plan(gt_segs, pr_segs)
            agg["tp"] += r["tp"]; agg["n_pred"] += r["n_pred"]
            agg["n_gt"] += r["n_gt"]; agg["matched_gt"] += r["matched_gt"]
            frag.append(r["n_pred"] / max(r["n_gt"], 1))

            for i, j, dev in r["pairs"]:
                perp_all.append(dev)
                tg = gt_segs[j]["t"]
                if tg > 1e-6:
                    thick_rel.append(abs(pr_segs[i]["t"] - tg) / tg)
            for j, idxs in r["matched_map"].items():
                if len(idxs) == 1:
                    p, g = pr_segs[idxs[0]], gt_segs[j]
                    e1 = min(float(np.linalg.norm(p["p1"] - g["p1"])),
                             float(np.linalg.norm(p["p1"] - g["p2"])))
                    e2 = min(float(np.linalg.norm(p["p2"] - g["p1"])),
                             float(np.linalg.norm(p["p2"] - g["p2"])))
                    endp_all.append((e1 + e2) / 2.0)

            gj = junctions_from(gt_segs, JUNC_TAU)
            pj = junctions_from(pr_segs, JUNC_TAU)
            if gj and pj:
                G = np.array(gj); P = np.array(pj)
                D = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=2)
                jR += float((D.min(axis=1) <= JUNC_TAU).mean())
                jP += float((D.min(axis=0) <= JUNC_TAU).mean())
                jn += 1

            gpts = np.concatenate([np.array([s["p1"], s["p2"]]) for s in gt_segs])
            ppts = np.concatenate([np.array([s["p1"], s["p2"]]) for s in pr_segs])
            gw = gpts[:, 0].max() - gpts[:, 0].min()
            gh = gpts[:, 1].max() - gpts[:, 1].min()
            pw = ppts[:, 0].max() - ppts[:, 0].min()
            ph = ppts[:, 1].max() - ppts[:, 1].min()
            if gw > 1 and gh > 1:
                dimx.append(abs(pw - gw) / gw); dimy.append(abs(ph - gh) / gh)

            # opening recall: rebuild the mask without openings, check for gaps
            ops = polys_of_class(d + r"\model.svg", ["Door", "Window"])
            if ops:
                m2 = mask.copy()
                for p in ops:
                    cv2.fillPoly(m2, [np.round(p).astype(np.int32)], 0)
                ps2, _ = pred_segments(run_stage2(m2))
                if ps2:
                    for p in ops:
                        c = p.mean(axis=0).astype(np.float64)
                        probe = dict(p1=c - 0.5, p2=c + 0.5)
                        covered = False
                        for s in ps2:
                            dd, ov = perp_and_overlap(probe, s)
                            if dd <= max(s["t"], 6.0) / 2 and ov > 0:
                                covered = True; break
                        open_tot += 1
                        if not covered:
                            open_gap += 1
            used += 1
        except Exception:
            skipped += 1
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(folders)} used={used} ({time.time()-t0:.0f}s)",
                  flush=True)

    P = agg["tp"] / max(agg["n_pred"], 1)
    R = agg["matched_gt"] / max(agg["n_gt"], 1)
    res = {
        "split": split, "n_plans": used, "n_skipped": skipped,
        "match_criteria": {"angle_deg": ANG_TOL, "overlap": OVERLAP,
                           "perp_tol_px": TAU, "junction_tau_px": JUNC_TAU},
        "segment_precision": round(P, 4),
        "segment_recall": round(R, 4),
        "segment_f1": round(2 * P * R / max(P + R, 1e-9), 4),
        "centreline_dev_px_mean": round(float(np.mean(perp_all)), 3) if perp_all else None,
        "centreline_dev_px_median": round(float(np.median(perp_all)), 3) if perp_all else None,
        "endpoint_err_px_mean": round(float(np.mean(endp_all)), 3) if endp_all else None,
        "endpoint_err_px_median": round(float(np.median(endp_all)), 3) if endp_all else None,
        "thickness_rel_err_mean": round(float(np.mean(thick_rel)), 4) if thick_rel else None,
        "thickness_rel_err_median": round(float(np.median(thick_rel)), 4) if thick_rel else None,
        "junction_precision": round(jP / max(jn, 1), 4),
        "junction_recall": round(jR / max(jn, 1), 4),
        "plan_dim_err_x_median": round(float(np.median(dimx)), 4) if dimx else None,
        "plan_dim_err_y_median": round(float(np.median(dimy)), 4) if dimy else None,
        "fragmentation_pred_per_gt_median": round(float(np.median(frag)), 3) if frag else None,
        "openings_total": open_tot,
        "openings_left_as_gap": open_gap,
        "opening_recall": round(open_gap / open_tot, 4) if open_tot else None,
    }
    print(json.dumps(res, indent=2), flush=True)
    open(SCR + r"\stage2_bench.json", "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(limit=lim, stride=st)
