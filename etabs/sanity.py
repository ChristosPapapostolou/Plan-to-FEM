"""
FEM Model Sanity Checker (roadmap step 5)
==========================================
Post-export structural sanity checks on the Stage 5 FEM JSON.  Every check
returns pass / warn / fail with a measured value, so the report doubles as a
quantitative evaluation metric for the GNN-generated layouts.

Checks:
  1. lateral_lines      >= 2 lateral-resisting SW lines in BOTH directions
  2. eccentricity       stiffness centre vs geometric centre (k ~ t*L^3)
  3. beam_support       every beam end lands on a column / wall / other beam
  4. slab_span          max clear distance from slab interior to a support
  5. column_continuity  columns stack through all stories
  6. orphan_nodes       nodes not referenced by any element

Usage:
  python etabs/sanity.py -i model_fem.json -o sanity_report.json
  python etabs/sanity.py -i model_fem.json --strict      # exit 1 on any fail
"""

from __future__ import annotations
import argparse, json, logging, math, os, sys

import numpy as np

logger = logging.getLogger(__name__)

PASS, WARN, FAIL = "pass", "warn", "fail"
_SCORE = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}


# =============================================================================
# helpers
# =============================================================================

def _nodes_xyz(model):
    return {n["id"]: np.array([n["x"], n["y"], n["z"]], float)
            for n in model.get("nodes", [])}


def _story_panels(model, nodes, story=0, sw_only=True):
    """SW wall panels of one story as bottom-edge segments [(p1,p2,t)]."""
    segs = []
    for wp in model.get("wall_panels", []):
        if wp.get("story") != story:
            continue
        if sw_only and not wp.get("is_shear_wall"):
            continue
        ids = wp.get("nodes", [])
        if len(ids) < 2 or ids[0] not in nodes or ids[1] not in nodes:
            continue
        p1, p2 = nodes[ids[0]][:2], nodes[ids[1]][:2]   # BL, BR
        segs.append((p1, p2, float(wp.get("thickness_m", 0.2))))
    return segs


def _columns_xy(model, nodes, story=0):
    out = []
    for fm in model.get("frame_members", []):
        if fm.get("type") == "column" and fm.get("story") == story:
            n = nodes.get(fm["start_node"])
            if n is not None:
                out.append(n[:2])
    return out


def _dist_pt_seg(pt, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = np.clip(np.dot(pt - a, ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(pt - (a + t * ab)))


def _orient(p1, p2):
    d = np.abs(p2 - p1)
    return "H" if d[0] >= d[1] else "V"


# =============================================================================
# checks
# =============================================================================

def check_lateral_lines(model, nodes, *, line_tol=0.40, min_lines=2):
    """>= min_lines distinct SW lines resisting each direction."""
    segs = _story_panels(model, nodes, 0, sw_only=True)
    offs = {"H": [], "V": []}   # H walls resist X, V walls resist Y
    for p1, p2, _t in segs:
        o = _orient(p1, p2)
        perp = 1 if o == "H" else 0
        offs[o].append(float((p1[perp] + p2[perp]) / 2))

    def _n_lines(vals):
        lines = []
        for v in sorted(vals):
            if not lines or v - lines[-1] > line_tol:
                lines.append(v)
        return len(lines)

    nx_, ny_ = _n_lines(offs["H"]), _n_lines(offs["V"])
    status = PASS if (nx_ >= min_lines and ny_ >= min_lines) else FAIL
    return dict(name="lateral_lines", status=status,
                value=dict(x_dir=nx_, y_dir=ny_),
                detail=f"{nx_} SW lines resisting X, {ny_} resisting Y "
                       f"(need >= {min_lines} each)")


def _merge_collinear(segs, *, off_tol=0.20, gap_tol=0.20):
    """Merge collinear touching wall panels back into continuous piers.

    The exporter splits long piers into <=3 m sub-panels; weighting each
    sub-panel by t*L^3 would massively understate the pier stiffness."""
    out = []
    for orient in ("H", "V"):
        ax = 0 if orient == "H" else 1
        perp = 1 - ax
        items = [(p1, p2, t) for (p1, p2, t) in segs
                 if _orient(p1, p2) == orient]
        used = [False] * len(items)
        for i, (p1, p2, t) in enumerate(items):
            if used[i]:
                continue
            used[i] = True
            band = (p1[perp] + p2[perp]) / 2
            lo = min(p1[ax], p2[ax]); hi = max(p1[ax], p2[ax])
            tmax = t
            changed = True
            while changed:
                changed = False
                for j, (q1, q2, tq) in enumerate(items):
                    if used[j]:
                        continue
                    if abs((q1[perp] + q2[perp]) / 2 - band) > off_tol:
                        continue
                    qlo = min(q1[ax], q2[ax]); qhi = max(q1[ax], q2[ax])
                    if qlo <= hi + gap_tol and qhi >= lo - gap_tol:
                        lo = min(lo, qlo); hi = max(hi, qhi)
                        tmax = max(tmax, tq)
                        used[j] = True
                        changed = True
            a, b = np.zeros(2), np.zeros(2)
            a[perp] = b[perp] = band
            a[ax], b[ax] = lo, hi
            out.append((a, b, tmax))
    return out


def check_eccentricity(model, nodes, *, warn_frac=0.10, fail_frac=0.20):
    """Stiffness centre (k ~ t*L^3 per direction, PIER level) vs plan centre."""
    segs = _merge_collinear(_story_panels(model, nodes, 0, sw_only=True))
    slab_pts = []
    for sl in model.get("slabs", []):
        if sl.get("story") == 0:
            slab_pts += [nodes[i][:2] for i in sl.get("nodes", [])
                         if i in nodes]
    if not segs or not slab_pts:
        return dict(name="eccentricity", status=WARN, value=None,
                    detail="no SW panels or slab found")

    slab_pts = np.vstack(slab_pts)
    lo, hi = slab_pts.min(axis=0), slab_pts.max(axis=0)
    gc = (lo + hi) / 2.0            # bbox centre (matches the SW selector)
    plan_w = hi - lo
    plan_w[plan_w < 1e-6] = 1e-6

    # walls along X (H) resist X: their stiffness centre y-position matters,
    # walls along Y (V) resist Y: x-position matters.
    ky_sum = kx_sum = 0.0
    x_cr = y_cr = 0.0
    for p1, p2, t in segs:
        L = float(np.linalg.norm(p2 - p1))
        k = t * L ** 3
        c = (p1 + p2) / 2
        if _orient(p1, p2) == "H":       # resists X -> y position
            kx_sum += k
            y_cr += k * c[1]
        else:                             # resists Y -> x position
            ky_sum += k
            x_cr += k * c[0]
    ecc = {}
    if ky_sum > 0:
        ecc["x"] = abs(x_cr / ky_sum - gc[0]) / plan_w[0]
    if kx_sum > 0:
        ecc["y"] = abs(y_cr / kx_sum - gc[1]) / plan_w[1]
    if not ecc:
        return dict(name="eccentricity", status=FAIL, value=None,
                    detail="no lateral stiffness in one or both directions")

    worst = max(ecc.values())
    status = PASS if worst < warn_frac else WARN if worst < fail_frac else FAIL
    sugg = ""
    if status != PASS:
        axis = max(ecc, key=ecc.get)
        sugg = (f"strengthen SW on the far side along {axis} "
                f"(or add columns) to re-centre stiffness")
    return dict(name="eccentricity", status=status,
                value={k: round(v, 4) for k, v in ecc.items()},
                detail=f"stiffness ecc: " +
                       ", ".join(f"{k}={100*v:.1f}%" for k, v in ecc.items()) +
                       f" of plan dim (warn>{100*warn_frac:.0f}%, "
                       f"fail>{100*fail_frac:.0f}%)",
                suggestion=sugg)


def check_beam_support(model, nodes, *, tol=0.20):
    """Every beam end must land on a column, a wall top edge, or another beam."""
    beams = [fm for fm in model.get("frame_members", [])
             if fm.get("type") == "beam" and fm.get("story") == 0]
    cols = _columns_xy(model, nodes, 0)
    walls = _story_panels(model, nodes, 0, sw_only=False)

    beam_segs = []
    for b in beams:
        a, c = nodes.get(b["start_node"]), nodes.get(b["end_node"])
        if a is not None and c is not None:
            beam_segs.append((a[:2], c[:2]))

    n_ends = n_direct = n_beam = n_unsup = 0
    unsupported = []
    for bi, (a, c) in enumerate(beam_segs):
        for pt in (a, c):
            n_ends += 1
            if cols and min(np.linalg.norm(pt - q) for q in cols) <= tol:
                n_direct += 1
                continue
            if walls and min(_dist_pt_seg(pt, w1, w2)
                             for w1, w2, _ in walls) <= tol:
                n_direct += 1
                continue
            hit = False
            for bj, (u, v) in enumerate(beam_segs):
                if bj == bi:
                    continue
                if _dist_pt_seg(pt, u, v) <= tol:
                    hit = True
                    break
            if hit:
                n_beam += 1
            else:
                n_unsup += 1
                unsupported.append([round(float(pt[0]), 2),
                                    round(float(pt[1]), 2)])

    if n_unsup:
        status = FAIL
    elif n_beam > 0.5 * max(n_ends, 1):
        status = WARN
    else:
        status = PASS
    return dict(name="beam_support", status=status,
                value=dict(ends=n_ends, direct=n_direct,
                           via_beam=n_beam, unsupported=n_unsup),
                detail=f"{n_direct}/{n_ends} ends on column/wall, "
                       f"{n_beam} on other beams, {n_unsup} unsupported"
                       + (f" at {unsupported[:5]}" if unsupported else ""))


def check_slab_span(model, nodes, *, max_span=6.0, grid=0.5):
    """Max clear distance from any slab interior point to a vertical support."""
    slab_pts = []
    for sl in model.get("slabs", []):
        if sl.get("story") == 0:
            slab_pts += [nodes[i][:2] for i in sl.get("nodes", [])
                         if i in nodes]
    if len(slab_pts) < 3:
        return dict(name="slab_span", status=WARN, value=None,
                    detail="no slab found")
    poly = np.vstack(slab_pts)

    supports = [np.asarray(c) for c in _columns_xy(model, nodes, 0)]
    for p1, p2, _t in _story_panels(model, nodes, 0, sw_only=False):
        L = np.linalg.norm(p2 - p1)
        n = max(2, int(L / grid) + 1)
        supports += [p1 + t * (p2 - p1) for t in np.linspace(0, 1, n)]
    if not supports:
        return dict(name="slab_span", status=FAIL, value=None,
                    detail="no vertical supports at all")
    sup = np.vstack(supports)

    lo, hi = poly.min(axis=0), poly.max(axis=0)
    from matplotlib.path import Path as _MplPath
    hullp = _MplPath(poly)

    worst = 0.0
    xs = np.arange(lo[0], hi[0] + grid, grid)
    ys = np.arange(lo[1], hi[1] + grid, grid)
    for x in xs:
        for y in ys:
            if not hullp.contains_point((x, y)):
                continue
            d = float(np.min(np.linalg.norm(sup - np.array([x, y]), axis=1)))
            worst = max(worst, d)

    warn_at, fail_at = 0.75 * max_span, 1.0 * max_span
    status = PASS if worst <= warn_at else WARN if worst <= fail_at else FAIL
    return dict(name="slab_span", status=status,
                value=round(worst, 2),
                detail=f"max distance from slab interior to a vertical "
                       f"support = {worst:.2f} m "
                       f"(warn>{warn_at:.1f}, fail>{fail_at:.1f})")


def check_column_continuity(model, nodes):
    """Columns should exist at the same plan position on every story."""
    n_stories = len(model.get("stories", [])) or 1
    locs: dict[tuple, set] = {}
    for fm in model.get("frame_members", []):
        if fm.get("type") != "column":
            continue
        n = nodes.get(fm["start_node"])
        if n is None:
            continue
        key = (round(float(n[0]), 2), round(float(n[1]), 2))
        locs.setdefault(key, set()).add(int(fm.get("story", 0)))

    broken = [k for k, st in locs.items() if len(st) != n_stories]
    status = PASS if not broken else WARN if len(broken) <= 2 else FAIL
    return dict(name="column_continuity", status=status,
                value=dict(locations=len(locs), discontinuous=len(broken)),
                detail=f"{len(locs)} column locations, "
                       f"{len(broken)} not continuous over all "
                       f"{n_stories} stories"
                       + (f" e.g. {broken[:3]}" if broken else ""))


def check_orphan_nodes(model, nodes):
    used = set()
    for wp in model.get("wall_panels", []):
        used.update(wp.get("nodes", []))
    for fm in model.get("frame_members", []):
        used.add(fm.get("start_node"))
        used.add(fm.get("end_node"))
    for sl in model.get("slabs", []):
        used.update(sl.get("nodes", []))
    orphans = [nid for nid in nodes if nid not in used]
    status = PASS if not orphans else WARN
    return dict(name="orphan_nodes", status=status,
                value=len(orphans),
                detail=f"{len(orphans)} nodes not referenced by any element")


# =============================================================================
# runner
# =============================================================================

def run_checks(model: dict, *, max_span=6.0) -> dict:
    nodes = _nodes_xyz(model)
    checks = [
        check_lateral_lines(model, nodes),
        check_eccentricity(model, nodes),
        check_beam_support(model, nodes),
        check_slab_span(model, nodes, max_span=max_span),
        check_column_continuity(model, nodes),
        check_orphan_nodes(model, nodes),
    ]
    score = sum(_SCORE[c["status"]] for c in checks) / len(checks)
    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    n_warn = sum(1 for c in checks if c["status"] == WARN)
    return dict(
        model_id=model.get("model_id", model.get("floor_id", "")),
        score=round(score, 3),
        n_pass=len(checks) - n_fail - n_warn,
        n_warn=n_warn, n_fail=n_fail,
        checks=checks,
    )


def main():
    ap = argparse.ArgumentParser(description="FEM model sanity checks")
    ap.add_argument("-i", "--input", required=True, help="model_fem.json")
    ap.add_argument("-o", "--output", default=None,
                    help="Report path (default: <input dir>/sanity_report.json)")
    ap.add_argument("--max-span", type=float, default=6.0)
    ap.add_argument("--strict", action="store_true",
                    help="Exit with code 1 if any check fails")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    with open(args.input) as f:
        model = json.load(f)
    report = run_checks(model, max_span=args.max_span)

    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)), "sanity_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print("  FEM SANITY REPORT")
    print("=" * 64)
    for c in report["checks"]:
        flag = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[c["status"]]
        print(f"  [{flag}] {c['name']:20s} {c['detail']}")
        if c.get("suggestion"):
            print(f"         -> {c['suggestion']}")
    print("-" * 64)
    print(f"  score = {report['score']:.2f}  "
          f"(pass={report['n_pass']}, warn={report['n_warn']}, "
          f"fail={report['n_fail']})")
    print(f"  report -> {out}")
    print("=" * 64)

    if args.strict and report["n_fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
