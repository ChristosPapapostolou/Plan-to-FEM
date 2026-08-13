# -*- coding: utf-8 -*-
"""Comment 22: evaluate the prior-based scale estimator on plans of known size.

CubiCasa5K F1_scaled.png is at 0.01 m/px (verified: per-plan medians 0.009989 -
0.010009 from the released room-dimension labels). Rescaling a plan by f gives a
drawing whose true scale is 0.01/f m/px while the pipeline still assumes its
0.01 default, so the estimator's job is to recover exactly 0.01/f.

Reports scale-factor error per simulated resolution, the propagated errors in
wall thickness and plan dimensions, and the validity of the 0.20 m prior itself.
"""
import sys, os, json, math, tempfile, time, logging
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\stages\stage_2")
import numpy as np, cv2

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
sys.path.insert(0, SCR)
import bench_stage2 as B2
ROOT = B2.ROOT
polys_of_class = B2.polys_of_class
seg_from_poly = B2.seg_from_poly
run_stage2 = B2.run_stage2
pred_segments = B2.pred_segments

TRUE_SCALE = 0.01                      # m/px of F1_scaled.png
FACTORS = [1.0, 0.5, 0.35, 0.25]       # -> 10, 20, 28.6, 40 mm/px


def main(limit=None, stride=3, split="test.txt"):
    folders = [l.strip().strip("/") for l in open(f"{ROOT}\\{split}") if l.strip()]
    folders = folders[::stride]
    if limit:
        folders = folders[:limit]

    rows = {f: [] for f in FACTORS}
    true_thick = []
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
            gt = [s for s in (seg_from_poly(p) for p in wp) if s]
            if len(gt) < 4:
                skipped += 1; continue

            base = np.zeros((H, W), np.uint8)
            for p in wp:
                cv2.fillPoly(base, [np.round(p).astype(np.int32)], 255)
            af = float((base > 0).sum()) / float(H * W)
            if not (0.005 <= af <= 0.45):
                skipped += 1; continue

            # length-weighted median true wall thickness (metres)
            th = np.array([s["t"] for s in gt]) * TRUE_SCALE
            wt = np.array([max(s["L"], 1e-3) for s in gt])
            o = np.argsort(th); c = np.cumsum(wt[o])
            t_true = float(th[o][int(np.searchsorted(c, c[-1] / 2.0))])
            true_thick.append(t_true)

            gpts = np.concatenate([np.array([s["p1"], s["p2"]]) for s in gt])
            gw_m = (gpts[:, 0].max() - gpts[:, 0].min()) * TRUE_SCALE
            gh_m = (gpts[:, 1].max() - gpts[:, 1].min()) * TRUE_SCALE

            for f in FACTORS:
                if f == 1.0:
                    m = base
                else:
                    m = cv2.resize(base, (max(int(W * f), 32), max(int(H * f), 32)),
                                   interpolation=cv2.INTER_AREA)
                    m = ((m > 127) * 255).astype(np.uint8)
                true_s = TRUE_SCALE / f
                try:
                    data = run_stage2(m)
                except Exception:
                    continue
                rec = float(data.get("scale_m_per_px", TRUE_SCALE))
                kf = float(data.get("scale_autocalibrated_factor", 1.0))
                segs, _ = pred_segments(data)
                if not segs:
                    continue
                # recovered metric quantities
                pth = np.array([s["t"] for s in segs]) * rec
                pl = np.array([max(s["L"], 1e-6) for s in segs])
                o2 = np.argsort(pth); c2 = np.cumsum(pl[o2])
                t_rec = float(pth[o2][int(np.searchsorted(c2, c2[-1] / 2.0))])
                ppts = np.concatenate([np.array([s["p1"], s["p2"]]) for s in segs])
                pw_m = (ppts[:, 0].max() - ppts[:, 0].min()) * rec
                ph_m = (ppts[:, 1].max() - ppts[:, 1].min()) * rec
                rows[f].append(dict(
                    scale_rel_err=rec / true_s - 1.0,
                    applied_factor=kf,
                    deadband_noop=(abs(kf - 1.0) < 1e-9),
                    thick_err_m=t_rec - t_true,
                    dim_rel_err=0.5 * (abs(pw_m - gw_m) / max(gw_m, 1e-6) +
                                       abs(ph_m - gh_m) / max(gh_m, 1e-6)),
                ))
            used += 1
        except Exception:
            skipped += 1
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(folders)} used={used} ({time.time()-t0:.0f}s)",
                  flush=True)

    def summ(v):
        a = np.array(v, float)
        return dict(median=round(float(np.median(a)), 4),
                    mean=round(float(np.mean(a)), 4),
                    p05=round(float(np.percentile(a, 5)), 4),
                    p95=round(float(np.percentile(a, 95)), 4))

    out = {"n_plans": used, "n_skipped": skipped, "true_scale_m_per_px": TRUE_SCALE,
           "prior_target_thickness_m": 0.20,
           "true_wall_thickness_m": summ(true_thick),
           "by_simulated_resolution": {}}
    for f in FACTORS:
        r = rows[f]
        if not r:
            continue
        se = [x["scale_rel_err"] for x in r]
        a = np.abs(np.array(se))
        out["by_simulated_resolution"][f"{TRUE_SCALE/f*1000:.0f} mm/px"] = {
            "n": len(r),
            "scale_rel_err": summ(se),
            "abs_err_within_5pct": round(float(np.mean(a <= 0.05)), 4),
            "abs_err_within_10pct": round(float(np.mean(a <= 0.10)), 4),
            "abs_err_within_20pct": round(float(np.mean(a <= 0.20)), 4),
            "deadband_noop_frac": round(float(np.mean([x["deadband_noop"] for x in r])), 4),
            "thickness_err_m": summ([x["thick_err_m"] for x in r]),
            "plan_dim_rel_err": summ([x["dim_rel_err"] for x in r]),
        }
    print(json.dumps(out, indent=2), flush=True)
    open(SCR + r"\scale_bench.json", "w").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(limit=lim, stride=st)
