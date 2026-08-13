# -*- coding: utf-8 -*-
"""Capture the RAW scale factor (0.20 / measured median thickness) per plan and
simulated resolution, then sweep the deadband offline to choose it from data.

The deadband decision is a pure function of the raw factor:
    |k-1| <= D  ->  no correction (factor 1.0)
    otherwise   ->  factor = clamp(k, 0.2, 6.0)
so every candidate D can be scored without re-running the pipeline.
"""
import sys, os, json, copy, time
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026")
sys.path.insert(0, r"C:\Dev\Plan_2_FEM_2026\stages\stage_2")
import numpy as np, cv2

SCR = r"C:\Users\OFFICE~1\AppData\Local\Temp\claude\D--Scada-chatbot\4fb47394-3717-4810-8c4b-ca6040e2bb42\scratchpad"
sys.path.insert(0, SCR)
import bench_stage2 as B2
import stage_2 as S2

ROOT = B2.ROOT
TRUE_SCALE = 0.01
FACTORS = [1.0, 0.5, 0.35, 0.25]
CLAMP = (0.2, 6.0)

_orig = S2.auto_calibrate_floor
RAW = {"k": None}

def _wrapper(data, **kw):
    # raw factor: deadband 0 means only an exact 1.0 is suppressed
    _, kraw = _orig(copy.deepcopy(data), deadband=0.0)
    RAW["k"] = float(kraw)
    return _orig(data, **kw)

S2.auto_calibrate_floor = _wrapper


def main(stride=3, limit=None):
    folders = [l.strip().strip("/") for l in open(f"{ROOT}\\test.txt") if l.strip()]
    folders = folders[::stride]
    if limit:
        folders = folders[:limit]
    recs = []
    used = 0
    t0 = time.time()
    for i, rel in enumerate(folders):
        d = f"{ROOT}\\{rel.replace('/', chr(92))}"
        try:
            img = cv2.imread(d + r"\F1_scaled.png")
            if img is None: continue
            wp = B2.polys_of_class(d + r"\model.svg", ["Wall"])
            if not wp: continue
            H, W = img.shape[:2]
            allp = np.concatenate(wp, axis=0)
            inside = ((allp[:, 0] >= 0) & (allp[:, 0] < W) &
                      (allp[:, 1] >= 0) & (allp[:, 1] < H)).mean()
            if (1.0 - inside) > 0.02: continue
            gt = [s for s in (B2.seg_from_poly(p) for p in wp) if s]
            if len(gt) < 4: continue
            base = np.zeros((H, W), np.uint8)
            for p in wp:
                cv2.fillPoly(base, [np.round(p).astype(np.int32)], 255)
            af = float((base > 0).sum()) / float(H * W)
            if not (0.005 <= af <= 0.45): continue

            for f in FACTORS:
                if f == 1.0:
                    m = base
                else:
                    m = cv2.resize(base, (max(int(W * f), 32), max(int(H * f), 32)),
                                   interpolation=cv2.INTER_AREA)
                    m = ((m > 127) * 255).astype(np.uint8)
                RAW["k"] = None
                try:
                    B2.run_stage2(m)
                except Exception:
                    continue
                if RAW["k"] is None:
                    continue
                recs.append({"f": f, "true_scale": TRUE_SCALE / f, "k_raw": RAW["k"]})
            used += 1
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(folders)} used={used} ({time.time()-t0:.0f}s)", flush=True)

    open(SCR + r"\raw_factors.json", "w").write(json.dumps(recs))
    print(f"captured {len(recs)} records from {used} plans", flush=True)

    def score(D):
        out = {}
        for f in FACTORS:
            rr = [r for r in recs if r["f"] == f]
            if not rr: continue
            errs, noop = [], 0
            for r in rr:
                k = r["k_raw"]
                if abs(k - 1.0) <= D:
                    applied = 1.0; noop += 1
                else:
                    applied = min(max(k, CLAMP[0]), CLAMP[1])
                errs.append((TRUE_SCALE * applied) / r["true_scale"] - 1.0)
            a = np.abs(np.array(errs))
            out[f] = dict(median=float(np.median(errs)),
                          med_abs=float(np.median(a)),
                          w10=float(np.mean(a <= 0.10)),
                          w20=float(np.mean(a <= 0.20)),
                          noop=noop / len(rr))
        return out

    print(f"\n{'D':>6s} {'native noop':>12s} {'native w20':>11s} "
          f"{'resc w20':>10s} {'resc medabs':>12s} {'overall medabs':>15s}")
    best = None
    for D in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        s = score(D)
        nat = s[1.0]
        resc = [s[f] for f in FACTORS if f != 1.0]
        rw20 = float(np.mean([x["w20"] for x in resc]))
        rma = float(np.mean([x["med_abs"] for x in resc]))
        oma = float(np.mean([x["med_abs"] for x in s.values()]))
        print(f"{D:6.2f} {nat['noop']:12.2f} {nat['w20']:11.2f} "
              f"{rw20:10.2f} {rma:12.3f} {oma:15.3f}")
        # objective: protect correctly scaled inputs, keep rescaled recovery
        obj = nat["w20"] + rw20
        if best is None or obj > best[1]:
            best = (D, obj)
    print(f"\nbest by (native w20 + rescaled w20): D = {best[0]:.2f}")
    json.dump({str(D): score(D) for D in [0.20, 0.30, 0.40, 0.45, 0.50, 0.60]},
              open(SCR + r"\deadband_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main(stride=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
