"""
MSD validation sweep
====================
Runs a sample of Modified Swiss Dwellings (MSD) plans through the geometry +
FEM half of the pipeline and collects the structural-sanity scores into a
distribution. This is the cross-typology robustness test discussed in
REVIEW_AND_ROADMAP.md: real Swiss residential low/mid-rise stock, much closer
to the target typology than the StructGAN Chinese high-rise corpus.

Per plan the sweep does (Stage 1 is SKIPPED — MSD gives us the wall mask):

    struct_in/<id>.npy --[msd_to_mask]--> mask.png
    Stage 2  vectorization    -> floor_output.json      (auto-calibrates scale)
    consolidate               -> floor_consolidated.json
    Stage 4  SW + GNN columns -> floor_enriched.json
    Stage 5  FEM export        -> model_fem.json
    sanity                    -> sanity_report.json     (0..1 score)

Outputs:
    results/msd_validation/<id>/   per-plan vis + sanity_report.json
    results/msd_validation/scores.csv     id, reached, score, n_warn, n_fail
    results/msd_validation/summary.txt    distribution (mean/median/quartiles,
                                          pass-rate, failure reasons)

NOTE ON LABELS: MSD has no shear-wall or column ground truth, so this validates
geometry extraction + structural realism (sanity), NOT Stage-4 prediction
accuracy. There is no IoU/precision here by design — see the roadmap.

Usage
-----
    python tools/run_msd_validation.py --msd-dir <dir-with-struct_in> --limit 50
    python tools/run_msd_validation.py --msd-dir .../train --sample 100 --stories 4
    python tools/run_msd_validation.py --msd-dir .../struct_in --wall-values 1,4

--msd-dir may point either at the `struct_in` folder itself or at a parent that
contains a `struct_in` subfolder (the extracted train dir).
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TOOLS = os.path.join(ROOT, "tools")
RESULTS_DIR = os.path.join(ROOT, "results", "msd_validation")
WORK = os.path.join(RESULTS_DIR, "_work")
STEP_TIMEOUT = 600


def _run(cmd, timeout=STEP_TIMEOUT):
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def find_struct_dir(msd_dir: str) -> str:
    """Accept either the struct_in folder or its parent."""
    if os.path.basename(os.path.normpath(msd_dir)) == "struct_in":
        return msd_dir
    cand = os.path.join(msd_dir, "struct_in")
    if os.path.isdir(cand):
        return cand
    # maybe the dir itself just contains .npy files
    if glob.glob(os.path.join(msd_dir, "*.npy")):
        return msd_dir
    raise FileNotFoundError(
        f"No struct_in folder or .npy files under {msd_dir}")


def run_one(npy_path, stories, floor_h, sw_threshold, wall_values,
            wall_channels, close_px, work=None, no_vis=False):
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    sub = os.path.join(RESULTS_DIR, stem)
    os.makedirs(sub, exist_ok=True)

    # Each worker needs its own scratch directory: run_one wipes it on entry,
    # so a shared one would have workers deleting each other's intermediates.
    # Keyed unconditionally on the process id -- on Windows the pool spawns
    # fresh processes that re-import this module, so anything captured at
    # import time is per-worker and cannot distinguish parent from child.
    work = work or f"{WORK}_{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    w = lambda f: os.path.join(work, f)  # noqa: E731

    # Step 0: MSD struct_in -> wall mask (replaces Stage 1)
    conv = ["tools/msd_to_mask.py", "-i", npy_path, "-o", w("output_mask.png")]
    if wall_values:
        conv += ["--wall-values", wall_values]
    if wall_channels:
        conv += ["--wall-channels", wall_channels]
    if close_px:
        conv += ["--close", str(close_px)]

    s2 = ["stages/stage_2/stage_2.py", "-i", w("output_mask.png"),
          "-o", w("floor_output.json")]
    s4 = ["stages/stage_4/stage_4.py", "-i", w("floor_consolidated.json"),
          "-o", w("floor_enriched.json"), "--sw-threshold", str(sw_threshold)]
    if not no_vis:
        s2 += ["-v", w("floor_vis.png")]
        s4 += ["-v", w("enrich_vis.png"),
               "--pred-vis", w("floor_predictions_vis.png"),
               "--graph-vis", w("gnn_graph_vis.png")]

    steps = [
        ("msd-mask", conv),
        ("stage2-vec", s2),
        ("consolidate", ["stages/stage_2/consolidate.py",
                         "-i", w("floor_output.json"),
                         "-o", w("floor_consolidated.json")]),
        ("stage4-sw", s4),
        ("stage5-fem", ["etabs/export.py", "-i", w("floor_enriched.json"),
                        "-o", w("model_fem.json"), "--stories", str(stories),
                        "--floor-height", str(floor_h)]),
        ("sanity", ["etabs/sanity.py", "-i", w("model_fem.json"),
                    "-o", w("sanity_report.json")]),
    ]
    if not no_vis:
        steps.insert(-1, ("fem-viz", ["etabs/fem_viz.py", "-i", w("model_fem.json"),
                                      "-o", w("model_fem_vis.png")]))

    reached = "-"
    for name, cmd in steps:
        try:
            rc, out, err = _run(cmd)
        except subprocess.TimeoutExpired:
            return reached, None, f"{name}: TIMEOUT"
        if rc != 0:
            tail = (err or out).strip().splitlines()[-4:]
            return reached, None, f"{name} rc={rc}: " + " | ".join(tail)
        reached = name

    for f in ("output_mask.png", "floor_vis.png", "floor_predictions_vis.png",
              "model_fem_vis.png", "sanity_report.json", "model_fem.json"):
        srcp = w(f)
        if os.path.exists(srcp):
            shutil.copy2(srcp, os.path.join(sub, f))

    try:
        with open(w("sanity_report.json")) as f:
            rep = json.load(f)
        return reached, rep, "OK"
    except Exception as e:  # noqa: BLE001
        return reached, None, f"no sanity report: {e}"


def main():
    ap = argparse.ArgumentParser(description="Sweep MSD plans through Stages 2-5 + sanity")
    ap.add_argument("--msd-dir", required=True,
                    help="path to struct_in/ (or a parent containing it)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap to first N plans when NOT sampling (0 = all)")
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly sample N plans (0 = take in sorted order)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stories", type=int, default=4)
    ap.add_argument("--floor-height", type=float, default=3.0)
    ap.add_argument("--sw-threshold", type=float, default=0.10)
    ap.add_argument("--wall-values", default=None,
                    help="struct_in label ids treated as wall (see msd_to_mask)")
    ap.add_argument("--wall-channels", default=None,
                    help="struct_in channels treated as wall (3D arrays)")
    ap.add_argument("--close", type=int, default=0,
                    help="morphological close px on the mask (bridge tiny gaps)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (plans are independent)")
    ap.add_argument("--no-vis", action="store_true",
                    help="skip the per-plan PNG renders (much faster)")
    ap.add_argument("--resume-after", default=None,
                    help="skip plans whose sanity_report.json is NEWER than this "
                         "timestamp, e.g. '2026-08-10 15:17'. Deliberately not a "
                         "bare --resume: results older than the cutoff are from a "
                         "previous configuration and MUST be recomputed, so "
                         "skipping on mere existence would silently mix runs.")
    args = ap.parse_args()

    struct_dir = find_struct_dir(args.msd_dir)
    files = sorted(glob.glob(os.path.join(struct_dir, "*.npy")))
    if not files:
        print(f"No .npy files in {struct_dir}")
        return
    if args.sample and args.sample < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.sample)
        files.sort()
    elif args.limit and args.limit > 0:
        # --limit caps only when not sampling (sample size is authoritative)
        files = files[:args.limit]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume: keep only plans not already computed under the current
    # configuration. Anything older than the cutoff is stale and is redone.
    done = {}
    if args.resume_after:
        cutoff = time.mktime(time.strptime(args.resume_after, "%Y-%m-%d %H:%M"))
        keep = []
        for npy in files:
            stem = os.path.splitext(os.path.basename(npy))[0]
            rep = os.path.join(RESULTS_DIR, stem, "sanity_report.json")
            if os.path.exists(rep) and os.path.getmtime(rep) > cutoff:
                done[stem] = rep
            else:
                keep.append(npy)
        print(f"resume: {len(done)} plan(s) already computed after "
              f"{args.resume_after}; {len(keep)} to run")
        files = keep

    print(f"MSD validation: {len(files)} plan(s) from {struct_dir}\n"
          f"stories={args.stories} floor_h={args.floor_height} "
          f"sw_thr={args.sw_threshold} jobs={args.jobs} "
          f"vis={'off' if args.no_vis else 'on'} -> {RESULTS_DIR}\n")

    rows = []
    scores = []
    check_fail = collections.Counter()
    check_warn = collections.Counter()
    t_all = time.time()

    def absorb(stem, reached, rep, status, tag=""):
        if rep is not None:
            score = float(rep.get("score", float("nan")))
            n_warn = int(rep.get("n_warn", -1))
            n_fail = int(rep.get("n_fail", -1))
            scores.append(score)
            for c in rep.get("checks", []):
                nm, st = c.get("name", "?"), c.get("status")
                if st == "fail":
                    check_fail[nm] += 1
                elif st == "warn":
                    check_warn[nm] += 1
            if tag:
                print(f"{tag} OK  sanity={score:.2f} (w={n_warn} f={n_fail})")
        else:
            score, n_warn, n_fail = float("nan"), -1, -1
            if tag:
                print(f"{tag} XX  {status}")
        rows.append({"id": stem, "reached": reached, "score": score,
                     "n_warn": n_warn, "n_fail": n_fail, "status": status})

    # fold in results carried over by --resume-after
    for stem, rep_path in sorted(done.items()):
        try:
            with open(rep_path) as f:
                absorb(stem, "sanity", json.load(f), "OK")
        except Exception as e:  # noqa: BLE001
            absorb(stem, "-", None, f"resume: unreadable report: {e}")

    task = (args.stories, args.floor_height, args.sw_threshold,
            args.wall_values, args.wall_channels, args.close)

    if args.jobs > 1 and files:
        import concurrent.futures as cf
        n = len(files)
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, npy, *task, None, args.no_vis):
                    os.path.splitext(os.path.basename(npy))[0] for npy in files}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                stem = futs[fut]
                try:
                    reached, rep, status = fut.result()
                except Exception as e:  # noqa: BLE001
                    reached, rep, status = "-", None, f"runner error: {e}"
                absorb(stem, reached, rep, status, tag=f"[{i}/{n}] {stem:>10s}")
    else:
        for i, npy in enumerate(files, 1):
            stem = os.path.splitext(os.path.basename(npy))[0]
            try:
                reached, rep, status = run_one(npy, *task, None, args.no_vis)
            except Exception as e:  # noqa: BLE001
                reached, rep, status = "-", None, f"runner error: {e}"
            absorb(stem, reached, rep, status,
                   tag=f"[{i}/{len(files)}] {stem:>10s}")

    # tidy the per-worker scratch directories
    for d in glob.glob(WORK + "_*"):
        shutil.rmtree(d, ignore_errors=True)

    rows.sort(key=lambda r: r["id"])

    # write scores.csv
    csv_path = os.path.join(RESULTS_DIR, "scores.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(
            f, fieldnames=["id", "reached", "score", "n_warn", "n_fail", "status"])
        wtr.writeheader()
        wtr.writerows(rows)

    # distribution summary
    n_ok = len(scores)
    lines = ["", "=" * 72, "  MSD VALIDATION SUMMARY", "=" * 72,
             f"  plans run        : {len(rows)}",
             f"  reached sanity   : {n_ok}",
             f"  failed pipeline  : {len(rows) - n_ok}"]
    if scores:
        scores_sorted = sorted(scores)
        def q(p):
            return scores_sorted[min(len(scores_sorted) - 1,
                                     int(p * (len(scores_sorted) - 1)))]
        lines += [
            "-" * 72,
            "  sanity score distribution (0..1, higher = more structurally sane)",
            f"    mean   : {statistics.mean(scores):.3f}",
            f"    median : {statistics.median(scores):.3f}",
            f"    min    : {min(scores):.3f}",
            f"    p25    : {q(0.25):.3f}",
            f"    p75    : {q(0.75):.3f}",
            f"    max    : {max(scores):.3f}",
            f"    >=0.80 : {100.0*sum(s>=0.80 for s in scores)/n_ok:.0f}% of plans",
            f"    >=0.90 : {100.0*sum(s>=0.90 for s in scores)/n_ok:.0f}% of plans",
        ]
    # which sanity checks fail most (the real validation signal)
    if scores and (check_fail or check_warn):
        lines += ["-" * 72, "  sanity-check breakdown (of %d scored plans):" % n_ok]
        allnames = sorted(set(check_fail) | set(check_warn),
                          key=lambda k: -(check_fail[k]*2 + check_warn[k]))
        for nm in allnames:
            lines.append("    {:<18s} fail={:>3d} ({:>3.0f}%)  warn={:>3d} ({:>3.0f}%)".format(
                nm, check_fail[nm], 100.0*check_fail[nm]/n_ok,
                check_warn[nm], 100.0*check_warn[nm]/n_ok))

    # failure reasons
    fails = [r for r in rows if r["status"] != "OK"]
    if fails:
        lines += ["-" * 72, "  pipeline failures (stage + reason):"]
        for r in fails[:20]:
            lines.append(f"    {r['id']:>10s}  reached={r['reached']:12s} "
                         f"{r['status'][:48]}")
    lines += ["-" * 72,
              f"  scores.csv -> {csv_path}",
              f"  elapsed    : {time.time()-t_all:.0f}s", "=" * 72]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
