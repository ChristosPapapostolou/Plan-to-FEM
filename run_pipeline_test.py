"""
Whole-pipeline test runner (no ETABS push)
===========================================
Runs every image in `images/` through the full pipeline:

  Stage 1  segmentation        -> output_mask.png
  Stage 2  vectorization       -> floor_output.json
  consolidate                  -> floor_consolidated.json
  Stage 4  SW prediction       -> floor_enriched.json + floor_predictions_vis.png
  Stage 5  FEM export (JSON)    -> model_fem.json        (NO --push-etabs)
  fem_viz  render              -> model_fem_vis.png

For each image it creates  results/<image_name>/  containing the
validation outputs:
    output_mask.png           Stage 1 wall mask
    floor_vis.png             Stage 2 vectorized geometry (walls/points/rects)
    floor_predictions_vis.png Stage 4 SW selection + GNN columns
    model_fem_vis.png         Stage 5 FEM model render
    sanity_report.json        step-5 structural checks + score

Intermediate files live in results/_work/ (overwritten per image).

Usage:
  .\\venv\\Scripts\\python.exe run_pipeline_test.py
  .\\venv\\Scripts\\python.exe run_pipeline_test.py --stories 5 --only floor_plan_1
"""

from __future__ import annotations
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
IMAGES_DIR = os.path.join(ROOT, "images")
RESULTS_DIR = os.path.join(ROOT, "results")
WORK = os.path.join(RESULTS_DIR, "_work")

FINAL_IMAGES = ["output_mask.png", "floor_vis.png", "gnn_graph_vis.png",
                "floor_predictions_vis.png", "model_fem_vis.png"]
STEP_TIMEOUT = 600  # seconds per stage


def _run(cmd, timeout=STEP_TIMEOUT):
    """Run a python subprocess from ROOT; return (rc, stdout, stderr)."""
    p = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run_one(img_path, stories, floor_h, sw_threshold):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    sub = os.path.join(RESULTS_DIR, stem)
    os.makedirs(sub, exist_ok=True)

    # fresh scratch dir
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)

    w = lambda f: os.path.join(WORK, f)  # noqa: E731
    steps = [
        ("stage1-seg", ["stages/stage_1/stage_1.py", "-i", img_path,
                        "--out-dir", WORK]),
        ("stage2-vec", ["stages/stage_2/stage_2.py", "-i", w("output_mask.png"),
                        "-o", w("floor_output.json"), "-v", w("floor_vis.png")]),
        ("consolidate", ["stages/stage_2/consolidate.py", "-i", w("floor_output.json"),
                         "-o", w("floor_consolidated.json")]),
        ("stage4-sw", ["stages/stage_4/stage_4.py", "-i", w("floor_consolidated.json"),
                       "-o", w("floor_enriched.json"), "-v", w("enrich_vis.png"),
                       "--pred-vis", w("floor_predictions_vis.png"),
                       "--graph-vis", w("gnn_graph_vis.png"),
                       "--sw-threshold", str(sw_threshold)]),
        ("stage5-fem", ["etabs/export.py", "-i", w("floor_enriched.json"),
                        "-o", w("model_fem.json"), "--stories", str(stories),
                        "--floor-height", str(floor_h)]),
        ("fem-viz", ["etabs/fem_viz.py", "-i", w("model_fem.json"),
                     "-o", w("model_fem_vis.png")]),
        ("sanity", ["etabs/sanity.py", "-i", w("model_fem.json"),
                    "-o", w("sanity_report.json")]),
    ]

    reached = "-"
    for name, cmd in steps:
        t0 = time.time()
        try:
            rc, out, err = _run(cmd)
        except subprocess.TimeoutExpired:
            print(f"    {name:12s} TIMEOUT")
            return reached, f"{name}: TIMEOUT"
        dt = time.time() - t0
        if rc != 0:
            print(f"    {name:12s} FAIL  ({dt:.0f}s)")
            tail = (err or out).strip().splitlines()[-6:]
            return reached, f"{name} rc={rc}: " + " | ".join(tail)
        reached = name
        print(f"    {name:12s} ok    ({dt:.0f}s)")

    # collect the three validation images + sanity report
    missing = []
    for f in FINAL_IMAGES + ["sanity_report.json", "model_fem.json"]:
        srcp = w(f)
        if os.path.exists(srcp):
            shutil.copy2(srcp, os.path.join(sub, f))
        else:
            missing.append(f)
    if missing:
        return reached, "missing outputs: " + ", ".join(missing)

    # surface the sanity score in the summary line
    try:
        import json as _json
        with open(w("sanity_report.json")) as f:
            rep = _json.load(f)
        return reached, (f"OK  sanity={rep['score']:.2f} "
                         f"(w={rep['n_warn']} f={rep['n_fail']})")
    except Exception:
        return reached, "OK"


def main():
    ap = argparse.ArgumentParser(description="Run the whole pipeline on all images (no ETABS)")
    ap.add_argument("--images-dir", default=IMAGES_DIR)
    ap.add_argument("--stories", type=int, default=3)
    ap.add_argument("--floor-height", type=float, default=3.0)
    ap.add_argument("--sw-threshold", type=float, default=0.10)
    ap.add_argument("--only", default=None,
                    help="Run only images whose name contains this substring")
    args = ap.parse_args()

    imgs = sorted(
        f for ext in ("*.png", "*.jpg", "*.jpeg")
        for f in glob.glob(os.path.join(args.images_dir, ext))
    )
    if args.only:
        imgs = [f for f in imgs if args.only in os.path.basename(f)]
    if not imgs:
        print(f"No images found in {args.images_dir}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Running {len(imgs)} image(s) | stories={args.stories} "
          f"floor_h={args.floor_height} | results -> {RESULTS_DIR}\n")

    results = []
    t_all = time.time()
    for i, img in enumerate(imgs, 1):
        name = os.path.basename(img)
        print(f"[{i}/{len(imgs)}] {name}")
        try:
            reached, status = run_one(img, args.stories, args.floor_height,
                                      args.sw_threshold)
        except Exception as e:  # noqa: BLE001
            reached, status = "-", f"runner error: {e}"
        results.append((name, reached, status))
        print(f"    => {status}\n")

    # summary
    lines = ["", "=" * 72, "  PIPELINE TEST SUMMARY", "=" * 72]
    n_ok = sum(1 for _, _, s in results if s.startswith("OK"))
    for name, reached, status in results:
        ok = status.startswith("OK")
        flag = "OK " if ok else "XX "
        lines.append(f"  {flag} {name:24s} reached={reached:12s} "
                     f"{(status[4:] if ok else status)[:60]}")
    lines.append("-" * 72)
    lines.append(f"  {n_ok}/{len(results)} succeeded in {time.time()-t_all:.0f}s")
    lines.append("=" * 72)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
