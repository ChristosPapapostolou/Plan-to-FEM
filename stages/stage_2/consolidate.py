"""
Apply wall-rect consolidation to an existing Stage 2 floor JSON.

De-fragments over-segmented walls (merges collinear same-wall rects into one
continuous rect) WITHOUT re-running Stage 1 segmentation / Stage 2 vectorization.

Usage:
  python stages/stage_2/consolidate.py -i floor_output.json
  python stages/stage_2/consolidate.py -i floor_output.json -o floor_consolidated.json

If -o is omitted the input is overwritten and the original is backed up to
<input>_unconsolidated.json (once).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_2 import consolidate_wall_rects  # noqa: E402


def _count(floor):
    return sum(len(w.get("rects", [])) for w in floor.get("walls", [])
               if not w.get("rejected"))


def main():
    ap = argparse.ArgumentParser(description="Consolidate Stage 2 wall rects")
    ap.add_argument("-i", "--input", required=True, help="Stage 2 floor JSON")
    ap.add_argument("-o", "--output", default=None,
                    help="Output path (default: overwrite input, with backup)")
    args = ap.parse_args()

    with open(args.input) as f:
        floor = json.load(f)

    before = _count(floor)
    consolidated = consolidate_wall_rects(floor)
    after = _count(consolidated)

    out = args.output or args.input
    if out == args.input:
        bak = os.path.splitext(args.input)[0] + "_unconsolidated.json"
        if not os.path.exists(bak):
            with open(bak, "w") as f:
                json.dump(floor, f, indent=2)
            print(f"Backed up original -> {bak}")

    with open(out, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"Rects: {before} -> {after}  ({100*(before-after)/max(before,1):.0f}% fewer)")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
