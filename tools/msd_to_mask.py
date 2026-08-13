"""
MSD struct_in -> Stage-2 wall mask adapter
==========================================
Converts a Modified Swiss Dwellings (MSD) struct_in/<index>.npy raster into a
binary wall mask PNG byte-compatible with Stage 1's output_mask.png (grayscale,
uint8, 255 = wall, 0 = else). Lets you skip Stage 1 and feed real Swiss
structural-wall geometry straight into Stage 2's vectorizer.

MSD v2 struct_in encoding (verified against the downloaded train split):
    shape (512, 512, 3), float16
    channel 0 : BINARY {0, 255} structural raster; WALLS = 0 (~2-7% of pixels)
    channel 1 : per-pixel x-coordinate field (meshgrid, ~ -20..20 m)
    channel 2 : per-pixel y-coordinate field (meshgrid, ~ -20..20 m)
Walls live in channel 0 as the MINORITY value. The default path picks them with
NO flags (structural channel = 0, wall = minority value, polarity-safe).
Channels 1/2 are coordinate ramps and are ignored.

Usage:
    python tools/msd_to_mask.py -i struct_in/10000.npy --inspect
    python tools/msd_to_mask.py -i struct_in/10000.npy -o mask.png
Overrides: --struct-channel N, --wall-values a,b, --wall-channels a,b,
           --invert, --close K
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def load_struct_in(path):
    if not os.path.exists(path):
        raise FileNotFoundError("struct_in not found: " + path)
    return np.load(path, allow_pickle=False)


def _to_hwc(arr):
    cax = int(np.argmin(arr.shape))
    return np.moveaxis(arr, cax, -1)


def inspect(arr):
    lines = ["shape={}  dtype={}  min={}  max={}".format(
        arr.shape, arr.dtype, arr.min(), arr.max())]
    if arr.ndim == 2:
        vals, counts = np.unique(arr, return_counts=True)
        total = arr.size
        lines.append("unique values (value: pixels, pct):")
        for v, c in zip(vals, counts):
            lines.append("    {:>6d}: {:>10d}  ({:5.2f}%)".format(
                int(v), int(c), 100.0 * c / total))
        lines.append("  -> 2D label raster. Use --wall-values, or default "
                     "(all non-zero = wall).")
    elif arr.ndim == 3:
        moved = _to_hwc(arr)
        C = moved.shape[-1]
        lines.append("  -> 3D array, {} channels:".format(C))
        struct_ch = None
        for ch in range(C):
            band = moved[..., ch]
            u = np.unique(band)
            nz = int(np.count_nonzero(band))
            if len(u) <= 3:
                mino = int(min((band == v).sum() for v in u))
                tag = "  [BINARY {} -> STRUCTURAL; walls=minority {:.1f}%]".format(
                    list(np.round(u, 1)), 100.0 * mino / band.size)
                if struct_ch is None:
                    struct_ch = ch
            else:
                tag = "  [{} distinct, range {:.1f}..{:.1f} -> coord/other]".format(
                    len(u), float(band.min()), float(band.max()))
            lines.append("    channel {}: {} nonzero ({:5.2f}%){}".format(
                ch, nz, 100.0 * nz / band.size, tag))
        if struct_ch is not None:
            lines.append("  -> DEFAULT uses channel {} (minority value = wall); "
                         "no flags needed for MSD v2.".format(struct_ch))
        else:
            lines.append("  -> no binary channel found; set --wall-channels.")
    else:
        lines.append("  -> unexpected ndim; inspect manually.")
    return "\n".join(lines)


def to_wall_mask(arr, wall_values=None, wall_channels=None, struct_channel=0,
                 invert=False, close_px=0):
    if arr.ndim == 3 and wall_channels is not None:
        moved = _to_hwc(arr)
        sel = np.zeros(moved.shape[:2], dtype=bool)
        for ch in wall_channels:
            sel |= moved[..., ch] > 0
    elif wall_values is not None:
        a2d = _to_hwc(arr).any(axis=-1).astype(arr.dtype) if arr.ndim == 3 else arr
        sel = np.isin(a2d, wall_values)
    elif arr.ndim == 3:
        ch = _to_hwc(arr)[..., struct_channel]
        vals, counts = np.unique(ch, return_counts=True)
        wall_val = vals[int(np.argmin(counts))]
        sel = ch == wall_val
    else:
        sel = arr != 0

    if invert:
        sel = ~sel

    mask = np.where(sel, 255, 0).astype(np.uint8)

    if close_px and close_px > 0:
        if cv2 is None:
            raise RuntimeError("--close needs opencv (cv2) installed")
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask


def write_mask(mask, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(out_path, mask)
    else:
        from PIL import Image
        Image.fromarray(mask).save(out_path)


def _parse_int_list(s):
    if not s:
        return None
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert an MSD struct_in .npy into a Stage-2 wall mask PNG.")
    ap.add_argument("--input", "-i", required=True, help="struct_in .npy path")
    ap.add_argument("--output", "-o", default=None, help="output mask .png")
    ap.add_argument("--inspect", action="store_true", help="print info and exit")
    ap.add_argument("--wall-values", default=None, help="wall label ids (2D)")
    ap.add_argument("--wall-channels", default=None, help="wall channels (3D)")
    ap.add_argument("--struct-channel", type=int, default=0, help="binary raster channel")
    ap.add_argument("--invert", action="store_true", help="flip selection")
    ap.add_argument("--close", type=int, default=0, help="morph close px")
    args = ap.parse_args(argv)

    arr = load_struct_in(args.input)
    if args.inspect:
        print(inspect(arr))
        return 0

    mask = to_wall_mask(
        arr,
        wall_values=_parse_int_list(args.wall_values),
        wall_channels=_parse_int_list(args.wall_channels),
        struct_channel=args.struct_channel,
        invert=args.invert,
        close_px=args.close,
    )
    out = args.output or (os.path.splitext(args.input)[0] + "_mask.png")
    write_mask(mask, out)
    frac = 100.0 * (mask > 0).mean()
    print("wrote {}  ({}x{}, wall={:.2f}% of pixels)".format(
        out, mask.shape[1], mask.shape[0], frac))
    return 0


if __name__ == "__main__":
    sys.exit(main())
