# MSD validation harness

Validate the geometry + FEM half of the pipeline against **Modified Swiss
Dwellings v2** (real Swiss residential low/mid-rise — closer to the target
typology than StructGAN). Stage 1 is skipped: MSD ships the structural-wall
raster directly.

## Verified MSD v2 `struct_in` encoding
Each `struct_in/<id>.npy` is `(512, 512, 3)` float16:

| channel | content | use |
|---|---|---|
| 0 | binary {0, 255} structural raster, **walls = 0** (~2–7% of pixels) | the wall mask |
| 1 | per-pixel x-coordinate field (meshgrid, ≈ −20…20 m) | ignored |
| 2 | per-pixel y-coordinate field | ignored |

`msd_to_mask.py` auto-detects this: structural channel = 0, wall = the minority
value (polarity-safe). **No flags needed.** Confirmed across sampled plans and
verified visually (walls render as a clean floor-plan outline).

## Files
- `msd_to_mask.py` — one `struct_in/<id>.npy` → a `255=wall` grayscale PNG
  (byte-compatible with Stage 1's `output_mask.png`).
- `run_msd_validation.py` — sweep N plans through Stage 2 → consolidate →
  Stage 4 → export → sanity; writes a sanity-score distribution.

## Run
```powershell
# inspect one file (optional — confirms the channel 0 auto-detection)
python tools/msd_to_mask.py -i data\msd\modified-swiss-dwellings-v2\train\struct_in\10000.npy --inspect

# full sweep (auto-handles MSD v2; no wall-channel flags needed)
python tools/run_msd_validation.py --msd-dir data\msd\modified-swiss-dwellings-v2\train --sample 100 --stories 4
```
`--msd-dir` may point at the `struct_in` folder or its parent.

## Outputs
```
results/msd_validation/<id>/       per-plan mask, floor_vis, predictions, FEM vis, sanity_report.json
results/msd_validation/scores.csv  id, reached, score, n_warn, n_fail, status
results/msd_validation/summary.txt mean/median/quartiles, %>=0.80, %>=0.90, failures
```

## What this validates (and what it doesn't)
- **Validates:** Stage 2 vectorization + Stage 3 graph + Stage 5 structural
  realism, across thousands of real residential plans → a `sanity` score
  *distribution* = a quantitative generalization result.
- **Does NOT validate:** Stage-4 prediction accuracy. MSD has no shear-wall or
  column ground truth, so no IoU/precision/recall here by design.

## Caveats observed on real data
- **Scale**: the Stage 2 auto-calibrator fires on MSD (≈2.8× on plan 10000) —
  leave it on, do **not** pass `--scale`.
- **Thin walls**: MSD wall lines are ~1–2 px, so post-calibration thicknesses
  can come out ~0.05–0.09 m (below the 0.20 m target). Sanity scores may be
  penalised on thin-wall plans; if so, raise the calibrator target or add a
  `--close 2` on the mask to thicken lines before vectorizing. Check a handful
  of `floor_vis.png` before trusting the aggregate.
