# Plan-to-FEM: reproducibility deposit

Companion deposit for the manuscript submitted to *Automation in Construction*.

It contains the identifiers of every data split used, the machine-readable
output behind every reported number, the analysis scripts that produced
them, and checksums for the trained checkpoints.


## Contents

| Directory | What it holds |
|---|---|
| `splits/` | Identifier lists for every split, including the deduplicated test set |
| `results/` | JSON outputs; one file per experiment |
| `analysis/` | The scripts that produced them |
| `checkpoints/` | SHA-256 checksums; the binaries themselves are archived separately |

## Provenance of reported numbers

| Script | Output | Establishes |
|---|---|---|
| `analysis/eval_stage1.py` | `results/stage1_eval.json` | Stage-1 wall segmentation on the CubiCasa5K test split, as deployed |
| `analysis/eval_refsubset.py` | `results/stage1_refsubset.json` | Stage-1 on the reference subset used to probe the unrecoverable 88.6% figure |
| `analysis/eval_cubicasa.py` | `results/cubicasa_eval.json` | CubiCasa5K Dice and IoU with bootstrap confidence intervals |
| `analysis/eval_cubicasa_deployed.py` | `results/cubicasa_deployed.json` | the same evaluation through the deployed pre-processing path |
| `analysis/bench_stage2.py` | `results/stage2_bench.json` | Stage-2 vectorisation accuracy against SVG wall polygons |
| `analysis/probe_scale.py` | `results/raw_factors.json` | raw scale factors before prior-based calibration |
| `analysis/bench_scale.py` | `results/scale_bench.json` | scale estimation error, before and after the deadband correction |
| `analysis/sweep_deadband.py` | `results/deadband_sweep.json` | the deadband sweep that justifies the 0.50 setting |
| `analysis/split_audit.py` | `results/split_audit.json` | graph statistics and the perceptual-hash leakage audit of the train/test split |
| `analysis/clean_split.py` | `results/clean_split.json` | results on the original 72, the deduplicated 24 and the clean 21 drawings |
| `analysis/indep_test.py` | `results/indep_test.json` | evaluation on the independent held-out graphs |
| `analysis/membership_test.py` | `results/membership.json` | train/test membership check behind the leakage discussion |
| `analysis/threshold_free.py` | `results/threshold_free.json` | ROC AUC, average precision and calibration error for both GNN heads |
| `analysis/sweep_psw.py` | `results/psw_sweep.json` | sensitivity to the minimum pier thickness |
| `analysis/sweep_grid.py` | `results/grid_sweep.json` | sensitivity to structural-grid spacing |
| `analysis/sweep_nms.py` | `results/nms_sweep.json` | the column non-maximum-suppression radius sweep |
| `analysis/ablate_prior.py` | `results/prior_ablation.json` | the rules-only versus deployed shear-wall ablation |
| `analysis/weight_sens.py` | `results/weight_sens.json` | sensitivity of the coherence score to check weighting |
| `analysis/density.py` | `results/density.json` | member density against tributary area |
| `analysis/quantities.py` | `results/quantities.json` | concrete volume per square metre |
| `analysis/rebatch.py` | `results/rebatch.json` | the ten-plan batch re-run after the checkpoint switch |
| `analysis/check_align.py` | `results/target_probe.json` | the beam-support tolerance mismatch diagnosis |

## Checkpoints

- `stages/stage_1/mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth` — 257.4 MB, `sha256:9c56c86723b0b509…`
- `train/gnn_multi1.pt` — 0.1 MB, `sha256:4d6645885bf13790…`
- `train/gnn_multi.pt` — 0.1 MB, `sha256:3297ae799f276749…`
- `train/gnn_ep4.pt` — 0.1 MB, `sha256:2bd63fadd4bdf0c3…`

The deployed graph network is `train/gnn_multi1.pt`. The Stage-1 segmentation
checkpoint is distributed as a binary artefact only: the training records that
produced it could not be recovered, so its recipe is not reproducible from this
deposit and the paper reports the checkpoint's measured performance on
CubiCasa5K instead of its original training metric. This limitation is stated
in the manuscript and is repeated here so that the deposit does not imply more
than it delivers.


## Third-party data

**StructGAN** (`train/data/`) — Shear-wall design pairs from the StructGAN corpus. NOT redistributed. Obtain from the original authors. This deposit contains only the identifiers of the pairs used in each split (splits/structgan_*.txt).

**CubiCasa5K** (`cubicasa5k/`) — Publicly available under its own licence. NOT redistributed. The official 4199/399/399 split files ship with the dataset and are used unmodified.

**Modified Swiss Dwellings (MSD)** (`data/msd/`) — Publicly available under its own licence. NOT redistributed. Plan identifiers used are listed in splits/msd_plan_ids.txt.


## Environment

`requirements-lock.txt` in the repository root pins the exact package versions
used for every result reported in the paper.


## Known limitation of these scripts

The analysis scripts were written against absolute paths on the development
machine. Each defines `ROOT` and `SCR` constants at the top; both must be set
to local paths before the script will run. They are deposited as they were run
rather than tidied afterwards, so that the deposited code is the code that
produced the numbers.
