# Plan-to-FEM

Automated generation of finite element models from architectural floor plans,
using a multi-task graph neural network with engineering-rule validation.

Code accompanying the manuscript submitted to *Automation in Construction*.

## What it does

A single raster floor-plan image is converted into a
three-dimensional finite element model through five stages, each of which
writes one inspectable artefact:

| Stage | Input | Output |
|---|---|---|
| 1. Segmentation | raster plan | binary wall mask |
| 2. Vectorisation | wall mask | wall rectangles, metric scale recovered |
| 3. Graph construction | rectangles | typed Edge-PSW-DW graph + structural grid |
| 4. Enrichment | graph | shear-wall ratios on edges, columns on nodes |
| 5. Export | enriched layout | ETABS/JSON finite element model |

A six-check heuristic module then scores the exported model for structural
coherence: lateral-line count, stiffness eccentricity, beam support, slab span,
column continuity and orphan nodes.

**The pipeline automates model generation, not structural design.** A coherence
score of 1.0 means the six heuristic checks passed; it is not a statement of
structural safety or adequacy, and it does not reduce the need for engineer
review.

### Independent solver check

`tools/opensees_check.py` rebuilds an exported model in OpenSeesPy and runs
gravity, modal and equivalent-lateral-force analyses; it reports floating
members (no element path to the base), deflections, periods, wall shear shares,
torsion ratios and the equilibrium error. `tools/opensees_batch.py` runs it over
a directory of models. This is a linear plausibility check, not a code-compliant
design.

### Joint connectivity and orientation

- `etabs/connect.py` merges member ends onto shared joints so that every member
  has a load path; it is on by default (`--no-connect` disables it).
- Stages 2 and 4, the exporter and the coherence checks work in the principal
  frame of the wall layout, so plans drawn rotated on the sheet keep their wall
  directions; `--no-principal-frame` restores drawing-axis processing. For
  drawings within 5 degrees of the axes the step is inactive.

### Outputs behind the paper

`release/outputs/` holds the machine-readable outputs read by the manuscript:
`numbers.json` (every number and table in the text), `figdata.json` (figure
data), `msd_per_plan.csv` (per-plan scores, checks and solver results for the
4,572 MSD plans), `msd_onmask.csv`, `solve_ten_plans.jsonl`,
`solve_msd150_without_connectivity.jsonl` and `msd_gallery_selection.json`.
`release/analysis/paper/` contains the MSD sweep worker and the aggregation
script that produced them (paths at the top of each script must be set).

## Layout

```
stages/        the five pipeline stages
models/        GraphSAGE-Frame multi-task network
etabs/         FE export, COM handler, coherence checks
train/         training scripts and the synthetic design-loop generator
tools/         MSD conversion and the large-scale validation sweep
finetune_stage1/  segmentation fine-tuning
release/       reproducibility deposit: splits, results, analysis scripts
```

`release/README.md` maps every reported quantity to the script that produced it
and the JSON file it was read from.

## Getting started

```bash
python -m venv venv
venv/Scripts/activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the exact versions used for every result in the
paper. `requirements.txt` carries loose bounds if you would rather resolve
fresh.

Running one plan end to end:

```bash
python stages/stage_1/stage_1.py -i plan.png -o mask.png
python stages/stage_2/stage_2.py -i mask.png -o floor.json
python stages/stage_2/consolidate.py -i floor.json -o consolidated.json
python stages/stage_4/stage_4.py -i consolidated.json -o enriched.json
python etabs/export.py -i enriched.json -o model_fem.json --stories 3 --floor-height 3.0
python etabs/sanity.py -i model_fem.json -o sanity_report.json
```

The large-scale sweep, parallel across plans:

```bash
python tools/run_msd_validation.py --msd-dir <path>/struct_in --jobs 8 --no-vis
```

## Checkpoints

| File | Size | Where |
|---|---|---|
| `train/gnn_multi1.pt` | 84 KB | this repository — **the deployed network** |
| `train/gnn_multi.pt` | 84 KB | this repository |
| `train/gnn_ep4.pt` | 77 KB | this repository, earlier single-task model |
| Stage-1 segmentation `.pth` | 257 MB | Zenodo record — over GitHub's file limit |

`release/checkpoints/checksums.json` carries SHA-256 for all four so a download
can be verified.

One limitation is stated here as it is in the paper: the training records for
the Stage-1 checkpoint could not be recovered. It is released as a binary
artefact whose recipe is not reproducible from this repository, which is why
the paper reports its measured performance on CubiCasa5K rather than its
original training metric.

## Data

No third-party dataset is redistributed here. StructGAN must be obtained from
its original authors; CubiCasa5K and Modified Swiss Dwellings are publicly
available under their own licences. `release/splits/` gives the identifiers of
every drawing used in each split, including the deduplicated test list that
supersedes the triplicated one described in the paper.

See `release/DATA_AVAILABILITY.md`.

## Citation

See `CITATION.cff`. The archived release carries a DOI; both are added here on
publication.

## Licence

MIT, for the code, the trained weights and the derived results. Third-party
datasets are not covered — see `LICENSE`.
