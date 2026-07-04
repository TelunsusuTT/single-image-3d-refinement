# Phase 2L.2C Data v2 Frame Panel Curation and Splits

## Goal

Phase 2L.2C turns the visually inspected manual ABO framed-art set into a
curated dataset manifest and reproducible train/validation/test split files.
This is still a preparation step: it does not render Hunyuan3D-Paint training
examples, run Hunyuan, run Blender, train, submit Slurm jobs, or load
checkpoints.

## Why Curate Before Training

The manual ABO set is semantically strong, but training should only use assets
that pass local checks and have input-view metadata. The curated manifest keeps
only assets that were found locally, imported in Blender, rendered all six
preview views, and were not listed in the optional reject list.

The default Data v2A curation policy is:

- `subclass=framed_wall_art`
- `human_decision=accept`
- `selected_input_view=005`
- `alternative_input_view=004`
- `primary_eval_views=004;005`

These defaults reflect the Phase 2K.4 finding that informative reference views
matter. The values can be revised later by human curation if a specific asset
is better represented by another view.

## Why Not Original Order

The original manual item-id list was produced by web browsing and user
selection order. That order may contain bursts of visually similar products,
near duplicates, or repeated sellers/styles. It must not define train/test
membership. Phase 2L.2C uses fixed-seed randomization and records split
membership so later training and evaluation are reproducible.

## Mini40 Split

The `mini40` split is for fast diagnostic training and early corrected-input
benchmarking:

- 32 train
- 4 validation
- 4 test

It should be the first split used for Data v2A training probes.

## Full101 Split

The `full101` split is for later larger training after the mini40 path works:

- 80 train
- 10 validation
- 11 test

Do not submit full101 training until mini40 training and evaluation succeeds.

## Leakage Precautions

Framed wall art assets can be near duplicates: identical frame geometry with
different art, repeated aspect ratios, or closely related product variants. The
curation script assigns a simple `group_key` using rounded bounding-box extents,
a face-count bin, and an approximate aspect ratio. The split script keeps
matching group keys in the same split where exact split sizes allow it, and
records any group that must be split to preserve requested counts.
