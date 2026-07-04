# Phase 2L.1B ABO Geometry Candidate Refinement

## Goal

Phase 2L.1B improves Data v2 candidate ranking for local ABO metadata by using
geometry-aware flat-panel heuristics from `data/metadata/abo/3dmodels.csv.gz`.

This phase does not download assets, run Blender, run Hunyuan, submit Slurm
jobs, train, or load checkpoints.

## Why Phase 2L.1A Was Insufficient

The Phase 2L.1A generic metadata miner worked structurally, but the local ABO
inventory currently exposes only:

- `data/metadata/abo/3dmodels.csv.gz`
- `data/metadata/abo/images.csv.gz`

The asset-level ABO `3dmodels.csv.gz` file has useful geometry, texture,
material, and image-count fields, but it does not provide semantic title,
category, or tag fields. As a result, generic text scoring produced empty
titles, no positive or negative keyword hits, and many tied scores.

## Why Geometry-Aware Filtering

ABO still has useful signals for the flat rectangular graphic panel subclass.
The extents can estimate whether an object is flat and panel-like:

- Sort `extent_x`, `extent_y`, and `extent_z` as `small <= mid <= large`.
- Estimate flatness with `flatness_ratio = small / max(mid, large)`.
- Estimate panel shape with `panel_aspect_ratio = large / max(mid, eps)`.

A likely panel should be thin in one dimension, rectangular in the other two
dimensions, textured, materialized, and not an extreme long rod. Face count and
mesh count are used as technical sanity checks, not as final acceptance rules.

## Candidate Ranking Only

This geometry scorer is still only a ranking tool. It cannot prove that an ABO
asset is a framed poster, wall art, plaque, sign, canvas print, book-cover-like
board, or package-front-like panel. It also cannot determine the final
`selected_input_view`.

Human review remains required before any download, Blender inspection, contact
sheet, rendering, or training step.

## Inputs and Outputs

Candidate rows come only from:

- `data/metadata/abo/3dmodels.csv.gz`

Auxiliary metadata such as `images.csv.gz` may be inventoried, but image rows
must not be treated as asset candidates.

The geometry miner writes:

- `data/candidates/datav2_abo_geometry_flat_panel_candidates.csv`
- `data/candidates/datav2_abo_geometry_flat_panel_candidates.md`
- `data/candidates/datav2_abo_geometry_flat_panel_summary.json`

The review-template tool writes:

- `data/candidates/datav2_abo_geometry_human_review_template.csv`

Generated CSVs and reports are planning artifacts and should not be committed
unless explicitly requested.

## Next Phase

After inspecting the top candidates, the next phase should curate a small
reviewed list. Only reviewed candidates should move to asset download,
inspection, contact-sheet generation, selected-input-view assignment, and
corrected-input base benchmarking.
