# Phase 2L.5B/C Mini40 Rendered-View Evaluation

## Goal

Evaluate the mini40 true-PBR 500-step checkpoint in final rendered-view space.
The evaluation compares fixed-camera renders of base and fine-tuned outputs
against the original `render_cond` reference views for the same assets.

## Why Rendered Views

UV texture comparison and direct GLB inspection are useful diagnostics, but they
are not enough to decide whether the checkpoint improved visual behavior. A UV
map can change without improving rendered appearance, and a GLB can look
acceptable from one manually chosen angle while failing across the six official
views. Rendered-view evaluation checks the actual view-space result.

## Comparison Design

For every mini40 eval case and view `000` through `005`, compare:

- base render vs fine-tuned render
- base render vs reference
- fine-tuned render vs reference

Base and fine-tuned outputs must use the same selected input view established in
Phase 2L.5A.

## Metrics

The rendered-view comparison records:

- MAE
- RMSE
- PSNR
- SSIM-like score
- histogram L1 distance
- edge difference
- color mean shift

## Required Breakdown

Report metrics separately for:

- all views `000` through `005`
- input view `005`
- front views `004` and `005`
- non-front views `000` through `003`
- validation split
- test split
- validation plus test
- `train_sanity` assets separately

## Decision Logic

If validation/test fine-tuned renders clearly improve front-view metrics without
non-front degradation, consider a full80 500-step run.

If the fine-tuned checkpoint is stable but the improvement is tiny or mixed,
then either full80 500-step or stopping for report discussion may be reasonable.

If fine-tuned renders worsen validation/test metrics, do not scale training yet.
No quality claim should be made until summaries and boards are reviewed.
