# Phase 2L.5A Mini40 True-PBR Evaluation Setup

## Goal

Phase 2L.5A prepares a safe evaluation workflow for the mini40 true-PBR
checkpoint:

```text
checkpoints/datav2_frame_mini40_truepbr_500_lr1e6/datav2_frame_mini40_truepbr_500_lr1e6-stepstep=500.ckpt
```

This phase creates review, case-generation, readiness, inference, render-eval,
and aggregation scaffolding. It does not run Hunyuan, submit Slurm jobs, run
Blender, train, load checkpoints in Codex, install packages, or make quality
claims.

## Why Selected Input View Is Critical

Phase 2K.4 showed that the input/reference view can dominate apparent output
quality. A weak or backside input view can make base and fine-tuned inference
look bad for reasons unrelated to checkpoint quality. For this mini40
evaluation, selected input view must be explicit before A100 inference.

The priority order is:

1. per-asset override CSV
2. curated manifest `selected_input_view`
3. config default, only as a last resort

The override CSV is a human-review artifact. It prevents the pipeline from
silently hard-coding view `004` or `005`.

## Corrected-Input Comparison

Base and fine-tuned inference must use the same `selected_input_image` for each
asset. This compares corrected-input base against corrected-input fine-tuned
output and avoids old wrong-input baselines.

## Why Val/Test First

Primary evaluation uses the mini40 validation and test assets, eight assets in
total. Two train assets may be included only as a clearly labeled sanity group.
Do not expand to full101 or full80 until mini40 rendered-view metrics and boards
are reviewed.

## Expected Stages

1. input-view review board and override CSV
2. eval case generation
3. readiness checks
4. A100 base and fine-tuned inference
5. local rendered-view evaluation
6. aggregate summary by split and view group

No quality claim should be made until rendered-view metrics and boards have
been inspected.
