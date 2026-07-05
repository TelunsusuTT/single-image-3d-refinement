# Phase 2L.7A Full80 Corrected-Input Evaluation Setup

## Goal

Phase 2L.7A prepares corrected-input evaluation for the full80 true-PBR
500-step checkpoint:

`checkpoints/datav2_frame_full80_truepbr_500_lr1e6/datav2_frame_full80_truepbr_500_lr1e6-stepstep=500.ckpt`

This phase compares base Hunyuan3D-Paint against the full80 fine-tuned
checkpoint on the full101 validation and test assets, using the same selected
input image for both base and fine-tuned inference.

## Why This Follows Mini40

Mini40 showed stable training and a train-sanity learning signal, but no clear
held-out front-view improvement. Full80 was run to test whether more data helps
before spending effort on a longer mini40 or full80 training schedule.

## Corrected Input

Input-view selection remains the dominant bottleneck. Evaluation must use
`selected_input_view=005` unless the human override CSV says otherwise. The same
selected input image must be used for base and fine-tuned inference. Do not
compare against old wrong-input baselines.

## Evaluation Scope

Primary evaluation includes all full101 validation and test assets:

- val: 10
- test: 11
- primary total: 21

An optional train-sanity group includes 3 train assets and must stay clearly
separate from held-out val/test reporting.

## Expected Stages

1. Create the input-view review board and override CSV.
2. Human-review or approve the selected input views.
3. Generate eval cases.
4. Run readiness checks.
5. Run A100 base and fine-tuned inference manually via sbatch.
6. Create render-eval cases from completed inference outputs.
7. Run local/manual rendered-view evaluation.
8. Aggregate val/test, train-sanity, front-view, input-view, and non-front
   summaries.

No 1000-step training should be prepared until the full80 500-step evaluation is
interpreted.
