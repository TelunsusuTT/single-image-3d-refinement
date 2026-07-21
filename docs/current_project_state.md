# Current Project State

## Summary

The project is in Phase 2N Week 1 Day 2, confirming the historical full80
training protocol before protocol-corrected, geometry-focused selective
fine-tuning is implemented. The technical pipeline is working end to end:
true-PBR initialization is fixed, Data v2 frame-panel examples can be rendered
and checked, official training runs can save checkpoints, and corrected-input
outputs can be evaluated in rendered-view space.

Phase 2M refview+DINO LoRA is complete and negative on held-out cases. Phase 2N
Day 1 architecture and reuse auditing is complete. No new Phase 2N training has
started.

## Initialization Fix

Earlier 50-step and 500-step checkpoints were initialized from an SD2/non-PBR
base, not from the official Hunyuan3D-Paint PBR inference weights. Phase 2J
located the local `hunyuan3d-paintpbr-v2-1` pipeline, verified training
initialization equivalence, and proved that official `train.py` can save a
base-equivalent checkpoint.

The later Data v2 runs use true-PBR initialization and do not show the old PBR
collapse failure.

## Mini40-500 Diagnostic Result

The mini40 true-PBR 500-step checkpoint was stable and showed a train-sanity
learning signal, but it did not clearly improve held-out corrected-input
front-view behavior. Rendered-view evaluation showed mixed or negative
validation/test behavior, especially on front/input views. This justified
scaling the dataset before increasing the step count.

## Full80-500 Result

The full80 true-PBR 500-step checkpoint completed successfully:

`checkpoints/datav2_frame_full80_truepbr_500_lr1e6/datav2_frame_full80_truepbr_500_lr1e6-stepstep=500.ckpt`

Corrected-input rendered-view evaluation completed with 24 cases and 144 rows:

`outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/summary/full80_eval_summary.md`

The result is stable and marginally positive in aggregate, but visually mixed:

- all views MAE improved from 6.1626 to 6.0654
- all views SSIM-like improved from 0.8466 to 0.8629
- val+test MAE improved from 6.3975 to 6.2725
- val+test SSIM-like improved from 0.8369 to 0.8561
- test MAE regressed slightly from 6.4182 to 6.4782
- train-sanity MAE regressed from 4.5181 to 4.6158

The main observed failure mode is front-to-back leakage or backside
contamination. This is a modeling/evaluation limitation of single-image texture
generation, not an infrastructure failure.

## Phase 2M LoRA Result

The refview+DINO LoRA adapter trained and ran successfully, but all evaluated
scales degraded held-out validation/test and front-view results compared with
corrected-input base. It remains a documented negative baseline and is not
selected for full evaluation expansion.

## Phase 2N Status

The Day 1 static audit established that full80-500 was broad partial
fine-tuning, not a narrow selective update. `attn_multiview` projections are the
main S1 selective candidate. S2 is not a distinct scope because no separate
trainable normal/position projection was found. S3 remains runtime-gated.

The current task is Day 2 historical protocol confirmation: learning-rate
ground truth, actual reference-view sampling, augmentation behavior, and the
proposed protocol-corrected replacement.

## No Full80-1000 Yet

Do not prepare or run full80-1000 yet. Full80-500 is not a strong enough signal
to justify a longer run by default. The improvements are small, split behavior
is mixed, and visual boards still show leakage/ambiguity.

## Current Decision

Complete the Phase 2N Week 1 protocol and runtime audits before authorizing any
new training. The next implementation step is a project-local,
protocol-corrected dataset path with explicit selected-view weighting and shared
spatial augmentation; the Day 5 A100 audit must confirm the selective scope and
learning rate before the Week 2 pilot.
