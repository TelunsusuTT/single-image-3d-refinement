# Current Project State

## Summary

The project is in Phase 2N Week 2 Stage 1, developing the shared deterministic
PC-S1/PC-Full pilot runner and freezing its evaluation cases. Week 1 Days 1-5
are complete. The Day 5 A100 runtime and numeric audits passed, and the
controlled Week 2 pilot is authorized with reviewed learning rates and runtime
limits.

Phase 2M refview+DINO LoRA is complete and negative on held-out cases. Phase 2N
now has a protocol-corrected no-augmentation reader, exact PC-S1 and PC-Full
scope helpers, optimizer/scheduler guards, and successful real-model Day 5
evidence. No Week 2 training has started.

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

Week 1 Days 1-5 are complete. The static audit established that full80-500 was
broad partial fine-tuning, while PC-S1 isolates 80 `attn_multiview` projection
tensors. Day 3 and Day 4 established the deterministic protocol-corrected
reader, exact scope selection, optimizer membership, and a 50-step
warmup-constant scheduler. Day 4B selected spatial augmentation `none` for both
controlled pilots.

Day 5 loaded the official true-PBR base on an 80 GB A100 and passed production
loss, gradient, update, memory, scheduler, and selective reload checks. Numeric
review authorized PC-S1 at peak LR `1e-6` and PC-Full at peak LR `5e-7` for a
320-update controlled pilot with checkpoints at 160 and 320.

The current task is Week 2 Stage 1 local runner/readiness development. One
shared deterministic schedule and a frozen six-validation/two-train-sanity
evaluation pilot are being prepared before any model run. Test data remains
excluded from training and model selection. No Week 2 training has started.

## No Full80-1000 Yet

Do not prepare or run full80-1000 yet. Full80-500 is not a strong enough signal
to justify a longer run by default. The improvements are small, split behavior
is mixed, and visual boards still show leakage/ambiguity.

## Current Decision

Finish and locally validate the shared Week 2 PC-S1/PC-Full runner, frozen
evaluation manifest, and manual A100 launcher. After assistant and user review,
the authorized pilot may run PC-S1 first and PC-Full second from fresh identical
true-PBR bases. Do not start Week 2 training before that gate, and do not use the
test split for selection.
