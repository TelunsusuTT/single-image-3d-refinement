# Phase 2J.3 True-PBR 50-Step Training Success Summary

## Goal

Phase 2J.3 ran the first real true-PBR fine-tuning job from the official Hunyuan3D-Paint PBR base.

## Result

Phase 2J.3 succeeded.

Job id: 255920

Key success signals:
- PHASE2J3_TRUEPBR50_PREFLIGHT_OK
- official strict checker passed on all 7 pilot_v1 samples
- CUBLAS_MATMUL_OK
- Setting learning rate to 1.00e-06
- Trainer.fit stopped at max_steps=50
- exactly one checkpoint saved
- PHASE2J3_TRUEPBR50_TRAIN_OK
- JOB END: SUCCESS

## Checkpoint

checkpoints/pilot_v1_truepbr_50_lr1e6/pilot_v1_truepbr_50_lr1e6-stepstep=50.ckpt

Approximate size: 9.5G

## Interpretation

This is the first checkpoint trained from the verified official Hunyuan3D-Paint PBR initialization path. It replaces the earlier collapsed 50-step and 500-step checkpoints that were not initialized from the official PBR base.

## Next Step

Evaluate this checkpoint with load-only, no-remesh inference, and base-vs-finetuned texture diagnostics.
