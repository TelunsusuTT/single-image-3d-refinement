# Phase 2J.2 True-PBR Checkpoint Save Smoke Success Summary

## Goal

Phase 2J.2 verified that official train.py can save a checkpoint initialized from the official Hunyuan3D-Paint PBR base.

## Result

Phase 2J.2 succeeded.

Job id: 255908

Key success signals:
- PHASE2J2_SAVE_SMOKE_PREFLIGHT_OK
- CUBLAS_MATMUL_OK
- Setting learning rate to 0.00e+00
- Trainer.fit stopped at max_steps=1
- exactly one checkpoint saved
- PHASE2I_BASE_VS_CHECKPOINT_COMPARE_OK
- mean_of_mean_abs_delta = 0.0
- mean_relative_delta = 0.0
- PHASE2J2_SAVED_DELTA_TINY_OK
- PHASE2J2_CHECKPOINT_SAVE_SMOKE_OK
- JOB END: SUCCESS

## Checkpoint

checkpoints/pilot_v1_truepbr_1step_lr0/pilot_v1_truepbr_1step_lr0-stepstep=1.ckpt

Approximate size: 9.5G

## Interpretation

This confirms that true-PBR initialization survives the official train.py checkpoint-saving path. The saved checkpoint is numerically equivalent to the official inference base UNet.

The previous collapsed checkpoints were therefore most likely caused by non-PBR-base initialization, not by checkpoint saving or inference loading.

## Next Step

Run true-PBR 50-step conservative fine-tuning from the local hunyuan3d-paintpbr-v2-1 base.
