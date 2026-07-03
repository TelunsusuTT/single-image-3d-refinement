# Phase 2K.1 True-PBR 200-Step Train-Eval Success Summary

## Goal

Phase 2K.1 increased true-PBR adaptation strength after the stable but nearly unchanged 50-step result.

## Result

Phase 2K.1 succeeded.

The workflow completed:
- 200-step true-PBR training
- exactly one checkpoint save
- load-only compatibility check
- base-vs-checkpoint delta audit
- no-remesh fine-tuned inference
- base-vs-fine-tuned texture comparison

## Checkpoint

checkpoints/pilot_v1_truepbr_200_lr1e6/pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt

## Diagnostic Metrics

Compared with base on B075YLTF7Q:

- albedo MAE: approximately 2.481
- metallic MAE: approximately 1.328
- roughness MAE: approximately 13.417
- metallic mean shift: approximately +0.839
- roughness mean shift: approximately -10.361

## Interpretation

The true-PBR 200-step checkpoint shows stronger changes than the 50-step checkpoint while still avoiding the earlier texture/PBR collapse. The largest visible change is in roughness, with smaller albedo changes and stable metallic behavior.

This is not yet a final quality improvement claim. It is evidence that true-PBR fine-tuning is now stable enough for multi-case evaluation.

## Next Step

Evaluate the true-PBR 200-step checkpoint on multiple pilot_v1 assets before increasing training strength or starting Data v2 training.
