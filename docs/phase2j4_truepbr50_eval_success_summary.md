# Phase 2J.4 True-PBR 50-Step Evaluation Success Summary

## Goal

Phase 2J.4 evaluated the correctly initialized true-PBR 50-step checkpoint.

## Result

Phase 2J.4 succeeded.

The true-PBR 50-step checkpoint loaded successfully, completed no-remesh inference, and produced texture maps without the severe PBR/texture collapse observed in the earlier wrong-initialized checkpoints.

## Key Diagnostic Results

- Albedo MAE: approximately 0.331
- Metallic MAE: approximately 0.355
- Roughness MAE: approximately 0.689
- Metallic mean shift: approximately +0.006
- Roughness mean shift: approximately -0.134

This is a major improvement over the earlier collapsed 50-step and 500-step checkpoints, where albedo/metallic/roughness differences were globally large and metallic mean shifted to about 140+.

## Interpretation

The true-PBR initialization path fixes the previous collapse failure. The 50-step lr=1e-6 checkpoint is stable and close to the official base output.

However, this checkpoint does not yet demonstrate meaningful quality improvement because the output remains very close to base.

## Next Step

Run a stronger but still conservative true-PBR training experiment, then evaluate it with the same load-only, delta audit, inference, and texture comparison workflow.
