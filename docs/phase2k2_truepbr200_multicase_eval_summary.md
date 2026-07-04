# Phase 2K.2 True-PBR 200-Step Multi-Case Evaluation Summary

## Result

Phase 2K.2 evaluated the true-PBR 200-step checkpoint on three additional pilot_v1 assets.

The checkpoint did not reproduce the earlier texture/PBR collapse.

## Aggregate Metrics

- mean albedo MAE: about 10.024
- mean metallic MAE: about 8.598
- mean roughness MAE: about 7.562
- mean metallic shift: about -7.154
- mean roughness shift: about -0.893

## Interpretation

The checkpoint is stable across multiple cases and produces measurable texture/PBR changes. However, quality improvement is not yet proven from UV texture maps alone.

## Next Step

Run rendered-view evaluation using fixed cameras to compare base GLB, fine-tuned GLB, and reference/training views.
