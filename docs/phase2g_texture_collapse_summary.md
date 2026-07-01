# Phase 2G Texture Collapse Summary

## Context

Phase 2G completed the first end-to-end fine-tuned Hunyuan3D-Paint inference test using the Phase 2F 500-step checkpoint.

## Engineering Result

The engineering pipeline succeeded:

- base no-remesh inference succeeded
- checkpoint key mapping was identified
- checkpoint strict load-only succeeded
- fine-tuned no-remesh inference succeeded
- fine-tuned OBJ/GLB and texture maps were generated

## Quality Result

The 500-step checkpoint produced severe texture/PBR collapse.

Observed symptoms:

- albedo was globally corrupted
- metallic map shifted strongly upward
- roughness map shifted upward
- texture maps changed almost globally
- visual appearance became dirty/rusty/metal-like

## Quantitative Evidence

From Phase 2G.6:

- albedo MAE approximately 105.8
- metallic MAE approximately 138.8
- roughness MAE approximately 68.1
- fine-tuned metallic mean shifted from approximately 6.5 to 144.6

From Phase 2G.7:

- training targets do not show a simple single-file corruption
- the result is more consistent with over-aggressive fine-tuning / catastrophic forgetting
- pilot_v1 is also not ideal as a strong texture-heavy dataset

## Interpretation

The current 500-step checkpoint should not be used as a positive result. It is useful as a failure analysis checkpoint showing that small-data large-UNet fine-tuning can destroy Hunyuan3D-Paint's texture/PBR prior.

## Next Step

Run conservative fine-tuning recovery:

- much lower learning rate
- fewer steps
- save small number of checkpoints
- verify no-collapse before any quality claim
