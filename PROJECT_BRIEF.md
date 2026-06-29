# Project Brief

This MSc project explores small-data partial fine-tuning of Hunyuan3D-Paint for texture-heavy image-to-3D assets.

## Scope

- We only target Hunyuan3D-Paint / texture stage.
- Shape generation is frozen / not trained.
- Mesh is fixed for evaluation.
- We use official training code from `Hunyuan3D2.1_Work` but keep configs, logs, checkpoints, and data in this project folder.
- No LoRA in Phase 0.
- No full fine-tuning in Phase 0.

## Phase 0 Goal

Phase 0 creates a safe external project scaffold and validates that the official Hunyuan3D-Paint overfit smoke test can be launched without writing outputs back into the upstream source tree.
