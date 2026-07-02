# Phase 2K.1 True-PBR 200-Step Train-Eval

Phase 2K.1 increases true-PBR adaptation strength after the stable 50-step
result. Phase 2J.4 showed that the true-PBR 50-step checkpoint avoided the
previous texture/PBR collapse, but its output stayed very close to the base
model. Phase 2K.1 trains the same `pilot_v1` dataset for 200 steps at the same
`base_learning_rate: 1e-6`.

This differs from Phase 2J.4:

- Phase 2J.4 evaluated the existing 50-step checkpoint and showed minimal
  change from base.
- Phase 2K.1 trains a new 200-step checkpoint from the verified local
  `hunyuan3d-paintpbr-v2-1` initialization path, then evaluates it immediately.

Evaluation stages:

1. Train.
2. Load-only checkpoint compatibility check.
3. Base-vs-checkpoint numeric delta audit.
4. No-remesh fine-tuned inference on `B075YLTF7Q`.
5. Diagnostic base-vs-fine-tuned texture comparison.

Non-goals:

- no final quality claim
- no Data v2 training
- no LoRA
- no automatic 500-step run

Expected success:

- `max_steps=200` is reached
- exactly one checkpoint is saved
- no texture/PBR collapse
- metrics are larger than the 50-step run but far smaller than the previous
  collapsed runs
