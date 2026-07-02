# Phase 2J.4 True-PBR 50-Step Evaluation

Phase 2J.4 evaluates the correctly initialized true-PBR 50-step checkpoint:

`checkpoints/pilot_v1_truepbr_50_lr1e6/pilot_v1_truepbr_50_lr1e6-stepstep=50.ckpt`

This differs from Phase 2H.1 evaluation:

- Phase 2H.1 evaluated a 50-step checkpoint that was later found to be
  initialized from the wrong SD2/non-PBR base.
- Phase 2J.4 evaluates the checkpoint trained from the verified official
  Hunyuan3D-Paint PBR initialization path.

Evaluation stages:

1. Load-only checkpoint compatibility check.
2. Base-vs-checkpoint numeric delta audit.
3. No-remesh fine-tuned inference on `B075YLTF7Q`.
4. Base-vs-fine-tuned diagnostic texture comparison.

Non-goals:

- no final quality claim
- no Data v2
- no LoRA
- no additional training

Expected outputs:

- `outputs/phase2j/eval_truepbr50/load_only/`
- `outputs/phase2j/eval_truepbr50/delta_audit/`
- `outputs/phase2j/eval_truepbr50/infer/B075YLTF7Q/`
- `outputs/phase2j/eval_truepbr50/compare/B075YLTF7Q/`
