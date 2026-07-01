# Phase 2H.1 Conservative 50-Step Evaluation

Phase 2H.1 evaluation checks whether the conservative 50-step low-learning-rate
checkpoint avoids the severe texture/PBR collapse seen with the earlier
500-step checkpoint.

Phase 2G used the collapsed `pilot_v1_overfit_500` checkpoint. This evaluation
uses:

```text
checkpoints/pilot_v1_conservative_50_lr1e6/pilot_v1_conservative_50_lr1e6-stepstep=50.ckpt
```

The evaluation has three stages:

1. Load-only checkpoint compatibility check.
2. Fine-tuned no-remesh inference on the prepared `B075YLTF7Q` case.
3. Base-vs-conservative diagnostic comparison using the existing Phase 2G.6
   comparison script.

Non-goals:

- no final quality claim
- no Data v2 training
- no LoRA
- no checkpoint loading from Codex
- no direct Blender or training run

Expected outputs:

```text
outputs/phase2h/load_only/pilot_v1_conservative_50_lr1e6/load_only_report.md
outputs/phase2h/infer_runs/B075YLTF7Q/pilot_v1_conservative_50_lr1e6_noremesh/
outputs/phase2h/compare/B075YLTF7Q/pilot_v1_conservative_50_lr1e6/
```

The comparison is diagnostic only. Compare its albedo, metallic, and roughness
metrics against the 500-step collapsed checkpoint before deciding whether the
conservative run is worth expanding.
