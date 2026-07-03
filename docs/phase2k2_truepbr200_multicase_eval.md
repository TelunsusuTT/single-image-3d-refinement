# Phase 2K.2 True-PBR 200-Step Multi-Case Evaluation

Phase 2K.2 evaluates the fixed true-PBR 200-step checkpoint across multiple
`pilot_v1` assets without training.

This differs from Phase 2K.1:

- Phase 2K.1 trained the 200-step checkpoint and evaluated it on `B075YLTF7Q`.
- Phase 2K.2 creates additional inference cases and evaluates the existing
  checkpoint on three more pilot assets.

Selected cases:

- `B07HSK7MXZ`
- `B073NZS57V`
- `B07B8MWCR8`

These are selected because they are already part of the pilot asset set and can
reuse the same raw-mesh plus rendered-reference-image case format as the
existing `B075YLTF7Q` inference case.

Evaluation stages:

1. Prepare case directories.
2. Run one load-only checkpoint check.
3. Run base no-remesh inference per case.
4. Run fine-tuned no-remesh inference per case.
5. Run diagnostic texture comparison per case.
6. Aggregate metrics across cases.

Non-goals:

- no new training
- no new checkpoint
- no Data v2 training
- no LoRA
- no final quality claim

Expected success:

- all selected cases produce base and fine-tuned GLB/OBJ/maps
- all selected cases produce comparison metrics and boards
- no asset shows the previous metallic/PBR collapse
