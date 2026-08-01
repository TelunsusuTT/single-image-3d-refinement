# Phase 2N Full Validation

## Goal

This stage expands the reviewed Week 2 pilot decision to the complete, frozen
`datav2_frame_panels_full101` validation split. It compares exactly four
variants on exactly ten validation assets:

1. `corrected_input_base`
2. `historical_full80_500`
3. `pc_s1_step160`, the safety-oriented candidate
4. `pc_full_step320`, the quality-oriented candidate

`pc_s1_step320` and `pc_full_step160` remain pilot-only checkpoint-step
ablations. They are not expanded. This stage contains no train-sanity or test
asset, does not rerun training, and does not select from test evidence.

## Canonical Cases

The manifest builder reads the actual full101 split and intersects it with the
actual Week 2 pilot case config. It does not hard-code the remaining set.

Canonical validation IDs, in split order:

```text
B073P1D981
B075YLG1B5
B075YM2VXJ
B075YLQTJ3
B075YLXSJC
B075YLQTNP
B073P52NDX
B073P1S8VZ
B073P51233
B077X6WSH1
```

The six existing pilot validation IDs are `B073P1D981`, `B075YLXSJC`,
`B075YLQTNP`, `B075YM2VXJ`, `B073P52NDX`, and `B073P1S8VZ`. The derived four
remaining IDs are `B075YLG1B5`, `B075YLQTJ3`, `B073P51233`, and `B077X6WSH1`.
Every case uses selected input view `005`, AL lighting, its fixed source mesh,
and references `000` through `005`.

## Reuse Audit

Configuration: `configs/phase2n_full_validation_manifest.json`

The resolver audits 40 asset-variant GLB sources:

| Source | Assets x variants | GLBs | Status before new inference |
|---|---:|---:|---|
| Phase 2L corrected-input base + historical full80 | 10 x 2 | 20 | Reused |
| Week 2 pilot PC-S1-160 + PC-Full-320 | 6 x 2 | 12 | Reused |
| Remaining validation candidates | 4 x 2 | 8 | New inference required |

Each source record includes asset and split, variant, source path and run,
selected view, mesh, all six references, reuse status, source manifest, and the
scope-checkpoint manifest where applicable. Check without writing:

```bash
python scripts/phase2n_build_full_validation_manifest.py \
  --config configs/phase2n_full_validation_manifest.json \
  --check-only
```

Expected tokens:

```text
PHASE2N_FULL_VALIDATION_SPLIT_OK
PHASE2N_FULL_VALIDATION_REUSE_OK
```

After review, `--write` creates the configured resolved JSON exactly once.
Existing output is never overwritten.

## New A100 Inference

Configuration: `configs/phase2n_full_validation_inference.json`

Only eight GLBs are new: the two selected candidates on the four remaining
validation assets. The reused Week 2 inference runner loads a fresh official
true-PBR base for each candidate, applies one strict scope checkpoint, rejects
missing or unexpected tensors, and runs fixed-mesh inference with remeshing
disabled. The fixed output is:

```text
outputs/phase2n/full_validation_inference/phase2n_full_validation_infer_v1/
```

Readiness:

```bash
python scripts/phase2n_week2_infer_pilots.py \
  --config configs/phase2n_full_validation_inference.json \
  --check-only
```

Success ends with `PHASE2N_FULL_VALIDATION_INFERENCE_READINESS_OK`. The runtime
must fail if the fixed run directory already exists. Per-candidate `_SUCCESS`
files and the root `_SUCCESS` are written only after all expected outputs pass
validation. Runtime success tokens are:

```text
PHASE2N_FULL_VALIDATION_PC_S1_STEP160_INFERENCE_OK
PHASE2N_FULL_VALIDATION_PC_FULL_STEP320_INFERENCE_OK
PHASE2N_FULL_VALIDATION_INFERENCE_OK
```

Manual submission after review:

```bash
sbatch env/run_phase2n_full_validation_infer_a100.sbatch
```

This uses one A100 80GB node and does not launch rendered evaluation.

## Clean Rendered Evaluation

Configuration: `configs/phase2n_full_validation_rendered_eval.json`

After the eight new GLBs exist, readiness resolves the complete 40-GLB matrix:

```bash
python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_full_validation_rendered_eval.json \
  --check-only
```

Expected token:

```text
PHASE2N_FULL_VALIDATION_RENDERED_EVAL_READINESS_OK
```

Use a new local run ID, never a Week 2 pilot directory:

```bash
python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_full_validation_rendered_eval.json \
  --run-all \
  --run-id <NEW_RUN_ID>
```

The existing Phase 2K renderer, camera, lighting, material handling,
background, resolution, and metric formulas remain unchanged. One isolated
Blender subprocess renders each GLB. The clean final matrix is 10 assets x 4
variants x 6 views: 240 PNGs and 240 per-view metric rows. It also creates ten
front boards and ten non-front/leakage boards.

Reports cover all views, input `005`, front `004/005`, non-front `000-003`,
each asset, improved/worsened view counts, direct change from corrected-input
base, and leakage-risk assets. Runtime stage tokens are:

```text
PHASE2N_FULL_VALIDATION_RENDER_STAGE_OK
PHASE2N_FULL_VALIDATION_METRICS_STAGE_OK
PHASE2N_FULL_VALIDATION_BOARDS_STAGE_OK
PHASE2N_FULL_VALIDATION_RENDERED_EVAL_OK
```

## Gate

Pass requires the exact 10/0/0 validation/train-sanity/test split, 40 non-empty
source GLBs, 240 valid renders, 240 metric rows, 20 boards, and all success
markers. Any split drift, missing reference, missing or unexpected checkpoint
tensor, existing fixed inference run, incomplete render unit, or test/train
asset is a hard failure. No full-validation quality conclusion exists until
the metrics and human boards have been reviewed.
