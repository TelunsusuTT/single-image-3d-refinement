# Phase 2N Final Test

## Goal And Freeze

This stage evaluates the already selected Phase 2N candidate on exactly the 11
assets in the canonical full101 test split. The frozen comparison is:

1. `corrected_input_base`
2. `historical_full80_500`
3. `pc_full_step320`

`pc_full_step320` is the quality-oriented final candidate. It uses the
`pc_full` scope, checkpoint step 320 from training run `slurm_264123`, and
checkpoint SHA-256
`b93f4d34291a21097b8d763b3d5c18d4b0a82796049ce0711aaac552b310d18e`.
`pc_s1_step160` remains a validation ablation only and is excluded from both
final-test inference and evaluation.

The freeze is based only on the completed ten-asset validation run
`phase2n_full_validation_eval_v1`. Its reviewed PC-Full evidence was MAE delta
`-0.44887108272976345`, SSIM-like delta `+0.024508462795182862`, positive
front/input/non-front mean checks, and four leakage regressions. The completed
run proves `test_data_used_for_selection=false`.

The test split was evaluated previously for the historical Phase 2L full80
baseline. It is therefore described as **held out from Phase 2N candidate and
checkpoint selection**, not as never inspected anywhere in the project. Phase
2N test results are report-only and must not trigger tuning, checkpoint
replacement, or protocol changes.

## Frozen Protocol

The authoritative record is `configs/phase2n_final_test_freeze.json`. It fixes:

- selected input view `005` and AL reference lighting;
- front views `004/005` and non-front views `000-003`;
- the original fixed mesh with remeshing disabled;
- the selected checkpoint, candidate list, and comparison variants;
- no further model, checkpoint, or protocol changes.

The exact test IDs, in canonical split order, are:

```text
B073NZS586
B073P1JNZZ
B075HXJ6Q9
B073P1N4C4
B073NZGLT1
B073P1CKJH
B073P5FLX9
B073P1H7MS
B07HSK626Y
B073P1H6D8
B075YNL763
```

There are zero validation and zero train-sanity assets in this stage.

## Manifest And Reuse Audit

The builder reads the real full101 split, verifies the completed validation
success and evidence, and audits historical run plans rather than assuming path
compatibility:

```bash
python scripts/phase2n_build_final_test_manifest.py \
  --config configs/phase2n_final_test_manifest.json \
  --check-only
```

Expected tokens are:

```text
PHASE2N_FINAL_TEST_FREEZE_OK
PHASE2N_FINAL_TEST_SPLIT_OK
PHASE2N_FINAL_TEST_REUSE_OK
```

The audited source matrix is:

| Source | GLBs | Action |
|---|---:|---|
| Corrected-input base, Phase 2L test outputs | 11 | Reuse |
| Historical full80-500, Phase 2L test outputs | 11 | Reuse |
| PC-Full step 320 | 11 | New A100 inference |
| **Total** | **33** | 22 reused + 11 new |

After review, write the canonical resolved manifest exactly once:

```bash
python scripts/phase2n_build_final_test_manifest.py \
  --config configs/phase2n_final_test_manifest.json \
  --write
```

This creates
`outputs/phase2n/final_test_manifest/phase2n_final_test_manifest_v1.json` and
refuses to overwrite an existing file.

## A100 Inference

Only `pc_full_step320` is run. Readiness verifies the 11 cases, both 11/11
baseline coverages, fixed protocol, training completion, strict checkpoint
scope/step/size/SHA-256, and a new fixed output directory:

```bash
python scripts/phase2n_week2_infer_pilots.py \
  --config configs/phase2n_final_test_inference.json \
  --check-only
```

Expected readiness token:

```text
PHASE2N_FINAL_TEST_INFERENCE_READINESS_OK
```

Manual submission, only after the manifest has been written and reviewed:

```bash
sbatch env/run_phase2n_final_test_infer_a100.sbatch
```

The fixed output is
`outputs/phase2n/final_test_inference/phase2n_final_test_infer_v1/`. The runner
loads a fresh official true-PBR pipeline, strictly loads only the frozen
PC-Full checkpoint, rejects missing or unexpected tensors, uses the original
mesh with `use_remesh=false`, and writes exactly 11 non-empty GLBs. It refuses
an existing run directory and writes case-level, variant-level, and root
success markers only after validation.

Runtime tokens:

```text
PHASE2N_FINAL_TEST_PC_FULL_STEP320_INFERENCE_OK
PHASE2N_FINAL_TEST_INFERENCE_OK
```

The sbatch does not launch rendered evaluation.

## Rendered Evaluation

Rendered-evaluation readiness is intentionally deferred until the final
inference root `_SUCCESS` exists. At that point run:

```bash
python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_final_test_rendered_eval.json \
  --check-only
```

Expected token:

```text
PHASE2N_FINAL_TEST_RENDERED_EVAL_READINESS_OK
```

Then use a new run ID:

```bash
python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_final_test_rendered_eval.json \
  --run-all \
  --run-id <NEW_FINAL_TEST_EVAL_RUN_ID>
```

The reused fixed Blender renderer and metric implementation remain unchanged.
The clean matrix is 11 assets x 3 variants x 6 views: 33 source GLBs, 198
rendered PNGs, 198 per-view metric rows, 11 front boards, and 11
non-front/leakage boards. Reports cover all views, input `005`, front
`004/005`, non-front `000-003`, per-asset changes, improved/worsened counts,
direct visual change, leakage regressions, and both comparisons against
corrected-input base.

The evaluation run embeds the freeze record, its SHA-256, the project Git
commit, source manifests, and checkpoint manifest. It reports the frozen
candidate; it cannot select or replace a checkpoint.

Runtime stage tokens:

```text
PHASE2N_FINAL_TEST_RENDER_STAGE_OK
PHASE2N_FINAL_TEST_METRICS_STAGE_OK
PHASE2N_FINAL_TEST_BOARDS_STAGE_OK
PHASE2N_FINAL_TEST_RENDERED_EVAL_OK
```

## Pass And Failure Gates

Pass requires the exact 0 validation / 0 train-sanity / 11 test split, 22/22
compatible reused baselines, 11 validated new candidate GLBs, 33 resolved
sources, 198 renders and metric rows, 22 boards, and all success markers. Any
split drift, protocol mismatch, missing baseline, checkpoint mismatch, missing
output, or pre-existing fixed inference directory fails closed. Final test is
an evaluation gate only; it authorizes no further Phase 2N tuning.
