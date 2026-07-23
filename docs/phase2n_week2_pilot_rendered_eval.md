# Phase 2N Week 2 Pilot Rendered-View Evaluation

## Purpose

Stage 5 measures the four controlled Week 2 pilot checkpoints against the
corrected-input base and historical full80-500 baselines. It uses the same
eight cases frozen before training and the same six-view rendered evaluation
protocol used by the earlier Phase 2K/2M work.

The completed Stage 4 inference run is:

`outputs/phase2n/week2_pilot_inference/slurm_264581`

Stage 5 is local Blender plus CPU/Pillow work on `gpu12`. It does not run
Hunyuan inference, load a training checkpoint, use CUDA, or require an A100.

## Frozen Comparison

The column and processing order is fixed:

1. `corrected_input_base`
2. `historical_full80_500`
3. `pc_s1_step160`
4. `pc_s1_step320`
5. `pc_full_step160`
6. `pc_full_step320`

The frozen cases are:

| Split | Stratum | Assets |
|---|---|---|
| validation | leakage risk | `B073P1D981`, `B075YLXSJC` |
| validation | high front complexity | `B075YLQTNP`, `B075YM2VXJ` |
| validation | historical median | `B073P52NDX`, `B073P1S8VZ` |
| train-sanity | diagnostic | `B073P16J7Y`, `B073P1H786` |

There are exactly six validation cases, two train-sanity cases, and no test
cases. The selected input is AL view `005`.

## Reused Protocol

The workflow deliberately reuses:

- `scripts/render_phase2k3_glb_views_blender.py` for normalized orthographic
  camera placement, six-view poses, lighting, background, material handling,
  and 512-pixel PNG rendering;
- `scripts/compare_phase2k3_rendered_views.py` for MAE, RMSE, PSNR,
  SSIM-like, histogram, edge, color-shift, and boosted-difference functions;
- the multi-variant board layout concepts established in Phase 2M.

The new driver adds orchestration and six-variant aggregation only. It launches
one isolated Blender subprocess per GLB, so Blender state and retained memory
cannot carry between the 48 render units. All six variants are rendered again
inside one new run; historical PNGs are not mixed into the comparison.

The complete plan is:

- 8 cases x 6 variants = 48 source GLBs;
- 48 GLBs x 6 views = 288 rendered PNGs;
- 8 cases x 6 views = 48 AL reference images;
- 288 per-view metric rows.

## Readiness

Run the standard-library preflight:

```bash
python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_week2_pilot_rendered_eval.json \
  --check-only
```

Success prints:

```text
PHASE2N_WEEK2_RENDER_INPUTS_OK
PHASE2N_WEEK2_RENDER_PROTOCOL_OK
PHASE2N_WEEK2_NO_TEST_OK
PHASE2N_WEEK2_RENDERED_EVAL_READINESS_OK
```

The check resolves baseline GLBs from the Stage 4 baseline reuse report and
pilot GLBs from each Stage 4 inference manifest. It verifies all 48 GLBs, all
48 references, the exact frozen split, the renderer and metric helpers, the
Blender executable, and the safe output root. It does not import `bpy`, call
Blender, or create an evaluation run.

## Runtime

Choose a new stable run ID and execute locally on `gpu12`:

```bash
python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_week2_pilot_rendered_eval.json \
  --run-all \
  --run-id <RUN_ID>
```

### V1 Worker Routing Failure

The first local run,
`outputs/phase2n/week2_pilot_rendered_eval/phase2n_week2_eval_v1`, is retained
read-only. Blender 3.6.0 and the clean background-startup smoke succeeded, but
the first worker stopped before GLB import because the Stage 5 script parsed
the complete Blender argv instead of parsing project arguments after the `--`
boundary. The internal worker action was therefore hidden from the required
argparse action group.

The repair changes only CLI routing and adds explicit run, case, variant, and
split metadata to the worker command. It does not change the stable renderer,
camera, views, lighting, materials, background, resolution, GLBs, metrics, or
aggregation. No model output or scientific result was invalidated. Any next
full evaluation must use a new run ID rather than reuse `phase2n_week2_eval_v1`.

The stages can also be run separately with the same run ID:

```bash
python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_week2_pilot_rendered_eval.json \
  --render-only --run-id <RUN_ID>

python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_week2_pilot_rendered_eval.json \
  --metrics-only --run-id <RUN_ID>

python scripts/phase2n_week2_evaluate_pilots.py \
  --config configs/phase2n_week2_pilot_rendered_eval.json \
  --boards-only --run-id <RUN_ID>
```

Complete render units are validated and skipped. A partial or invalid unit,
changed runtime manifest, or success-marked run with a missing artifact fails
closed rather than being silently overwritten. A complete `_SUCCESS` run is
fully revalidated and returned idempotently.

## Outputs

Each run writes:

```text
outputs/phase2n/week2_pilot_rendered_eval/<run_id>/
  00_RUNTIME_MANIFEST.json
  resolved_cases.json
  resolved_variants.json
  renders/
  metrics/
    per_view.jsonl
    per_view.csv
    per_asset.json
    aggregate.json
    aggregate.md
  boards/
    front/
    nonfront_leakage/
  summary.json
  summary.md
  _SUCCESS
```

Every board has reference plus all six variants. Front boards show views
`004/005`; non-front/leakage boards show views `000-003`. Boosted direct
difference-to-base panels use the stable Phase 2K difference helper.

## Evidence And Selection

Metrics are aggregated over all views, input view `005`, front views `004/005`,
non-front views `000-003`, validation, train-sanity, each asset, each view, and
the two leakage-risk validation assets.

Each candidate records validation MAE/SSIM deltas, improved and worsened view
counts, direct visual change from base, front/input/non-front evidence,
leakage-risk regressions, and train-sanity behavior. The explicit evidence
fields include:

- `val_front_mean_better_than_base`
- `val_front_view_win_count`
- `val_input_mean_better_than_base`
- `val_nonfront_mean_better_than_base`
- `leakage_risk_regression_count`
- `train_sanity_mean_better_than_base`

No winner is selected automatically.

- Use validation only for model selection.
- Treat train-sanity as diagnostic only.
- Check non-front and leakage safety before considering a front improvement.
- Prefer step 160 when step 160 and step 320 are materially similar.
- Prefer PC-S1 when PC-S1 and PC-Full are materially similar.
- A small front gain does not compensate for a clear backside leakage
  regression.

The preliminary leakage concern remains a hypothesis. No Week 2 checkpoint
quality conclusion exists until this runtime evaluation and human board review
are complete.
