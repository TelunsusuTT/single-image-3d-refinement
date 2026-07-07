# Phase 2L Runbook

## Phase 2L.1A Metadata Candidate Mining

Phase 2L.1A is metadata-only. Do not download assets, run Blender, run Hunyuan,
submit Slurm jobs, train, or load checkpoints.

Run static checks first:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_metadata_inventory.py \
  tests/test_datav2_flat_panel_candidate_mining.py \
  tests/test_datav2_human_review_template.py
```

Inventory available local metadata:

```bash
python scripts/datav2_inventory_metadata_sources.py \
  --config configs/datav2_flat_panel_candidate_mining.json \
  --out-dir outputs/phase2l/metadata_inventory
```

Inspect:

```bash
less outputs/phase2l/metadata_inventory/metadata_sources_report.md
```

Run candidate mining as a dry run first:

```bash
python scripts/datav2_mine_flat_panel_candidates.py \
  --config configs/datav2_flat_panel_candidate_mining.json \
  --dry-run \
  --max-rows-per-source 0
```

Only after the inventory is understood, run full metadata candidate mining:

```bash
python scripts/datav2_mine_flat_panel_candidates.py \
  --config configs/datav2_flat_panel_candidate_mining.json \
  --max-rows-per-source 0 \
  --top-k 500
```

Generate the human-review template:

```bash
python scripts/datav2_make_human_review_template.py \
  --candidates-csv data/candidates/datav2_flat_panel_metadata_candidates.csv \
  --out-csv data/candidates/datav2_flat_panel_human_review_template.csv \
  --top-k 150
```

The candidate CSV, candidate Markdown, summary JSON, and human-review template
are planning artifacts. They should not be committed unless explicitly
requested.

Human review decides final membership. Do not treat rank alone as acceptance.
Later curation must fill `selected_input_view`, `primary_eval_views`, rejection
reasons, and notes before download, inspection, contact sheets, or training.

## Phase 2L.1B ABO Geometry Candidate Refinement

Phase 2L.1B is still metadata-only. It ranks ABO asset-level rows from
`data/metadata/abo/3dmodels.csv.gz` using flatness, panel aspect ratio,
texture/material counts, mesh count, face count, and image-resolution signals.
Do not download assets in this phase.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_abo_geometry_candidates.py \
  tests/test_datav2_abo_geometry_review_template.py
```

Run geometry mining as a dry run first:

```bash
python scripts/datav2_mine_abo_geometry_candidates.py \
  --config configs/datav2_flat_panel_candidate_mining.json \
  --dry-run \
  --top-k 300
```

Run full geometry candidate mining only after the dry run looks sensible:

```bash
python scripts/datav2_mine_abo_geometry_candidates.py \
  --config configs/datav2_flat_panel_candidate_mining.json \
  --top-k 300
```

Generate the human-review template:

```bash
python scripts/datav2_make_abo_geometry_review_template.py \
  --candidates-csv data/candidates/datav2_abo_geometry_flat_panel_candidates.csv \
  --out-csv data/candidates/datav2_abo_geometry_human_review_template.csv \
  --top-k 120
```

Inspect the top 40 rows before any download:

```bash
head -n 41 data/candidates/datav2_abo_geometry_flat_panel_candidates.csv
```

`images.csv.gz` is auxiliary metadata only. Do not treat image rows as asset
candidates. Candidate rank is not acceptance; human review still decides
accepted/rejected rows and must fill `selected_input_view` before later phases.

## Phase 2L.2A Manual ABO Item-ID Workflow

Use this path when manual ABO product-page inspection identifies promising item
IDs that may not rank highly by geometry alone. Do not run Blender, Hunyuan,
Slurm, training, package installs, or checkpoint loading.

Create the item-id text file with one ID per line:

```text
data/candidates/datav2_manual_abo_item_ids.txt
```

Blank lines are ignored, lines starting with `#` are ignored, and inline notes
after `#` are allowed.

Resolve IDs against local `3dmodels.csv.gz` metadata:

```bash
python scripts/datav2_resolve_manual_abo_item_ids.py \
  --item-ids data/candidates/datav2_manual_abo_item_ids.txt \
  --metadata data/metadata/abo/3dmodels.csv.gz \
  --out-csv data/candidates/datav2_manual_abo_download_manifest.csv
```

Inspect found and missing counts:

```bash
less outputs/phase2l/manual_abo_download/manual_abo_resolve_summary.md
head -n 20 data/candidates/datav2_manual_abo_download_manifest.csv
```

Run the downloader only as a dry run first:

```bash
python scripts/datav2_download_manual_abo_glbs.py \
  --manifest data/candidates/datav2_manual_abo_download_manifest.csv \
  --out-root data/raw_assets/abo \
  --dry-run
```

Real download should happen only after reviewing the dry-run summary and only
when explicitly requested:

```bash
python scripts/datav2_download_manual_abo_glbs.py \
  --manifest data/candidates/datav2_manual_abo_download_manifest.csv \
  --out-root data/raw_assets/abo
```

Create the human-review template:

```bash
python scripts/datav2_make_manual_abo_review_template.py \
  --manifest data/candidates/datav2_manual_abo_download_manifest.csv \
  --out-csv data/candidates/datav2_manual_abo_human_review.csv
```

Later Blender contact sheets should be used to decide whether each asset is a
flat rectangular graphic panel and to fill `selected_input_view`.

## Phase 2L.2B Manual ABO Visual Inspection

Phase 2L.2B visually inspects the manually downloaded ABO GLBs. Codex may
create or check scripts, but Blender should be run manually by the user. Do not
run Hunyuan, A100 jobs, training, Slurm, package installs, or checkpoint loads.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_manual_abo_contact_sheet_index.py \
  tests/test_datav2_manual_abo_review_merge.py
```

Run a local Blender smoke with five assets:

```bash
/vol/bitbucket/ct1022/tools/bin/blender -b \
  --python scripts/datav2_inspect_manual_abo_blender.py -- \
  --config configs/datav2_manual_abo_visual_inspection.json \
  --limit 5
```

Generate contact sheets for the smoke outputs:

```bash
python scripts/datav2_make_manual_abo_contact_sheets.py \
  --config configs/datav2_manual_abo_visual_inspection.json
```

If the smoke looks correct, run the full local Blender inspection:

```bash
/vol/bitbucket/ct1022/tools/bin/blender -b \
  --python scripts/datav2_inspect_manual_abo_blender.py -- \
  --config configs/datav2_manual_abo_visual_inspection.json \
  --only-missing
```

Regenerate the full contact sheets:

```bash
python scripts/datav2_make_manual_abo_contact_sheets.py \
  --config configs/datav2_manual_abo_visual_inspection.json
```

Create the inspection-enriched review CSV:

```bash
python scripts/datav2_update_manual_abo_review_with_inspection.py \
  --config configs/datav2_manual_abo_visual_inspection.json \
  --out-csv data/candidates/datav2_manual_abo_human_review_with_inspection.csv
```

Human curation should fill accept/reject decisions, `selected_input_view`,
`alternative_input_view`, quality scores, leakage risk, and notes. The first
Data v2A training subset should use 30-40 high-quality accepted assets; do not
train on the full 101-asset set until QA supports a larger split.

## Phase 2L.2C Data v2 Frame-Panel Curation and Splits

Phase 2L.2C creates a curated manifest and reproducible train/validation/test
splits for the manually inspected framed-wall-art assets. Do not run Hunyuan,
Blender, A100 jobs, training, Slurm, package installs, or checkpoint loads.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_panel_curated_manifest.py \
  tests/test_datav2_frame_panel_splits.py
```

Optionally create a reject list before building the manifest:

```text
data/candidates/datav2_manual_abo_reject_item_ids.txt
```

Build the curated manifest:

```bash
python scripts/datav2_build_frame_panel_curated_manifest.py \
  --config configs/datav2_frame_panels_split.json
```

Generate fixed-seed mini40 and full101 splits:

```bash
python scripts/datav2_make_frame_panel_splits.py \
  --config configs/datav2_frame_panels_split.json
```

Inspect the split summary:

```bash
less outputs/phase2l/datav2_frame_panels/split_summary.md
head -n 20 data/manifests/datav2_frame_panels/datav2_frame_panels_split_membership.csv
```

Export the training plan:

```bash
python scripts/datav2_export_frame_panel_training_plan.py \
  --config configs/datav2_frame_panels_split.json
```

Do not train yet. The next phase is Hunyuan train-example rendering for the
curated mini40 split, followed by local structure and framing QA.

## Phase 2L.3A Render Mini40 Hunyuan Examples

Phase 2L.3A renders only the `datav2_frame_panels` mini40 split into official
Hunyuan3D-Paint-style training examples. Do not run Hunyuan, A100 jobs,
training, Slurm, package installs, or checkpoint loads. Blender commands are
manual user actions, not Codex actions.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_panel_render_plan.py \
  tests/test_datav2_frame_panel_examples_json.py \
  tests/test_datav2_frame_panel_examples_checker.py
```

Build the render plan:

```bash
python scripts/datav2_build_frame_panel_render_plan.py \
  --config configs/datav2_frame_panels_mini40_render.json
```

Run a local Blender smoke render with three assets:

```bash
/vol/bitbucket/ct1022/tools/bin/blender -b \
  --python scripts/datav2_render_frame_panel_examples_blender.py -- \
  --config configs/datav2_frame_panels_mini40_render.json \
  --limit 3
```

Build examples JSON files from successful render results:

```bash
python scripts/datav2_build_frame_panel_examples_json.py \
  --config configs/datav2_frame_panels_mini40_render.json
```

Run the local rendered-example checker:

```bash
python scripts/check_datav2_frame_panel_examples.py \
  --config configs/datav2_frame_panels_mini40_render.json
```

If the smoke passes, run the full local Blender render:

```bash
/vol/bitbucket/ct1022/tools/bin/blender -b \
  --python scripts/datav2_render_frame_panel_examples_blender.py -- \
  --config configs/datav2_frame_panels_mini40_render.json \
  --only-missing
```

Then rebuild examples JSON and run the checker again. Do not start A100
training yet; the next phase should run strict example validation and prepare
the mini40 training sbatch.

## Phase 2L.4A Mini40 True-PBR Training Setup

Phase 2L.4A prepares a conservative 500-step true-PBR training job for the
`datav2_frame_panels_mini40` rendered examples. It is a readiness and sbatch
setup phase: Codex should not submit Slurm jobs, run Hunyuan, load
checkpoints, or start full101 training.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_mini40_training_readiness.py \
  tests/test_datav2_frame_mini40_checkpoint_inspection.py
bash -n env/run_datav2_frame_mini40_train_a100.sbatch
```

Run the mini40 training readiness gate:

```bash
python scripts/check_datav2_frame_mini40_training_readiness.py \
  --config configs/datav2_frame_mini40_train.json
```

Expected success marker:

```text
PHASE2L4A_MINI40_TRAINING_READINESS_OK
```

Submit the A100 job manually only after the readiness gate passes:

```bash
sbatch env/run_datav2_frame_mini40_train_a100.sbatch
```

After the job finishes, inspect the Slurm log under
`logs/slurm/datav2_frame_mini40_truepbr_500_lr1e6-<jobid>.out`. Success means
the job reached max steps, found exactly one new checkpoint under
`checkpoints/datav2_frame_mini40_truepbr_500_lr1e6`, printed
`PHASE2L4A_MINI40_CHECKPOINT_INSPECTION_OK`, and ended with:

```text
PHASE2L4A_MINI40_TRUEPBR_TRAIN_OK
```

Inspect the checkpoint without loading it:

```bash
python scripts/inspect_datav2_frame_mini40_checkpoint.py \
  --config configs/datav2_frame_mini40_train.json
```

If the readiness gate fails, fix the package paths, examples JSONs, true-PBR
source path, or checkpoint configuration before submitting. If training fails,
review the Slurm log and do not proceed to full101 or comparison runs until the
mini40 checkpoint behavior is understood.

## Phase 2L.5A Mini40 True-PBR Evaluation Setup

Phase 2L.5A evaluates the mini40 true-PBR checkpoint with explicit per-asset
selected input views. Do not run Hunyuan, submit Slurm, run Blender, train,
load checkpoints in Codex, install packages, or compare against old wrong-input
baselines.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_mini40_input_view_review.py \
  tests/test_datav2_frame_mini40_eval_cases.py \
  tests/test_datav2_frame_mini40_eval_readiness.py \
  tests/test_datav2_frame_mini40_eval_aggregate.py
```

Generate the input-view review board and default override CSV:

```bash
python scripts/make_datav2_frame_mini40_input_view_review.py \
  --config configs/datav2_frame_mini40_eval.json
```

Review:

```text
outputs/phase2l/datav2_frame_panels/mini40_eval_truepbr500/input_view_review/input_view_review_board.jpg
data/manifests/datav2_frame_panels/datav2_frame_mini40_eval_input_view_overrides.csv
```

The user should confirm or edit `selected_input_view` for each val/test asset,
normally choosing `004` or `005`. The override CSV is the explicit human-review
artifact; do not silently hard-code a view.

Create eval cases after the override CSV is reviewed:

```bash
python scripts/make_datav2_frame_mini40_eval_cases.py \
  --config configs/datav2_frame_mini40_eval.json
```

Run readiness:

```bash
python scripts/check_datav2_frame_mini40_eval_readiness.py \
  --config configs/datav2_frame_mini40_eval.json
```

Expected marker:

```text
PHASE2L5A_MINI40_EVAL_READINESS_OK
```

Static-check the sbatch before submission:

```bash
bash -n env/run_datav2_frame_mini40_eval_infer_a100.sbatch
```

Commit the setup only if desired, then submit the A100 inference job manually
after readiness is OK:

```bash
sbatch env/run_datav2_frame_mini40_eval_infer_a100.sbatch
```

The job writes corrected-input base and fine-tuned outputs under:

```text
outputs/phase2l/datav2_frame_panels/mini40_eval_truepbr500/infer/base/<split>/<item_id>/
outputs/phase2l/datav2_frame_panels/mini40_eval_truepbr500/infer/finetuned/<split>/<item_id>/
```

After inference, create a render-eval config:

```bash
python scripts/make_datav2_frame_mini40_render_eval_configs.py \
  --config configs/datav2_frame_mini40_eval.json
```

Run the existing rendered-view evaluation path locally/manual as appropriate,
then aggregate:

```bash
python scripts/aggregate_datav2_frame_mini40_eval.py \
  --config configs/datav2_frame_mini40_eval.json
```

Inspect metrics and boards before making any quality claim. Do not run full101
or full80 until the mini40 val/test and train-sanity results have been
interpreted.

## Phase 2L.5B/C Mini40 Rendered-View Evaluation

Phase 2L.5B/C evaluates the mini40 base and fine-tuned outputs in final
rendered-view space. Codex may prepare and run local Python comparison scripts,
but must not run Blender, Hunyuan, Slurm, training, package installs, or
checkpoint loading.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_mini40_render_eval_cases.py \
  tests/test_datav2_frame_mini40_render_eval_readiness.py \
  tests/test_datav2_frame_mini40_rendered_compare.py \
  tests/test_datav2_frame_mini40_eval_aggregate_rendered.py
```

Create render-eval cases from completed A100 inference outputs:

```bash
python scripts/make_datav2_frame_mini40_render_eval_configs.py \
  --config configs/datav2_frame_mini40_eval.json
```

Run readiness before Blender:

```bash
python scripts/check_datav2_frame_mini40_render_eval_readiness.py \
  --config configs/datav2_frame_mini40_eval.json
```

Expected marker:

```text
PHASE2L5B_MINI40_RENDER_EVAL_READINESS_OK
```

Run a manual Blender smoke with two cases:

```bash
blender -b --python scripts/render_datav2_frame_mini40_eval_views_blender.py -- \
  --config configs/datav2_frame_mini40_eval.json \
  --limit 2
```

If the smoke looks correct, render missing views for all cases manually:

```bash
blender -b --python scripts/render_datav2_frame_mini40_eval_views_blender.py -- \
  --config configs/datav2_frame_mini40_eval.json \
  --only-missing
```

Compare rendered views locally:

```bash
python scripts/compare_datav2_frame_mini40_rendered_views.py \
  --config configs/datav2_frame_mini40_eval.json
```

Aggregate and inspect:

```bash
python scripts/aggregate_datav2_frame_mini40_eval.py \
  --config configs/datav2_frame_mini40_eval.json
```

Review:

```text
outputs/phase2l/datav2_frame_panels/mini40_eval_truepbr500/render_eval/boards/
outputs/phase2l/datav2_frame_panels/mini40_eval_truepbr500/summary/mini40_eval_summary.md
```

Do not run full80 or full101 until the rendered-view summary and boards are
interpreted. If val/test fine improves front metrics without non-front
degradation, consider full80 500-step. If results are tiny or mixed, stop for
report discussion or consider full80 cautiously. If fine worsens, do not scale
training.

## Phase 2L.6A Render Full101 Hunyuan Examples

Phase 2L.6A scales the curated frame-panel dataset from mini40 to full101
rendered Hunyuan3D-Paint examples. The mini40 checkpoint was stable but did not
clearly improve held-out val/test front views, so scale data before increasing
mini40 steps. Do not run Hunyuan, Blender in Codex, Slurm, training, package
installs, or checkpoint loading.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_panel_render_plan.py \
  tests/test_datav2_frame_panel_examples_json.py \
  tests/test_datav2_frame_panel_examples_checker.py
```

Build the full101 render plan:

```bash
python scripts/datav2_build_frame_panel_render_plan.py \
  --config configs/datav2_frame_panels_full101_render.json
```

Run an optional manual Blender smoke with three assets:

```bash
/vol/bitbucket/ct1022/tools/bin/blender -b \
  --python scripts/datav2_render_frame_panel_examples_blender.py -- \
  --config configs/datav2_frame_panels_full101_render.json \
  --limit 3
```

If the smoke looks correct, run the full manual Blender render resumably:

```bash
/vol/bitbucket/ct1022/tools/bin/blender -b \
  --python scripts/datav2_render_frame_panel_examples_blender.py -- \
  --config configs/datav2_frame_panels_full101_render.json \
  --only-missing
```

Build examples JSON files:

```bash
python scripts/datav2_build_frame_panel_examples_json.py \
  --config configs/datav2_frame_panels_full101_render.json
```

Run the local checker:

```bash
python scripts/check_datav2_frame_panel_examples.py \
  --config configs/datav2_frame_panels_full101_render.json
```

Expected counts are train 80, val 10, test 11, and all 101. Do not prepare or
submit A100 full80 training until the checker passes and the official-style
strict checker has also passed on the full101 examples JSONs.

## Phase 2L.6B Full80 True-PBR Training Setup

Phase 2L.6B prepares the `datav2_frame_full80_truepbr_500_lr1e6` A100 job using
the full101 rendered examples with an 80/10/11 train/val/test split. Codex
should not run Hunyuan, submit Slurm, train, load checkpoints, run Blender,
install packages, or prepare a 1000-step run in this phase.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_full80_training_readiness.py \
  tests/test_datav2_frame_full80_checkpoint_inspection.py
bash -n env/run_datav2_frame_full80_train_a100.sbatch
```

Run the full80 readiness gate:

```bash
python scripts/check_datav2_frame_full80_training_readiness.py \
  --config configs/datav2_frame_full80_train.json
```

Expected marker:

```text
PHASE2L6B_FULL80_TRAINING_READINESS_OK
```

The readiness report is written under:

```text
outputs/phase2l/datav2_frame_panels/full80_train_readiness/
```

Static-check the sbatch, then submit manually only after readiness passes:

```bash
bash -n env/run_datav2_frame_full80_train_a100.sbatch
sbatch env/run_datav2_frame_full80_train_a100.sbatch
```

The job should run official `train.py` for `max_steps: 500`, use the local
true-PBR source directory, and save exactly one new checkpoint under:

```text
checkpoints/datav2_frame_full80_truepbr_500_lr1e6/
```

After the job finishes, inspect the checkpoint without loading it:

```bash
python scripts/inspect_datav2_frame_full80_checkpoint.py \
  --config configs/datav2_frame_full80_train.json
```

Expected markers in the Slurm log:

```text
PHASE2L6B_FULL80_TRAINING_READINESS_OK
CUBLAS_MATMUL_OK
PHASE2L6B_FULL80_CHECKPOINT_INSPECTION_OK
PHASE2L6B_FULL80_TRUEPBR_TRAIN_OK
===== JOB END: SUCCESS =====
```

If readiness fails, fix rendered examples, counts, YAML true-PBR references, or
checkpoint isolation before submission. If the 500-step full80 run succeeds,
evaluate corrected-input base versus full80 fine-tuned outputs before deciding
whether a 1000-step run is justified.

## Phase 2L.7A Full80 Corrected-Input Evaluation Setup

Phase 2L.7A evaluates the full80 true-PBR 500-step checkpoint against corrected
input base inference on the full101 validation and test assets. Do not run
Hunyuan, submit Slurm, run Blender, train, install packages, load checkpoints,
or compare against old wrong-input baselines in Codex.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_frame_full80_input_view_review.py \
  tests/test_datav2_frame_full80_eval_cases.py \
  tests/test_datav2_frame_full80_eval_readiness.py \
  tests/test_datav2_frame_full80_render_eval_cases.py \
  tests/test_datav2_frame_full80_rendered_compare.py \
  tests/test_datav2_frame_full80_eval_aggregate.py
bash -n env/run_datav2_frame_full80_eval_infer_a100.sbatch
```

Create the input-view review board and default override CSV:

```bash
python scripts/make_datav2_frame_full80_input_view_review.py \
  --config configs/datav2_frame_full80_eval.json
```

Review or edit:

```text
outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/input_view_review/input_view_review_board.jpg
data/manifests/datav2_frame_panels/datav2_frame_full80_eval_input_view_overrides.csv
```

The default selected input view is `005`; use `004` only when human review says
it is more informative for an asset. Base and fine-tuned inference must use the
same selected input image.

Generate eval cases and run readiness:

```bash
python scripts/make_datav2_frame_full80_eval_cases.py \
  --config configs/datav2_frame_full80_eval.json

python scripts/check_datav2_frame_full80_eval_readiness.py \
  --config configs/datav2_frame_full80_eval.json
```

Expected marker:

```text
PHASE2L7A_FULL80_EVAL_READINESS_OK
```

Static-check the sbatch and submit manually only after readiness passes:

```bash
bash -n env/run_datav2_frame_full80_eval_infer_a100.sbatch
sbatch env/run_datav2_frame_full80_eval_infer_a100.sbatch
```

The sbatch writes base/fine outputs under:

```text
outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/infer/base/<eval_split>/<item_id>/
outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/infer/finetuned/<eval_split>/<item_id>/
```

After inference completes, create render-eval cases:

```bash
python scripts/make_datav2_frame_full80_render_eval_configs.py \
  --config configs/datav2_frame_full80_eval.json
```

Run render-eval readiness before Blender:

```bash
python scripts/check_datav2_frame_full80_render_eval_readiness.py \
  --config configs/datav2_frame_full80_eval.json
```

Run a manual Blender smoke with two cases:

```bash
blender -b --python scripts/render_datav2_frame_full80_eval_views_blender.py -- \
  --config configs/datav2_frame_full80_eval.json \
  --limit 2
```

If the smoke looks correct, render missing views for all cases manually:

```bash
blender -b --python scripts/render_datav2_frame_full80_eval_views_blender.py -- \
  --config configs/datav2_frame_full80_eval.json \
  --only-missing
```

Compare rendered views and aggregate:

```bash
python scripts/compare_datav2_frame_full80_rendered_views.py \
  --config configs/datav2_frame_full80_eval.json

python scripts/aggregate_datav2_frame_full80_eval.py \
  --config configs/datav2_frame_full80_eval.json
```

Review:

```text
outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/
outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/summary/full80_eval_summary.md
```

Do not prepare 1000-step training until the full80 500-step val/test and
train-sanity summaries and boards are interpreted.

## Phase 2L.7C Full80 Evaluation Closeout

Phase 2L.7C is documentation-only. Do not run Hunyuan, Blender, Slurm,
training, package installs, checkpoint loading, or file deletion.

Review the closeout and report materials:

```text
docs/phase2l7c_full80_eval_closeout.md
docs/phase2l_full80_failure_cases.md
docs/phase2l_report_materials.md
docs/current_project_state.md
```

The closeout records that full80-500 is stable and marginally positive in
aggregate, but visually mixed and not strong enough to claim clear superiority
over corrected-input base. Do not prepare full80-1000 unless the report-first
path is set aside and an optional low-learning-rate rescue is explicitly chosen.

## Phase 2L.2A ABO Probe Inspection Setup

Phase 2L.2A checks whether the top ABO geometry candidates are visually useful
enough to keep ABO in Data v2. It prepares manifests, a non-executing download
plan, Blender inspection/contact-sheet tooling, and a human-review template.
Do not download assets, run Blender, run Hunyuan, submit Slurm jobs, train, or
load checkpoints in Codex.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_abo_probe_manifest.py \
  tests/test_datav2_abo_probe_review_template.py
```

Prepare the top-40 probe manifest and availability report:

```bash
python scripts/datav2_prepare_abo_probe_manifest.py \
  --config configs/datav2_abo_probe_inspection.json
```

Create a safe download plan for missing assets:

```bash
python scripts/datav2_make_abo_download_plan.py \
  --probe-manifest data/candidates/datav2_abo_probe_top40_manifest.csv \
  --out-sh outputs/phase2l/abo_probe/download_missing_abo_assets.sh
```

The generated shell script only echoes planned downloads and contains commented
commands. Review it manually before editing or running anything.

After any selected GLBs exist locally, run Blender manually:

```bash
blender -b --python scripts/datav2_inspect_abo_probe_blender.py -- \
  --probe-manifest data/candidates/datav2_abo_probe_top40_manifest.csv \
  --output-root outputs/phase2l/abo_probe
```

Create the human-review template. This works even before Blender inspection:

```bash
python scripts/datav2_make_abo_probe_human_review_template.py \
  --probe-manifest data/candidates/datav2_abo_probe_top40_manifest.csv \
  --inspection-csv outputs/phase2l/abo_probe/blender_inspection_results.csv \
  --out-csv data/candidates/datav2_abo_probe_human_review_template.csv
```

Decision rule: if 8-12 acceptable assets are found, keep ABO as part of Data
v2. If fewer are found, prioritize Objaverse or Objaverse-XL metadata
acquisition.

## Phase 2L.2B ABO Deduped Acquisition Prep

Phase 2L.2B deduplicates the ABO geometry-ranked candidates before any GLB
download. It uses rounded extents, flatness/aspect ratios, face buckets, and
texture/material counts to avoid spending the probe on near-identical shapes.
Do not download assets, run Blender, run Hunyuan, submit Slurm jobs, train, or
load checkpoints in Codex.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q tests/test_datav2_abo_dedup_acquisition.py
```

Create the deduplicated candidate set and summary:

```bash
python scripts/datav2_dedupe_abo_probe_candidates.py \
  --config configs/datav2_abo_dedup_acquisition.json
```

Create the download-ready manifest without downloading:

```bash
python scripts/datav2_make_abo_dedup_download_manifest.py \
  --config configs/datav2_abo_dedup_acquisition.json
```

Inspect before any acquisition:

```bash
head -n 31 data/candidates/datav2_abo_dedup_probe_candidates.csv
head -n 31 data/candidates/datav2_abo_dedup_probe_download_manifest.csv
less outputs/phase2l/abo_dedup_acquisition/dedup_summary.md
```

The manifest is download-ready, but download still requires explicit user
approval and should happen manually outside Codex automation.

## Phase 2L.2C ABO Visual Inspection

Phase 2L.2C inspects the 30 downloaded deduped ABO GLBs and builds six-view
contact sheets for human curation. Do not run Blender, Hunyuan, Slurm, training,
package installs, or checkpoint loading in Codex.

Run static checks:

```bash
python -m compileall scripts tests
python -m pytest -q \
  tests/test_datav2_abo_visual_human_review.py \
  tests/test_datav2_abo_contact_sheet_inputs.py
```

Run Blender manually when ready:

```bash
/vol/bitbucket/ct1022/tools/bin/blender -b \
  --python scripts/datav2_inspect_abo_dedup_blender.py -- \
  --config configs/datav2_abo_visual_inspection.json
```

Build contact sheets after renders exist:

```bash
python scripts/datav2_make_abo_visual_contact_sheets.py \
  --config configs/datav2_abo_visual_inspection.json
```

Create the human-review CSV. This works before inspection, but it is most useful
after contact sheets exist:

```bash
python scripts/datav2_make_abo_visual_human_review.py \
  --config configs/datav2_abo_visual_inspection.json
```

Accept only clear flat rectangular graphic panels with readable or meaningful
front texture. Reject plain, broken, not-panel-like, cluttered, bad-import, or
weak-texture assets. If at least 8-12 are acceptable, keep ABO as a Data v2
seed; otherwise prioritize Objaverse or Objaverse-XL metadata acquisition.

## Phase 2M LoRA Rescue

Phase 2M is a conservative LoRA rescue path after the full80-500 evaluation showed only marginal aggregate improvement and persistent front-to-back leakage. It is not a full fine-tuning run yet.

Run local static checks first:

```bash
python -m compileall src scripts tests
python -m pytest -q tests/test_phase2m_lora_targeting.py
bash -n env/run_phase2m_lora_inventory_a100.sbatch
bash -n env/run_phase2m_zero_lora_smoke_a100.sbatch
```

Manual A100 M0 inventory:

```bash
sbatch env/run_phase2m_lora_inventory_a100.sbatch
```

Expected success token:

```text
PHASE2M_M0_INVENTORY_OK
```

Manual A100 M1 zero-LoRA smoke, only after M0 passes:

```bash
sbatch env/run_phase2m_zero_lora_smoke_a100.sbatch
```

Expected success token:

```text
PHASE2M_M1_ZERO_LORA_SMOKE_OK
```

M1 writes the canonical summary `outputs/phase2m/zero_lora_smoke/zero_lora_smoke_summary.json`. The compatibility alias `outputs/phase2m/zero_lora_smoke/smoke_summary.json` is also accepted, but future checks should prefer the canonical filename.

Check M2 readiness without training:

```bash
python scripts/check_phase2m_lora_training_ready.py --dry-run
```

The initial `ref_dino` preset targets only exact `nn.Linear` projection modules under `attn_refview` and `attn_dino`: `to_q`, `to_k`, `to_v`, and `to_out.0`. Do not target `attn_multiview`, `attn1`, `attn2`, feed-forward layers, convolutions, normalization layers, `learned_text_clip`, or DINO encoder weights until a later explicit phase.

Phase 2M outputs belong under `outputs/phase2m/`. Do not merge LoRA into the base model, do not call `save_pretrained` on a full model, and do not overwrite checkpoints or upstream Hunyuan files.

M2 adapter-only training preset:

```bash
python scripts/check_phase2m_lora_training_ready.py --dry-run
bash -n env/run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch
sbatch env/run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch
```

Only after the one-step smoke passes, run the 300-step job:

```bash
bash -n env/run_phase2m_train_lora_refdino_r4_lr5e5_300_a100.sbatch
sbatch env/run_phase2m_train_lora_refdino_r4_lr5e5_300_a100.sbatch
```

Expected M2 output directory:

```text
outputs/phase2m/lora_train_refdino_r4_lr5e5_300/
```

M2 saves adapter-only files: `adapter_step_000100.pt`, `adapter_step_000200.pt`, `adapter_step_000300.pt`, `adapter_final.pt`, `adapter_config.json`, and `training_summary.json`. It must not save a `.ckpt` or full Hunyuan model.

