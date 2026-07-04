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
