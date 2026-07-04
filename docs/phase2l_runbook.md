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
