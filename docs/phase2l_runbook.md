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
