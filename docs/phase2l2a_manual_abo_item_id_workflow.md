# Phase 2L.2A Manual ABO Item-ID Workflow

## Goal

Manual ABO item-id selection lets the user bring product-page judgment back
into the Data v2 funnel without downloading broad assets. The workflow starts
from a small text file of manually chosen ABO item IDs, verifies which IDs have
local 3D model metadata, writes a download manifest, and prepares a curation
CSV for later human review.

This phase does not run Blender, run Hunyuan, train, submit Slurm jobs, load
checkpoints, install packages, or download anything unless the user explicitly
runs the download script.

## Why Manual IDs Help

The geometry-ranked ABO list is useful for finding flat objects, but
`3dmodels.csv.gz` has weak semantic fields. Manual inspection of ABO product
pages can identify package fronts, signs, covers, labels, and other visually
informative panels that geometry alone cannot distinguish from plain boards or
furniture parts.

Product pages can also be misleading: some items have product images but no
matching row in the local `data/metadata/abo/3dmodels.csv.gz` metadata. The
resolver treats those safely as `missing_in_metadata` and never creates a
download URL for them.

## Item-ID File

Write manual candidates to:

```text
data/candidates/datav2_manual_abo_item_ids.txt
```

Rules:

- one item ID per line
- blank lines are ignored
- lines starting with `#` are ignored
- inline notes after `#` are allowed
- duplicate IDs are removed while preserving the first occurrence

Example:

```text
# flat product boxes found by manual web review
B000EXAMPLE1 # cereal-like front panel
B000EXAMPLE2
```

## Resolve Metadata

Resolve item IDs against local ABO metadata:

```bash
python scripts/datav2_resolve_manual_abo_item_ids.py \
  --item-ids data/candidates/datav2_manual_abo_item_ids.txt \
  --metadata data/metadata/abo/3dmodels.csv.gz \
  --out-csv data/candidates/datav2_manual_abo_download_manifest.csv
```

The resolver writes:

- `data/candidates/datav2_manual_abo_download_manifest.csv`
- `outputs/phase2l/manual_abo_download/manual_abo_resolve_summary.md`
- `outputs/phase2l/manual_abo_download/manual_abo_resolve_summary.json`

For found IDs, it records the ABO relative GLB path, HTTPS download URL, and
expected local path under:

```text
data/raw_assets/abo/<relative_path_from_metadata>
```

For missing IDs, `status` is `missing_in_metadata` and the URL is blank.

## Optional Download

Dry-run first:

```bash
python scripts/datav2_download_manual_abo_glbs.py \
  --manifest data/candidates/datav2_manual_abo_download_manifest.csv \
  --out-root data/raw_assets/abo \
  --dry-run
```

Only after reviewing the dry-run summary should the real download be run
manually:

```bash
python scripts/datav2_download_manual_abo_glbs.py \
  --manifest data/candidates/datav2_manual_abo_download_manifest.csv \
  --out-root data/raw_assets/abo
```

The downloader processes only rows with `status=found`, skips missing metadata
rows, and skips existing files unless `--overwrite` is passed.

## Human Review

Create the manual curation template:

```bash
python scripts/datav2_make_manual_abo_review_template.py \
  --manifest data/candidates/datav2_manual_abo_download_manifest.csv \
  --out-csv data/candidates/datav2_manual_abo_human_review.csv
```

No asset is accepted automatically. Human review must fill fields such as
`human_decision`, `subclass`, `selected_input_view`, `primary_eval_views`,
quality scores, leakage risk, and notes.

After selected GLBs exist locally, later Blender contact-sheet inspection can
decide whether the object is truly a flat rectangular graphic panel and which
view should become `selected_input_view`.
