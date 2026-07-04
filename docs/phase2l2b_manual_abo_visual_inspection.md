# Phase 2L.2B Manual ABO Visual Inspection

## Goal

Phase 2L.2B inspects the 101 manually selected and downloaded ABO GLBs with a
local Blender workflow. It renders six fixed views per asset, creates contact
sheets, and prepares an inspection-enriched review CSV so the user can choose
accepted Data v2A assets and set `selected_input_view`.

This phase does not run Hunyuan, train, submit Slurm jobs, load checkpoints, or
install packages.

## Why Contact Sheets Are Required

The manual item-id set is semantically strong, but product-page semantics do
not prove that the downloaded GLB imports correctly, has useful textures, or
shows a clear front-facing graphic panel from the standard six-view turntable.
Contact sheets make the curation decision visual and repeatable before any
training data is rendered.

## Local Blender, Not A100

This workflow uses Blender only for import inspection and lightweight preview
renders. It is a local asset QA step, not Hunyuan inference or training, and it
does not require an A100.

## Manual Decisions

Human curation should fill:

- `human_decision`
- `reject_reason`
- `subclass`
- `selected_input_view`
- `alternative_input_view`
- `primary_eval_views`
- `front_quality_score`
- `texture_quality_score`
- `leakage_risk`
- `notes`

Preferred candidate input views are currently `004` and `005`, but the contact
sheets should decide per asset.

## Decision Rule

The first training subset should use only 30-40 high-quality accepted assets.
The full 101-item manual set should not become training data automatically. A
larger 80/10/11 train/validation/holdout split should only be considered after
Blender inspection, contact-sheet review, and Hunyuan-style render QA.

## Outputs

The default output root is:

```text
outputs/phase2l/manual_abo_visual_inspection
```

Expected outputs are:

- `renders/<ITEM_ID>/000.png` through `005.png`
- `inspection_results.csv`
- `inspection_summary.md`
- `inspection_summary.json`
- `contact_sheets/contact_sheet_001.jpg`, etc.
- `contact_sheets/contact_sheet_index.md`
- `data/candidates/datav2_manual_abo_human_review_with_inspection.csv`
