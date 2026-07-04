# Phase 2L.2A ABO Probe Inspection

## Goal

Phase 2L.2A validates whether ABO geometry-ranked flat-panel candidates are
visually useful enough to keep ABO as a Data v2 source. It prepares a top-40
availability manifest, a download/acquisition plan, Blender inspection and
contact-sheet runtime tooling, and a human-curation CSV.

This phase does not run Hunyuan, train, submit Slurm jobs, download assets,
install packages, or load checkpoints. Blender scripts may be prepared, but
Blender is run manually later.

## Why Metadata Is Insufficient

Phase 2L.1B ranked ABO assets using flatness, panel aspect ratio, mesh count,
face count, texture count, material count, and image-resolution metadata. That
produced technically clean candidates, but metadata still cannot show whether an
asset actually looks like a useful graphic panel, has an informative front view,
or contains visually meaningful texture.

ABO candidates therefore need visual inspection before download effort expands.

## Why Top 40

The top 40 candidates are enough for a small source-quality probe. They are
large enough to reveal whether ABO contains usable flat rectangular graphic
panels, but small enough to inspect manually without turning the phase into a
dataset download or training run.

## Expected Outputs

Phase 2L.2A prepares:

- availability report:
  `outputs/phase2l/abo_probe/availability_report.md`
- availability JSON:
  `outputs/phase2l/abo_probe/availability_report.json`
- acquisition/download plan:
  `outputs/phase2l/abo_probe/download_missing_abo_assets.sh`
- top-40 probe manifest:
  `data/candidates/datav2_abo_probe_top40_manifest.csv`
- Blender inspection outputs:
  `outputs/phase2l/abo_probe/blender_inspection_results.csv`
  `outputs/phase2l/abo_probe/blender_inspection_summary.md`
  `outputs/phase2l/abo_probe/blender_inspection_summary.json`
- contact sheets:
  `outputs/phase2l/abo_probe/contact_sheets/`
- human curation CSV:
  `data/candidates/datav2_abo_probe_human_review_template.csv`

Generated reports and candidate CSVs are planning artifacts and should not be
committed unless explicitly requested.

## Decision Rule

After manual inspection:

- If at least 8-12 acceptable assets are found, keep ABO as part of Data v2.
- If fewer than 8 acceptable assets are found, prioritize Objaverse and
  Objaverse-XL metadata acquisition for Data v2 instead of spending more effort
  on ABO.

Acceptance still requires human review. Rank, geometry, importability, and
contact sheets are evidence, not final dataset membership.
