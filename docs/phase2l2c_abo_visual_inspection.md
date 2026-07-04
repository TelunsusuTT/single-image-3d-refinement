# Phase 2L.2C ABO Visual Inspection

## Goal

Phase 2L.2C verifies whether the 30 downloaded, deduplicated ABO candidates are
visually useful enough to seed Data v2. It prepares Blender import inspection,
six-view thumbnail rendering, paginated contact sheets, and a human-review CSV.

This phase does not run Hunyuan, train, submit Slurm jobs, install packages, or
load checkpoints. Blender is run manually by the user, not by Codex.

## Why Geometry Metadata Is Insufficient

Phase 2L.1B and 2L.2B selected technically plausible flat panels using geometry,
texture/material counts, image resolution, and deduplication. Those signals do
not prove that an asset has readable front graphics, a meaningful label, useful
texture, clean import behavior, or a clear selected input view.

Visual inspection is required before ABO can become a Data v2 seed.

## Blender Inspection

The Blender runtime script checks each local GLB for:

- import success
- object and mesh counts
- material count
- detected texture image count
- bounding-box extents
- face count
- six fixed-view rendered thumbnails

The script writes `inspection_results.csv`, Markdown/JSON summaries, and
per-asset rendered views under `outputs/phase2l/abo_visual_inspection/`.

## Contact Sheets

Contact sheets make human review fast. Each page shows 10 assets with metadata
and six rendered views. The sheets are for deciding whether each asset is a
clear flat rectangular graphic panel and which view should become
`selected_input_view`.

## Human Decision Rule

Accept an asset only if it is a clear flat rectangular graphic panel with a
readable or meaningful front texture.

Reject assets that are plain, not texture-heavy, broken, multi-object clutter,
bad imports, not panel-like, or otherwise unsuitable for selected-input-view
curation.

## Pass/Fail Gate

If at least 8-12 assets are acceptable, keep ABO as a Data v2 seed source. If
fewer than 8 acceptable assets are found, switch priority to Objaverse or
Objaverse-XL metadata acquisition.

Human decisions remain final. Automated import/render success is evidence, not
dataset membership.
