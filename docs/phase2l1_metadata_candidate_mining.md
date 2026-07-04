# Phase 2L.1A Metadata Candidate Mining

## Goal

Phase 2L.1A creates a reusable metadata-first workflow for finding Data v2
candidates in the flat rectangular graphic panel subclass. It inventories local
metadata, normalizes heterogeneous metadata rows, ranks likely candidates, and
produces CSV/Markdown files for human review.

This phase does not download assets, run Blender, run Hunyuan, submit Slurm
jobs, train, or load checkpoints.

## Why Metadata First

Metadata-first filtering keeps the next dataset pass small and inspectable. It
lets us use local ABO, Objaverse, and Objaverse-XL metadata to find likely
assets before spending time or storage on GLBs, thumbnails, Blender inspection,
or rendering.

The expected mining target is roughly 300-500 ranked candidate records if
metadata is available. Later human curation should narrow this to about 30
accepted assets.

## Why Flat Rectangular Graphic Panels

Phase 2K.4 showed that the selected input view is a dominant bottleneck.
Flat rectangular graphic panels are a narrower class where front/back semantics
are meaningful and selected input views can be curated consistently.

Examples include framed posters, wall art, plaques, flat signs, canvas prints,
decorative graphic panels, book-cover-like flat boards, and
package-front-like flat panels.

The mining workflow down-ranks bottles, cans, mugs, cylinders, generic
furniture, characters, vehicles, scenes, plants, sculpture, glass objects, pure
wood boards with no graphic texture, and cluttered multi-object scenes.

## Human Review Remains Required

Candidate mining does not decide final dataset membership. The scripts only
produce ranked suggestions. Human curation will later mark accepted/rejected
rows and fill fields such as `selected_input_view`, `primary_eval_views`,
front-view quality, texture quality, leakage risk, and notes.

This is important because metadata rarely proves that an asset has a dominant
usable graphic panel or an informative front view.

## Expected Outputs

The inventory step writes:

- `outputs/phase2l/metadata_inventory/metadata_sources_report.md`
- `outputs/phase2l/metadata_inventory/metadata_sources_report.json`

The mining step writes:

- `data/candidates/datav2_flat_panel_metadata_candidates.csv`
- `data/candidates/datav2_flat_panel_metadata_candidates.md`
- `data/candidates/datav2_flat_panel_candidate_summary.json`

The human-review template step writes:

- `data/candidates/datav2_flat_panel_human_review_template.csv`

Generated candidate CSVs and reports should not be committed unless explicitly
requested.

## Connection to Later Phases

Phase 2L.1A feeds later Data v2 steps:

1. Inventory local metadata sources.
2. Mine and rank candidate assets from metadata only.
3. Create a human-review template for curation.
4. Later, after review, download only selected assets.
5. Run Blender inspection only on selected assets.
6. Build contact sheets and select input views.
7. Build corrected-input base benchmarks before any new training.

No asset is training data until it passes human curation, local download,
inspection, rendering, and packaging checks.
