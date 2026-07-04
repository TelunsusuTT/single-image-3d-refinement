# Phase 2L.2B ABO Deduped Acquisition Prep

## Goal

Phase 2L.2B prepares a deduplicated ABO acquisition set before downloading any
GLBs. It takes the geometry-ranked ABO candidate pool and selects about 30
unique candidates for a small download/inspection probe.

This phase does not download assets, run Blender, run Hunyuan, submit Slurm
jobs, install packages, train, or load checkpoints.

## Why Deduplicate

The Phase 2L.1B geometry ranking produced a technically clean top set, but many
ABO candidates can still be near-duplicates: same shape family, same aspect
ratio, similar face counts, and identical texture/material counts. Downloading
all of those would waste time and storage before visual quality is known.

Deduplication reduces the probe to a broader sample of geometry types while
preserving the highest-ranked candidates.

## Deduplication Key

The dedupe key is built from rounded geometry and technical fields:

- `extent_x`, `extent_y`, `extent_z`
- `flatness_ratio`
- `panel_aspect_ratio`
- `faces`
- `textures`
- `materials`

Extents and ratios are rounded so tiny numeric differences do not create fake
uniqueness. Face counts are bucketed to keep similar-complexity models together.

## Selection Preference

Within each duplicate group, the chosen row prefers:

1. earlier rank
2. higher image resolution
3. moderate face count

The output is still a candidate acquisition plan, not final dataset membership.
Human review and visual inspection remain required.

## Outputs

Phase 2L.2B writes:

- `data/candidates/datav2_abo_dedup_probe_candidates.csv`
- `data/candidates/datav2_abo_dedup_probe_download_manifest.csv`
- `outputs/phase2l/abo_dedup_acquisition/dedup_summary.md`
- `outputs/phase2l/abo_dedup_acquisition/dedup_summary.json`

The download manifest is download-ready, but Codex must not download anything.
The user should inspect the manifest before running any acquisition step.
