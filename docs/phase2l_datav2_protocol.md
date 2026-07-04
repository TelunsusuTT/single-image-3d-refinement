# Phase 2L Data v2 Protocol

## Goal

Create an input-view-aware Data v2 plan before mining more metadata or training
again.

## Target Subclass

Focus on flat rectangular graphic panels:

- product-package fronts
- book or box covers
- posters and signs
- labeled flat containers
- rectangular graphic surfaces with clear front/back semantics

## Why Not Broad Texture-Heavy

The broad "texture-heavy" category mixes too many different failure modes:

- furniture and shelves can dominate metadata rankings
- cylindrical products have different view-selection rules
- mugs, cans, boxes, signs, and books need different camera/front heuristics
- texture presence alone does not guarantee a useful front/reference view

Phase 2K.4 showed that input-view selection can dominate apparent quality. Data
v2 should reduce category variance before increasing training strength.

## Source Pools

Use metadata-first filtering over:

- ABO
- Objaverse
- Objaverse-XL

Do not download broad datasets at this stage. Start with metadata and thumbnails,
then select a small curated pool.

## Metadata-First Filtering

Candidate metadata should capture:

- source and source id
- category/subclass
- candidate front-view evidence
- expected graphic-panel quality
- license
- mesh availability
- texture/material availability
- thumbnail or preview availability
- rejection reason when rejected

## Human Curation

Human review is required before rendering or training. Review should reject:

- ambiguous front view
- weak or absent graphic panel
- large scene assets
- pure geometry
- mostly single-color objects
- character, animal, furry, or hair-heavy objects
- assets likely to fail import/rendering

## Required Selected Input View

Every accepted Data v2 asset must include:

- `selected_input_view`
- evidence image path
- curation note explaining why that view is informative
- front/back/all-view evaluation assignment

The selected input view should be used for corrected-input base benchmarking
before any training experiment.

## Evaluation Split

Report metrics separately for:

- front/input-visible views
- back/unseen views
- all views

This keeps front fidelity, all-view consistency, and backside texture leakage
separate.

## Training Gate

Do not train on Data v2 until a corrected-input base benchmark has been built.
The baseline must use the selected informative input view for each asset. New
training should only proceed after the base benchmark establishes the remaining
gap that fine-tuning is meant to improve.
