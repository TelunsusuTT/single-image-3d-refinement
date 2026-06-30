# Phase 2A Asset Selection

Phase 2A builds a lightweight, metadata-driven selection funnel for choosing
texture-heavy, product-like ABO candidates for later Hunyuan3D-Paint pilot
fine-tuning. It produces candidate manifests and a local review gallery only.

This phase does not put assets directly into training data. Later phases will
download selected GLBs, inspect them with Blender, render Hunyuan3D-Paint-style
examples, and run the official structure and training smoke checkers.

## Why ABO First

ABO is the first target because it is product-like by design. It has catalog
metadata, catalog thumbnails for human review, and glTF/PBR 3D models that are a
better fit for texture-heavy object fine-tuning than broad web-scale 3D data.

Objaverse and Objaverse-XL are deferred because they are larger and more diverse,
but noisier for this specific pilot. They will require more filtering before the
asset download and inspection phases.

## Selection Funnel

The Phase 2A funnel is:

1. Metadata filter
2. Candidate scoring
3. Thumbnail gallery
4. Manual selection
5. `selected_assets.csv`
6. Phase 2B GLB download and inspection

The metadata filter should only use small ABO metadata files. ABO listing
metadata currently uses `listings_0.json.gz` through `listings_9.json.gz`; later
listing indices are not expected by default. The gallery should only use small
catalog thumbnails. Thumbnails use the ABO small-image prefix
`images/small/<image_path>`, while original catalog image links use
`images/original/<image_path>`. Do not download GLBs, render data, or large
archives in Phase 2A.

## Texture-Heavy Criteria

Prefer assets with:

- several texture or image files
- high maximum texture or image resolution
- semantic keywords such as `package`, `packaging`, `box`, `bottle`, `can`,
  `label`, `book`, `cover`, `poster`, `sign`, `sticker`, `logo`, `mug`,
  `container`, `carton`, or related product terms
- moderate face counts that are likely to import and render reliably
- thumbnails that visibly confirm labels, graphics, packaging, or printed
  detail

The candidate index computes approximate metadata scores. Human thumbnail review
is still required before marking assets for Phase 2B. Furniture-heavy top results
are a scoring signal: adjust semantic scoring and filters rather than manually
deleting metadata rows.

## Rejection Rules

Reject candidates with:

- no texture signal
- pure single-color appearance
- furniture with no labels or graphics
- huge scenes or room-scale assets
- characters, humans, animals, hair, fur, or plush-like objects
- extreme polygon counts
- weak, missing, or ambiguous previews

## Download Rule

Do not download large archives such as full ABO 3D model bundles. Phase 2A may
download only small metadata files and selected top-k thumbnails when the user
runs those scripts manually. GLB downloads belong to Phase 2B.
