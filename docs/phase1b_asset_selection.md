# Phase 1B Asset Selection

Phase 1B plans how to select 1-3 texture-heavy candidate assets for later
Hunyuan3D-Paint packaging and fine-tuning checks. The output of this phase is a
small, reviewable candidate list, not a trained model or converted dataset.

This phase does not train, download large datasets, run Blender, run Hunyuan, or
submit Slurm jobs.

## Preferred Asset Types

Prefer compact objects where image-space texture detail matters more than raw
geometry:

- packaging box
- product box
- drink can
- book cover
- logo mug
- poster/sign
- carton
- labeled container
- sticker-like object

These objects are useful first targets because a small training set can show
whether fine-tuning improves labels, graphics, decals, and layout preservation.

## Rejection Rules

Reject candidates with any of these properties:

- no texture
- no UV
- missing texture files
- pure geometry
- pure single-color object
- large scene asset
- character / human / animal
- furry or hair-heavy object
- extremely high polygon count
- Blender import likely to fail

The first pass should favor boring, reliable assets over impressive but fragile
ones. A clean textured box is more useful for Phase 1 than a complex scene that
fails conversion.

## Data Source Priority

Use this order when filling the candidate list:

1. ABO product-like assets
2. Objaverse / Objaverse-XL selected texture-heavy assets
3. synthetic packaging fallback

ABO is preferred because product-like assets are closer to the packaging and
labeled-object use case. Objaverse can broaden coverage when specific
texture-heavy objects are easy to identify. Synthetic packaging is a fallback
when real assets are hard to license, inspect, or convert.

## First Target Scale

Start with one asset for a smoke test. Once the packaging path is understood,
expand to a three-asset mini set.

The one-asset smoke should answer whether the conversion process can produce the
expected Hunyuan3D-Paint structure. The three-asset mini set should then check
whether the process generalizes across a few texture-heavy categories.
