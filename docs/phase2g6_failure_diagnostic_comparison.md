# Phase 2G.6 Failure Diagnostic Comparison

Phase 2G.6 diagnoses why the fine-tuned GLB appears visually degraded. This is
a diagnostic comparison only, not a quality claim.

Inputs:

- base run: `outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke/`
- fine-tuned run: `outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke/`
- case reference image: `outputs/phase2g/infer_cases/B075YLTF7Q/input/image.png`

The comparison separates likely failure sources:

- albedo map differences
- metallic map differences
- roughness map differences
- GLB/OBJ inventory and material-binding clues

Non-goals:

- no final quality claim
- no retraining decision yet
- no checkpoint loading change yet

Expected outputs:

- `outputs/phase2g/compare/B075YLTF7Q/base_vs_finetuned_texture_board.jpg`
- `outputs/phase2g/compare/B075YLTF7Q/base_vs_finetuned_metrics.json`
- `outputs/phase2g/compare/B075YLTF7Q/base_vs_finetuned_report.md`

## Phase 2G.6a Grayscale Handling

The initial diagnostic board run failed because metallic and roughness maps may
be grayscale, while `image_stats()` assumed RGB-style extrema. PIL returns
`(min, max)` for grayscale images and `((min, max), ...)` for multi-channel
images. Phase 2G.6a normalizes both forms and compares albedo as RGB while
comparing metallic and roughness as grayscale.
