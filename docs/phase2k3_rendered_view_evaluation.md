# Phase 2K.3 Rendered-View Evaluation

Phase 2K.3 evaluates the true-PBR 200-step checkpoint in final rendered-view
space. The earlier UV texture-map comparisons are useful diagnostics, but they
do not prove visual quality improvement because UV-space differences can be
hard to interpret after projection onto the mesh, lighting, visibility, and
camera framing.

Cases included:

- `B075YLTF7Q`
- `B07HSK7MXZ`
- `B073NZS57V`
- `B07B8MWCR8`

Comparison design:

- base render vs fine-tuned render
- base render vs reference `render_cond` view
- fine-tuned render vs reference `render_cond` view

Metrics:

- MAE
- RMSE
- PSNR
- simple global SSIM-like score
- histogram distance
- color mean shift
- edge/gradient difference
- foreground-mask full-image and foreground-only metrics

Non-goals:

- no training
- no Hunyuan inference
- no new checkpoint
- no final thesis claim yet

Expected outputs:

- per-case rendered images
- per-case view boards
- per-case metrics JSON/MD
- aggregate summary JSON/MD
