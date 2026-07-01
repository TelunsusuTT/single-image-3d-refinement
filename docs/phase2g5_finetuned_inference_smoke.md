# Phase 2G.5 Fine-Tuned Inference Smoke

Phase 2G.5 is the first real fine-tuned Hunyuan3D-Paint inference smoke. It
loads the Phase 2F checkpoint into the official inference pipeline UNet and
runs one no-remesh inference on the prepared `B075YLTF7Q` case.

This differs from Phase 2G.4: Phase 2G.4 loaded the checkpoint only and exited.
Phase 2G.5 loads the checkpoint and then calls the official paint pipeline.

Mapping:

- checkpoint keys: `unet.*`
- transform: strip `unet.`
- target: `paint_pipeline.models["multiview_model"].pipeline.unet`

The run uses `--no-remesh` because fixed mesh comparison is required, and the
earlier remesh path failed through `pymeshlab` / Qt.

Non-goals:

- no final quality claim
- no base-vs-fine-tuned board yet
- no generalization claim

Expected outputs:

- `outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke/`
- `finetuned_textured_mesh.obj`
- `finetuned_textured_mesh.glb`
- fine-tuned texture maps emitted by the official pipeline
