# Phase 2G.4 Checkpoint Load-Only Smoke

Phase 2G.4 prepares the first fine-tuned checkpoint load-only smoke test. It
builds the official base Hunyuan3D-Paint inference pipeline, loads the Phase 2F
checkpoint tensors into the recommended inference UNet target, and exits without
running inference.

Mapping:

- checkpoint prefix: `unet.`
- target: `paint_pipeline.models["multiview_model"].pipeline.unet`
- load mode: `strict=True`

Phase 2G.3 only compared key names. Phase 2G.4 actually loads tensors, so it
must also verify tensor shapes before calling `load_state_dict`.

Non-goals:

- no full inference
- no visual comparison
- no base-vs-fine-tuned quality claim

Expected success signs:

- official pipeline is built
- checkpoint is loaded on CPU
- `1747` transformed keys are found
- transformed key set exactly matches the target UNet key set
- tensor shape check passes
- strict `load_state_dict` succeeds
- `PHASE2G4_CHECKPOINT_LOAD_ONLY_OK`
