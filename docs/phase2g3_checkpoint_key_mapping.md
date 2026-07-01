# Phase 2G.3 Checkpoint Key Mapping

Phase 2G.3 prepares a safe A100 key-inspection job for the Phase 2F
`pilot_v1_overfit_500` checkpoint. The goal is to understand whether the
training checkpoint keys line up with the base inference pipeline UNet keys
before any fine-tuned inference is attempted.

Fine-tuned inference cannot be run directly yet because the official
`demo.py` does not expose a checkpoint argument, and the Phase 2F checkpoint is
a Lightning training checkpoint rather than a confirmed inference UNet
state_dict. Loading it blindly could silently miss keys, load the wrong nested
module, or accidentally compare base output against base output again.

Phase 2G.3 inspects:

- checkpoint top-level structure
- checkpoint `state_dict` key count and sampled keys
- likely training prefixes such as `unet.unet.`, `unet.unet.unet.`, and
  `unet.controlnet.`
- base inference pipeline UNet `state_dict` keys
- key overlap after stripping candidate prefixes

The A100 job writes:

- checkpoint key summary JSON and Markdown
- inference UNet key summary JSON and Markdown
- candidate prefix overlap JSON and Markdown

Non-goals:

- no full inference
- no visual comparison
- no checkpoint loading into an inference model
- no official Hunyuan source modification

Expected success means the checkpoint key summary is generated, the inference
UNet key summary is generated, and the overlap report either identifies a
recommended load strategy or explicitly marks the mapping as `UNKNOWN`.

## Phase 2G.3b Multi-Candidate Compare

The first A100 key inspection succeeded, but the initial compare report returned
`UNKNOWN` because it compared against a single inference candidate. The logged
counts suggest a likely nested mapping: checkpoint keys under `unet.unet.*` have
count `1061`, and `paint_pipeline.models['multiview_model'].pipeline.unet.unet`
also has `1061` keys.

Phase 2G.3b refines the comparison across every discovered inference candidate
and every candidate strip transform. It should identify whether stripping
`unet.unet.` maps the checkpoint subset onto the nested inference UNet.
