# Phase 2G Checkpoint Inference Sanity

Phase 2G prepares a base-vs-fine-tuned inference sanity check for the
`pilot_v1` checkpoint. Phase 2F saved a 500-step checkpoint; Phase 2G checks
whether that checkpoint can be wired into the official Hunyuan3D-Paint
inference path.

This is not training. Do not load the checkpoint in Codex, run GPU inference, or
modify the official Hunyuan source tree during the preparation step.

## Interface Inspection

The first step is read-only source inspection of the official `hy3dpaint`
directory. The current inspection target is:

```text
/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint
```

Initial read-only inspection found `demo.py` as the quick inference entrypoint.
It constructs `Hunyuan3DPaintConfig`, creates `Hunyuan3DPaintPipeline`, and uses
hardcoded inputs under `./assets/case_1/`. The README describes `max_num_view`
and `resolution` as optional settings, but `demo.py` does not expose an argparse
checkpoint argument. Model loading appears to happen through
`utils/multiview_utils.py`, which uses `DiffusionPipeline.from_pretrained` for
the base Hunyuan paint pipeline. Training checkpoint loading logic is visible in
`train.py`, but it is not directly exposed by `demo.py`.

If no custom checkpoint argument is found, Phase 2G will likely need a
project-local wrapper or config patch for an A100 inference job. That wrapper
must live in this project, not in the official Hunyuan checkout.

## Comparison Design

The base and fine-tuned comparison should use:

- the same mesh
- the same reference image
- the same inference settings
- the same later rendering and visual board layout

This keeps Phase 2G focused on whether the 500-step checkpoint can be loaded and
used, not on broad quality claims.

## First Test Case

The first Phase 2G asset is:

```text
asset_id: B075YLTF7Q
mesh: data/raw_assets/phase2b_abo_selected/B075YLTF7Q.glb
reference: data/hy3dpaint_train_examples/pilot_v1/B075YLTF7Q/render_cond/001_light_AL.png
checkpoint: checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt
```

## Non-Goals

- No final quality claim.
- No generalization claim.
- No long training.
- No direct Hunyuan inference until the loading interface is reviewed.

## Expected Stages

1. Inspect the official inference/checkpoint-loading interface read-only.
2. Prepare one local inference input case.
3. Run base inference later on A100.
4. Run fine-tuned inference later on A100.
5. Build a visual comparison board.
