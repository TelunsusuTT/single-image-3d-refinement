# Phase 2G.2 Base Inference Smoke

Phase 2G.2 prepares the first real base Hunyuan3D-Paint inference through the
project-local wrapper on one A100 job.

This differs from Phase 2G.1 dry-runs because the wrapper will actually import
and execute the official base inference pipeline. It still does not load the
Phase 2F checkpoint and does not run fine-tuned inference.

## Goal

Run one base inference on the prepared `B075YLTF7Q` case using:

```text
outputs/phase2g/infer_cases/B075YLTF7Q
```

Expected output directory:

```text
outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke/
```

## Non-Goals

- No checkpoint loading.
- No fine-tuned inference.
- No base-vs-fine-tuned comparison yet.
- No final quality claim.

## Expected Success

- `CUBLAS_MATMUL_OK` appears in the Slurm log.
- The wrapper starts in base mode.
- An output OBJ and/or GLB is created under the base smoke output directory.
- The job prints output file paths and sizes.
- `JOB END: SUCCESS` appears only after output validation succeeds.

A failure at this phase means the base wrapper pathing, official cache/model
availability, runtime environment, or official inference assumptions need to be
fixed before any fine-tuned checkpoint loading work begins.

## Run 255495 Failure Note

A first Phase 2G.2 A100 attempt, job `255495`, reached CUDA sanity, loaded the
official base model, and printed `Models Loaded.` It failed after model loading
inside the official default remesh path:

```text
Hunyuan3DPaintPipeline.__call__ -> remesh_mesh(...) -> import pymeshlab
```

The observed error was a `pymeshlab` import failure involving
`libQt5OpenGL.so.5` and `Qt_5_PRIVATE_API`. This is not a CUDA, data, or base
model-load failure. The Phase 2G fix is to use fixed-mesh inference with
`use_remesh=False`, exposed by the project wrapper as `--no-remesh`.

Skipping remesh is also the better experimental choice for the later
base-vs-fine-tuned comparison because geometry should remain fixed across both
runs.
