# Phase 2G.2 Run 255495 Failure

Job `255495` was the first A100 base inference smoke attempt for Phase 2G.2.

## What Passed

- Slurm job startup reached the project environment.
- Phase 2G.2 readiness passed.
- CUDA sanity passed and printed `CUBLAS_MATMUL_OK`.
- The official base Hunyuan3D-Paint model loaded successfully.
- The log printed `Models Loaded.`

## Failure Stage

The failure happened after model loading, inside official inference when the
wrapper called `Hunyuan3DPaintPipeline.__call__` with the official default
`use_remesh=True` behavior.

Call chain:

```text
scripts/run_phase2g_paint_infer.py
-> Hunyuan3DPaintPipeline.__call__
-> textureGenPipeline.py line 109
-> remesh_mesh(...)
-> import pymeshlab
-> Qt symbol error
```

The error involved `pymeshlab`, `libQt5OpenGL.so.5`, and `Qt_5_PRIVATE_API`.

## Interpretation

This was not a CUDA failure, data failure, or base model-load failure. The model
had already loaded, and CUDA matmul had passed. The failure was isolated to the
optional remeshing dependency path.

## Decision

Do not repair or reinstall `pymeshlab` for Phase 2G. Instead, bypass remeshing
by running fixed-mesh inference with `use_remesh=False`. This is also preferable
for the experiment because base and fine-tuned comparisons should keep geometry
fixed.

## Next Run

The next A100 smoke should use:

```text
outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke
```

and invoke the project wrapper with:

```text
--no-remesh
```
