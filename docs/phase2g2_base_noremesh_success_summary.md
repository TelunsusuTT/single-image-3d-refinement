# Phase 2G.2 Base No-Remesh Inference Success Summary

## Goal

Phase 2G.2 tested whether the project-local wrapper can run real base Hunyuan3D-Paint inference on the prepared B075YLTF7Q case.

This phase did not load the Phase 2F fine-tuned checkpoint.

## Result

Phase 2G.2 succeeded with no-remesh inference.

Job id: 255509

Key success signals:
- PHASE2G2_BASE_INFER_PREFLIGHT_OK
- CUBLAS_MATMUL_OK
- mode: base
- use_remesh: False
- Models Loaded.
- output OBJ created
- output GLB created
- JOB END: SUCCESS

## Output Directory

outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke/

Expected main outputs:
- base_textured_mesh.obj
- base_textured_mesh.glb
- base_textured_mesh.jpg
- base_textured_mesh_metallic.jpg
- base_textured_mesh_roughness.jpg

## Interpretation

This confirms that the project-local wrapper can call the official Hunyuan3D-Paint base inference pipeline on our prepared case.

The previous pymeshlab / Qt failure was caused by the default remesh path. The no-remesh fix bypasses that path and is aligned with the project requirement of fixed-mesh base-vs-fine-tuned comparison.

## Non-Goals

This result does not yet prove:
- fine-tuned checkpoint loading
- fine-tuned inference
- base-vs-fine-tuned visual difference
- final visual improvement
- generalization
