# Phase 2E Multi-Asset Training Smoke

Phase 2E creates an A100 smoke setup for the 7-sample `pilot_v1`
Hunyuan3D-Paint-style dataset.

The goal is to check whether the official Hunyuan3D-Paint `train.py` can read
the multi-asset `pilot_v1` dataset and complete a short training-loop smoke run.
This is not formal fine-tuning.

## Inputs

```text
data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json
data/candidates/phase2c_pilot_v1_render_manifest.csv
outputs/boards/phase2c_pilot_v1_dataset_check.csv
```

## Expected Success Signals

Look for these in the Slurm logs:

- `CONFIG_TARGET_OK`
- `DATASET_PREFLIGHT_OK`
- official strict checker reports `Checked 7 sample(s): 7 OK, 0 failed`
- `CUBLAS_MATMUL_OK`
- dataset length = 7
- max steps reached
- `JOB END: SUCCESS`

## Non-Goals

This smoke run is not expected to provide:

- a useful checkpoint
- a visual quality conclusion
- a generalization claim
- a comparison against other datasets or models

If this smoke fails, debug dataset path wiring, package structure, dataloader
assumptions, or CUDA/runtime setup before interpreting anything about model
quality.
