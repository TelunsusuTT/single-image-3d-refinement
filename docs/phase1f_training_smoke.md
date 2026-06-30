# Phase 1F Custom ABO One-Asset Training Smoke

Phase 1F creates a minimal A100 smoke setup for the custom ABO one-asset
Hunyuan3D-Paint-style training example:

```text
data/hy3dpaint_train_examples/abo_one_asset/B07H469871
```

The goal is to check whether the official Hunyuan3D-Paint `train.py` can read
the custom example and enter the training loop. This is a dataloader and
training-loop smoke test, not formal fine-tuning.

## Absolute Manifest

The Slurm job runs `train.py` from the official HYPAINT directory, not from this
project root. To avoid relative-path ambiguity, Phase 1F uses an absolute-path
examples JSON:

```text
data/hy3dpaint_train_examples/abo_one_asset/examples_train_abs.json
```

That file should contain exactly one entry: the absolute path to the custom
sample directory.

## Expected Success Signals

Look for these in the Slurm logs:

- `CONFIG_TARGET_OK`
- `DATASET_PREFLIGHT_OK`
- `CUBLAS_MATMUL_OK`
- train starts
- max steps are reached
- `JOB END: SUCCESS`

## Non-Goals

This smoke test is not expected to produce:

- a useful checkpoint
- visual quality evaluation
- model comparison
- evidence that the synthetic supervision is high quality

Checkpoint saving remains effectively disabled for the smoke run.

## Likely Failure Causes

If Phase 1F fails, likely causes include:

- relative or ambiguous dataset paths
- `transforms.json` format mismatch
- image channel mismatch
- normal or position encoding mismatch
- assumptions inside the official dataloader that the custom packaging does not
  yet satisfy

Fix packaging and dataloader compatibility first. Do not interpret a failed
Phase 1F smoke as a model quality result.
