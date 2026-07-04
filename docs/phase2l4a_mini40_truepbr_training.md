# Phase 2L.4A Mini40 True-PBR Training Readiness

## Goal

Phase 2L.4A prepares a safe A100 training workflow for:

```text
datav2_frame_mini40_truepbr_500_lr1e6
```

This phase only creates configuration, preflight checks, an sbatch file, and a
checkpoint metadata inspector. It does not run Hunyuan, submit Slurm jobs,
train, load checkpoints, install packages, or modify official Hunyuan source.

## Dataset

Dataset: `datav2_frame_panels_mini40`

Expected split counts:

- train: 32
- validation: 4
- test: 4

Training uses:

```text
data/hy3dpaint_train_examples/datav2_frame_panels_mini40/examples_train_abs.json
```

Validation metadata is retained for evaluation planning, but this conservative
training config keeps `limit_val_batches: 0` and `num_sanity_val_steps: 0`.

## Why Mini40 First

Mini40 is the fast diagnostic split for the curated framed-panel Data v2 set.
It should prove that the new dataset can train from true-PBR initialization
before expanding to the full101 set. Do not run full101, full80, or parallel
larger training until mini40 train/eval succeeds.

## Hyperparameters

- steps: 500
- learning rate: `1e-6`
- batch size: 1
- num views: 6
- view size: 512
- initialization: local official Hunyuan3D-Paint PBR pipeline

True-PBR source:

```text
caches/hf/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1
```

## Outputs

Checkpoint directory:

```text
checkpoints/datav2_frame_mini40_truepbr_500_lr1e6
```

Expected checkpoint pattern:

```text
datav2_frame_mini40_truepbr_500_lr1e6-stepstep=500.ckpt
```

Lightning may format the `{step}` placeholder as `step=500`, so the exact file
name is confirmed after training by:

```text
scripts/inspect_datav2_frame_mini40_checkpoint.py
```

Slurm logs:

```text
logs/slurm/datav2_frame_mini40_truepbr_500_lr1e6-%j.out
logs/slurm/datav2_frame_mini40_truepbr_500_lr1e6-%j.err
```

## Success Tokens

Readiness:

```text
PHASE2L4A_MINI40_TRAINING_READINESS_OK
```

Training job:

```text
PHASE2L4A_MINI40_TRUEPBR_TRAIN_OK
===== JOB END: SUCCESS =====
```

Checkpoint inspection:

```text
PHASE2L4A_MINI40_CHECKPOINT_INSPECTION_OK
```

## What Not To Do

- do not run full101 yet
- do not run full80 in parallel
- do not use old wrong-init checkpoints
- do not reference `pilot_v1_overfit_500`
- do not reference `pilot_v1_conservative_50_lr1e6`
- do not reference `sd2-community`
