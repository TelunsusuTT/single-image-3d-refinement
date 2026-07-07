# Phase 2M M2 Failure 257331

## Context

- job id: `257331`
- phase: Phase 2M.2A one-step `ref_dino` LoRA smoke
- preset: `phase2m_refdino_r4_lr5e5_300`
- backend: `local_linear_fallback`
- intended checkpoint policy: adapter-only outputs under `outputs/phase2m/`

## What Passed

The job passed the M2 readiness checker and progressed beyond the previous `model.logdir` failure. The log showed:

- project-local `model_logdir` was set correctly
- the official Hunyuan3D-Paint base model loaded
- all 128 M0 `ref_dino` LoRA targets were injected
- non-LoRA parameters were frozen
- `trainable_parameter_count = 829,952`
- `optimizer_parameter_count = 829,952`
- train dataset length was 80
- validation dataset length was 10

This confirms that the Phase 2M.2A lifecycle-path patch worked.

## Failure

The job failed before the first training step with:

```text
ValueError: Default process group has not been initialized
```

The failure came from the official DataModule path:

```text
src/data/objaverse_hunyuan.py
train_dataloader()
sampler = DistributedSampler(datasets)
```

The official DataModule constructs a distributed sampler without explicit `num_replicas` and `rank`, so PyTorch asks the default distributed process group for world size/rank. In the project-local wrapper's previous single-device Trainer path, that process group had not been initialized.

## Diagnosis

This is not an OOM, not a LoRA target issue, not a dataset packaging issue, and not a model-load failure. It is a trainer/DataModule interface mismatch: the official training path expects DDP initialization before the official DataModule builds its distributed sampler.

Read-only inspection of official `train.py` showed that official training uses:

```python
from pytorch_lightning.strategies import DDPStrategy
trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
trainer = Trainer(..., num_nodes=opt.num_nodes, inference_mode=False)
trainer.logdir = logdir
model.logdir = logdir
```

## Patch Plan

The project-local M2 wrapper now supports two trainer modes:

- `official_ddp`, the default, mirrors official training with `DDPStrategy(find_unused_parameters=False)`, `num_nodes=1`, `devices=1`, and `inference_mode=False`.
- `single_rank_data`, an explicit fallback that reuses the same dataset configs but creates single-rank DataLoaders without implicit distributed sampling.

The default smoke sbatch uses:

```bash
--trainer-mode official_ddp
```

A separate fallback smoke sbatch is available only if the official-DDP smoke still fails:

```bash
--trainer-mode single_rank_data
```

Both paths keep adapter-only checkpointing, project-local log directories, no full Lightning checkpoints, and no writes to `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work`.

## Next Gate

Run the official-DDP one-step smoke first:

```bash
sbatch env/run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch
```

Only if that still fails because of distributed setup should the fallback be tried:

```bash
sbatch env/run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_single_rank_a100.sbatch
```

Do not rerun the full 300-step M2 job until a one-step smoke reaches the first training step and writes adapter-only smoke outputs.
