# Phase 2M M2 Failure 257295

## Job

- job id: `257295`
- phase: Phase 2M.2 ref_dino LoRA adapter-only training
- preset: `phase2m_refdino_r4_lr5e5_300`

## What Succeeded

The job passed the project-local readiness checks, loaded the original Hunyuan3D-Paint base, injected the 128 `ref_dino` LoRA targets with `local_linear_fallback`, froze non-LoRA parameters, and reached dataset setup.

## Failure

The job failed before the first training step with:

```text
AttributeError: 'HunyuanPaint' object has no attribute 'logdir'
```

The traceback pointed to the official `HunyuanPaint.on_fit_start()` lifecycle hook in:

```text
hy3dpaint/hunyuanpaintpbr/unet/model.py
```

The hook creates:

```text
os.path.join(self.logdir, "images_val")
```

The official `train.py` normally sets `model.logdir = logdir` before `trainer.fit(...)`. The project-local LoRA wrapper had not mirrored that attribute.

## Diagnosis

This is not a CUDA OOM, not a dataset packaging failure, and not a LoRA target selection failure. The model and dataset progressed far enough to show that the wrapper was missing an official training lifecycle attribute expected by the Lightning module.

## Patch Plan

The project-local wrapper now sets safe project-local lifecycle paths before `trainer.fit(...)`:

- `model.logdir`
- `model.ckptdir`
- `model.cfgdir`
- `model.codedir`
- `model.logdir/images_val`

These paths are under project `logs/train/phase2m/...` and are explicitly rejected if they resolve under `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work`. No official source files are edited.

## Next Gate

Run a one-step smoke before rerunning the 300-step job:

```bash
sbatch env/run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch
```

Only after the smoke reaches one training step and writes adapter-only outputs should the full 300-step M2 job be resubmitted.
