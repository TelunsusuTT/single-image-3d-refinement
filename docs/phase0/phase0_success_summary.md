# Phase 0 Success Summary

Date: 2026-06-28  
Job: hy3dpaint-smoke-254378  
Goal: Verify that official Hunyuan3D-Paint training can run safely on Imperial A100 from the project-local control directory.

## Result

Phase 0 succeeded.

The A100 smoke test passed:

- CONFIG_TARGET_OK
- MODULE_PREFLIGHT_OK
- HF_REPO_PREFLIGHT_OK
- DATASET_PREFLIGHT_OK
- CUBLAS_MATMUL_OK
- Training loop started
- max_steps=50 reached
- JOB END: SUCCESS

## Important setup decisions

- Hunyuan source root:
  /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1

- Hunyuan Paint root:
  /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint

- Project control root:
  /vol/bitbucket/ct1022/hy3dpaint_finetune

- Conda env:
  /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/env/conda_envs/hunyuan3d21_work

- Smoke config:
  configs/ft_overfit_smoke.yaml

- Slurm script:
  env/run_official_overfit_smoke_a100.sbatch

## Fixes applied during Phase 0

1. Corrected HY21/HYPAINT to point to the real source root.
2. Patched copied project config target:
   hunyuanpaintpbr.model.HunyuanPaint -> hunyuanpaintpbr.HunyuanPaint
3. Patched Stable Diffusion 2.1 repo:
   stabilityai/stable-diffusion-2-1 -> sd2-community/stable-diffusion-2-1
4. Created project-local absolute examples.json for official overfit data.
5. Added config/module/HF/dataset/CUDA preflight checks.
6. Created a 50-step smoke config with checkpoint saving disabled.
7. Set num_workers=1 because Hunyuan DataModule uses prefetch_factor=2.

## Interpretation

This does not produce a useful fine-tuned checkpoint. It proves that the official Hunyuan3D-Paint training code can run on A100 safely and reproducibly from the project-local setup.

Next phase: Phase 1, custom data preparation and format conversion.
