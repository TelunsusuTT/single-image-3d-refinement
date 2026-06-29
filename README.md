# Small-Data Partial Fine-Tuning of Hunyuan3D-Paint for Texture-Heavy Image-to-3D Assets

This project is a Phase 0 scaffold for safely running the official Hunyuan3D-Paint training smoke test from outside the upstream Hunyuan3D 2.1 source tree.

The upstream source tree is expected at:

```bash
/vol/bitbucket/ct1022/Hunyuan3D2.1_Work
```

This project keeps local configs, logs, checkpoints, data, caches, and temporary files under:

```bash
/vol/bitbucket/ct1022/hy3dpaint_finetune
```

Phase 0 is intentionally small. It does not add Python packages, custom modules, dataset loaders, LoRA code, or full fine-tuning code. Its goal is to prepare a safe external working area for the official overfit smoke test.

## Main Entry Points

- `env/env.sh`: shared environment setup for local checks and Slurm jobs.
- `env/local.env.example`: template for local environment overrides.
- `scripts/prepare_official_overfit_config.sh`: copies the official upstream paint config into this project.
- `scripts/local_sanity.sh`: checks expected upstream paths and imports `torch`.
- `env/run_official_overfit_smoke_a100.sbatch`: Slurm A100 smoke-test launcher.

## Phase 0 Flow

1. Create `env/local.env` from `env/local.env.example`.
2. Set `ENV_NAME` to the intended conda environment.
3. Run `scripts/prepare_official_overfit_config.sh`.
4. Run `scripts/local_sanity.sh`.
5. Submit `env/run_official_overfit_smoke_a100.sbatch` from the project root.
6. Inspect logs under `logs/slurm/` and `logs/train/`.

See `docs/phase0_runbook.md` for the step-by-step runbook.
