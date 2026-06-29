# Phase 0 Runbook

Run these steps from the project root:

```bash
cd /vol/bitbucket/ct1022/hy3dpaint_finetune
```

## 1. Create Local Environment Overrides

Copy the example file:

```bash
cp env/local.env.example env/local.env
```

Edit `env/local.env` and set the conda environment name:

```bash
export ENV_NAME=hunyuan21
```

If Hugging Face credentials are needed later, add them there rather than committing them.

## 2. Prepare The Official Overfit Config

Copy the upstream Hunyuan3D-Paint config into this project:

```bash
bash scripts/prepare_official_overfit_config.sh
```

The script creates:

```bash
configs/ft_overfit_official.yaml
```

Only edit the project-local config. Do not edit the upstream config in `Hunyuan3D2.1_Work`.

## 3. Run Local Sanity Checks

Run:

```bash
bash scripts/local_sanity.sh
```

This checks that the expected upstream files exist, activates the conda environment, imports `torch`, and prints the torch version and torch CUDA build version.

This local check does not require GPU availability.

## 4. Submit The A100 Smoke Test

Submit the Slurm job from the project root:

```bash
sbatch env/run_official_overfit_smoke_a100.sbatch
```

The job requests one A100 GPU, runs `nvidia-smi`, performs a small fp16 CUDA matrix multiplication, and then runs the official training command with a default 45 minute timeout.

To change the timeout:

```bash
sbatch --export=ALL,SMOKE_TIMEOUT=30m env/run_official_overfit_smoke_a100.sbatch
```

Timeout is treated as a successful smoke-test stop. Other training errors exit nonzero.

## 5. Inspect Logs

Slurm logs are written to:

```bash
logs/slurm/
```

Training logs are written under:

```bash
logs/train/
```

Look for:

- `CUBLAS_MATMUL_OK`
- the printed training command output
- timeout message, if the smoke test reaches the configured timeout
- Python stack traces or missing-file errors

## 6. Report Errors

When reporting an error, include:

- the command that was run
- the Slurm job id
- the relevant file from `logs/slurm/`
- the last 100 lines around the first error
- whether `scripts/local_sanity.sh` passed
