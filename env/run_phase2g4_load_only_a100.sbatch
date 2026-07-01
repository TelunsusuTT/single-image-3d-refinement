#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2g4-load-only
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2g4_load_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CHECKPOINT="$PROJ/checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt"
OUT_ROOT="$PROJ/outputs/phase2g/load_only/pilot_v1_overfit_500"

mkdir -p "$PROJ/logs/slurm"
mkdir -p "$OUT_ROOT"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUT_ROOT=$OUT_ROOT"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2G.4 READINESS ====="
python "$PROJ/scripts/check_phase2g4_load_readiness.py" \
  --checkpoint "$CHECKPOINT" \
  --hypaint "$HYPAINT" \
  --output-dir "$OUT_ROOT"

echo "===== CUDA SANITY ====="
python3 - <<'PY'
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job")

a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
print("matmul sum:", float(c.sum().item()))
print("CUBLAS_MATMUL_OK")
PY

echo "===== CHECKPOINT LOAD-ONLY SMOKE ====="
python "$PROJ/scripts/load_phase2g4_finetuned_checkpoint_only.py" \
  --checkpoint "$CHECKPOINT" \
  --hypaint "$HYPAINT" \
  --out-json "$OUT_ROOT/load_only_report.json" \
  --out-md "$OUT_ROOT/load_only_report.md" \
  --max-num-view 6 \
  --resolution 512

echo "===== REPORTS ====="
ls -lh "$OUT_ROOT"/*.json "$OUT_ROOT"/*.md

echo "===== JOB END: SUCCESS ====="
date
