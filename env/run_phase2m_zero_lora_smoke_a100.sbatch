#!/bin/bash
#SBATCH --job-name=hy3dpaint-phase2m-zero-lora-smoke
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -euo pipefail

PROJ=/vol/bitbucket/ct1022/hy3dpaint_finetune
source "$PROJ/env/env.sh"
conda activate "$ENV_NAME"

JOB_ID=${SLURM_JOB_ID:-manual}
JOB_TMPDIR=/tmp/${USER}/phase2m_zero_lora_smoke_${JOB_ID}
mkdir -p "$JOB_TMPDIR" "$PROJ/outputs/phase2m/zero_lora_smoke"
export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/src:${HYPAINT}:${HY21}:${PYTHONPATH:-}"

cd "$PROJ"

echo "===== PHASE 2M ZERO LORA SMOKE ====="
date
hostname
nvidia-smi

echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "TMPDIR=$TMPDIR"

python "$PROJ/scripts/phase2m_zero_lora_smoke.py" \
  --out-dir "$PROJ/outputs/phase2m/zero_lora_smoke" \
  --backend auto \
  --rank 4 \
  --alpha 4 \
  --dropout 0.0 \
  --preset ref_dino \
  --max-num-view 6 \
  --resolution 512 \
  --device cuda

echo "===== JOB END: SUCCESS ====="
