#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase2n-day5-a100-preflight
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

CUDA_SETUP=/vol/cuda/12.4.0/setup.sh
if [[ ! -f "$CUDA_SETUP" ]]; then
  echo "CUDA 12.4 setup is missing: $CUDA_SETUP" >&2
  exit 1
fi
source "$CUDA_SETUP"
export CUDA_HOME=/vol/cuda/12.4.0

export JOB_TMPDIR="/tmp/${USER}/phase2n_day5_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="$PROJ/configs/phase2n_day5_a100_preflight.json"

echo "===== ENVIRONMENT ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "CUDA_HOME=$CUDA_HOME"
echo "PYTHONPATH=$PYTHONPATH"
echo "TMPDIR=$TMPDIR"
echo "CONFIG=$CONFIG"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== GIT HEAD ====="
git -C "$PROJ" rev-parse HEAD

echo "===== DISK USAGE ====="
df -h "$PROJ" /tmp

echo "===== CHECK-ONLY READINESS ====="
python "$PROJ/scripts/phase2n_day5_a100_preflight.py" \
  --config "$CONFIG" \
  --check-only

echo "===== REAL-HUNYUAN ONE-BATCH AUDIT ====="
python "$PROJ/scripts/phase2n_day5_a100_preflight.py" \
  --config "$CONFIG" \
  --run-audit

echo "===== JOB END: SUCCESS ====="
date
