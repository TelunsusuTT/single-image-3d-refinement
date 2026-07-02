#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2i-init-audit
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2i_init_audit_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

OUT_ROOT="$PROJ/outputs/phase2i/pretrained_initialization_audit"
CONFIG_50="$PROJ/configs/ft_pilot_v1_conservative_50_lr1e6.yaml"
CONFIG_500="$PROJ/configs/ft_pilot_v1_overfit_500.yaml"
CKPT_50="$PROJ/checkpoints/pilot_v1_conservative_50_lr1e6/pilot_v1_conservative_50_lr1e6-stepstep=50.ckpt"
CKPT_500="$PROJ/checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt"

mkdir -p "$PROJ/logs/slurm"
mkdir -p "$OUT_ROOT"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "OUT_ROOT=$OUT_ROOT"
echo "CONFIG_50=$CONFIG_50"
echo "CONFIG_500=$CONFIG_500"
echo "CKPT_50=$CKPT_50"
echo "CKPT_500=$CKPT_500"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2I READINESS ====="
python "$PROJ/scripts/check_phase2i_audit_readiness.py" \
  --hypaint "$HYPAINT" \
  --config "$CONFIG_50" "$CONFIG_500" \
  --checkpoint "$CKPT_50" "$CKPT_500" \
  --output-dir "$OUT_ROOT"

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2I_CUDA_PY'
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
PHASE2I_CUDA_PY

echo "===== CONFIG INITIALIZATION INSPECTION ====="
python "$PROJ/scripts/inspect_phase2i_training_initialization.py" \
  --configs "$CONFIG_50" "$CONFIG_500" \
  --out-json "$OUT_ROOT/config_initialization_report.json" \
  --out-md "$OUT_ROOT/config_initialization_report.md"

echo "===== BASE UNET VS CHECKPOINTS ====="
python "$PROJ/scripts/compare_phase2i_base_unet_to_checkpoints.py" \
  --hypaint "$HYPAINT" \
  --checkpoint "conservative50=$CKPT_50" "overfit500=$CKPT_500" \
  --out-json "$OUT_ROOT/base_vs_checkpoint_weight_deltas.json" \
  --out-md "$OUT_ROOT/base_vs_checkpoint_weight_deltas.md" \
  --max-sample-keys 100

echo "===== REPORTS ====="
echo "CONFIG_REPORT_MD: $OUT_ROOT/config_initialization_report.md"
echo "CONFIG_REPORT_JSON: $OUT_ROOT/config_initialization_report.json"
echo "WEIGHT_DELTA_MD: $OUT_ROOT/base_vs_checkpoint_weight_deltas.md"
echo "WEIGHT_DELTA_JSON: $OUT_ROOT/base_vs_checkpoint_weight_deltas.json"
ls -lh \
  "$OUT_ROOT/config_initialization_report.md" \
  "$OUT_ROOT/config_initialization_report.json" \
  "$OUT_ROOT/base_vs_checkpoint_weight_deltas.md" \
  "$OUT_ROOT/base_vs_checkpoint_weight_deltas.json"

echo "===== JOB END: SUCCESS ====="
date
