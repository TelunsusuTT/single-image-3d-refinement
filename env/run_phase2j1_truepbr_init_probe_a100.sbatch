#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2j1-init-probe
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

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2j1_init_probe_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CONFIG_PATH="$PROJ/configs/ft_pilot_v1_truepbr_init_probe.yaml"
PBR_DIR="$PROJ/caches/hf/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1"
OUT_ROOT="$PROJ/outputs/phase2j/true_pbr_initialization/init_probe"

mkdir -p "$PROJ/logs/slurm"
mkdir -p "$OUT_ROOT"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "PBR_DIR=$PBR_DIR"
echo "OUT_ROOT=$OUT_ROOT"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2J.1 TRUE-PBR INIT READINESS ====="
python "$PROJ/scripts/check_phase2j1_truepbr_init_readiness.py" \
  --hypaint "$HYPAINT" \
  --config "$CONFIG_PATH" \
  --pbr-dir "$PBR_DIR" \
  --output-dir "$OUT_ROOT"

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2J1_CUDA_PY'
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
PHASE2J1_CUDA_PY

echo "===== TRUE-PBR INIT EQUIVALENCE COMPARE ====="
python "$PROJ/scripts/compare_phase2j1_training_init_to_infer_base.py" \
  --hypaint "$HYPAINT" \
  --config "$CONFIG_PATH" \
  --out-json "$OUT_ROOT/truepbr_init_compare.json" \
  --out-md "$OUT_ROOT/truepbr_init_compare.md" \
  --max-sample-keys 100

echo "===== REPORTS ====="
echo "COMPARE_MD: $OUT_ROOT/truepbr_init_compare.md"
echo "COMPARE_JSON: $OUT_ROOT/truepbr_init_compare.json"
ls -lh "$OUT_ROOT/truepbr_init_compare.md" "$OUT_ROOT/truepbr_init_compare.json"

echo "===== JOB END: SUCCESS ====="
date
