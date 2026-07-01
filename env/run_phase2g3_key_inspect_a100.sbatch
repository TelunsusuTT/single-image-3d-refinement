#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2g3-key-inspect
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

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2g3_keys_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CHECKPOINT="$PROJ/checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt"
OUT_ROOT="$PROJ/outputs/phase2g/key_inspection/pilot_v1_overfit_500"

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

echo "===== PHASE 2G.3 READINESS ====="
python "$PROJ/scripts/check_phase2g3_key_inspect_readiness.py" \
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

echo "===== CHECKPOINT KEY INSPECTION ====="
python "$PROJ/scripts/inspect_phase2g3_checkpoint_keys.py" \
  --checkpoint "$CHECKPOINT" \
  --out-json "$OUT_ROOT/checkpoint_keys.json" \
  --out-md "$OUT_ROOT/checkpoint_keys.md" \
  --max-sample-keys 200

echo "===== INFERENCE UNET KEY INSPECTION ====="
python "$PROJ/scripts/inspect_phase2g3_infer_unet_keys.py" \
  --hypaint "$HYPAINT" \
  --out-json "$OUT_ROOT/infer_unet_keys.json" \
  --out-md "$OUT_ROOT/infer_unet_keys.md" \
  --max-sample-keys 200 \
  --max-num-view 6 \
  --resolution 512

echo "===== KEYSPACE COMPARE ====="
python "$PROJ/scripts/compare_phase2g3_keyspaces.py" \
  --checkpoint-json "$OUT_ROOT/checkpoint_keys.json" \
  --infer-json "$OUT_ROOT/infer_unet_keys.json" \
  --out-json "$OUT_ROOT/keyspace_compare.json" \
  --out-md "$OUT_ROOT/keyspace_compare.md"

echo "===== REPORTS ====="
ls -lh "$OUT_ROOT"/*.json "$OUT_ROOT"/*.md

echo "===== JOB END: SUCCESS ====="
date
