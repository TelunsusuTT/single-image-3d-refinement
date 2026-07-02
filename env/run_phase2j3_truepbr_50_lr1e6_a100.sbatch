#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2j3-truepbr50
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2j3_truepbr50_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CONFIG_PATH="$PROJ/configs/ft_pilot_v1_truepbr_50_lr1e6.yaml"
EXAMPLES_JSON="$PROJ/data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json"
CHECKPOINT_ROOT="$PROJ/checkpoints/pilot_v1_truepbr_50_lr1e6"
LOGDIR="$PROJ/logs/train"
JOB_MARKER="$JOB_TMPDIR/job_start.marker"

mkdir -p "$PROJ/logs/slurm"
mkdir -p "$CHECKPOINT_ROOT"
touch "$JOB_MARKER"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "EXAMPLES_JSON=$EXAMPLES_JSON"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "LOGDIR=$LOGDIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2J.3 READINESS ====="
python "$PROJ/scripts/check_phase2j3_truepbr50_readiness.py" \
  --examples-json "$EXAMPLES_JSON" \
  --config "$CONFIG_PATH" \
  --hypaint "$HYPAINT" \
  --checkpoint-root "$CHECKPOINT_ROOT"

echo "===== OFFICIAL-STYLE PACKAGING CHECK ====="
python "$PROJ/scripts/check_hy3dpaint_example.py" \
  --examples-json "$EXAMPLES_JSON" \
  --num-view 6 \
  --strict

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2J3_CUDA_PY'
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
PHASE2J3_CUDA_PY

TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-90m}"

echo "===== TRAIN START ====="
echo "train starts"
echo "timeout: $TRAIN_TIMEOUT"
echo "HYPAINT: $HYPAINT"
echo "config: $CONFIG_PATH"
echo "logdir: $LOGDIR"

cd "$HYPAINT"

set +e
timeout "$TRAIN_TIMEOUT" python3 train.py \
  --base "$CONFIG_PATH" \
  --name pilot_v1_truepbr_50_lr1e6 \
  --logdir "$LOGDIR" \
  --gpus 1
train_status=$?
set -e

if [[ "$train_status" -eq 124 || "$train_status" -eq 137 ]]; then
  echo "true-PBR 50-step conservative run timed out after $TRAIN_TIMEOUT." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

if [[ "$train_status" -ne 0 ]]; then
  echo "true-PBR 50-step conservative run failed with exit code $train_status." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

echo "===== CHECKPOINT SEARCH ====="
checkpoint_count=0
FOUND_CKPT=""
while IFS= read -r -d '' ckpt; do
  size="$(du -h "$ckpt" | awk '{print $1}')"
  bytes="$(stat -c '%s' "$ckpt")"
  echo "CHECKPOINT: $ckpt size=$size bytes=$bytes"
  FOUND_CKPT="$ckpt"
  checkpoint_count=$((checkpoint_count + 1))
done < <(find "$CHECKPOINT_ROOT" -type f -name "*.ckpt" -newer "$JOB_MARKER" -print0)

if [[ "$checkpoint_count" -ne 1 ]]; then
  echo "EXPECTED_EXACTLY_ONE_NEW_CHECKPOINT, found $checkpoint_count" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "FOUND_CKPT=$FOUND_CKPT"
echo "PHASE2J3_TRUEPBR50_TRAIN_OK"
echo "===== JOB END: SUCCESS ====="
date
