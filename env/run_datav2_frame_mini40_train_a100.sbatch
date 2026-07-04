#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=datav2_frame_mini40_truepbr_500_lr1e6
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/datav2_frame_mini40_truepbr_500_lr1e6-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/datav2_frame_mini40_truepbr_500_lr1e6-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/datav2_frame_mini40_truepbr_500_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

TRAIN_CONFIG_JSON="$PROJ/configs/datav2_frame_mini40_train.json"
TRAIN_YAML="$PROJ/configs/ft_datav2_frame_mini40_truepbr_500_lr1e6.yaml"
TRAIN_EXAMPLES_JSON="$PROJ/data/hy3dpaint_train_examples/datav2_frame_panels_mini40/examples_train_abs.json"
VAL_EXAMPLES_JSON="$PROJ/data/hy3dpaint_train_examples/datav2_frame_panels_mini40/examples_val_abs.json"
TEST_EXAMPLES_JSON="$PROJ/data/hy3dpaint_train_examples/datav2_frame_panels_mini40/examples_test_abs.json"
CHECKPOINT_ROOT="$PROJ/checkpoints/datav2_frame_mini40_truepbr_500_lr1e6"
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
echo "TRAIN_CONFIG_JSON=$TRAIN_CONFIG_JSON"
echo "TRAIN_YAML=$TRAIN_YAML"
echo "TRAIN_EXAMPLES_JSON=$TRAIN_EXAMPLES_JSON"
echo "VAL_EXAMPLES_JSON=$VAL_EXAMPLES_JSON"
echo "TEST_EXAMPLES_JSON=$TEST_EXAMPLES_JSON"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "LOGDIR=$LOGDIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2L.4A READINESS ====="
python "$PROJ/scripts/check_datav2_frame_mini40_training_readiness.py" \
  --config "$TRAIN_CONFIG_JSON"

echo "===== OFFICIAL-STYLE PACKAGING CHECKS ====="
python "$PROJ/scripts/check_hy3dpaint_example.py" \
  --examples-json "$TRAIN_EXAMPLES_JSON" \
  --num-view 6 \
  --strict
python "$PROJ/scripts/check_hy3dpaint_example.py" \
  --examples-json "$VAL_EXAMPLES_JSON" \
  --num-view 6 \
  --strict
python "$PROJ/scripts/check_hy3dpaint_example.py" \
  --examples-json "$TEST_EXAMPLES_JSON" \
  --num-view 6 \
  --strict

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2L4A_CUDA_PY'
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
PHASE2L4A_CUDA_PY

TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-4h}"

echo "===== TRAIN START ====="
echo "train starts"
echo "timeout: $TRAIN_TIMEOUT"
echo "HYPAINT: $HYPAINT"
echo "config: $TRAIN_YAML"
echo "logdir: $LOGDIR"

cd "$HYPAINT"

set +e
timeout "$TRAIN_TIMEOUT" python3 train.py \
  --base "$TRAIN_YAML" \
  --name datav2_frame_mini40_truepbr_500_lr1e6 \
  --logdir "$LOGDIR" \
  --gpus 1
train_status=$?
set -e

if [[ "$train_status" -eq 124 || "$train_status" -eq 137 ]]; then
  echo "mini40 true-PBR 500-step run timed out after $TRAIN_TIMEOUT." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

if [[ "$train_status" -ne 0 ]]; then
  echo "mini40 true-PBR 500-step run failed with exit code $train_status." >&2
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

echo "===== CHECKPOINT INSPECTION ====="
python "$PROJ/scripts/inspect_datav2_frame_mini40_checkpoint.py" \
  --config "$TRAIN_CONFIG_JSON"

echo "PHASE2L4A_MINI40_TRUEPBR_TRAIN_OK"
echo "===== JOB END: SUCCESS ====="
date
