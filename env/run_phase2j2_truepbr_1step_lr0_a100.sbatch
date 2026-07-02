#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2j2-1step-lr0
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2j2_1step_lr0_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CONFIG_PATH="$PROJ/configs/ft_pilot_v1_truepbr_1step_lr0.yaml"
EXAMPLES_JSON="$PROJ/data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json"
CHECKPOINT_ROOT="$PROJ/checkpoints/pilot_v1_truepbr_1step_lr0"
OUT_ROOT="$PROJ/outputs/phase2j/true_pbr_training_smoke/one_step_lr0"
LOGDIR="$PROJ/logs/train"
JOB_MARKER="$JOB_TMPDIR/job_start.marker"

mkdir -p "$PROJ/logs/slurm"
mkdir -p "$CHECKPOINT_ROOT"
mkdir -p "$OUT_ROOT"
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
echo "OUT_ROOT=$OUT_ROOT"
echo "LOGDIR=$LOGDIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2J.2 READINESS ====="
python "$PROJ/scripts/check_phase2j2_save_smoke_readiness.py" \
  --examples-json "$EXAMPLES_JSON" \
  --config "$CONFIG_PATH" \
  --hypaint "$HYPAINT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-dir "$OUT_ROOT"

echo "===== OFFICIAL-STYLE PACKAGING CHECK ====="
python "$PROJ/scripts/check_hy3dpaint_example.py" \
  --examples-json "$EXAMPLES_JSON" \
  --num-view 6 \
  --strict

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2J2_CUDA_PY'
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
PHASE2J2_CUDA_PY

TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-45m}"

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
  --name pilot_v1_truepbr_1step_lr0 \
  --logdir "$LOGDIR" \
  --gpus 1
train_status=$?
set -e

if [[ "$train_status" -eq 124 || "$train_status" -eq 137 ]]; then
  echo "true-PBR 1-step lr0 smoke timed out after $TRAIN_TIMEOUT." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

if [[ "$train_status" -ne 0 ]]; then
  echo "true-PBR 1-step lr0 smoke failed with exit code $train_status." >&2
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

echo "===== BASE VS SAVED CHECKPOINT AUDIT ====="
python "$PROJ/scripts/compare_phase2i_base_unet_to_checkpoints.py" \
  --hypaint "$HYPAINT" \
  --checkpoint "truepbr1step_lr0=$FOUND_CKPT" \
  --out-json "$OUT_ROOT/base_vs_truepbr1step_lr0_delta.json" \
  --out-md "$OUT_ROOT/base_vs_truepbr1step_lr0_delta.md" \
  --max-sample-keys 100 2>&1 | tee "$OUT_ROOT/base_vs_truepbr1step_lr0_delta_stdout.log"

if ! grep -q "PHASE2I_BASE_VS_CHECKPOINT_COMPARE_OK" "$OUT_ROOT/base_vs_truepbr1step_lr0_delta_stdout.log"; then
  echo "BASE_VS_CHECKPOINT_COMPARE_TOKEN_MISSING" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

python "$PROJ/scripts/check_phase2j2_saved_delta.py" \
  --report-json "$OUT_ROOT/base_vs_truepbr1step_lr0_delta.json" \
  --max-mean-abs-delta 1e-6

echo "===== REPORTS ====="
echo "CHECKPOINT: $FOUND_CKPT"
echo "DELTA_MD: $OUT_ROOT/base_vs_truepbr1step_lr0_delta.md"
echo "DELTA_JSON: $OUT_ROOT/base_vs_truepbr1step_lr0_delta.json"
ls -lh "$FOUND_CKPT" "$OUT_ROOT/base_vs_truepbr1step_lr0_delta.md" "$OUT_ROOT/base_vs_truepbr1step_lr0_delta.json"

echo "PHASE2J2_CHECKPOINT_SAVE_SMOKE_OK"
echo "===== JOB END: SUCCESS ====="
date
