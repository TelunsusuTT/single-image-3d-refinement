#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase2m_refdino_lora_smoke
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_refdino_lora_smoke-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_refdino_lora_smoke-%j.err

set -euo pipefail

PROJ=/vol/bitbucket/ct1022/hy3dpaint_finetune
source "$PROJ/env/env.sh"
conda activate "$ENV_NAME"

JOB_ID=${SLURM_JOB_ID:-manual}
JOB_TMPDIR=/tmp/${USER}/phase2m_refdino_lora_smoke_${JOB_ID}
mkdir -p "$JOB_TMPDIR" "$PROJ/logs/slurm" "$PROJ/logs/train/phase2m"
export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/src:${HYPAINT}:${HY21}:${PYTHONPATH:-}"

TRAIN_CONFIG="$PROJ/configs/ft_datav2_frame_full80_truepbr_500_lr1e6.yaml"
INVENTORY_JSON="$PROJ/outputs/phase2m/lora_module_inventory.json"
M1_SMOKE_DIR="$PROJ/outputs/phase2m/zero_lora_smoke"
OUT_DIR="$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300_smoke"
TRAIN_LOG_DIR="$PROJ/logs/train/phase2m"
SBATCH_PATH="$PROJ/env/run_phase2m_train_lora_refdino_r4_lr5e5_300_a100.sbatch"
SMOKE_SBATCH_PATH="$PROJ/env/run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch"
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-45m}

cd "$PROJ"

echo "===== PHASE 2M M2 REFDINO LORA SMOKE ====="
date
hostname
nvidia-smi

echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "TMPDIR=$TMPDIR"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "OUT_DIR=$OUT_DIR"
echo "TRAIN_LOG_DIR=$TRAIN_LOG_DIR"

echo "===== READINESS ====="
python "$PROJ/scripts/check_phase2m_lora_training_ready.py" \
  --inventory-json "$INVENTORY_JSON" \
  --smoke-dir "$M1_SMOKE_DIR" \
  --full80-config "$TRAIN_CONFIG" \
  --m2-output-dir "$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300" \
  --m2-checkpoint-dir "$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300" \
  --smoke-output-dir "$OUT_DIR" \
  --logdir "$TRAIN_LOG_DIR/phase2m_refdino_r4_lr5e5_300_smoke" \
  --sbatch "$SBATCH_PATH" \
  --smoke-sbatch "$SMOKE_SBATCH_PATH"

echo "===== SMOKE TRAIN START ====="
echo "one-step adapter-only smoke starts"
echo "timeout: $TRAIN_TIMEOUT"
cd "$HYPAINT"

timeout "$TRAIN_TIMEOUT" python "$PROJ/scripts/phase2m_train_lora_refdino.py" \
  --train-config "$TRAIN_CONFIG" \
  --inventory-json "$INVENTORY_JSON" \
  --m1-smoke-dir "$M1_SMOKE_DIR" \
  --trainer-log-dir "$TRAIN_LOG_DIR" \
  --backend local_linear_fallback \
  --rank 4 \
  --alpha 4 \
  --dropout 0.05 \
  --learning-rate 5e-5 \
  --device cuda \
  --trainer-mode official_ddp \
  --smoke

cd "$PROJ"

echo "===== SMOKE ADAPTER OUTPUT CHECK ====="
expected=(
  "$OUT_DIR/adapter_step_000001.pt"
  "$OUT_DIR/adapter_config.json"
  "$OUT_DIR/training_summary.json"
)
for path in "${expected[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "missing or empty expected smoke adapter output: $path" >&2
    echo "===== JOB END: FAILED ====="
    date
    exit 1
  fi
  bytes=$(stat -c '%s' "$path")
  echo "SMOKE_ADAPTER_OUTPUT: $path bytes=$bytes"
done

if [[ -e "$OUT_DIR/adapter_final.pt" ]]; then
  echo "unexpected smoke adapter_final.pt: $OUT_DIR/adapter_final.pt" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

if find "$OUT_DIR" -type f -name '*.ckpt' | grep -q .; then
  echo "unexpected full checkpoint-like .ckpt file under $OUT_DIR" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "PHASE2M_M2_REFDINO_LORA_SMOKE_OK"
echo "===== JOB END: SUCCESS ====="
date
