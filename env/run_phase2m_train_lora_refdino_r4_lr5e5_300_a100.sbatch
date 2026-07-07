#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase2m_refdino_r4_lr5e5_300
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_refdino_r4_lr5e5_300-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_refdino_r4_lr5e5_300-%j.err

set -euo pipefail

PROJ=/vol/bitbucket/ct1022/hy3dpaint_finetune
source "$PROJ/env/env.sh"
conda activate "$ENV_NAME"

JOB_ID=${SLURM_JOB_ID:-manual}
JOB_TMPDIR=/tmp/${USER}/phase2m_refdino_lora_${JOB_ID}
mkdir -p "$JOB_TMPDIR" "$PROJ/logs/slurm" "$PROJ/logs/train/phase2m"
export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/src:${HYPAINT}:${HY21}:${PYTHONPATH:-}"

TRAIN_CONFIG="$PROJ/configs/ft_datav2_frame_full80_truepbr_500_lr1e6.yaml"
INVENTORY_JSON="$PROJ/outputs/phase2m/lora_module_inventory.json"
M1_SMOKE_DIR="$PROJ/outputs/phase2m/zero_lora_smoke"
OUT_DIR="$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300"
TRAIN_LOG_DIR="$PROJ/logs/train/phase2m"
SBATCH_PATH="$PROJ/env/run_phase2m_train_lora_refdino_r4_lr5e5_300_a100.sbatch"
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-6h}

cd "$PROJ"

echo "===== PHASE 2M M2 REFDINO LORA TRAIN ====="
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
  --m2-output-dir "$OUT_DIR" \
  --m2-checkpoint-dir "$OUT_DIR" \
  --sbatch "$SBATCH_PATH"

echo "===== TRAIN START ====="
echo "adapter-only training starts"
echo "timeout: $TRAIN_TIMEOUT"
cd "$HYPAINT"

timeout "$TRAIN_TIMEOUT" python "$PROJ/scripts/phase2m_train_lora_refdino.py" \
  --train-config "$TRAIN_CONFIG" \
  --inventory-json "$INVENTORY_JSON" \
  --m1-smoke-dir "$M1_SMOKE_DIR" \
  --out-dir "$OUT_DIR" \
  --trainer-log-dir "$TRAIN_LOG_DIR" \
  --backend local_linear_fallback \
  --rank 4 \
  --alpha 4 \
  --dropout 0.05 \
  --learning-rate 5e-5 \
  --max-train-steps 300 \
  --save-every 100 \
  --device cuda \
  --trainer-mode official_ddp

cd "$PROJ"

echo "===== ADAPTER OUTPUT CHECK ====="
expected=(
  "$OUT_DIR/adapter_step_000100.pt"
  "$OUT_DIR/adapter_step_000200.pt"
  "$OUT_DIR/adapter_step_000300.pt"
  "$OUT_DIR/adapter_final.pt"
  "$OUT_DIR/adapter_config.json"
  "$OUT_DIR/training_summary.json"
)
for path in "${expected[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "missing or empty expected adapter output: $path" >&2
    echo "===== JOB END: FAILED ====="
    date
    exit 1
  fi
  bytes=$(stat -c '%s' "$path")
  echo "ADAPTER_OUTPUT: $path bytes=$bytes"
done

if find "$OUT_DIR" -type f -name '*.ckpt' | grep -q .; then
  echo "unexpected full checkpoint-like .ckpt file under $OUT_DIR" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "PHASE2M_M2_REFDINO_LORA_TRAIN_OK"
echo "===== JOB END: SUCCESS ====="
date
