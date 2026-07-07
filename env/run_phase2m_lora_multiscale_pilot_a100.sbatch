#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase2m_lora_multiscale_pilot
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_lora_multiscale_pilot-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_lora_multiscale_pilot-%j.err

set -euo pipefail

PROJ=/vol/bitbucket/ct1022/hy3dpaint_finetune
source "$PROJ/env/env.sh"
conda activate "$ENV_NAME"

JOB_ID=${SLURM_JOB_ID:-manual}
JOB_TMPDIR=/tmp/${USER}/phase2m_lora_multiscale_pilot_${JOB_ID}
mkdir -p "$JOB_TMPDIR" "$PROJ/logs/slurm"
export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/src:${HYPAINT}:${HY21}:${PYTHONPATH:-}"

ADAPTER_PATH="$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_final.pt"
ADAPTER_CONFIG="$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_config.json"
EVAL_CASES="$PROJ/outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/eval_cases.json"
M3A_SUMMARY="$PROJ/outputs/phase2m/lora_infer_smoke_scale075/smoke_summary.json"
OUTPUT_DIR="$PROJ/outputs/phase2m/lora_multiscale_pilot"
SBATCH_PATH="$PROJ/env/run_phase2m_lora_multiscale_pilot_a100.sbatch"

cd "$PROJ"

echo "===== PHASE 2M.3B LORA MULTISCALE PILOT ====="
date
hostname
nvidia-smi

echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "TMPDIR=$TMPDIR"
echo "ADAPTER_PATH=$ADAPTER_PATH"
echo "EVAL_CASES=$EVAL_CASES"
echo "OUTPUT_DIR=$OUTPUT_DIR"

echo "===== READINESS ====="
python "$PROJ/scripts/check_phase2m_lora_multiscale_pilot_ready.py" \
  --adapter-path "$ADAPTER_PATH" \
  --adapter-config "$ADAPTER_CONFIG" \
  --eval-cases-json "$EVAL_CASES" \
  --m3a-summary "$M3A_SUMMARY" \
  --output-dir "$OUTPUT_DIR" \
  --sbatch "$SBATCH_PATH" \
  --scales 0.5 0.75 1.0

echo "===== MULTISCALE PILOT START ====="
python "$PROJ/scripts/phase2m_lora_multiscale_pilot.py" \
  --eval-cases-json "$EVAL_CASES" \
  --adapter-path "$ADAPTER_PATH" \
  --adapter-config "$ADAPTER_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --scales 0.5 0.75 1.0 \
  --max-num-view 6 \
  --resolution 512 \
  --device cuda

echo "===== OUTPUT CHECK ====="
for path in "$OUTPUT_DIR/pilot_summary.json" "$OUTPUT_DIR/cases_used.json" "$OUTPUT_DIR/per_variant_outputs.json"; do
  if [[ ! -s "$path" ]]; then
    echo "missing or empty pilot output: $path" >&2
    echo "===== JOB END: FAILED ====="
    date
    exit 1
  fi
  bytes=$(stat -c '%s' "$path")
  echo "PILOT_OUTPUT: $path bytes=$bytes"
done

python - "$OUTPUT_DIR/pilot_summary.json" <<'PHASE2M_M3B_SUMMARY_CHECK'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "OK" or summary.get("success") is not True:
    raise SystemExit(f"pilot summary not OK: {summary.get('status')} success={summary.get('success')}")
if summary.get("scales") != [0.5, 0.75, 1.0]:
    raise SystemExit(f"unexpected scales: {summary.get('scales')}")
if summary.get("case_count") != 3:
    raise SystemExit(f"unexpected case_count: {summary.get('case_count')}")
print("PHASE2M_M3B_SUMMARY_CHECK_OK")
PHASE2M_M3B_SUMMARY_CHECK

echo "PHASE2M_M3B_LORA_MULTISCALE_PILOT_JOB_OK"
echo "===== JOB END: SUCCESS ====="
date
