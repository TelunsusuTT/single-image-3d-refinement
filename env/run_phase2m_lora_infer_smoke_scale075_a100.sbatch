#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase2m_lora_infer_smoke075
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_lora_infer_smoke075-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_lora_infer_smoke075-%j.err

set -euo pipefail

PROJ=/vol/bitbucket/ct1022/hy3dpaint_finetune
source "$PROJ/env/env.sh"
conda activate "$ENV_NAME"

JOB_ID=${SLURM_JOB_ID:-manual}
JOB_TMPDIR=/tmp/${USER}/phase2m_lora_infer_smoke075_${JOB_ID}
mkdir -p "$JOB_TMPDIR" "$PROJ/logs/slurm"
export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/src:${HYPAINT}:${HY21}:${PYTHONPATH:-}"

ADAPTER_PATH="$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_final.pt"
ADAPTER_CONFIG="$PROJ/outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_config.json"
EVAL_CASES="$PROJ/outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/eval_cases.json"
OUTPUT_DIR="$PROJ/outputs/phase2m/lora_infer_smoke_scale075"
SBATCH_PATH="$PROJ/env/run_phase2m_lora_infer_smoke_scale075_a100.sbatch"
LORA_SCALE=0.75

cd "$PROJ"

echo "===== PHASE 2M.3A LORA INFER SMOKE SCALE 0.75 ====="
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
python "$PROJ/scripts/check_phase2m_lora_infer_smoke_ready.py"   --adapter-path "$ADAPTER_PATH"   --adapter-config "$ADAPTER_CONFIG"   --eval-cases-json "$EVAL_CASES"   --output-dir "$OUTPUT_DIR"   --sbatch "$SBATCH_PATH"

echo "===== LORA INFERENCE SMOKE START ====="
python "$PROJ/scripts/phase2m_lora_infer_smoke.py"   --eval-cases-json "$EVAL_CASES"   --adapter-path "$ADAPTER_PATH"   --adapter-config "$ADAPTER_CONFIG"   --output-dir "$OUTPUT_DIR"   --lora-scale 0.75   --max-num-view 6   --resolution 512   --device cuda

echo "===== OUTPUT CHECK ====="
SUMMARY="$OUTPUT_DIR/smoke_summary.json"
if [[ ! -s "$SUMMARY" ]]; then
  echo "missing smoke summary: $SUMMARY" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

python - "$SUMMARY" <<'PHASE2M_M3A_SUMMARY_CHECK'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "OK" or summary.get("success") is not True:
    raise SystemExit(f"smoke summary not OK: {summary.get('status')} success={summary.get('success')}")
for key in ("base_output_glb_path", "lora_output_glb_path"):
    path = Path(summary[key])
    alt_obj = path.with_suffix(".obj")
    if not ((path.is_file() and path.stat().st_size > 0) or (alt_obj.is_file() and alt_obj.stat().st_size > 0)):
        raise SystemExit(f"missing output for {key}: {path} or {alt_obj}")
    print(f"{key}: {path}")
print("PHASE2M_M3A_SUMMARY_CHECK_OK")
PHASE2M_M3A_SUMMARY_CHECK

echo "PHASE2M_M3A_LORA_INFER_SMOKE_JOB_OK"
echo "===== JOB END: SUCCESS ====="
date
