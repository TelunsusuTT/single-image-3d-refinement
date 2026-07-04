#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2k3-render-eval
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
pwd

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2k3_render_eval_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CASES_CONFIG="$PROJ/configs/phase2k3_rendered_view_eval_cases.json"
OUTPUT_ROOT="$PROJ/outputs/phase2k/rendered_view_eval_truepbr200"

if [[ -z "${BLENDER_BIN:-}" ]]; then
  BLENDER_BIN="$(command -v blender || true)"
fi
if [[ -z "${BLENDER_BIN:-}" && -x /vol/bitbucket/ct1022/tools/bin/blender ]]; then
  BLENDER_BIN="/vol/bitbucket/ct1022/tools/bin/blender"
fi
export BLENDER_BIN

mkdir -p "$PROJ/logs/slurm"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CASES_CONFIG=$CASES_CONFIG"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "BLENDER_BIN=$BLENDER_BIN"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2K.3 READINESS ====="
python "$PROJ/scripts/check_phase2k3_render_eval_readiness.py" \
  --cases-config "$CASES_CONFIG" \
  --output-root "$OUTPUT_ROOT"

echo "===== BLENDER RENDER ====="
"$BLENDER_BIN" -b --python "$PROJ/scripts/render_phase2k3_glb_views_blender.py" -- \
  --cases-config "$CASES_CONFIG" \
  --output-root "$OUTPUT_ROOT" 2>&1 | tee "$OUTPUT_ROOT/render_stdout.log"

if ! grep -q "PHASE2K3_RENDERED_VIEWS_OK" "$OUTPUT_ROOT/render_stdout.log"; then
  echo "RENDER_STDOUT_MISSING_OK_TOKEN: $OUTPUT_ROOT/render_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== RENDERED VIEW COMPARISON ====="
python "$PROJ/scripts/compare_phase2k3_rendered_views.py" \
  --cases-config "$CASES_CONFIG" \
  --output-root "$OUTPUT_ROOT" 2>&1 | tee "$OUTPUT_ROOT/compare_stdout.log"

if ! grep -q "PHASE2K3_RENDERED_VIEW_COMPARE_OK" "$OUTPUT_ROOT/compare_stdout.log"; then
  echo "COMPARE_STDOUT_MISSING_OK_TOKEN: $OUTPUT_ROOT/compare_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== RENDERED METRICS AGGREGATION ====="
python "$PROJ/scripts/aggregate_phase2k3_rendered_metrics.py" \
  --cases-config "$CASES_CONFIG" \
  --output-root "$OUTPUT_ROOT" 2>&1 | tee "$OUTPUT_ROOT/aggregate_stdout.log"

if ! grep -q "PHASE2K3_RENDERED_METRICS_AGGREGATED_OK" "$OUTPUT_ROOT/aggregate_stdout.log"; then
  echo "AGGREGATE_STDOUT_MISSING_OK_TOKEN: $OUTPUT_ROOT/aggregate_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== REPORTS ====="
for ASSET_ID in B075YLTF7Q B07HSK7MXZ B073NZS57V B07B8MWCR8; do
  echo "CASE_BOARD: $OUTPUT_ROOT/boards/$ASSET_ID/rendered_view_board.jpg"
  echo "CASE_REPORT: $OUTPUT_ROOT/reports/$ASSET_ID/rendered_view_report.md"
  echo "CASE_METRICS: $OUTPUT_ROOT/metrics/$ASSET_ID/rendered_view_metrics.json"
done
echo "SUMMARY_JSON: $OUTPUT_ROOT/summary/rendered_view_metrics_summary.json"
echo "SUMMARY_MD: $OUTPUT_ROOT/summary/rendered_view_metrics_summary.md"
ls -lh \
  "$OUTPUT_ROOT/summary/rendered_view_metrics_summary.json" \
  "$OUTPUT_ROOT/summary/rendered_view_metrics_summary.md"

echo "PHASE2K3_RENDERED_VIEW_EVAL_OK"
echo "===== JOB END: SUCCESS ====="
date
