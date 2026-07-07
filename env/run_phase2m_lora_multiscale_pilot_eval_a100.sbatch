#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase2m_lora_multiscale_pilot_eval
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_lora_multiscale_pilot_eval-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2m_lora_multiscale_pilot_eval-%j.err

set -euo pipefail

PROJ=/vol/bitbucket/ct1022/hy3dpaint_finetune
source "$PROJ/env/env.sh"
conda activate "$ENV_NAME"

JOB_ID=${SLURM_JOB_ID:-manual}
JOB_TMPDIR=/tmp/${USER}/phase2m_lora_multiscale_pilot_eval_${JOB_ID}
mkdir -p "$JOB_TMPDIR" "$PROJ/logs/slurm"
export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/scripts:$PROJ/src:${PYTHONPATH:-}"

PILOT_SUMMARY="$PROJ/outputs/phase2m/lora_multiscale_pilot/pilot_summary.json"
PER_VARIANT_OUTPUTS="$PROJ/outputs/phase2m/lora_multiscale_pilot/per_variant_outputs.json"
RENDER_ROOT="$PROJ/outputs/phase2m/lora_multiscale_pilot_rendered"
EVAL_ROOT="$PROJ/outputs/phase2m/lora_multiscale_pilot_eval"
SBATCH_PATH="$PROJ/env/run_phase2m_lora_multiscale_pilot_eval_a100.sbatch"

if [[ -z "${BLENDER_BIN:-}" ]]; then
  BLENDER_BIN="$(command -v blender || true)"
fi
if [[ -z "${BLENDER_BIN:-}" && -x /vol/bitbucket/ct1022/tools/bin/blender ]]; then
  BLENDER_BIN="/vol/bitbucket/ct1022/tools/bin/blender"
fi
export BLENDER_BIN

cd "$PROJ"

echo "===== PHASE 2M.3C LORA MULTISCALE PILOT EVAL ====="
date
hostname
nvidia-smi

echo "PROJ=$PROJ"
echo "ENV_NAME=$ENV_NAME"
echo "TMPDIR=$TMPDIR"
echo "BLENDER_BIN=$BLENDER_BIN"
echo "PILOT_SUMMARY=$PILOT_SUMMARY"
echo "RENDER_ROOT=$RENDER_ROOT"
echo "EVAL_ROOT=$EVAL_ROOT"

echo "===== READINESS ====="
python "$PROJ/scripts/check_phase2m_lora_multiscale_pilot_eval_ready.py" \
  --pilot-summary "$PILOT_SUMMARY" \
  --per-variant-outputs "$PER_VARIANT_OUTPUTS" \
  --render-root "$RENDER_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --sbatch "$SBATCH_PATH"

echo "===== BLENDER RENDER ====="
"$BLENDER_BIN" -b --python "$PROJ/scripts/phase2m_render_lora_multiscale_pilot.py" -- \
  --pilot-summary "$PILOT_SUMMARY" \
  --per-variant-outputs "$PER_VARIANT_OUTPUTS" \
  --output-root "$RENDER_ROOT" \
  --resolution 512 \
  --only-missing 2>&1 | tee "$RENDER_ROOT/render_stdout.log"

if ! grep -q "PHASE2M_M3C_RENDER_LORA_PILOT_OK" "$RENDER_ROOT/render_stdout.log"; then
  echo "RENDER_STDOUT_MISSING_OK_TOKEN: $RENDER_ROOT/render_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== AGGREGATE METRICS ====="
python "$PROJ/scripts/aggregate_phase2m_lora_multiscale_pilot_eval.py" \
  --render-root "$RENDER_ROOT" \
  --eval-root "$EVAL_ROOT" 2>&1 | tee "$EVAL_ROOT/aggregate_stdout.log"

if ! grep -q "PHASE2M_M3C_AGGREGATE_LORA_PILOT_OK" "$EVAL_ROOT/aggregate_stdout.log"; then
  echo "AGGREGATE_STDOUT_MISSING_OK_TOKEN: $EVAL_ROOT/aggregate_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== MAKE BOARDS ====="
python "$PROJ/scripts/make_phase2m_lora_multiscale_pilot_boards.py" \
  --render-root "$RENDER_ROOT" \
  --eval-root "$EVAL_ROOT" 2>&1 | tee "$EVAL_ROOT/boards_stdout.log"

if ! grep -q "PHASE2M_M3C_LORA_PILOT_BOARDS_OK" "$EVAL_ROOT/boards_stdout.log"; then
  echo "BOARDS_STDOUT_MISSING_OK_TOKEN: $EVAL_ROOT/boards_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== OUTPUTS ====="
for path in \
  "$RENDER_ROOT/render_eval_cases.json" \
  "$RENDER_ROOT/render_summary.json" \
  "$EVAL_ROOT/metrics_rows.csv" \
  "$EVAL_ROOT/aggregate_summary.json" \
  "$EVAL_ROOT/scale_comparison.json" \
  "$EVAL_ROOT/boards/board_summary.json"; do
  if [[ ! -s "$path" ]]; then
    echo "missing or empty output: $path" >&2
    echo "===== JOB END: FAILED ====="
    date
    exit 1
  fi
  bytes=$(stat -c '%s' "$path")
  echo "M3C_OUTPUT: $path bytes=$bytes"
done

echo "PHASE2M_M3C_LORA_MULTISCALE_PILOT_EVAL_JOB_OK"
echo "===== JOB END: SUCCESS ====="
date
