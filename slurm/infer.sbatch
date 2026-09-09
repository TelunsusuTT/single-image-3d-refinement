#!/usr/bin/env bash
#SBATCH --job-name=hy3dpaint-infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: sbatch slurm/infer.sbatch CONFIG_JSON CASE_DIR OUTPUT_DIR [CHECKPOINT]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

resolve_from_project() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "$1"
  fi
}

CONFIG="$(resolve_from_project "$1")"
CASE_DIR="$(resolve_from_project "$2")"
OUTPUT_DIR="$(resolve_from_project "$3")"
CHECKPOINT="${4:-}"
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT="$(resolve_from_project "$CHECKPOINT")"
fi

if [[ -n "${HY3DFT_ENV_FILE:-}" ]]; then
  if [[ ! -f "$HY3DFT_ENV_FILE" ]]; then
    echo "HY3DFT_ENV_FILE does not exist: $HY3DFT_ENV_FILE" >&2
    exit 2
  fi
  source "$HY3DFT_ENV_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${HY3DFT_RUN_ID:-slurm_${SLURM_JOB_ID:-manual}}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --config "$CONFIG"
  --case-dir "$CASE_DIR"
  --output-dir "$OUTPUT_DIR"
  --run-id "$RUN_ID"
)
if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/infer.py "${ARGS[@]}" --check-only
"$PYTHON_BIN" scripts/infer.py "${ARGS[@]}" --run
