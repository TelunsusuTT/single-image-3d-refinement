#!/usr/bin/env bash
#SBATCH --job-name=hy3dpaint-evaluate
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: sbatch slurm/evaluate.sbatch EVALUATION_JSON CASES_JSON OUTPUT_ROOT" >&2
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
CASES_CONFIG="$(resolve_from_project "$2")"
OUTPUT_ROOT="$(resolve_from_project "$3")"

if [[ -n "${HY3DFT_ENV_FILE:-}" ]]; then
  if [[ ! -f "$HY3DFT_ENV_FILE" ]]; then
    echo "HY3DFT_ENV_FILE does not exist: $HY3DFT_ENV_FILE" >&2
    exit 2
  fi
  source "$HY3DFT_ENV_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
EVALUATION_STAGE="${HY3DFT_EVALUATION_STAGE:-all}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --config "$CONFIG"
  --cases-config "$CASES_CONFIG"
  --output-root "$OUTPUT_ROOT"
  --stage "$EVALUATION_STAGE"
)

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/evaluate.py "${ARGS[@]}" --check-only
"$PYTHON_BIN" scripts/evaluate.py "${ARGS[@]}" --run
