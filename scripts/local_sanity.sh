#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env/env.sh"

missing=0

require_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    echo "OK: $path"
  else
    echo "MISSING: $path" >&2
    missing=1
  fi
}

require_path "$HYPAINT"
require_path "$HYPAINT/train.py"
require_path "$HYPAINT/cfgs/hunyuan-paint-pbr.yaml"
require_path "$HYPAINT/train_examples"
require_path "$HYPAINT/train_examples/examples.json"

if [[ "$missing" -ne 0 ]]; then
  echo "One or more required upstream paths are missing." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command is not available after sourcing env/env.sh" >&2
  exit 1
fi

conda activate "$ENV_NAME"

python3 - <<'PY'
import torch

print(f"torch version: {torch.__version__}")
print(f"torch cuda version: {torch.version.cuda}")
PY

echo "Local sanity checks completed. GPU availability is intentionally not required here."
