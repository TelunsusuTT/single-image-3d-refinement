#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-abo-smoke
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

echo "===== LOAD ENV ====="
source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_abo_one_asset_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

# Avoid writing __pycache__ into the external Hunyuan source tree.
export PYTHONDONTWRITEBYTECODE=1

CONFIG_PATH="$PROJ/configs/ft_abo_one_asset_smoke.yaml"
EXAMPLES_JSON="$PROJ/data/hy3dpaint_train_examples/abo_one_asset/examples_train_abs.json"
LOGDIR="$PROJ/logs/train"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "CUDA_HOME=$CUDA_HOME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "EXAMPLES_JSON=$EXAMPLES_JSON"
echo "LOGDIR=$LOGDIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== CONFIG EXISTS CHECK ====="
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "CONFIG_MISSING: $CONFIG_PATH" >&2
  exit 1
fi
echo "CONFIG_EXISTS_OK: $CONFIG_PATH"

echo "===== EXAMPLES JSON EXISTS CHECK ====="
if [[ ! -f "$EXAMPLES_JSON" ]]; then
  echo "EXAMPLES_JSON_MISSING: $EXAMPLES_JSON" >&2
  exit 1
fi
echo "EXAMPLES_JSON_EXISTS_OK: $EXAMPLES_JSON"

echo "===== CONFIG TARGET CHECK ====="
python3 - <<PY
from pathlib import Path

cfg = Path("$CONFIG_PATH")
text = cfg.read_text()

bad = "target: hunyuanpaintpbr.model.HunyuanPaint"
good = "target: hunyuanpaintpbr.HunyuanPaint"

if bad in text:
    raise SystemExit(f"BAD_CONFIG_TARGET_FOUND: {bad}")

if good not in text:
    print("WARNING: expected HunyuanPaint target not found. Current target lines:")
    for i, line in enumerate(text.splitlines(), 1):
        if "target:" in line:
            print(f"{i}: {line}")
    raise SystemExit("CONFIG_TARGET_CHECK_FAILED")

print("CONFIG_TARGET_OK:", good)
PY

echo "===== DATASET READINESS PREFLIGHT ====="
python "$PROJ/scripts/check_phase1f_readiness.py" --examples-json "$EXAMPLES_JSON" --config "$CONFIG_PATH"

echo "===== OFFICIAL-STYLE PACKAGING CHECK ====="
python "$PROJ/scripts/check_hy3dpaint_example.py" --examples-json "$EXAMPLES_JSON" --num-view 6 --strict

echo "===== CUDA SANITY ====="
python3 - <<'PY'
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
PY

SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-60m}"

echo "===== TRAIN START ====="
echo "train starts"
echo "timeout: $SMOKE_TIMEOUT"
echo "HYPAINT: $HYPAINT"
echo "config: $CONFIG_PATH"
echo "logdir: $LOGDIR"

cd "$HYPAINT"

set +e
timeout "$SMOKE_TIMEOUT" python3 train.py --base "$CONFIG_PATH" --name abo_one_asset_smoke --logdir "$LOGDIR" --gpus 1
train_status=$?
set -e

if [[ "$train_status" -eq 124 || "$train_status" -eq 137 ]]; then
  echo "ABO one-asset smoke test timed out after $SMOKE_TIMEOUT." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

if [[ "$train_status" -ne 0 ]]; then
  echo "ABO one-asset smoke test failed with exit code $train_status." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

echo "ABO one-asset smoke test completed."
echo "max_steps reached"
echo "===== JOB END: SUCCESS ====="
date
