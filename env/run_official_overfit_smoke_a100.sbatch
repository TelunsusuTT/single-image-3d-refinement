#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-smoke
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

# Use node-local temporary storage for short-lived worker/temp files.
# Persistent project data, logs and HF/torch/pip caches remain under /vol/bitbucket via env.sh.
export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

# Avoid writing __pycache__ into the external Hunyuan source tree.
export PYTHONDONTWRITEBYTECODE=1

CONFIG_PATH="$PROJ/configs/ft_overfit_smoke.yaml"
LOGDIR="$PROJ/logs/train"

cd "$HYPAINT"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "CUDA_HOME=$CUDA_HOME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "LOGDIR=$LOGDIR"

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

echo "===== MODULE PREFLIGHT ====="
python3 - <<'PY'
import importlib
import os
import sys

print("cwd:", os.getcwd())
print("sys.path first 12:")
for p in sys.path[:12]:
    print(" ", p)

checks = [
    "torch",
    "src.utils.train_util",
    "hunyuanpaintpbr",
    "hunyuanpaintpbr.unet.model",
    "hunyuanpaintpbr.pipeline",
]

for m in checks:
    importlib.import_module(m)
    print("OK:", m)

from hunyuanpaintpbr import HunyuanPaint
print("OK: from hunyuanpaintpbr import HunyuanPaint ->", HunyuanPaint)

target = "hunyuanpaintpbr.HunyuanPaint"
module, cls = target.rsplit(".", 1)
obj = getattr(importlib.import_module(module), cls)
print("OK: config target import ->", target, obj)

print("MODULE_PREFLIGHT_OK")
PY

echo "===== HF REPO PREFLIGHT ====="
python3 - <<PY
from pathlib import Path
from omegaconf import OmegaConf
from huggingface_hub import model_info

cfg_path = Path("$CONFIG_PATH")
cfg = OmegaConf.load(cfg_path)

sd_cfg = cfg.model.params.stable_diffusion_config
repo_id = sd_cfg.get("pretrained_model_name_or_path", None)

if repo_id is None:
    raise SystemExit("stable_diffusion_config.pretrained_model_name_or_path not found")

if repo_id == "stabilityai/stable-diffusion-2-1":
    raise SystemExit(
        "BAD_SD21_REPO_ID: stabilityai/stable-diffusion-2-1 is not accessible. "
        "Use sd2-community/stable-diffusion-2-1 in the copied project config."
    )

info = model_info(repo_id)
print("HF_REPO_OK:", repo_id)
print("HF_REPO_SHA:", info.sha)
print("HF_REPO_PREFLIGHT_OK")
PY

echo "===== DATASET PREFLIGHT ====="
python3 - <<PY
import json
from pathlib import Path
from omegaconf import OmegaConf

cfg_path = Path("$CONFIG_PATH")
cfg = OmegaConf.load(cfg_path)

json_paths = []

for split in ["train", "validation"]:
    entries = cfg.data.params[split]
    for entry in entries:
        json_paths.append(str(entry.params.json_path))

for json_path in json_paths:
    jp = Path(json_path)
    if not jp.is_file():
        raise SystemExit(f"DATASET_JSON_MISSING: {jp}")

    data = json.loads(jp.read_text())
    print("JSON_OK:", jp, "num_entries:", len(data))

    if not data:
        raise SystemExit(f"DATASET_JSON_EMPTY: {jp}")

    for sample in data:
        sample_dir = Path(sample)
        render_cond = sample_dir / "render_cond"
        render_tex = sample_dir / "render_tex"

        if not sample_dir.is_dir():
            raise SystemExit(f"SAMPLE_DIR_MISSING: {sample_dir}")
        if not render_cond.is_dir():
            raise SystemExit(f"RENDER_COND_DIR_MISSING: {render_cond}")
        if not render_tex.is_dir():
            raise SystemExit(f"RENDER_TEX_DIR_MISSING: {render_tex}")

        cond_count = (
            len(list(render_cond.glob("*.png")))
            + len(list(render_cond.glob("*.jpg")))
            + len(list(render_cond.glob("*.jpeg")))
        )

        albedo_count = (
            len(list(render_tex.glob("*_albedo.png")))
            + len(list(render_tex.glob("*_albedo.jpg")))
            + len(list(render_tex.glob("*_albedo.jpeg")))
        )

        print("SAMPLE_OK:", sample_dir)
        print("  render_cond images:", cond_count)
        print("  render_tex albedo:", albedo_count)

        if cond_count == 0:
            raise SystemExit(f"NO_CONDITION_IMAGES: {render_cond}")
        if albedo_count < 6:
            raise SystemExit(f"TOO_FEW_ALBEDO_VIEWS: {render_tex}")

print("DATASET_PREFLIGHT_OK")
PY

echo "===== NVIDIA SMI ====="
nvidia-smi

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
print("CUBLAS_MATMUL_OK")
PY

SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-45m}"

echo "===== TRAINING SMOKE TEST ====="
echo "timeout: $SMOKE_TIMEOUT"
echo "HYPAINT: $HYPAINT"
echo "config: $CONFIG_PATH"
echo "logdir: $LOGDIR"

set +e
timeout "$SMOKE_TIMEOUT" python3 train.py \
  --base "$CONFIG_PATH" \
  --name official_overfit_smoke \
  --logdir "$LOGDIR"
train_status=$?
set -e

if [[ "$train_status" -eq 124 || "$train_status" -eq 137 ]]; then
  echo "Official overfit smoke test reached timeout after $SMOKE_TIMEOUT; exiting 0."
  echo "===== JOB END: TIMEOUT OK ====="
  date
  exit 0
fi

if [[ "$train_status" -ne 0 ]]; then
  echo "Official overfit smoke test failed with exit code $train_status." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

echo "Official overfit smoke test completed before timeout."
echo "===== JOB END: SUCCESS ====="
date
