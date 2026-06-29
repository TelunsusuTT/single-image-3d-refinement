#!/usr/bin/env bash

# Shared Phase 0 environment. Intended to be sourced from this project.

export PROJ=/vol/bitbucket/ct1022/hy3dpaint_finetune
export HY21=/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1
export HYPAINT="$HY21/hy3dpaint"

CONDA_SH=/vol/bitbucket/ct1022/miniconda3/etc/profile.d/conda.sh
if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
else
  echo "WARNING: conda setup file not found: $CONDA_SH" >&2
fi

if [[ -f "$PROJ/env/local.env" ]]; then
  source "$PROJ/env/local.env"
fi

export ENV_NAME="${ENV_NAME:-/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/env/conda_envs/hunyuan3d21_work}"

CUDA_SETUP=/vol/cuda/12.4.0/setup.sh
if [[ -f "$CUDA_SETUP" ]]; then
  source "$CUDA_SETUP"
else
  echo "WARNING: CUDA setup file not found: $CUDA_SETUP" >&2
fi

export CUDA_HOME=/vol/cuda/12.4.0

# Cache redirection: keep all generated cache files inside this project.
export HF_HOME="$PROJ/caches/hf"
export HUGGINGFACE_HUB_CACHE="$PROJ/caches/hf/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TORCH_HOME="$PROJ/caches/torch"
export PIP_CACHE_DIR="$PROJ/caches/pip"
export XDG_CACHE_HOME="$PROJ/caches/xdg"
export MPLCONFIGDIR="$PROJ/caches/matplotlib"
export TMPDIR="$PROJ/tmp"

mkdir -p \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR" \
  "$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENCV_IO_ENABLE_OPENEXR=1
export TOKENIZERS_PARALLELISM=false

# Important:
# HYPAINT is needed for imports like src.utils.train_util and hunyuanpaintpbr.
# HY21 is kept for project-level imports.
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

printf 'PROJ=%s\n' "$PROJ"
printf 'HY21=%s\n' "$HY21"
printf 'HYPAINT=%s\n' "$HYPAINT"
printf 'ENV_NAME=%s\n' "$ENV_NAME"
printf 'CUDA_HOME=%s\n' "$CUDA_HOME"