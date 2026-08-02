#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase2n-final-test-infer
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2n-final-test-infer-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/phase2n-final-test-infer-%j.err

set -euo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

CUDA_SETUP=/vol/cuda/12.4.0/setup.sh
if [[ ! -f "$CUDA_SETUP" ]]; then
  echo "CUDA 12.4 setup is missing: $CUDA_SETUP" >&2
  exit 1
fi
source "$CUDA_SETUP"
export CUDA_HOME=/vol/cuda/12.4.0

export JOB_TMPDIR="/tmp/${USER}/phase2n_final_test_infer_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJ/src:$PROJ/scripts${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="$PROJ/configs/phase2n_final_test_inference.json"
MANIFEST_CONFIG="$PROJ/configs/phase2n_final_test_manifest.json"
MANIFEST="$PROJ/outputs/phase2n/final_test_manifest/phase2n_final_test_manifest_v1.json"
RUN_ID="phase2n_final_test_infer_v1"
RUN_DIR="$PROJ/outputs/phase2n/final_test_inference/$RUN_ID"

mkdir -p "$PROJ/logs/slurm"

echo "===== ENVIRONMENT ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "CUDA_HOME=$CUDA_HOME"
echo "PYTHONPATH=$PYTHONPATH"
echo "TMPDIR=$TMPDIR"
echo "CONFIG=$CONFIG"
echo "RUN_DIR=$RUN_DIR"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing final-test inference run: $RUN_DIR" >&2
  exit 1
fi
if [[ ! -s "$MANIFEST" ]]; then
  echo "Write and review the final-test manifest before submission: $MANIFEST" >&2
  exit 1
fi

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== GIT HEAD ====="
git -C "$PROJ" rev-parse HEAD

echo "===== DISK USAGE ====="
df -h "$PROJ" /tmp

echo "===== FINAL-TEST FREEZE, SPLIT, AND REUSE AUDIT ====="
python "$PROJ/scripts/phase2n_build_final_test_manifest.py" \
  --config "$MANIFEST_CONFIG" \
  --check-only

echo "===== INFERENCE READINESS ====="
python "$PROJ/scripts/phase2n_week2_infer_pilots.py" \
  --config "$CONFIG" \
  --check-only

echo "===== PC-FULL STEP 320 TEST INFERENCE ====="
python "$PROJ/scripts/phase2n_week2_infer_pilots.py" \
  --config "$CONFIG" \
  --run-all \
  --run-id "$RUN_ID"

GLB_COUNT=$(find "$RUN_DIR/pc_full_step320" -type f -name textured_mesh.glb | wc -l)
CASE_SUCCESS_COUNT=$(find "$RUN_DIR/pc_full_step320" -mindepth 2 -maxdepth 2 -type f -name _SUCCESS | wc -l)
if [[ "$GLB_COUNT" -ne 11 ]]; then
  echo "Expected exactly 11 new GLBs, found $GLB_COUNT" >&2
  exit 1
fi
if [[ "$CASE_SUCCESS_COUNT" -ne 11 ]]; then
  echo "Expected exactly 11 case success markers, found $CASE_SUCCESS_COUNT" >&2
  exit 1
fi
test -s "$RUN_DIR/pc_full_step320/_SUCCESS"
test -s "$RUN_DIR/_SUCCESS"
grep -Fxq "PHASE2N_FINAL_TEST_PC_FULL_STEP320_INFERENCE_OK" \
  "$RUN_DIR/pc_full_step320/_SUCCESS"
grep -Fxq "PHASE2N_FINAL_TEST_INFERENCE_OK" "$RUN_DIR/_SUCCESS"

echo "new_glb_count=$GLB_COUNT"
echo "case_success_count=$CASE_SUCCESS_COUNT"
echo "===== JOB END: SUCCESS ====="
date
