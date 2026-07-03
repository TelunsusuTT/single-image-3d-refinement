#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2k2-multicase200
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname
pwd

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2k2_multicase200_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CASES_CONFIG="$PROJ/configs/phase2k2_truepbr200_cases.json"
CHECKPOINT="$PROJ/checkpoints/pilot_v1_truepbr_200_lr1e6/pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt"
OUTPUT_ROOT="$PROJ/outputs/phase2k/multicase_truepbr200"
LOAD_DIR="$OUTPUT_ROOT/load_only"
SUMMARY_DIR="$OUTPUT_ROOT/summary"
ASSET_IDS=("B07HSK7MXZ" "B073NZS57V" "B07B8MWCR8")

mkdir -p "$PROJ/logs/slurm"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CASES_CONFIG=$CASES_CONFIG"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "LOAD_DIR=$LOAD_DIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2K.2 READINESS ====="
python "$PROJ/scripts/check_phase2k2_truepbr200_multicase_readiness.py" \
  --cases-config "$CASES_CONFIG" \
  --hypaint "$HYPAINT"

echo "===== CASE PREPARATION ====="
python "$PROJ/scripts/prepare_phase2k2_multicase_cases.py" \
  --cases-config "$CASES_CONFIG"

echo "===== LOAD-ONLY CHECK ====="
mkdir -p "$LOAD_DIR"
python "$PROJ/scripts/load_phase2g4_finetuned_checkpoint_only.py" \
  --checkpoint "$CHECKPOINT" \
  --hypaint "$HYPAINT" \
  --out-json "$LOAD_DIR/load_only_report.json" \
  --out-md "$LOAD_DIR/load_only_report.md" \
  --max-num-view 6 \
  --resolution 512 2>&1 | tee "$LOAD_DIR/load_only_stdout.log"

if ! grep -q "PHASE2G4_CHECKPOINT_LOAD_ONLY_OK" "$LOAD_DIR/load_only_report.md"; then
  echo "LOAD_ONLY_REPORT_MISSING_OK_TOKEN: $LOAD_DIR/load_only_report.md" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

if ! grep -q "PHASE2G4_CHECKPOINT_LOAD_ONLY_OK" "$LOAD_DIR/load_only_stdout.log"; then
  echo "LOAD_ONLY_STDOUT_MISSING_OK_TOKEN: $LOAD_DIR/load_only_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2K2_CUDA_PY'
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
PHASE2K2_CUDA_PY

validate_textured_outputs() {
  local label="$1"
  local output_dir="$2"
  local prefix="$3"
  local obj="$output_dir/${prefix}_textured_mesh.obj"
  local glb="$output_dir/${prefix}_textured_mesh.glb"
  local albedo="$output_dir/${prefix}_textured_mesh.jpg"
  local metallic="$output_dir/${prefix}_textured_mesh_metallic.jpg"
  local roughness="$output_dir/${prefix}_textured_mesh_roughness.jpg"

  for path in "$obj" "$glb" "$albedo" "$metallic" "$roughness"; do
    if [[ ! -s "$path" ]]; then
      echo "MISSING_OR_EMPTY_${label}_OUTPUT: $path" >&2
      echo "===== JOB END: FAILED ====="
      date
      exit 1
    fi
    size="$(du -h "$path" | awk '{print $1}')"
    bytes="$(stat -c '%s' "$path")"
    echo "${label}_OUTPUT: $path size=$size bytes=$bytes"
  done
}

for ASSET_ID in "${ASSET_IDS[@]}"; do
  CASE_DIR="$OUTPUT_ROOT/cases/$ASSET_ID"
  BASE_DIR="$OUTPUT_ROOT/base/$ASSET_ID"
  FINETUNED_DIR="$OUTPUT_ROOT/finetuned/$ASSET_ID"
  COMPARE_DIR="$OUTPUT_ROOT/compare/$ASSET_ID"

  echo "===== CASE $ASSET_ID: BASE NO-REMESH INFERENCE ====="
  mkdir -p "$BASE_DIR"
  python "$PROJ/scripts/run_phase2g_paint_infer.py" \
    --case-dir "$CASE_DIR" \
    --output-dir "$BASE_DIR" \
    --mode base \
    --max-num-view 6 \
    --resolution 512 \
    --device cuda \
    --no-remesh 2>&1 | tee "$BASE_DIR/phase2k2_base_stdout.log"
  validate_textured_outputs "BASE_$ASSET_ID" "$BASE_DIR" "base"

  echo "===== CASE $ASSET_ID: FINE-TUNED NO-REMESH INFERENCE ====="
  mkdir -p "$FINETUNED_DIR"
  python "$PROJ/scripts/run_phase2g_paint_infer.py" \
    --case-dir "$CASE_DIR" \
    --output-dir "$FINETUNED_DIR" \
    --mode finetuned \
    --checkpoint "$CHECKPOINT" \
    --max-num-view 6 \
    --resolution 512 \
    --device cuda \
    --no-remesh 2>&1 | tee "$FINETUNED_DIR/phase2k2_finetuned_stdout.log"

  if ! grep -q "PHASE2G5_FINETUNED_CHECKPOINT_LOADED_OK" "$FINETUNED_DIR/phase2k2_finetuned_stdout.log"; then
    echo "INFER_STDOUT_MISSING_CHECKPOINT_LOAD_TOKEN: $FINETUNED_DIR/phase2k2_finetuned_stdout.log" >&2
    echo "===== JOB END: FAILED ====="
    date
    exit 1
  fi
  validate_textured_outputs "FINETUNED_$ASSET_ID" "$FINETUNED_DIR" "finetuned"

  echo "===== CASE $ASSET_ID: TEXTURE COMPARISON ====="
  mkdir -p "$COMPARE_DIR"
  python "$PROJ/scripts/make_phase2g6_texture_comparison.py" \
    --case-dir "$CASE_DIR" \
    --base-dir "$BASE_DIR" \
    --finetuned-dir "$FINETUNED_DIR" \
    --output-dir "$COMPARE_DIR" 2>&1 | tee "$COMPARE_DIR/phase2k2_compare_stdout.log"

  if ! grep -q "PHASE2G6_TEXTURE_COMPARISON_OK" "$COMPARE_DIR/phase2k2_compare_stdout.log"; then
    echo "COMPARE_STDOUT_MISSING_OK_TOKEN: $COMPARE_DIR/phase2k2_compare_stdout.log" >&2
    echo "===== JOB END: FAILED ====="
    date
    exit 1
  fi
done

echo "===== AGGREGATE METRICS ====="
python "$PROJ/scripts/aggregate_phase2k2_multicase_metrics.py" \
  --cases-config "$CASES_CONFIG" \
  --output-root "$OUTPUT_ROOT" 2>&1 | tee "$SUMMARY_DIR/aggregate_stdout.log"

if ! grep -q "PHASE2K2_MULTICASE_METRICS_AGGREGATED_OK" "$SUMMARY_DIR/aggregate_stdout.log"; then
  echo "AGGREGATE_STDOUT_MISSING_OK_TOKEN: $SUMMARY_DIR/aggregate_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== REPORTS ====="
for ASSET_ID in "${ASSET_IDS[@]}"; do
  echo "CASE_REPORT_MD: $OUTPUT_ROOT/compare/$ASSET_ID/base_vs_finetuned_report.md"
  echo "CASE_METRICS_JSON: $OUTPUT_ROOT/compare/$ASSET_ID/base_vs_finetuned_metrics.json"
  echo "CASE_BOARD: $OUTPUT_ROOT/compare/$ASSET_ID/base_vs_finetuned_texture_board.jpg"
done
echo "SUMMARY_JSON: $OUTPUT_ROOT/summary/multicase_metrics_summary.json"
echo "SUMMARY_MD: $OUTPUT_ROOT/summary/multicase_metrics_summary.md"
ls -lh \
  "$OUTPUT_ROOT/summary/multicase_metrics_summary.json" \
  "$OUTPUT_ROOT/summary/multicase_metrics_summary.md"

echo "PHASE2K2_TRUEPBR200_MULTICASE_EVAL_OK"
echo "===== JOB END: SUCCESS ====="
date
