#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2k1-truepbr200
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2k1_truepbr200_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CONFIG_PATH="$PROJ/configs/ft_pilot_v1_truepbr_200_lr1e6.yaml"
EXAMPLES_JSON="$PROJ/data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json"
CHECKPOINT_ROOT="$PROJ/checkpoints/pilot_v1_truepbr_200_lr1e6"
CASE_DIR="$PROJ/outputs/phase2g/infer_cases/B075YLTF7Q"
BASE_DIR="$PROJ/outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke"
EVAL_ROOT="$PROJ/outputs/phase2k/eval_truepbr200"
LOAD_DIR="$EVAL_ROOT/load_only"
DELTA_DIR="$EVAL_ROOT/delta_audit"
INFER_DIR="$EVAL_ROOT/infer/B075YLTF7Q"
COMPARE_DIR="$EVAL_ROOT/compare/B075YLTF7Q"
LOGDIR="$PROJ/logs/train"
JOB_MARKER="$JOB_TMPDIR/job_start.marker"

mkdir -p "$PROJ/logs/slurm"
mkdir -p "$CHECKPOINT_ROOT"
touch "$JOB_MARKER"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "EXAMPLES_JSON=$EXAMPLES_JSON"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "CASE_DIR=$CASE_DIR"
echo "BASE_DIR=$BASE_DIR"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "LOAD_DIR=$LOAD_DIR"
echo "DELTA_DIR=$DELTA_DIR"
echo "INFER_DIR=$INFER_DIR"
echo "COMPARE_DIR=$COMPARE_DIR"
echo "LOGDIR=$LOGDIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2K.1 READINESS ====="
python "$PROJ/scripts/check_phase2k1_truepbr200_readiness.py" \
  --examples-json "$EXAMPLES_JSON" \
  --config "$CONFIG_PATH" \
  --hypaint "$HYPAINT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --case-dir "$CASE_DIR" \
  --base-dir "$BASE_DIR" \
  --eval-output-root "$EVAL_ROOT"

echo "===== OFFICIAL-STYLE PACKAGING CHECK ====="
python "$PROJ/scripts/check_hy3dpaint_example.py" \
  --examples-json "$EXAMPLES_JSON" \
  --num-view 6 \
  --strict

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2K1_CUDA_PY'
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
PHASE2K1_CUDA_PY

TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-3h}"

echo "===== STAGE 1: TRUE-PBR 200-STEP TRAIN ====="
echo "train starts"
echo "timeout: $TRAIN_TIMEOUT"
echo "HYPAINT: $HYPAINT"
echo "config: $CONFIG_PATH"
echo "logdir: $LOGDIR"

cd "$HYPAINT"

set +e
timeout "$TRAIN_TIMEOUT" python3 train.py \
  --base "$CONFIG_PATH" \
  --name pilot_v1_truepbr_200_lr1e6 \
  --logdir "$LOGDIR" \
  --gpus 1
train_status=$?
set -e

if [[ "$train_status" -eq 124 || "$train_status" -eq 137 ]]; then
  echo "true-PBR 200-step train timed out after $TRAIN_TIMEOUT." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

if [[ "$train_status" -ne 0 ]]; then
  echo "true-PBR 200-step train failed with exit code $train_status." >&2
  echo "===== JOB END: FAILED ====="
  date
  exit "$train_status"
fi

echo "===== CHECKPOINT SEARCH ====="
checkpoint_count=0
FOUND_CKPT=""
while IFS= read -r -d '' ckpt; do
  size="$(du -h "$ckpt" | awk '{print $1}')"
  bytes="$(stat -c '%s' "$ckpt")"
  echo "CHECKPOINT: $ckpt size=$size bytes=$bytes"
  FOUND_CKPT="$ckpt"
  checkpoint_count=$((checkpoint_count + 1))
done < <(find "$CHECKPOINT_ROOT" -type f -name "*.ckpt" -newer "$JOB_MARKER" -print0)

if [[ "$checkpoint_count" -ne 1 ]]; then
  echo "EXPECTED_EXACTLY_ONE_NEW_CHECKPOINT, found $checkpoint_count" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "FOUND_CKPT=$FOUND_CKPT"

echo "===== STAGE 2: LOAD-ONLY CHECK ====="
mkdir -p "$LOAD_DIR"
python "$PROJ/scripts/load_phase2g4_finetuned_checkpoint_only.py" \
  --checkpoint "$FOUND_CKPT" \
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

echo "===== STAGE 3: BASE VS TRUE-PBR200 DELTA AUDIT ====="
mkdir -p "$DELTA_DIR"
python "$PROJ/scripts/compare_phase2i_base_unet_to_checkpoints.py" \
  --hypaint "$HYPAINT" \
  --checkpoint "truepbr200=$FOUND_CKPT" \
  --out-json "$DELTA_DIR/base_vs_truepbr200_delta.json" \
  --out-md "$DELTA_DIR/base_vs_truepbr200_delta.md" \
  --max-sample-keys 100 2>&1 | tee "$DELTA_DIR/base_vs_truepbr200_delta_stdout.log"

if ! grep -q "PHASE2I_BASE_VS_CHECKPOINT_COMPARE_OK" "$DELTA_DIR/base_vs_truepbr200_delta_stdout.log"; then
  echo "DELTA_STDOUT_MISSING_OK_TOKEN: $DELTA_DIR/base_vs_truepbr200_delta_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== STAGE 4: FINE-TUNED NO-REMESH INFERENCE ====="
mkdir -p "$INFER_DIR"
python "$PROJ/scripts/run_phase2g_paint_infer.py" \
  --case-dir "$CASE_DIR" \
  --output-dir "$INFER_DIR" \
  --mode finetuned \
  --checkpoint "$FOUND_CKPT" \
  --max-num-view 6 \
  --resolution 512 \
  --device cuda \
  --no-remesh 2>&1 | tee "$INFER_DIR/phase2k1_infer_stdout.log"

if ! grep -q "PHASE2G5_FINETUNED_CHECKPOINT_LOADED_OK" "$INFER_DIR/phase2k1_infer_stdout.log"; then
  echo "INFER_STDOUT_MISSING_CHECKPOINT_LOAD_TOKEN: $INFER_DIR/phase2k1_infer_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

obj_count=0
while IFS= read -r -d '' obj; do
  size="$(du -h "$obj" | awk '{print $1}')"
  bytes="$(stat -c '%s' "$obj")"
  echo "OUTPUT_OBJ: $obj size=$size bytes=$bytes"
  obj_count=$((obj_count + 1))
done < <(find "$INFER_DIR" -type f -name "*.obj" -print0)

glb_count=0
while IFS= read -r -d '' glb; do
  size="$(du -h "$glb" | awk '{print $1}')"
  bytes="$(stat -c '%s' "$glb")"
  echo "OUTPUT_GLB: $glb size=$size bytes=$bytes"
  glb_count=$((glb_count + 1))
done < <(find "$INFER_DIR" -type f -name "*.glb" -print0)

if [[ "$obj_count" -lt 1 || "$glb_count" -lt 1 ]]; then
  echo "EXPECTED_AT_LEAST_ONE_OBJ_AND_ONE_GLB, found obj=$obj_count glb=$glb_count" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== STAGE 5: BASE VS TRUE-PBR200 TEXTURE COMPARISON ====="
mkdir -p "$COMPARE_DIR"
python "$PROJ/scripts/make_phase2g6_texture_comparison.py" \
  --case-dir "$CASE_DIR" \
  --base-dir "$BASE_DIR" \
  --finetuned-dir "$INFER_DIR" \
  --output-dir "$COMPARE_DIR" 2>&1 | tee "$COMPARE_DIR/phase2k1_compare_stdout.log"

if ! grep -q "PHASE2G6_TEXTURE_COMPARISON_OK" "$COMPARE_DIR/phase2k1_compare_stdout.log"; then
  echo "COMPARE_STDOUT_MISSING_OK_TOKEN: $COMPARE_DIR/phase2k1_compare_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== REPORTS ====="
echo "CHECKPOINT: $FOUND_CKPT"
echo "LOAD_REPORT: $LOAD_DIR/load_only_report.md"
echo "DELTA_REPORT: $DELTA_DIR/base_vs_truepbr200_delta.md"
echo "COMPARE_REPORT: $COMPARE_DIR/base_vs_finetuned_report.md"
echo "COMPARE_METRICS: $COMPARE_DIR/base_vs_finetuned_metrics.json"
echo "COMPARE_BOARD: $COMPARE_DIR/base_vs_finetuned_texture_board.jpg"
ls -lh \
  "$FOUND_CKPT" \
  "$LOAD_DIR/load_only_report.md" \
  "$DELTA_DIR/base_vs_truepbr200_delta.md" \
  "$COMPARE_DIR/base_vs_finetuned_report.md" \
  "$COMPARE_DIR/base_vs_finetuned_metrics.json" \
  "$COMPARE_DIR/base_vs_finetuned_texture_board.jpg"

echo "PHASE2K1_TRUEPBR200_TRAIN_EVAL_OK"
echo "===== JOB END: SUCCESS ====="
date
