#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2h1-eval50
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2h1_eval50_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CHECKPOINT="$PROJ/checkpoints/pilot_v1_conservative_50_lr1e6/pilot_v1_conservative_50_lr1e6-stepstep=50.ckpt"
CASE_DIR="$PROJ/outputs/phase2g/infer_cases/B075YLTF7Q"
BASE_DIR="$PROJ/outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke"
LOAD_DIR="$PROJ/outputs/phase2h/load_only/pilot_v1_conservative_50_lr1e6"
INFER_DIR="$PROJ/outputs/phase2h/infer_runs/B075YLTF7Q/pilot_v1_conservative_50_lr1e6_noremesh"
COMPARE_DIR="$PROJ/outputs/phase2h/compare/B075YLTF7Q/pilot_v1_conservative_50_lr1e6"

mkdir -p "$PROJ/logs/slurm"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CHECKPOINT=$CHECKPOINT"
echo "CASE_DIR=$CASE_DIR"
echo "BASE_DIR=$BASE_DIR"
echo "LOAD_DIR=$LOAD_DIR"
echo "INFER_DIR=$INFER_DIR"
echo "COMPARE_DIR=$COMPARE_DIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2H.1 EVAL READINESS ====="
python "$PROJ/scripts/check_phase2h1_eval_readiness.py" \
  --case-dir "$CASE_DIR" \
  --base-dir "$BASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --hypaint "$HYPAINT" \
  --load-output-dir "$LOAD_DIR" \
  --infer-output-dir "$INFER_DIR" \
  --compare-output-dir "$COMPARE_DIR"

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2H1_CUDA_PY'
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
PHASE2H1_CUDA_PY

echo "===== STAGE 1: LOAD-ONLY CHECK ====="
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

echo "===== STAGE 2: FINE-TUNED NO-REMESH INFERENCE ====="
mkdir -p "$INFER_DIR"
python "$PROJ/scripts/run_phase2g_paint_infer.py" \
  --case-dir "$CASE_DIR" \
  --output-dir "$INFER_DIR" \
  --mode finetuned \
  --checkpoint "$CHECKPOINT" \
  --max-num-view 6 \
  --resolution 512 \
  --device cuda \
  --no-remesh 2>&1 | tee "$INFER_DIR/phase2h1_infer_stdout.log"

if ! grep -q "PHASE2G5_FINETUNED_CHECKPOINT_LOADED_OK" "$INFER_DIR/phase2h1_infer_stdout.log"; then
  echo "INFER_STDOUT_MISSING_CHECKPOINT_LOAD_TOKEN: $INFER_DIR/phase2h1_infer_stdout.log" >&2
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

echo "===== STAGE 3: BASE VS CONSERVATIVE COMPARISON ====="
mkdir -p "$COMPARE_DIR"
python "$PROJ/scripts/make_phase2g6_texture_comparison.py" \
  --case-dir "$CASE_DIR" \
  --base-dir "$BASE_DIR" \
  --finetuned-dir "$INFER_DIR" \
  --output-dir "$COMPARE_DIR" 2>&1 | tee "$COMPARE_DIR/phase2h1_compare_stdout.log"

if ! grep -q "PHASE2G6_TEXTURE_COMPARISON_OK" "$COMPARE_DIR/phase2h1_compare_stdout.log"; then
  echo "COMPARE_STDOUT_MISSING_OK_TOKEN: $COMPARE_DIR/phase2h1_compare_stdout.log" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== REPORTS ====="
echo "LOAD_REPORT: $LOAD_DIR/load_only_report.md"
echo "COMPARE_REPORT: $COMPARE_DIR/base_vs_finetuned_report.md"
echo "COMPARE_METRICS: $COMPARE_DIR/base_vs_finetuned_metrics.json"
echo "COMPARE_BOARD: $COMPARE_DIR/base_vs_finetuned_texture_board.jpg"
ls -lh \
  "$LOAD_DIR/load_only_report.md" \
  "$COMPARE_DIR/base_vs_finetuned_report.md" \
  "$COMPARE_DIR/base_vs_finetuned_metrics.json" \
  "$COMPARE_DIR/base_vs_finetuned_texture_board.jpg"

echo "===== JOB END: SUCCESS ====="
date
