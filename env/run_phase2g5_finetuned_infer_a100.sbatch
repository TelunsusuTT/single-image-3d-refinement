#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hy3dpaint-phase2g5-ft-infer
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/%x-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/hy3dpaint_phase2g5_ft_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
echo "JOB_TMPDIR=$JOB_TMPDIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CASE_DIR="$PROJ/outputs/phase2g/infer_cases/B075YLTF7Q"
OUTPUT_DIR="$PROJ/outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke"
CHECKPOINT="$PROJ/checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt"
WRAPPER="$PROJ/scripts/run_phase2g_paint_infer.py"

mkdir -p "$PROJ/logs/slurm"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "CASE_DIR=$CASE_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CHECKPOINT=$CHECKPOINT"
echo "WRAPPER=$WRAPPER"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== PHASE 2G.5 READINESS ====="
python "$PROJ/scripts/check_phase2g5_finetuned_infer_readiness.py" \
  --case-dir "$CASE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --wrapper "$WRAPPER" \
  --hypaint "$HYPAINT" \
  --output-dir "$OUTPUT_DIR"

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

echo "===== FINE-TUNED INFERENCE START ====="
python "$WRAPPER" \
  --case-dir "$CASE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mode finetuned \
  --checkpoint "$CHECKPOINT" \
  --max-num-view 6 \
  --resolution 512 \
  --device cuda \
  --no-remesh

echo "===== OUTPUT SEARCH ====="
mesh_count=0
while IFS= read -r -d '' mesh; do
  size="$(du -h "$mesh" | awk '{print $1}')"
  bytes="$(stat -c '%s' "$mesh")"
  echo "OUTPUT_MESH: $mesh size=$size bytes=$bytes"
  mesh_count=$((mesh_count + 1))
done < <(find "$OUTPUT_DIR" -type f \( -name "*.obj" -o -name "*.glb" \) -print0)

if [[ "$mesh_count" -lt 1 ]]; then
  echo "NO_OUTPUT_MESH_FOUND under $OUTPUT_DIR" >&2
  echo "===== JOB END: FAILED ====="
  date
  exit 1
fi

echo "===== JOB END: SUCCESS ====="
date
