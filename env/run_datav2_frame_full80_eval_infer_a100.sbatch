#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=datav2-frame-full80-eval
#SBATCH --chdir=/vol/bitbucket/ct1022/hy3dpaint_finetune
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/datav2_frame_full80_eval_truepbr500-%j.out
#SBATCH --error=/vol/bitbucket/ct1022/hy3dpaint_finetune/logs/slurm/datav2_frame_full80_eval_truepbr500-%j.err

set -eo pipefail

echo "===== JOB START ====="
date
hostname

source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"

export JOB_TMPDIR="/tmp/${USER}/datav2_frame_full80_eval_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HYPAINT:$HY21:${PYTHONPATH:-}"

CONFIG="$PROJ/configs/datav2_frame_full80_eval.json"
OUTPUT_ROOT="$PROJ/outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500"
CHECKPOINT="$PROJ/checkpoints/datav2_frame_full80_truepbr_500_lr1e6/datav2_frame_full80_truepbr_500_lr1e6-stepstep=500.ckpt"
CASE_TSV="$JOB_TMPDIR/eval_cases.tsv"

mkdir -p "$PROJ/logs/slurm"
mkdir -p "$OUTPUT_ROOT"

echo "===== PATHS ====="
echo "PROJ=$PROJ"
echo "HY21=$HY21"
echo "HYPAINT=$HYPAINT"
echo "ENV_NAME=$ENV_NAME"
echo "CONFIG=$CONFIG"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "CHECKPOINT=$CHECKPOINT"
echo "JOB_TMPDIR=$JOB_TMPDIR"

echo "===== NVIDIA SMI ====="
nvidia-smi

echo "===== CASE GENERATION ====="
python "$PROJ/scripts/make_datav2_frame_full80_eval_cases.py" \
  --config "$CONFIG"

echo "===== READINESS ====="
python "$PROJ/scripts/check_datav2_frame_full80_eval_readiness.py" \
  --config "$CONFIG"

echo "===== CUDA SANITY ====="
python3 - <<'PHASE2L7A_CUDA_PY'
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job")
a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
print("matmul sum:", float(c.sum().item()))
print("CUBLAS_MATMUL_OK")
PHASE2L7A_CUDA_PY

python3 - "$OUTPUT_ROOT/eval_cases.json" "$CASE_TSV" <<'PHASE2L7A_CASES_PY'
import json
import sys
from pathlib import Path

eval_cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_tsv = Path(sys.argv[2])
with out_tsv.open("w", encoding="utf-8") as handle:
    for case in eval_cases["cases"]:
        handle.write(
            "\t".join(
                [
                    case["eval_split"],
                    case["item_id"],
                    case["case_dir"],
                ]
            )
            + "\n"
        )
PHASE2L7A_CASES_PY

echo "===== BASE INFERENCE ====="
while IFS=$'\t' read -r eval_split item_id case_dir; do
  out_dir="$OUTPUT_ROOT/infer/base/$eval_split/$item_id"
  mkdir -p "$out_dir"
  echo "BASE $eval_split $item_id"
  python "$PROJ/scripts/run_phase2g_paint_infer.py" \
    --case-dir "$case_dir" \
    --output-dir "$out_dir" \
    --mode base \
    --max-num-view 6 \
    --resolution 512 \
    --device cuda \
    --no-remesh
  if [[ ! -f "$out_dir/base_textured_mesh.glb" && ! -f "$out_dir/base_textured_mesh.obj" ]]; then
    echo "missing base output for $eval_split/$item_id" >&2
    exit 1
  fi
done < "$CASE_TSV"
echo "PHASE2L7A_FULL80_BASE_INFER_OK"

echo "===== FINETUNED INFERENCE ====="
while IFS=$'\t' read -r eval_split item_id case_dir; do
  out_dir="$OUTPUT_ROOT/infer/finetuned/$eval_split/$item_id"
  mkdir -p "$out_dir"
  echo "FINETUNED $eval_split $item_id"
  python "$PROJ/scripts/run_phase2g_paint_infer.py" \
    --case-dir "$case_dir" \
    --output-dir "$out_dir" \
    --mode finetuned \
    --checkpoint "$CHECKPOINT" \
    --max-num-view 6 \
    --resolution 512 \
    --device cuda \
    --no-remesh
  if [[ ! -f "$out_dir/finetuned_textured_mesh.glb" && ! -f "$out_dir/finetuned_textured_mesh.obj" ]]; then
    echo "missing fine-tuned output for $eval_split/$item_id" >&2
    exit 1
  fi
done < "$CASE_TSV"
echo "PHASE2L7A_FULL80_FINETUNED_INFER_OK"

echo "===== UV TEXTURE COMPARISON ====="
while IFS=$'\t' read -r eval_split item_id case_dir; do
  python "$PROJ/scripts/make_phase2g6_texture_comparison.py" \
    --case-dir "$case_dir" \
    --base-dir "$OUTPUT_ROOT/infer/base/$eval_split/$item_id" \
    --finetuned-dir "$OUTPUT_ROOT/infer/finetuned/$eval_split/$item_id" \
    --output-dir "$OUTPUT_ROOT/texture_compare/$eval_split/$item_id"
done < "$CASE_TSV"

echo "PHASE2L7A_FULL80_EVAL_INFER_OK"
echo "===== JOB END: SUCCESS ====="
date
