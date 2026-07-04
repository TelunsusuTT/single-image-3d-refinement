# Phase 2L.3A Render Mini40 Hunyuan3D-Paint Examples

## Goal

Phase 2L.3A converts the `datav2_frame_panels` mini40 split from raw ABO GLBs
into official Hunyuan3D-Paint-style training examples. The output is a dataset
root containing one sample directory per asset, each with `render_tex/` and
`render_cond/` folders plus camera metadata.

This phase prepares local training data only. It does not run Hunyuan, train,
submit Slurm jobs, load checkpoints, install packages, or run Blender inside
Codex.

## Why This Comes Before A100 Training

Official `train.py` expects Hunyuan3D-Paint training examples, not raw GLBs.
Rendering first lets us run local structure checks, strict example validation,
and framing QA before any A100 smoke or fine-tuning job.

## Why Mini40 First

The mini40 split is the fast diagnostic split: 32 train, 4 validation, and 4
test assets. Rendering mini40 first keeps iteration cheap while preserving a
held-out evaluation path. The full101 split should wait until mini40 rendering,
checking, and training smoke succeed.

## Expected Outputs

Dataset root:

```text
data/hy3dpaint_train_examples/datav2_frame_panels_mini40
```

Each sample should contain:

- `render_tex/000.png` through `005.png`
- `render_tex/000_albedo.png`, `000_mr.png`, `000_normal.png`, `000_pos.png`,
  and corresponding files for all six views
- `render_cond/000_light_AL.png`, `000_light_ENVMAP.png`, `000_light_PL.png`,
  and corresponding files for all six views
- `render_tex/transforms.json`

The examples JSON files are:

- `examples_train_abs.json`
- `examples_val_abs.json`
- `examples_test_abs.json`
- `examples_all_abs.json`

## Pass/Fail Criteria

Pass requires:

- render plan contains 40 mini40 assets
- expected train/val/test counts are 32/4/4
- every rendered sample has `render_tex/` and `render_cond/`
- every expected view file exists
- `render_tex/transforms.json` exists
- images are 512x512 when Pillow is available for checking

## Next Phase

The next phase is strict local checking and A100 mini40 training sbatch
preparation. Do not train before the mini40 rendered dataset passes local
structure checks and the existing official-style checker.
