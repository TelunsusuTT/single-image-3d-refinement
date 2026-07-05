# Phase 2L.6A Render Full101 Hunyuan3D-Paint Examples

## Goal

Phase 2L.6A prepares the full101 `datav2_frame_panels` split as official
Hunyuan3D-Paint-style training examples.

Mini40 true-PBR evaluation showed a stable checkpoint and train-sanity learning
signal, but held-out validation/test front views did not clearly improve over
corrected-input base. The next step is to scale data before increasing mini40
training steps.

This phase does not run Hunyuan, submit Slurm jobs, train, load checkpoints,
install packages, or run Blender inside Codex. Blender rendering is a manual
user action.

## Expected Splits

- train: 80
- val: 10
- test: 11
- all: 101

## Expected Output Structure

Dataset root:

```text
data/hy3dpaint_train_examples/datav2_frame_panels_full101
```

Each sample directory should contain:

- `render_tex/`
- `render_cond/`
- `render_tex/transforms.json`
- six views `000` through `005`
- five `render_tex` images per view: RGB, albedo, MR, normal, position
- three `render_cond` lighting images per view: `AL`, `ENVMAP`, `PL`

Examples JSON files:

- `examples_train_abs.json`
- `examples_val_abs.json`
- `examples_test_abs.json`
- `examples_all_abs.json`

## Pass/Fail Criteria

Pass requires:

- render plan contains 101 assets
- split counts are 80/10/11
- all referenced GLBs exist before Blender rendering
- every successful sample has `render_tex/` and `render_cond/`
- every expected view file exists
- `render_tex/transforms.json` exists
- images are 512x512 when Pillow is available for checking
- `examples_all_abs.json` matches the train/val/test union

## Next Phase

After full101 rendering and local checks pass, the next phase is full80
true-PBR 500-step A100 training readiness. Do not start A100 training until the
full101 rendered dataset checker and official-style strict checker pass.
