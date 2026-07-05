# Phase 2L.6B Full80 True-PBR Training

Phase 2L.6B prepares the first full80 Data v2 frame-panel true-PBR training
run:

`datav2_frame_full80_truepbr_500_lr1e6`

This is a conservative scale-up from mini40. Phase 2L.5B showed that the
mini40 500-step checkpoint was stable and had a train-sanity learning signal,
but held-out val/test and corrected-input front views did not clearly beat the
base model. The next probe should therefore increase dataset coverage before
increasing the mini40 step count.

## Dataset

The training setup uses the already rendered full101 Hunyuan3D-Paint examples:

- train: `data/hy3dpaint_train_examples/datav2_frame_panels_full101/examples_train_abs.json`
- val: `data/hy3dpaint_train_examples/datav2_frame_panels_full101/examples_val_abs.json`
- test: `data/hy3dpaint_train_examples/datav2_frame_panels_full101/examples_test_abs.json`

Expected counts:

- train: 80
- val: 10
- test: 11

Each sample must keep the official training-directory shape:

- `render_tex/`
- `render_cond/`
- `render_tex/transforms.json`

## Training Configuration

The run uses true-PBR initialization from the local official PBR weights:

`/vol/bitbucket/ct1022/hy3dpaint_finetune/caches/hf/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1`

Core settings:

- `base_learning_rate: 1e-6`
- `max_steps: 500`
- `batch_size: 1`
- `num_view: 6`
- `view_size: 512`
- checkpoint every 500 train steps
- `save_top_k: -1`
- `save_last: false`
- `save_weights_only: true`

The checkpoint directory is isolated:

`checkpoints/datav2_frame_full80_truepbr_500_lr1e6`

The 500 optimizer steps are about 6.25 passes over the 80-sample training split.
This is still a pilot run, not a final experiment.

## Success Criteria

The setup is ready only when:

- train/val/test examples JSONs exist and have counts 80/10/11
- all JSON sample paths are absolute
- every sample has the expected Hunyuan-style rendered files
- the YAML points at the true-PBR source directory, not `sd2-community`
- the YAML references the full101 train and val examples JSONs
- checkpoint output points only to the full80 checkpoint directory
- readiness prints `PHASE2L6B_FULL80_TRAINING_READINESS_OK`
- sbatch syntax passes `bash -n`

The A100 job succeeds only if training reaches max steps and exactly one new
step-500 checkpoint is found under the isolated checkpoint directory.

## Non-Goals

Phase 2L.6B does not:

- run Hunyuan in Codex
- submit Slurm in Codex
- load or inspect checkpoint tensor contents
- run Blender
- make a final quality claim
- prepare a 1000-step run yet

## Next Step

If the 500-step full80 run is stable and produces exactly one checkpoint, the
next phase should evaluate corrected-input base versus full80 fine-tuned outputs
on val/test and train-sanity cases. Prepare a 1000-step run only if the full80
500-step result is stable but clearly under-adapted.
