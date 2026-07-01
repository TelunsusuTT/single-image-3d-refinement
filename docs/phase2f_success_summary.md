# Phase 2F Success Summary

## Goal

Phase 2F tested a conservative 500-step tiny overfit run on the 7-sample pilot_v1 dataset. Unlike Phase 2E, this run was intended to save a real checkpoint for later base-vs-finetuned inference testing.

## Result

Phase 2F succeeded.

The run reached max_steps=500 and saved one checkpoint:

checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt

The checkpoint is ignored by Git through the existing checkpoints/ rule in .gitignore.

## Dataset

Dataset: pilot_v1  
Sample count: 7  
Examples JSON:

data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json

The dataset had previously passed:
- local rendered dataset check
- framing QA
- official Hunyuan-style strict checker

## Checkpoint

Checkpoint path:

checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt

Observed size:
- ls: about 9.5G
- du: checkpoints/pilot_v1_overfit_500 about 15G

This confirms that checkpoint saving works, but future experiments must carefully control checkpoint count and storage usage.

## Interpretation

This run confirms:
- pilot_v1 can train for 500 steps without crashing
- no checkpoint-saving failure remains
- the first fine-tuned checkpoint is available for Phase 2G

This run does not yet prove:
- visual improvement
- generalization
- final model quality

## Next Phase

Phase 2G should test whether this checkpoint can be loaded for Hunyuan3D-Paint inference and compared against the base model under the same mesh/reference/camera setup.
