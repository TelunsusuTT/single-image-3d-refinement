# Phase 2F Tiny Overfit

Phase 2F is the first real fine-tuning probe on the 7-sample `pilot_v1`
dataset. It runs a conservative 500-step tiny overfit and saves exactly one
final checkpoint.

Phase 2E proved that the official Hunyuan3D-Paint training loop can read the
custom multi-asset dataset and complete a short smoke run. Phase 2F goes one
step further: it tests a longer run, checks for numerical/runtime stability, and
verifies that checkpoint saving works without creating a pile of large files.

This is still not the final experiment. It should not be used to make a final
visual quality claim, a generalization claim, or a formal comparison against
other data or model settings.

Expected success signs:

- `max_steps=500` is reached.
- No NaN failure appears in the logs.
- No CUDA out-of-memory failure appears in the logs.
- Exactly one checkpoint is saved.
- The checkpoint path and size are recorded in the Slurm log.
- No unexpected large checkpoint set is produced.

Storage caution:

- Save one step-500 checkpoint in `checkpoints/pilot_v1_overfit_500`.
- Prefer `save_weights_only: true` when the official training stack supports it.
- Check disk usage after the job before moving on to larger experiments.

Checkpoint fix note:

The first 500-step Phase 2F run reached `max_steps=500` but produced no
checkpoint. The checkpoint interval was `every_n_train_steps: 1000000`, and
`save_top_k: 0` also disabled ordinary checkpoint saves. The patched setup saves
at step 500 with `every_n_train_steps: 500`, `save_top_k: -1`, and
`save_last: false`, so the post-run check expects exactly one new `.ckpt` under
`checkpoints/pilot_v1_overfit_500`.
