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

- Save only one checkpoint with `save_top_k: 0` and `save_last: true`.
- Prefer `save_weights_only: true` when the official training stack supports it.
- Check disk usage after the job before moving on to larger experiments.
