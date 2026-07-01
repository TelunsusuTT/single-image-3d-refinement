# Phase 2H.1 Conservative Recovery

Phase 2H.1 creates a conservative 50-step recovery fine-tuning run on the
7-sample `pilot_v1` dataset. The goal is to test whether a much shorter run with
`base_learning_rate: 1e-6` can produce a clean checkpoint without repeating the
texture-collapse behavior seen after the 500-step probe.

This differs from Phase 2F in three important ways:

- Phase 2F ran 500 steps and produced a checkpoint that later showed degraded
  inference behavior.
- Phase 2H.1 runs only 50 steps with a lower learning rate.
- Phase 2H.1 starts fresh and must not resume from, load, or reference the
  `pilot_v1_overfit_500` checkpoint path.

This is not the final experiment. It does not claim visual quality, does not
compare base and fine-tuned outputs, does not tune hyperparameters broadly, and
does not resume from the collapsed checkpoint.

Expected success signs:

- `PHASE2H1_PREFLIGHT_OK`
- official strict checker passes on all 7 `pilot_v1` samples
- `CUBLAS_MATMUL_OK`
- `max_steps=50 reached`
- no NaN
- no CUDA OOM
- exactly one new checkpoint under
  `checkpoints/pilot_v1_conservative_50_lr1e6`
- `JOB END: SUCCESS`

Storage caution: this run should save exactly one weights-only checkpoint. Check
disk usage after the job and do not keep retry checkpoints in the same checkpoint
directory.
