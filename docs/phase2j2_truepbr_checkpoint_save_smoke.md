# Phase 2J.2 True-PBR Checkpoint Save Smoke

Phase 2J.2 verifies that the official `train.py` can save a checkpoint
initialized from the official Hunyuan3D-Paint PBR base. It runs one training
step with `base_learning_rate: 0.0`, saves exactly one checkpoint, then compares
that checkpoint against the official base inference UNet.

This differs from Phase 2J.1:

- Phase 2J.1 only instantiated the official inference model and training model.
- Phase 2J.2 runs official `train.py` for `max_steps: 1`.
- Phase 2J.2 saves one checkpoint and checks that it has not drifted from the
  official base weights.

Configuration:

- true PBR `pretrained_model_name_or_path`
- `max_steps: 1`
- `base_learning_rate: 0.0`
- exactly one weights-only checkpoint

Non-goals:

- no quality claim
- no real fine-tuning
- no inference
- no LoRA
- no Data v2

Expected success:

- training starts
- `max_steps=1` reached
- exactly one checkpoint saved
- checkpoint-vs-base numeric delta is zero or tiny
- `PHASE2J2_CHECKPOINT_SAVE_SMOKE_OK`
