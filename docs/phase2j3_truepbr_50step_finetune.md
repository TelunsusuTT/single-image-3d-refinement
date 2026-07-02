# Phase 2J.3 True-PBR 50-Step Fine-Tuning

Phase 2J.3 is the first real true-PBR fine-tuning run. It trains the
`pilot_v1` seven-sample dataset for 50 steps from the same local
`hunyuan3d-paintpbr-v2-1` weights that official inference uses.

This replaces the previous collapsed conservative checkpoints:

- the earlier 500-step and 50-step runs were not initialized from the official
  Hunyuan3D-Paint PBR base
- this config points `pretrained_model_name_or_path` at the local official
  `hunyuan3d-paintpbr-v2-1` pipeline
- Phase 2J.2 proved that this initialization path can save a checkpoint that is
  numerically equivalent to the official base before learning

Configuration:

- `max_steps: 50`
- `base_learning_rate: 1e-6`
- one weights-only checkpoint under
  `checkpoints/pilot_v1_truepbr_50_lr1e6`
- `every_n_train_steps: 50`
- `save_top_k: -1`
- `save_last: false`

Non-goals:

- no quality claim yet
- no inference in this phase
- no LoRA
- no Data v2

Expected success:

- strict example check passes
- `max_steps=50` is reached
- no NaN or CUDA OOM
- exactly one checkpoint is saved under
  `checkpoints/pilot_v1_truepbr_50_lr1e6`
- the checkpoint can be evaluated later against the official base
