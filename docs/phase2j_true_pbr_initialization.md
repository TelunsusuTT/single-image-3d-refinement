# Phase 2J.0 True PBR Initialization Planning

Phase 2J.0 plans how to make training start from the same official
Hunyuan3D-Paint PBR weights used by inference. This phase does not train,
does not run inference, and does not modify the official Hunyuan source tree.

Phase 2I showed why this is necessary. Both previous training configs used
`sd2-community/stable-diffusion-2-1`, had `resume_from: null`, and did not
reference `tencent/Hunyuan3D-2.1` or `hunyuan3d-paintpbr-v2-1`. The numeric
audit also found both training checkpoints far from the official base inference
UNet, including the 50-step conservative checkpoint.

Candidate strategies:

- Strategy A: point training `pretrained_model_name_or_path` to a local
  `hunyuan3d-paintpbr-v2-1` pipeline directory.
- Strategy B: create a training-compatible resume checkpoint from the official
  inference UNet weights.
- Strategy C: use a project-local training wrapper or patch only if Strategy A
  and Strategy B fail.

Non-goals:

- no training yet
- no LoRA yet
- no Data v2 yet
- no official source modification

Phase 2J.0 should first locate local official PBR weights, inspect the official
training/model initialization text paths, and then choose the least invasive
safe strategy for a later training smoke.
