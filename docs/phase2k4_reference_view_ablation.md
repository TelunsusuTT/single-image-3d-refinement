# Phase 2K.4 Reference-View Ablation

Phase 2K.4 tests whether using front, texture-informative input views improves
texture generation for the true-PBR 200-step checkpoint.

Phase 2K.3 showed that base and fine-tuned rendered views are close to each
other but still far from the reference views, especially on the front-facing,
texture-informative views. The user manually confirmed that views `004` and
`005` are the front/informative views for all four pilot assets.

This phase does not rerun input view `001`. The existing Phase 2K.3 rendered
evaluation is treated as the previous-current-input baseline. Phase 2K.4 focuses
only on input-view sensitivity for `004` and `005`.

Workflow:

- A100: run base and fine-tuned Hunyuan inference for input views `004` and `005`
- local gpu12: run rendered-view evaluation from generated configs

Non-goals:

- no training
- no new checkpoint
- no automatic front detector
- no LoRA
- no Data v2 yet

Expected decision:

- if input views `004` or `005` improve base substantially, input-view selection
  is a major bottleneck
- if not, model/data fidelity is the bottleneck
