# Current Project State

## Summary

The project has reached the end of Phase 2K.4. The old texture/PBR collapse has
been traced to incorrect training initialization and fixed by using the local
official Hunyuan3D-Paint PBR weights.

## Initialization Fix

Earlier 50-step and 500-step checkpoints were initialized from an SD2/non-PBR
base, not from the official Hunyuan3D-Paint PBR inference weights. Phase 2J
located the local `hunyuan3d-paintpbr-v2-1` pipeline, verified exact training
initialization equivalence, and proved that official `train.py` can save a
base-equivalent checkpoint.

## Training Findings

- True-PBR 50-step at `lr=1e-6`: stable but very weak.
- True-PBR 200-step at `lr=1e-6`: stable and stronger than 50-step, with no
  return of the previous PBR collapse.
- The 200-step checkpoint does not consistently outperform corrected-input base
  inference.

## Phase 2K.3 Rendered-View Evaluation

Rendered-view evaluation showed that base and fine-tuned renders are much closer
to each other than to reference/training views. This made clear that UV texture
map comparison alone is not enough to claim visual quality improvement.

## Phase 2K.4 Reference-View Ablation

The user manually confirmed that views `004` and `005` are the
front/texture-informative views for the current pilot assets. Phase 2K.4 showed
that using these informative input views substantially improves rendered-view
fidelity for base inference. Input-view selection is therefore the dominant
bottleneck for the current pilot setting.

The true-PBR 200-step checkpoint remains stable, but it does not consistently
outperform the corrected-input base model.

## Current Decision

Proceed to Phase 2L: input-view-aware, subclass-specific, human-curated Data v2.

The next dataset should focus on a narrower visual category and explicitly store
`selected_input_view` metadata before any new training run. No Data v2 training
should begin until a corrected-input base benchmark exists.
