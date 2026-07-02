# Phase 2I Pretrained Initialization Audit

Phase 2I audits whether the training checkpoints are numerically close to the
official Hunyuan3D-Paint PBR inference UNet. The goal is to determine whether
the Phase 2F and Phase 2H.1 checkpoints were initialized from the same official
paint PBR weights used by inference.

Strict key matching is not enough. A checkpoint can have the right key names and
tensor shapes, load with `strict=True`, and still be far from the official base
weights if training started from a different model family or checkpoint.

The suspected issue is that the training configs may use:

```text
sd2-community/stable-diffusion-2-1
resume_from: null
```

while inference uses the official Hunyuan3D-Paint PBR model:

```text
tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1
```

Non-goals:

- no training
- no inference
- no LoRA
- no Data v2

Expected outputs:

- config initialization report
- base-vs-checkpoint weight delta report
- recommendation for the next step

If the 50-step checkpoint already has large deltas from the official base UNet,
the likely next step is to build a training config that explicitly initializes
from the official Hunyuan3D-Paint PBR weights. If the 50-step checkpoint is very
close to the base UNet but inference still collapses, the next investigation
should look for high sensitivity, non-UNet mismatch, export/material issues, or
data-target mismatch.
