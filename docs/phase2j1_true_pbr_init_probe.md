# Phase 2J.1 True PBR Initialization Probe

Phase 2J.1 verifies whether a training model initialized from the local
`hunyuan3d-paintpbr-v2-1` pipeline directory matches the official inference base
UNet numerically.

This probe is needed because the Phase 2I audit showed that previous checkpoints
had compatible keys and shapes, but were far from the official base inference
UNet. Key compatibility alone does not prove the training run started from the
same PBR weights used by inference.

Strategy A for this probe is to set:

```text
stable_diffusion_config.pretrained_model_name_or_path:
/vol/bitbucket/ct1022/hy3dpaint_finetune/caches/hf/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1
```

Expected success:

- the training model can be instantiated from the project config
- the official inference base can be instantiated
- the best training UNet candidate has exact key and shape match to the
  inference base UNet
- weight deltas are tiny or near zero
- `PHASE2J1_TRUEPBR_INIT_COMPARE_OK`

Non-goals:

- no training
- no inference
- no checkpoint output
- no LoRA
- no Data v2

If the probe succeeds, the next step is a true-PBR 1-step or 50-step training
smoke. If no tiny-delta candidate is found, move to Strategy B: create a
training-compatible resume checkpoint from the official inference UNet.
