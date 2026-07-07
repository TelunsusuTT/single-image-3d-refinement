# Phase 2M.3A LoRA Inference Smoke Protocol

## Purpose

Phase 2M.3A is the first runtime check that the trained `ref_dino` LoRA adapter can be injected into the original Hunyuan3D-Paint inference UNet and used for corrected-input inference. It compares one case only:

1. corrected-input base inference
2. corrected-input LoRA inference at adapter scale `0.75`

This is a load/inference smoke, not the full 24-case Phase 2M evaluation.

## Why One Case First

M2 proved adapter-only training can finish and save adapter states, but it did not prove the inference path can inject and load those adapters into the official base pipeline. A single corrected-input case is enough to catch:

- adapter state/key mismatch
- wrong LoRA backend
- wrong target list
- unsafe full-model save/merge behavior
- output layout problems
- Hunyuan runtime failures before launching a full evaluation

Full eval should wait until this smoke writes both base and LoRA outputs successfully.

## Inputs

Adapter checkpoint:

```text
outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_final.pt
```

Adapter config:

```text
outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_config.json
```

Corrected-input cases:

```text
outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/eval_cases.json
```

The selected smoke case must use `selected_input_view=005`, matching the corrected-input full80 evaluation protocol.

## Output

Default output root:

```text
outputs/phase2m/lora_infer_smoke_scale075/
```

Expected outputs include:

```text
outputs/phase2m/lora_infer_smoke_scale075/base/<split>/<case_id>/base_textured_mesh.glb
outputs/phase2m/lora_infer_smoke_scale075/lora_scale075/<split>/<case_id>/lora_scale075_textured_mesh.glb
outputs/phase2m/lora_infer_smoke_scale075/smoke_summary.json
```

OBJ outputs are also acceptable if GLB creation fails but the official pipeline still writes a valid textured OBJ.

## Pass/Fail Criteria

Pass means:

- readiness checker passes
- base corrected-input inference writes OBJ and/or GLB
- LoRA scale `0.75` inference writes OBJ and/or GLB
- `smoke_summary.json` has `status=OK` and `success=true`
- stdout prints `PHASE2M_M3A_LORA_INFER_SMOKE_OK`

Fail means any missing adapter/config/case input, adapter mismatch, unsafe output path, Hunyuan runtime error, or missing base/LoRA output.

## Safety

Phase 2M.3A uses `local_linear_fallback`, matching M2 training. It does not use PEFT. It must not merge LoRA into the base model and must not call `save_pretrained` on a full Hunyuan model. All outputs stay under project-local `outputs/phase2m/`.

Do not edit or write under `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work`.
