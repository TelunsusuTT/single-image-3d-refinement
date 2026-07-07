# Phase 2M.3B LoRA Multi-Scale Pilot Protocol

## Purpose

Phase 2M.3B runs a small corrected-input LoRA inference pilot before any full 24-case evaluation. It uses the Phase 2M `ref_dino` adapter trained for 300 steps and compares three LoRA scales on exactly three full80 evaluation cases:

- one `val` case
- one `test` case
- one `train_sanity` or train-like sanity case

For each case it writes one corrected-input base output and LoRA outputs at scales `0.5`, `0.75`, and `1.0`.

## Why Not Full Eval Yet

M3A proved the adapter can load and produce one LoRA output at scale `0.75`. M3B checks whether scale choice is stable and worth evaluating before spending A100 time on all 24 cases and later rendered-view boards.

This pilot should catch:

- scale-specific runtime failures
- adapter key/load problems across repeated variants
- output layout problems for multiple splits
- obvious scale sensitivity before full evaluation

## Scale Accumulation Bug And Fix

The M3A helper originally set runtime scale by multiplying the current module scale:

```python
module.scale = module.scale * lora_scale
```

That is safe only when a fresh model is used once. It is unsafe for multi-scale evaluation in one Python process because applying `0.5` and then `0.75` would produce an effective multiplier of `0.375` instead of `0.75`.

The fixed behavior records the original LoRA module scale as `_phase2m_base_lora_scale` and sets scale absolutely:

```python
module.scale = module._phase2m_base_lora_scale * runtime_lora_scale
```

The M3B pilot also fresh-loads a Hunyuan pipeline and adapter for each LoRA scale, so scale accumulation is avoided by both policy and helper behavior.

## Inputs

Adapter:

```text
outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_final.pt
```

Adapter config:

```text
outputs/phase2m/lora_train_refdino_r4_lr5e5_300/adapter_config.json
```

Corrected-input eval cases:

```text
outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/eval_cases.json
```

M3A gate:

```text
outputs/phase2m/lora_infer_smoke_scale075/smoke_summary.json
```

## Output Structure

Default output root:

```text
outputs/phase2m/lora_multiscale_pilot/
```

Expected structure:

```text
base/<split>/<case_id>/
lora_scale050/<split>/<case_id>/
lora_scale075/<split>/<case_id>/
lora_scale100/<split>/<case_id>/
```

Summary files:

```text
outputs/phase2m/lora_multiscale_pilot/pilot_summary.json
outputs/phase2m/lora_multiscale_pilot/cases_used.json
outputs/phase2m/lora_multiscale_pilot/per_variant_outputs.json
```

## Pass/Fail Criteria

Pass means:

- readiness checker passes
- M3A smoke summary exists and is `OK`
- selected pilot cases include one val, one test, and one train-like sanity case
- all selected cases use `selected_input_view=005`
- base outputs are generated or reused
- LoRA outputs are generated or reused for scales `0.5`, `0.75`, and `1.0`
- `pilot_summary.json` has `status=OK`, `success=true`, and `case_count=3`
- stdout prints `PHASE2M_M3B_LORA_MULTISCALE_PILOT_OK`

Fail means any missing adapter/config/case input, wrong backend, wrong target count, unsafe output path, scale list mismatch, Hunyuan runtime error, or missing pilot summary.

## Safety

This phase uses `local_linear_fallback`. It must not use PEFT, merge LoRA into the base model, call `save_pretrained` on a full Hunyuan model, modify `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work`, or write outputs outside project-local `outputs/phase2m/`.

## Next Step

After M3B succeeds, the next step is a rendered-view pilot evaluation and visual board for these three cases. Only after that should the full 24-case LoRA evaluation be launched.
