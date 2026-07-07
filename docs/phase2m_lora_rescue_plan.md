# Phase 2M LoRA Rescue Plan

## Goal

Phase 2M prepares a conservative Hunyuan3D-Paint LoRA rescue experiment. The
first target is a zero-LoRA smoke and module inventory, not training. The goal
is to confirm that a small adapter can be attached to the custom Hunyuan
multiview UNet without modifying upstream source files, upstream model weights,
or existing checkpoints.

## Safety Rules

- `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work` remains read-only.
- Do not edit original Hunyuan source files.
- Do not save logs, checkpoints, adapters, or generated files into
  `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work`.
- Do not merge LoRA weights into the base model.
- Do not call `save_pretrained` on the full Hunyuan model.
- Do not overwrite existing checkpoints.
- Adapter outputs must be adapter-only and live under `outputs/phase2m/`.

## Imperial A100 Workflow

Phase 2M runtime jobs use project `env/*.sbatch` files. Each sbatch sources:

```bash
source /vol/bitbucket/ct1022/hy3dpaint_finetune/env/env.sh
conda activate "$ENV_NAME"
```

Slurm logs go under `logs/slurm/`. Persistent outputs go under
`outputs/phase2m/`. Cache and temporary paths should follow `env/env.sh`; do not
write persistent artifacts under `/tmp`.

## First Target Modules

The first preset is `ref_dino`:

- `attn_refview.*.to_q`
- `attn_refview.*.to_k`
- `attn_refview.*.to_v`
- `attn_refview.*.to_out.0`
- `attn_dino.*.to_q`
- `attn_dino.*.to_k`
- `attn_dino.*.to_v`
- `attn_dino.*.to_out.0`

Only exact `torch.nn.Linear` modules are wrapped.

## Why Ref/DINO First

The current failure mode is front-to-back leakage and non-front contamination.
The reference-view and DINO projection paths are the most direct place to
adjust how the selected input image and visual features condition texture
generation while keeping the trainable footprint small.

## Why Multiview Is Deferred

`attn_multiview` is deferred because it is likely to affect cross-view
consistency more broadly. Targeting it first could make it harder to tell
whether any change comes from better reference conditioning or from broader
view-interaction changes. The first rescue should stay narrow.

The first implementation also excludes `attn1`, `attn2`, feed-forward layers,
convolutions, normalization layers, `learned_text_clip`, and DINO encoder
weights.

## Gates

### M0 Inventory

Run a module inventory on A100. Confirm that `ref_dino` selects a non-empty set
of exact Linear projections and excludes all forbidden groups.

Success token:

```text
PHASE2M_M0_INVENTORY_OK
```

### M1 Zero-LoRA Smoke

Inject rank-4 LoRA, verify that the adapter is a no-op at initialization, save
adapter-only weights, reload them, and confirm the upstream Hunyuan tree is
unchanged by size and mtime manifest.

Success token:

```text
PHASE2M_M1_ZERO_LORA_SMOKE_OK
```

The canonical M1 summary path is `outputs/phase2m/zero_lora_smoke/zero_lora_smoke_summary.json`.
For compatibility with early notes, checks also accept `outputs/phase2m/zero_lora_smoke/smoke_summary.json`, but should prefer the canonical filename when both exist.

### M2 Training Smoke

Only after M0/M1 pass, prepare a tiny adapter-only training smoke. It must save
adapter-only outputs under `outputs/phase2m/` and must not save a full model.

### M3 Evaluation

Only after a LoRA training smoke succeeds, evaluate corrected-input base versus
LoRA output with the same rendered-view protocol used in Phase 2L.

## Backend Policy

Prefer PEFT or Diffusers adapter APIs only if they are already installed and
can safely target the exact custom Hunyuan UNet modules. Do not install
packages. If exact safe targeting is unavailable or uncertain, use the local
`LoRALinear` fallback in this repo.

The backend used by each runtime smoke must be reported as one of:

- `peft`
- `diffusers`
- `local_linear_fallback`
