# Phase 2M.3C LoRA Multi-Scale Pilot Rendered-View Evaluation

## Purpose

Phase 2M.3C evaluates the three-case Phase 2M.3B LoRA pilot in rendered-view space. M3B only proved that the base and LoRA adapter outputs could be generated for three corrected-input cases at scales `0.5`, `0.75`, and `1.0`. M3C renders those GLBs from the same six fixed views used by the full80 evaluation and compares the rendered views against the original full101 reference views.

This is still a pilot. It does not run the full 24-case LoRA evaluation.

## Why Rendered-View Evaluation

GLB generation success is not enough to judge texture quality. The textured mesh must be rendered from fixed views so that base and LoRA outputs can be compared against the same `render_cond/*_light_AL.png` references used in prior full80 evaluation. This catches view-dependent failures such as front-to-back leakage, non-front contamination, and scale choices that look plausible from one angle but degrade other views.

## Inputs

- M3B summary: `outputs/phase2m/lora_multiscale_pilot/pilot_summary.json`
- M3B variant outputs: `outputs/phase2m/lora_multiscale_pilot/per_variant_outputs.json`
- Variants: `base`, `lora_scale050`, `lora_scale075`, `lora_scale100`
- Cases: `B073P1D981` val, `B073NZS586` test, `B073P16J7Y` train_sanity
- Selected input view: `005` for all cases

## Outputs

Rendered PNGs are written under:

```text
outputs/phase2m/lora_multiscale_pilot_rendered/
```

Metrics and boards are written under:

```text
outputs/phase2m/lora_multiscale_pilot_eval/
```

Key files:

- `outputs/phase2m/lora_multiscale_pilot_rendered/render_eval_cases.json`
- `outputs/phase2m/lora_multiscale_pilot_rendered/render_summary.json`
- `outputs/phase2m/lora_multiscale_pilot_eval/metrics_rows.csv`
- `outputs/phase2m/lora_multiscale_pilot_eval/aggregate_summary.json`
- `outputs/phase2m/lora_multiscale_pilot_eval/scale_comparison.json`
- `outputs/phase2m/lora_multiscale_pilot_eval/boards/`

## Metrics

M3C reuses the existing rendered-view metric path where possible:

- MAE
- RMSE
- simple SSIM-like score
- base-versus-variant MAE
- LoRA improvement over corrected-input base

Negative `variant_minus_base_mae` means the LoRA variant is closer to the reference than corrected-input base. Positive `variant_minus_base_ssim_like` means the LoRA variant has a higher SSIM-like score than base.

## View Splits

The aggregate separates:

- all views `000`-`005`
- selected input view `005`
- front views `004`/`005`
- non-front views `000`-`003`
- each split and each case

Front-view improvement alone is not enough if non-front views visibly degrade.

## Pass/Fail Criteria

A successful M3C run means:

- readiness passes
- all 12 GLBs render into 72 PNGs
- metric aggregation completes
- one visual board is produced per case
- no output is written into the official Hunyuan tree

A useful pilot signal means at least one LoRA scale improves front views without obvious non-front degradation on the boards. If all LoRA scales are clearly worse than base, stop before the 24-case full evaluation.

## Next Decision

After M3C:

1. Pick the best scale for the 24-case full LoRA evaluation if a scale is promising.
2. Stop the LoRA branch if the pilot is clearly worse than corrected-input base.
3. If metrics are mixed, inspect boards before making any claim.
