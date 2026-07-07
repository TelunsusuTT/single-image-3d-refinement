# Phase 2M LoRA Negative Result Report

## Experiment Goal

Phase 2M tested whether a small adapter-only LoRA update on the reference-conditioning attention path could improve corrected-input Hunyuan3D-Paint outputs without modifying the official Hunyuan source tree or saving a full model checkpoint.

## Setup

- Target modules: `attn_refview + attn_dino`
- Trainable parameters: `829,952`
- Total parameters: `3,099,557,192`
- Adapter size: `3.3 MB`
- Training steps: `300`
- Backend: `local_linear_fallback`
- Adapter-only safety: `merge_into_base=false`, `save_pretrained_full_model=false`
- Scales evaluated: `0.5`, `0.75`, `1.0`
- Pilot cases: B073P1D981 (val), B073NZS586 (test), B073P16J7Y (train_sanity)

The evaluated cases cover one validation asset, one test asset, and one train-sanity asset, all using selected input view `005`.

## All-View Table

| Variant | Views | Base MAE | Variant MAE | MAE Delta | Base SSIM-like | Variant SSIM-like | SSIM Delta | MAE Improved Views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 18 | 5.0146 | 5.0146 | 0.0000 | 0.9063 | 0.9063 | 0.0000 | 0 |
| LoRA 0.50 | 18 | 5.0146 | 5.0324 | 0.0178 | 0.9063 | 0.8940 | -0.0123 | 6 |
| LoRA 0.75 | 18 | 5.0146 | 5.3462 | 0.3316 | 0.9063 | 0.8739 | -0.0324 | 3 |
| LoRA 1.00 | 18 | 5.0146 | 5.7947 | 0.7801 | 0.9063 | 0.8370 | -0.0693 | 2 |

## Front 004/005 Table

| Variant | Views | Base MAE | Variant MAE | MAE Delta | Base SSIM-like | Variant SSIM-like | SSIM Delta | MAE Improved Views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 6 | 8.8970 | 8.8970 | 0.0000 | 0.8847 | 0.8847 | 0.0000 | 0 |
| LoRA 0.50 | 6 | 8.8970 | 9.1721 | 0.2751 | 0.8847 | 0.8717 | -0.0130 | 2 |
| LoRA 0.75 | 6 | 8.8970 | 10.2061 | 1.3091 | 0.8847 | 0.8487 | -0.0360 | 0 |
| LoRA 1.00 | 6 | 8.8970 | 11.2935 | 2.3966 | 0.8847 | 0.8240 | -0.0606 | 0 |

## Non-Front 000-003 Table

| Variant | Views | Base MAE | Variant MAE | MAE Delta | Base SSIM-like | Variant SSIM-like | SSIM Delta | MAE Improved Views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 12 | 3.0734 | 3.0734 | 0.0000 | 0.9172 | 0.9172 | 0.0000 | 0 |
| LoRA 0.50 | 12 | 3.0734 | 2.9626 | -0.1108 | 0.9172 | 0.9052 | -0.0120 | 4 |
| LoRA 0.75 | 12 | 3.0734 | 2.9162 | -0.1571 | 0.9172 | 0.8866 | -0.0306 | 3 |
| LoRA 1.00 | 12 | 3.0734 | 3.0453 | -0.0281 | 0.9172 | 0.8435 | -0.0736 | 2 |

## Split Table

| Split | Variant | Views | Base MAE | Variant MAE | MAE Delta | SSIM Delta | MAE Improved Views |
|---|---|---:|---:|---:|---:|---:|---:|
| val | LoRA 0.50 | 6 | 4.2283 | 4.7392 | 0.5109 | -0.0392 | 0 |
| val | LoRA 0.75 | 6 | 4.2283 | 5.5329 | 1.3046 | -0.0993 | 0 |
| val | LoRA 1.00 | 6 | 4.2283 | 6.1263 | 1.8980 | -0.1430 | 0 |
| test | LoRA 0.50 | 6 | 5.3359 | 5.5967 | 0.2608 | -0.0136 | 0 |
| test | LoRA 0.75 | 6 | 5.3359 | 5.9294 | 0.5935 | -0.0231 | 0 |
| test | LoRA 1.00 | 6 | 5.3359 | 6.2946 | 0.9587 | -0.0692 | 0 |
| train_sanity | LoRA 0.50 | 6 | 5.4795 | 4.7613 | -0.7182 | 0.0158 | 6 |
| train_sanity | LoRA 0.75 | 6 | 5.4795 | 4.5763 | -0.9033 | 0.0252 | 3 |
| train_sanity | LoRA 1.00 | 6 | 5.4795 | 4.9632 | -0.5163 | 0.0043 | 2 |

## Qualitative Board Observations

The visual boards show scale-dependent front-to-back leakage. Higher LoRA scales increase visible drift on front/input views and do not produce a reliable held-out improvement over corrected-input base. The only consistent metric improvement appears on the train_sanity case, which is compatible with overfitting or narrow domain-specific drift rather than a robust improvement.

Board paths are included in the report packet under `boards/`.

## Final Conclusion

`ref_dino` LoRA is not selected for full 24-case expansion.

Decision string: `stop_full_lora_eval`

All LoRA scales degrade held-out val/test and front views versus corrected-input base; train_sanity-only gains suggest overfitting or domain-specific drift.

## Recommended Future Work

- LoRA with an explicit non-front/base preservation objective.
- Visibility-aware loss so the model is not rewarded for leaking front texture onto unseen surfaces.
- Avoid a plain reference-conditioning adapter as the only trainable path; pair it with constraints or supervision that preserve corrected-input base behavior.
