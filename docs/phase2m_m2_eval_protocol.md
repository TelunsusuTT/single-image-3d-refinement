# Phase 2M M2 Evaluation Protocol

Phase 2M M2 trains a small `ref_dino` LoRA adapter. Evaluation should remain cautious and compare against the strongest corrected-input baselines rather than the original weak-input baseline.

## Comparisons

Compare three conditions on the same assets and views:

- corrected-input base Hunyuan3D-Paint
- full80-500 true-PBR checkpoint
- Phase 2M `ref_dino` LoRA adapter

Use `selected_input_view=005` unless a later explicit review changes the selected input view for a case.

## Eval Set

Reuse the same 24 evaluation cases and 144 rendered views from the full80 evaluation protocol. This keeps the result comparable to the Phase 2L.7A/B full80-500 result.

## LoRA Scales

Evaluate at least these adapter scales:

- 0.5
- 0.75
- 1.0

Do not merge LoRA into the base model for evaluation. Adapter scale should be applied at runtime only.

## Primary Front Success Criterion

The primary signal is front-view improvement on views `004` and `005`:

- front `004/005` improved count must be greater than 24 out of 48
- preferably front `004/005` improved count should be at least 28 out of 48

Use rendered-view metrics as a screen, then inspect boards before making a claim.

## Leakage Watch

The main failure mode remains front-to-back or non-front texture leakage. The non-front visual board must not get worse. Do not claim leakage improvement from M2 unless the visual boards support it; metric movement alone is not enough.

## Reporting

Report base, full80-500, and LoRA side by side. Use cautious language: M2 is a rescue probe, not a final quality claim.
