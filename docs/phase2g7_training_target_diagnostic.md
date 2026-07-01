# Phase 2G.7 Training Target Diagnostic

Phase 2G.7 diagnoses whether the `pilot_v1` training targets contributed to
the observed PBR and texture collapse in the fine-tuned output.

Hunyuan3D-Paint training examples use `render_tex/*_mr.png` for
metallic/roughness supervision and `render_tex/*_albedo.png` for albedo/color
supervision. Since the Phase 2G.6 comparison showed severe output map changes,
we need to inspect the target distributions before changing checkpoint loading
or retraining.

Current suspicion:

- fine-tuned albedo is globally corrupted
- fine-tuned metallic output became much higher than base
- fine-tuned roughness also shifted upward
- manual inspection suggests the whole texture map degraded, not only MR

Questions:

- Are `pilot_v1` MR target channels high?
- Is channel packing suspicious?
- Are R/G/B channels plausible?
- Are albedo targets normal or already degraded?
- Is `B075YLTF7Q` representative or an outlier?

Non-goals:

- no retraining yet
- no checkpoint changes
- no quality claim
