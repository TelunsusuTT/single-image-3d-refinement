# Phase 2M M1 Success Summary

Phase 2M M0 and M1 completed successfully. M1 was a zero-update LoRA smoke, not training and not inference.

## Confirmed Results

- M0 selected target count: 128
- M1 backend: `local_linear_fallback`
- PEFT available: false
- Diffusers available: true
- Trainable parameters: 829,952
- Total parameters: 1,963,076,424
- Trainable ratio: approximately 0.0423%
- Adapter state estimate: approximately 1.66 MB
- Official Hunyuan tree unchanged: yes
- Adapter-only save/load successful: yes
- Full checkpoint saved: no

## Output Hygiene

The canonical M1 summary is:

```text
outputs/phase2m/zero_lora_smoke/zero_lora_smoke_summary.json
```

For compatibility with earlier checks, this alias is also accepted:

```text
outputs/phase2m/zero_lora_smoke/smoke_summary.json
```

Future checks should prefer `zero_lora_smoke_summary.json` and fall back to `smoke_summary.json` only when the canonical file is absent.

## Next Step

The next step is M2 adapter-only training readiness. M2 should preserve the same safety constraints: adapter-only outputs, no full-model `save_pretrained`, no LoRA merge into the base model, and no overwriting old checkpoints.
