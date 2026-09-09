# 3D Generation and Texturing Pipelines

Research code for controlled Hunyuan3D-Paint adaptation and evaluation,
together with a TRELLIS.2 + MV-Adapter image-to-textured-GLB pipeline.

## Repository layout

- `src/hy3dft/` — datasets, adaptation, checkpoint handling, LoRA, inference,
  and view-selective conditioning.
- `configs/` — baseline, adaptation, gating, and evaluation specifications.
- `scripts/` — configuration-driven workflow and evaluation tools.
- `trellis2_mv_adapter/` — shape generation, mesh processing, multi-view
  synthesis, and texture baking.
- `data/` — portable manifests and dataset splits.
- `tests/` — lightweight validation tests.
- `docs/` — method, experiment, result, and reproducibility notes.
- `slurm/` — optional cluster entry points.

## Entry points

```bash
python scripts/train.py --help
python scripts/infer.py --help
python scripts/evaluate.py --help
python trellis2_mv_adapter/pipeline.py --help
```

Upstream source trees, model weights, source assets, checkpoints, generated
models, renders, and logs are kept outside Git. See
[THIRD_PARTY.md](THIRD_PARTY.md).
