# Phase 2C Batch Rendering

Phase 2C batch-renders the Phase 2B-passed ABO assets into a pilot
Hunyuan3D-Paint-style training dataset. It uses the existing Phase 1E Blender
renderer and the existing local structure and framing checkers.

This phase prepares rendered training examples only. It does not train Hunyuan
and does not submit Slurm jobs.

## Inputs

Phase 2C starts from:

```text
data/candidates/phase2b_abo_download_manifest.csv
outputs/boards/phase2b_abo_inspection_summary.csv
data/raw_assets/phase2b_abo_selected/<source_id>.glb
```

Only assets marked as passing Phase 2B inspection should be rendered.

## Output Dataset

The output dataset root is:

```text
data/hy3dpaint_train_examples/pilot_v1/
```

Each rendered sample should be:

```text
data/hy3dpaint_train_examples/pilot_v1/<source_id>/
  render_tex/
  render_cond/
```

For `num_view=6`, each asset should contain:

- 6 views
- 30 `render_tex` PNG images
- 18 `render_cond` PNG images
- `render_tex/transforms.json`
- 49 expected files total

## QA

Run local QA after rendering:

- `scripts/check_phase1e_outputs.py`
- `scripts/check_render_framing.py`
- `scripts/check_hy3dpaint_example.py --strict`

Framing QA masks and reports are sidecar artifacts. They must live outside the
training data tree under:

```text
outputs/qa/framing/pilot_v1/
```

The training dataset directory should remain Hunyuan-style: `render_tex/`,
`render_cond/`, and the renderer metadata files needed by the dataloader.
