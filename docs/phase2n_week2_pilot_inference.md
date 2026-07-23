# Phase 2N Week 2 Scope-Checkpoint Pilot Inference

## Status

The workflow is prepared for local validation and later manual A100 submission.
No Week 2 checkpoint inference has run yet, and no checkpoint-quality conclusion
exists.

## Goal

This stage reconstructs four controlled training variants on top of identical,
fresh official true-PBR bases and runs each one on the same frozen pilot cases:

- `pc_s1_step160`
- `pc_s1_step320`
- `pc_full_step160`
- `pc_full_step320`

The resulting GLBs will support checkpoint and scope selection before the more
expensive rendered-view evaluation. This stage does not train, run Blender,
calculate rendered metrics, or use test data.

## Reuse Decision

The historical one-case wrapper
`scripts/run_phase2g_paint_infer.py` was safe but could not retain one loaded
pipeline across eight cases or apply a trainable-scope-only checkpoint before
inference. A limited extraction was therefore necessary.

`src/hy3dft/hunyuan_inference.py` now contains only the reusable official
pipeline initialization and case-execution helpers. The old CLI calls those
helpers and keeps its historical dry-run, base, fine-tuned, and remesh behavior.
The new runner reuses the same fixed-mesh/no-remesh inference call; it does not
copy the official pipeline or introduce another inference algorithm.

## Frozen Cases

A frozen pilot case is an asset whose identity, split, mesh, and corrected input
image were committed before Week 2 training. Freezing the cases prevents the
evaluation set from drifting after seeing training artifacts.

`configs/phase2n_week2_pilot_eval_cases.json` contains exactly:

| Evaluation split | Count | Assets |
| --- | ---: | --- |
| Validation | 6 | `B073P1D981`, `B075YLXSJC`, `B075YLQTNP`, `B075YM2VXJ`, `B073P52NDX`, `B073P1S8VZ` |
| Train-sanity | 2 | `B073P16J7Y`, `B073P1H786` |
| Test | 0 | None |

Every case uses selected input view `005`, AL reference lighting, the original
mesh, resolution 512, seed 0, and fixed-mesh inference with remeshing disabled.
The six validation cases support pilot model selection. The two train-sanity
cases are diagnostic only: they can reveal whether a checkpoint learned an
obvious training-domain signal, but they must not select the final model.

The full set of 101 assets is not used for pilot selection because this stage is
a controlled, low-cost checkpoint gate, not a final dataset-wide quality claim.
The test split remains untouched for later reporting.

## Scope-Only Reconstruction

A scope checkpoint is not a standalone Hunyuan model. It contains only the
parameters that were trainable for PC-S1 or PC-Full. For each variant the runner:

1. loads a new official true-PBR base;
2. derives the exact expected live names for that scope;
3. validates the checkpoint manifest, byte size, and SHA-256;
4. loads the plain tensor mapping on CPU;
5. requires exact keys, tensor count, numel, shapes, dtypes, and finite values;
6. casts and copies each tensor to its live parameter under `torch.no_grad()`;
7. verifies loaded equality and an unchanged deterministic frozen sample; and
8. places the model in evaluation mode.

The loader rejects full-model, frozen-parameter, optimizer, and scheduler
payloads. Step 320 is never layered on step 160, and PC-Full is never layered on
PC-S1. A new base per variant is essential because otherwise the comparison
would measure accumulated checkpoint state rather than the four trained
variants.

PC-Full specifically means the subset preserved by the official training
initializer. Because an inference model may not retain those training-time
`requires_grad` flags, the loader reconciles the audited PC-Full trainable and
frozen name lists against the entire fresh inference keyspace before copying
anything. It does not infer PC-Full from inference defaults.

## Audited Checkpoints

Training run: `outputs/phase2n/week2_pilot_training/slurm_264123`

| Variant | Scope | Step | Tensors | Parameters | Bytes | SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `pc_s1_step160` | PC-S1 | 160 | 80 | 49,574,080 | 198,326,586 | `a61fa7e4ab11ceee5bfe7e208e557f32b01fa212b9fab1e3a55db317259348ed` |
| `pc_s1_step320` | PC-S1 | 320 | 80 | 49,574,080 | 198,326,586 | `fea99e0ae82d2ac0faa7c5b76ca7c2a116b9def90c15ee688834eea728601c4c` |
| `pc_full_step160` | PC-Full | 160 | 981 | 1,046,761,668 | 4,187,401,329 | `1ce474ba1246331b1d55152a1cafa4f08be6d9b8f746c18b4bf4701efe51f9fc` |
| `pc_full_step320` | PC-Full | 320 | 981 | 1,046,761,668 | 4,187,401,329 | `b93f4d34291a21097b8d763b3d5c18d4b0a82796049ce0711aaac552b310d18e` |

The completion gate is the root summary plus the root, PC-S1, and PC-Full
`_SUCCESS` markers. The historical runtime manifest still says `RUNNING`; that
stale field is recorded but does not override the completed summaries and
markers.

## Baseline Reuse

Baseline inference is not rerun. Check-only requires compatible output and run
plan coverage for all eight frozen cases at these exact roots:

- corrected-input base:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/infer/base`
- historical full80-500:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/infer/finetuned`
- historical full80 checkpoint:
  `checkpoints/datav2_frame_full80_truepbr_500_lr1e6/datav2_frame_full80_truepbr_500_lr1e6-stepstep=500.ckpt`

Reuse is valid only when each run plan records the same mesh and view-005 image,
resolution 512, six views, fixed mesh, and `use_remesh=false`. Reusing these
artifacts saves four unnecessary baseline reruns and keeps the comparison tied
to the already reviewed corrected-input protocol.

## Why Loss Is Not Selection

Training loss measures the optimization objective on training batches. It does
not directly measure whether generated texture placement is visually faithful
from front and non-front views, and its scale is not a fair quality ranking
across different trainable scopes. The four checkpoints therefore remain
scientifically unselected until inference and rendered-view evidence exist.

## Static Readiness

The static gate uses only the Python standard library. It does not import Torch
or Hunyuan, load model weights, require CUDA, or write runtime outputs:

```bash
python scripts/phase2n_week2_infer_pilots.py \
  --config configs/phase2n_week2_pilot_inference.json \
  --check-only
```

Expected final tokens:

```text
PHASE2N_WEEK2_SCOPE_CHECKPOINTS_OK
PHASE2N_WEEK2_FROZEN_CASES_OK
PHASE2N_WEEK2_BASELINE_REUSE_OK
PHASE2N_WEEK2_PILOT_INFERENCE_READINESS_OK
```

The check fails with the exact incompatible or missing case when baseline
coverage is incomplete.

## Runtime Contract

The later manual job requires one A100 with at least 70 GiB and at least 20 GiB
free project storage. Within each variant, the first frozen case is a smoke
gate; only after its GLB passes strict header and finite-value checks do the
remaining seven run. The same loaded variant serves all eight cases, then it is
released and CUDA cache is cleared before the next fresh base is loaded.

A rerun with the same run ID skips a completed variant only if all eight GLBs,
case manifests, hashes, summaries, and `_SUCCESS` marker validate. Any partial
or inconsistent existing variant fails closed and requires a new run ID.

Manual submission, after review:

```bash
sbatch env/run_phase2n_week2_infer_pilots_a100.sbatch
```

## Expected Outputs

```text
outputs/phase2n/week2_pilot_inference/<run_id>/
  00_RUNTIME_MANIFEST.json
  resolved_cases.json
  baseline_reuse_report.json
  checkpoint_validation/
    pc_s1_step160.json
    pc_s1_step320.json
    pc_full_step160.json
    pc_full_step320.json
  pc_s1_step160/
    <ASSET_ID>/
      textured_mesh.glb
      inference_manifest.json
    summary.json
    summary.md
    _SUCCESS
  pc_s1_step320/
  pc_full_step160/
  pc_full_step320/
  summary.json
  summary.md
  _SUCCESS
```

## Next Gate

After all 32 pilot GLBs validate, the later `gpu12` stage may render the frozen
views with Blender and compare front views 004/005, non-front views 000-003,
and all views against the reused base and full80 outputs. That stage, not this
readiness work, will determine whether either scope or checkpoint step merits
further evaluation.
