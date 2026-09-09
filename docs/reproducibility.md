# Reproducibility

This document covers the Hunyuan3D-Paint workflows under `src/hy3dft/`,
`configs/`, and `scripts/`. The separate `trellis2_mv_adapter/` pipeline
uses its own upstream environments and weights.

## What is versioned

The repository versions the implementation, experiment descriptors, split and
evaluation contracts, public launchers, tests, and concise result summaries.
Large or machine-specific artefacts are intentionally kept outside Git:

- upstream Hunyuan3D 2.1 source and official model weights;
- upstream TRELLIS.2 and MV-Adapter source and model weights;
- ABO source assets and generated training examples;
- checkpoints, generated GLBs, renders, caches, and runtime logs;
- local environment credentials and scheduler output.

Third-party code, weights, and data remain governed by their respective terms.
See [Third-Party Components](../THIRD_PARTY.md).

## Prerequisites

A full Hunyuan3D-Paint run requires:

1. an external Hunyuan3D 2.1 checkout with the Hunyuan3D-Paint dependencies;
2. access to the official Hunyuan3D-Paint PBR weights;
3. the prepared framed-panel dataset and split manifests;
4. a compatible CUDA GPU environment for training or inference;
5. Blender only for the external fixed-view rerendering stage;
6. Slurm only when using the optional cluster launchers.

The upstream checkout must be treated as an external dependency. Project
outputs should remain under this repository's ignored runtime directories, not
inside the upstream source tree.

## Safe configuration checks

Each public entry point supports a check-only mode that validates the selected
descriptor without starting a training, inference, or rendering job:

```bash
python scripts/train.py \
  --config configs/adaptation/protocol_corrected_broad_finetuning.json \
  --check-only

python scripts/infer.py \
  --config configs/gating/view_selective_conditioning_gating.json \
  --check-only

python scripts/evaluate.py \
  --config configs/evaluation/fixed_view_rerendering.json \
  --check-only
```

Actual execution requires the explicit `--run` flag, a configuration whose
`runner.status` marks its canonical backend as available, and a complete
local runtime environment. This separation makes accidental heavyweight
execution less likely.

## Environment and Slurm launchers

Copy `.env.example` to `.env` or another ignored local file and set
`HUNYUAN3D_ROOT`. The Paint source defaults to
`${HUNYUAN3D_ROOT}/hy3dpaint`; set `HUNYUAN3D_PAINT_SOURCE_ROOT` only when
the upstream checkout uses a different layout.

```bash
cp .env.example .env
# Edit .env, then load it in the current shell.
source .env
```

The Slurm launchers cover the runtime backends marked `ready` in this public
snapshot:

```bash
sbatch slurm/infer.sbatch \
  configs/baseline/corrected_conditioning_baseline.json \
  /path/to/case \
  outputs/inference/baseline

sbatch slurm/evaluate.sbatch \
  configs/evaluation/fixed_view_rerendering.json \
  data/manifests/evaluation_cases.example.json \
  outputs/evaluation
```

Training descriptors and View-Selective Conditioning Gating can be validated
with `--check-only`; their canonical launch backends remain explicitly marked
`planned` rather than being represented as runnable.

The launchers resolve the project root from their own location. A deployment
may provide `PYTHON_BIN`, `BLENDER_BIN`, and an optional
`HY3DFT_ENV_FILE`. Cluster partitions, account names, environment locations,
and local data roots are deployment choices. Slurm's default `*.out` files
are ignored; deployments may provide their own `--output` path.

## Checkpoint contracts

The ready Broad-Scope inference backend accepts the complete UNet state stored
under the `unet.` prefix of a Lightning checkpoint and verifies exact key and
shape equality before loading. Protocol-corrected configurations use the
separate `trainable_scope_checkpoint_v1` manifest contract implemented in
`hy3dft.checkpoint`. The scope loader validates every tensor before copying
any parameter. A manifest's recorded checkpoint path is provenance rather than
a location lock: the supplied binary is authenticated by byte size, SHA-256,
exact keys, shapes, dtypes, and per-name element counts, so an intact pair can
be relocated. Protocol-corrected public inference runners remain `planned`
until that scope-only loader is connected end to end.

## Reproduction order

The scientific workflow is:

1. Verify the authoritative 80/10/11 asset split and prepared sample structure.
2. Validate the Corrected-Conditioning Baseline descriptor.
3. Run adaptation configurations using training assets only.
4. Select configuration/checkpoint settings from training-sanity and validation
   evidence.
5. Freeze the selected method and evaluation descriptor.
6. Run inference independently for each asset and method.
7. Rerender each GLB from the six fixed evaluation cameras.
8. Compute paired per-asset/per-view metrics and generate visual boards.
9. Report every final-test asset, including failures and regressions.

## Identity to record for every run

For an independently auditable result, retain:

- repository commit;
- upstream Hunyuan3D commit and official weight identity;
- canonical configuration and its content hash;
- dataset manifest and split identity;
- seed, conditioning-view metadata, target ordering, and augmentation policy;
- training scope, optimizer schedule, step, and checkpoint identity;
- inference scheduler, step count, guidance, resolution, and remeshing setting;
- generation-camera metadata and gate mask when gating is enabled;
- renderer cameras, lighting, foreground policy, and metric implementation;
- source and output hashes for reused or generated GLBs.

Paths recorded in portable manifests should be repository-relative or expressed
through explicit local root variables. Usernames and machine-specific absolute
paths are not part of the scientific identity.

## Determinism and isolation

The protocol-corrected adaptation runs use a fixed target order, deterministic
conditioning schedule, and no spatial augmentation. Shared examples drive the
Broad and MVA scopes. Inference comparisons initialise variants separately so
that checkpoint or runtime state cannot accumulate across methods. The ready
inference backend records the declared conditioning view and seed, resets
Python, NumPy, and Torch random generators immediately before each Paint call,
and refuses an existing output directory so that a stale GLB cannot satisfy a
new run.

The canonical evaluation descriptor owns the camera poses, render settings,
metric definitions, and the all/front/conditioning/non-front view groups.
Reference and generated images must match its declared resolution; a mismatch
fails the comparison instead of triggering an implicit resize.

The gating runtime validates the live camera layout and target tensor structure
before activation, checks that the intended branches were reached, verifies
selected-slot behaviour and base-weight sentinels, and restores the original
runtime state after each run. A failed structural check terminates the case
instead of silently changing the intervention.
