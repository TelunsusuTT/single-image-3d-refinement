# Phase 2N Week 2 Controlled Pilot Training

## Status

This document prepares the shared PC-S1 and PC-Full controlled-pilot runner.
**No Week 2 training has started.** The code and frozen evaluation manifest
must pass local review before the A100 job is submitted manually.

## Goal

The pilot asks a narrow question: does updating only the multi-view attention
projections (PC-S1) behave differently from the official broad partial
trainable scope (PC-Full) when every other training choice is held fixed?

Both scopes use the same 80 training assets, the same deterministic sample
order, the same reference decisions, the same no-augmentation reader, and the
same 320-update duration. PC-S1 runs first because it is the smaller,
lower-memory scope and provides the earliest fail-closed check before the
broader PC-Full run.

This stage performs no validation or test inference during training. It also
does not replace the existing rendered-view evaluation tools.

## Runtime Reuse Decision

The reusable functions in
`scripts/phase2n_day5_a100_preflight.py` are safe to import: that module is
standard-library-only at import time, has a normal main guard, and imports
Torch, OmegaConf, and Hunyuan only inside runtime functions.

The Week 2 runner therefore directly reuses the Day 5 true-PBR initializer,
production-loss call, deterministic RNG reset, batch transfer, and optimizer
membership audit. No shared helper extraction was needed, and the official
initialization or Hunyuan loss formula was not copied.

At runtime, each scope starts from a separately loaded official true-PBR base.
The runner calls the Day 5 production path rather than reconstructing the loss.

## Frozen Training Contract

Configuration:
`configs/phase2n_week2_pilot_training.json`

| Setting | Value |
| --- | --- |
| Train split | Existing full80 absolute examples JSON |
| Train assets | Exactly 80 |
| Batch size / workers | 1 / 0 |
| Updates | 320 |
| Epochs | Four complete deterministic 80-asset epochs |
| Base / schedule seed | 42 / 42 |
| Image size | 512 |
| Spatial augmentation | None |
| Precision | bf16 mixed precision |
| Warmup | 50-step linear warmup, then constant |
| Gradient clipping | 1.0 |
| Checkpoints | After updates 160 and 320 |
| Scope order | PC-S1, then PC-Full |
| PC-S1 peak LR | 1e-6 |
| PC-Full peak LR | 5e-7 |
| Runtime hardware | One A100 with at least 70 GiB |

The learning rates differ because Day 5 measured very different trainable
scope sizes and update magnitudes. PC-S1 updates 49,574,080 parameters; PC-Full
updates 1,046,761,668. The reviewed conservative pilot rates are safe runtime
settings, not proven optima.

The reference reader retains the Day 3 probabilities:

| Reference view | Probability |
| --- | ---: |
| 005 | 0.50 |
| 004 | 0.30 |
| 000, 001, 002, 003 | 0.05 each |

All six target views remain in canonical order 000 through 005.

## Shared Deterministic Schedule

Before either model is loaded, the runner creates one
`training_schedule.json`. It has exactly 320 records. For each of four
zero-based epochs, a deterministic permutation contains every one of the 80
assets exactly once.

Every record fixes:

- one-based global optimizer update;
- zero-based epoch and within-epoch position;
- original asset index, ID, and absolute sample path; and
- a deterministic production-loss seed.

The same schedule file drives both scopes. Immediately before the official
production loss, the reused Day 5 helper resets Python, NumPy, Torch CPU, and
Torch CUDA RNG state from the recorded step seed. The Day 3 reader uses its own
deterministic asset/epoch decision for the reference view and lighting pair.

This matters because a comparison is only interpretable when PC-S1 and PC-Full
see the same stochastic problem. At the end, the runner requires byte-identical
sampling-trace hashes and separately checks asset order plus reference
view/light decisions.

Only the six historical model keys enter the Hunyuan training step.
`protocol_metadata` remains outside the model batch and is retained in the
sampling trace.

## Training Guards

At every update the runner:

1. sets the protocol dataset epoch explicitly;
2. fetches the scheduled asset and collates a one-item batch;
3. verifies the exact scheduled path and no-augmentation metadata;
4. checks that live trainable names have not drifted;
5. checks optimizer identities exactly equal the trainable identities;
6. invokes the Day 5 production loss under bf16 autocast;
7. requires a finite scalar loss and present, finite, nonzero gradients;
8. rejects any gradient on a frozen tensor;
9. records the pre-clip gradient norm and clips to 1.0;
10. applies one optimizer step and exactly one scheduler step; and
11. records loss, LR, clipping, CUDA memory, asset, epoch, reference view, and
    lighting pair.

NaN, Inf, empty gradients, scope drift, optimizer mismatch, unexpected
augmentation, or sample-order drift is a hard failure.

## Checkpoint Contract

Each scope retains only:

- `step_160_scope_state.pt`; and
- `step_320_scope_state.pt`.

These are plain mappings of exact trainable parameter names to tensors in
their original dtype. They contain no optimizer state, scheduler state, frozen
parameter, or complete model. They live only under the new Week 2 output root,
never under `checkpoints/` or the official Hunyuan tree.

Each checkpoint is written to a temporary sibling and atomically renamed. Its
manifest records exact names, tensor count, numel, dtype, byte size, SHA-256,
scope report, official base identifier, update, LR, schedule hash, and
sampling-trace hash. The runner reloads the retained file and requires its keys
to exactly equal the trainable scope.

PC-Full is a broad partial scope, but its retained state still excludes every
parameter the official model marked frozen. It is not a full Hunyuan model
save.

## Stage-Level Rerun Safety

Each scope gets its own `_SUCCESS` marker. A repeated `--run-all` with the
same run ID skips a scope only when both checkpoints, both manifests, all 320
metric and trace rows, hashes, summaries, and the success marker validate.

An existing incomplete or inconsistent scope is never overwritten and has no
optimizer-resume path. Use a new run ID after diagnosing such a failure. This
keeps partial evidence from being mistaken for a completed pilot.

The final run-level `_SUCCESS` appears only after both scope outputs validate
and trace equivalence passes.

## Frozen Eight-Case Evaluation Pilot

Configuration:
`configs/phase2n_week2_pilot_eval_cases.json`

The cases were selected before Week 2 training using only the completed
corrected-input base/full80 rendered-view evidence. All use input view 005 and
AL lighting. No test case is present.

| Asset | Split | Stratum | Objective rationale |
| --- | --- | --- | --- |
| B073P1D981 | val | Non-front leakage risk | Largest full80-minus-base non-front MAE regression (+0.228972) |
| B075YLXSJC | val | Non-front leakage risk | Second-largest non-front regression (+0.162636) |
| B075YLQTNP | val | Front/reference complexity | Highest remaining base front MAE (18.614899) |
| B075YM2VXJ | val | Front/reference complexity | Second-highest remaining base front MAE (13.847439) |
| B073P52NDX | val | Historical median | Closest remaining case below validation all-view base median |
| B073P1S8VZ | val | Historical median | Closest remaining case above validation all-view base median |
| B073P16J7Y | train_sanity | Deterministic sanity | First lexicographically sorted existing train-sanity ID |
| B073P1H786 | train_sanity | Deterministic sanity | Second lexicographically sorted existing train-sanity ID |

The historical validation median base all-view MAE used for the final stratum
was 5.875447. Mesh and view-005 AL reference paths point to existing project
artifacts.

Test is excluded because it must remain an untouched final reporting split.
The six validation cases can support controlled pilot comparison, while the
two train-sanity cases reveal whether either scope learned an obvious
training-domain signal. Neither category establishes generalization alone.

## Expected Output Layout

```text
outputs/phase2n/week2_pilot_training/<run_id>/
  00_RUNTIME_MANIFEST.json
  training_schedule.json
  pc_s1/
    scope_report.json
    training_metrics.jsonl
    sampling_trace.jsonl
    checkpoints/
      step_160_scope_state.pt
      step_160_manifest.json
      step_320_scope_state.pt
      step_320_manifest.json
    summary.json
    summary.md
    _SUCCESS
  pc_full/
    ...
  comparison/
    trace_equivalence.json
  summary.json
  summary.md
  _SUCCESS
```

## Local Readiness

These checks do not import Hunyuan, load weights, or require CUDA:

```bash
python -m compileall src scripts tests
python scripts/phase2n_week2_train_pilots.py \
  --config configs/phase2n_week2_pilot_training.json \
  --check-only
bash -n env/run_phase2n_week2_train_pilots_a100.sbatch
```

Expected readiness token:

```text
PHASE2N_WEEK2_PILOT_TRAINING_READINESS_OK
```

Before manual submission, review:

- the frozen eight-case manifest contains six val and two train-sanity cases;
- no test path appears in the training config, schedule source, or manifest;
- the output filesystem has at least 15 GiB free;
- the requested GPU is an A100 with at least 70 GiB;
- the working tree contains only intended Stage 1 changes; and
- the generated run ID does not collide with an incomplete historical run.

## Manual Submission

After review, the user may submit:

```bash
sbatch env/run_phase2n_week2_train_pilots_a100.sbatch
```

Codex does not submit this job. The sbatch performs check-only first, then runs
PC-S1 followed by a fresh-base PC-Full. It runs no inference or evaluation and
prints `===== JOB END: SUCCESS =====` only after the runner succeeds.

At the time this document was written, no Week 2 model was loaded, no GPU was
used, and no training had started.
