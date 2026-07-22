# Phase 2N Day 5: A100 Runtime Preflight

## Purpose

Day 5 is a real-model runtime audit for the two Phase 2N trainable scopes. It
loads the same local official true-PBR base used by the successful historical
full80 run, reads one real protocol-corrected batch, and performs one optimizer
update for PC-S1 followed by one optimizer update for PC-Full.

This is **not formal training**. It is not a 320-step run, does not measure
texture quality, and cannot support a model-quality conclusion. Week 2 remains
unauthorized until a human reviews the resulting gradients, update magnitudes,
and memory use.

## Why An A100 Audit Is Needed

Day 3 and Day 4 tests prove the reader, trainable-scope selection, optimizer
membership, and scheduler behavior using local synthetic or lightweight
models. Those tests intentionally do not prove that the complete official
Hunyuan model:

- loads from the intended true-PBR pipeline directory;
- accepts the six-key protocol-corrected batch;
- produces a finite production loss under bf16 autocast;
- propagates useful gradients through the exact selected tensors;
- fits in A100 memory for each scope; or
- can restore the selective PC-S1 checkpoint format.

The Day 5 job answers those runtime questions with the smallest useful real
audit: one deterministic batch and exactly one peak-learning-rate update per
scope.

## Fixed Contract

The configuration is
[`configs/phase2n_day5_a100_preflight.json`](../configs/phase2n_day5_a100_preflight.json).
It pins:

- the full80 training JSON with exactly 80 absolute sample paths;
- sample index `0`, batch size `1`, workers `0`, seed `42`, and epoch `0`;
- `spatial_augmentation = none` for both scopes;
- six 512-pixel views and the historical six model keys;
- warmup for 50 optimizer steps;
- candidate peak LR `1e-6` for PC-S1 and `5e-7` for PC-Full; and
- the only permitted report root,
  `outputs/phase2n/day5_a100_preflight/`.

The script validates the committed historical YAML and then reuses its model
target, local `hunyuan3d-paintpbr-v2-1` path, custom pipeline, 12-channel input
setting, and null resume/control initialization. It reproduces the official
`train.py` input-convolution expansion and calls
`HunyuanPaint.training_step` directly. No Hunyuan loss formula is copied into
project code.

Lightning normally transfers a batch to the model device before
`training_step`. Because this audit calls that method directly, the wrapper
performs the same transfer first, then passes only:

```text
images_cond
images_albedo
images_mr
images_normal
images_position
name
```

`protocol_metadata` remains separate and is written to the audit report.

## Stage Order

### Stage 0: Environment

The job records the project and upstream Git state, hostname, GPU, CUDA,
PyTorch, bf16 support, relevant environment paths, source hashes, and CUDA
memory. It fails without CUDA or A100-class bf16 support. It does not write to
the upstream tree.

### Stage 1: Batch And Production Loss

`ProtocolCorrectedDataModule` is built with no shuffle and no augmentation.
The job verifies that the first batch is sample index `0`, records tensor
shapes/dtypes/ranges/devices, and evaluates the official production loss. The
loss must be finite.

Before either optimizer update, the script evaluates the same seeded loss
before and after applying the scope. Scope application should only change
`requires_grad`; the report records absolute and relative differences. The
tolerance is explicit (`1e-5` absolute plus `1e-4` relative), so the report
does not claim bitwise equality.

### Stage 2: PC-S1

PC-S1 starts from a fresh true-PBR model. The audit:

1. Selects only `attn_multiview` q/k/v/output projection tensors.
2. Proves the optimizer identity set exactly equals the selected trainable set.
3. Runs one production forward/backward and checks finite, nonzero gradients.
4. Fails if any frozen tensor has a gradient.
5. Clips gradients to norm `1.0` and applies exactly one AdamW update at the
   candidate peak LR `1e-6`.
6. Measures exact update-to-weight ratios for every selected tensor when the
   configured CPU snapshot budget permits, otherwise for a deterministic
   bounded subset whose names are recorded.
7. Saves a temporary selective checkpoint containing selected parameters,
   optimizer state, scheduler state, scope report, and initialization metadata.
8. Loads a fresh base, restores selected tensors exactly, confirms a
   deterministic frozen sample remains base-equivalent, and proves optimizer
   and scheduler state can load.
9. Records the temporary file size and SHA-256, then deletes only that binary.

The JSON checkpoint manifest remains. No temporary `.pt` file should remain
after a successful job.

### Stage 3: PC-Full

PC-Full starts from another fresh true-PBR model. Applying PC-Full must preserve
the official pre-existing `requires_grad` state exactly. The job repeats the
finite/nonzero gradient checks, proves exact optimizer membership, and applies
one update at candidate peak LR `5e-7`. Update ratios use at most 16 tensors,
selected evenly over sorted trainable names so the sample is reproducible.

No PC-Full checkpoint is saved. Historical full-checkpoint writing is already
proven and is outside this preflight's purpose.

## Peak LR Versus Scheduler Step 0

The Day 4 warmup-constant scheduler intentionally has multiplier `0.0` at step
0. Testing an optimizer while that zero LR is active would produce no update
and say nothing about the proposed peak LR. Day 5 therefore applies its one
audited update directly at the candidate peak LR, without an attached
scheduler, and audits scheduler behavior separately with a disposable scalar
parameter.

Expected scheduler evidence is:

| Step | Multiplier |
|---:|---:|
| 0 | 0.0 |
| 25 | 0.5 |
| 50 | 1.0 |
| 160 | 1.0 |
| 320 | 1.0 |

There is no cosine or cycle decay.

## Output Layout

Each run uses a new UTC timestamp plus Slurm job ID:

```text
outputs/phase2n/day5_a100_preflight/<run_id>/
  00_RUNTIME_MANIFEST.json
  batch/
    protocol_metadata.json
    batch_summary.json
  pc_s1/
    scope_report.json
    zero_update_report.json
    gradient_report.json
    update_report.json
    checkpoint_reload_report.json
    checkpoint_manifest.json
  pc_full/
    scope_report.json
    zero_update_report.json
    gradient_report.json
    update_report.json
  scheduler_report.json
  memory_report.json
  summary.json
  summary.md
```

Nothing may be written to `data/`, `checkpoints/`, historical experiment
directories, or `Hunyuan3D2.1_Work`.

## Gates

The runtime passes only when all of these are true:

- exact true-PBR initialization succeeds without fallback;
- the deterministic real batch and official loss are finite;
- zero-update comparisons stay within their documented tolerance;
- both scope reports and optimizer identity sets are exact;
- selected gradients are finite and at least one is nonzero;
- no frozen parameter receives a gradient;
- each one-step update is finite, nonzero, and below the configured `0.01`
  update-to-weight safety ceiling;
- PC-S1 selective reload is exact and its temporary binary is removed;
- PC-Full creates no checkpoint;
- scheduler evidence exactly matches the warmup-constant contract;
- the upstream Git HEAD and status do not change; and
- the final `PHASE2N_DAY5_A100_PREFLIGHT_OK` and job-success markers print.

A failure is a stop signal, not permission to weaken a guard silently.

## Human Review

Passing still does not authorize the final learning rates. A human must review:

- gradient norms and how broadly nonzero gradients are distributed;
- maximum and per-tensor update-to-weight ratios;
- whether `0.01` was comfortably avoided rather than narrowly passed;
- peak and reserved CUDA memory with adequate headroom;
- live normal/position tensor ranges and semantic plausibility; and
- all zero-update and checkpoint-reload evidence.

Only after that review may a later task decide whether Week 2 training is safe.

## Commands

The local readiness command is lightweight and does not import Torch,
Lightning, or Hunyuan:

```bash
python scripts/phase2n_day5_a100_preflight.py \
  --config configs/phase2n_day5_a100_preflight.json \
  --check-only
```

Expected final line:

```text
PHASE2N_DAY5_PREFLIGHT_READINESS_OK
```

After code and readiness review, the user may submit the real audit manually:

```bash
sbatch env/run_phase2n_day5_a100_preflight_a100.sbatch
```

Codex does not submit this job. The sbatch runs check-only first and prints
`===== JOB END: SUCCESS =====` only after the real audit returns successfully.

## Final Reviewed Results And Week 2 Authorization

The real A100 runtime preflight completed successfully as Slurm job `263814`.
The reviewed result directory is:

```text
outputs/phase2n/day5_a100_preflight/20260721T142743Z_263814/
```

The reports record a shared scope root containing exactly `1,962,246,472`
parameters (`1.962B`) across 1,747 tensors. Both scope applications changed the
deterministic pre-update loss by exactly `0.0` in absolute and relative terms.
This confirms zero-update equivalence for the audited batch; it is not a claim
that the two scopes will train identically.

### PC-S1 Reviewed Numbers

| Measure | Reviewed value |
|---|---:|
| Trainable tensors | 80 |
| Trainable parameters | 49,574,080 |
| Trainable fraction | 2.526394% |
| Total gradient norm | 0.6060911421 |
| Finite gradients | 80/80 |
| Nonzero gradients | 80/80 |
| Frozen tensors with gradients | 0 |
| Candidate peak LR | `1e-6` |
| Mean update/weight ratio | `3.0536568507e-5` |
| Maximum update/weight ratio | `9.4664885327e-5` |
| Peak reserved CUDA memory | 45,608,861,696 bytes (42.48 GiB) |

PC-S1 update ratios were measured over **all 80 trainable tensors**. The
snapshot report confirms `all_trainable_selected: true`; these are not sampled
statistics.

The temporary selective checkpoint was `595,122,538` bytes (about `595.1 MB`
decimal) and included the 80 selected parameter tensors plus optimizer and
scheduler state. Reload into a fresh true-PBR model restored all selected
parameters exactly, preserved base equivalence for the deterministic sample of
16 frozen tensors, and loaded both optimizer and scheduler state. Its SHA-256
was recorded in `checkpoint_manifest.json`, and the binary was removed after
successful verification.

### PC-Full Reviewed Numbers

| Measure | Reviewed value |
|---|---:|
| Trainable tensors | 981 |
| Trainable parameters | 1,046,761,668 |
| Trainable fraction | 53.345066% |
| Total gradient norm | 2.3940957912 |
| Finite gradients | 981/981 |
| Nonzero gradients | 981/981 |
| Frozen tensors with gradients | 0 |
| Candidate peak LR | `5e-7` |
| Sampled mean update/weight ratio | `1.1545617114e-5` |
| Sampled maximum update/weight ratio | `6.4392209989e-5` |
| Peak reserved CUDA memory | 66,748,153,856 bytes (62.16 GiB) |

PC-Full update ratios were measured on a **deterministic sample of 16 tensors**
drawn evenly over the 981 sorted trainable names. They must not be described as
full-scope ratio statistics. All 16 sampled updates were finite and nonzero.
No PC-Full checkpoint was saved.

### Scheduler Evidence

The live Day 4 scheduler check produced the expected warmup-constant behavior
for both scopes:

| Optimizer step | Multiplier | PC-S1 LR | PC-Full LR |
|---:|---:|---:|---:|
| 0 | 0.0 | 0 | 0 |
| 25 | 0.5 | `5e-7` | `2.5e-7` |
| 50 | 1.0 | `1e-6` | `5e-7` |
| 160 | 1.0 | `1e-6` | `5e-7` |
| 320 | 1.0 | `1e-6` | `5e-7` |

### Reviewed Conclusion

The numeric review is complete, and no additional Day 5 A100 audit is
required. The measured gradients and one-step updates were finite and nonzero,
optimizer membership remained exact, update/weight ratios were comfortably
below the `0.01` safety ceiling, PC-S1 selective reload was exact, and both
scopes fit on the audited 80 GB A100.

The candidate learning rates are therefore authorized as **safe controlled
pilot settings**, not as proven optimal learning rates. Day 5 establishes
runtime safety only; it does not establish quality improvement, convergence,
generalization, or superiority of either scope. The generated `summary.json`
retains `week2_authorized: false` because it was written before this required
human numeric review; this reviewed closeout is the subsequent authorization
record.

The controlled Week 2 pilot is authorized with this fixed contract:

- PC-S1 peak LR: `1e-6`;
- PC-Full peak LR: `5e-7`;
- 50-step linear warmup followed by constant LR;
- maximum steps: `320`;
- checkpoints at steps `160` and `320` only;
- gradient clipping: `1.0`;
- batch size: `1`;
- precision: bf16 mixed precision;
- spatial augmentation: `none` for both scopes;
- hardware: A100 80 GB only; and
- no test-set use for model selection.

This authorization applies only to the two controlled pilot settings above.
Any broader scope, different learning rate, longer schedule, augmentation
change, or test-informed selection requires a separate review.
