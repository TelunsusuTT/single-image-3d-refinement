# Phase 2N Day 4 Training Support

## Purpose

Day 4 adds a project-local connection layer for the future Phase 2N PC-Full
and PC-S1 experiments. It does not create a new dataset and does not alter any
rendered training image. Both planned experiments still use the existing
full101 split: 80 train assets, 10 validation assets, and 11 test assets.

The implementation is CPU-testable and contains no Hunyuan model import,
checkpoint load, training launch, inference call, GPU operation, or file-writing
runtime. No training has started.

## Data Connection Layer

`ProtocolCorrectedDatasetFromJson` reads the existing examples JSON, requires
one non-empty list of absolute sample-directory paths, and delegates all sample
inventory, reference selection, image loading, target ordering, and protocol
metadata construction to the Day 3 `ProtocolCorrectedTextureDataset`. It does
not duplicate that logic.

The wrapper preserves the historical model keys:

- `images_cond`
- `images_albedo`
- `images_mr`
- `images_normal`
- `images_position`
- `name`

It also preserves `protocol_metadata`. At each sample access it obtains the
actual PyTorch DataLoader worker ID with `get_worker_info()`. Each worker owns a
private copied dataset instance, so updating that copy's worker ID does not
change another worker or the parent process. Rank remains an explicit project
input. Reference selection continues to use the Day 3 private SHA-256-derived
RNG and never the global Python random state.

`protocol_collate_fn` passes the six historical keys through PyTorch's normal
`default_collate`. It separately preserves `protocol_metadata` as a list of
JSON-serializable Python dictionaries. This is important because a SHA-256
derived seed can be larger than signed int64; keeping the metadata outside
normal integer-tensor collation avoids overflow without dropping audit data.

`ProtocolCorrectedDataModule` is a small Lightning DataModule, not a copy of
the upstream framework. `setup("fit")` builds the train reader and optional
validation reader, both dataloaders use `protocol_collate_fn`, and
`val_dataloader()` returns `None` when no validation JSON is configured. Its
`set_epoch()` propagates to all owned datasets. `set_epoch_recursive()` also
supports common `Subset` and `ConcatDataset` wrappers while tracking object
identity to prevent cycles and repeated updates.

Spatial augmentation remains exactly `none`. Day 4 changes only how the tested
Day 3 reader is connected to batching and training support. It does not add
rotation, scale, translation, horizontal flip, perspective, affine warps, or
normal-map decoding.

## Trainable Scopes

`TrainableScopeReport` records unique tensor counts, element counts, and exact
trainable and frozen names. It is immutable and its `to_dict()` output is JSON
serializable.

### PC-Full

PC-Full means the official broad partial scope, not every parameter in the
model. `apply_trainable_scope(model, "pc_full")` reads and reports the
`requires_grad` state already established by the freshly instantiated official
Hunyuan model. It changes none of those flags, never reimplements the upstream
string-based freeze policy, and fails if the official state contains zero
trainable parameters.

This preserves the original initialization boundary: parameters frozen by the
official pipeline remain frozen, and no frozen parameter is broadened into the
experiment by project-local code.

### PC-S1

PC-S1 first discovers and validates its complete selection, then freezes the
model and enables only weight and optional bias tensors owned by exact
`nn.Linear` projection modules under an `.attn_multiview.` path segment:

```text
*.attn_multiview.to_q.{weight,bias}
*.attn_multiview.to_k.{weight,bias}
*.attn_multiview.to_v.{weight,bias}
*.attn_multiview.to_out.0.{weight,bias}
```

Any valid model prefix is accepted. Zero matches, non-Linear owners, duplicate
parameter identities, registration mismatches, and selected paths containing
refview, DINO, ordinary attention, feed-forward, convolution, resnet, norm, or
dual-UNet segments fail closed.

`conv_in` is deliberately excluded. Normal and position latents enter through
that convolution, but albedo/MR diffusion inputs also share it. Updating the
whole convolution would broaden S1 beyond a name-isolatable multi-view
attention experiment and could not honestly be described as geometry-only.

S2 is not implemented because the static audit found no separate trainable
normal or position projector. S3 remains deferred until runtime evidence can
justify an exact scope.

## Optimizer Safety

`build_trainable_adamw()` passes only tensors whose current
`requires_grad=True` state is set. It rejects an empty selection and duplicate
parameter identities, then verifies that optimizer membership exactly equals
the current unique trainable identity set. Frozen parameters are absent from
optimizer parameter groups and therefore cannot receive optimizer state.

The helper does not select a scope itself. The intended order is:

1. instantiate the official model;
2. call `apply_trainable_scope()`;
3. inspect and save the returned report;
4. call `build_trainable_adamw()`;
5. verify live gradients and updates during the Day 5 audit.

## Learning-Rate Schedule

The new scheduler is a 50-step linear warmup followed by a constant multiplier.
It has no cosine decay, cycle decay, or historical 1000-step warmup.

| Scheduler step | Multiplier | Absolute LR at provisional peak `P` |
|---:|---:|---:|
| 0 | 0.0 | 0 |
| 25 | 0.5 | `0.5 * P` |
| 50 | 1.0 | `P` |
| 160 | 1.0 | `P` |
| 320 | 1.0 | `P` |

With the installed PyTorch 2.5.1 behavior tested on CPU, constructing
`LambdaLR` performs its initial scheduler update at `last_epoch == 0`, so the
optimizer LR immediately becomes multiplier 0. The training loop must then
call `optimizer.step()` followed by `scheduler.step()` once per optimizer
update. After 25 such updates the multiplier is 0.5; after 50 it is 1.0 and it
remains there through step 320. A future Lightning `configure_optimizers()`
integration must mark this scheduler with `interval: step` and `frequency: 1`.

The checkpoint comparison plan remains steps 160 and 320. Candidate peak LRs
remain provisional: the Day 2 values were `5e-7` for PC-Full and `1e-6` for
PC-S1, but neither is authorized until the Day 5 A100 audit measures real
gradients, update ratios, memory, and initial output drift.

## Day 4B Historical-Augmentation Evidence

Day 4B completed the planned comparison as a temporary CPU-only evidence run
under `tmp/phase2n_day4b_reader_comparison`. It created no persistent runtime
script and did not modify the source training examples. The controlled design
used exactly three entries from the full80 examples list:

- list index 0: `B073NZFTH9`
- list index 39: `B073P5KN5N`
- list index 79: `B07W5BDJ73`

For every asset, the unaugmented control and historical augmentation path read
the same six files: view-005 `AL` and `PL` references plus view-005 albedo, MR,
normal, and position maps. Controls were loaded as 512 x 512 RGB without any
spatial operation and were checked against the original pixel layout. The
controlled historical path used neutral fill RGB `(127, 127, 127)`, freshly
reloaded every source, seeded Python and NumPy with each seed from 1000 through
1011, and called the upstream `augment_image()` helper separately in the
historical order: reference AL, reference PL, albedo, MR, normal, position.

This produced 12 draws per asset and 36 draws in total. Exact same-file pixel
comparison found that all 36 draws changed only a subset of the six inputs:

| Pixel-change class | Count | Fraction |
|---|---:|---:|
| All six unchanged | 0 | 0.0000 |
| Only a subset changed | 36 | 1.0000 |
| All six changed | 0 | 0.0000 |

The per-input change frequencies and mean absolute differences were:

| Input | Changed draws | Fraction | Mean MAE (0-255) |
|---|---:|---:|---:|
| Reference AL | 15/36 | 0.4167 | 16.1909 |
| Reference PL | 12/36 | 0.3333 | 13.6539 |
| Albedo | 12/36 | 0.3333 | 13.9969 |
| MR | 24/36 | 0.6667 | 18.7783 |
| Normal | 30/36 | 0.8333 | 26.7899 |
| Position | 15/36 | 0.4167 | 15.4701 |

### Alignment Diagnostics

The diagnostic mask estimated each image's background as the per-channel
median of an 8-pixel border and classified pixels whose Euclidean RGB distance
from that estimate exceeded 30. It is a compact image-space heuristic, not a
physical foreground mask or semantic ground truth.

| Pair group | Pair count | Median IoU | Worst IoU | Median centroid difference | Worst centroid difference | Median bbox difference | Worst bbox difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference AL/PL | 36 | 0.6893 | 0.3167 | 0.0160 | 0.0786 | 0.0532 | 0.1675 |
| Target modality pairs | 216 | 0.6912 | 0.3345 | 0.0144 | 0.0938 | 0.0679 | 0.1670 |

Read-only human review of the three asset contact sheets and the compact
worst-case board found visible relative rotation, scale, translation, and
perspective differences across inputs that should describe the same aligned
view. This visual evidence is considered alongside the exact pixel-change
evidence and heuristic masks; the mask values are not used as physical truth.

### Final Augmentation Decision

The first PC-Full and PC-S1 pilots will use:

```text
spatial_augmentation = none
```

Both pilots must use the identical protocol-corrected, no-augmentation data
path. Their comparison may differ in trainable scope and approved learning
rate, but not in reference selection, target ordering, source images, or
spatial preprocessing. This keeps the first small-data adaptation comparison
from introducing the observed independent spatial-transform behavior and its
potential cross-channel alignment risk.

This decision does not label the official augmentation strategy a bug and does
not claim that it explains the historical full80 result. Independent spatial
augmentation may be appropriate in the original large-scale training context.
A shared mild-augmentation policy remains a future ablation, considered only
after normal-map encoding and direction handling have been calibrated.

## Day 5 Gate

Day 5 must use a no-update or tightly controlled A100 audit to confirm:

- both PC-Full and PC-S1 consume the identical no-augmentation data path, with
  no hidden call to the historical spatial augmentation helper;
- exact live PC-Full and PC-S1 names, tensor counts, and element counts;
- exact optimizer membership and absence of duplicate identities;
- nonzero gradients on intended PC-S1 tensors and no gradients elsewhere;
- update-to-weight ratios at candidate learning rates;
- step-0 output equivalence before any update;
- scheduler values at live optimizer steps;
- one-batch reference/target alignment and normal/position background behavior;
- memory use and whether the candidate peak LRs remain conservative.

No config, sbatch launcher, checkpoint policy, or training entrypoint is added
on Day 4. No Phase 2N training has started, and no pilot is authorized until
the Day 5 A100 runtime gate passes.
