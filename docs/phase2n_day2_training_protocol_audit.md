# Phase 2N Day 2 Training Protocol Audit

## Scope and Evidence Labels

This audit establishes the historical full80-500 training protocol and one
concrete proposed Phase 2N protocol before implementation. It is based only on
static source, existing project metadata, existing logs, and one existing
TensorBoard event file. No model or checkpoint was loaded, and no training,
inference, Blender, GPU, or Slurm command was run.

Project paths are relative to:

`/vol/bitbucket/ct1022/hy3dpaint_finetune`

`HYPAINT` below means this exact read-only upstream path:

`/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint`

Evidence is labeled as follows:

- **OBSERVED:** directly present in a historical log, event file, manifest, or
  existing file inventory.
- **INFERRED_FROM_CODE:** calculated from the committed config and exact source
  used by the run.
- **PROPOSED:** a Phase 2N decision that has not yet been implemented or run.

## 1. Historical Learning-Rate Ground Truth

### Run identity and evidence

The historical run was Slurm job `256864` using:

- config: `configs/ft_datav2_frame_full80_truepbr_500_lr1e6.yaml`
- saved runtime config:
  `logs/train/ft_datav2_frame_full80_truepbr_500_lr1e6-datav2_frame_full80_truepbr_500_lr1e6/configs/project.yaml`
- stdout:
  `logs/slurm/datav2_frame_full80_truepbr_500_lr1e6-256864.out`
- stderr:
  `logs/slurm/datav2_frame_full80_truepbr_500_lr1e6-256864.err`
- TensorBoard event:
  `logs/train/ft_datav2_frame_full80_truepbr_500_lr1e6-datav2_frame_full80_truepbr_500_lr1e6/tensorboard/0/events.out.tfevents.1783259793.linnet.doc.ic.ac.uk.2895185.0`

**OBSERVED:** stdout prints `Setting learning rate to 1.00e-06`, and the saved
runtime YAML records `base_learning_rate: 1.0e-06`. The run reached
`max_steps=500`, produced its step-500 checkpoint, and ended successfully.

### Exact scheduler

**INFERRED_FROM_CODE:**

- optimizer: `torch.optim.AdamW`
- configured base/nominal LR: `1e-6`
- scheduler: `torch.optim.lr_scheduler.LambdaLR`
- scheduler interval: every step
- scheduler frequency: 1
- warm-up: 1000 scheduler steps
- post-warm-up cosine interval: 9000 steps
- cycle-length expression: `1000 + 9000 = 10000` steps
- cycle peak multiplier decay: `gamma = 0.9`
- starting multiplier: `0.0`
- initial cycle peak multiplier: `1.0`

The source is `$HYPAINT/hunyuanpaintpbr/unet/model.py:589-622`. Official
`train.py` assigns the unscaled configured LR at `$HYPAINT/train.py:385-395`.
For every step in this 500-step run, the scheduler was still in its first
linear warm-up, so the relevant formula is simply:

```text
multiplier(step) = step / 1000
absolute_lr(step) = 1e-6 * multiplier(step)
```

### Requested step table

The Slurm progress stream contains all displayed global steps 0 through 499.
TensorBoard logs scalar snapshots every 50 training records, at steps 49, 99,
149, ..., 499. It has no scalar record at step 500.

| Step | OBSERVED Slurm `lr_abs` | OBSERVED TensorBoard exact step | INFERRED_FROM_CODE multiplier | INFERRED_FROM_CODE absolute LR |
|---:|---:|---|---:|---:|
| 0 | `0` | no exact event | `0.000` | `0` |
| 1 | `1e-9` | no exact event | `0.001` | `1e-9` |
| 50 | `5e-8` | no exact event; step 49 is `4.9e-8` | `0.050` | `5e-8` |
| 100 | `1e-7` | no exact event; step 99 is `9.9e-8` | `0.100` | `1e-7` |
| 160 | `1.6e-7` | no exact LR event | `0.160` | `1.6e-7` |
| 320 | `3.2e-7` | no exact LR event | `0.320` | `3.2e-7` |
| 500 | no displayed step-500 scalar | no exact event | `0.500` | `5e-7` |

**OBSERVED:** the final logged global step is 499 with
`lr_abs=4.99e-7`. TensorBoard reports the same value, within float32 precision,
under both `lr_abs` and `lr-AdamW`.

**INFERRED_FROM_CODE:** decay had not begun by step 500. The scheduler would
first reach multiplier `1.0`, and therefore nominal LR `1e-6`, at step 1000.
Cosine decay would begin after that point. The 500-step run therefore never
reached its nominal learning rate; its last observed LR was just under half of
that value.

### TensorBoard and validation evidence

The event file is readable with the already-installed project environment and
contains these scalar tags:

- `lr-AdamW`
- `lr_abs`
- `global_step`
- `epoch`
- `train/albedo_loss_step` and `train/albedo_loss_epoch`
- `train/mr_loss_step` and `train/mr_loss_epoch`
- `train/cons_loss_step` and `train/cons_loss_epoch`

It contains 10 step-level snapshots and seven epoch-level loss snapshots. The
final event has global step 499 and epoch 6. There is no scalar named
`train/loss`, no validation scalar, no image event, and no graph event.

The runtime config has:

- `val_check_interval: 1000`
- `num_sanity_val_steps: 0`
- `limit_val_batches: 0`
- `max_steps: 500`

Therefore normal validation was not expected before step 500, and
`limit_val_batches: 0` disabled validation batches anyway. The historical
`images_val/` directory is empty. The validation dataset was configured, but it
was not exercised by this run.

## 2. Historical Reference-View Protocol

### Actual 80-sample inventory

The training JSON is:

`data/hy3dpaint_train_examples/datav2_frame_panels_full101/examples_train_abs.json`

**OBSERVED:** it is a list of exactly 80 absolute sample-directory strings. It
does not contain objects or a `selected_input_view` field.

All 80 `render_cond/` directories were enumerated using the exact extensions
accepted by the historical loader (`.png`, `.jpg`, `.jpeg`). The inventory is
uniform:

- samples checked: 80
- samples with a missing `render_cond/`: 0
- condition files per sample: 18 for all 80 samples
- total condition files: 1440
- view prefixes in every sample: `000`, `001`, `002`, `003`, `004`, `005`
- lighting files for every view: `AL`, `ENVMAP`, `PL`
- unparsed or unexpected supported-extension filenames: 0

The common filename layout is:

```text
render_cond/000_light_AL.png
render_cond/000_light_ENVMAP.png
render_cond/000_light_PL.png
...
render_cond/005_light_AL.png
render_cond/005_light_ENVMAP.png
render_cond/005_light_PL.png
```

### Historical selection probabilities

The loader builds one flat list of all condition files and calls
`random.choice(cond_images)` at
`$HYPAINT/src/data/dataloader/objaverse_loader_forTexturePBR.py:63-95`.
Because every sample has the same 18-file layout:

| Event | Historical probability per sample access |
|---|---:|
| A particular condition file | `1/18 = 5.5556%` |
| View 000 | `3/18 = 1/6 = 16.6667%` |
| View 001 | `1/6 = 16.6667%` |
| View 002 | `1/6 = 16.6667%` |
| View 003 | `1/6 = 16.6667%` |
| View 004 | `1/6 = 16.6667%` |
| View 005 | `1/6 = 16.6667%` |
| A primary lighting type, conditional on the chosen view | `1/3` |
| Either remaining alternate light, conditional on the primary file | `1/2` |
| A particular ordered view and two-light pair | `1/36 = 2.7778%` |

The alternate path is constructed by replacing the selected light suffix with
one of the other two entries in `lighting_suffix_pool`, chosen uniformly. It
therefore keeps the same view prefix. The actual inventory contains both
alternate files for every possible primary file.

The target side separately finds six albedo files and calls
`random.sample(available_views, 6)` at
`$HYPAINT/src/data/dataloader/objaverse_loader_forTexturePBR.py:58-77`. Since
there are exactly six albedo views, all six are retained on every access, but
their order is randomly permuted. Each selected albedo filename determines the
corresponding MR, normal, and position filename.

### Curated metadata versus training

**OBSERVED:** existing project metadata is explicit and consistent:

- curated manifest: 101 of 101 rows have `selected_input_view=005` and
  `alternative_input_view=004`
- full80 eval override CSV: 24 of 24 rows select view 005
- full80 eval cases: 24 of 24 cases select view 005

That metadata did not enter historical training. The YAML passes only the
80-string examples JSON to the upstream `TextureDataset`, whose constructor
reads only sample paths. Its `__getitem__` never reads the project manifests.

Corrected-input inference differs in two concrete ways. The case builder
resolves override, then curated, then default selected view at
`scripts/make_datav2_frame_mini40_eval_cases.py:63-76`. It then fixes the
single input to `render_cond/005_light_AL.png` at `:104-140`. Historical
training instead used two lighting images from one uniformly random view, with
the primary light also random. In short:

| Protocol | Input view | Input lighting | Target views |
|---|---|---|---|
| Historical full80 training | uniform over 000-005 | random ordered pair of two distinct lights | all six, random order |
| Corrected-input evaluation | fixed curated 005 | one fixed AL image | all six evaluated in fixed view order |

## 3. Historical Augmentation Protocol

### Exact operations and probabilities

`augment_image()` is defined at
`$HYPAINT/src/data/dataloader/loader_util.py:169-219`.

There is first a 50% identity gate. If that gate does not return, each operation
is sampled independently:

| Operation | Conditional probability after non-identity gate | Unconditional probability | Range and interpolation |
|---|---:|---:|---|
| Rotation | 0.30 | 0.15 | uniform `-30` to `+30` degrees; PIL bilinear; expanded then center-cropped |
| Scaling | 0.50 | 0.25 | factor uniform `0.8` to `1.2`; PIL bilinear; center crop/pad |
| Translation | 0.50 | 0.25 | integer x/y offsets independently sampled within `+/-10%` of width/height; `ImageChops.offset` then constant edge fill |
| Perspective | 0.30 | 0.15 | each corner moves inward within 20% of width/height; OpenCV perspective warp with default linear interpolation and constant border |

When scaling is selected, there is a 50% proportional branch using one factor
for both axes and a 50% anisotropic branch using independently sampled width and
height factors. There is no horizontal-flip operation in this augmentation
path.

`load_image()` always resizes to the configured size but does not specify a
resampling argument (`loader_util.py:42-61`). The historical source therefore
delegates that first resize to the installed Pillow default. The files are
already 512 x 512 and the configured training size is 512. The model later
passes numeric `interpolation=3` with antialiasing for condition, target,
normal, and position tensors at
`$HYPAINT/hunyuanpaintpbr/unet/model.py:151-198`.

### Transform sharing

The two same-view reference images use the same first-reference background
choice, but each calls `augment_image()` separately
(`objaverse_loader_forTexturePBR.py:98-113`). Their spatial transforms are
therefore independent.

For every one of the six target views, albedo, MR, normal, and position each
call `augment_image()` separately
(`objaverse_loader_forTexturePBR.py:115-125`). Transform parameters are not
shared across modalities or views. This establishes a **potential cross-channel
alignment risk**. Static source alone does not establish how often the
independent draws produce visible harmful misalignment in an actual batch.

The training and validation YAML entries instantiate the same `TextureDataset`
class with the same default augmentation behavior. If validation were enabled,
it would also be stochastic. The historical 500-step run disabled validation,
so no validation batch provides observed augmentation evidence.

### Background and geometry-map handling

For reference images, the first background choice has probability 0.60 gray,
0.20 black, and 0.20 white. That same chosen color is passed for both reference
images, although the code draws and discards a second background choice. Target
albedo, MR, normal, and position maps always pass `127/255` gray as their load
and augmentation fill color.

The actual training-map metadata adds an important detail:

- all 480 normal PNGs are 512 x 512, 8-bit RGB, without alpha
- all 480 position PNGs are 512 x 512, 8-bit RGB, without alpha
- a corner sample across 18 normal and 18 position files contained dark-gray
  values `(57,57,57)` and `(58,58,58)`
- the render config's nominal background is `(71,71,71)`

Because these files are RGB, `load_image()` does not alpha-composite them onto
the requested 127-gray background. Original dark-gray pixels remain when the
identity path is taken. A non-identity spatial transform can introduce new
127-gray border pixels, and bilinear operations can blend foreground and
background values.

The model resizes and clamps normal/position tensors to `[0,1]`, VAE-encodes
both, and also passes the position image into the voxel-index path
(`$HYPAINT/hunyuanpaintpbr/unet/model.py:182-198,314-357`). Position validity
uses `(position != 1).all(channel)`
(`$HYPAINT/hunyuanpaintpbr/unet/modules.py:121-177,203-250`). Thus a position
pixel with any channel exactly equal to 1 is treated as invalid; sampled
57/58-gray corners do not satisfy that sentinel test. No loader or model code
renormalizes normal vectors after interpolation.

The project renderer connects Blender Geometry `Normal` and `Position` outputs
directly to emission without an explicit normal encoding transform
(`scripts/blender_render_hy3dpaint_example.py:268-278,323-329`). It is therefore
not safe to assume that the stored normal RGB can be inverted with the common
`2 * rgb - 1` formula.

### Certainty boundary

Certain from source and metadata:

- sampling probabilities and operation ranges above;
- absence of horizontal flip;
- separate augmentation calls for both references and every target modality;
- identical augmentation implementation for train and configured validation;
- RGB/no-alpha normal and position maps in all 80 training samples;
- no explicit normal-vector renormalization;
- no validation execution in the historical run.

Requires a future runtime batch visualization:

- the visible frequency and severity of cross-modality displacement;
- foreground/background masks after each operation;
- how dark-gray position backgrounds affect live voxel indices;
- the semantic coordinate space and invertibility of the rendered normal map;
- whether bicubic model resizing creates material geometry artifacts at edges.

### Other historical randomness

The sbatch did not pass `--seed`, so official `train.py` used its default seed
42 and called `seed_everything(42 + global_rank)`
(`$HYPAINT/train.py:92-97,191-196`). The dataset draws from the module-level
Python `random` generator, while conditioning dropout uses NumPy and diffusion
timesteps use PyTorch. The timestep is sampled uniformly once per batch element
and repeated across both PBR materials and all six target views
(`$HYPAINT/hunyuanpaintpbr/unet/model.py:314-321`). Exact worker-level random
sequences remain `RUNTIME_CONFIRMATION_REQUIRED` because the run used one
DataLoader worker and the source does not expose a dataset-local RNG record.

## 4. Proposed Phase 2N Protocol

This is one ordered proposal, not a menu. It remains a proposal until Day 3
unit tests and the Day 5 A100 audit pass.

### Reference and target sampling

For every training sample, draw exactly one reference view with:

| View | Probability |
|---|---:|
| 005 | 0.50 |
| 004 | 0.30 |
| 000 | 0.05 |
| 001 | 0.05 |
| 002 | 0.05 |
| 003 | 0.05 |

This keeps the informative front/oblique-front views dominant without teaching
the model that side and back views can never be inputs. In beginner terms, half
the examples show the best front view, three in ten show the alternate front
view, and the remaining two in ten keep broad view coverage.

Within the chosen view, choose the primary light uniformly from AL, ENVMAP, and
PL, then choose the secondary light uniformly from the other two. Both images
must use the same view. Retain all six target views in canonical order
`000,001,002,003,004,005`; do not randomly permute them. This makes each batch
record easier to reproduce and inspect.

Keep true-PBR initialization, the existing albedo/MR/consistency loss, and
`drop_cond_prob=0.1` unchanged for the first pilot so the protocol and parameter
scope are the controlled changes.

### Shared spatial augmentation

Sample one affine transform record per asset access and apply that exact record
to both reference images and every albedo, MR, normal, and position target view:

- horizontal flip: disabled
- perspective: disabled
- rotation: uniform from `-5` to `+5` degrees
- isotropic scale: uniform from `0.95` to `1.05`
- x translation: uniform from `-3%` to `+3%` of image width
- y translation: uniform from `-3%` to `+3%` of image height
- no separate identity gate

The transform is deliberately mild. The goal is to prevent memorization of
exact pixels without changing which object regions correspond across PBR and
geometry maps.

Horizontal flip is disabled because it changes left/right handedness and would
require a proven normal-vector convention. Perspective is disabled because the
position maps already describe rendered 3D geometry; an extra independent 2D
projective distortion would make that meaning harder to preserve. The narrow
rotation, scale, and translation ranges add small framing variation while
remaining close to the rendered camera protocol.

Use bilinear interpolation for reference, albedo, MR, and position values. Use
nearest-neighbor interpolation for a foreground/background mask. Newly exposed
pixels must use the original modality's measured background triplet rather
than hard-coded 127 gray, so existing full101 files remain semantically
unchanged.

For normal maps, the initial implementation must preserve RGB values under the
shared spatial warp and must not assume a standard invertible vector encoding.
Use nearest-neighbor interpolation until the renderer's normal encoding is
calibrated. If a later calibration establishes a valid vector decode, switch to
bilinear foreground interpolation, decode vectors, renormalize each valid
foreground vector to unit length, re-encode, and restore background pixels
exactly. If the normals are camera-space, the x/y vector components must also
be rotated by the same in-plane angle. This calibration is a training gate, not
an optional cleanup.

### Randomness and audit records

Use a project-local RNG, not the upstream module-level `random` state. Set the
base seed to `42`. Derive a stable per-sample seed with SHA-256 over the explicit
tuple `(base_seed, rank, worker_id, epoch, asset_id)` rather than the
process-randomized Python `hash()`. The dataset must expose `set_epoch()` so
this tuple is complete. Day 3 unit tests must prove that the same tuple
reproduces the same view, lighting pair, and transform record.

Every training step should record:

- asset ID
- chosen reference view
- ordered lighting pair
- rotation, scale, and x/y translation
- diffusion timestep
- current absolute LR
- global step and epoch

The dataset should return the sampling/augmentation record as batch metadata.
A later project-local training hook should add the diffusion timestep and LR to
the same structured record. Write one append-only
`protocol_trace_rank{rank}.jsonl` per run under its project-local log directory;
do not write one file per step.

### Learning rate and checkpoints

Use a 50-step linear warm-up from multiplier 0 to the selected peak, then hold
that peak constant through step 320. A short constant plateau makes the 160 and
320 checkpoints easy to compare; the historical 1000-step warm-up never let a
500-step run reach its configured peak.

- PC-Full peak LR candidate: `5e-7`
- PC-S1 peak LR candidate: `1e-6`
- checkpoint steps: 160 and 320

PC-Full updates a broad parameter set, so its candidate peak is conservative.
PC-S1 updates only multiview attention projections, so its candidate peak can
be higher. These relative values are not final: the Day 5 A100 gradient/update
audit must compare gradient norms, update-to-weight ratios, memory, and one-step
output drift before either training run is authorized.

## 5. Day 3 Implementation Decision

### Decision

Use a project-local minimal dataset implementation that reuses the upstream
`BaseDataset.load_image()` helper where safe, but does not subclass or call the
upstream `TextureDataset.__getitem__()`.

This is safer than subclassing `TextureDataset` because its view selection,
lighting selection, background sampling, and independent augmentation calls are
interleaved inside one method. Overriding that behavior would replace nearly
the entire method anyway while retaining hidden coupling to its random calls.
A small project-local implementation makes selected-view weighting and one
shared transform explicit, deterministic, and unit-testable without copying the
training framework or touching upstream source.

The class may inherit only from upstream `BaseDataset` to reuse basic image
loading and length behavior. Sampling and transformation should be implemented
as small pure project-local functions, with the dataset assembling their
results.

### Smallest proposed Day 3 footprint

Do not create these files on Day 2. The smallest useful Day 3 change is:

1. Add `src/hy3dft/protocol_corrected_dataset.py` containing:
   - explicit condition-file inventory validation;
   - weighted reference-view and same-view lighting-pair selection;
   - an immutable shared affine-parameter record;
   - background-preserving transform application;
   - canonical six-target loading;
   - returned audit metadata and deterministic `set_epoch()` seeding.
2. Add `tests/test_phase2n_protocol_corrected_dataset.py` using synthetic files
   to test exact probabilities through deterministic draws, same-view lighting,
   shared transforms, background preservation, canonical view order, and seed
   reproducibility.

No upstream file, existing Phase 2L data file, training config, sbatch, or
evaluation script needs to change on Day 3. A later day can add the minimum
training config/entrypoint and structured timestep/LR logging after the dataset
tests and normal-encoding decision pass.

## 6. Day 2 Decision and Remaining Gates

The historical full80 run was not a true `1e-6` run in the ordinary sense: its
1000-step warm-up limited the observed LR to `4.99e-7` by global step 499. It
also trained with uniformly random reference views, while accepted evaluation
used fixed view 005. Phase 2N should correct that protocol before judging S1.

The proposed first selective scope remains S1 (`attn_multiview` projections).
S2 has no distinct statically justified geometry projection. S3 remains gated
on runtime confirmation of late albedo-specific paths.

Before training starts, later audits must still confirm:

- live S1 names, counts, optimizer membership, gradients, and update ratios;
- normal-map encoding and any valid vector renormalization procedure;
- live position-mask behavior for the existing dark-gray background;
- deterministic RNG behavior with the actual DataLoader worker lifecycle;
- one-batch visual alignment across reference, albedo, MR, normal, and position;
- safe compact logging of sampling metadata, timestep, and LR;
- whether PC-Full `5e-7` versus PC-S1 `1e-6` remains appropriate after the
  Day 5 A100 audit.

No new Phase 2N training has started.
