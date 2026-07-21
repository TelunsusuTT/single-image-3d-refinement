# Phase 2N Day 1 Architecture and Reuse Audit

## Audit Scope

Phase 2N proposes protocol-corrected, geometry-focused selective fine-tuning.
Day 1 is a static architecture and reuse audit only. No model was imported or
instantiated, no checkpoint was loaded, and no training, inference, Blender,
GPU, or Slurm command was run.

Project path references below are relative to:

`/vol/bitbucket/ct1022/hy3dpaint_finetune`

Upstream references use `HYPAINT` as an alias for this exact read-only path:

`/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint`

Line references describe the checkout audited on 2026-07-21. Runtime names,
counts, tensor shapes, optimizer state, and gradient flow are explicitly marked
`RUNTIME_CONFIRMATION_REQUIRED` where static source is insufficient.

## A. Current Repository Snapshot

- Branch: `main`
- HEAD: `c22905d3f5feaeefa6c8a561800f3a448b374a26`
- HEAD subject: `add phase 2m LoRA training and evaluation pipeline`
- Worktree before this audit: clean.

Relevant Phase 2L training and evaluation files include:

- `configs/datav2_frame_full80_train.json`
- `configs/ft_datav2_frame_full80_truepbr_500_lr1e6.yaml`
- `env/run_datav2_frame_full80_train_a100.sbatch`
- `scripts/check_datav2_frame_full80_training_readiness.py`
- `scripts/make_datav2_frame_full80_eval_cases.py`
- `scripts/check_datav2_frame_full80_eval_readiness.py`
- `env/run_datav2_frame_full80_eval_infer_a100.sbatch`
- `scripts/render_datav2_frame_full80_eval_views_blender.py`
- `scripts/compare_datav2_frame_full80_rendered_views.py`
- `scripts/aggregate_datav2_frame_full80_eval.py`
- `docs/phase2l7c_full80_eval_closeout.md`
- `docs/phase2l_full80_failure_cases.md`

Relevant Phase 2M implementation and result files include:

- `src/hy3dft/lora/targeting.py`
- `src/hy3dft/lora/local_linear.py`
- `src/hy3dft/lora/peft_lora.py`
- `src/hy3dft/lora/io.py`
- `scripts/phase2m_lora_inventory.py`
- `scripts/phase2m_train_lora_refdino.py`
- `scripts/phase2m_lora_infer_smoke.py`
- `scripts/phase2m_lora_multiscale_pilot.py`
- `scripts/phase2m_render_lora_multiscale_pilot.py`
- `scripts/aggregate_phase2m_lora_multiscale_pilot_eval.py`
- `scripts/make_phase2m_lora_multiscale_pilot_boards.py`
- `docs/phase2m_lora_negative_result_report.md`
- `docs/phase2m_lora_rescue_plan.md`

The project-level Python package is deliberately small:

```text
src/hy3dft/
  __init__.py
  lora/
    __init__.py
    targeting.py
    local_linear.py
    peft_lora.py
    io.py
```

Most orchestration remains in `scripts/`, static experiment settings in
`configs/`, and manual launchers in `env/`. The repository has no general
selective full-parameter training package yet. The existing
`docs/current_project_state.md` still closes at Phase 2L.7C; the newer Phase 2M
negative-result documents are the authoritative evidence for the LoRA branch.

## B. Full/Partial Fine-Tuning Call Chain

There is no project-local Python training wrapper in the full80 path. The sbatch
performs project-local preflight and then directly invokes upstream `train.py`.

1. **Sbatch and project config.**
   `env/run_datav2_frame_full80_train_a100.sbatch:31-37` declares the project
   config, examples JSON, checkpoint root, and log paths. Its readiness and
   strict dataset checks are at `:61-77`. It changes to `HYPAINT` and directly
   runs `python3 train.py` at `:107-114`.
2. **Training YAML.**
   `configs/ft_datav2_frame_full80_truepbr_500_lr1e6.yaml:1-11` instantiates
   `hunyuanpaintpbr.HunyuanPaint` from the local true-PBR pipeline and sets the
   learning rate. `:12-26` selects the upstream data module and PBR texture
   dataset. `:27-35` configures one weights-only step-500 checkpoint. `:37-48`
   sets the 500-step trainer and leaves `init_control_from` and `resume_from`
   null.
3. **Official entrypoint.**
   `$HYPAINT/train.py:198-215` loads the YAML and instantiates the model.
   `$HYPAINT/train.py:217-221` identifies the inner UNet and its checkpoint key
   prefix. `$HYPAINT/train.py:247-263` replaces `conv_in` when
   `noise_in_channels` is configured. `$HYPAINT/train.py:288-297` implements
   optional resume loading. `$HYPAINT/train.py:302-377` assigns `model.logdir`,
   builds the logger and checkpoint callbacks, and constructs the bf16 DDP
   trainer.
4. **Dataloader.**
   `$HYPAINT/train.py:380-383` instantiates and sets up the configured data
   module. `$HYPAINT/src/data/objaverse_hunyuan.py:22-54` instantiates its
   datasets, and `:59-75` creates train/validation data loaders around the
   concatenated datasets. The sample implementation is
   `$HYPAINT/src/data/dataloader/objaverse_loader_forTexturePBR.py:25-39`.
5. **Current input protocol.**
   The dataset randomly samples target albedo views at
   `$HYPAINT/src/data/dataloader/objaverse_loader_forTexturePBR.py:58-77` and
   randomly chooses any condition image plus alternate lighting at `:63-96`.
   It does not read the curated `selected_input_view` field. Reference images
   are independently augmented at `:98-113`; albedo, MR, normal, and position
   maps each receive separate random augmentation calls at `:115-125`.
   `$HYPAINT/src/data/dataloader/loader_util.py:169-219` shows that an
   augmentation call can independently rotate, scale, translate, or apply
   perspective. This is the first protocol issue Phase 2N must correct.
6. **Model construction.**
   `$HYPAINT/hunyuanpaintpbr/unet/model.py:46-103` creates the diffusion
   pipeline, wraps the pretrained 2D UNet as `UNet2p5DConditionModel`, and calls
   `pipeline.set_learned_parameters()`. It exposes the wrapped module as
   `HunyuanPaint.unet` at `:112` and creates the frozen DINO encoder at
   `:118-120`.
7. **Training step and losses.**
   `$HYPAINT/hunyuanpaintpbr/unet/model.py:314-399` prepares PBR, reference,
   DINO, normal, and position inputs. `:401-467` predicts and splits albedo/MR,
   then calculates albedo, MR, and consistency losses.
8. **Optimizer parameter selection.**
   `$HYPAINT/hunyuanpaintpbr/unet/model.py:589-622` constructs
   `AdamW(self.unet.parameters(), lr=...)`. Thus every parameter object in the
   wrapped `UNet2p5DConditionModel` is placed in the optimizer; only those with
   `requires_grad=True` can receive updates. The effective freeze policy comes
   from `$HYPAINT/hunyuanpaintpbr/pipeline.py:118-143` and is detailed in
   Section G.
9. **Scheduler.**
   The same `$HYPAINT/hunyuanpaintpbr/unet/model.py:593-622` block constructs a
   step-interval `LambdaLR` with warm-up, cosine decay, and cycle decay.
   `$HYPAINT/train.py:385-395` assigns the configured base learning rate without
   batch-size scaling.
10. **Checkpoint save.**
    `$HYPAINT/train.py:319-371` merges the YAML ModelCheckpoint settings into
    the callback configuration. The full80 YAML requests `save_weights_only:
    true`, `every_n_train_steps: 500`, `save_top_k: -1`, and `save_last: false`
    at `configs/ft_datav2_frame_full80_truepbr_500_lr1e6.yaml:27-35`. The sbatch
    validates the newly created checkpoint at
    `env/run_datav2_frame_full80_train_a100.sbatch:132-157`.

## C. LoRA Call Chain

1. **Runtime target inventory.**
   `scripts/phase2m_lora_inventory.py:65-85` loads the inference UNet only in
   the A100 runtime path. `:127-151` inventories eligible linear modules,
   selects the refview+DINO target set, and writes the exact target list.
2. **Target classification and validation.**
   `src/hy3dft/lora/targeting.py:9-21` defines suffix, segment, and exclusion
   rules. `:47-66` classifies candidates, `:69-92` inventories/selects them,
   and `:95-110` verifies that every stored target still resolves exactly.
3. **Injection backend.**
   `src/hy3dft/lora/peft_lora.py:63-107` chooses the explicitly allowed
   backend. The completed Phase 2M run used `local_linear_fallback`, whose
   exact-module replacement is implemented by
   `src/hy3dft/lora/local_linear.py:12-65`.
4. **Freeze and trainability policy.**
   `src/hy3dft/lora/peft_lora.py:35-60` freezes all base parameters and asserts
   that only LoRA parameters remain trainable. Each replacement also freezes
   its base linear layer and adds trainable A/B matrices at
   `src/hy3dft/lora/local_linear.py:12-33`.
5. **M2 model/data setup and optimizer.**
   `scripts/phase2m_train_lora_refdino.py:242-280` reuses the full80 YAML to
   instantiate model and data and locates the target root. `:502-525` freezes,
   injects, verifies trainability, and replaces `configure_optimizers` so the
   optimizer receives only trainable LoRA A/B parameters.
6. **Adapter-only save/load.**
   `src/hy3dft/lora/local_linear.py:68-92` extracts and restores only adapter
   tensors. `src/hy3dft/lora/io.py:15-66` rejects unsafe full-checkpoint-like
   names and guards adapter-only persistence. M2 checkpoint callbacks are at
   `scripts/phase2m_train_lora_refdino.py:331-425`; Lightning full-model
   checkpointing is disabled at `:559-589`.
7. **Corrected-input inference.**
   `scripts/phase2m_lora_infer_smoke.py:102-143` selects an existing full80
   case and requires corrected input view 005. `:237-289` builds base and LoRA
   pipelines, injects and loads the adapter, and applies an absolute runtime
   scale. `:291-326` runs the one-case LoRA output and records no-merge safety.
8. **Multi-scale inference.**
   `scripts/phase2m_lora_multiscale_pilot.py:62-111` selects val, test, and
   train-sanity cases. `:131-230` uses a fresh pipeline for base and each LoRA
   scale, preventing accumulated scaling. `:289-330` writes variant outputs
   and summaries.
9. **Rendered-view evaluation.**
   `scripts/phase2m_render_lora_multiscale_pilot.py:176-260` renders the four
   variants through the existing Phase 2K renderer.
   `scripts/aggregate_phase2m_lora_multiscale_pilot_eval.py:114-185` collects
   per-view metrics and `:227-297` aggregates by view group and scale.
   `scripts/make_phase2m_lora_multiscale_pilot_boards.py:44-99` creates the
   fixed LoRA comparison boards.

The Phase 2M safety ideas are reusable. Its `ref_dino` target selector,
adapter-only injection, and experiment entrypoints are historical baselines,
not the proposed Phase 2N selective full-parameter method.

## D. Evaluation Call Chain

The current corrected-input full80 path is:

1. `scripts/make_datav2_frame_full80_input_view_review.py` produces the human
   review board and override CSV.
2. `scripts/make_datav2_frame_full80_eval_cases.py:63-67` delegates case
   construction to `scripts/make_datav2_frame_mini40_eval_cases.py`. The latter
   selects val/test/train-sanity cases at `:40-51`, resolves override then
   curated then default input view at `:63-76`, and creates mesh/image/reference
   case records at `:104-148`.
3. `env/run_datav2_frame_full80_eval_infer_a100.sbatch:50-56` builds and checks
   cases. `:96-135` invokes `scripts/run_phase2g_paint_infer.py` once for base
   and once for the full80 checkpoint, using the same corrected-input case.
4. `scripts/run_phase2g_paint_infer.py:28-61` validates mode and paths,
   `:178-197` maps a full checkpoint into the inference UNet when requested,
   and `:200-250` runs fixed-mesh official inference to GLB.
5. `scripts/make_datav2_frame_full80_render_eval_configs.py:40-48` delegates
   render-case creation to the mini40 implementation.
6. `scripts/render_datav2_frame_full80_eval_views_blender.py:85-105` imports and
   calls the common renderer for base and fine variants.
   `scripts/render_phase2k3_glb_views_blender.py:84-290` contains reusable
   world-bbox normalization, fixed camera setup, and `render_variant()`.
7. `scripts/compare_datav2_frame_full80_rendered_views.py:22-31` delegates to
   the mini40 comparator. `scripts/compare_datav2_frame_mini40_rendered_views.py:11-14`
   imports Phase 2K metric helpers, `:71-90` compares one view, `:93-123`
   builds a board, and `:155-182` compares all cases.
   `scripts/compare_phase2k3_rendered_views.py:93-171` provides MAE, RMSE,
   PSNR metadata, SSIM-like, and pair-level metric functions.
8. `scripts/aggregate_datav2_frame_full80_eval.py:73-77` delegates to
   `scripts/aggregate_datav2_frame_mini40_eval.py`, whose `:42-87` defines
   summaries and view groups and whose `:99-160` collects split/view metrics.

For a Phase 2N multi-variant pilot, directly reuse corrected case construction,
`run_phase2g_paint_infer.py`, Phase 2K fixed-view rendering, and Phase 2K pair
metrics. Parameterize the Phase 2M renderer/aggregator/board layer to remove its
exactly-three-case and fixed LoRA-scale assumptions. Do not create a new
evaluation framework.

## E. Existing Reusable Components Matrix

| File or function | Classification | Purpose and input/output contract | Imports Hunyuan | A100 | Blender | Phase 2N rationale |
|---|---|---|---:|---:|---:|---|
| `scripts/make_datav2_frame_full80_eval_cases.py` plus mini40 `build_case()` | DIRECT_REUSE | Split/override/curation inputs to corrected-input case JSON, symlinks, and references | No | No | No | Already enforces the selected-input protocol used by the accepted benchmark. |
| `scripts/check_datav2_frame_full80_eval_readiness.py` | DIRECT_REUSE | Eval config/cases to local readiness report | No | No | No | Keep the established held-out case gate. |
| `scripts/run_phase2g_paint_infer.py` | DIRECT_REUSE | Case plus base/full checkpoint mode to fixed-mesh GLB | Runtime only | Yes | No | Existing trusted base and full-checkpoint inference path. Add a new mode only if selective checkpoints need different loading. |
| `scripts/render_phase2k3_glb_views_blender.py::render_variant` | DIRECT_REUSE | GLB and fixed render config to six rendered PNGs | No | No | Yes | Common bbox, camera, and view convention. |
| `scripts/compare_phase2k3_rendered_views.py::pair_metrics` | DIRECT_REUSE | Reference/render pair to MAE, RMSE, PSNR metadata, SSIM-like | No | No | No | Common numeric metric implementation across 2K/2L/2M. |
| `scripts/aggregate_datav2_frame_mini40_eval.py` | DIRECT_REUSE | Per-view rows to split and front/non-front summaries | No | No | No | Existing corrected-input group definitions; retain input/front/non-front reporting. |
| `src/hy3dft/lora/targeting.py` | PARAMETERIZE_OR_EXTEND | Loaded module to exact target inventory/validation | No direct import; receives model | Runtime model needed | No | Reuse exact-name validation and exclusions, but add generic parameter-scope rules rather than LoRA-only Linear selection. |
| `scripts/phase2m_train_lora_refdino.py` lifecycle helpers | PARAMETERIZE_OR_EXTEND | YAML/model/data setup, safe log paths, parameter summary, fit lifecycle | Runtime only | Yes | No | Reuse lifecycle and freeze assertions. Do not retrofit LoRA adapter semantics into selective full-parameter training. |
| `scripts/phase2m_render_lora_multiscale_pilot.py::build_render_config` | PARAMETERIZE_OR_EXTEND | Pilot summary and variant outputs to render config | No | No | Runtime script uses Blender | Remove exactly-three-case and fixed LoRA variant assumptions. |
| `scripts/aggregate_phase2m_lora_multiscale_pilot_eval.py` generic row/group functions | PARAMETERIZE_OR_EXTEND | Multi-variant rendered PNGs to rows and grouped summaries | No | No | No | `collect_metric_rows()` and `summarize_rows()` are generic; replace LoRA-scale-specific comparison naming. |
| `scripts/make_phase2m_lora_multiscale_pilot_boards.py` | PARAMETERIZE_OR_EXTEND | Fixed reference/base/scale columns to JPG boards | No | No | No | Make variant columns and labels config-driven. |
| `src/hy3dft/lora/local_linear.py`, `peft_lora.py`, `io.py` | HISTORICAL_ONLY | Exact Linear LoRA injection and adapter-only IO | Runtime torch only | Runtime dependent | No | Preserve as the Phase 2M baseline. Phase 2N changes selected existing weights rather than injecting A/B adapters. |
| `scripts/phase2m_lora_inventory.py` and M1/M2/M3 entrypoints | HISTORICAL_ONLY | Reproduce the refview+DINO LoRA experiment | Runtime only | Yes | M3C render only | Retain for provenance and negative-result reproduction, not as Phase 2N defaults. |
| `configs/ft_datav2_frame_full80_truepbr_500_lr1e6.yaml` | HISTORICAL_ONLY | Broad full80-500 training configuration to full checkpoint | Via upstream train | Yes | No | Baseline/reference only; its broad freeze policy is not selective enough. |
| Phase 2F/2H configs initialized from Stable Diffusion rather than true PBR | DO_NOT_USE_FOR_PHASE2N | Historical smoke/overfit paths | Yes | Yes | No | Phase 2I established the initialization mismatch. |
| Old wrong-input evaluation outputs and cases | DO_NOT_USE_FOR_PHASE2N | Historical inference using uncorrected reference views | N/A | N/A | N/A | Phase 2K.4 showed input-view choice dominates; comparisons must use corrected inputs. |
| Shared `conv_in`, shared `conv_out`, or all-UNet unlock by name | DO_NOT_USE_FOR_PHASE2N | Broad parameter updates | N/A | Yes if implemented | No | Geometry passes through these modules, but they also control unrelated PBR behavior and are not isolated geometry scopes. |

## F. Static Hunyuan Architecture Map

### Naming root

`HunyuanPaint.unet` is the `UNet2p5DConditionModel`
(`$HYPAINT/hunyuanpaintpbr/unet/model.py:95-112`). That wrapper stores the
main pretrained UNet as `.unet` and a copied reference stream as `.unet_dual`
(`$HYPAINT/hunyuanpaintpbr/unet/modules.py:758-807`). Therefore canonical full
model names begin `unet.unet.` for the main UNet. Existing Phase 2M inventory
uses the wrapper-relative root and consequently records the same modules with
one fewer leading `unet.`.

### Modules and pathways

| Component | Static symbol and canonical parameter path | Trainable parameters | PBR relationship and isolation | Evidence |
|---|---|---:|---|---|
| Multi-view attention | `Basic2p5DTransformerBlock.attn_multiview`; `unet.unet.{down_blocks.*,mid_block,up_blocks.*}.attentions.*.transformer_blocks.*.attn_multiview.{to_q,to_k,to_v,to_out.0}.*` | Yes | One projection set operates over the flattened material/view dimension, so it is geometry-aware but shared by albedo and MR. Projection modules are name-isolatable; material effects are not. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:352-364,611-631` |
| Reference attention | `.attn_refview.{to_q,to_k,to_v,to_out.0}.*` plus processor MR branches | Yes | Q/K are shared. Base V/output serve albedo; MR uses separate processor V/output. Individual projection modules are isolatable. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:366-388,580-607`; `$HYPAINT/hunyuanpaintpbr/unet/attn_processor.py:758-839` |
| DINO attention | `.attn_dino.{to_q,to_k,to_v,to_out.0}.*` | Yes | Shared conditioning attention across flattened PBR samples; projection modules are isolatable, but not geometry-specific. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:390-401,665-676` |
| Ordinary self-attention | `.transformer.attn1.{to_q,to_k,to_v,to_out.0}.*`; MR-specific branches under `.transformer.attn1.processor.*_mr` | Yes | Base projections are used for albedo; registered `_mr` projections are separately used for MR. These are material-isolatable. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:552-576`; `$HYPAINT/hunyuanpaintpbr/unet/attn_processor.py:638-755` |
| Ordinary cross-attention | `.transformer.attn2.{to_q,to_k,to_v,to_out.0}.*` | Yes | Shared by albedo and MR in the flattened batch/material stream; name-isolatable as attention, not material-isolatable. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:639-663` |
| Feed-forward | `.transformer.ff.*` | Yes | Shared transformer computation for both materials; no safe albedo-only split. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:678-703` |
| Down/mid/up hierarchy | Main UNet `.down_blocks`, `.mid_block`, `.up_blocks`; each cross-attention transformer block is recursively replaced by `Basic2p5DTransformerBlock` | Yes | Resnets, norms, attention, and sampling layers are broad shared representation paths. Individual added attention projections can be isolated; entire blocks cannot. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:859-913` |
| Normal input | Normal images are VAE-encoded, then concatenated with noisy sample and position latent before shared `conv_in` | No dedicated trainable geometry projector | VAE is outside the UNet optimizer. Shared `conv_in` is trainable but cannot isolate normal channels through ordinary `requires_grad` selection. | `$HYPAINT/hunyuanpaintpbr/unet/model.py:314-357`; `$HYPAINT/hunyuanpaintpbr/unet/modules.py:959-970` |
| Position input | Position images are VAE-encoded and concatenated before shared `conv_in`; original position map also generates voxel indices | No dedicated trainable geometry projector | Same shared-conv limitation as normal. Position-to-index operations are computed, not learned. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:959-998` |
| 3D-aware masks and RoPE | `compute_voxel_grid_mask`, `compute_discrete_voxel_indice`, `calc_multires_voxel_idxs`, `PoseRoPEAttnProcessor`, `RotaryEmbedding` | No independent learned geometry weights | Mask/index/rotation calculations alter attention geometry but own no projection parameters. Their Q/K/V weights belong to `attn_multiview`. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:121-274,986-998`; `$HYPAINT/hunyuanpaintpbr/unet/attn_processor.py:367-465,553-635` |
| Albedo pathway | Albedo latent, learned albedo text tokens, base ordinary self-attention, base ref V/output, then shared main UNet output split | Mixed | Albedo tokens and ordinary/ref material branches can be isolated. Shared conv/resnet/FF/cross/multiview/output cannot. There is no dedicated albedo output head. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:833-857`; `$HYPAINT/hunyuanpaintpbr/unet/model.py:401-467`; self/ref evidence above |
| Metallic-roughness pathway | MR latent, learned MR text tokens, `_mr` self/ref processor projections, then shared main UNet output split | Mixed | MR-specific tokens and processor projections are isolatable; the remainder is shared. | `$HYPAINT/hunyuanpaintpbr/unet/attn_processor.py:308-363,638-839`; `$HYPAINT/hunyuanpaintpbr/unet/model.py:401-467` |
| DINO image projection | `UNet2p5DConditionModel.image_proj_model` with Linear and LayerNorm | Yes | Separately named but semantic/reference conditioning, not geometry conditioning. | `$HYPAINT/hunyuanpaintpbr/unet/modules.py:710-755,833-857,1000-1007` |

The architecture has trainable geometry-aware attention projections, but no
standalone trainable normal projector, position projector, or albedo output
head. Consequently, a name-based selective method can isolate
`attn_multiview` and material-specific attention pieces. It cannot honestly
label shared `conv_in`, `conv_out`, resnets, feed-forward layers, or ordinary
cross-attention as geometry-only or albedo-only.

## G. Current Full80 Trainable Scope

### Optimizer membership

The exact optimizer argument is `self.unet.parameters()`
(`$HYPAINT/hunyuanpaintpbr/unet/model.py:589-592`). `self.unet` is the complete
`UNet2p5DConditionModel` wrapper, including its main UNet, dual reference UNet,
added attention blocks, material-specific attention processor parameters,
learned PBR tokens, and DINO image projector. PyTorch therefore receives every
parameter object under this wrapper, including objects with
`requires_grad=False`.

### Effective update subset

Before optimizer construction, `set_learned_parameters()` applies this rule
(`$HYPAINT/hunyuanpaintpbr/pipeline.py:118-143`):

- Freeze a parameter if its name contains `attn1` or `unet_dual` and contains
  none of `albedo`, `mr`, or `dino`.
- Set every other wrapped-UNet parameter to trainable.

Thus full80-500 was broad partial fine-tuning, not complete-UNet fine-tuning and
not a small selective scope. It allowed updates to most main-UNet conv/resnet,
norm, ordinary cross-attention, feed-forward, multiview, refview, DINO,
material-specific, and learned-token parameters. Ordinary base `attn1` and most
of `unet_dual` were frozen by construction, subject to the string exceptions.

The VAE and text encoder live on the pipeline rather than under `self.unet`, so
they were not passed to this optimizer. The separate DINO encoder is created
outside `self.unet` and explicitly frozen in the DINO wrapper
(`$HYPAINT/hunyuanpaintpbr/unet/modules.py:38-57`). Its trainable image
projection inside the UNet wrapper was eligible to update.

### Remaining ambiguity

`RUNTIME_CONFIRMATION_REQUIRED`:

- the exact instantiated optimizer parameter names, tensor shapes, and counts;
- the exact trainable/frozen totals after all string rules and dynamic MR
  processor registration;
- whether any parameter was duplicated or omitted by module aliasing;
- which eligible parameters received nonzero gradients on a protocol-corrected
  batch;
- the actual optimizer-state footprint of the 500-step run.

A no-update runtime inventory should record parameter identity, full name,
shape, `requires_grad`, optimizer membership, and module class before Phase 2N
training is allowed.

## H. Candidate Phase 2N Selective Scopes

These are static candidates, not recommendations to run. Patterns are relative
to `HunyuanPaint.named_parameters()` and should match only names that actually
exist. Bias entries are included only when the corresponding Linear has a bias.

### Scope S1: Multi-view attention projections only

Candidate pattern:

```text
unet.unet.{down_blocks.*,mid_block,up_blocks.*}.attentions.*.
  transformer_blocks.*.attn_multiview.
  {to_q,to_k,to_v,to_out.0}.{weight,bias}
```

- Expected benefit: tune the projections directly used by cross-view attention
  and 3D-aware RoPE while preserving convolutional, text/reference, ordinary
  attention, feed-forward, and output paths.
- Preservation risk: the same multiview weights process both albedo and MR and
  occur at every UNet resolution. They can still shift global PBR consistency
  or propagate front texture into non-front views.
- Unresolved: exact module count, whether all resolutions are equally useful,
  and whether q/k-only would preserve appearance better than q/k/v/out.
- Week 1 runtime confirmation: exact matched names/count/parameter total,
  duplicate-identity check, zero unmatched patterns, nonzero gradients on a
  corrected-input batch, and step-0 output equivalence before any update.

### Scope S2: S1 plus clearly separable geometry-condition projections

Static extension set:

```text
{}
```

The audited graph exposes no dedicated trainable normal or position projection.
Both latents enter through channel slices of one shared `conv_in`; voxel masks
and RoPE indices are parameter-free. Therefore S2 currently collapses to S1 and
is **not a distinct runnable scope**. Broadly unfreezing `conv_in` would violate
the requested preservation boundary.

- Expected benefit if a runtime-discovered dedicated projector exists: permit
  geometry-condition adaptation without changing texture/noise channels.
- Preservation risk: a falsely classified shared projection would alter base
  denoising and both materials.
- Unresolved: whether the actual configured runtime wraps geometry inputs in a
  separately named module not visible in the static source path.
- Week 1 runtime confirmation: enumerate the instantiated module graph and
  trace normal/position tensors to every parameterized operation. Promote a
  module into S2 only if its parameter identities are exclusive to those
  condition paths. Otherwise reject S2 rather than using `conv_in`.

### Scope S3: S2 plus clearly separable late albedo/output modules

No separate albedo output head exists. The narrow static candidate is the
albedo-specific ordinary self-attention in the last up block, in addition to
S1 (and any future valid S2 extension):

```text
unet.unet.up_blocks.3.attentions.*.transformer_blocks.*.
  transformer.attn1.{to_q,to_k,to_v,to_out.0}.{weight,bias}
```

`SelfAttnProcessor` uses the base `attn1` projections for albedo and separate
`*_mr` processor projections for MR
(`$HYPAINT/hunyuanpaintpbr/unet/attn_processor.py:638-755`). This makes the
base late `attn1` projection set more separable than shared `conv_out`, FF, or
resnet weights. `up_blocks.3` is supported by the existing Phase 2M runtime
inventory, but its status as the final active high-resolution block remains a
runtime/config check.

- Expected benefit: give the late albedo representation limited adaptation
  capacity after geometry-focused multiview processing.
- Preservation risk: even late albedo self-attention can sharpen training-domain
  graphics, overfit train-sanity assets, or worsen backside leakage. It does not
  guarantee local output-only behavior.
- Unresolved: the exact active last-up-block index, whether these base `attn1`
  parameters are frozen at baseline, and whether gradients improve front views
  without degrading non-front views.
- Week 1 runtime confirmation: verify exact names/classes and material dispatch,
  compare S1/S3 trainable counts, prove MR-specific `_mr` modules and shared
  `conv_out` are excluded, and run one-step gradient/output-preservation audits
  before any scheduled experiment.

S1 is the only distinct scope fully justified by static source. S2 and S3 must
remain gated until runtime evidence confirms their claimed separation.

## I. Phase 2N Minimal Reuse Plan

Later implementation should have the smallest practical footprint:

1. Extend the exact-name inventory/validation ideas in
   `src/hy3dft/lora/targeting.py` to support named full-parameter scopes,
   parameter identity checks, and trainable/optimizer summaries. Keep the LoRA
   selectors unchanged for Phase 2M reproducibility.
2. Reuse the safe model/data lifecycle, project-local log paths, freeze
   assertions, and explicit optimizer construction from
   `scripts/phase2m_train_lora_refdino.py`. A selective full-parameter runtime
   should not use adapter injection or adapter-only IO.
3. Add one project-local protocol-corrected dataset implementation only if
   static review confirms the upstream dataset cannot accept
   `selected_input_view` and synchronized PBR augmentation. It should reuse
   upstream image loading where possible and must not patch upstream source.
4. Reuse the existing full80 true-PBR initialization and examples/splits. Add
   configuration fields for selected input view, scope name, exact parameter
   patterns, and preservation assertions rather than cloning the entire config
   stack.
5. Reuse full80 corrected-input case creation and
   `run_phase2g_paint_infer.py` for base/full-checkpoint outputs. Parameterize
   checkpoint variant labels only if needed.
6. Reuse `render_phase2k3_glb_views_blender.py` and Phase 2K metrics unchanged.
   Parameterize the Phase 2M render-config, aggregate, and board functions for
   arbitrary cases and variant columns.
7. Keep one common evaluation schema for base, full80-500, and S1/S2/S3. Do not
   introduce a Phase 2N-only renderer or metric implementation.

Likely later code changes are one generic selective-scope helper extension, one
protocol-corrected data path, one selective training entrypoint/config/launcher
with readiness tests, and small parameterization of the existing multi-variant
evaluation scripts. No such files or changes are made on Day 1.

## J. Unresolved Questions for Day 2-5

### Answerable by further static source reading

- Does the exact configured SD2 UNet topology always make `up_blocks.3` the
  final attention-bearing high-resolution block?
- Can the upstream dataset be cleanly subclassed while reusing `load_image`, or
  is a small project-local dataset clearer and safer?
- Which random transforms must share one sampled transform across albedo, MR,
  normal, and position to preserve pixel alignment?
- Which checkpoint loader path should be parameterized for Phase 2N variant
  names while preserving the existing base/full80 paths?
- Can the generic portions of the Phase 2M aggregator and board generator be
  lifted without changing historical M3C outputs?

### Requires CPU/local runtime

- Parse the configured UNet JSON/YAML topology and confirm the last up-block
  index without loading weights.
- Unit-test exact parameter pattern matching against a lightweight synthetic
  module tree, including duplicate identities and empty scopes.
- Unit-test deterministic selected-input resolution and one sampled transform
  applied identically to all PBR/geometry maps.
- Validate that generated configs and summaries stay project-local and never
  overwrite Phase 2L/2M artifacts.
- Exercise multi-variant aggregation and boards with synthetic PNGs and more
  than four arbitrary variant labels.

### Requires the planned A100 preflight

- Load the original true-PBR model and record exact S1/S2/S3 names, shapes,
  counts, classes, identities, and baseline `requires_grad` states.
- Confirm optimizer membership equals exactly the selected scope, with every
  non-selected parameter frozen and no selected parameter omitted.
- Trace normal/position tensors through the live model and determine whether
  any dedicated trainable geometry projection exists.
- Confirm final-up-block material dispatch and that S3 excludes MR-specific and
  shared output parameters.
- Run no-update and one-step checks for finite loss, finite gradients, nonzero
  selected gradients, zero non-selected gradients, memory use, and checkpoint
  reload equivalence.
- Verify the corrected dataset always selects view 005 (or the explicit curated
  view) and maintains target/normal/position alignment in an actual batch.
- Gate any pilot on step-0 equality with base and on front/non-front preservation
  reporting, not training loss alone.

## Day 1 Decision

Phase 2N should begin with protocol correction and an S1 runtime audit, not a
broad UNet rerun. The static graph supports exact isolation of multiview
attention projections. It does not support claiming a dedicated trainable
geometry projection or albedo output head. Existing corrected-input evaluation,
fixed-view rendering, metrics, and aggregation logic provide nearly all of the
evaluation infrastructure needed for a small multi-variant pilot.
