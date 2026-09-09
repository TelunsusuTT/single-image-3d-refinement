# Experiments

## Controlled dataset

The study uses 101 manually screened framed-panel assets derived from Amazon
Berkeley Objects (ABO):

| Split | Assets | Role |
| --- | ---: | --- |
| Training | 80 | Parameter adaptation only |
| Validation | 10 | Configuration and checkpoint selection |
| Test | 11 | Reporting after the configuration is frozen |

Geometry-informed grouping is used while constructing the split to reduce the
risk of closely related shapes crossing split boundaries. Assets are centred,
scale-normalised, and represented in the official Hunyuan3D-Paint multi-view
PBR training-example structure. Automated structure checks and manual framing
review form the dataset quality-control layer.

The source assets and generated data are not versioned in this repository.
Their use remains subject to the ABO terms and the provenance recorded in the
dataset manifests.

## Common controls

Unless a method definition says otherwise, comparisons share:

- a fixed input mesh with `use_remesh=false`;
- the official Hunyuan3D-Paint PBR initialisation;
- conditioning view 005;
- a fixed input resolution and seed, with scheduler and guidance inherited from
  the same upstream runtime within each comparison;
- six predefined external evaluation cameras;
- common rendering, foreground handling, and metric implementations.

The conditioning view, internal generation-view slots, and evaluation cameras
remain separate pieces of metadata throughout the pipeline.

## Canonical configurations

The public configuration names match the terminology used in the report.
Run/checkpoint labels such as a step number are configuration details rather
than method names.

| Method | Canonical configuration | Evidence role |
| --- | --- | --- |
| Corrected-Conditioning Baseline | `configs/baseline/corrected_conditioning_baseline.json` | Common comparison baseline |
| Broad-Scope Fine-Tuning | `configs/adaptation/broad_scope_finetuning.json` | Early broad-adaptation study |
| Reference-Conditioning LoRA | `configs/adaptation/reference_conditioning_lora.json` | Constrained adaptation study |
| Protocol-Corrected Broad-Scope Fine-Tuning | `configs/adaptation/protocol_corrected_broad_finetuning.json` | Quality-oriented corrected-protocol candidate |
| Protocol-Corrected Multi-View-Attention Fine-Tuning | `configs/adaptation/protocol_corrected_mva_finetuning.json` | Selective-scope corrected-protocol comparison |
| View-Selective Conditioning Gating | `configs/gating/view_selective_conditioning_gating.json` | Frozen inference-time control |
| Fixed-View Rerendering | `configs/evaluation/fixed_view_rerendering.json` | Shared external evaluation |

## Adaptation comparison

The protocol-corrected Broad and MVA configurations use the same training split,
accepted sequence of examples, target order, conditioning probabilities,
augmentation setting, and 320-update horizon. Their principal controlled
difference is trainable scope, although peak learning rate and selected
checkpoint also differ. They should therefore be interpreted as complete
configuration comparisons, not as a single-factor proof of the effect of
parameter count.

Reference-Conditioning LoRA is evaluated as one trained adapter at three scales.
The scales alter inference strength; they are not three independently trained
models.

## Gating comparison

The gating study compares the Corrected-Conditioning Baseline against a frozen
camera-aware gate with a `120` degree threshold. The gate acts only on Reference
Attention and DINO-conditioning branch outputs. It introduces no checkpoint,
adapter, weight merge, or training stage.

An all-ones mask is used as an operator sanity check. The scientific comparison
uses the camera-derived mask `[1, 1, 0, 1, 1, 1]`. Runtime checks verify branch
inventory, call counts, selected-slot suppression, preservation of kept values,
base-weight sentinels, and restoration of patched state.

## Evaluation and selection discipline

Evaluation rerenders each generated GLB from views 000 through 005. The reports
separate front views 004/005, selected input view 005, non-front views 000–003,
and the all-view aggregate. Quantitative comparisons are paired by asset and
view. Visual boards support review of content leakage and newly introduced
artefacts.

Configuration choices are made using training-sanity and validation evidence.
The method and evaluation protocol are then frozen before final-test reporting.
No final-test observation may be used to change a threshold, trainable scope,
checkpoint, camera policy, or evaluation rule.

The 11 test assets were reserved for the frozen comparisons reported here, but
some appeared in earlier completed sub-studies during project development. The
final results should therefore be described as locked, report-only evaluations
for the stated configuration—not as evidence from assets that were globally
unseen throughout the entire research history.
