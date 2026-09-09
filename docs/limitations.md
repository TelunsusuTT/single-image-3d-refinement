# Limitations

## Study scope

The evidence comes from a controlled, fixed-mesh study of 101 framed-panel
assets. This restricted category is useful for separating front and non-front
behaviour, but it does not represent the geometric or material diversity of
general image-to-3D generation. Results should not be assumed to transfer to
other asset classes, model versions, or camera conventions without new tests.

The Hunyuan3D-Paint experiments do not evaluate shape generation, geometry
correction, remeshing, or joint shape-and-texture adaptation.

## Hidden appearance is underdetermined

A single front-informative image does not observe the true side and rear
materials. Non-front rendered error is a repeatable proxy for hidden-surface
preservation, not direct evidence that the generated hidden appearance is
semantically correct. A method can reduce pixel error or visible leakage while
still producing an implausible unseen surface.

View-Selective Conditioning Gating suppresses direct reference-conditioning
branches for selected internal generation slots. It does not infer missing
material evidence. Multi-View Attention is not gated directly, and information
can still propagate through modified hidden states, later generation steps,
texture baking, and the final mesh representation.

## Metrics and visual review

MAE and the reported SSIM-like measure are sensitive to rendering alignment,
lighting, foreground masks, and small spatial shifts. Aggregate improvement can
also hide asset-level regressions. For these reasons, metrics are split by view
role and paired with visual inspection of leakage and newly introduced
artefacts.

The distinction between front views, selected input view, and non-front views
is specific to the controlled camera protocol. It should not be transferred to
a different camera rig by index alone.

## Small-data adaptation

The adaptation dataset contains 80 training assets. At this scale, broad
fine-tuning can overfit or drift from pretrained behaviour, while narrow update
strategies may lack sufficient capacity. The tested LoRA rank, target modules,
learning rate, scales, and update budget cover one constrained design rather
than the full LoRA design space.

The protocol-corrected Broad and MVA configurations differ not only in trainable
scope but also in peak learning rate and selected checkpoint. They are compared
as complete configurations; the evidence does not isolate parameter count as a
single causal factor.

## Selection and test history

The controlled split separates 80 training, 10 validation, and 11 test assets,
and the reported configurations are frozen before their final-test evaluation.
However, some test assets appeared in earlier completed sub-studies during
project development. The results are therefore locked, report-only tests for
the stated comparisons, not a claim of globally untouched assets across the
entire project history.

The validation and test sets are also small enough that a few difficult assets
can materially affect an aggregate. Per-asset tables and failure cases should
remain available whenever a result is presented.

## Supported conclusion

The experiments do not support a claim that any tested method universally
improves Hunyuan3D-Paint, eliminates front-to-back leakage, or reconstructs true
hidden materials. They support the narrower conclusion that:

- corrected conditioning is an important comparison control;
- small-data adaptation can improve front-oriented fidelity for some complete
  configurations;
- camera-aware inference gating can shift the balance toward lower non-front
  error;
- under the evaluated settings, neither route consistently improves both
  front-surface fidelity and hidden-surface preservation.
