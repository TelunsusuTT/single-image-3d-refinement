# Results

## Summary

The experiments support a trade-off-based conclusion. Corrected conditioning
provides a stronger and more consistent comparison point, but none of the
tested adaptation or control strategies consistently improves both visible
front-surface fidelity and hidden-surface preservation.

| Configuration | Supported observation | Interpretation |
| --- | --- | --- |
| Corrected-Conditioning Baseline | Front-informative conditioning is associated with higher rerendering fidelity in the four diagnostic training cases. | Use conditioning view 005 as the common input rule. |
| Broad-Scope Fine-Tuning | The 32- and 80-asset, 500-update runs alter appearance, but held-out gains vary across assets and splits. | More broad-scope adaptation is not automatically better. |
| Reference-Conditioning LoRA | The rank-4 adapter shows no advantage in the evaluated held-out pilot across scales 0.50, 0.75, and 1.00. | A narrower update alone does not resolve the problem under this setting. |
| Protocol-Corrected Broad-Scope Fine-Tuning | Front-oriented fidelity improves more clearly, but gains vary and can coincide with worse non-front behaviour. | Useful as an optional front-quality configuration, not a universal replacement for the baseline. |
| Protocol-Corrected Multi-View-Attention Fine-Tuning | Produces smaller deviations with far fewer trainable parameters, but no clear advantage in either objective. | Selective capacity is conservative but not sufficient here. |
| View-Selective Conditioning Gating | Non-front rendered error tends to decrease, while front and conditioning-view fidelity decrease; the all-view benefit is inconsistent. | Gating changes the fidelity/preservation balance rather than solving both objectives. |

## Evidence convention

The numeric values below are compact summaries from the frozen run aggregates.
Source GLBs, rendered PNGs, and per-view metric files are generated artefacts
and are intentionally kept outside Git; they are required for independent
recomputation and remain covered by the identity requirements in
[Reproducibility](reproducibility.md).

A **leakage-risk regression** means that a candidate has higher per-asset mean
MAE than the Corrected-Conditioning Baseline over non-front evaluation views
000–003 (candidate-minus-baseline delta greater than zero). This is a numeric
screening proxy, not by itself a semantic judgement that visible content has
leaked; the corresponding renders still require visual review.

## Broad-Scope Fine-Tuning

The early true-PBR 80-asset, 500-update experiment is stable and marginally
positive in its full aggregate: all-view MAE changes from 6.1626 to 6.0654 and
the SSIM-like score from 0.8466 to 0.8629. The held-out validation-plus-test
aggregate also improves, but the test-only MAE changes from 6.4182 to 6.4782.
This split dependence, together with mixed visual boards, prevents a general
improvement claim.

The dominant qualitative failure is transfer of recognisable front content to
side or rear surfaces. That observation motivates separating front fidelity
from non-front behaviour in all later comparisons.

## Protocol-corrected adaptation

On the ten-asset validation split, Protocol-Corrected Broad-Scope Fine-Tuning
shows a mean all-view MAE change of `-0.4489` and a mean SSIM-like change of
`+0.0245` relative to the Corrected-Conditioning Baseline. Front, input, and
non-front mean metrics improve in that validation aggregate, although four
assets show non-front leakage-risk regressions.

The frozen 11-asset evaluation is less uniform. The Broad configuration has a
modest all-view MAE gain and stronger mean front/input fidelity, but improves
all-view mean MAE for only 5 of 11 assets. Mean non-front MAE regresses, and
seven assets meet the predefined leakage-risk regression criterion. The
Corrected-Conditioning Baseline therefore remains the conservative default;
the Broad checkpoint is an optional front-quality configuration.

Protocol-Corrected Multi-View-Attention Fine-Tuning updates about 2.53% of the instantiated Paint
UNet, compared with about 53.35% for the Broad configuration. Its smaller output
changes do not translate into a clear fidelity or preservation advantage in
the evaluated cases, so it remains a controlled comparison rather than the
selected final configuration.

## View-Selective Conditioning Gating

Gating suppresses the rear internal generation slot in Reference Attention and
DINO conditioning while leaving model weights unchanged. The evaluated setting
often reduces non-front rendered error, but the effect varies by asset and
comes with a measurable front/conditioning-view cost. It does not reconstruct
unobserved materials and it does not guarantee removal of front-to-back
leakage.

The appropriate interpretation is therefore operational: the gate can reduce
part of the direct reference-conditioning transfer to a rear-facing generation
slot. It is not evidence that the model has learned the true hidden appearance.

## Overall conclusion

The strongest evidence from this controlled study is that conditioning protocol
and intervention location both matter, but the two target objectives remain in
tension. Small-data weight adaptation can improve visible-front quality under
some configurations. Inference-time gating can shift behaviour toward more
stable non-front renders. Neither intervention consistently improves both, and
all reported comparisons should retain per-asset results and visible failure
cases alongside aggregate metrics.
