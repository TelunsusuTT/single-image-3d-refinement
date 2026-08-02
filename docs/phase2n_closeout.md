# Phase 2N Closeout

## Executive Summary

Phase 2N is **CLOSED**. Controlled training, the validation-only selection
stage, frozen final-test inference, and final rendered-view evaluation all
completed successfully. The final test compared exactly
`corrected_input_base`, `historical_full80_500`, and
`pc_full_step320` on 11 test assets and 198 rendered views.

The result is useful but deliberately narrow. `pc_full_step320` improves mean
front and input-view fidelity over the corrected-input base, but its all-view
gain is modest and asset-dependent. It improves all-view mean MAE on 5 of 11
assets and regresses on 6. Non-front mean MAE is worse, and seven assets meet
the recorded leakage-regression criterion. The default safety recommendation
therefore remains `corrected_input_base`; `pc_full_step320` is an optional
front-quality-oriented checkpoint.

No further Phase 2N training, checkpoint, scope, learning-rate, dataset,
input-view, or evaluation-protocol change is permitted. Final-test evidence is
report-only.

## Research Question

The phase asked whether controlled partial fine-tuning of the official
Hunyuan3D-Paint true-PBR base could improve texture fidelity for flat,
rectangular graphic panels without sacrificing the corrected-input base's
safer non-front behavior. It compared a narrow attention scope, PC-S1, with a
broader PC-Full scope under matched data order and protocol.

## Frozen Dataset And Evaluation Protocol

The fixed Data v2 split contains:

| Split | Assets |
| --- | ---: |
| Train | 80 |
| Validation | 10 |
| Test | 11 |

Both training scopes used the same no-augmentation data, deterministic
MVA-active sampling schedule, selected input view `005`, and AL references.
Rendered evaluation used views `000-005`, with front views `004/005` and
non-front views `000-003`. Geometry remained fixed and remeshing was disabled.

The ten validation assets alone were used for Phase 2N candidate and checkpoint
selection. The 11 test assets were not used for that selection. They had,
however, been inspected previously during the historical Phase 2L full80
evaluation, so they are not described as unseen throughout the entire project.

## Historical Baselines

`corrected_input_base` is the official true-PBR base run with corrected input
view `005`. It is the safety reference and final default recommendation.

`historical_full80_500` is the earlier broad full80 checkpoint. It remains a
comparison baseline only. On the final test it slightly worsens all-view MAE
and non-front MAE relative to the corrected-input base, despite improving the
mean SSIM-like score.

## Phase 2N Training Scopes

Training run `slurm_264123` completed 320 optimizer updates per scope with a
50-step linear warmup, constant peak learning rate thereafter, gradient
clipping at 1.0, batch size 1, bf16 mixed precision, and checkpoints at steps
160 and 320.

| Scope | Trainable tensors | Trainable parameters | Peak LR | Role |
| --- | ---: | ---: | ---: | --- |
| PC-S1 | 80 | 49,574,080 | 1e-6 | Narrow `attn_multiview` safety probe |
| PC-Full | 981 | 1,046,761,668 | 5e-7 | Broad partial fine-tuning quality probe |

Both scopes loaded a fresh official true-PBR base, used the same accepted
320-record schedule, and saved scope-only model state. No test data,
validation, or inference was used during training.

## Pilot And Full-Validation Progression

The eight-case pilot used six validation and two train-sanity assets with zero
test assets. It compared the corrected-input base, historical full80-500, and
both scopes at steps 160 and 320. Human and numeric review retained
`pc_s1_step160` as the conservative validation candidate and
`pc_full_step320` as the stronger quality candidate. The other two
checkpoint-step combinations remained pilot-only ablations.

The complete validation then evaluated four variants on all ten frozen
validation assets and 240 rendered rows:

| Validation variant | All-view MAE | Delta MAE vs base | SSIM-like | Delta SSIM-like | Better/worse views |
| --- | ---: | ---: | ---: | ---: | ---: |
| `corrected_input_base` | 6.374827 | 0.000000 | 0.818142 | 0.000000 | baseline |
| `historical_full80_500` | 6.046239 | -0.328588 | 0.838309 | +0.020167 | 34 / 26 |
| `pc_s1_step160` | 6.391484 | +0.016657 | 0.817202 | -0.000940 | 24 / 36 |
| `pc_full_step320` | 5.925956 | -0.448871 | 0.842651 | +0.024508 | 35 / 25 |

This table contains validation evidence only. It is not combined with the final
test.

## Validation-Only Checkpoint Selection

The completed ten-asset validation review selected
`pc_full_step320`, scope `pc_full`, step 320, from training run
`slurm_264123`. Its validation means were better than the corrected-input
base for all views, input view `005`, front views, and non-front views. Four
validation assets met the leakage-regression criterion.

The frozen checkpoint SHA-256 is:

`b93f4d34291a21097b8d763b3d5c18d4b0a82796049ce0711aaac552b310d18e`

`pc_s1_step160` was excluded because it was not selected by the completed
validation review. It remains a validation ablation only.

## Final-Test Protocol

After the selection freeze, final-test inference generated exactly 11 new
`pc_full_step320` GLBs. The evaluation reused 22 compatible Phase 2L
baseline GLBs and compared exactly three variants:

1. `corrected_input_base`
2. `historical_full80_500`
3. `pc_full_step320`

The final matrix contains 11 test assets, zero validation assets, zero
train-sanity assets, 33 source GLBs, 198 rendered PNGs, 198 metric rows, and 22
boards. It uses selected input view `005`, AL lighting, fixed meshes, no
remeshing, and the unchanged six-view renderer and metric formulas. The
evaluation did not select a winner and did not allow checkpoint replacement.

## Final-Test Quantitative Results

All values below come only from the frozen 11-asset final test. Delta columns
are relative to `corrected_input_base`.

| View group | Variant | Rows | Mean MAE | Delta MAE | SSIM-like | Delta SSIM-like | Better/worse views |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | `corrected_input_base` | 66 | 6.418191 | 0.000000 | 0.853896 | 0.000000 | baseline |
| All | `historical_full80_500` | 66 | 6.478250 | +0.060058 | 0.872292 | +0.018396 | 28 / 38 |
| All | `pc_full_step320` | 66 | 6.312487 | -0.105705 | 0.883728 | +0.029833 | 33 / 33 |
| Input 005 | `corrected_input_base` | 11 | 13.410379 | 0.000000 | 0.814117 | 0.000000 | baseline |
| Input 005 | `historical_full80_500` | 11 | 13.333286 | -0.077094 | 0.822664 | +0.008548 | 5 / 6 |
| Input 005 | `pc_full_step320` | 11 | 12.885914 | -0.524466 | 0.833382 | +0.019265 | 7 / 4 |
| Front 004/005 | `corrected_input_base` | 22 | 13.403545 | 0.000000 | 0.814359 | 0.000000 | baseline |
| Front 004/005 | `historical_full80_500` | 22 | 13.326030 | -0.077515 | 0.822723 | +0.008364 | 10 / 12 |
| Front 004/005 | `pc_full_step320` | 22 | 12.874511 | -0.529033 | 0.833616 | +0.019257 | 14 / 8 |
| Non-front 000-003 | `corrected_input_base` | 44 | 2.925515 | 0.000000 | 0.873665 | 0.000000 | baseline |
| Non-front 000-003 | `historical_full80_500` | 44 | 3.054359 | +0.128845 | 0.897076 | +0.023412 | 18 / 26 |
| Non-front 000-003 | `pc_full_step320` | 44 | 3.031474 | +0.105960 | 0.908785 | +0.035120 | 19 / 25 |

PC-Full's all-view MAE change is favorable but small:
`-0.10570479884292135`. The result is not a uniform improvement; its 66
per-view MAE comparisons split evenly between 33 improvements and 33
regressions.

## Per-Asset Heterogeneity

Asset-level outcomes use the sign of each asset's mean MAE delta from the
corrected-input base.

| Variant | Group | Improved assets | Regressed assets | Equal |
| --- | --- | ---: | ---: | ---: |
| `historical_full80_500` | All views | 5 | 6 | 0 |
| `historical_full80_500` | Input 005 | 5 | 6 | 0 |
| `historical_full80_500` | Front 004/005 | 5 | 6 | 0 |
| `historical_full80_500` | Non-front 000-003 | 3 | 8 | 0 |
| `pc_full_step320` | All views | 5 | 6 | 0 |
| `pc_full_step320` | Input 005 | 7 | 4 | 0 |
| `pc_full_step320` | Front 004/005 | 7 | 4 | 0 |
| `pc_full_step320` | Non-front 000-003 | 4 | 7 | 0 |

Thus PC-Full improves a majority of assets only for the front/input grouping,
not for the all-view result. Its aggregate all-view gain is driven by effect
size on some assets rather than consistent asset-level wins.

## Front/Input Findings

PC-Full gives the clearest positive signal on observed front content:

- input-view mean MAE improves by `0.5244656187115291`;
- front-view mean MAE improves by `0.5290334874933413`;
- input view improves for 7 of 11 assets and 7 of 11 view rows;
- front views improve for 7 of 11 assets and 14 of 22 view rows.

This supports an optional front-quality role. It does not support replacing the
base as an unconditional default.

## Non-Front And Leakage Findings

Non-front mean MAE does not improve. It changes from
`2.9255145968812886` for the base to `3.0314741423635776` for PC-Full, a
regression of `0.10595954548228867`. Only 4 of 11 assets improve on
non-front mean MAE, while 7 regress.

The recorded leakage-risk test identifies seven PC-Full regression assets:
`B073NZS586`, `B073P1JNZZ`, `B075HXJ6Q9`, `B073P1N4C4`,
`B073P5FLX9`, `B073P1H7MS`, and `B075YNL763`. Visual and numeric
evidence therefore leave hidden-surface and front-to-back leakage unresolved.
This is a model limitation under single-image conditioning, not an
infrastructure failure.

## Comparison With Historical Full80

Historical full80-500 is not recommended. On the final test its all-view MAE
is `6.478249520966501`, worse than the base's
`6.418191389604048`. Its non-front MAE also regresses by
`0.12884469465775916`, only 3 of 11 assets improve on non-front mean MAE,
and eight assets meet the leakage-regression criterion. Its higher mean
SSIM-like score does not erase those safety and consistency concerns.

## PC-S1 Ablation Conclusion

PC-S1 step 160 remains a useful narrow-scope ablation but is not a final-test
or deployment model. On complete validation its all-view MAE delta was
`+0.016656939188639325`, its SSIM-like delta was
`-0.000940259609546167`, and 24 of 60 views improved while 36 worsened.
Its non-front validation mean did not beat the base, and seven validation
assets met the leakage-regression criterion. It was therefore excluded before
the test split was run.

## Final Model Roles

| Role | Model |
| --- | --- |
| Default and safety recommendation | `corrected_input_base` |
| Optional front-quality-oriented model | `pc_full_step320` |
| Historical comparison only | `historical_full80_500` |
| Validation ablation only | `pc_s1_step160` |

## Deployment Recommendation

Use `corrected_input_base` by default. It has the safer non-front MAE profile
and avoids presenting the modest PC-Full aggregate gain as universal.

Use `pc_full_step320` only when observed front/input fidelity is the explicit
priority and the downstream workflow can tolerate asset-dependent behavior and
review non-front surfaces. Do not use `historical_full80_500` as the preferred
model.

## Limitations And Uncertainty

- The final test contains 11 assets from one narrow frame-panel subclass.
- Asset-level outcomes are heterogeneous, and mean metrics can be dominated by
  larger changes on a minority of assets.
- MAE and SSIM-like trends disagree in some non-front comparisons.
- Hidden and backside texture correctness cannot be inferred reliably from one
  observed input image.
- The test assets were held out from Phase 2N selection, but the same split had
  been evaluated previously in Phase 2L.
- The final test estimates behavior under this frozen protocol; it does not
  establish broad deployment generalization.

## Valid Final Claims

- Phase 2N selected `pc_full_step320` using the ten-asset validation split
  only.
- PC-Full improves mean final-test front and input-view fidelity relative to
  the corrected-input base.
- Its all-view MAE gain is modest and asset-dependent.
- It improves 5 of 11 assets and regresses 6 of 11 on all-view mean MAE.
- Its non-front mean MAE does not improve.
- Hidden-surface and front-to-back leakage remain unresolved.
- The corrected-input base remains the default recommendation, with PC-Full as
  an optional front-quality model.

## Claims That Must Not Be Made

- PC-Full is universally or consistently superior to the corrected-input base.
- PC-Full improves a majority of test assets on all-view mean MAE.
- Phase 2N solved hidden-surface or front-to-back leakage.
- Historical full80-500 is recommended for deployment.
- PC-S1 step 160 is a selected final model.
- The 11 test assets were never inspected anywhere else in the project.
- Test evidence authorizes additional Phase 2N tuning or checkpoint selection.
- Eleven assets establish broad generalization beyond the frozen protocol.

## Reproducibility And Artifact Map

Authoritative machine record:

- `configs/phase2n_closeout.json`

Training:

- `outputs/phase2n/week2_pilot_training/slurm_264123/`

Selection freeze:

- `configs/phase2n_final_test_freeze.json`
- freeze SHA-256:
  `d7f65ea6c6f7fd2002f6cf39969d8cc7aca4f3412203eb3ea74ce2d3d48be65e`
- frozen project Git commit:
  `b940a737e1cdec4199a0ea3c8a433a99997647fd`

Validation:

- `outputs/phase2n/full_validation_rendered_eval/phase2n_full_validation_eval_v1/`
- analysis packet:
  `logs/local/phase2n_full_validation_eval_v1_analysis_packet.tar.gz`
- packet SHA-256:
  `1b395c6afbee83509abb319fb6384e25c2bd7d03c5f04596489195c1e17b2d5f`

Final test:

- inference:
  `outputs/phase2n/final_test_inference/phase2n_final_test_infer_v1/`
- rendered evaluation:
  `outputs/phase2n/final_test_rendered_eval/phase2n_final_test_eval_v1/`
- analysis packet:
  `logs/local/phase2n_final_test_eval_v1_analysis_packet.tar.gz`
- packet SHA-256:
  `20afab5dbde9c9b911de1002004fc2c9f1b84774bb6a845fc4354c450b064e43`

Read-only verification:

```bash
python scripts/check_phase2n_closeout.py \
  --config configs/phase2n_closeout.json
```

A valid closeout prints `PHASE2N_CLOSEOUT_OK`.

## Final Phase 2N Status

Phase 2N is **CLOSED**. Training, validation-only selection, the frozen final
test, quantitative aggregation, visual boards, and artifact packaging are
complete. Test results are report-only. No further Phase 2N model,
checkpoint, scope, learning-rate, dataset, input-view, or protocol change is
permitted.

## Recommended Next Project Activity

Proceed to report writing and broader project consolidation. Any future
scientific work should be defined as a new phase with a new hypothesis and
protocol, not as test-driven continuation of Phase 2N.

