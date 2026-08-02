# Current Project State

## Summary

Phase 2N is **CLOSED**. Controlled training, validation-only selection, frozen
final-test inference, and the 11-asset rendered final test are complete.
`corrected_input_base` is the default safety recommendation, while
`pc_full_step320` is an optional front-quality checkpoint. PC-Full improves
mean front/input fidelity but does not improve non-front mean MAE, and
hidden-surface/front-to-back leakage remains unresolved.

No further Phase 2N training, checkpoint replacement, model selection, or
test-driven tuning is permitted. The next activity is report writing and
broader project consolidation. Phase 2M refview+DINO LoRA remains a completed
negative result on held-out cases.

## Initialization Fix

Earlier 50-step and 500-step checkpoints were initialized from an SD2/non-PBR
base, not from the official Hunyuan3D-Paint PBR inference weights. Phase 2J
located the local `hunyuan3d-paintpbr-v2-1` pipeline, verified training
initialization equivalence, and proved that official `train.py` can save a
base-equivalent checkpoint.

The later Data v2 runs use true-PBR initialization and do not show the old PBR
collapse failure.

## Mini40-500 Diagnostic Result

The mini40 true-PBR 500-step checkpoint was stable and showed a train-sanity
learning signal, but it did not clearly improve held-out corrected-input
front-view behavior. Rendered-view evaluation showed mixed or negative
validation/test behavior, especially on front/input views. This justified
scaling the dataset before increasing the step count.

## Full80-500 Result

The full80 true-PBR 500-step checkpoint completed successfully:

`checkpoints/datav2_frame_full80_truepbr_500_lr1e6/datav2_frame_full80_truepbr_500_lr1e6-stepstep=500.ckpt`

Corrected-input rendered-view evaluation completed with 24 cases and 144 rows:

`outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/summary/full80_eval_summary.md`

The result is stable and marginally positive in aggregate, but visually mixed:

- all views MAE improved from 6.1626 to 6.0654
- all views SSIM-like improved from 0.8466 to 0.8629
- val+test MAE improved from 6.3975 to 6.2725
- val+test SSIM-like improved from 0.8369 to 0.8561
- test MAE regressed slightly from 6.4182 to 6.4782
- train-sanity MAE regressed from 4.5181 to 4.6158

The main observed failure mode is front-to-back leakage or backside
contamination. This is a modeling/evaluation limitation of single-image texture
generation, not an infrastructure failure.

## Phase 2M LoRA Result

The refview+DINO LoRA adapter trained and ran successfully, but all evaluated
scales degraded held-out validation/test and front-view results compared with
corrected-input base. It remains a documented negative baseline and is not
selected for full evaluation expansion.

## Phase 2N Status

Week 1 Days 1-5 are complete. The static audit established that full80-500 was
broad partial fine-tuning, while PC-S1 isolates 80 `attn_multiview` projection
tensors. Day 3 and Day 4 established the deterministic protocol-corrected
reader, exact scope selection, optimizer membership, and a 50-step
warmup-constant scheduler. Day 4B selected spatial augmentation `none` for both
controlled pilots.

Day 5 loaded the official true-PBR base on an 80 GB A100 and passed production
loss, gradient, update, memory, scheduler, and selective reload checks. Numeric
review authorized PC-S1 at peak LR `1e-6` and PC-Full at peak LR `5e-7` for a
320-update controlled pilot with checkpoints at 160 and 320.

Job `264014` exposed a schedule contract failure: update 1 used candidate seed
`2775693311`, whose audited MVA/reference branch draw
`0.09623141240259903` disabled MVA and therefore gave PC-S1 an expected
all-zero gradient. Original update 8 was also MVA-inactive. The job stopped
before metrics/traces, a valid optimizer update, or a checkpoint, and PC-Full
did not start.

The repaired shared schedule preserves all four 80-asset permutations and
original candidate seeds, then deterministically retries only MVA-inactive
production-loss seeds. It has 320 MVA-active records, zero inactive records,
and 50 records with at least one retry. The same accepted schedule drives both
scopes; Day 3 reference/light decisions, no augmentation, data, learning rates,
warmup, checkpoint steps, scopes, and the frozen six-validation/two-train-sanity
evaluation cases remain unchanged. The repaired controlled run completed under
`outputs/phase2n/week2_pilot_training/slurm_264123` and retained four audited
scope-only checkpoints: PC-S1 and PC-Full at updates 160 and 320.

The A100 inference stage is complete under
`outputs/phase2n/week2_pilot_inference/slurm_264581`. PC-S1 and PC-Full at
steps 160 and 320 each produced eight successful fixed-mesh outputs. Existing
corrected-input base and historical full80 GLBs provide complete compatible
coverage for those same cases. No baseline inference was rerun, checkpoint
state did not accumulate across variants, and no test case was used.

Stage 5 used a local, process-isolated Blender rendered-view comparison of six
variants: corrected-input base, historical full80-500, and PC-S1/PC-Full at
steps 160/320. It used the frozen six validation and two train-sanity cases, AL
references, views `000-005`, and the historical Phase 2K camera and metric
implementations. The completed plan contains 48 GLBs and 288 PNG renders.

The first local Stage 5 run, `phase2n_week2_eval_v1`, reached a clean Blender
3.6 background startup but stopped before the first GLB import. The internal
worker action was placed after the Blender `--` boundary while the project
parser still consumed the unsliced Blender argv. This was a CLI routing failure
only: no render, metric, model, GLB, or checkpoint result was invalidated. The
failed run remains read-only, and the repair leaves the scientific
render/evaluation protocol unchanged.

The repaired run `phase2n_week2_eval_v2` then completed all 288 renders and
metric rows for six validation and two train-sanity assets. Human and numeric
review retained `pc_s1_step160` as the conservative safety candidate and
`pc_full_step320` as the stronger quality candidate. `pc_s1_step320` and
`pc_full_step160` remain useful pilot-only checkpoint-step ablations and will
not be expanded.

Full validation completed on exactly all ten assets in the frozen full101
validation split, with zero train-sanity and zero test assets. The clean run
contains 40 source GLBs and 240 six-view rows. Review selected
`pc_full_step320` as the quality-oriented final candidate: validation MAE delta
was `-0.44887108272976345`, SSIM-like delta was
`+0.024508462795182862`, front/input/non-front means were better, and four
assets had non-front leakage regressions. `pc_s1_step160` remains a validation
ablation and is excluded from final test.

The frozen final test is complete under
`outputs/phase2n/final_test_rendered_eval/phase2n_final_test_eval_v1`. It
contains exactly 11 test assets, three variants, 198 rendered rows, zero
validation rows, and zero train-sanity rows. PC-Full has a modest all-view MAE
gain and stronger mean front/input fidelity, but improves only 5 of 11 assets
on all-view mean MAE. Non-front mean MAE regresses and seven assets meet the
leakage-regression criterion. These report-only results do not authorize
further tuning.

## No Full80-1000 Yet

Do not prepare or run full80-1000 yet. Full80-500 is not a strong enough signal
to justify a longer run by default. The improvements are small, split behavior
is mixed, and visual boards still show leakage/ambiguity.

## Current Decision

Keep `corrected_input_base` as the default safety recommendation and treat
`pc_full_step320` as an optional front-quality checkpoint. Keep
`historical_full80_500` for comparison only and `pc_s1_step160` as a
validation ablation only. Phase 2N is closed; proceed to report writing and
broader project consolidation without additional Phase 2N training or
test-driven tuning.
