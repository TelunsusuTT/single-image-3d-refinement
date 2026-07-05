# Phase 2L.7C Full80 Evaluation Closeout

## Phase Goal

Phase 2L.7C closes out the full80 true-PBR 500-step experiment and turns the
rendered-view evaluation into report-ready project notes. The goal is not to
claim that the fine-tuned model is clearly better than corrected-input base, but
to record what was tested, what improved marginally, what failed visually, and
why a full80-1000 run should not be started yet.

## Completed Artifacts

- Full101 rendered Hunyuan3D-Paint training examples:
  `data/hy3dpaint_train_examples/datav2_frame_panels_full101/`
- Full80 true-PBR checkpoint:
  `checkpoints/datav2_frame_full80_truepbr_500_lr1e6/datav2_frame_full80_truepbr_500_lr1e6-stepstep=500.ckpt`
- Full80 corrected-input base and fine-tuned inference outputs:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/infer/`
- Rendered-view evaluation outputs:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/`
- Aggregated evaluation summary:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/summary/full80_eval_summary.md`
- Selected board directory:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/`

Representative boards to inspect:

- `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/val/B075YLQTNP_rendered_view_board.jpg`
- `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/test/B073P1H6D8_rendered_view_board.jpg`
- `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/train_sanity/B073P16J7Y_rendered_view_board.jpg`

## Main Metrics

Lower MAE is better. Higher SSIM-like is better. Delta values are
fine-tuned minus base, so negative MAE deltas and positive SSIM-like deltas are
favorable.

| Group | Views | Base MAE | Fine MAE | Delta MAE | Base SSIM-like | Fine SSIM-like | Delta SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| all views | 144 | 6.1626 | 6.0654 | -0.0972 | 0.8466 | 0.8629 | +0.0163 |
| input view 005 | 24 | 11.6160 | 11.5530 | -0.0630 | 0.8357 | 0.8373 | +0.0017 |
| front 004/005 | 48 | 11.6206 | 11.5509 | -0.0697 | 0.8356 | 0.8373 | +0.0017 |
| non-front 000-003 | 96 | 3.4336 | 3.3227 | -0.1109 | 0.8521 | 0.8757 | +0.0236 |
| val | 60 | 6.3748 | 6.0462 | -0.3286 | 0.8181 | 0.8383 | +0.0202 |
| test | 66 | 6.4182 | 6.4782 | +0.0601 | 0.8539 | 0.8723 | +0.0184 |
| val+test | 126 | 6.3975 | 6.2725 | -0.1250 | 0.8369 | 0.8561 | +0.0192 |
| train_sanity | 18 | 4.5181 | 4.6158 | +0.0977 | 0.9147 | 0.9103 | -0.0044 |

## Comparison to Mini40

Mini40-500 was stable and showed a clearer train-sanity learning signal, but it
did not improve held-out front/input views. Its val+test MAE worsened from
5.6161 to 5.8273 and SSIM-like decreased from 0.8767 to 0.8651. Front-view
metrics also worsened.

Full80-500 is more encouraging numerically: val+test MAE improves from 6.3975
to 6.2725 and SSIM-like improves from 0.8369 to 0.8561. However, the effect is
small, the test split MAE regresses slightly, and train-sanity no longer shows a
clear positive signal. The result is therefore best described as stable and
marginally positive in aggregate, but visually mixed.

## Visual Findings

The rendered boards show that the fine-tuned checkpoint can make small local
changes without consistently improving the view-space result. Some cases look
slightly closer to the reference, but others remain similar to base or show
ambiguous changes. The most important visual issue is not a crash, data-format
problem, or training initialization failure. It is that a single selected input
view often does not provide enough information to reconstruct coherent texture
behavior across front, side, and back views.

## Failure Mode

The dominant failure mode is front-to-back leakage or backside contamination:
visual evidence from the informative front view can appear on non-front regions
where it should not. This is especially visible when the object has a clear
graphic front and a simpler or different back. Fine-tuning on the current data
does not reliably solve this ambiguity.

## Final Stage Diagnosis

The engineering pipeline is now functioning:

- true-PBR initialization is fixed
- full101 rendering and strict example checks are complete
- full80 training reaches 500 steps and saves a checkpoint
- corrected-input base and fine-tuned inference run on val/test cases
- rendered-view evaluation produces interpretable metrics and boards

The remaining issue is methodological/modeling rather than infrastructure:
single-image texture generation has limited information about unseen surfaces,
and the current fine-tuning signal is not strong enough to overcome that.

## Why Not Full80-1000 Now

A 1000-step run would cost more time without a strong signal that more steps are
the limiting factor. Full80-500 is stable, but the gains are small, mixed across
splits, and visually ambiguous. The test split MAE regresses slightly, and
train-sanity does not improve. Running longer now risks optimizing noise or
amplifying leakage rather than producing a cleaner conclusion.

## Recommended Next Paths

### A. Report-First Closeout

Use the current result as the main experimental endpoint. The report can state
that corrected-input evaluation and true-PBR initialization were essential, that
Data v2 full80 produced a stable but modest aggregate improvement, and that
front-to-back leakage remains the main limitation.

### B. Optional Low-LR Rescue

Only if time allows, consider one conservative rescue run with a lower learning
rate or stronger regularization target. This should be treated as exploratory
and should not delay the report unless it can be run and evaluated quickly. The
success criterion should be a clear held-out front-view improvement without
increased non-front contamination.
