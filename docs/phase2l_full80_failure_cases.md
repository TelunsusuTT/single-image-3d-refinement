# Phase 2L Full80 Failure Cases

## How To Read The Boards

Rendered-view boards compare reference renders, corrected-input base outputs,
fine-tuned outputs, and difference images across views `000` through `005`.
They should be read alongside metrics, not replaced by metrics. A lower MAE can
still be visually ambiguous if the change is small, if the output shifts color
globally, or if a front-facing detail leaks onto another surface.

The main questions for each board are:

- Does fine-tuned output look closer to the reference than base?
- Is the improvement visible on the selected/front view or only on easy
  non-front views?
- Does the back or side surface receive front texture that should not be there?
- Are metric gains aligned with visible gains?
- Does the test split show the same behavior as validation?

## Representative Board Placeholders

Fill these with specific cases when preparing final figures:

- Front-to-back leakage:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/<split>/<item_id>_rendered_view_board.jpg`
- Non-front contamination:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/<split>/<item_id>_rendered_view_board.jpg`
- No visible improvement over base:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/<split>/<item_id>_rendered_view_board.jpg`
- Metric improvement with visual ambiguity:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/<split>/<item_id>_rendered_view_board.jpg`
- Test split regression:
  `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/test/<item_id>_rendered_view_board.jpg`

Known board directories:

- `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/val/`
- `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/test/`
- `outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/render_eval/boards/train_sanity/`

## Failure Categories

### Front-To-Back Leakage

Front-to-back leakage occurs when graphic information from the selected front
input view appears on the back or side of the object. This is the main observed
failure mode. It suggests the model is using front-view evidence, but lacks
enough information to decide where that evidence should stop.

### Non-Front Contamination

Non-front contamination is a broader category covering unwanted texture,
contrast, or color changes on views `000` through `003`. Some aggregate
non-front metrics improve, but board inspection can still show contamination if
the model adds plausible-looking detail to the wrong surface.

### No Visible Improvement Over Base

Several cases show base and fine-tuned outputs that are nearly identical in
rendered view. This is not an execution failure. It means the checkpoint's
change is too small to matter visually for that asset, even if small metric
differences are measurable.

### Metric Improvement But Visual Ambiguity

Small MAE or SSIM-like improvements can coexist with ambiguous visual changes.
For example, a color shift may reduce mean error while still failing to preserve
the intended graphic structure. These cases should not be counted as clear
successes without human inspection.

### Test Split Regression

The full80 test split has a slight MAE regression: base 6.4182 versus fine
6.4782. SSIM-like improves, so the result is mixed rather than simply worse.
This is a warning against claiming a robust held-out improvement.

## Interpretation

Leakage is not an engineering failure. The data pipeline, true-PBR
initialization, checkpointing, inference, and rendered-view evaluation all
worked. The limitation is that single-image texture generation is under
constrained for assets with distinct front and back surfaces. Without additional
view supervision, explicit surface semantics, or stronger curation/evaluation
rules, the model can learn useful front-view priors while still contaminating
unseen sides.
