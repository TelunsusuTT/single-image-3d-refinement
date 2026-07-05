# Phase 2L Report Materials

## Dataset Construction

We constructed an input-view-aware Data v2 subset focused on flat rectangular
graphic-panel assets. The final rendered dataset contains 101 Hunyuan3D-Paint
style training examples, split into 80 training assets, 10 validation assets,
and 11 test assets. Each example follows the official training layout with
`render_tex/`, `render_cond/`, and `render_tex/transforms.json`. The curated
assets include an explicit selected input view, usually view `005`, so that base
and fine-tuned evaluation use the most informative reference image available.

## Training Setup

Training used the official Hunyuan3D-Paint `train.py` with true PBR
initialization from the local `hunyuan3d-paintpbr-v2-1` weights. This avoided
the earlier wrong-initialization failure mode where training began from an
SD2/non-PBR base. The full80 experiment trained for 500 steps at learning rate
`1e-6`, batch size 1, six views, and 512 pixel view size, saving a single
step-500 checkpoint.

## Evaluation Protocol

Evaluation compared corrected-input base inference against corrected-input
fine-tuned inference. For each validation, test, and train-sanity asset, base
and fine-tuned runs used the same selected input image. The resulting textured
meshes were rendered from the six fixed Hunyuan-style views and compared against
the corresponding `render_cond` references. Metrics were reported separately
for all views, selected/input view `005`, front views `004` and `005`,
non-front views `000` through `003`, validation, test, validation plus test, and
train-sanity groups.

## Results

The full80-500 checkpoint was stable and marginally positive in aggregate. Over
all 144 rendered views, MAE decreased from 6.1626 to 6.0654 and SSIM-like
increased from 0.8466 to 0.8629. On the held-out validation plus test set, MAE
decreased from 6.3975 to 6.2725 and SSIM-like increased from 0.8369 to 0.8561.
However, the effect was small and not uniformly positive: the test split MAE
increased slightly from 6.4182 to 6.4782, and train-sanity MAE also increased
from 4.5181 to 4.6158.

## Limitations

The main limitation is front-to-back leakage or backside contamination. The
model can use the selected front input view, but it does not reliably know where
front-view graphic evidence should stop on assets with different front and back
surfaces. This limitation is expected for single-image texture generation and
should not be interpreted as a data-format, checkpointing, or initialization
failure. It does mean that metric improvements require visual inspection before
they can support a quality claim.

## Final Conclusion

The project successfully repaired the training initialization path, built a
curated input-view-aware Data v2 frame-panel dataset, trained a stable full80
true-PBR checkpoint, and evaluated it with corrected-input rendered-view
metrics. The full80 checkpoint shows a small aggregate improvement over
corrected-input base, but the result is visually mixed and not strong enough to
claim clear superiority. The most defensible conclusion is that the current
pipeline is technically sound, while single-view supervision remains
insufficient to resolve front/back texture ambiguity for this asset class.
