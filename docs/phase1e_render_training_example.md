# Phase 1E Render Training Example

Phase 1E creates a minimal Hunyuan3D-Paint-style training example from the one
downloaded GLB. This is a structure and rendering smoke test, not final
high-quality training data.

This phase does not train a model or run Hunyuan. The user runs Blender
manually to render the sample.

## Expected Structure

The rendered sample should be listed from an `examples.json` file containing a
JSON list of sample directories:

```json
[
  "data/hy3dpaint_train_examples/phase1e/B07H469871"
]
```

Each sample directory should contain:

```text
B07H469871/
  render_tex/
    000.png
    000_albedo.png
    000_mr.png
    000_normal.png
    000_pos.png
    ...
    005.png
    005_albedo.png
    005_mr.png
    005_normal.png
    005_pos.png
    transforms.json
  render_cond/
    000_light_AL.png
    000_light_ENVMAP.png
    000_light_PL.png
    ...
    005_light_AL.png
    005_light_ENVMAP.png
    005_light_PL.png
  sample_summary.json
```

Use `num_view=6` and `resolution=512` for the first smoke render.

## Dataloader Naming Rules

The official Hunyuan3D-Paint dataloader discovers training views from
`*_albedo` files. The albedo file is the anchor. For each albedo file, the
matching `_mr`, `_normal`, and `_pos` files must use the same prefix:

```text
000_albedo.png
000_mr.png
000_normal.png
000_pos.png
```

The condition images must also be paired by prefix. For each view, all three
lighting condition files should exist:

```text
000_light_AL.png
000_light_ENVMAP.png
000_light_PL.png
```

Phase 1E only checks that these files exist and are named correctly. The first
renderer uses approximate Blender material overrides for albedo,
metallic-roughness, normal, and position passes. Improving render quality and
matching the official renderer more closely can happen after the structure smoke
test succeeds.

## Phase 1E.1 Automatic Framing

Visual inspection of the first structure smoke render showed that fixed camera
settings can crop or mis-center assets with unusual proportions. Manual
per-asset camera adjustment is not acceptable because later phases need a
repeatable conversion path for multiple ABO, Objaverse, or synthetic assets.

The renderer now uses object-centric camera framing:

- compute a world-space bounding box across all imported mesh objects
- translate imported root objects so the bounding box center is at the origin
- uniformly scale the asset so its largest extent becomes a standard size
- recompute the normalized bounding box
- use an orthographic camera for every view
- set the orthographic scale from the normalized bounding box plus a margin
- when `--qa-root` is provided, write sidecar QA masks and
  `camera_framing_report.json` under `outputs/qa/framing/<sample-name>/`

After rendering, run the lightweight framing checker:

```bash
python scripts/check_render_framing.py --sample-dir data/hy3dpaint_train_examples/phase1e/B07H469871 --qa-dir outputs/qa/framing/B07H469871 --num-view 6 --border-margin-px 8 --out-json outputs/qa/framing/B07H469871/framing_report.json
```

The checker analyzes `render_tex/000.png` through `render_tex/005.png`. Mask-based
QA is preferred because RGB background thresholding can false fail when the whole
image is textured or colored. QA masks may have dark gray backgrounds, such as
luminance around 58, while visible object pixels are white. The default
`--mask-threshold` is 128 and remains configurable for unusual masks. QA masks
and QA reports are sidecar artifacts and must live under `outputs/qa/framing/...`,
not inside the Hunyuan training sample. They do not affect Hunyuan training. The
official sample directory should remain limited to `render_tex/` and
`render_cond/` plus the existing official metadata files. The checker fails only
when foreground clearly touches the image border within the configured margin. It
warns when the object is very small, very large, or noticeably off-center.
