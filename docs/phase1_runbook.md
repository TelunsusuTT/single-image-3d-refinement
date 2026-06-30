# Phase 1 Runbook

## Phase 1A Data Format Checks

Run these commands from the project root:

```bash
cd /vol/bitbucket/ct1022/hy3dpaint_finetune
source env/env.sh
conda activate "$ENV_NAME"
```

Validate the official overfit examples manifest:

```bash
python scripts/check_hy3dpaint_example.py --examples-json data/hy3dpaint_train_examples/official_overfit/examples.json --num-view 6 --strict
```

Write JSON and CSV summaries:

```bash
python scripts/summarize_train_examples.py --examples-json data/hy3dpaint_train_examples/official_overfit/examples.json --out-json outputs/boards/phase1_official_summary.json --out-csv outputs/boards/phase1_official_summary.csv --num-view 6
```

## Expected Success Signs

The checker should print one readable block per sample. A successful strict
check ends with:

```text
Checked N sample(s): N OK, 0 failed
```

The summarizer should print the paths it wrote:

```text
Wrote JSON summary: outputs/boards/phase1_official_summary.json
Wrote CSV summary: outputs/boards/phase1_official_summary.csv
```

The JSON output contains the source manifest path, sample count, per-sample
checks, per-sample map counts, and aggregate min/max counts. The CSV output has
one row per sample for quick spreadsheet inspection.

## Error Meanings

- `examples-json missing`: the manifest path is wrong or has not been created.
- `examples-json is empty`: the manifest exists but contains no sample paths.
- `sample directory missing`: a path listed in the manifest does not exist.
- `render_cond/ missing`: the sample is missing condition/reference images.
- `render_tex/ missing`: the sample is missing texture supervision images.
- `condition image count is 0`: `render_cond/` exists but contains no supported
  image files.
- `albedo image count ... < num-view`: too few albedo maps were found.
- `normal image count ... < num-view`: too few normal maps were found.
- `position image count ... < num-view`: too few position maps were found.
- `metallic-roughness image count ... < num-view`: strict mode found too few MR
  maps.
- `transforms.json missing from sample dir or render_tex/`: strict mode could
  not find camera/view metadata in either supported location.

These checks only inspect file packaging. They do not train a model or verify
image contents.

## Phase 1B Candidate Asset Planning

Phase 1B is only a planning step for selecting 1-3 texture-heavy assets. Do not
download large datasets, run Blender, run Hunyuan, or submit Slurm jobs yet.

Inspect the candidate CSV template:

```bash
head -n 5 data/candidates/asset_candidate_template.csv
```

Inspect the seeded placeholder candidates:

```bash
head -n 12 data/candidates/phase1b_candidates_seed.csv
```

Filter texture-heavy candidates that have not been rejected:

```bash
python scripts/filter_asset_candidates.py --input-csv data/candidates/phase1b_candidates_seed.csv --output-csv data/candidates/phase1b_candidates_filtered.csv
```

The filter keeps rows where `expected_texture_heavy` is `yes`, `true`, or `1`,
and where `status` is not `rejected`. It prints total rows, kept rows, rejected
rows, and kept-row counts by source and category.

The filtered CSV is only a planning artifact for Phase 1C. It is not a dataset
manifest and should not trigger downloads or conversion work by itself.

## Phase 1C One-Asset Acquisition

Phase 1C prepares one real asset first, preferably from ABO, then up to three
assets after the one-asset path is understood. Do not download full archives
such as `abo-3dmodels.tar`.

Create a selected-assets CSV from the template:

```bash
cp data/candidates/phase1c_selected_assets_template.csv data/candidates/phase1c_selected_assets.csv
```

Edit `data/candidates/phase1c_selected_assets.csv` and replace the TODO row with
one real selected asset. For ABO, fill `source`, `source_id`, `abo_path`, `name`,
`category`, `license`, `selection_reason`, `status`, and any useful notes.

Build a download manifest without downloading anything:

```bash
python scripts/make_abo_download_manifest.py --input-csv data/candidates/phase1c_selected_assets.csv --output-csv data/candidates/phase1c_download_manifest.csv --asset-root data/raw_assets/phase1c
```

Manually download exactly one selected GLB from the generated manifest:

```bash
aws s3 cp --no-sign-request s3://amazon-berkeley-objects/3dmodels/original/REPLACE_WITH_ABO_PATH data/raw_assets/phase1c/REPLACE_WITH_SOURCE_ID.glb
```

Do not run archive downloads, recursive syncs, or broad dataset fetches in Phase
1C.

After the one GLB is present locally, check only basic local file properties:

```bash
python scripts/check_local_glb_assets.py --manifest-csv data/candidates/phase1c_download_manifest.csv --min-size-mb 0.1 --max-size-mb 500
```

This check only verifies path existence, `.glb` or `.gltf` suffix, and file size
bounds. It does not import Blender, inspect UVs, inspect materials, convert data,
or train a model.

## Phase 1D Blender Asset Inspection

Phase 1D checks whether the selected GLB imports into Blender and has mesh, UV,
material, and texture signals. Do not run Hunyuan or training yet.

Find Blender:

```bash
which blender
```

If `blender` is not on `PATH`, check the local tools path:

```bash
ls /vol/bitbucket/ct1022/tools/bin/blender
```

Run the inspection script manually with Blender:

```bash
blender --background --python scripts/blender_inspect_glb.py -- --input-glb data/raw_assets/phase1c/B07H469871.glb --out-dir outputs/boards/phase1d_B07H469871
```

If using the local tools path:

```bash
/vol/bitbucket/ct1022/tools/bin/blender --background --python scripts/blender_inspect_glb.py -- --input-glb data/raw_assets/phase1c/B07H469871.glb --out-dir outputs/boards/phase1d_B07H469871
```

The Blender script writes:

```text
outputs/boards/phase1d_B07H469871/asset_inspection.json
outputs/boards/phase1d_B07H469871/asset_inspection_summary.md
```

Check the JSON report without Blender:

```bash
python scripts/check_asset_inspection_report.py --report-json outputs/boards/phase1d_B07H469871/asset_inspection.json
```

A pass means Blender imported the asset, found at least one mesh, found faces,
found at least one UV layer, and found at least one material. Missing texture
images are a warning, not an automatic failure.

## Phase 1E Render Training Example

Phase 1E renders the one selected GLB into a minimal Hunyuan3D-Paint-style
training example. This is still a structure/rendering smoke test. Do not run
Hunyuan training yet.

Run the Blender renderer manually:

```bash
blender --background --python scripts/blender_render_hy3dpaint_example.py -- --input-glb data/raw_assets/phase1c/B07H469871.glb --sample-name B07H469871 --out-root data/hy3dpaint_train_examples/phase1e --num-view 6 --resolution 512 --qa-root outputs/qa/framing
```

If using the local tools path:

```bash
/vol/bitbucket/ct1022/tools/bin/blender --background --python scripts/blender_render_hy3dpaint_example.py -- --input-glb data/raw_assets/phase1c/B07H469871.glb --sample-name B07H469871 --out-root data/hy3dpaint_train_examples/phase1e --num-view 6 --resolution 512 --qa-root outputs/qa/framing
```

Create an `examples.json` that points to the rendered sample:

```bash
python scripts/make_hy3dpaint_examples_json.py --sample-dir data/hy3dpaint_train_examples/phase1e/B07H469871 --out-json data/hy3dpaint_train_examples/phase1e/examples.json
```

Check the exact Phase 1E expected filenames:

```bash
python scripts/check_phase1e_outputs.py --sample-dir data/hy3dpaint_train_examples/phase1e/B07H469871 --num-view 6
```

Then run the existing Phase 1A checker in strict mode:

```bash
python scripts/check_hy3dpaint_example.py --examples-json data/hy3dpaint_train_examples/phase1e/examples.json --num-view 6 --strict
```

Phase 1E only confirms that the rendered sample has the expected directories,
filenames, and basic manifest shape. It does not prove training quality.

## Phase 1E.1 Automatic Render Framing QA

Phase 1E.1 improves the Blender renderer so assets are centered, normalized, and
framed automatically. Manual per-asset camera adjustment is not acceptable for
the later conversion pipeline.

The renderer computes a mesh bounding box, moves the asset center to the origin,
scales the largest extent to a standard size, and uses an orthographic camera
with margin. Each render writes:

```text
outputs/qa/framing/B07H469871/camera_framing_report.json
outputs/qa/framing/B07H469871/000_mask.png
...
outputs/qa/framing/B07H469871/005_mask.png
```

After rerendering with the Phase 1E command above, run the framing checker:

```bash
python scripts/check_render_framing.py --sample-dir data/hy3dpaint_train_examples/phase1e/B07H469871 --qa-dir outputs/qa/framing/B07H469871 --num-view 6 --border-margin-px 8 --out-json outputs/qa/framing/B07H469871/framing_report.json
```

The checker prefers `*_mask.png` files from the sidecar QA directory. RGB
background thresholding remains as a fallback for old outputs, but it can false
fail when the full render is colored or textured. QA masks may have dark gray
backgrounds, such as luminance around 58, while visible object pixels are white.
The default `--mask-threshold` is 128 and remains configurable for unusual masks.
The checker fails only for clear border-cropping. It warns if the object appears
too small, too large, or too far from image center.

Do not create `sample_dir/qa`, and do not place masks under `render_tex/` or
`render_cond/`. QA masks and reports belong under `outputs/qa/framing/...` and do
not affect Hunyuan training. Hunyuan training sample directories should remain
official-only: `render_tex/` and `render_cond/`. Do not run Hunyuan training
until the filename checks and mask-based framing QA both pass.

## Phase 1F Custom ABO One-Asset Training Smoke

Phase 1F checks whether the official Hunyuan3D-Paint `train.py` can read the
custom ABO one-asset training example and start the training loop. This is only
a dataloader/training-loop smoke test, not formal fine-tuning.

Create an absolute-path examples JSON so the official HYPAINT working directory
cannot change dataset path resolution:

```bash
python scripts/make_phase1f_train_json.py --sample-dir data/hy3dpaint_train_examples/abo_one_asset/B07H469871 --out-json data/hy3dpaint_train_examples/abo_one_asset/examples_train_abs.json
```

Run the local readiness check:

```bash
python scripts/check_phase1f_readiness.py --examples-json data/hy3dpaint_train_examples/abo_one_asset/examples_train_abs.json --config configs/ft_abo_one_asset_smoke.yaml
```

Run the official-style strict packaging check:

```bash
python scripts/check_hy3dpaint_example.py --examples-json data/hy3dpaint_train_examples/abo_one_asset/examples_train_abs.json --num-view 6 --strict
```

Submit the A100 smoke job manually:

```bash
sbatch env/run_abo_one_asset_smoke_a100.sbatch
```

Inspect Slurm logs:

```bash
ls -lt logs/slurm
tail -n 120 logs/slurm/hy3dpaint-abo-smoke-<jobid>.out
tail -n 120 logs/slurm/hy3dpaint-abo-smoke-<jobid>.err
```

Expected success signs are `CONFIG_TARGET_OK`, `DATASET_PREFLIGHT_OK`,
`CUBLAS_MATMUL_OK`, train startup, max steps reached, and `JOB END: SUCCESS`.
Failure usually means path ambiguity, `transforms.json` mismatch, image channel
mismatch, normal or position encoding mismatch, or an official dataloader
assumption that the custom sample does not satisfy yet. A successful Phase 1F
run does not imply useful checkpoint quality or visual quality.

## Phase 2A Metadata-Driven Asset Selection

Phase 2A builds an ABO candidate index and thumbnail gallery for manual review.
Do not download GLBs, full archives, or large datasets in this phase.

Download only small ABO metadata files. ABO listing metadata currently uses
`listings_0.json.gz` through `listings_9.json.gz` by default:

```bash
python scripts/download_abo_metadata.py --out-dir data/metadata/abo_phase2a --skip-existing
```

Build a scored candidate index:

```bash
python scripts/build_abo_candidate_index.py --metadata-dir data/metadata/abo_phase2a --out-csv data/candidates/phase2a_abo_candidates.csv --min-textures 3 --min-images 3 --min-resolution 2048 --max-faces 150000
```

Download only top-k catalog thumbnails for review. The candidate CSV should
contain URLs using `images/small/<image_path>`:

```bash
python scripts/download_abo_candidate_thumbnails.py --candidates-csv data/candidates/phase2a_abo_candidates.csv --thumb-dir data/candidates/phase2a_thumbnails --top-k 200 --skip-existing
```

Build a local gallery that can be opened directly in a browser:

```bash
python scripts/make_candidate_gallery.py --candidates-csv data/candidates/phase2a_abo_candidates.csv --thumb-dir data/candidates/phase2a_thumbnails --out-html outputs/boards/phase2a_abo_gallery.html --top-k 200
```

Mark a manually reviewed candidate:

```bash
python scripts/mark_phase2a_candidates.py --candidates-csv data/candidates/phase2a_abo_candidates.csv --selected-csv data/candidates/phase2a_selected_assets.csv --candidate-id REPLACE_WITH_ID --status selected --reason "texture-heavy product packaging"
```

The selected CSV is only a Phase 2B planning manifest. Metadata filtering is a
first pass; human gallery review remains required. If furniture-heavy results
dominate the top of the gallery, adjust semantic scoring or use
`--require-positive-keyword` rather than manually deleting metadata rows. Phase
2B will download selected GLBs, inspect them with Blender, render training
examples, and run the official Hunyuan-style checkers.
