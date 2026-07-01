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

## Phase 2B Selected Asset Download And Inspection

Phase 2B downloads only the Phase 2A selected ABO GLBs and checks whether they
are technically usable before rendering training examples.

Create a selected-asset download manifest:

```bash
python scripts/make_phase2b_download_manifest.py --selected-csv data/candidates/phase2a_abo_selected_assets.csv --out-csv data/candidates/phase2b_abo_download_manifest.csv --raw-dir data/raw_assets/phase2b_abo_selected
```

Download the selected GLBs manually:

```bash
python scripts/download_phase2b_assets.py --manifest-csv data/candidates/phase2b_abo_download_manifest.csv --skip-existing
```

Check that the expected local GLB files exist and are non-empty:

```bash
python scripts/check_phase2b_local_assets.py --manifest-csv data/candidates/phase2b_abo_download_manifest.csv
```

Run Blender inspection manually for each downloaded asset:

```bash
blender --background --python scripts/blender_inspect_glb.py -- --input-glb data/raw_assets/phase2b_abo_selected/REPLACE_WITH_SOURCE_ID.glb --out-dir outputs/boards/phase2b_inspections/REPLACE_WITH_SOURCE_ID
```

Summarize the inspection reports:

```bash
python scripts/summarize_phase2b_inspections.py --inspection-root outputs/boards/phase2b_inspections --manifest-csv data/candidates/phase2b_abo_download_manifest.csv --out-csv outputs/boards/phase2b_inspection_summary.csv --out-md outputs/boards/phase2b_inspection_summary.md
```

Assets should pass Blender import, mesh, material, texture image, UV, and
reasonable polygon-count checks before Phase 2C. Reject or hold weak assets for
review rather than rendering them into Hunyuan3D-Paint training examples.

## Phase 2C Batch Render Pilot Dataset

Phase 2C renders only Phase 2B-passed assets into the `pilot_v1`
Hunyuan3D-Paint-style training dataset. Do not run training in this phase.

Create the render manifest:

```bash
python scripts/make_phase2c_render_manifest.py --download-manifest data/candidates/phase2b_abo_download_manifest.csv --inspection-summary outputs/boards/phase2b_abo_inspection_summary.csv --out-csv data/candidates/phase2c_pilot_v1_render_manifest.csv --dataset-root data/hy3dpaint_train_examples/pilot_v1 --qa-root outputs/qa/framing/pilot_v1
```

Generate the Blender render commands:

```bash
python scripts/make_phase2c_render_commands.py --render-manifest data/candidates/phase2c_pilot_v1_render_manifest.csv --out-sh outputs/boards/run_phase2c_pilot_v1_renders.sh --blender-bin /vol/bitbucket/ct1022/tools/bin/blender --num-view 6 --resolution 512
```

Run the generated shell script manually when ready:

```bash
bash outputs/boards/run_phase2c_pilot_v1_renders.sh
```

Create dataset manifests:

```bash
python scripts/make_phase2c_examples_json.py --render-manifest data/candidates/phase2c_pilot_v1_render_manifest.csv --out-json data/hy3dpaint_train_examples/pilot_v1/examples.json
python scripts/make_phase2c_examples_json.py --render-manifest data/candidates/phase2c_pilot_v1_render_manifest.csv --out-json data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json --absolute
```

Run local rendered-dataset structure checks:

```bash
python scripts/check_phase2c_rendered_dataset.py --render-manifest data/candidates/phase2c_pilot_v1_render_manifest.csv --num-view 6 --framing-qa-root outputs/qa/framing/pilot_v1 --out-csv outputs/boards/phase2c_pilot_v1_render_summary.csv --out-md outputs/boards/phase2c_pilot_v1_render_summary.md
```

Run framing QA per asset after rendering:

```bash
python scripts/check_render_framing.py --sample-dir data/hy3dpaint_train_examples/pilot_v1/REPLACE_WITH_SOURCE_ID --qa-dir outputs/qa/framing/pilot_v1/REPLACE_WITH_SOURCE_ID --num-view 6 --border-margin-px 8 --out-json outputs/qa/framing/pilot_v1/REPLACE_WITH_SOURCE_ID/framing_report.json
```

Run the official-style strict checker on the pilot manifest:

```bash
python scripts/check_hy3dpaint_example.py --examples-json data/hy3dpaint_train_examples/pilot_v1/examples.json --num-view 6 --strict
```

A pass means the pilot dataset has the expected Hunyuan-style directories,
filenames, transforms, and sidecar framing QA. A failure means the asset should
be rerendered, reframed, or removed from the pilot manifest before Phase 2D.

## Phase 2E Multi-Asset Training Smoke

Phase 2E checks whether official Hunyuan3D-Paint `train.py` can read the
7-sample `pilot_v1` dataset and complete a 50-step A100 smoke run. This is not
formal fine-tuning.

Run the readiness check:

```bash
python scripts/check_phase2e_readiness.py --examples-json data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json --config configs/ft_pilot_v1_smoke.yaml --expected-count 7
```

Static-check the sbatch syntax:

```bash
bash -n env/run_pilot_v1_smoke_a100.sbatch
```

Submit the job manually when ready:

```bash
sbatch env/run_pilot_v1_smoke_a100.sbatch
```

Inspect logs:

```bash
ls -lt logs/slurm
tail -n 160 logs/slurm/hy3dpaint-pilot-v1-smoke-<jobid>.out
tail -n 160 logs/slurm/hy3dpaint-pilot-v1-smoke-<jobid>.err
```

Expected success signs are `CONFIG_TARGET_OK`, `DATASET_PREFLIGHT_OK`, the
official strict checker reporting `Checked 7 sample(s): 7 OK, 0 failed`,
`CUBLAS_MATMUL_OK`, `dataset length = 7`, `max_steps reached`, and
`JOB END: SUCCESS`. Failure means the pilot dataset, config path wiring, CUDA
runtime, or official dataloader assumptions need debugging before any real
fine-tuning claims.

## Phase 2F Tiny Pilot Overfit

Phase 2F runs the first real 500-step tiny overfit on the 7-sample `pilot_v1`
dataset. It tests longer training and one-checkpoint saving, but it is not a
final quality or generalization experiment.

Run the readiness check:

```bash
python scripts/check_phase2f_readiness.py --examples-json data/hy3dpaint_train_examples/pilot_v1/examples_train_abs.json --config configs/ft_pilot_v1_overfit_500.yaml --expected-count 7 --checkpoint-root logs/train
```

Static-check the sbatch syntax and inspect the checkpoint settings:

```bash
bash -n env/run_pilot_v1_overfit_500_a100.sbatch
grep -n "max_steps\|save_top_k\|save_last\|save_weights_only" configs/ft_pilot_v1_overfit_500.yaml
```

Submit the job manually when ready:

```bash
sbatch env/run_pilot_v1_overfit_500_a100.sbatch
```

Inspect logs:

```bash
ls -lt logs/slurm
tail -n 200 logs/slurm/hy3dpaint-pilot-v1-overfit500-<jobid>.out
tail -n 200 logs/slurm/hy3dpaint-pilot-v1-overfit500-<jobid>.err
```

Check checkpoint size after the job:

```bash
find logs/train checkpoints -type f -name "*.ckpt" -printf "%p %s bytes\n" 2>/dev/null
du -sh logs/train checkpoints 2>/dev/null
```

Expected success signs are `PHASE2F_PREFLIGHT_OK`, the official strict checker
passing on all 7 samples, `CUBLAS_MATMUL_OK`, `max_steps=500 reached`,
one printed checkpoint path and size, and `JOB END: SUCCESS`. Failure usually
means config path drift, checkpoint callback behavior, CUDA memory pressure, or
a training stability issue such as NaN.

## Phase 2G Checkpoint Inference Sanity

Phase 2G prepares a base-vs-fine-tuned inference sanity check for the 500-step
`pilot_v1` checkpoint. Do not run Hunyuan inference directly yet.

Inspect the official Hunyuan3D-Paint inference/checkpoint-loading interface
read-only:

```bash
python scripts/inspect_phase2g_hy3dpaint_interfaces.py --hypaint /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint --out-json outputs/boards/phase2g_hy3dpaint_interface.json --out-md outputs/boards/phase2g_hy3dpaint_interface.md
```

Prepare the first local inference case:

```bash
python scripts/prepare_phase2g_infer_case.py --asset-id B075YLTF7Q --mesh data/raw_assets/phase2b_abo_selected/B075YLTF7Q.glb --reference-image data/hy3dpaint_train_examples/pilot_v1/B075YLTF7Q/render_cond/001_light_AL.png --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --out-dir outputs/phase2g/cases/B075YLTF7Q
```

Run readiness without loading the checkpoint:

```bash
python scripts/check_phase2g_readiness.py --case-dir outputs/phase2g/cases/B075YLTF7Q --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --hypaint /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint
```

Stop after readiness and review the interface report before creating an A100
inference sbatch. Phase 2G should compare base and fine-tuned outputs with the
same mesh, same reference image, same settings, and the same later board layout.

## Phase 2G.1 Project-Local Inference Wrapper

Phase 2G.1 prepares a project-local wrapper for later A100 inference. Do not run
Hunyuan, load the checkpoint, submit Slurm, or modify the official Hunyuan repo.

Inspect checkpoint filesystem metadata without opening the checkpoint:

```bash
python scripts/inspect_phase2g1_checkpoint_metadata.py --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --out-json outputs/boards/phase2g1_checkpoint_metadata.json --out-md outputs/boards/phase2g1_checkpoint_metadata.md
```

Dry-run the wrapper in base mode:

```bash
python scripts/run_phase2g_paint_infer.py --case-dir outputs/phase2g/infer_cases/B075YLTF7Q --output-dir outputs/phase2g/base/B075YLTF7Q --mode base --max-num-view 6 --resolution 512 --dry-run
```

Dry-run the wrapper in fine-tuned mode:

```bash
python scripts/run_phase2g_paint_infer.py --case-dir outputs/phase2g/infer_cases/B075YLTF7Q --output-dir outputs/phase2g/finetuned/B075YLTF7Q --mode finetuned --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --max-num-view 6 --resolution 512 --dry-run
```

Run wrapper readiness:

```bash
python scripts/check_phase2g1_wrapper_readiness.py --case-dir outputs/phase2g/infer_cases/B075YLTF7Q --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --wrapper scripts/run_phase2g_paint_infer.py --hypaint /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint
```

Stop after these checks and review the outputs before creating any A100
inference sbatch. Fine-tuned non-dry-run mode must fail loudly until checkpoint
key mapping is confirmed.

## Phase 2G.2 Base Inference A100 Smoke

Phase 2G.2 runs one real base Hunyuan3D-Paint inference through the
project-local wrapper. It does not load the Phase 2F checkpoint and does not run
fine-tuned inference.

Run readiness first:

```bash
python scripts/check_phase2g2_base_infer_readiness.py --case-dir outputs/phase2g/infer_cases/B075YLTF7Q --wrapper scripts/run_phase2g_paint_infer.py --hypaint /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint --output-dir outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke
```

Static-check the sbatch syntax:

```bash
bash -n env/run_phase2g_base_infer_a100.sbatch
```

Submit manually only after assistant review:

```bash
sbatch env/run_phase2g_base_infer_a100.sbatch
```

Inspect logs and outputs:

```bash
ls -lt logs/slurm
tail -n 200 logs/slurm/hy3dpaint-phase2g-base-infer-<jobid>.out
tail -n 200 logs/slurm/hy3dpaint-phase2g-base-infer-<jobid>.err
find outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke -type f \( -name "*.obj" -o -name "*.glb" -o -name "run_plan.json" \) -printf "%p %s bytes\n"
```

The sbatch uses `--no-remesh` to bypass the `pymeshlab` remesh path that failed
in job `255495`.

Success means the base project-local wrapper can execute official inference and
produce at least one mesh output. Failure means pathing, cache/model
availability, CUDA runtime, or official inference assumptions need debugging
before any fine-tuned checkpoint loading is attempted.

## Phase 2G.3 Checkpoint Key Mapping

Phase 2G.3 inspects the Phase 2F checkpoint key structure and compares it with
the base inference pipeline UNet keys. It does not run fine-tuned inference and
does not load checkpoint weights into the inference model.

Run readiness first:

```bash
python scripts/check_phase2g3_key_inspect_readiness.py --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --hypaint /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint --output-dir outputs/phase2g/key_inspection/pilot_v1_overfit_500
```

Static-check the sbatch syntax:

```bash
bash -n env/run_phase2g3_key_inspect_a100.sbatch
```

Submit manually only after assistant review:

```bash
sbatch env/run_phase2g3_key_inspect_a100.sbatch
```

Inspect the generated reports:

```bash
ls -lh outputs/phase2g/key_inspection/pilot_v1_overfit_500
sed -n '1,160p' outputs/phase2g/key_inspection/pilot_v1_overfit_500/keyspace_compare.md
```

Expected success signs are `PHASE2G3_KEY_INSPECT_PREFLIGHT_OK`,
`CUBLAS_MATMUL_OK`, `PHASE2G3_CHECKPOINT_KEYS_OK`,
`PHASE2G3_INFER_UNET_KEYS_OK`, `PHASE2G3_KEYSPACE_COMPARE_OK`, and
`JOB END: SUCCESS`. The compare report should either recommend a prefix mapping
such as stripping `unet.unet.` or explicitly mark the load strategy as
`UNKNOWN`.

Do not run fine-tuned inference yet. Phase 2G.3 only decides whether the
checkpoint key mapping is understood well enough to design the next wrapper
patch.

## Phase 2G.4 Checkpoint Load-Only Smoke

Phase 2G.4 builds the official base inference pipeline, loads the Phase 2F
checkpoint into `paint_pipeline.models["multiview_model"].pipeline.unet`, and
exits before inference. It uses the Phase 2G.3b mapping recommendation:
strip checkpoint prefix `unet.` and load with `strict=True`.

Run readiness first:

```bash
python scripts/check_phase2g4_load_readiness.py --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --hypaint /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint --output-dir outputs/phase2g/load_only/pilot_v1_overfit_500
```

Static-check the sbatch syntax:

```bash
bash -n env/run_phase2g4_load_only_a100.sbatch
```

Submit manually only after assistant review:

```bash
sbatch env/run_phase2g4_load_only_a100.sbatch
```

Inspect the load-only report:

```bash
ls -lh outputs/phase2g/load_only/pilot_v1_overfit_500
sed -n '1,180p' outputs/phase2g/load_only/pilot_v1_overfit_500/load_only_report.md
```

Expected success signs are `PHASE2G4_LOAD_PREFLIGHT_OK`,
`CUBLAS_MATMUL_OK`, exact key and shape compatibility,
`PHASE2G4_CHECKPOINT_LOAD_ONLY_OK`, and `JOB END: SUCCESS`.

Do not run fine-tuned inference until the load-only smoke succeeds.

## Phase 2G.5 Fine-Tuned No-Remesh Inference Smoke

Phase 2G.5 is the first real fine-tuned inference smoke. It loads the Phase 2F
checkpoint into `paint_pipeline.models["multiview_model"].pipeline.unet` using
the Phase 2G.3b `strip:unet.` mapping, then runs one no-remesh inference on the
prepared `B075YLTF7Q` case.

Run readiness first:

```bash
python scripts/check_phase2g5_finetuned_infer_readiness.py --case-dir outputs/phase2g/infer_cases/B075YLTF7Q --checkpoint checkpoints/pilot_v1_overfit_500/pilot_v1_overfit_500-stepstep=500.ckpt --wrapper scripts/run_phase2g_paint_infer.py --hypaint /vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/hy3dpaint --output-dir outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke
```

Static-check the sbatch syntax:

```bash
bash -n env/run_phase2g5_finetuned_infer_a100.sbatch
```

Submit manually only after assistant review:

```bash
sbatch env/run_phase2g5_finetuned_infer_a100.sbatch
```

Inspect outputs:

```bash
ls -lh outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke
find outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke -type f \( -name "*.obj" -o -name "*.glb" -o -name "run_plan.json" \) -printf "%p %s bytes\n"
```

Expected success signs are `PHASE2G5_FINETUNED_INFER_PREFLIGHT_OK`,
`CUBLAS_MATMUL_OK`, `PHASE2G5_FINETUNED_CHECKPOINT_LOADED_OK`, at least one
mesh output, and `JOB END: SUCCESS`.

Do not make any quality claim until Phase 2G.6 compares base and fine-tuned
outputs side by side.

## Phase 2G.6 Failure Diagnostic Comparison

Phase 2G.6 compares the successful base no-remesh inference against the
fine-tuned no-remesh inference. This is diagnostic only: it should identify
whether the visible degradation is concentrated in albedo, metallic, roughness,
or mesh/material binding signals.

Run readiness first:

```bash
python scripts/check_phase2g6_compare_readiness.py --case-dir outputs/phase2g/infer_cases/B075YLTF7Q --base-dir outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke --finetuned-dir outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke --output-dir outputs/phase2g/compare/B075YLTF7Q
```

Run the texture comparison:

```bash
python scripts/make_phase2g6_texture_comparison.py --case-dir outputs/phase2g/infer_cases/B075YLTF7Q --base-dir outputs/phase2g/infer_runs/B075YLTF7Q/base_a100_noremesh_smoke --finetuned-dir outputs/phase2g/infer_runs/B075YLTF7Q/finetuned_a100_noremesh_smoke --output-dir outputs/phase2g/compare/B075YLTF7Q
```

Inspect the board and report:

```bash
ls -lh outputs/phase2g/compare/B075YLTF7Q
sed -n '1,220p' outputs/phase2g/compare/B075YLTF7Q/base_vs_finetuned_report.md
```

Use the metrics to decide the next step: if albedo differs strongly, inspect
color/texture learning; if metallic or roughness shifts strongly, inspect PBR
channel prediction; if maps look sane but GLB appearance is bad, inspect
material binding and exported mesh references. Do not make a final quality claim
or retraining decision from this diagnostic alone.
