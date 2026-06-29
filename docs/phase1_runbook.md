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
