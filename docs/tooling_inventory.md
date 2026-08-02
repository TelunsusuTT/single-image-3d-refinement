# Tooling Inventory

This inventory is the first place to check before adding a new script. Prefer
reusing or extending existing tools over creating duplicate phase-specific
versions.

Status labels:

- `reuse`: preferred stable building block
- `runtime`: useful but may run Hunyuan, Blender, torch, network, or large jobs
- `diagnostic`: phase-specific investigation tool
- `legacy`: keep for history; avoid extending unless needed

## Data Format and Local QA

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|---|
| `check_hy3dpaint_example.py` | Validate Hunyuan3D-Paint sample directories | examples JSON, num views | readable pass/fail | No | No | reuse | Use for any train examples JSON before training. |
| `summarize_train_examples.py` | Summarize train example counts | examples JSON | JSON and CSV summaries | No | No | reuse | Use after dataset conversion or pilot rendering. |
| `check_phase1e_outputs.py` | Check one rendered training example structure | sample dir | readable structure report | No | No | reuse | Use for Phase 1E/2C-style rendered examples. |
| `check_render_framing.py` | QA rendered object framing using masks/RGB fallback | sample dir, optional QA dir | stdout and optional JSON | No | No | reuse | Prefer mask QA sidecars under `outputs/qa/framing`. |
| `make_hy3dpaint_examples_json.py` | Create examples JSON from sample dirs | sample root | examples JSON | No | No | reuse | Use when building local train-example manifests. |

## Asset Candidate Selection and Metadata

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `filter_asset_candidates.py` | Filter planning CSV for texture-heavy unrejected assets | candidate CSV | filtered CSV | No | No | legacy | Simple Phase 1B utility; prefer newer ABO candidate tools for ABO. |
| `download_abo_metadata.py` | Download small ABO metadata files | metadata output dir | `.csv.gz`/`.json.gz` metadata | No | No | runtime | Network script; do not run unless explicitly requested. |
| `build_abo_candidate_index.py` | Score ABO assets from metadata | ABO metadata | candidate CSV | No | No | reuse | Prefer before manual gallery review. |
| `download_abo_candidate_thumbnails.py` | Download selected thumbnails | candidate CSV | local thumbnails, failures CSV | No | No | runtime | Network script; thumbnail URL should use `images/small`. |
| `make_candidate_gallery.py` | Build HTML gallery for review | candidate CSV, thumbnails | gallery HTML | No | No | reuse | Human review entry point. |
| `mark_phase2a_candidates.py` | Mark selected/rejected candidates | candidate CSV | updated selection CSV | No | No | legacy | Reuse if its CSV schema matches; otherwise prefer a new CSV transform. |
| `datav2_inventory_metadata_sources.py` | Inventory local ABO/Objaverse metadata files | Data v2 mining config | metadata report JSON/MD | No | No | reuse | Phase 2L.1 metadata-first entry point; run before mining candidates. |
| `datav2_mine_flat_panel_candidates.py` | Normalize and rank flat-panel metadata candidates | Data v2 mining config, local metadata | ranked candidate CSV/MD and summary JSON | No | No | reuse | Preferred Data v2A candidate miner; use `--dry-run` before writing outputs. |
| `datav2_make_human_review_template.py` | Create curation template from ranked candidates | ranked candidate CSV | human-review CSV | No | No | reuse | Use after mining; does not auto-accept candidates. |
| `datav2_mine_abo_geometry_candidates.py` | Rank ABO flat-panel candidates from asset geometry | Data v2 mining config, ABO `3dmodels.csv.gz` | geometry candidate CSV/MD and summary JSON | No | No | reuse | Preferred when ABO semantic title/category fields are unavailable; does not score `images.csv.gz` as candidates. |
| `datav2_make_abo_geometry_review_template.py` | Create curation template from ABO geometry candidates | ABO geometry candidate CSV | human-review CSV | No | No | reuse | Use after geometry mining; does not auto-accept candidates. |
| `datav2_resolve_manual_abo_item_ids.py` | Resolve manually selected ABO item IDs against local 3D metadata | item-id text file, ABO `3dmodels.csv.gz` | download manifest CSV, JSON/MD resolution summary | No | No | reuse | Use when web-page semantic review identifies promising ABO IDs; missing IDs do not get URLs. |
| `datav2_download_manual_abo_glbs.py` | Optionally download manually resolved ABO GLBs | manual ABO download manifest | local GLBs, JSON/MD download summary | No | No | runtime | Network script; always dry-run first and do not run in Codex unless explicitly requested. |
| `datav2_make_manual_abo_review_template.py` | Create curation template from manual ABO manifest | manual ABO download manifest | human-review CSV | No | No | reuse | Use after resolving manual IDs; does not auto-accept candidates. |
| `datav2_make_manual_abo_contact_sheets.py` | Create contact sheets from manual ABO Blender preview renders | manual ABO visual config, inspection CSV, six-view PNGs | contact sheet JPG pages and index MD | No | No | reuse | Use after manual ABO Blender renders exist; imports Pillow only at runtime. |
| `datav2_update_manual_abo_review_with_inspection.py` | Merge manual ABO manifest/review with inspection results | manual ABO visual config, manifest, review CSV, inspection CSV | inspection-enriched review CSV | No | No | reuse | Preferred before human curation of manual ABO set. |
| `datav2_build_frame_panel_curated_manifest.py` | Build curated Data v2 framed-panel manifest | frame-panel split config, manual review-with-inspection CSV, optional reject IDs | curated manifest CSV/JSON and summary JSON/MD | No | No | reuse | Use after manual ABO visual QA before any training-example rendering. |
| `datav2_make_frame_panel_splits.py` | Create fixed-seed group-aware mini40/full101 splits | frame-panel split config, curated manifest | split JSONs, membership CSV, summary JSON/MD | No | No | reuse | Avoids original manual list order and reduces near-duplicate leakage. |
| `datav2_export_frame_panel_training_plan.py` | Export Data v2 frame-panel training plan | frame-panel split config and split files | training plan Markdown | No | No | reuse | Planning only; do not submit training from this script. |
| `datav2_build_frame_panel_render_plan.py` | Build Hunyuan-example render plan for frame panels | frame-panel render config, curated manifest, split file | render plan CSV and summary JSON/MD | No | No | reuse | Use for mini40 or full101 before Blender rendering; validates local GLB paths. |
| `datav2_build_frame_panel_examples_json.py` | Build train/val/test/all examples JSON files | frame-panel render config, render results CSV | absolute examples JSON files | No | No | reuse | Use after successful mini40 or full101 rendering. |
| `check_datav2_frame_panel_examples.py` | Check rendered frame-panel Hunyuan examples | frame-panel render config, examples JSONs, sample dirs | check summary JSON/MD | No | No | reuse | Use before strict checker or A100 training prep for mini40 or full101. |
| `datav2_prepare_abo_probe_manifest.py` | Prepare top-k ABO probe availability manifest | ABO geometry candidate CSV, probe config | probe manifest CSV, availability JSON/MD | No | No | reuse | Use before any Phase 2L.2A download or visual inspection. |
| `datav2_make_abo_download_plan.py` | Create non-executing download plan for missing ABO probe assets | probe manifest CSV | safe shell plan | No | No | reuse | Emits commented download commands only; does not download. |
| `datav2_make_abo_probe_human_review_template.py` | Create curation template from probe manifest and optional inspection CSV | probe manifest, optional inspection CSV | human-review CSV | No | No | reuse | Works before Blender inspection; human review remains required. |
| `datav2_dedupe_abo_probe_candidates.py` | Deduplicate ABO geometry candidates for acquisition | dedup config, geometry candidate CSV | dedup candidate CSV, JSON/MD summary | No | No | reuse | Use before downloading top ABO probe assets to avoid near-duplicates. |
| `datav2_make_abo_dedup_download_manifest.py` | Convert deduped ABO candidates into download-ready manifest | dedup config, dedup candidate CSV | download manifest CSV | No | No | reuse | Download-ready only; does not download. |
| `datav2_make_abo_visual_contact_sheets.py` | Build paginated contact sheets from rendered ABO visual thumbnails | visual inspection config, inspection CSV, rendered PNGs | contact sheet JPG pages and index MD | No | No | reuse | Imports Pillow only after validating expected renders exist. |
| `datav2_make_abo_visual_human_review.py` | Create Data v2 visual curation CSV for ABO candidates | visual inspection config, manifest, optional inspection CSV | human-review CSV | No | No | reuse | Works before Blender inspection; does not auto-accept candidates. |

## Asset Download and Inspection

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `make_abo_download_manifest.py` | Build one-asset ABO S3 manifest | selected CSV | manifest CSV | No | No | legacy | Phase 1C one-asset helper. |
| `check_local_glb_assets.py` | Check local GLB/GLTF presence and size | manifest CSV | stdout pass/fail | No | No | reuse | Use for any local asset presence check. |
| `make_phase2b_download_manifest.py` | Build selected ABO GLB download manifest | selected CSV | manifest CSV | No | No | reuse | Preferred for selected ABO GLBs. |
| `download_phase2b_assets.py` | Download selected GLBs from manifest | manifest CSV | local GLBs | No | No | runtime | Network script; do not run by default. |
| `check_phase2b_local_assets.py` | Check selected Phase 2B GLBs exist | manifest CSV | stdout pass/fail | No | No | reuse | Use before Blender inspection. |
| `blender_inspect_glb.py` | Blender import/UV/material/texture inspection | GLB | inspection JSON/MD | No | Yes | runtime | Run manually with Blender only when requested. |
| `datav2_inspect_abo_probe_blender.py` | Blender inspect/render contact sheets for local ABO probe GLBs | probe manifest | inspection CSV/JSON/MD and contact sheets | No | Yes | runtime | Manual Blender-only Phase 2L.2A visual probe; not run in Codex. |
| `datav2_inspect_abo_dedup_blender.py` | Blender inspect/render six fixed views for downloaded ABO dedup GLBs | visual inspection config | inspection CSV/JSON/MD and rendered PNGs | No | Yes | runtime | Manual Blender-only Phase 2L.2C visual inspection; not run in Codex. |
| `datav2_inspect_manual_abo_blender.py` | Blender inspect/render six fixed views for manually selected ABO GLBs | manual ABO visual config | inspection CSV/JSON/MD and rendered PNGs | No | Yes | runtime | Manual Blender-only Phase 2L.2B visual inspection; supports limit/start/only-missing. |
| `check_asset_inspection_report.py` | Validate inspection JSON without Blender | inspection JSON | pass/fail | No | No | reuse | Use after Blender inspection. |
| `summarize_phase2b_inspections.py` | Join inspection reports with manifest | inspection root, manifest | CSV/MD summary | No | No | reuse | Preferred gate before rendering. |

## Rendering Training Examples

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `blender_render_hy3dpaint_example.py` | Render GLB into Hunyuan-style train sample | GLB, sample name, output root | `render_tex`, `render_cond`, transforms, QA sidecars | No | Yes | reuse | Preferred training-example renderer; reuses normalized orthographic framing. |
| `datav2_render_frame_panel_examples_blender.py` | Batch-render mini40 frame-panel GLBs into Hunyuan examples | frame-panel render config, render plan CSV | mini40 train examples, render results JSON/MD/CSV | No | Yes | runtime | Blender-only wrapper around `blender_render_hy3dpaint_example.py`; not run in Codex. |
| `make_phase2c_render_manifest.py` | Build batch-render manifest from passed inspections | download manifest, inspection summary | render manifest CSV | No | No | reuse | Use for pilot-style render batches. |
| `make_phase2c_render_commands.py` | Generate Blender commands for batch rendering | render manifest | shell script | No | No | reuse | Generates but does not run Blender. |
| `make_phase2c_examples_json.py` | Create examples JSON from render manifest | render manifest | JSON list | No | No | reuse | Use for relative or absolute examples JSON. |
| `check_phase2c_rendered_dataset.py` | Check rendered dataset file inventory | render manifest | CSV/MD dataset check | No | No | reuse | Use before official strict checker. |

## Training Readiness and Smoke Jobs

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `make_phase1f_train_json.py` | Create absolute one-sample train JSON | sample dir | examples JSON | No | No | reuse | Use to avoid relative-path ambiguity. |
| `check_phase1f_readiness.py` | Check one-asset smoke readiness | examples JSON, config | stdout preflight | No | No | reuse | One-asset official training smoke gate. |
| `check_phase2e_readiness.py` | Check pilot_v1 smoke readiness | examples JSON, config | stdout preflight | No | No | reuse | Multi-asset smoke gate. |
| `check_phase2f_readiness.py` | Check 500-step overfit readiness | examples JSON, config, checkpoint root | stdout preflight | No | No | legacy | Superseded by true-PBR initialization work. |
| `check_phase2h1_readiness.py` | Check conservative wrong-init recovery readiness | examples/config/checkpoint paths | stdout preflight | No | No | legacy | Keep for history; do not base new runs on wrong-init configs. |
| `check_phase2j3_truepbr50_readiness.py` | Check true-PBR 50-step train readiness | examples/config/HYPAINT/checkpoint root | stdout preflight | No | No | reuse | Template for true-PBR training preflights. |
| `check_phase2k1_truepbr200_readiness.py` | Check true-PBR 200-step train-eval readiness | examples/config/case/base/eval paths | stdout preflight | No | No | reuse | Preferred pattern for combined train-eval gates. |
| `check_datav2_frame_mini40_training_readiness.py` | Check Data v2 mini40 true-PBR training readiness | mini40 training JSON config | stdout, readiness JSON/MD | No | No | reuse | Preferred static gate before the Phase 2L.4A mini40 A100 sbatch. |
| `inspect_datav2_frame_mini40_checkpoint.py` | Stat-only mini40 checkpoint inspection | mini40 training JSON config | checkpoint inspection JSON/MD | No | No | reuse | Use after Phase 2L.4A training to verify a step-500 checkpoint exists without loading it. |
| `check_datav2_frame_full80_training_readiness.py` | Check full80 true-PBR 500-step training readiness | full80 training JSON config | stdout, readiness JSON/MD | No | No | reuse | Preferred static gate before the Phase 2L.6B full80 A100 sbatch; validates split counts 80/10/11 and true-PBR YAML text. |
| `inspect_datav2_frame_full80_checkpoint.py` | Stat-only full80 checkpoint inspection | full80 training JSON config | checkpoint inspection JSON/MD | No | No | reuse | Use after Phase 2L.6B training to verify exactly one step-500 checkpoint exists without loading it. |

## Hunyuan Inference and Checkpoint Diagnostics

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `run_phase2g_paint_infer.py` | Project-local wrapper around official paint inference | case dir, output dir, mode, optional checkpoint | textured OBJ/GLB/maps, run plan | Yes | No | reuse | Preferred inference wrapper; supports `--no-remesh`; do not run in Codex. |
| `prepare_phase2g_infer_case.py` | Prepare one inference case dir | mesh/image paths | case dir | No | No | reuse | Use case layout `input/mesh.glb`, `input/image.png`. |
| `check_phase2g_readiness.py` | Check Phase 2G broad readiness | paths | stdout preflight | No | No | legacy | Prefer newer specific readiness checks. |
| `check_phase2g1_wrapper_readiness.py` | Check wrapper dry-run readiness | case/wrapper/checkpoint paths | stdout preflight | No | No | reuse | Use before wrapper-only dry runs. |
| `check_phase2g2_base_infer_readiness.py` | Check base inference readiness | case, wrapper, HYPAINT, output | stdout preflight | No | No | reuse | Use before base A100 inference. |
| `check_phase2g4_load_readiness.py` | Check checkpoint load-only readiness | checkpoint/HYPAINT/output | stdout preflight | No | No | reuse | Use before load-only A100 jobs. |
| `check_phase2g5_finetuned_infer_readiness.py` | Check fine-tuned inference readiness | case, checkpoint, HYPAINT, output | stdout preflight | No | No | reuse | Use before fine-tuned A100 inference. |
| `load_phase2g4_finetuned_checkpoint_only.py` | Load checkpoint into inference UNet without inference | checkpoint, HYPAINT | JSON/MD load report | Yes | No | runtime | Requires checkpoint/model load; A100/runtime only. |
| `make_phase2g6_texture_comparison.py` | Compare base/fine texture maps | case, base dir, fine dir | metrics JSON, report MD, board JPG | No | No | reuse | Preferred UV texture-map diagnostic; imports PIL at runtime. |
| `check_phase2g6_compare_readiness.py` | Check texture comparison readiness | case/base/fine/output | stdout preflight | No | No | reuse | Use before UV map comparison. |

## Keyspace, Initialization, and Audit Diagnostics

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `inspect_phase2g1_checkpoint_metadata.py` | Inspect checkpoint metadata | checkpoint | JSON/MD | Maybe | No | diagnostic | Avoid unless checkpoint inspection is explicitly requested. |
| `inspect_phase2g3_checkpoint_keys.py` | Inspect checkpoint keyspace | checkpoint | key JSON | Maybe | No | diagnostic | Checkpoint loading; runtime only when requested. |
| `inspect_phase2g3_infer_unet_keys.py` | Inspect inference UNet key candidates | HYPAINT | key JSON | Yes | No | diagnostic | Imports official model; A100/runtime only. |
| `compare_phase2g3_keyspaces.py` | Compare checkpoint keys to inference candidates | key JSON files | JSON/MD recommendation | No | No | reuse | Safe stdlib comparison; reuse for key mapping. |
| `inspect_phase2g_hy3dpaint_interfaces.py` | Inspect official inference interfaces | HYPAINT text files | JSON/MD | No | No | diagnostic | Read-only official text inspection only. |
| `check_phase2g3_key_inspect_readiness.py` | Check key-inspection readiness | checkpoint/HYPAINT/output | stdout preflight | No | No | diagnostic | Use before A100 key inspection. |
| `check_phase2g7_target_diagnostic_readiness.py` | Check target diagnostic readiness | checkpoint/HYPAINT/output | stdout preflight | No | No | diagnostic | Phase 2G.7 only. |
| `analyze_phase2g7_training_targets.py` | Analyze training target diagnostics | JSON/paths | report | No | No | diagnostic | Phase 2G.7 only. |
| `inspect_phase2i_training_initialization.py` | Inspect training configs/init text | configs/HYPAINT | JSON/MD audit | No | No | reuse | Use before new training config families. |
| `compare_phase2i_base_unet_to_checkpoints.py` | Numeric base-vs-checkpoint audit | HYPAINT, checkpoints | JSON/MD delta report | Yes | No | runtime | Loads official model/checkpoints; A100/runtime only. |
| `check_phase2i_audit_readiness.py` | Check Phase 2I audit readiness | HYPAINT/checkpoints/output | stdout preflight | No | No | reuse | Use before initialization audits. |
| `locate_phase2j_official_pbr_weights.py` | Locate local official PBR pipeline | search roots | JSON/MD candidates | No | No | reuse | Use before changing PBR weight paths. |
| `inspect_phase2j_training_init_interfaces.py` | Inspect official train init interfaces | HYPAINT text files | JSON/MD | No | No | reuse | Read-only text inspection. |
| `check_phase2j0_readiness.py` | Check Phase 2J.0 planning readiness | HYPAINT/search roots/output | stdout preflight | No | No | reuse | Use before PBR init planning. |
| `compare_phase2j1_training_init_to_infer_base.py` | Compare instantiated training init to inference base | HYPAINT/config/output | JSON/MD | Yes | No | runtime | Imports official model; A100/runtime only. |
| `check_phase2j1_truepbr_init_readiness.py` | Check true-PBR init probe readiness | HYPAINT/config/PBR dir/output | stdout preflight | No | No | reuse | Gate for PBR init probes. |
| `check_phase2j2_save_smoke_readiness.py` | Check 1-step save smoke readiness | examples/config/HYPAINT/checkpoint/output | stdout preflight | No | No | reuse | Template for checkpoint-save smoke gates. |
| `check_phase2j2_saved_delta.py` | Verify saved delta is tiny | delta JSON | stdout pass/fail | No | No | reuse | Use after base-equivalence save smoke. |

## True-PBR Evaluation and Input-View Workflows

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `check_phase2j4_truepbr50_eval_readiness.py` | Check true-PBR 50 eval readiness | case/base/checkpoint/HYPAINT/output dirs | stdout preflight | No | No | reuse | Template for evaluation preflights. |
| `check_phase2k2_truepbr200_multicase_readiness.py` | Check true-PBR 200 multicase inference readiness | cases config, HYPAINT | stdout preflight | No | No | reuse | Preferred multicase inference preflight. |
| `prepare_phase2k2_multicase_cases.py` | Prepare multicase inference case dirs | cases config | `cases/<asset>/input` dirs | No | No | reuse | Reuse case-dir creation pattern. |
| `aggregate_phase2k2_multicase_metrics.py` | Aggregate UV texture metrics across cases | cases config, output root | JSON/MD summary | No | No | reuse | For texture-map comparison summaries. |
| `check_phase2k3_render_eval_readiness.py` | Check rendered-view eval readiness | render eval config, output root | stdout preflight | No | No | reuse | Preferred before Blender rendered-view eval. |
| `render_phase2k3_glb_views_blender.py` | Render GLBs from fixed Phase 1E-like views | render eval config, output root | per-view PNGs, render config JSON | No | Yes | runtime | Blender runtime only; reuse for rendered-view evaluation. |
| `compare_phase2k3_rendered_views.py` | Compare rendered views against references | render eval config, output root | per-case metrics/report/board | No | No | reuse | Imports PIL at runtime; preferred rendered-view comparison. |
| `aggregate_phase2k3_rendered_metrics.py` | Aggregate rendered-view metrics | render eval config, output root | JSON/MD summary | No | No | reuse | Preferred rendered-view aggregate. |
| `check_phase2k4_reference_view_ablation_readiness.py` | Check input-view ablation readiness | ablation config, HYPAINT | stdout preflight | No | No | reuse | Use before Phase 2K.4-style ablations. |
| `prepare_phase2k4_reference_view_cases.py` | Prepare input-view case dirs | ablation config | case dirs for each asset/view | No | No | reuse | Reuse for input-view ablation case prep. |
| `make_phase2k4_render_eval_configs.py` | Generate rendered-eval configs for ablations | ablation config | config JSON per input view | No | No | reuse | Bridges inference outputs to Phase 2K.3 renderer. |
| `aggregate_phase2k4_reference_view_ablation.py` | Compare baseline/input_004/input_005 summaries | ablation config | JSON/MD ablation summary | No | No | reuse | Preferred for input-view ablation interpretation. |
| `make_datav2_frame_mini40_input_view_review.py` | Build mini40 selected-input-view review board and override CSV | mini40 eval config, rendered examples | review board, override CSV, JSON/MD summary | No | No | reuse | Run before A100 inference so per-asset input views are human-reviewed. |
| `make_datav2_frame_mini40_eval_cases.py` | Create mini40 eval cases with explicit selected input views | mini40 eval config, split/curation/override CSVs | eval cases JSON/MD/summary and case input symlinks | No | No | reuse | Uses override CSV first, curated manifest second, config default last. |
| `check_datav2_frame_mini40_eval_readiness.py` | Check mini40 corrected-input eval readiness | mini40 eval config and eval cases | readiness JSON/MD/stdout | No | No | reuse | Gate before Phase 2L.5A A100 inference. |
| `make_datav2_frame_mini40_render_eval_configs.py` | Create render-eval config for mini40 base/fine outputs | mini40 eval config and eval cases | render eval cases JSON/MD | No | No | reuse | Produces split-aware Phase 2L.5B render-eval cases after inference outputs exist. |
| `check_datav2_frame_mini40_render_eval_readiness.py` | Check mini40 rendered-view eval readiness | mini40 eval config and render eval cases | readiness JSON/MD/stdout | No | No | reuse | Gate before manual Blender rendered-view evaluation. |
| `render_datav2_frame_mini40_eval_views_blender.py` | Render mini40 base/fine GLBs from fixed views | mini40 eval config and render eval cases | rendered PNGs and render summary | No | Yes | runtime | Blender-only; run manually, supports `--limit` and `--only-missing`. |
| `compare_datav2_frame_mini40_rendered_views.py` | Compare mini40 rendered views against references | mini40 eval config, rendered PNGs, references | per-case metrics/reports/boards | No | No | reuse | Local rendered-view comparison using Phase 2K metric logic. |
| `aggregate_datav2_frame_mini40_eval.py` | Aggregate mini40 rendered-view metrics by split/view group | mini40 eval config and rendered metrics | summary JSON/MD | No | No | reuse | Separates all, input 005, front, non-front, val/test, and train-sanity metrics. |
| `make_datav2_frame_full80_input_view_review.py` | Build full80 selected-input-view review board and override CSV | full80 eval config, full101 rendered examples | review board, override CSV, JSON/MD summary | No | No | reuse | Phase 2L.7A entry point; defaults to selected input view 005 and includes all val/test plus train sanity. |
| `make_datav2_frame_full80_eval_cases.py` | Create full80 eval cases with explicit selected input views | full80 eval config, split/curation/override CSVs | eval cases JSON/MD/summary and case input symlinks | No | No | reuse | Uses override CSV first, curated manifest second, config default last; never compare to wrong-input baselines. |
| `check_datav2_frame_full80_eval_readiness.py` | Check full80 corrected-input eval readiness | full80 eval config and eval cases | readiness JSON/MD/stdout | No | No | reuse | Gate before Phase 2L.7A A100 inference; checks expected 21 primary eval cases plus optional train sanity when configured. |
| `make_datav2_frame_full80_render_eval_configs.py` | Create render-eval config for full80 base/fine outputs | full80 eval config and eval cases | render eval cases JSON/MD | No | No | reuse | Produces split-aware Phase 2L.7A render-eval cases after inference outputs exist. |
| `check_datav2_frame_full80_render_eval_readiness.py` | Check full80 rendered-view eval readiness | full80 eval config and render eval cases | readiness JSON/MD/stdout | No | No | reuse | Gate before manual Blender rendered-view evaluation. |
| `render_datav2_frame_full80_eval_views_blender.py` | Render full80 base/fine GLBs from fixed views | full80 eval config and render eval cases | rendered PNGs and render summary | No | Yes | runtime | Blender-only; run manually, supports `--limit` and `--only-missing`. |
| `compare_datav2_frame_full80_rendered_views.py` | Compare full80 rendered views against references | full80 eval config, rendered PNGs, references | per-case metrics/reports/boards | No | No | reuse | Local rendered-view comparison using the mini40/Phase 2K metric path. |
| `aggregate_datav2_frame_full80_eval.py` | Aggregate full80 rendered-view metrics by split/view group | full80 eval config and rendered metrics | summary JSON/MD | No | No | reuse | Separates all, input 005, front, non-front, val/test, and train-sanity metrics; mini40 summary is optional context. |

## Shell Helpers

| Script | Purpose | Inputs | Outputs | A100 | Blender | Status | Reuse notes |
|---|---|---|---|---|---|---|
| `local_sanity.sh` | Lightweight local sanity checks | repo env | stdout | No | No | reuse | Safe only if contents are reviewed for current phase. |
| `prepare_official_overfit_config.sh` | Prepare official overfit config | env/configs | config files | No | No | legacy | Phase 0/1 only; avoid for true-PBR work. |
| `storage_report.sh` | Report storage usage | repo paths | stdout | No | No | reuse | Read-only storage check. |

## Preferred Reuse Patterns

- Dataset/package checks: start with `check_hy3dpaint_example.py`,
  `summarize_train_examples.py`, and `check_phase2c_rendered_dataset.py`.
- Case directory creation: reuse `prepare_phase2g_infer_case.py`,
  `prepare_phase2k2_multicase_cases.py`, or
  `prepare_phase2k4_reference_view_cases.py`.
- Hunyuan inference: always go through `run_phase2g_paint_infer.py`.
- UV texture diagnostics: use `make_phase2g6_texture_comparison.py` and
  `aggregate_phase2k2_multicase_metrics.py`.
- Rendered-view diagnostics: use `render_phase2k3_glb_views_blender.py`,
  `compare_phase2k3_rendered_views.py`, and
  `aggregate_phase2k3_rendered_metrics.py`.
- Training initialization changes: use the Phase 2I/2J inspection and
  comparison tools before adding any new training configs.
- Data v2 metadata mining: use `datav2_inventory_metadata_sources.py`,
  `datav2_mine_flat_panel_candidates.py`, and
  `datav2_make_human_review_template.py`; do not download assets during
  Phase 2L.1A.
- ABO geometry metadata mining: use `datav2_mine_abo_geometry_candidates.py`
  and `datav2_make_abo_geometry_review_template.py` when local ABO metadata has
  geometry fields but no useful semantic titles/tags.
- Manual ABO item-id acquisition: use `datav2_resolve_manual_abo_item_ids.py`,
  dry-run `datav2_download_manual_abo_glbs.py`, and
  `datav2_make_manual_abo_review_template.py` when the user has manually
  collected promising ABO IDs from product-page inspection.
- Manual ABO visual inspection: run `datav2_inspect_manual_abo_blender.py`
  manually in Blender, then use `datav2_make_manual_abo_contact_sheets.py` and
  `datav2_update_manual_abo_review_with_inspection.py` before curation.
- Data v2 framed-panel curation/splits: use
  `datav2_build_frame_panel_curated_manifest.py`,
  `datav2_make_frame_panel_splits.py`, and
  `datav2_export_frame_panel_training_plan.py`; do not use original manual
  item-id order for train/test splits.
- Data v2 mini40 Hunyuan rendering: use
  `datav2_build_frame_panel_render_plan.py`,
  manually run `datav2_render_frame_panel_examples_blender.py`, then use
  `datav2_build_frame_panel_examples_json.py` and
  `check_datav2_frame_panel_examples.py` before any A100 training prep.
- Data v2 full101 Hunyuan rendering: reuse the same frame-panel render
  workflow with `configs/datav2_frame_panels_full101_render.json`; build the
  render plan, manually run Blender smoke/full renders, build examples JSON,
  and run `check_datav2_frame_panel_examples.py` before full80 training prep.
- Data v2 mini40 true-PBR training prep: use
  `check_datav2_frame_mini40_training_readiness.py`,
  `env/run_datav2_frame_mini40_train_a100.sbatch`, and
  `inspect_datav2_frame_mini40_checkpoint.py`; do not start full101 training
  until the mini40 run and checkpoint are reviewed.
- Data v2 full80 true-PBR training prep: use
  `check_datav2_frame_full80_training_readiness.py`,
  `env/run_datav2_frame_full80_train_a100.sbatch`, and
  `inspect_datav2_frame_full80_checkpoint.py`; this is the preferred scale-up
  after full101 rendering and mini40 rendered-view review, and it should remain
  a 500-step run until the resulting checkpoint is evaluated.
- Data v2 mini40 corrected-input evaluation: run
  `make_datav2_frame_mini40_input_view_review.py` first, have the user review
  the override CSV, then use `make_datav2_frame_mini40_eval_cases.py`,
  `check_datav2_frame_mini40_eval_readiness.py`,
  `env/run_datav2_frame_mini40_eval_infer_a100.sbatch`,
  `make_datav2_frame_mini40_render_eval_configs.py`,
  `check_datav2_frame_mini40_render_eval_readiness.py`, manually run
  `render_datav2_frame_mini40_eval_views_blender.py`, compare with
  `compare_datav2_frame_mini40_rendered_views.py`, and aggregate with
  `aggregate_datav2_frame_mini40_eval.py`; never compare against old
  wrong-input baselines.
- Data v2 full80 corrected-input evaluation: run
  `make_datav2_frame_full80_input_view_review.py`, review the full80 override
  CSV, then use `make_datav2_frame_full80_eval_cases.py`,
  `check_datav2_frame_full80_eval_readiness.py`,
  `env/run_datav2_frame_full80_eval_infer_a100.sbatch`,
  `make_datav2_frame_full80_render_eval_configs.py`,
  `check_datav2_frame_full80_render_eval_readiness.py`, manually run
  `render_datav2_frame_full80_eval_views_blender.py`, compare with
  `compare_datav2_frame_full80_rendered_views.py`, and aggregate with
  `aggregate_datav2_frame_full80_eval.py`; evaluate all full101 val/test
  assets before considering any 1000-step training.
- ABO visual probe setup: use `datav2_prepare_abo_probe_manifest.py`,
  `datav2_make_abo_download_plan.py`,
  `datav2_inspect_abo_probe_blender.py`, and
  `datav2_make_abo_probe_human_review_template.py`; do not download assets or
  run Blender in Codex.
- ABO dedup acquisition prep: use `datav2_dedupe_abo_probe_candidates.py` and
  `datav2_make_abo_dedup_download_manifest.py` before downloading probe GLBs.
- ABO visual inspection: use `datav2_inspect_abo_dedup_blender.py`,
  `datav2_make_abo_visual_contact_sheets.py`, and
  `datav2_make_abo_visual_human_review.py` after downloaded GLBs exist; do not
  run Blender in Codex.

## Phase 2M LoRA Rescue Tools

| Tool | Purpose | Inputs | Outputs | Requires A100? | Requires Blender? | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/hy3dft/lora/targeting.py` | Exact `ref_dino` LoRA target inventory and validation. | Loaded PyTorch model object. | Target-name list and module metadata. | No by itself | No | Stable/reuse | Reuse before adding new LoRA target selectors. |
| `src/hy3dft/lora/local_linear.py` | Local exact `nn.Linear` LoRA fallback. | PyTorch model and exact target names. | Wrapped modules and adapter-only state dict. | No by itself | No | Stable/reuse | Freezes base parameters and zero-inits B. |
| `src/hy3dft/lora/peft_lora.py` | Backend detection, safe injection policy, trainable parameter checks. | Model, target names, rank/alpha/dropout. | Injection summary. | No by itself | No | Stable/reuse | Defaults to local fallback; PEFT/Diffusers are not auto-used until exact target safety is proven. |
| `src/hy3dft/lora/io.py` | Adapter-only save/load guards. | Adapter state dict. | `*_adapter_state.pt` and `adapter_config.json`. | No | No | Stable/reuse | Refuses full-checkpoint-like names and overwrites. |
| `scripts/phase2m_lora_inventory.py` | Runtime inventory of official inference UNet Linear modules and selected `ref_dino` targets. | Official Hunyuan paths from `env/env.sh`. | `outputs/phase2m/lora_module_inventory.{json,md}`. | Yes | No | Ready/manual | Imports Hunyuan only inside runtime `main()`. |
| `scripts/phase2m_zero_lora_smoke.py` | Zero-update adapter smoke: inject, save adapter-only state, reload, verify upstream tree unchanged. | Official Hunyuan paths from `env/env.sh`. | `outputs/phase2m/zero_lora_smoke/`. | Yes | No | Ready/manual | Does not train, run inference, merge adapters, or save a full model. |
| `env/run_phase2m_lora_inventory_a100.sbatch` | Manual A100 launcher for M0 inventory. | Project env. | Slurm log and inventory outputs. | Yes | No | Ready/manual | Submit manually only. |
| `env/run_phase2m_zero_lora_smoke_a100.sbatch` | Manual A100 launcher for M1 zero-LoRA smoke. | Project env. | Slurm log and adapter smoke outputs. | Yes | No | Ready/manual | Submit manually only after M0 passes. |
| `scripts/check_phase2m_lora_training_ready.py` | Stdlib M2 readiness checker for M0/M1 artifacts, adapter safety flags, M2 output paths, and sbatch partition hygiene. | M0 inventory, M1 smoke output, full80 config, planned M2 paths. | Readiness stdout. | No | No | Stable/reuse | Prefer `zero_lora_smoke_summary.json`, accepts `smoke_summary.json` for compatibility. |
| `scripts/phase2m_train_lora_refdino.py` | Adapter-only M2 training entrypoint for `phase2m_refdino_r4_lr5e5_300`. | Full80 YAML, M0 target inventory, M1 smoke output. | Adapter-only `adapter_step_*.pt`, `adapter_final.pt`, `adapter_config.json`, `training_summary.json`. | Yes | No | Ready/manual | Uses local_linear_fallback only; disables Lightning checkpointing and saves no full model. |
| `env/run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch` | Manual A100 one-step gate for M2 after job 257295 logdir failure. | Project env and M2 readiness artifacts. | Slurm logs and smoke adapter outputs. | Yes | No | Ready/manual | Run before the 300-step M2 rerun; no evaluation. |
| `env/run_phase2m_train_lora_refdino_r4_lr5e5_300_a100.sbatch` | Manual A100 launcher for M2 adapter-only training. | Project env and M2 readiness artifacts. | Slurm logs and Phase 2M adapter outputs. | Yes | No | Ready/manual | Runs readiness first; does not run evaluation. |
| `scripts/check_phase2m_lora_infer_smoke_ready.py` | Stdlib M3A readiness checker for adapter-only LoRA inference smoke. | M2 adapter/config, full80 corrected-input eval cases, M3A sbatch. | Readiness stdout. | No | No | Stable/reuse | Checks local_linear_fallback, target_count 128, selected input view 005, project-local outputs, and Slurm partition hygiene. |
| `scripts/phase2m_lora_infer_smoke.py` | One-case corrected-input base vs LoRA scale-0.75 inference smoke. | Full80 eval cases, adapter_final.pt, adapter_config.json. | `outputs/phase2m/lora_infer_smoke_scale075/` and `smoke_summary.json`. | Yes | No | Ready/manual | Imports Hunyuan and loads adapter only at runtime; no merge and no full-model save. |
| `env/run_phase2m_lora_infer_smoke_scale075_a100.sbatch` | Manual A100 launcher for M3A one-case LoRA inference smoke. | Project env, M2 adapter outputs, full80 eval cases. | Slurm logs and Phase 2M LoRA smoke inference outputs. | Yes | No | Ready/manual | Run before any 24-case LoRA eval; does not run Blender or full eval. |
| `scripts/check_phase2m_lora_multiscale_pilot_ready.py` | Stdlib M3B readiness checker for the LoRA multi-scale pilot. | M2 adapter/config, full80 eval cases, M3A summary, M3B sbatch. | Readiness stdout. | No | No | Stable/reuse | Checks M3A succeeded, exact scales 0.5/0.75/1.0, val/test/train-sanity availability, and Slurm hygiene. |
| `scripts/phase2m_lora_multiscale_pilot.py` | Three-case corrected-input base plus LoRA scale sweep pilot. | Full80 eval cases, adapter_final.pt, adapter_config.json. | `outputs/phase2m/lora_multiscale_pilot/`, `pilot_summary.json`, `cases_used.json`, `per_variant_outputs.json`. | Yes | No | Ready/manual | Fresh-loads model+adapter per LoRA scale and uses absolute scale setting; no merge or full-model save. |
| `env/run_phase2m_lora_multiscale_pilot_a100.sbatch` | Manual A100 launcher for M3B multi-scale pilot. | Project env, M2 adapter outputs, M3A smoke summary, full80 eval cases. | Slurm logs and Phase 2M pilot outputs. | Yes | No | Ready/manual | Run before rendered-view pilot boards or any 24-case LoRA eval. |
| `scripts/check_phase2m_lora_multiscale_pilot_eval_ready.py` | Stdlib M3C readiness checker for the rendered-view LoRA pilot. | M3B summary, per-variant outputs, planned M3C sbatch. | Readiness stdout. | No | No | Stable/reuse | Checks exactly three cases, four variants, existing GLBs, project-local outputs, wrapper scripts, and Slurm hygiene. |
| `scripts/phase2m_render_lora_multiscale_pilot.py` | Blender runtime renderer for M3B base/LoRA pilot GLBs from six fixed views. | M3B `pilot_summary.json` and `per_variant_outputs.json`. | `outputs/phase2m/lora_multiscale_pilot_rendered/` renders and render config. | Yes in sbatch | Yes | Ready/manual | Reuses `render_phase2k3_glb_views_blender.py`; run only through Blender/manual sbatch, not inside Codex. |
| `scripts/aggregate_phase2m_lora_multiscale_pilot_eval.py` | Aggregate rendered-view metrics for base vs LoRA scales. | M3C rendered PNGs and references. | `metrics_rows.csv`, `aggregate_summary.json`, `scale_comparison.json`. | No | No | Stable/reuse | Uses the Phase 2K rendered-view metric functions and splits front/non-front/input/all views. |
| `scripts/make_phase2m_lora_multiscale_pilot_boards.py` | Create one visual board per M3B pilot case. | M3C rendered PNGs and reference images. | `outputs/phase2m/lora_multiscale_pilot_eval/boards/`. | No | No | Stable/reuse | Columns are reference, base, LoRA 0.50, LoRA 0.75, LoRA 1.00; front/input views are labeled. |
| `env/run_phase2m_lora_multiscale_pilot_eval_a100.sbatch` | Manual A100 launcher for M3C rendered-view pilot evaluation. | M3B outputs and project env. | Rendered PNGs, metrics, boards, Slurm logs. | Yes | Yes | Ready/manual | Runs readiness, Blender render, aggregation, and board generation for three cases only. |

## Phase 2N Controlled Evaluation Tools

| Tool | Purpose | Inputs | Outputs | Requires A100? | Requires Blender? | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `scripts/phase2n_build_full_validation_manifest.py` | Resolve and audit the canonical ten-asset validation matrix, six pilot reuses, and four remaining assets. | Full101 split, pilot case config, Phase 2L/2N manifests. | Check-only stdout or one resolved manifest under `outputs/phase2n/full_validation_manifest/`. | No | No | Stable/reuse | Preferred source of truth for full-validation IDs and the 32 reused / 8 new GLB split; never runs models. |
| `scripts/phase2n_build_final_test_manifest.py` | Freeze the selected candidate and audit the canonical 11-asset test matrix. | Completed ten-asset validation, full101 split, Phase 2L baseline run plans, PC-Full checkpoint manifest. | Check-only stdout or one resolved manifest under `outputs/phase2n/final_test_manifest/`. | No | No | Ready/reuse | Preferred source of truth for the 22 reused / 11 new / 33 total final-test GLBs; does not load checkpoints or models. |
| `scripts/phase2n_week2_infer_pilots.py` | Strict scope-checkpoint fixed-mesh inference runner. | Pilot, full-validation, or final-test inference config, audited scope checkpoints, case metadata. | Project-local GLBs, manifests, summaries, and success markers. | Yes for runtime | No | Stable/reuse | Preserves pilot/full-validation profiles and adds the frozen one-candidate/11-test-asset profile; check-only is runtime-free. |
| `scripts/phase2n_week2_evaluate_pilots.py` | Isolated Blender rendering, stable Phase 2K metrics, aggregation, and front/non-front boards. | Pilot, full-validation, or final-test rendered-eval config and resolved GLBs. | Dedicated rendered-evaluation run under `outputs/phase2n/`. | No | Yes for render stage | Stable/reuse | Supports 8 x 6 pilot, 10 x 4 validation, and 11 x 3 final-test matrices without changing renderer or metric formulas. |
| `scripts/check_phase2n_closeout.py` | Read-only, fail-closed verification of the completed Phase 2N record. | `configs/phase2n_closeout.json`, frozen summaries, metrics, manifests, success markers, and analysis packets. | Verification stdout ending in `PHASE2N_CLOSEOUT_OK`; writes nothing. | No | No | Stable/reuse | Preferred final integrity check; validates split isolation, frozen checkpoint identity, exact metrics, asset outcomes, leakage evidence, and packet hashes without loading a checkpoint. |
| `env/run_phase2n_full_validation_infer_a100.sbatch` | Manual A100 launcher for the eight missing full-validation candidate GLBs. | Full-validation manifest and inference configs. | Fixed `phase2n_full_validation_infer_v1` run plus Slurm logs. | Yes | No | Ready/manual | Runs split/reuse and inference readiness first; refuses an existing fixed run and performs no evaluation. |
| `env/run_phase2n_final_test_infer_a100.sbatch` | Manual A100 launcher for the 11 frozen PC-Full final-test GLBs. | Written final-test manifest, freeze and inference configs. | Fixed `phase2n_final_test_infer_v1` run plus Slurm logs. | Yes | No | Ready/manual | Runs freeze/split/reuse and inference readiness first; requires a new output run and never starts rendered evaluation. |

## Phase 2N Reuse Candidates

See `docs/phase2n_day1_architecture_reuse_audit.md` for the exact call chains,
static Hunyuan architecture evidence, and candidate selective scopes.

- **Direct reuse:** full80 corrected-input case creation/readiness,
  `run_phase2g_paint_infer.py`, Phase 2K `render_variant()` and pair metrics,
  and the mini40/full80 split and front/non-front aggregation logic.
- **Parameterize or extend:** reuse the exact-name validation and trainable
  safety ideas in `src/hy3dft/lora/targeting.py` for full-parameter scopes;
  make the Phase 2M render config, aggregator, and board columns accept arbitrary
  case counts and variant names.
- **Historical baselines only:** keep the Phase 2M refview+DINO inventory,
  adapter injection/IO, M2 training, and M3 inference entrypoints unchanged for
  reproducibility. They are not Phase 2N selective full-parameter defaults.
- **Do not reuse:** wrong-initialization Phase 2F/2H training paths, old
  wrong-input evaluations, or broad shared `conv_in`/`conv_out`/all-UNet
  unlocking as a substitute for a proven geometry-focused scope.
