#!/usr/bin/env python3
"""Resolve the frozen Phase 2N final-test matrix without running models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PHASE = "phase2n_final_test_manifest"
EXPECTED_FREEZE_PHASE = "phase2n_final_test_freeze"
VIEW_IDS = ("000", "001", "002", "003", "004", "005")
VARIANT_ORDER = (
    "corrected_input_base",
    "historical_full80_500",
    "pc_full_step320",
)
CANDIDATE_VARIANT = "pc_full_step320"
EXPECTED_CHECKPOINT_SHA256 = (
    "b93f4d34291a21097b8d763b3d5c18d4b0a82796049ce0711aaac552b310d18e"
)
EXPECTED_TEST_IDS = (
    "B073NZS586",
    "B073P1JNZZ",
    "B075HXJ6Q9",
    "B073P1N4C4",
    "B073NZGLT1",
    "B073P1CKJH",
    "B073P5FLX9",
    "B073P1H7MS",
    "B07HSK626Y",
    "B073P1H6D8",
    "B075YNL763",
)
EXPECTED_COUNTS = {
    "test_assets": 11,
    "validation_assets": 0,
    "train_sanity_assets": 0,
    "variants": 3,
    "source_glbs": 33,
    "reused_glbs": 22,
    "new_inference_glbs": 11,
    "planned_renders": 198,
}
EXPECTED_CONFIG_KEYS = {
    "phase",
    "freeze_record",
    "full101_split",
    "historical_eval_cases",
    "historical_base_root",
    "historical_full80_root",
    "historical_full80_checkpoint",
    "training_run_dir",
    "new_inference_run",
    "output_manifest",
    "selected_input_view",
    "reference_lighting",
    "fixed_mesh",
    "use_remesh",
    "view_ids",
    "variant_order",
    "candidate_variant",
    "expected_counts",
}
EXPECTED_FREEZE_KEYS = {
    "phase",
    "selection_source_run",
    "selection_source_split",
    "validation_asset_count",
    "test_data_used_for_selection",
    "selected_candidate",
    "validation_evidence",
    "excluded_candidate",
    "frozen_comparison_variants",
    "selected_input_view",
    "front_views",
    "non_front_views",
    "reference_lighting",
    "fixed_mesh",
    "no_remesh",
    "no_further_model_checkpoint_or_protocol_changes",
    "phase2l_test_history_caveat",
    "phase2n_test_results_must_not_drive_tuning",
}


class ManifestError(RuntimeError):
    """Raised when the frozen final-test source matrix is not exact."""


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"expected JSON object: {path}")
    return value


def require_nonempty(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ManifestError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ManifestError(f"empty {label}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).expanduser().resolve() == expected.resolve()


def load_shared_builder(project_root: Path) -> Any:
    source = project_root / "scripts" / "phase2n_build_full_validation_manifest.py"
    spec = importlib.util.spec_from_file_location("_phase2n_shared_manifest_helpers", source)
    if spec is None or spec.loader is None:
        raise ManifestError(f"could not load shared manifest helpers: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_manifest_config(config: Mapping[str, Any]) -> None:
    if set(config) != EXPECTED_CONFIG_KEYS:
        raise ManifestError("final-test manifest config fields differ from the frozen schema")
    expected = {
        "phase": EXPECTED_PHASE,
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "fixed_mesh": True,
        "use_remesh": False,
        "view_ids": list(VIEW_IDS),
        "variant_order": list(VARIANT_ORDER),
        "expected_counts": EXPECTED_COUNTS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ManifestError(f"final-test manifest {key} differs from the frozen value")
    candidate = config.get("candidate_variant")
    expected_candidate = {
        "variant_id": CANDIDATE_VARIANT,
        "scope": "pc_full",
        "step": 320,
        "checkpoint_manifest_path": (
            "outputs/phase2n/week2_pilot_training/slurm_264123/pc_full/"
            "checkpoints/step_320_manifest.json"
        ),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    if candidate != expected_candidate:
        raise ManifestError("final-test candidate spec is not frozen PC-Full step 320")


def validate_freeze_record(path: Path, project_root: Path) -> dict[str, Any]:
    freeze = load_json(path)
    if set(freeze) != EXPECTED_FREEZE_KEYS:
        raise ManifestError("final-test freeze fields differ from the frozen schema")
    exact = {
        "phase": EXPECTED_FREEZE_PHASE,
        "selection_source_split": "validation",
        "validation_asset_count": 10,
        "test_data_used_for_selection": False,
        "frozen_comparison_variants": list(VARIANT_ORDER),
        "selected_input_view": "005",
        "front_views": ["004", "005"],
        "non_front_views": ["000", "001", "002", "003"],
        "reference_lighting": "AL",
        "fixed_mesh": True,
        "no_remesh": True,
        "no_further_model_checkpoint_or_protocol_changes": True,
        "phase2n_test_results_must_not_drive_tuning": True,
    }
    for key, value in exact.items():
        if freeze.get(key) != value:
            raise ManifestError(f"freeze record {key} differs from the reviewed decision")
    caveat = freeze.get("phase2l_test_history_caveat")
    if not isinstance(caveat, str) or "held out from Phase 2N candidate" not in caveat:
        raise ManifestError("freeze record omits the Phase 2L test-history caveat")

    selected = freeze.get("selected_candidate")
    if not isinstance(selected, dict):
        raise ManifestError("freeze record selected_candidate is invalid")
    selected_expected = {
        "variant_id": CANDIDATE_VARIANT,
        "role": "quality-oriented final candidate",
        "scope": "pc_full",
        "step": 320,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    for key, value in selected_expected.items():
        if selected.get(key) != value:
            raise ManifestError(f"freeze selected candidate {key} mismatch")
    excluded = freeze.get("excluded_candidate")
    if not isinstance(excluded, dict) or excluded.get("variant_id") != "pc_s1_step160":
        raise ManifestError("freeze must explicitly exclude PC-S1 step 160")
    if excluded.get("status") != "validation ablation only":
        raise ManifestError("excluded PC-S1 status is not frozen")

    source_run = resolve_project_path(freeze["selection_source_run"], project_root)
    success = require_nonempty(source_run / "_SUCCESS", "completed validation success marker")
    if success.read_text(encoding="utf-8") != "PHASE2N_FULL_VALIDATION_RENDERED_EVAL_SUCCESS\n":
        raise ManifestError("completed validation success marker is invalid")
    summary = load_json(require_nonempty(source_run / "summary.json", "validation summary"))
    aggregate = load_json(
        require_nonempty(source_run / "metrics" / "aggregate.json", "validation aggregate")
    )
    if summary.get("status") != "OK" or aggregate.get("status") != "OK":
        raise ManifestError("completed validation status is not OK")
    if summary.get("split_counts") != {"val": 10, "train_sanity": 0, "test": 0}:
        raise ManifestError("completed validation is not the frozen ten-asset val-only run")
    if summary.get("test_data_used") is not False or aggregate.get("test_data_used") is not False:
        raise ManifestError("completed validation does not prove test exclusion")
    if aggregate.get("row_count") != 240 or aggregate.get("case_count") != 10:
        raise ManifestError("completed validation row/case counts are not 240/10")
    evidence = aggregate.get("candidate_evidence_relative_to_corrected_input_base", {}).get(
        CANDIDATE_VARIANT
    )
    if not isinstance(evidence, dict):
        raise ManifestError("completed validation lacks PC-Full step 320 evidence")
    observed = {
        "val_mae_delta": evidence.get("mean_delta_mae"),
        "val_ssim_delta": evidence.get("mean_delta_ssim_like"),
        "front_better": evidence.get("val_front_mean_better_than_base"),
        "input_better": evidence.get("val_input_mean_better_than_base"),
        "nonfront_better": evidence.get("val_nonfront_mean_better_than_base"),
        "leakage_regressions": evidence.get("leakage_risk_regression_count"),
    }
    if freeze.get("validation_evidence") != observed:
        raise ManifestError("freeze validation evidence differs from the completed aggregate")

    checkpoint = require_nonempty(
        resolve_project_path(selected.get("checkpoint_path", ""), project_root),
        "selected checkpoint",
    )
    checkpoint_manifest_path = require_nonempty(
        resolve_project_path(selected.get("checkpoint_manifest_path", ""), project_root),
        "selected checkpoint manifest",
    )
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    if checkpoint_manifest.get("scope") != "pc_full":
        raise ManifestError("selected checkpoint manifest scope is not pc_full")
    if checkpoint_manifest.get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ManifestError("selected checkpoint manifest SHA-256 mismatch")
    if checkpoint_manifest.get("byte_size") != checkpoint.stat().st_size:
        raise ManifestError("selected checkpoint byte size differs from its manifest")
    if not _same_path(checkpoint_manifest.get("checkpoint_path"), checkpoint):
        raise ManifestError("selected checkpoint path differs from its manifest")
    return {
        **freeze,
        "freeze_record_path": str(path),
        "freeze_record_sha256": sha256_file(path),
        "selection_source_run_resolved": str(source_run),
        "checkpoint_path_resolved": str(checkpoint),
        "checkpoint_manifest_path_resolved": str(checkpoint_manifest_path),
    }


def _validate_candidate_manifest(
    manifest_path: Path,
    *,
    asset_id: str,
    output_glb: Path,
    checkpoint_manifest: Path,
) -> None:
    manifest = load_json(require_nonempty(manifest_path, "final-test inference manifest"))
    success = require_nonempty(manifest_path.parent / "_SUCCESS", "final-test case success marker")
    if success.read_text(encoding="utf-8") != "PHASE2N_FINAL_TEST_CASE_OK\n":
        raise ManifestError(f"invalid final-test case success marker: {success}")
    expected = {
        "status": "OK",
        "asset_id": asset_id,
        "variant": CANDIDATE_VARIANT,
        "eval_split": "test",
        "source_split": "test",
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "checkpoint_step": 320,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "test_data_used": True,
        "test_data_used_for_selection": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ManifestError(f"final-test candidate manifest {key} mismatch: {manifest_path}")
    if not _same_path(manifest.get("output_glb_path"), output_glb):
        raise ManifestError(f"final-test candidate output path mismatch: {manifest_path}")
    settings = manifest.get("exact_inference_settings")
    if not isinstance(settings, dict):
        raise ManifestError(f"final-test candidate settings are missing: {manifest_path}")
    if settings.get("fixed_mesh") is not True or settings.get("use_remesh") is not False:
        raise ManifestError(f"final-test candidate does not prove fixed mesh: {manifest_path}")
    require_nonempty(checkpoint_manifest, "PC-Full step 320 checkpoint manifest")


def build_manifest(
    config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    *,
    require_new_outputs: bool = False,
    validate_files: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = resolve_project_path(config_path, root)
    config = load_json(path)
    validate_manifest_config(config)
    shared = load_shared_builder(root)

    freeze_path = resolve_project_path(config["freeze_record"], root)
    freeze = validate_freeze_record(freeze_path, root)
    split_path = resolve_project_path(config["full101_split"], root)
    historical_path = resolve_project_path(config["historical_eval_cases"], root)
    base_root = resolve_project_path(config["historical_base_root"], root)
    full80_root = resolve_project_path(config["historical_full80_root"], root)
    full80_checkpoint = resolve_project_path(config["historical_full80_checkpoint"], root)
    training_run = resolve_project_path(config["training_run_dir"], root)
    new_run = resolve_project_path(config["new_inference_run"], root)
    output_manifest = resolve_project_path(config["output_manifest"], root)
    allowed_output = (root / "outputs" / "phase2n").resolve()
    if not is_within(output_manifest, allowed_output):
        raise ManifestError(f"output_manifest is outside outputs/phase2n: {output_manifest}")
    expected_new_run = (
        root / "outputs/phase2n/final_test_inference/phase2n_final_test_infer_v1"
    ).resolve()
    if new_run != expected_new_run:
        raise ManifestError(f"final-test inference run must be exactly {expected_new_run}")
    if training_run != (
        root / "outputs/phase2n/week2_pilot_training/slurm_264123"
    ).resolve():
        raise ManifestError("training run is not the audited slurm_264123 directory")
    if validate_files:
        require_nonempty(full80_checkpoint, "historical full80 checkpoint")

    split = load_json(split_path)
    if split.get("counts") != {"train": 80, "val": 10, "test": 11}:
        raise ManifestError("full101 declared split counts are not 80/10/11")
    splits = split.get("splits")
    if not isinstance(splits, dict):
        raise ManifestError("full101 split has no splits object")
    split_ids: dict[str, list[str]] = {}
    for split_name, expected_count in (("train", 80), ("val", 10), ("test", 11)):
        rows = splits.get(split_name)
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ManifestError(f"canonical {split_name} split must contain {expected_count} assets")
        ids = [str(row.get("item_id")) for row in rows if isinstance(row, dict)]
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise ManifestError(f"canonical {split_name} IDs are invalid or duplicated")
        split_ids[split_name] = ids
    train_ids, validation_ids, test_ids = (
        split_ids["train"],
        split_ids["val"],
        split_ids["test"],
    )
    if tuple(test_ids) != EXPECTED_TEST_IDS:
        raise ManifestError("canonical test IDs/order differ from the frozen full101 split")
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids):
        raise ManifestError("train/validation/test sets are not disjoint")
    if set(validation_ids) & set(test_ids):
        raise ManifestError("validation/test sets are not disjoint")

    historical = shared._historical_cases(historical_path)
    test_rows = {str(row["item_id"]): row for row in splits["test"]}
    cases: list[dict[str, Any]] = []
    for asset_id in test_ids:
        split_row = test_rows[asset_id]
        if str(split_row.get("selected_input_view")) != "005":
            raise ManifestError(f"canonical selected input view is not 005 for {asset_id}")
        historical_row = historical.get(asset_id)
        if historical_row is None:
            raise ManifestError(f"historical eval metadata is missing test asset {asset_id}")
        if historical_row.get("eval_split") != "test" or historical_row.get("source_split") != "test":
            raise ManifestError(f"historical split mismatch for {asset_id}")
        if historical_row.get("selected_input_view") != "005":
            raise ManifestError(f"historical selected input view is not 005 for {asset_id}")
        mesh = resolve_project_path(split_row.get("local_glb_path", ""), root)
        if not _same_path(historical_row.get("local_mesh_path"), mesh):
            raise ManifestError(f"historical mesh mismatch for {asset_id}")
        raw_references = historical_row.get("reference_images")
        if not isinstance(raw_references, dict) or list(raw_references) != list(VIEW_IDS):
            raise ManifestError(f"references are not exactly 000-005 for {asset_id}")
        references = {
            view_id: str(resolve_project_path(raw_references[view_id], root))
            for view_id in VIEW_IDS
        }
        if Path(references["005"]).name != "005_light_AL.png":
            raise ManifestError(f"selected reference is not 005_light_AL.png for {asset_id}")
        if validate_files:
            require_nonempty(mesh, f"{asset_id} mesh")
            for view_id, reference in references.items():
                require_nonempty(Path(reference), f"{asset_id} reference {view_id}")
        cases.append(
            {
                "asset_id": asset_id,
                "source_split": "test",
                "eval_split": "test",
                "selection_stratum": "phase2n_final_test",
                "selection_rationale": "Canonical full101 test split; held out from Phase 2N selection.",
                "selected_input_view": "005",
                "reference_lighting": "AL",
                "fixed_mesh": True,
                "use_remesh": False,
                "mesh_path": str(mesh),
                "input_image_path": references["005"],
                "reference_image_path": references["005"],
                "reference_paths": references,
            }
        )

    candidate = config["candidate_variant"]
    checkpoint_manifest = resolve_project_path(candidate["checkpoint_manifest_path"], root)
    expected_checkpoint_manifest = (
        training_run / "pc_full" / "checkpoints" / "step_320_manifest.json"
    ).resolve()
    if checkpoint_manifest != expected_checkpoint_manifest:
        raise ManifestError("candidate checkpoint manifest path mismatch")
    if checkpoint_manifest != Path(freeze["checkpoint_manifest_path_resolved"]):
        raise ManifestError("manifest candidate differs from the frozen checkpoint manifest")
    if validate_files:
        require_nonempty(checkpoint_manifest, "PC-Full step 320 checkpoint manifest")

    if require_new_outputs and validate_files:
        root_marker = require_nonempty(new_run / "_SUCCESS", "final-test inference success marker")
        if root_marker.read_text(encoding="utf-8") != "PHASE2N_FINAL_TEST_INFERENCE_OK\n":
            raise ManifestError("invalid final-test root success marker")
        variant_marker = require_nonempty(
            new_run / CANDIDATE_VARIANT / "_SUCCESS",
            "final-test PC-Full step 320 success marker",
        )
        if variant_marker.read_text(encoding="utf-8") != (
            "PHASE2N_FINAL_TEST_PC_FULL_STEP320_INFERENCE_OK\n"
        ):
            raise ManifestError("invalid final-test candidate success marker")
        root_summary = load_json(
            require_nonempty(new_run / "summary.json", "final-test inference summary")
        )
        root_expected = {
            "phase": "phase2n_final_test_inference",
            "status": "OK",
            "variant_order": [CANDIDATE_VARIANT],
            "case_count_per_variant": 11,
            "total_inference_outputs": 11,
            "split_counts_per_variant": {"val": 0, "train_sanity": 0, "test": 11},
            "test_data_used": True,
            "test_data_used_for_selection": False,
        }
        for key, value in root_expected.items():
            if root_summary.get(key) != value:
                raise ManifestError(f"final-test inference summary {key} mismatch")
        variant_summary = load_json(
            require_nonempty(
                new_run / CANDIDATE_VARIANT / "summary.json",
                "final-test candidate summary",
            )
        )
        variant_expected = {
            "phase": "phase2n_final_test_inference",
            "status": "OK",
            "variant": CANDIDATE_VARIANT,
            "scope": "pc_full",
            "checkpoint_step": 320,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "case_count": 11,
            "split_counts": {"val": 0, "train_sanity": 0, "test": 11},
            "test_data_used": True,
            "test_data_used_for_selection": False,
        }
        for key, value in variant_expected.items():
            if variant_summary.get(key) != value:
                raise ManifestError(f"final-test candidate summary {key} mismatch")
        candidate_dir = new_run / CANDIDATE_VARIANT
        actual_case_dirs = {path.name for path in candidate_dir.iterdir() if path.is_dir()}
        if actual_case_dirs != set(test_ids):
            raise ManifestError("final-test candidate case directories are not the exact 11")
        load_report = load_json(
            require_nonempty(
                new_run / "checkpoint_validation" / f"{CANDIDATE_VARIANT}.json",
                "final-test strict checkpoint load report",
            )
        ).get("scope_checkpoint_load")
        if not isinstance(load_report, dict):
            raise ManifestError("final-test strict checkpoint load report is invalid")
        load_expected = {
            "status": "OK",
            "scope": "pc_full",
            "step": 320,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "exact_key_equality": True,
            "loaded_target_equality_verified": True,
            "nonfinite_entry_count": 0,
        }
        for key, value in load_expected.items():
            if load_report.get(key) != value:
                raise ManifestError(f"strict checkpoint load report {key} mismatch")

    source_records: list[dict[str, Any]] = []
    for case in cases:
        asset_id = case["asset_id"]
        mesh = Path(case["mesh_path"])
        references = case["reference_paths"]
        for variant_id in VARIANT_ORDER:
            if variant_id == "corrected_input_base":
                source_path = base_root / "test" / asset_id / "base_textured_mesh.glb"
                source_manifest = base_root / "test" / asset_id / "run_plan.json"
                source_run = base_root.parent.parent
                checkpoint_manifest_for_row = None
                reuse_status = "reused_phase2l"
                if validate_files:
                    require_nonempty(source_path, f"base GLB for {asset_id}")
                    try:
                        shared._validate_historical_run_plan(
                            source_manifest,
                            mode="base",
                            mesh=mesh,
                            input_image=Path(references["005"]),
                            output_glb=source_path,
                            checkpoint=None,
                        )
                    except Exception as exc:
                        raise ManifestError(f"base protocol mismatch for {asset_id}: {exc}") from exc
            elif variant_id == "historical_full80_500":
                source_path = full80_root / "test" / asset_id / "finetuned_textured_mesh.glb"
                source_manifest = full80_root / "test" / asset_id / "run_plan.json"
                source_run = full80_root.parent.parent
                checkpoint_manifest_for_row = None
                reuse_status = "reused_phase2l"
                if validate_files:
                    require_nonempty(source_path, f"full80 GLB for {asset_id}")
                    try:
                        shared._validate_historical_run_plan(
                            source_manifest,
                            mode="finetuned",
                            mesh=mesh,
                            input_image=Path(references["005"]),
                            output_glb=source_path,
                            checkpoint=full80_checkpoint,
                        )
                    except Exception as exc:
                        raise ManifestError(f"full80 protocol mismatch for {asset_id}: {exc}") from exc
            else:
                source_run = new_run
                source_path = source_run / variant_id / asset_id / "textured_mesh.glb"
                source_manifest = source_run / variant_id / asset_id / "inference_manifest.json"
                checkpoint_manifest_for_row = checkpoint_manifest
                reuse_status = "new_inference_required"
                if validate_files and require_new_outputs:
                    require_nonempty(source_path, f"{variant_id} GLB for {asset_id}")
                    _validate_candidate_manifest(
                        source_manifest,
                        asset_id=asset_id,
                        output_glb=source_path,
                        checkpoint_manifest=checkpoint_manifest,
                    )
            source_records.append(
                {
                    "asset_id": asset_id,
                    "eval_split": "test",
                    "source_split": "test",
                    "variant_id": variant_id,
                    "source_path": str(source_path.resolve()),
                    "source_run": str(source_run.resolve()),
                    "source_manifest": str(source_manifest.resolve()),
                    "selected_input_view": "005",
                    "reference_lighting": "AL",
                    "fixed_mesh": True,
                    "use_remesh": False,
                    "mesh_path": str(mesh),
                    "reference_paths": dict(references),
                    "reuse_status": reuse_status,
                    "protocol_compatible": True,
                    "checkpoint_manifest": (
                        str(checkpoint_manifest_for_row)
                        if checkpoint_manifest_for_row is not None
                        else None
                    ),
                }
            )

    reuse_counts = {
        "reused_phase2l": sum(row["reuse_status"] == "reused_phase2l" for row in source_records),
        "new_inference_required": sum(
            row["reuse_status"] == "new_inference_required" for row in source_records
        ),
    }
    actual_counts = {
        "test_assets": len(cases),
        "validation_assets": 0,
        "train_sanity_assets": 0,
        "variants": len(VARIANT_ORDER),
        "source_glbs": len(source_records),
        "reused_glbs": reuse_counts["reused_phase2l"],
        "new_inference_glbs": reuse_counts["new_inference_required"],
        "planned_renders": len(source_records) * len(VIEW_IDS),
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ManifestError(f"resolved counts {actual_counts} differ from {EXPECTED_COUNTS}")

    return {
        "phase": EXPECTED_PHASE,
        "status": "OK",
        "config_path": str(path),
        "freeze_record_path": str(freeze_path),
        "freeze_record_sha256": freeze["freeze_record_sha256"],
        "freeze": freeze,
        "canonical_split_path": str(split_path),
        "historical_eval_cases_path": str(historical_path),
        "output_manifest_path": str(output_manifest),
        "test_ids": test_ids,
        "canonical_train_ids": train_ids,
        "canonical_validation_ids": validation_ids,
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "fixed_mesh": True,
        "use_remesh": False,
        "view_ids": list(VIEW_IDS),
        "variant_order": list(VARIANT_ORDER),
        "candidate_variants": [CANDIDATE_VARIANT],
        "excluded_variants": ["pc_s1_step160"],
        "candidate_checkpoint_manifest_path": str(checkpoint_manifest),
        "split_counts": {"val": 0, "train_sanity": 0, "test": 11},
        "counts": actual_counts,
        "reuse_counts": reuse_counts,
        "new_outputs_validated": bool(require_new_outputs),
        "test_data_used": True,
        "test_data_used_for_selection": False,
        "cases": cases,
        "source_records": source_records,
    }


def write_manifest(manifest: Mapping[str, Any]) -> Path:
    output = Path(str(manifest["output_manifest_path"]))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary manifest already exists: {temporary}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.link(temporary, output)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return output


def print_summary(manifest: Mapping[str, Any]) -> None:
    print("test_ids=" + ",".join(manifest["test_ids"]))
    counts = manifest["counts"]
    print(
        f"reused_glbs={counts['reused_glbs']} "
        f"new_inference_glbs={counts['new_inference_glbs']} "
        f"source_glbs={counts['source_glbs']} planned_renders={counts['planned_renders']}"
    )
    print("PHASE2N_FINAL_TEST_FREEZE_OK")
    print("PHASE2N_FINAL_TEST_SPLIT_OK")
    print("PHASE2N_FINAL_TEST_REUSE_OK")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or write the frozen Phase 2N final-test source manifest."
    )
    parser.add_argument("--config", required=True, help="Final-test manifest config")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="Validate without writing")
    modes.add_argument("--write", action="store_true", help="Write the resolved manifest once")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.config)
    print_summary(manifest)
    if args.write:
        print(f"wrote_manifest={write_manifest(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
