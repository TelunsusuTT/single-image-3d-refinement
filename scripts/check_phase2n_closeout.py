#!/usr/bin/env python3
"""Verify the completed Phase 2N closeout without running model code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLIT_COUNTS = {"train": 80, "validation": 10, "test": 11}
FINAL_VARIANTS = [
    "corrected_input_base",
    "historical_full80_500",
    "pc_full_step320",
]
VALIDATION_VARIANTS = [
    "corrected_input_base",
    "historical_full80_500",
    "pc_s1_step160",
    "pc_full_step320",
]
NON_BASE_VARIANTS = ["historical_full80_500", "pc_full_step320"]
VIEW_IDS = ["000", "001", "002", "003", "004", "005"]
METRIC_KEYS = [
    "row_count",
    "mean_mae",
    "mean_rmse",
    "mean_psnr",
    "mean_ssim_like",
    "mean_delta_mae",
    "mean_delta_ssim_like",
    "improved_view_count",
    "worsened_view_count",
    "equal_view_count",
]
SUCCESS_TOKENS = {
    "training": "PHASE2N_WEEK2_PILOT_TRAINING_OK",
    "full_validation": "PHASE2N_FULL_VALIDATION_RENDERED_EVAL_SUCCESS",
    "final_test_inference": "PHASE2N_FINAL_TEST_INFERENCE_OK",
    "final_test_evaluation": "PHASE2N_FINAL_TEST_RENDERED_EVAL_SUCCESS",
}
EXPECTED_RECOMMENDATIONS = {
    "default_safety_recommendation": "corrected_input_base",
    "optional_front_quality_model": "pc_full_step320",
    "historical_comparison_only": "historical_full80_500",
    "validation_ablation_only": "pc_s1_step160",
}


class CloseoutError(RuntimeError):
    """Raised when any frozen closeout invariant fails."""


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def resolve_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, text in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not text.strip():
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _equal(errors: list[str], actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def _required_file(
    errors: list[str], path: Path, label: str, *, nonempty: bool = True
) -> bool:
    if not path.is_file():
        errors.append(f"{label} missing: {path}")
        return False
    if nonempty and path.stat().st_size <= 0:
        errors.append(f"{label} is empty: {path}")
        return False
    return True


def _check_success_marker(
    errors: list[str], run_root: Path, token: str, label: str
) -> Path:
    marker = run_root / "_SUCCESS"
    if _required_file(errors, marker, f"{label} success marker"):
        actual = marker.read_text(encoding="utf-8").strip()
        _equal(errors, actual, token, f"{label} success token")
    return marker


def _run_root(
    record: dict[str, Any], run_name: str, project_root: Path, errors: list[str]
) -> Path:
    try:
        run = record["runs"][run_name]
        _equal(errors, run.get("status"), "OK", f"{run_name} recorded status")
        return resolve_path(run["root"], project_root)
    except (KeyError, TypeError) as exc:
        errors.append(f"invalid runs.{run_name} record: {exc}")
        return project_root / "__missing_phase2n_run__"


def metric_projection(aggregate: dict[str, Any]) -> dict[str, Any]:
    groups = aggregate.get("by_view_group")
    if not isinstance(groups, dict):
        raise ValueError("aggregate by_view_group is missing or invalid")
    projection: dict[str, Any] = {}
    for group_name in (
        "all_views",
        "input_005",
        "front_004_005",
        "nonfront_000_003",
    ):
        variants = groups.get(group_name)
        if not isinstance(variants, dict):
            raise ValueError(f"aggregate view group missing: {group_name}")
        projection[group_name] = {}
        for variant in FINAL_VARIANTS:
            metrics = variants.get(variant)
            if not isinstance(metrics, dict):
                raise ValueError(f"aggregate metrics missing: {group_name}/{variant}")
            projection[group_name][variant] = {
                key: metrics.get(key) for key in METRIC_KEYS
            }
    return projection


def _sign_counts(values: Iterable[float]) -> dict[str, int]:
    result = {"improved": 0, "regressed": 0, "equal": 0}
    for value in values:
        if value < 0:
            result["improved"] += 1
        elif value > 0:
            result["regressed"] += 1
        else:
            result["equal"] += 1
    return result


def asset_level_outcomes(
    aggregate: dict[str, Any], per_view_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_asset = aggregate.get("by_asset")
    if not isinstance(by_asset, dict):
        raise ValueError("aggregate by_asset is missing or invalid")

    outcomes: dict[str, Any] = {}
    for variant in NON_BASE_VARIANTS:
        variant_outcomes: dict[str, Any] = {}
        group_sources = {
            "all_views": "variants",
            "front_004_005": "front_004_005",
            "nonfront_000_003": "nonfront_000_003",
        }
        for group_name, source_key in group_sources.items():
            values: list[float] = []
            for asset_id, asset in by_asset.items():
                try:
                    value = asset[source_key][variant]["mean_delta_mae"]
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        f"missing per-asset metric for {asset_id}/{group_name}/{variant}"
                    ) from exc
                values.append(float(value))
            variant_outcomes[group_name] = _sign_counts(values)

        input_by_asset: dict[str, float] = {}
        for row in per_view_rows:
            if row.get("variant") != variant or str(row.get("view_id")).zfill(3) != "005":
                continue
            asset_id = str(row.get("asset_id"))
            if asset_id in input_by_asset:
                raise ValueError(f"duplicate input-view row for {asset_id}/{variant}")
            input_by_asset[asset_id] = float(row["delta_mae"])
        if set(input_by_asset) != set(by_asset):
            raise ValueError(
                f"input-view asset coverage mismatch for {variant}: "
                f"{len(input_by_asset)} vs {len(by_asset)}"
            )
        variant_outcomes["input_005"] = _sign_counts(input_by_asset.values())
        outcomes[variant] = {
            group: variant_outcomes[group]
            for group in (
                "all_views",
                "input_005",
                "front_004_005",
                "nonfront_000_003",
            )
        }
    return outcomes


def validation_selection_projection(aggregate: dict[str, Any]) -> dict[str, Any]:
    evidence = aggregate.get(
        "candidate_evidence_relative_to_corrected_input_base"
    )
    if not isinstance(evidence, dict):
        raise ValueError("validation candidate evidence is missing")
    projection: dict[str, Any] = {}
    for variant in ("pc_full_step320", "pc_s1_step160"):
        row = evidence.get(variant)
        if not isinstance(row, dict):
            raise ValueError(f"validation evidence missing for {variant}")
        projection[variant] = {
            "all_views_mean_delta_mae": row.get("mean_delta_mae"),
            "all_views_mean_delta_ssim_like": row.get("mean_delta_ssim_like"),
            "improved_view_count": row.get("improved_view_count"),
            "worsened_view_count": row.get("worsened_view_count"),
            "front_mean_better_than_base": row.get(
                "val_front_mean_better_than_base"
            ),
            "input_mean_better_than_base": row.get(
                "val_input_mean_better_than_base"
            ),
            "nonfront_mean_better_than_base": row.get(
                "val_nonfront_mean_better_than_base"
            ),
            "leakage_regression_count": row.get(
                "leakage_risk_regression_count"
            ),
        }
    return projection


def leakage_projection(aggregate: dict[str, Any]) -> dict[str, Any]:
    evidence = aggregate.get(
        "candidate_evidence_relative_to_corrected_input_base"
    )
    if not isinstance(evidence, dict):
        raise ValueError("final-test candidate evidence is missing")
    result: dict[str, Any] = {}
    for variant in NON_BASE_VARIANTS:
        row = evidence.get(variant)
        if not isinstance(row, dict):
            raise ValueError(f"final-test evidence missing for {variant}")
        result[variant] = {
            "count": row.get("leakage_risk_regression_count"),
            "asset_ids": row.get("leakage_risk_regression_assets"),
        }
    return result


def collect_evidence_paths(
    record: dict[str, Any], project_root: Path
) -> list[Path]:
    """Return every small evidence file read by the checker."""
    paths: list[Path] = [
        resolve_path(record["dataset_split_manifest"], project_root),
        resolve_path(record["model_selection"]["freeze_record"]["path"], project_root),
        resolve_path(
            record["model_selection"]["final_candidate"][
                "checkpoint_manifest_path"
            ],
            project_root,
        ),
    ]
    for run_name in (
        "training",
        "full_validation",
        "final_test_inference",
        "final_test_evaluation",
    ):
        root = resolve_path(record["runs"][run_name]["root"], project_root)
        paths.extend([root / "_SUCCESS", root / "summary.json"])
    validation_root = resolve_path(
        record["runs"]["full_validation"]["root"], project_root
    )
    final_infer_root = resolve_path(
        record["runs"]["final_test_inference"]["root"], project_root
    )
    final_eval_root = resolve_path(
        record["runs"]["final_test_evaluation"]["root"], project_root
    )
    paths.extend(
        [
            validation_root / "metrics/aggregate.json",
            validation_root / "metrics/per_view.jsonl",
            final_infer_root / "checkpoint_validation/pc_full_step320.json",
            final_eval_root / "metrics/aggregate.json",
            final_eval_root / "metrics/per_view.jsonl",
            final_eval_root / "00_RUNTIME_MANIFEST.json",
            final_eval_root / "checkpoint_manifest.json",
        ]
    )
    for packet in record["analysis_packets"]:
        paths.append(resolve_path(packet["path"], project_root))
    return sorted(set(paths))


def check_closeout(
    config_path: Path, *, project_root: Path | None = None
) -> dict[str, Any]:
    project_root = (project_root or PROJECT_ROOT).resolve()
    config_path = config_path.resolve()
    record = load_json(config_path)
    errors: list[str] = []

    _equal(errors, record.get("phase"), "phase2n", "phase")
    _equal(errors, record.get("phase_status"), "CLOSED", "phase status")
    _equal(
        errors,
        record.get("fixed_split_counts"),
        EXPECTED_SPLIT_COUNTS,
        "fixed split counts",
    )
    _expect(
        errors,
        record.get("test_results_are_report_only") is True,
        "test_results_are_report_only must be true",
    )
    _expect(
        errors,
        record.get("no_further_phase2n_tuning_permitted") is True,
        "no_further_phase2n_tuning_permitted must be true",
    )

    split_path = resolve_path(record.get("dataset_split_manifest", ""), project_root)
    if _required_file(errors, split_path, "fixed split manifest"):
        split = load_json(split_path)
        counts = split.get("counts", {})
        actual_counts = {
            "train": counts.get("train"),
            "validation": counts.get("val"),
            "test": counts.get("test"),
        }
        _equal(errors, actual_counts, EXPECTED_SPLIT_COUNTS, "split manifest counts")

    roots = {
        name: _run_root(record, name, project_root, errors)
        for name in (
            "training",
            "full_validation",
            "final_test_inference",
            "final_test_evaluation",
        )
    }
    for name, root in roots.items():
        _check_success_marker(errors, root, SUCCESS_TOKENS[name], name)

    summaries: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        path = root / "summary.json"
        if _required_file(errors, path, f"{name} summary"):
            try:
                summary = load_json(path)
                summaries[name] = summary
                _equal(errors, summary.get("status"), "OK", f"{name} summary status")
                _equal(
                    errors,
                    summary.get("run_id"),
                    record["runs"][name]["run_id"],
                    f"{name} run identity",
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{name} summary unreadable: {exc}")

    training = summaries.get("training", {})
    _equal(errors, training.get("optimizer_updates_per_scope"), 320, "training updates")
    for scope in ("pc_s1", "pc_full"):
        scope_summary = training.get("scope_summaries", {}).get(scope, {})
        _equal(errors, scope_summary.get("status"), "OK", f"training {scope} status")
        _expect(
            errors,
            scope_summary.get("test_data_used") is False,
            f"training {scope} must not use test data",
        )

    validation = summaries.get("full_validation", {})
    _equal(
        errors,
        validation.get("split_counts"),
        {"val": 10, "train_sanity": 0, "test": 0},
        "validation split counts",
    )
    _equal(errors, validation.get("source_glb_count"), 40, "validation GLB count")
    _equal(errors, validation.get("metric_row_count"), 240, "validation row count")
    _equal(
        errors,
        validation.get("variant_order"),
        VALIDATION_VARIANTS,
        "validation variants",
    )
    _expect(
        errors,
        validation.get("test_data_used") is False,
        "validation must not use test data",
    )

    validation_aggregate_path = roots["full_validation"] / "metrics/aggregate.json"
    validation_rows_path = roots["full_validation"] / "metrics/per_view.jsonl"
    validation_aggregate: dict[str, Any] = {}
    validation_rows: list[dict[str, Any]] = []
    if _required_file(errors, validation_aggregate_path, "validation aggregate"):
        validation_aggregate = load_json(validation_aggregate_path)
        _equal(
            errors,
            validation_aggregate.get("status"),
            "OK",
            "validation aggregate status",
        )
    if _required_file(errors, validation_rows_path, "validation per-view rows"):
        validation_rows = read_jsonl(validation_rows_path)
        _equal(errors, len(validation_rows), 240, "validation per-view row count")
        _equal(
            errors,
            {row.get("eval_split") for row in validation_rows},
            {"val"},
            "validation row split coverage",
        )
        _equal(
            errors,
            len({row.get("asset_id") for row in validation_rows}),
            10,
            "validation asset count",
        )
        _equal(
            errors,
            {row.get("variant") for row in validation_rows},
            set(VALIDATION_VARIANTS),
            "validation row variants",
        )

    inference = summaries.get("final_test_inference", {})
    _equal(
        errors,
        inference.get("split_counts_per_variant"),
        {"test": 11, "train_sanity": 0, "val": 0},
        "final inference split counts",
    )
    _equal(errors, inference.get("total_inference_outputs"), 11, "inference outputs")
    _equal(
        errors,
        inference.get("variant_order"),
        ["pc_full_step320"],
        "inference variants",
    )
    _expect(
        errors,
        inference.get("test_data_used_for_selection") is False,
        "final inference must not use test data for selection",
    )

    final_summary = summaries.get("final_test_evaluation", {})
    _equal(
        errors,
        final_summary.get("split_counts"),
        {"val": 0, "train_sanity": 0, "test": 11},
        "final evaluation split counts",
    )
    _equal(
        errors,
        final_summary.get("variant_order"),
        FINAL_VARIANTS,
        "final evaluation variants",
    )
    actual_final_counts = {
        "validation_assets": final_summary.get("split_counts", {}).get("val"),
        "train_sanity_assets": final_summary.get("split_counts", {}).get(
            "train_sanity"
        ),
        "test_assets": final_summary.get("split_counts", {}).get("test"),
        "variants": len(final_summary.get("variant_order", [])),
        "source_glbs": final_summary.get("source_glb_count"),
        "rendered_pngs": final_summary.get("rendered_png_count"),
        "metric_rows": final_summary.get("metric_row_count"),
        "boards": final_summary.get("board_count"),
    }
    _equal(
        errors,
        record.get("final_test_counts"),
        actual_final_counts,
        "recorded final-test counts",
    )
    _expect(
        errors,
        final_summary.get("automatic_winner") is None,
        "final evaluation must not select an automatic winner",
    )
    _expect(
        errors,
        final_summary.get("checkpoint_replacement_allowed") is False,
        "final evaluation must prohibit checkpoint replacement",
    )
    _expect(
        errors,
        final_summary.get("test_data_used_for_selection") is False,
        "final evaluation must not use test data for selection",
    )

    final_aggregate_path = roots["final_test_evaluation"] / "metrics/aggregate.json"
    final_rows_path = roots["final_test_evaluation"] / "metrics/per_view.jsonl"
    final_aggregate: dict[str, Any] = {}
    final_rows: list[dict[str, Any]] = []
    if _required_file(errors, final_aggregate_path, "final-test aggregate"):
        final_aggregate = load_json(final_aggregate_path)
        _equal(errors, final_aggregate.get("status"), "OK", "aggregate status")
        _equal(errors, final_aggregate.get("case_count"), 11, "aggregate case count")
        _equal(errors, final_aggregate.get("row_count"), 198, "aggregate row count")
        _equal(
            errors,
            final_aggregate.get("variant_order"),
            FINAL_VARIANTS,
            "aggregate variants",
        )
        _expect(
            errors,
            final_aggregate.get("automatic_winner") is None,
            "aggregate must not select an automatic winner",
        )
        _expect(
            errors,
            final_aggregate.get("checkpoint_replacement_allowed") is False,
            "aggregate must prohibit checkpoint replacement",
        )
        _expect(
            errors,
            final_aggregate.get("test_data_used_for_selection") is False,
            "aggregate must not use test data for selection",
        )
    if _required_file(errors, final_rows_path, "final-test per-view rows"):
        final_rows = read_jsonl(final_rows_path)
        _equal(errors, len(final_rows), 198, "final-test per-view row count")
        _equal(
            errors,
            {row.get("eval_split") for row in final_rows},
            {"test"},
            "final-test row split coverage",
        )
        _equal(
            errors,
            len({row.get("asset_id") for row in final_rows}),
            11,
            "final-test asset count",
        )
        _equal(
            errors,
            {row.get("variant") for row in final_rows},
            set(FINAL_VARIANTS),
            "final-test row variants",
        )

    selection = record.get("model_selection", {})
    _equal(errors, selection.get("basis"), "validation_only", "selection basis")
    _equal(
        errors,
        selection.get("selection_source_split"),
        "validation",
        "selection source split",
    )
    _expect(
        errors,
        selection.get("test_data_used_for_selection") is False,
        "closeout selection must not use test data",
    )
    _expect(
        errors,
        selection.get("new_candidate_selected_from_test") is False,
        "no candidate may be selected from final-test evidence",
    )

    freeze_info = selection.get("freeze_record", {})
    freeze_path = resolve_path(freeze_info.get("path", ""), project_root)
    freeze: dict[str, Any] = {}
    if _required_file(errors, freeze_path, "freeze record"):
        freeze = load_json(freeze_path)
        _equal(
            errors,
            sha256_file(freeze_path),
            freeze_info.get("sha256"),
            "freeze record SHA-256",
        )
        _equal(
            errors,
            freeze.get("selection_source_split"),
            "validation",
            "freeze selection split",
        )
        _equal(
            errors,
            freeze.get("validation_asset_count"),
            10,
            "freeze validation count",
        )
        _expect(
            errors,
            freeze.get("test_data_used_for_selection") is False,
            "freeze evidence must set test_data_used_for_selection=false",
        )
        _expect(
            errors,
            freeze.get("phase2n_test_results_must_not_drive_tuning") is True,
            "freeze must prohibit test-driven tuning",
        )

    candidate = selection.get("final_candidate", {})
    expected_candidate = {
        "variant_id": "pc_full_step320",
        "scope": "pc_full",
        "step": 320,
        "training_run_id": "slurm_264123",
    }
    for key, expected in expected_candidate.items():
        _equal(errors, candidate.get(key), expected, f"final candidate {key}")
    _equal(
        errors,
        selection.get("excluded_candidate"),
        freeze.get("excluded_candidate"),
        "excluded candidate",
    )
    freeze_candidate = freeze.get("selected_candidate", {})
    for key in ("variant_id", "scope", "step", "checkpoint_sha256"):
        _equal(
            errors,
            candidate.get(key),
            freeze_candidate.get(key),
            f"candidate/freeze {key}",
        )

    configured_checkpoint = resolve_path(candidate.get("checkpoint_path", ""), project_root)
    freeze_checkpoint = resolve_path(
        freeze_candidate.get("checkpoint_path", ""), project_root
    )
    _equal(
        errors,
        configured_checkpoint,
        freeze_checkpoint,
        "candidate checkpoint path",
    )
    if _required_file(errors, configured_checkpoint, "frozen checkpoint"):
        _equal(
            errors,
            configured_checkpoint.stat().st_size,
            candidate.get("checkpoint_byte_size"),
            "checkpoint byte size",
        )

    checkpoint_manifest_path = resolve_path(
        candidate.get("checkpoint_manifest_path", ""), project_root
    )
    if _required_file(errors, checkpoint_manifest_path, "checkpoint manifest"):
        checkpoint_manifest = load_json(checkpoint_manifest_path)
        _equal(errors, checkpoint_manifest.get("scope"), "pc_full", "manifest scope")
        _equal(
            errors,
            checkpoint_manifest.get("sha256"),
            candidate.get("checkpoint_sha256"),
            "manifest checkpoint SHA-256",
        )
        _equal(
            errors,
            checkpoint_manifest.get("byte_size"),
            candidate.get("checkpoint_byte_size"),
            "manifest checkpoint size",
        )
        _equal(
            errors,
            resolve_path(checkpoint_manifest.get("checkpoint_path", ""), project_root),
            configured_checkpoint,
            "manifest checkpoint path",
        )

    final_eval_checkpoint_manifest = (
        roots["final_test_evaluation"] / "checkpoint_manifest.json"
    )
    if _required_file(
        errors,
        final_eval_checkpoint_manifest,
        "final-test checkpoint manifest",
    ):
        eval_manifest = load_json(final_eval_checkpoint_manifest)
        _equal(errors, eval_manifest.get("scope"), "pc_full", "eval manifest scope")
        _equal(
            errors,
            eval_manifest.get("sha256"),
            candidate.get("checkpoint_sha256"),
            "eval manifest checkpoint SHA-256",
        )

    checkpoint_validation_path = (
        roots["final_test_inference"]
        / "checkpoint_validation/pc_full_step320.json"
    )
    if _required_file(errors, checkpoint_validation_path, "checkpoint validation"):
        checkpoint_validation = load_json(checkpoint_validation_path)
        identity = checkpoint_validation.get("artifact_validation", {})
        for key, expected in (
            ("scope", "pc_full"),
            ("step", 320),
            ("sha256", candidate.get("checkpoint_sha256")),
        ):
            _equal(
                errors,
                identity.get(key),
                expected,
                f"inference checkpoint identity {key}",
            )

    runtime_path = roots["final_test_evaluation"] / "00_RUNTIME_MANIFEST.json"
    if _required_file(errors, runtime_path, "final-test runtime manifest"):
        runtime = load_json(runtime_path)
        provenance = runtime.get("final_test_provenance", {})
        _equal(
            errors,
            provenance.get("freeze_record_sha256"),
            freeze_info.get("sha256"),
            "runtime freeze SHA-256",
        )
        _equal(
            errors,
            provenance.get("project_git_head"),
            freeze_info.get("git_commit"),
            "freeze git commit",
        )
        _expect(
            errors,
            runtime.get("test_data_used_for_selection") is False,
            "runtime must not use test data for selection",
        )

    protocol = record.get("protocol", {})
    if freeze:
        expected_protocol = {
            "comparison_variants": freeze.get("frozen_comparison_variants"),
            "selected_input_view": freeze.get("selected_input_view"),
            "front_views": freeze.get("front_views"),
            "non_front_views": freeze.get("non_front_views"),
            "all_views": VIEW_IDS,
            "reference_lighting": freeze.get("reference_lighting"),
            "fixed_mesh": freeze.get("fixed_mesh"),
            "use_remesh": not freeze.get("no_remesh"),
        }
        _equal(errors, protocol, expected_protocol, "frozen protocol")

    if validation_aggregate:
        try:
            _equal(
                errors,
                record.get("validation_selection_evidence"),
                validation_selection_projection(validation_aggregate),
                "validation selection evidence",
            )
        except ValueError as exc:
            errors.append(str(exc))

    if final_aggregate:
        try:
            _equal(
                errors,
                record.get("final_test_metrics"),
                metric_projection(final_aggregate),
                "final-test metric projection",
            )
            _equal(
                errors,
                record.get("leakage_regressions_relative_to_base"),
                leakage_projection(final_aggregate),
                "leakage regression evidence",
            )
        except ValueError as exc:
            errors.append(str(exc))
    if final_aggregate and final_rows:
        try:
            _equal(
                errors,
                record.get("asset_level_mae_outcomes"),
                asset_level_outcomes(final_aggregate, final_rows),
                "asset-level MAE outcomes",
            )
        except ValueError as exc:
            errors.append(str(exc))

    _equal(
        errors,
        record.get("deployment_recommendation"),
        EXPECTED_RECOMMENDATIONS,
        "deployment recommendation",
    )
    _equal(
        errors,
        record.get("phase2l_test_exposure_caveat"),
        freeze.get("phase2l_test_history_caveat"),
        "Phase 2L test-exposure caveat",
    )
    _expect(
        errors,
        isinstance(record.get("accepted_scientific_claims"), list)
        and len(record["accepted_scientific_claims"]) >= 7,
        "accepted_scientific_claims must contain the reviewed claims",
    )
    _expect(
        errors,
        isinstance(record.get("prohibited_overclaims"), list)
        and len(record["prohibited_overclaims"]) >= 8,
        "prohibited_overclaims must contain the reviewed boundaries",
    )

    packets = record.get("analysis_packets")
    _expect(
        errors,
        isinstance(packets, list) and len(packets) == 2,
        "exactly two analysis packets must be recorded",
    )
    if isinstance(packets, list):
        _equal(
            errors,
            {packet.get("role") for packet in packets},
            {"full_validation", "final_test"},
            "analysis packet roles",
        )
        for packet in packets:
            packet_path = resolve_path(packet.get("path", ""), project_root)
            role = str(packet.get("role", "unknown"))
            if _required_file(errors, packet_path, f"{role} analysis packet"):
                _equal(
                    errors,
                    packet_path.stat().st_size,
                    packet.get("byte_size"),
                    f"{role} packet byte size",
                )
                _equal(
                    errors,
                    sha256_file(packet_path),
                    packet.get("sha256"),
                    f"{role} packet SHA-256",
                )

    if errors:
        raise CloseoutError("\n".join(errors))
    return {
        "status": "OK",
        "phase_status": "CLOSED",
        "validation_assets": 10,
        "test_assets": 11,
        "metric_rows": 198,
        "final_candidate": "pc_full_step320",
        "default_recommendation": "corrected_input_base",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen Phase 2N closeout record and completed evidence "
            "without loading checkpoints or invoking runtime tools."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/phase2n_closeout.json.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = check_closeout(args.config, project_root=args.project_root)
    except (CloseoutError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print("Phase 2N closeout verification failed:")
        for line in str(exc).splitlines():
            print(f"  FAIL: {line}")
        print("PHASE2N_CLOSEOUT_FAIL")
        return 1

    print("Phase 2N closeout verification")
    print(f"  status: {result['phase_status']}")
    print(f"  validation assets: {result['validation_assets']}")
    print(f"  final-test assets: {result['test_assets']}")
    print(f"  final-test metric rows: {result['metric_rows']}")
    print(f"  frozen candidate: {result['final_candidate']}")
    print(f"  default recommendation: {result['default_recommendation']}")
    print("PHASE2N_CLOSEOUT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

