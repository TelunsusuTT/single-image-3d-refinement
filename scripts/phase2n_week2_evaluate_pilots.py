#!/usr/bin/env python3
"""Run the frozen Phase 2N Week 2 rendered-view pilot evaluation.

Check-only mode uses the Python standard library. Runtime rendering delegates to
the stable Phase 2K Blender renderer in one fresh Blender process per GLB.
Metric and boosted-difference calculations delegate to the stable Phase 2K
comparison helpers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase2n_week2_pilot_rendered_eval.json"
EXPECTED_PHASE = "phase2n_week2_pilot_rendered_eval"
FULL_VALIDATION_PHASE = "phase2n_full_validation_rendered_eval"
FINAL_TEST_PHASE = "phase2n_final_test_rendered_eval"
EXPECTED_VARIANTS = [
    "corrected_input_base",
    "historical_full80_500",
    "pc_s1_step160",
    "pc_s1_step320",
    "pc_full_step160",
    "pc_full_step320",
]
FULL_VALIDATION_VARIANTS = [
    "corrected_input_base",
    "historical_full80_500",
    "pc_s1_step160",
    "pc_full_step320",
]
FINAL_TEST_VARIANTS = [
    "corrected_input_base",
    "historical_full80_500",
    "pc_full_step320",
]
EXPECTED_CASE_IDS = [
    "B073P1D981",
    "B075YLXSJC",
    "B075YLQTNP",
    "B075YM2VXJ",
    "B073P52NDX",
    "B073P1S8VZ",
    "B073P16J7Y",
    "B073P1H786",
]
EXPECTED_VIEW_IDS = ["000", "001", "002", "003", "004", "005"]
EXPECTED_VIEW_GROUPS = {
    "all_views": EXPECTED_VIEW_IDS,
    "input_005": ["005"],
    "front_004_005": ["004", "005"],
    "nonfront_000_003": ["000", "001", "002", "003"],
}
EXPECTED_SPLIT_COUNTS = {"val": 6, "train_sanity": 2, "test": 0}
FULL_VALIDATION_SPLIT_COUNTS = {"val": 10, "train_sanity": 0, "test": 0}
FINAL_TEST_SPLIT_COUNTS = {"val": 0, "train_sanity": 0, "test": 11}
BASE_VARIANT = "corrected_input_base"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class EvaluationError(RuntimeError):
    """Raised when the frozen evaluation protocol is not satisfied."""


def is_full_validation_config(config: Mapping[str, Any]) -> bool:
    return config.get("phase") == FULL_VALIDATION_PHASE


def is_final_test_config(config: Mapping[str, Any]) -> bool:
    return config.get("phase") == FINAL_TEST_PHASE


def expected_variants(config: Mapping[str, Any]) -> list[str]:
    if is_final_test_config(config):
        return FINAL_TEST_VARIANTS
    return FULL_VALIDATION_VARIANTS if is_full_validation_config(config) else EXPECTED_VARIANTS


def expected_split_counts(config: Mapping[str, Any]) -> dict[str, int]:
    if is_final_test_config(config):
        return FINAL_TEST_SPLIT_COUNTS
    if is_full_validation_config(config):
        return FULL_VALIDATION_SPLIT_COUNTS
    return EXPECTED_SPLIT_COUNTS


def evaluation_split(config: Mapping[str, Any]) -> str:
    return "test" if is_final_test_config(config) else "val"


def test_data_used(config: Mapping[str, Any]) -> bool:
    return is_final_test_config(config)


def protocol_variants(protocol: Mapping[str, Any]) -> list[str]:
    return expected_variants(protocol["config"])


def protocol_phase(protocol: Mapping[str, Any]) -> str:
    return str(protocol["config"]["phase"])


def runtime_token(config: Mapping[str, Any], stage: str) -> str:
    if is_final_test_config(config):
        return {
            "render": "PHASE2N_FINAL_TEST_RENDER_STAGE_OK",
            "metrics": "PHASE2N_FINAL_TEST_METRICS_STAGE_OK",
            "boards": "PHASE2N_FINAL_TEST_BOARDS_STAGE_OK",
            "complete": "PHASE2N_FINAL_TEST_RENDERED_EVAL_OK",
            "success_file": "PHASE2N_FINAL_TEST_RENDERED_EVAL_SUCCESS",
            "worker": "PHASE2N_FINAL_TEST_RENDER_ONE_OK",
        }[stage]
    if is_full_validation_config(config):
        return {
            "render": "PHASE2N_FULL_VALIDATION_RENDER_STAGE_OK",
            "metrics": "PHASE2N_FULL_VALIDATION_METRICS_STAGE_OK",
            "boards": "PHASE2N_FULL_VALIDATION_BOARDS_STAGE_OK",
            "complete": "PHASE2N_FULL_VALIDATION_RENDERED_EVAL_OK",
            "success_file": "PHASE2N_FULL_VALIDATION_RENDERED_EVAL_SUCCESS",
            "worker": "PHASE2N_FULL_VALIDATION_RENDER_ONE_OK",
        }[stage]
    return {
        "render": "PHASE2N_WEEK2_RENDER_STAGE_OK",
        "metrics": "PHASE2N_WEEK2_METRICS_STAGE_OK",
        "boards": "PHASE2N_WEEK2_BOARDS_STAGE_OK",
        "complete": "PHASE2N_WEEK2_RENDERED_EVAL_OK",
        "success_file": "PHASE2N_WEEK2_RENDERED_EVAL_SUCCESS",
        "worker": "PHASE2N_WEEK2_RENDER_ONE_OK",
    }[stage]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvaluationError(f"expected a JSON object: {path}")
    return data


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise EvaluationError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise EvaluationError(f"empty {label}: {path}")


def validate_png(path: Path, expected_resolution: int | None = None) -> tuple[int, int]:
    """Validate PNG structure and CRCs with the standard library."""

    validate_nonempty_file(path, "PNG")
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise EvaluationError(f"invalid PNG signature: {path}")
        width = height = None
        saw_iend = False
        while True:
            length_data = handle.read(4)
            if not length_data:
                break
            if len(length_data) != 4:
                raise EvaluationError(f"truncated PNG chunk length: {path}")
            length = struct.unpack(">I", length_data)[0]
            chunk_type = handle.read(4)
            chunk_data = handle.read(length)
            crc_data = handle.read(4)
            if len(chunk_type) != 4 or len(chunk_data) != length or len(crc_data) != 4:
                raise EvaluationError(f"truncated PNG chunk: {path}")
            expected_crc = struct.unpack(">I", crc_data)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise EvaluationError(f"PNG CRC mismatch: {path}")
            if chunk_type == b"IHDR":
                if length != 13:
                    raise EvaluationError(f"invalid PNG IHDR length: {path}")
                width, height = struct.unpack(">II", chunk_data[:8])
            elif chunk_type == b"IEND":
                saw_iend = True
                break
        if width is None or height is None or not saw_iend:
            raise EvaluationError(f"incomplete PNG structure: {path}")
        if expected_resolution is not None and (width, height) != (
            expected_resolution,
            expected_resolution,
        ):
            raise EvaluationError(
                f"unexpected PNG dimensions for {path}: {width}x{height}, "
                f"expected {expected_resolution}x{expected_resolution}"
            )
        return int(width), int(height)


def validate_exact_protocol_config(config: dict[str, Any]) -> None:
    phase = config.get("phase")
    if phase not in {EXPECTED_PHASE, FULL_VALIDATION_PHASE, FINAL_TEST_PHASE}:
        raise EvaluationError(
            f"unsupported rendered-evaluation phase: {phase!r}"
        )
    variants = expected_variants(config)
    if config.get("variant_order") != variants:
        raise EvaluationError("variant_order does not match the frozen phase profile")
    if config.get("view_ids") != EXPECTED_VIEW_IDS:
        raise EvaluationError("view_ids must be exactly 000 through 005")
    if config.get("view_groups") != EXPECTED_VIEW_GROUPS:
        raise EvaluationError("view_groups do not match the frozen input/front/non-front protocol")
    if config.get("selected_input_view") != "005":
        raise EvaluationError("selected_input_view must be 005")
    if config.get("reference_lighting") != "AL":
        raise EvaluationError("reference_lighting must be AL")
    if int(config.get("render_resolution", 0)) != 512:
        raise EvaluationError("render_resolution must be 512")
    if config.get("isolated_blender_process_per_glb") is not True:
        raise EvaluationError("isolated_blender_process_per_glb must be true")
    sources = config.get("variant_sources")
    if not isinstance(sources, dict) or list(sources) != variants:
        raise EvaluationError("variant_sources must preserve the exact phase variant order")
    if is_full_validation_config(config):
        if any(source != {"kind": "full_validation_manifest"} for source in sources.values()):
            raise EvaluationError("full validation sources must use the audited manifest")
    if is_final_test_config(config):
        if any(source != {"kind": "final_test_manifest"} for source in sources.values()):
            raise EvaluationError("final-test sources must use the audited manifest")


def validate_frozen_cases(
    frozen: dict[str, Any],
    view_ids: list[str],
    reference_lighting: str,
    project_root: Path,
    validate_files: bool,
) -> list[dict[str, Any]]:
    cases = frozen.get("cases")
    if not isinstance(cases, list):
        raise EvaluationError("frozen case config must contain a cases list")
    case_ids = [str(case.get("asset_id", "")) for case in cases if isinstance(case, dict)]
    if case_ids != EXPECTED_CASE_IDS:
        raise EvaluationError(
            "frozen case IDs/order changed; expected " + ", ".join(EXPECTED_CASE_IDS)
        )
    split_counts = {"val": 0, "train_sanity": 0, "test": 0}
    resolved: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise EvaluationError("frozen case entry is not an object")
        asset_id = str(case["asset_id"])
        split = str(case.get("eval_split", ""))
        if split == "test":
            raise EvaluationError(f"test case is forbidden: {asset_id}")
        if split not in {"val", "train_sanity"}:
            raise EvaluationError(f"unsupported split for {asset_id}: {split!r}")
        split_counts[split] += 1
        if str(case.get("selected_input_view")) != "005":
            raise EvaluationError(f"{asset_id} does not use selected_input_view=005")
        if str(case.get("reference_lighting")) != reference_lighting:
            raise EvaluationError(f"{asset_id} does not use {reference_lighting} lighting")
        selected_reference = resolve_project_path(case["reference_image_path"], project_root)
        render_cond = selected_reference.parent
        references = {
            view_id: render_cond / f"{view_id}_light_{reference_lighting}.png"
            for view_id in view_ids
        }
        if validate_files:
            for reference in references.values():
                validate_png(reference, 512)
        resolved.append(
            {
                "asset_id": asset_id,
                "eval_split": split,
                "source_split": str(case.get("source_split", "")),
                "selection_stratum": str(case.get("selection_stratum", "")),
                "selection_rationale": str(case.get("selection_rationale", "")),
                "selected_input_view": "005",
                "reference_lighting": reference_lighting,
                "reference_images": {
                    view_id: str(path) for view_id, path in references.items()
                },
            }
        )
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise EvaluationError(
            f"frozen split counts changed: {split_counts}, expected {EXPECTED_SPLIT_COUNTS}"
        )
    declared = frozen.get("split_counts")
    if declared != EXPECTED_SPLIT_COUNTS:
        raise EvaluationError(
            f"declared frozen split counts changed: {declared}, expected {EXPECTED_SPLIT_COUNTS}"
        )
    if frozen.get("selection_frozen_before_training") is not True:
        raise EvaluationError("case selection must remain frozen before training")
    return resolved


def baseline_case_index(report: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    section = report.get(key)
    if not isinstance(section, dict) or section.get("status") != "OK":
        raise EvaluationError(f"baseline reuse report section {key!r} is not OK")
    if section.get("all_cases_compatible") is not True:
        raise EvaluationError(f"baseline reuse report section {key!r} is incompatible")
    rows = section.get("cases")
    if not isinstance(rows, list):
        raise EvaluationError(f"baseline reuse report section {key!r} has no cases")
    return {str(row.get("asset_id")): row for row in rows if isinstance(row, dict)}


def resolve_variant_glbs(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    inference_run_dir: Path,
    project_root: Path,
    validate_files: bool,
) -> list[dict[str, Any]]:
    stage4_summary = load_json(inference_run_dir / "summary.json")
    if stage4_summary.get("status") != "OK":
        raise EvaluationError("Stage 4 inference summary is not OK")
    if stage4_summary.get("test_data_used") is not False:
        raise EvaluationError("Stage 4 inference summary indicates test data use")
    if stage4_summary.get("case_count_per_variant") != 8:
        raise EvaluationError("Stage 4 inference did not report eight cases per variant")
    if stage4_summary.get("total_inference_outputs") != 32:
        raise EvaluationError("Stage 4 inference did not report exactly 32 new outputs")
    if stage4_summary.get("variant_order") != EXPECTED_VARIANTS[2:]:
        raise EvaluationError(
            "Stage 4 inference variant order does not match the frozen protocol"
        )
    if stage4_summary.get("split_counts_per_variant") != EXPECTED_SPLIT_COUNTS:
        raise EvaluationError(
            "Stage 4 inference split counts do not match the frozen protocol"
        )
    for variant in EXPECTED_VARIANTS[2:]:
        validate_nonempty_file(
            inference_run_dir / variant / "_SUCCESS",
            f"Stage 4 {variant} success token",
        )
    baseline_report = load_json(inference_run_dir / "baseline_reuse_report.json")
    if baseline_report.get("status") != "OK":
        raise EvaluationError("Stage 4 baseline reuse report is not OK")
    if baseline_report.get("complete_compatible_coverage") is not True:
        raise EvaluationError("Stage 4 baseline reuse coverage is incomplete")

    case_ids = [case["asset_id"] for case in cases]
    sources = config["variant_sources"]
    resolved: list[dict[str, Any]] = []
    for variant in EXPECTED_VARIANTS:
        source = sources[variant]
        kind = source.get("kind")
        per_case: dict[str, Any] = {}
        if kind == "baseline_reuse_report":
            report_key = str(source.get("report_key", ""))
            indexed = baseline_case_index(baseline_report, report_key)
            for case_id in case_ids:
                row = indexed.get(case_id)
                if row is None:
                    raise EvaluationError(f"missing baseline GLB record for {variant}/{case_id}")
                glb = resolve_project_path(row["output_glb_path"], project_root)
                per_case[case_id] = {
                    "glb_path": str(glb),
                    "source_manifest": str(inference_run_dir / "baseline_reuse_report.json"),
                    "source_kind": kind,
                }
        elif kind == "stage4_inference":
            template = str(source.get("manifest_template", ""))
            for case_id in case_ids:
                manifest_path = inference_run_dir / template.format(
                    variant=variant, asset_id=case_id
                )
                manifest = load_json(manifest_path)
                if manifest.get("status") != "OK":
                    raise EvaluationError(f"Stage 4 manifest is not OK: {manifest_path}")
                if manifest.get("variant") != variant:
                    raise EvaluationError(f"variant mismatch in {manifest_path}")
                if manifest.get("asset_id") != case_id:
                    raise EvaluationError(f"asset mismatch in {manifest_path}")
                if manifest.get("test_data_used") is not False:
                    raise EvaluationError(f"test data use reported in {manifest_path}")
                glb = resolve_project_path(manifest["output_glb_path"], project_root)
                per_case[case_id] = {
                    "glb_path": str(glb),
                    "source_manifest": str(manifest_path.resolve()),
                    "source_kind": kind,
                }
        else:
            raise EvaluationError(f"unsupported source kind for {variant}: {kind!r}")
        if list(per_case) != case_ids:
            raise EvaluationError(f"case order mismatch while resolving {variant}")
        if validate_files:
            for case_id, item in per_case.items():
                glb = Path(item["glb_path"])
                validate_nonempty_file(glb, f"source GLB {variant}/{case_id}")
                if glb.suffix.lower() != ".glb":
                    raise EvaluationError(f"source is not a GLB for {variant}/{case_id}: {glb}")
                item["byte_size"] = glb.stat().st_size
        resolved.append(
            {
                "variant": variant,
                "source_kind": kind,
                "cases": per_case,
            }
        )
    return resolved


def load_full_validation_builder(project_root: Path) -> Any:
    source = project_root / "scripts" / "phase2n_build_full_validation_manifest.py"
    spec = importlib.util.spec_from_file_location("_phase2n_full_validation_manifest_eval", source)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"could not load full-validation manifest builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_full_validation_manifest(
    config: dict[str, Any],
    manifest_config_path: Path,
    project_root: Path,
    validate_files: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    builder = load_full_validation_builder(project_root)
    try:
        manifest = builder.build_manifest(
            manifest_config_path,
            project_root,
            require_new_outputs=validate_files,
            validate_files=validate_files,
        )
    except Exception as exc:
        raise EvaluationError(f"full-validation manifest resolution failed: {exc}") from exc
    if manifest.get("validation_ids") != config.get("leakage_risk_assets"):
        raise EvaluationError("full-validation case order differs from the canonical validation split")
    if manifest.get("split_counts") != FULL_VALIDATION_SPLIT_COUNTS:
        raise EvaluationError("full-validation manifest is not val-only 10/0/0")
    cases = [
        {
            "asset_id": row["asset_id"],
            "eval_split": "val",
            "source_split": "val",
            "selection_stratum": "full_validation",
            "selection_rationale": "Canonical full101 validation split.",
            "selected_input_view": "005",
            "reference_lighting": "AL",
            "reference_images": dict(row["reference_paths"]),
        }
        for row in manifest["cases"]
    ]
    source_index = {
        (row["variant_id"], row["asset_id"]): row
        for row in manifest["source_records"]
    }
    variants: list[dict[str, Any]] = []
    for variant in FULL_VALIDATION_VARIANTS:
        per_case = {}
        for case in cases:
            key = (variant, case["asset_id"])
            row = source_index.get(key)
            if row is None:
                raise EvaluationError(f"missing full-validation source record: {variant}/{case['asset_id']}")
            glb = Path(row["source_path"])
            if validate_files:
                validate_nonempty_file(glb, f"source GLB {variant}/{case['asset_id']}")
                if glb.suffix.lower() != ".glb":
                    raise EvaluationError(f"source is not a GLB: {glb}")
            per_case[case["asset_id"]] = {
                "glb_path": str(glb),
                "source_manifest": row["source_manifest"],
                "source_kind": "full_validation_manifest",
                "reuse_status": row["reuse_status"],
                **({"byte_size": glb.stat().st_size} if validate_files else {}),
            }
        variants.append(
            {
                "variant": variant,
                "source_kind": "full_validation_manifest",
                "cases": per_case,
            }
        )
    if len(source_index) != 40:
        raise EvaluationError("full-validation manifest must resolve exactly 40 source GLBs")
    return manifest, cases, variants


def load_final_test_builder(project_root: Path) -> Any:
    source = project_root / "scripts" / "phase2n_build_final_test_manifest.py"
    spec = importlib.util.spec_from_file_location("_phase2n_final_test_manifest_eval", source)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"could not load final-test manifest builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_final_test_manifest(
    config: dict[str, Any],
    manifest_config_path: Path,
    project_root: Path,
    validate_files: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    builder = load_final_test_builder(project_root)
    try:
        manifest = builder.build_manifest(
            manifest_config_path,
            project_root,
            require_new_outputs=validate_files,
            validate_files=validate_files,
        )
    except Exception as exc:
        raise EvaluationError(f"final-test manifest resolution failed: {exc}") from exc
    if manifest.get("test_ids") != config.get("leakage_risk_assets"):
        raise EvaluationError("final-test case order differs from the canonical test split")
    if manifest.get("split_counts") != FINAL_TEST_SPLIT_COUNTS:
        raise EvaluationError("final-test manifest is not test-only 0/0/11")
    if manifest.get("candidate_variants") != ["pc_full_step320"]:
        raise EvaluationError("final-test manifest includes an unfrozen candidate")
    cases = [
        {
            "asset_id": row["asset_id"],
            "eval_split": "test",
            "source_split": "test",
            "selection_stratum": "phase2n_final_test",
            "selection_rationale": (
                "Canonical full101 test split; held out from Phase 2N candidate "
                "and checkpoint selection."
            ),
            "selected_input_view": "005",
            "reference_lighting": "AL",
            "reference_images": dict(row["reference_paths"]),
        }
        for row in manifest["cases"]
    ]
    source_index = {
        (row["variant_id"], row["asset_id"]): row
        for row in manifest["source_records"]
    }
    variants: list[dict[str, Any]] = []
    for variant in FINAL_TEST_VARIANTS:
        per_case = {}
        for case in cases:
            row = source_index.get((variant, case["asset_id"]))
            if row is None:
                raise EvaluationError(
                    f"missing final-test source record: {variant}/{case['asset_id']}"
                )
            glb = Path(row["source_path"])
            if validate_files:
                validate_nonempty_file(glb, f"source GLB {variant}/{case['asset_id']}")
                if glb.suffix.lower() != ".glb":
                    raise EvaluationError(f"source is not a GLB: {glb}")
            per_case[case["asset_id"]] = {
                "glb_path": str(glb),
                "source_manifest": row["source_manifest"],
                "source_kind": "final_test_manifest",
                "reuse_status": row["reuse_status"],
                **({"byte_size": glb.stat().st_size} if validate_files else {}),
            }
        variants.append(
            {
                "variant": variant,
                "source_kind": "final_test_manifest",
                "cases": per_case,
            }
        )
    if len(source_index) != 33:
        raise EvaluationError("final-test manifest must resolve exactly 33 source GLBs")
    return manifest, cases, variants


def validate_helpers(config: dict[str, Any], project_root: Path) -> tuple[Path, Path]:
    renderer = resolve_project_path(config["stable_renderer_script"], project_root)
    metrics = resolve_project_path(config["stable_metrics_script"], project_root)
    validate_nonempty_file(renderer, "stable renderer helper")
    validate_nonempty_file(metrics, "stable metrics helper")
    renderer_text = renderer.read_text(encoding="utf-8")
    metrics_text = metrics.read_text(encoding="utf-8")
    if "def render_variant(" not in renderer_text:
        raise EvaluationError(f"stable renderer lacks render_variant: {renderer}")
    for function_name in ("pair_metrics", "rgb255", "diff_image"):
        if f"def {function_name}(" not in metrics_text:
            raise EvaluationError(
                f"stable metrics helper lacks {function_name}: {metrics}"
            )
    return renderer, metrics


def validate_output_root(output_root: Path, inference_run_dir: Path, project_root: Path) -> None:
    allowed_parent = (project_root / "outputs" / "phase2n").resolve()
    if not is_within(output_root, allowed_parent):
        raise EvaluationError(f"output_root is outside outputs/phase2n: {output_root}")
    if output_root == allowed_parent:
        raise EvaluationError("output_root must be a dedicated Stage 5 directory")
    if output_root == inference_run_dir or is_within(output_root, inference_run_dir):
        raise EvaluationError("output_root must not modify the Stage 4 inference run")
    if output_root.exists() and not output_root.is_dir():
        raise EvaluationError(f"output_root exists but is not a directory: {output_root}")


def build_plan(
    cases: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    view_ids: list[str],
    variant_order: Iterable[str] = EXPECTED_VARIANTS,
) -> list[dict[str, Any]]:
    variant_map = {row["variant"]: row for row in variants}
    rows: list[dict[str, Any]] = []
    for case in cases:
        for variant in variant_order:
            source = variant_map[variant]["cases"][case["asset_id"]]
            for view_id in view_ids:
                rows.append(
                    {
                        "asset_id": case["asset_id"],
                        "eval_split": case["eval_split"],
                        "selection_stratum": case["selection_stratum"],
                        "variant": variant,
                        "view_id": view_id,
                        "source_glb_path": source["glb_path"],
                        "reference_image_path": case["reference_images"][view_id],
                    }
                )
    return rows


def resolve_protocol(
    config_path: Path,
    project_root: Path = PROJECT_ROOT,
    validate_files: bool = True,
) -> dict[str, Any]:
    config_path = resolve_project_path(config_path, project_root)
    config = load_json(config_path)
    validate_exact_protocol_config(config)
    inference_run_dir = resolve_project_path(config["inference_run_dir"], project_root)
    frozen_path = resolve_project_path(config["frozen_cases_config"], project_root)
    output_root = resolve_project_path(config["output_root"], project_root)
    resolved_manifest = None
    if is_final_test_config(config):
        resolved_manifest, cases, variants = resolve_final_test_manifest(
            config,
            frozen_path,
            project_root,
            validate_files,
        )
    elif is_full_validation_config(config):
        resolved_manifest, cases, variants = resolve_full_validation_manifest(
            config,
            frozen_path,
            project_root,
            validate_files,
        )
    else:
        validate_nonempty_file(inference_run_dir / "_SUCCESS", "Stage 4 success token")
        frozen = load_json(frozen_path)
        cases = validate_frozen_cases(
            frozen,
            list(config["view_ids"]),
            str(config["reference_lighting"]),
            project_root,
            validate_files,
        )
        variants = resolve_variant_glbs(
            config,
            cases,
            inference_run_dir,
            project_root,
            validate_files,
        )
    renderer, metrics = validate_helpers(config, project_root)
    blender = resolve_project_path(config["blender_bin"], project_root)
    if validate_files:
        if not blender.is_file() or not os.access(blender, os.X_OK):
            raise EvaluationError(f"Blender executable is missing or not executable: {blender}")
    validate_output_root(output_root, inference_run_dir, project_root)
    plan = build_plan(
        cases,
        variants,
        list(config["view_ids"]),
        expected_variants(config),
    )
    expected = config.get("expected_counts", {})
    actual = {
        "cases": len(cases),
        "val": sum(case["eval_split"] == "val" for case in cases),
        "train_sanity": sum(case["eval_split"] == "train_sanity" for case in cases),
        "test": sum(case["eval_split"] == "test" for case in cases),
        "variants": len(variants),
        "source_glbs": len(cases) * len(variants),
        "references": len(cases) * len(config["view_ids"]),
        "planned_renders": len(plan),
    }
    if expected != actual:
        raise EvaluationError(f"resolved counts {actual} do not match config {expected}")
    return {
        "config_path": str(config_path),
        "config": config,
        "inference_run_dir": str(inference_run_dir),
        "frozen_cases_path": str(frozen_path),
        "output_root": str(output_root),
        "blender_bin": str(blender),
        "stable_renderer_script": str(renderer),
        "stable_metrics_script": str(metrics),
        "cases": cases,
        "variants": variants,
        "plan": plan,
        "counts": actual,
        "resolved_full_validation_manifest": (
            resolved_manifest if is_full_validation_config(config) else None
        ),
        "resolved_final_test_manifest": (
            resolved_manifest if is_final_test_config(config) else None
        ),
    }


def run_check_only(config_path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    protocol = resolve_protocol(config_path, project_root=project_root, validate_files=True)
    config = protocol["config"]
    title = (
        "Phase 2N final-test rendered-view evaluation readiness"
        if is_final_test_config(config)
        else "Phase 2N full-validation rendered-view evaluation readiness"
        if is_full_validation_config(config)
        else "Phase 2N Week 2 rendered-view evaluation readiness"
    )
    print(title)
    print(f"  variants: {', '.join(expected_variants(config))}")
    print(f"  cases: {', '.join(case['asset_id'] for case in protocol['cases'])}")
    print(f"  source_glbs: {protocol['counts']['source_glbs']}")
    print(f"  references: {protocol['counts']['references']}")
    print(f"  planned_renders: {protocol['counts']['planned_renders']}")
    if is_final_test_config(config):
        print("PHASE2N_FINAL_TEST_RENDERED_EVAL_READINESS_OK")
    elif is_full_validation_config(config):
        print("PHASE2N_FULL_VALIDATION_RENDERED_EVAL_READINESS_OK")
    else:
        print("PHASE2N_WEEK2_RENDER_INPUTS_OK")
        print("PHASE2N_WEEK2_RENDER_PROTOCOL_OK")
        print("PHASE2N_WEEK2_NO_TEST_OK")
        print("PHASE2N_WEEK2_RENDERED_EVAL_READINESS_OK")
    return protocol


def render_output_dir(run_root: Path, case: dict[str, Any], variant: str) -> Path:
    return run_root / "renders" / case["eval_split"] / case["asset_id"] / variant


def expected_render_paths(
    run_root: Path,
    case: dict[str, Any],
    variant: str,
    view_ids: Iterable[str],
) -> list[Path]:
    output_dir = render_output_dir(run_root, case, variant)
    return [output_dir / f"{view_id}.png" for view_id in view_ids]


def render_unit_state(
    output_dir: Path,
    view_ids: list[str],
    resolution: int,
) -> str:
    if not output_dir.exists():
        return "missing"
    if not output_dir.is_dir():
        return "inconsistent"
    pngs = sorted(output_dir.glob("*.png"))
    report = output_dir / "render_report.json"
    if not pngs and not report.exists():
        return "missing"
    expected_names = {f"{view_id}.png" for view_id in view_ids}
    if {path.name for path in pngs} != expected_names:
        return "inconsistent"
    try:
        for path in pngs:
            validate_png(path, resolution)
    except EvaluationError:
        return "inconsistent"
    return "complete"


def validate_render_inventory(
    protocol: dict[str, Any],
    run_root: Path,
    require_all: bool,
) -> dict[str, Any]:
    config = protocol["config"]
    view_ids = list(config["view_ids"])
    resolution = int(config["render_resolution"])
    states = {"complete": 0, "missing": 0, "inconsistent": 0}
    units: list[dict[str, str]] = []
    for case in protocol["cases"]:
        for variant in protocol_variants(protocol):
            output_dir = render_output_dir(run_root, case, variant)
            state = render_unit_state(output_dir, view_ids, resolution)
            states[state] += 1
            units.append(
                {
                    "asset_id": case["asset_id"],
                    "eval_split": case["eval_split"],
                    "variant": variant,
                    "output_dir": str(output_dir),
                    "state": state,
                }
            )
    if states["inconsistent"]:
        raise EvaluationError(
            "render inventory is incomplete/inconsistent: "
            f"{states['inconsistent']} inconsistent unit(s)"
        )
    if require_all and states["missing"]:
        raise EvaluationError(
            f"render inventory is incomplete: {states['missing']} unit(s) missing"
        )
    return {"states": states, "units": units}


def runtime_manifest(protocol: dict[str, Any], run_id: str) -> dict[str, Any]:
    source_inventory = []
    variant_map = {row["variant"]: row for row in protocol["variants"]}
    for case in protocol["cases"]:
        for variant in protocol_variants(protocol):
            item = variant_map[variant]["cases"][case["asset_id"]]
            glb = Path(item["glb_path"])
            source_inventory.append(
                {
                    "asset_id": case["asset_id"],
                    "eval_split": case["eval_split"],
                    "variant": variant,
                    "glb_path": str(glb),
                    "byte_size": glb.stat().st_size,
                }
            )
    manifest = {
        "phase": protocol_phase(protocol),
        "run_id": run_id,
        "config_path": protocol["config_path"],
        "config_sha256": sha256_file(Path(protocol["config_path"])),
        "frozen_cases_path": protocol["frozen_cases_path"],
        "frozen_cases_sha256": sha256_file(Path(protocol["frozen_cases_path"])),
        "inference_run_dir": protocol["inference_run_dir"],
        "variant_order": protocol_variants(protocol),
        "case_order": [case["asset_id"] for case in protocol["cases"]],
        "view_ids": list(protocol["config"]["view_ids"]),
        "view_groups": dict(protocol["config"]["view_groups"]),
        "render_resolution": int(protocol["config"]["render_resolution"]),
        "background_color": list(protocol["config"]["background_color"]),
        "stable_renderer_script": protocol["stable_renderer_script"],
        "stable_metrics_script": protocol["stable_metrics_script"],
        "isolated_blender_process_per_glb": True,
        "counts": protocol["counts"],
        "source_inventory": source_inventory,
        "test_data_used": test_data_used(protocol["config"]),
    }
    final_manifest = protocol.get("resolved_final_test_manifest")
    if isinstance(final_manifest, dict):
        manifest["test_data_used_for_selection"] = False
        source_manifests = sorted(
            {
                item["source_manifest"]
                for variant in protocol["variants"]
                for item in variant["cases"].values()
            }
        )
        manifest["final_test_provenance"] = {
            "freeze_record_path": final_manifest["freeze_record_path"],
            "freeze_record_sha256": final_manifest["freeze_record_sha256"],
            "project_git_head": git_head(PROJECT_ROOT),
            "source_manifests": [
                {
                    "path": path,
                    "sha256": sha256_file(Path(path)),
                    "document": load_json(Path(path)),
                }
                for path in source_manifests
            ],
            "checkpoint_manifest_path": final_manifest[
                "candidate_checkpoint_manifest_path"
            ],
            "checkpoint_manifest_sha256": sha256_file(
                Path(final_manifest["candidate_checkpoint_manifest_path"])
            ),
        }
    return manifest


def resolved_cases_document(protocol: dict[str, Any]) -> dict[str, Any]:
    document = {
        "phase": protocol_phase(protocol),
        "case_count": len(protocol["cases"]),
        "split_counts": expected_split_counts(protocol["config"]),
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "test_data_used": test_data_used(protocol["config"]),
        "cases": protocol["cases"],
    }
    if is_final_test_config(protocol["config"]):
        document["test_data_used_for_selection"] = False
    return document


def resolved_variants_document(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": protocol_phase(protocol),
        "variant_count": len(protocol["variants"]),
        "variant_order": protocol_variants(protocol),
        "source_glb_count": protocol["counts"]["source_glbs"],
        "variants": protocol["variants"],
    }


def prepare_run(protocol: dict[str, Any], run_id: str, create: bool) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise EvaluationError(
            "run_id must contain only letters, numbers, dot, underscore, and hyphen"
        )
    output_root = Path(protocol["output_root"])
    run_root = (output_root / run_id).resolve()
    if not is_within(run_root, output_root):
        raise EvaluationError(f"unsafe run directory: {run_root}")
    if run_root.exists() and create and is_final_test_config(protocol["config"]):
        raise EvaluationError(
            f"final-test run already exists; use a new run ID: {run_root}"
        )
    manifest_path = run_root / "00_RUNTIME_MANIFEST.json"
    expected_manifest = runtime_manifest(protocol, run_id)
    if run_root.exists():
        if not manifest_path.is_file():
            raise EvaluationError(
                f"existing run directory has no runtime manifest; refusing overwrite: {run_root}"
            )
        existing = load_json(manifest_path)
        if existing != expected_manifest:
            raise EvaluationError(
                f"existing runtime manifest differs; refusing overwrite: {manifest_path}"
            )
    elif create:
        run_root.mkdir(parents=True)
        write_json_atomic(manifest_path, expected_manifest)
        write_json_atomic(run_root / "resolved_cases.json", resolved_cases_document(protocol))
        write_json_atomic(
            run_root / "resolved_variants.json", resolved_variants_document(protocol)
        )
        final_manifest = protocol.get("resolved_final_test_manifest")
        if isinstance(final_manifest, dict):
            write_json_atomic(run_root / "final_test_freeze.json", final_manifest["freeze"])
            write_json_atomic(run_root / "final_test_source_manifest.json", final_manifest)
            write_json_atomic(
                run_root / "checkpoint_manifest.json",
                load_json(Path(final_manifest["candidate_checkpoint_manifest_path"])),
            )
    else:
        raise EvaluationError(f"run does not exist for requested stage: {run_root}")
    return run_root


def load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stable_metric_helpers(
    metrics_script: Path,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    module = load_module_from_path("_phase2n_stable_metrics", metrics_script)
    return module.pair_metrics, module.rgb255, module.diff_image


def blender_render_one(args: argparse.Namespace) -> int:
    required = {
        "config": args.config,
        "worker_run_dir": args.worker_run_dir,
        "worker_case_id": args.worker_case_id,
        "worker_variant_id": args.worker_variant_id,
        "worker_eval_split": args.worker_eval_split,
        "worker_source_split": args.worker_source_split,
        "input_glb": args.input_glb,
        "output_dir": args.output_dir,
        "stable_renderer": args.stable_renderer,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise EvaluationError(
            "internal Blender mode is missing: " + ", ".join(missing)
        )
    config_path = Path(args.config).resolve()
    worker_config = load_json(config_path)
    validate_exact_protocol_config(worker_config)
    run_dir = Path(args.worker_run_dir).resolve()
    input_glb = Path(args.input_glb).resolve()
    output_dir = Path(args.output_dir).resolve()
    renderer_path = Path(args.stable_renderer).resolve()
    case_id = str(args.worker_case_id)
    variant = str(args.worker_variant_id)
    eval_split = str(args.worker_eval_split)
    source_split = str(args.worker_source_split)
    if is_final_test_config(worker_config):
        manifest_config = resolve_project_path(
            worker_config["frozen_cases_config"], config_path.parents[1]
        )
        builder = load_final_test_builder(config_path.parents[1])
        resolved = builder.build_manifest(
            manifest_config,
            config_path.parents[1],
            require_new_outputs=True,
            validate_files=True,
        )
        allowed_case_ids = resolved["test_ids"]
    elif is_full_validation_config(worker_config):
        manifest_config = resolve_project_path(
            worker_config["frozen_cases_config"], config_path.parents[1]
        )
        builder = load_full_validation_builder(config_path.parents[1])
        resolved = builder.build_manifest(
            manifest_config,
            config_path.parents[1],
            require_new_outputs=True,
            validate_files=True,
        )
        allowed_case_ids = resolved["validation_ids"]
    else:
        allowed_case_ids = EXPECTED_CASE_IDS
    if case_id not in allowed_case_ids:
        raise EvaluationError(f"unexpected internal worker case ID: {case_id}")
    if variant not in expected_variants(worker_config):
        raise EvaluationError(f"unexpected internal worker variant ID: {variant}")
    allowed_splits = (
        {"test"}
        if is_final_test_config(worker_config)
        else {"val"}
        if is_full_validation_config(worker_config)
        else {"val", "train_sanity"}
    )
    if eval_split not in allowed_splits:
        raise EvaluationError(f"unexpected internal worker eval split: {eval_split}")
    expected_source_split = (
        "test"
        if eval_split == "test"
        else "val"
        if eval_split == "val"
        else "train"
    )
    if source_split != expected_source_split:
        raise EvaluationError(
            f"internal worker source split {source_split!r} does not match "
            f"{eval_split!r}"
        )
    configured_output_root = resolve_project_path(
        worker_config["output_root"], config_path.parents[1]
    )
    if not is_within(run_dir, configured_output_root):
        raise EvaluationError(
            f"internal worker run directory is outside output_root: {run_dir}"
        )
    expected_output_dir = run_dir / "renders" / eval_split / case_id / variant
    if output_dir != expected_output_dir:
        raise EvaluationError(
            f"internal worker output directory mismatch: {output_dir} != "
            f"{expected_output_dir}"
        )
    expected_renderer = resolve_project_path(
        worker_config["stable_renderer_script"], config_path.parents[1]
    )
    if renderer_path != expected_renderer:
        raise EvaluationError(
            f"internal worker renderer mismatch: {renderer_path} != "
            f"{expected_renderer}"
        )
    if int(args.internal_resolution) != int(worker_config["render_resolution"]):
        raise EvaluationError("internal worker resolution differs from config")
    if list(args.internal_view_ids) != list(worker_config["view_ids"]):
        raise EvaluationError("internal worker views differ from config")
    if list(args.internal_background_color) != list(
        worker_config["background_color"]
    ):
        raise EvaluationError("internal worker background differs from config")
    validate_nonempty_file(input_glb, "internal render GLB")
    validate_nonempty_file(renderer_path, "internal stable renderer")
    import bpy  # type: ignore

    module = load_module_from_path("_phase2n_stable_renderer", renderer_path)
    report = module.render_variant(
        bpy,
        input_glb,
        output_dir,
        list(args.internal_view_ids),
        int(args.internal_resolution),
        list(args.internal_background_color),
    )
    report["worker_context"] = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "case_id": case_id,
        "variant": variant,
        "eval_split": eval_split,
        "source_split": source_split,
        "input_glb": str(input_glb),
        "output_dir": str(output_dir),
    }
    write_json_atomic(output_dir / "render_report.json", report)
    print(f"rendered isolated GLB: {case_id}/{variant}: {input_glb}")
    print(runtime_token(worker_config, "worker"))
    return 0


def build_blender_worker_command(
    protocol: dict[str, Any],
    run_root: Path,
    case: dict[str, Any],
    variant: str,
    source_glb: Path,
    output_dir: Path,
    script_path: Path | None = None,
) -> list[str]:
    config = protocol["config"]
    script_path = (script_path or Path(__file__)).resolve()
    return [
        protocol["blender_bin"],
        "--background",
        "--python",
        str(script_path),
        "--",
        "--_render-one",
        "--config",
        protocol["config_path"],
        "--worker-run-dir",
        str(run_root),
        "--worker-case-id",
        case["asset_id"],
        "--worker-variant-id",
        variant,
        "--worker-eval-split",
        case["eval_split"],
        "--worker-source-split",
        case["source_split"],
        "--input-glb",
        str(source_glb),
        "--output-dir",
        str(output_dir),
        "--stable-renderer",
        protocol["stable_renderer_script"],
        "--internal-resolution",
        str(config["render_resolution"]),
        "--internal-view-ids",
        *config["view_ids"],
        "--internal-background-color",
        *(str(value) for value in config["background_color"]),
    ]


def run_render_stage(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    inventory = validate_render_inventory(protocol, run_root, require_all=False)
    config = protocol["config"]
    view_ids = list(config["view_ids"])
    resolution = int(config["render_resolution"])
    script_path = Path(__file__).resolve()
    variant_map = {row["variant"]: row for row in protocol["variants"]}
    rows: list[dict[str, Any]] = []
    for case in protocol["cases"]:
        for variant in protocol_variants(protocol):
            output_dir = render_output_dir(run_root, case, variant)
            state = render_unit_state(output_dir, view_ids, resolution)
            if state == "inconsistent":
                raise EvaluationError(
                    f"inconsistent render unit; refusing overwrite: {output_dir}"
                )
            if state == "complete":
                rows.append(
                    {
                        "asset_id": case["asset_id"],
                        "eval_split": case["eval_split"],
                        "variant": variant,
                        "status": "skipped_complete",
                        "output_dir": str(output_dir),
                    }
                )
                print(f"skipped complete: {case['asset_id']}/{variant}")
                continue
            source = variant_map[variant]["cases"][case["asset_id"]]
            command = build_blender_worker_command(
                protocol,
                run_root,
                case,
                variant,
                Path(source["glb_path"]),
                output_dir,
                script_path=script_path,
            )
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log_path = (
                run_root
                / "render_logs"
                / case["eval_split"]
                / case["asset_id"]
                / f"{variant}.log"
            )
            write_text_atomic(log_path, completed.stdout or "")
            if completed.returncode != 0:
                raise EvaluationError(
                    f"Blender failed for {case['asset_id']}/{variant}; see {log_path}"
                )
            if render_unit_state(output_dir, view_ids, resolution) != "complete":
                raise EvaluationError(
                    f"Blender returned success but render unit is incomplete: {output_dir}"
                )
            rows.append(
                {
                    "asset_id": case["asset_id"],
                    "eval_split": case["eval_split"],
                    "variant": variant,
                    "status": "rendered",
                    "output_dir": str(output_dir),
                    "log_path": str(log_path),
                }
            )
            print(f"rendered: {case['asset_id']}/{variant}")
    final_inventory = validate_render_inventory(protocol, run_root, require_all=True)
    summary = {
        "phase": protocol_phase(protocol),
        "status": "OK",
        "isolated_blender_process_per_glb": True,
        "unit_count": len(rows),
        "rendered_png_count": protocol["counts"]["planned_renders"],
        "inventory": final_inventory["states"],
        "rows": rows,
    }
    write_json_atomic(run_root / "render_summary.json", summary)
    print(runtime_token(config, "render"))
    return summary


def row_view_groups(view_id: str, config: dict[str, Any]) -> list[str]:
    return [
        name
        for name, members in config["view_groups"].items()
        if view_id in members
    ]


def subtract_optional(a: Any, b: Any) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return None


def compute_metric_rows(
    protocol: dict[str, Any],
    run_root: Path,
) -> list[dict[str, Any]]:
    from PIL import Image

    pair_metrics, rgb255, _ = load_stable_metric_helpers(
        Path(protocol["stable_metrics_script"])
    )
    config = protocol["config"]
    background_rgb = rgb255(list(config["background_color"]))
    rows: list[dict[str, Any]] = []
    for case in protocol["cases"]:
        split = case["eval_split"]
        for view_id in config["view_ids"]:
            reference_path = Path(case["reference_images"][view_id])
            base_path = (
                render_output_dir(run_root, case, BASE_VARIANT) / f"{view_id}.png"
            )
            reference = Image.open(reference_path).convert("RGB")
            base = Image.open(base_path).convert("RGB")
            if base.size != reference.size:
                base = base.resize(reference.size, Image.Resampling.BICUBIC)
            base_reference = pair_metrics(reference, base, background_rgb)
            for variant in protocol_variants(protocol):
                candidate_path = (
                    render_output_dir(run_root, case, variant) / f"{view_id}.png"
                )
                candidate = Image.open(candidate_path).convert("RGB")
                resized = candidate.size != reference.size
                if resized:
                    candidate = candidate.resize(reference.size, Image.Resampling.BICUBIC)
                candidate_reference = pair_metrics(reference, candidate, background_rgb)
                direct = pair_metrics(base, candidate, background_rgb)
                rows.append(
                    {
                        "asset_id": case["asset_id"],
                        "eval_split": split,
                        "selection_stratum": case["selection_stratum"],
                        "variant": variant,
                        "view_id": view_id,
                        "view_groups": row_view_groups(view_id, config),
                        "selected_input_view": case["selected_input_view"],
                        "reference_path": str(reference_path),
                        "rendered_path": str(candidate_path),
                        "base_rendered_path": str(base_path),
                        "resized_to_reference": resized,
                        "base_mae": base_reference["mae"],
                        "candidate_mae": candidate_reference["mae"],
                        "delta_mae": candidate_reference["mae"]
                        - base_reference["mae"],
                        "base_rmse": base_reference["rmse"],
                        "candidate_rmse": candidate_reference["rmse"],
                        "delta_rmse": candidate_reference["rmse"]
                        - base_reference["rmse"],
                        "base_psnr": base_reference["psnr"],
                        "candidate_psnr": candidate_reference["psnr"],
                        "delta_psnr": subtract_optional(
                            candidate_reference["psnr"], base_reference["psnr"]
                        ),
                        "base_ssim_like": base_reference["ssim_like"],
                        "candidate_ssim_like": candidate_reference["ssim_like"],
                        "delta_ssim_like": candidate_reference["ssim_like"]
                        - base_reference["ssim_like"],
                        "base_histogram_l1": base_reference["histogram_l1"],
                        "candidate_histogram_l1": candidate_reference["histogram_l1"],
                        "delta_histogram_l1": candidate_reference["histogram_l1"]
                        - base_reference["histogram_l1"],
                        "base_edge_difference": base_reference["edge_difference"],
                        "candidate_edge_difference": candidate_reference[
                            "edge_difference"
                        ],
                        "delta_edge_difference": candidate_reference[
                            "edge_difference"
                        ]
                        - base_reference["edge_difference"],
                        "base_mean_abs_color_shift": base_reference[
                            "mean_abs_color_shift"
                        ],
                        "candidate_mean_abs_color_shift": candidate_reference[
                            "mean_abs_color_shift"
                        ],
                        "delta_mean_abs_color_shift": candidate_reference[
                            "mean_abs_color_shift"
                        ]
                        - base_reference["mean_abs_color_shift"],
                        "base_color_mean_shift": base_reference["color_mean_shift"],
                        "candidate_color_mean_shift": candidate_reference[
                            "color_mean_shift"
                        ],
                        "base_foreground_mae": base_reference["foreground_mae"],
                        "candidate_foreground_mae": candidate_reference[
                            "foreground_mae"
                        ],
                        "direct_visual_change_mae": direct["mae"],
                        "direct_visual_change_rmse": direct["rmse"],
                        "direct_visual_change_ssim_like": direct["ssim_like"],
                        "direct_visual_change_histogram_l1": direct["histogram_l1"],
                        "direct_visual_change_edge_difference": direct[
                            "edge_difference"
                        ],
                        "direct_visual_change_mean_abs_color_shift": direct[
                            "mean_abs_color_shift"
                        ],
                    }
                )
    return rows


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := numeric(row, key)) is not None]
    return statistics.fmean(values) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improved = sum(
        1 for row in rows if (value := numeric(row, "delta_mae")) is not None and value < 0
    )
    worsened = sum(
        1 for row in rows if (value := numeric(row, "delta_mae")) is not None and value > 0
    )
    equal = len(rows) - improved - worsened
    improved_ssim = sum(
        1
        for row in rows
        if (value := numeric(row, "delta_ssim_like")) is not None and value > 0
    )
    worsened_ssim = sum(
        1
        for row in rows
        if (value := numeric(row, "delta_ssim_like")) is not None and value < 0
    )
    return {
        "row_count": len(rows),
        "mean_mae": mean_of(rows, "candidate_mae"),
        "mean_rmse": mean_of(rows, "candidate_rmse"),
        "mean_psnr": mean_of(rows, "candidate_psnr"),
        "mean_ssim_like": mean_of(rows, "candidate_ssim_like"),
        "mean_histogram_l1": mean_of(rows, "candidate_histogram_l1"),
        "mean_edge_difference": mean_of(rows, "candidate_edge_difference"),
        "mean_abs_color_shift": mean_of(rows, "candidate_mean_abs_color_shift"),
        "mean_delta_mae": mean_of(rows, "delta_mae"),
        "mean_delta_rmse": mean_of(rows, "delta_rmse"),
        "mean_delta_psnr": mean_of(rows, "delta_psnr"),
        "mean_delta_ssim_like": mean_of(rows, "delta_ssim_like"),
        "mean_delta_histogram_l1": mean_of(rows, "delta_histogram_l1"),
        "mean_delta_edge_difference": mean_of(rows, "delta_edge_difference"),
        "mean_delta_abs_color_shift": mean_of(rows, "delta_mean_abs_color_shift"),
        "mean_direct_visual_change_mae": mean_of(
            rows, "direct_visual_change_mae"
        ),
        "improved_view_count": improved,
        "worsened_view_count": worsened,
        "equal_view_count": equal,
        "improved_ssim_view_count": improved_ssim,
        "worsened_ssim_view_count": worsened_ssim,
    }


def rows_for(
    rows: list[dict[str, Any]],
    *,
    variant: str | None = None,
    split: str | None = None,
    view_group: str | None = None,
    asset_id: str | None = None,
    view_id: str | None = None,
    asset_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (variant is None or row["variant"] == variant)
        and (split is None or row["eval_split"] == split)
        and (view_group is None or view_group in row["view_groups"])
        and (asset_id is None or row["asset_id"] == asset_id)
        and (view_id is None or row["view_id"] == view_id)
        and (asset_ids is None or row["asset_id"] in asset_ids)
    ]


def mean_better(summary: dict[str, Any]) -> bool | None:
    value = summary.get("mean_delta_mae")
    return bool(value < 0) if isinstance(value, (int, float)) else None


def build_aggregate_artifacts(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = protocol["config"]
    variants = protocol_variants(protocol)
    split_names = [
        split for split, count in expected_split_counts(config).items() if count > 0
    ]
    primary_split = evaluation_split(config)
    leakage_assets = set(config["leakage_risk_assets"])
    by_variant = {
        variant: summarize_rows(rows_for(rows, variant=variant))
        for variant in variants
    }
    by_split = {
        split: {
            variant: summarize_rows(rows_for(rows, variant=variant, split=split))
            for variant in variants
        }
        for split in split_names
    }
    by_view_group = {
        group: {
            variant: summarize_rows(
                rows_for(rows, variant=variant, view_group=group)
            )
            for variant in variants
        }
        for group in config["view_groups"]
    }
    by_asset = {
        case["asset_id"]: {
            "eval_split": case["eval_split"],
            "selection_stratum": case["selection_stratum"],
            "variants": {
                variant: summarize_rows(
                    rows_for(rows, variant=variant, asset_id=case["asset_id"])
                )
                for variant in variants
            },
            "front_004_005": {
                variant: summarize_rows(
                    rows_for(
                        rows,
                        variant=variant,
                        asset_id=case["asset_id"],
                        view_group="front_004_005",
                    )
                )
                for variant in variants
            },
            "nonfront_000_003": {
                variant: summarize_rows(
                    rows_for(
                        rows,
                        variant=variant,
                        asset_id=case["asset_id"],
                        view_group="nonfront_000_003",
                    )
                )
                for variant in variants
            },
        }
        for case in protocol["cases"]
    }
    by_view = {
        view_id: {
            variant: summarize_rows(
                rows_for(rows, variant=variant, view_id=view_id)
            )
            for variant in variants
        }
        for view_id in config["view_ids"]
    }
    leakage_subset = {
        "asset_ids": list(config["leakage_risk_assets"]),
        "all_views": {
            variant: summarize_rows(
                rows_for(
                    rows,
                    variant=variant,
                    split=primary_split,
                    asset_ids=leakage_assets,
                )
            )
            for variant in variants
        },
        "nonfront_000_003": {
            variant: summarize_rows(
                rows_for(
                    rows,
                    variant=variant,
                    split=primary_split,
                    view_group="nonfront_000_003",
                    asset_ids=leakage_assets,
                )
            )
            for variant in variants
        },
        "per_asset_nonfront": {
            asset_id: {
                variant: summarize_rows(
                    rows_for(
                        rows,
                        variant=variant,
                        split=primary_split,
                        view_group="nonfront_000_003",
                        asset_id=asset_id,
                    )
                )
                for variant in variants
            }
            for asset_id in config["leakage_risk_assets"]
        },
    }
    evidence: dict[str, Any] = {}
    for variant in variants[1:]:
        primary_all = summarize_rows(
            rows_for(rows, variant=variant, split=primary_split)
        )
        primary_front = summarize_rows(
            rows_for(
                rows,
                variant=variant,
                split=primary_split,
                view_group="front_004_005",
            )
        )
        primary_input = summarize_rows(
            rows_for(
                rows,
                variant=variant,
                split=primary_split,
                view_group="input_005",
            )
        )
        primary_nonfront = summarize_rows(
            rows_for(
                rows,
                variant=variant,
                split=primary_split,
                view_group="nonfront_000_003",
            )
        )
        train_sanity = summarize_rows(
            rows_for(rows, variant=variant, split="train_sanity")
        )
        leakage_regressions = []
        for asset_id in config["leakage_risk_assets"]:
            asset_summary = leakage_subset["per_asset_nonfront"][asset_id][variant]
            delta = asset_summary["mean_delta_mae"]
            if isinstance(delta, (int, float)) and delta > 0:
                leakage_regressions.append(asset_id)
        row = {
            "train_sanity_all_views": train_sanity,
            "mean_delta_mae": primary_all["mean_delta_mae"],
            "mean_delta_ssim_like": primary_all["mean_delta_ssim_like"],
            "improved_view_count": primary_all["improved_view_count"],
            "worsened_view_count": primary_all["worsened_view_count"],
            "direct_visual_change_magnitude_from_base": primary_all[
                "mean_direct_visual_change_mae"
            ],
            f"{primary_split}_front_mean_better_than_base": mean_better(primary_front),
            f"{primary_split}_front_view_win_count": primary_front[
                "improved_view_count"
            ],
            f"{primary_split}_input_mean_better_than_base": mean_better(primary_input),
            f"{primary_split}_nonfront_mean_better_than_base": mean_better(
                primary_nonfront
            ),
            "leakage_risk_regression_count": len(leakage_regressions),
            "leakage_risk_regression_assets": leakage_regressions,
            "train_sanity_mean_better_than_base": mean_better(train_sanity),
        }
        if primary_split == "val":
            row.update(
                {
                    "validation_all_views": primary_all,
                    "validation_front_004_005": primary_front,
                    "validation_input_005": primary_input,
                    "validation_nonfront_000_003": primary_nonfront,
                }
            )
        else:
            row.update(
                {
                    "test_all_views": primary_all,
                    "test_front_004_005": primary_front,
                    "test_input_005": primary_input,
                    "test_nonfront_000_003": primary_nonfront,
                }
            )
        evidence[variant] = row
    leakage_key = (
        "leakage_risk_test" if is_final_test_config(config) else "leakage_risk_validation"
    )
    aggregate = {
        "phase": protocol_phase(protocol),
        "status": "OK" if len(rows) == protocol["counts"]["planned_renders"] else "FAIL",
        "row_count": len(rows),
        "case_count": len(protocol["cases"]),
        "variant_order": variants,
        "view_ids": list(config["view_ids"]),
        "by_variant": by_variant,
        "by_split": by_split,
        "by_view_group": by_view_group,
        "by_asset": by_asset,
        "by_view": by_view,
        leakage_key: leakage_subset,
        "candidate_evidence_relative_to_corrected_input_base": evidence,
        "automatic_winner": None,
        "selection_requires_validation_and_human_board_review": True,
        "test_data_used": test_data_used(config),
    }
    if is_final_test_config(config):
        aggregate.update(
            {
                "selection_requires_validation_and_human_board_review": False,
                "frozen_candidate_evaluation_only": True,
                "checkpoint_replacement_allowed": False,
                "test_data_used_for_selection": False,
            }
        )
    per_asset = {
        "phase": protocol_phase(protocol),
        "asset_count": len(by_asset),
        "assets": by_asset,
    }
    return aggregate, per_asset


def json_value_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def write_metric_rows(metrics_dir: Path, rows: list[dict[str, Any]]) -> None:
    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    write_text_atomic(metrics_dir / "per_view.jsonl", jsonl)
    fields = list(rows[0]) if rows else []
    csv_path = metrics_dir / "per_view.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_value_for_csv(row.get(key)) for key in fields})
    temporary.replace(csv_path)


def aggregate_markdown(aggregate: dict[str, Any]) -> str:
    full_validation = aggregate["phase"] == FULL_VALIDATION_PHASE
    final_test = aggregate["phase"] == FINAL_TEST_PHASE
    split_prefix = "test" if final_test else "val"
    lines = [
        (
            "# Phase 2N Final-Test Rendered-View Aggregate"
            if final_test
            else "# Phase 2N Full Validation Rendered-View Aggregate"
            if full_validation
            else "# Phase 2N Week 2 Pilot Rendered-View Aggregate"
        ),
        "",
        f"status: `{aggregate['status']}`",
        f"rows: `{aggregate['row_count']}`",
        f"cases: `{aggregate['case_count']}`",
        f"test data used: `{str(aggregate['test_data_used']).lower()}`",
        "",
        (
            "This report evaluates the already frozen candidate and cannot select or "
            "replace a checkpoint."
            if final_test
            else "No winner is selected automatically. Validation evidence and visual boards"
        ),
        (
            "Test results are report-only and must not trigger Phase 2N tuning."
            if final_test
            else "must be reviewed with non-front leakage safety checked before front gains."
        ),
        "",
        "## Test Evidence" if final_test else "## Validation Evidence",
        "",
        "| Variant | Split MAE delta | Split SSIM delta | Front better | Front wins | Input better | Non-front better | Leakage regressions | Train-sanity better | Direct change MAE |",
        "|---|---:|---:|---|---:|---|---|---:|---|---:|",
    ]
    evidence = aggregate["candidate_evidence_relative_to_corrected_input_base"]
    for variant in aggregate["variant_order"][1:]:
        row = evidence[variant]
        lines.append(
            f"| `{variant}` | `{row['mean_delta_mae']}` | "
            f"`{row['mean_delta_ssim_like']}` | "
            f"`{row[f'{split_prefix}_front_mean_better_than_base']}` | "
            f"`{row[f'{split_prefix}_front_view_win_count']}` | "
            f"`{row[f'{split_prefix}_input_mean_better_than_base']}` | "
            f"`{row[f'{split_prefix}_nonfront_mean_better_than_base']}` | "
            f"`{row['leakage_risk_regression_count']}` | "
            f"`{row['train_sanity_mean_better_than_base']}` | "
            f"`{row['direct_visual_change_magnitude_from_base']}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def run_metrics_stage(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    metrics_dir = run_root / "metrics"
    if is_final_test_config(protocol["config"]) and metrics_dir.exists():
        raise EvaluationError(
            f"final-test metrics already exist; use a new run ID: {metrics_dir}"
        )
    validate_render_inventory(protocol, run_root, require_all=True)
    rows = compute_metric_rows(protocol, run_root)
    if len(rows) != protocol["counts"]["planned_renders"]:
        raise EvaluationError(
            f"metric row count is {len(rows)}, expected {protocol['counts']['planned_renders']}"
        )
    aggregate, per_asset = build_aggregate_artifacts(rows, protocol)
    if aggregate["status"] != "OK":
        raise EvaluationError("aggregate metrics are incomplete")
    write_metric_rows(metrics_dir, rows)
    write_json_atomic(metrics_dir / "per_asset.json", per_asset)
    write_json_atomic(metrics_dir / "aggregate.json", aggregate)
    write_text_atomic(metrics_dir / "aggregate.md", aggregate_markdown(aggregate))
    print(runtime_token(protocol["config"], "metrics"))
    return aggregate


def board_image_path(
    run_root: Path,
    case: dict[str, Any],
    column: str,
    view_id: str,
) -> Path:
    if column == "reference":
        return Path(case["reference_images"][view_id])
    return render_output_dir(run_root, case, column) / f"{view_id}.png"


def make_one_board(
    protocol: dict[str, Any],
    run_root: Path,
    case: dict[str, Any],
    view_ids: list[str],
    output_path: Path,
    title_suffix: str,
) -> None:
    from PIL import Image, ImageDraw

    _, _, diff_image = load_stable_metric_helpers(
        Path(protocol["stable_metrics_script"])
    )
    columns = ["reference", *protocol_variants(protocol)]
    tile = (170, 170)
    header_height = 58
    row_label_height = 24
    diff_label_height = 20
    block_height = row_label_height + tile[1] + diff_label_height + tile[1]
    board = Image.new(
        "RGB",
        (len(columns) * tile[0], header_height + len(view_ids) * block_height),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(board)
    draw.text(
        (8, 6),
        f"{case['asset_id']} [{case['eval_split']}] - {title_suffix}",
        fill=(0, 0, 0),
    )
    for column_index, column in enumerate(columns):
        draw.text(
            (column_index * tile[0] + 6, 34),
            column,
            fill=(0, 0, 0),
        )
    for row_index, view_id in enumerate(view_ids):
        top = header_height + row_index * block_height
        labels = [view_id]
        if view_id == "005":
            labels.append("input")
        if view_id in {"004", "005"}:
            labels.append("front")
        draw.text((6, top + 5), " / ".join(labels), fill=(0, 0, 0))
        base = Image.open(
            board_image_path(run_root, case, BASE_VARIANT, view_id)
        ).convert("RGB")
        for column_index, column in enumerate(columns):
            x = column_index * tile[0]
            image = Image.open(
                board_image_path(run_root, case, column, view_id)
            ).convert("RGB")
            image = image.resize(tile, Image.Resampling.BICUBIC)
            board.paste(image, (x, top + row_label_height))
            draw.text(
                (x + 5, top + row_label_height + tile[1] + 3),
                "boosted diff to base",
                fill=(40, 40, 40),
            )
            source = Image.open(
                board_image_path(run_root, case, column, view_id)
            ).convert("RGB")
            if source.size != base.size:
                source = source.resize(base.size, Image.Resampling.BICUBIC)
            difference = diff_image(base, source).resize(
                tile, Image.Resampling.BICUBIC
            )
            board.paste(
                difference,
                (x, top + row_label_height + tile[1] + diff_label_height),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path, quality=92)
    validate_nonempty_file(output_path, "board")


def run_boards_stage(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    boards_dir = run_root / "boards"
    if is_final_test_config(protocol["config"]) and boards_dir.exists():
        raise EvaluationError(
            f"final-test boards already exist; use a new run ID: {boards_dir}"
        )
    validate_render_inventory(protocol, run_root, require_all=True)
    rows = []
    for case in protocol["cases"]:
        front_path = (
            run_root
            / "boards"
            / "front"
            / case["eval_split"]
            / f"{case['asset_id']}_front_004_005.jpg"
        )
        nonfront_path = (
            run_root
            / "boards"
            / "nonfront_leakage"
            / case["eval_split"]
            / f"{case['asset_id']}_nonfront_000_003.jpg"
        )
        make_one_board(
            protocol,
            run_root,
            case,
            ["004", "005"],
            front_path,
            "front views 004/005",
        )
        make_one_board(
            protocol,
            run_root,
            case,
            ["000", "001", "002", "003"],
            nonfront_path,
            "non-front/leakage views 000-003",
        )
        rows.append(
            {
                "asset_id": case["asset_id"],
                "eval_split": case["eval_split"],
                "front_board": str(front_path),
                "nonfront_leakage_board": str(nonfront_path),
            }
        )
    summary = {
        "phase": protocol_phase(protocol),
        "status": "OK",
        "asset_count": len(rows),
        "board_count": len(rows) * 2,
        "columns": ["reference", *protocol_variants(protocol)],
        "rows": rows,
    }
    write_json_atomic(run_root / "boards" / "board_summary.json", summary)
    print(runtime_token(protocol["config"], "boards"))
    return summary


def required_final_paths(run_root: Path, protocol: dict[str, Any]) -> list[Path]:
    paths = [
        run_root / "00_RUNTIME_MANIFEST.json",
        run_root / "resolved_cases.json",
        run_root / "resolved_variants.json",
        run_root / "render_summary.json",
        run_root / "metrics" / "per_view.jsonl",
        run_root / "metrics" / "per_view.csv",
        run_root / "metrics" / "per_asset.json",
        run_root / "metrics" / "aggregate.json",
        run_root / "metrics" / "aggregate.md",
        run_root / "boards" / "board_summary.json",
        run_root / "summary.json",
        run_root / "summary.md",
    ]
    if is_final_test_config(protocol["config"]):
        paths.extend(
            [
                run_root / "final_test_freeze.json",
                run_root / "final_test_source_manifest.json",
                run_root / "checkpoint_manifest.json",
            ]
        )
    for case in protocol["cases"]:
        paths.extend(
            [
                run_root
                / "boards"
                / "front"
                / case["eval_split"]
                / f"{case['asset_id']}_front_004_005.jpg",
                run_root
                / "boards"
                / "nonfront_leakage"
                / case["eval_split"]
                / f"{case['asset_id']}_nonfront_000_003.jpg",
            ]
        )
    return paths


def validate_complete_run(
    protocol: dict[str, Any],
    run_root: Path,
    require_success_token: bool = True,
) -> None:
    validate_render_inventory(protocol, run_root, require_all=True)
    for path in required_final_paths(run_root, protocol):
        validate_nonempty_file(path, "final evaluation artifact")
    aggregate = load_json(run_root / "metrics" / "aggregate.json")
    expected_rows = protocol["counts"]["planned_renders"]
    if aggregate.get("status") != "OK" or aggregate.get("row_count") != expected_rows:
        raise EvaluationError(
            f"aggregate metrics are not a complete {expected_rows}-row result"
        )
    expected_test_use = test_data_used(protocol["config"])
    if aggregate.get("test_data_used") is not expected_test_use:
        raise EvaluationError("aggregate test-data-use flag does not match the phase")
    if (
        is_final_test_config(protocol["config"])
        and aggregate.get("test_data_used_for_selection") is not False
    ):
        raise EvaluationError("aggregate permits test data to influence selection")
    if require_success_token:
        validate_nonempty_file(run_root / "_SUCCESS", "evaluation success token")


def finalize_run(
    protocol: dict[str, Any],
    run_root: Path,
    aggregate: dict[str, Any],
    board_summary: dict[str, Any],
) -> dict[str, Any]:
    config = protocol["config"]
    variants = protocol_variants(protocol)
    counts = protocol["counts"]
    full_validation = is_full_validation_config(config)
    final_test = is_final_test_config(config)
    if final_test:
        existing = [
            path
            for path in (run_root / "summary.json", run_root / "summary.md", run_root / "_SUCCESS")
            if path.exists()
        ]
        if existing:
            raise EvaluationError(
                "final-test summary artifacts already exist; use a new run ID: "
                + ", ".join(str(path) for path in existing)
            )
    summary = {
        "phase": protocol_phase(protocol),
        "status": "OK",
        "run_id": run_root.name,
        "variant_order": variants,
        "case_order": [case["asset_id"] for case in protocol["cases"]],
        "split_counts": expected_split_counts(config),
        "view_ids": list(config["view_ids"]),
        "source_glb_count": counts["source_glbs"],
        "rendered_png_count": counts["planned_renders"],
        "metric_row_count": aggregate["row_count"],
        "board_count": board_summary["board_count"],
        "automatic_winner": None,
        "selection_requires_validation_and_human_board_review": True,
        "test_data_used": test_data_used(config),
    }
    if final_test:
        summary.update(
            {
                "selection_requires_validation_and_human_board_review": False,
                "frozen_candidate_evaluation_only": True,
                "checkpoint_replacement_allowed": False,
                "test_data_used_for_selection": False,
            }
        )
    lines = [
        (
            "# Phase 2N Final-Test Rendered-View Evaluation"
            if final_test
            else "# Phase 2N Full Validation Rendered-View Evaluation"
            if full_validation
            else "# Phase 2N Week 2 Pilot Rendered-View Evaluation"
        ),
        "",
        "status: `OK`",
        f"run_id: `{run_root.name}`",
        f"variants: `{len(variants)}`",
        f"cases: `{counts['cases']}` (`{counts['val']}` validation, "
        f"`{counts['train_sanity']}` train-sanity, `{counts['test']}` test)",
        f"rendered PNGs: `{counts['planned_renders']}`",
        f"metric rows: `{aggregate['row_count']}`",
        f"boards: `{board_summary['board_count']}`",
        "",
        (
            "The candidate was frozen before this test; this report cannot select or "
            "replace a checkpoint."
            if final_test
            else "No winner was selected automatically. Use validation only for selection,"
        ),
        (
            "Test results are report-only and must not trigger further Phase 2N tuning."
            if final_test
            else "treat train-sanity as diagnostic, and review non-front leakage boards before"
        ),
        "" if final_test else "considering front-view gains.",
    ]
    write_json_atomic(run_root / "summary.json", summary)
    write_text_atomic(run_root / "summary.md", "\n".join(lines) + "\n")
    write_text_atomic(run_root / "_SUCCESS", runtime_token(config, "success_file") + "\n")
    validate_complete_run(protocol, run_root, require_success_token=True)
    return summary


def execute_runtime(
    protocol: dict[str, Any],
    run_id: str,
    mode: str,
) -> int:
    create = mode in {"run_all", "render_only"}
    run_root = prepare_run(protocol, run_id, create=create)
    success_token = run_root / "_SUCCESS"
    if success_token.is_file():
        validate_complete_run(protocol, run_root, require_success_token=True)
        print(f"complete run already validated: {run_root}")
        print(runtime_token(protocol["config"], "complete"))
        return 0
    aggregate = None
    boards = None
    if mode in {"run_all", "render_only"}:
        run_render_stage(protocol, run_root)
    if mode in {"run_all", "metrics_only"}:
        aggregate = run_metrics_stage(protocol, run_root)
    if mode in {"run_all", "boards_only"}:
        boards = run_boards_stage(protocol, run_root)
    if mode == "run_all":
        assert aggregate is not None and boards is not None
        finalize_run(protocol, run_root, aggregate, boards)
        print(f"evaluation root: {run_root}")
        print(runtime_token(protocol["config"], "complete"))
    return 0


def project_arguments(argv: list[str]) -> list[str]:
    """Return project arguments after the Blender argv boundary."""

    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or run the frozen Phase 2N Week 2 pilot rendered-view evaluation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--run-all", action="store_true")
    modes.add_argument("--render-only", action="store_true")
    modes.add_argument("--metrics-only", action="store_true")
    modes.add_argument("--boards-only", action="store_true")
    modes.add_argument("--_render-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id")
    parser.add_argument("--worker-run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-case-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-eval-split", help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-split", help=argparse.SUPPRESS)
    parser.add_argument("--input-glb", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stable-renderer", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-resolution", type=int, default=512, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--internal-view-ids",
        nargs="+",
        default=list(EXPECTED_VIEW_IDS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-background-color",
        nargs=3,
        type=float,
        default=[0.28, 0.28, 0.28],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(project_arguments(raw_argv))
    try:
        if args._render_one:
            return blender_render_one(args)
        if args.check_only:
            run_check_only(args.config)
            return 0
        if not args.run_id:
            raise EvaluationError("--run-id is required for runtime modes")
        protocol = resolve_protocol(args.config, validate_files=True)
        mode = (
            "run_all"
            if args.run_all
            else "render_only"
            if args.render_only
            else "metrics_only"
            if args.metrics_only
            else "boards_only"
        )
        return execute_runtime(protocol, args.run_id, mode)
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
