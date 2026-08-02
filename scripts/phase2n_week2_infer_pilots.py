#!/usr/bin/env python3
"""Check or run Phase 2N Week 2 scope-checkpoint pilot inference.

Check-only mode is standard-library-only and never imports Torch or Hunyuan.
The runtime path is intended only for the accompanying A100 Slurm job.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.hunyuan_inference import (  # noqa: E402
    initialize_base_paint_pipeline,
    run_fixed_mesh_inference,
)
from hy3dft.scope_checkpoint import (  # noqa: E402
    load_scope_checkpoint_into_model,
    sha256_file,
    validate_checkpoint_artifact,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase2n_week2_pilot_inference.json"
PILOT_PHASE = "phase2n_week2_pilot_inference"
FULL_VALIDATION_PHASE = "phase2n_full_validation_inference"
FINAL_TEST_PHASE = "phase2n_final_test_inference"
EXPECTED_OUTPUT_RELATIVE = Path("outputs/phase2n/week2_pilot_inference")
FULL_VALIDATION_OUTPUT_RELATIVE = Path("outputs/phase2n/full_validation_inference")
FINAL_TEST_OUTPUT_RELATIVE = Path("outputs/phase2n/final_test_inference")
EXPECTED_TRAINING_RUN_RELATIVE = Path("outputs/phase2n/week2_pilot_training/slurm_264123")
EXPECTED_EVAL_CONFIG_RELATIVE = Path("configs/phase2n_week2_pilot_eval_cases.json")
FULL_VALIDATION_EVAL_CONFIG_RELATIVE = Path("configs/phase2n_full_validation_manifest.json")
FINAL_TEST_EVAL_CONFIG_RELATIVE = Path("configs/phase2n_final_test_manifest.json")
EXPECTED_VARIANT_ORDER = (
    "pc_s1_step160",
    "pc_s1_step320",
    "pc_full_step160",
    "pc_full_step320",
)
EXPECTED_VARIANTS = {
    "pc_s1_step160": ("pc_s1", 160, 80, 49_574_080),
    "pc_s1_step320": ("pc_s1", 320, 80, 49_574_080),
    "pc_full_step160": ("pc_full", 160, 981, 1_046_761_668),
    "pc_full_step320": ("pc_full", 320, 981, 1_046_761_668),
}
FULL_VALIDATION_VARIANT_ORDER = (
    "pc_s1_step160",
    "pc_full_step320",
)
FINAL_TEST_VARIANT_ORDER = ("pc_full_step320",)
VARIANT_SUCCESS_TOKENS = {
    "pc_s1_step160": "PHASE2N_WEEK2_PC_S1_STEP160_INFERENCE_OK",
    "pc_s1_step320": "PHASE2N_WEEK2_PC_S1_STEP320_INFERENCE_OK",
    "pc_full_step160": "PHASE2N_WEEK2_PC_FULL_STEP160_INFERENCE_OK",
    "pc_full_step320": "PHASE2N_WEEK2_PC_FULL_STEP320_INFERENCE_OK",
}
FULL_VALIDATION_VARIANT_SUCCESS_TOKENS = {
    "pc_s1_step160": "PHASE2N_FULL_VALIDATION_PC_S1_STEP160_INFERENCE_OK",
    "pc_full_step320": "PHASE2N_FULL_VALIDATION_PC_FULL_STEP320_INFERENCE_OK",
}
FINAL_TEST_VARIANT_SUCCESS_TOKENS = {
    "pc_full_step320": "PHASE2N_FINAL_TEST_PC_FULL_STEP320_INFERENCE_OK",
}
FINAL_SUCCESS_TOKEN = "PHASE2N_WEEK2_PILOT_INFERENCE_OK"
FULL_VALIDATION_FINAL_SUCCESS_TOKEN = "PHASE2N_FULL_VALIDATION_INFERENCE_OK"
FINAL_TEST_FINAL_SUCCESS_TOKEN = "PHASE2N_FINAL_TEST_INFERENCE_OK"
FINAL_TEST_CASE_SUCCESS_TOKEN = "PHASE2N_FINAL_TEST_CASE_OK"
TRAINING_SUCCESS_TOKENS = {
    "root": "PHASE2N_WEEK2_PILOT_TRAINING_OK",
    "pc_s1": "PHASE2N_WEEK2_PC_S1_TRAINING_OK",
    "pc_full": "PHASE2N_WEEK2_PC_FULL_TRAINING_OK",
}
EXPECTED_CONFIG_KEYS = {
    "phase",
    "training_run_dir",
    "eval_cases_config",
    "output_root",
    "inference_seed",
    "selected_input_view",
    "reference_lighting",
    "max_num_view",
    "resolution",
    "minimum_gpu_memory_gib",
    "minimum_free_disk_gib",
    "fixed_mesh",
    "use_remesh",
    "save_glb",
    "fresh_official_true_pbr_base_per_variant",
    "variant_order",
    "variants",
    "baseline_reuse",
}
FULL_VALIDATION_CONFIG_KEYS = EXPECTED_CONFIG_KEYS | {
    "fixed_run_id",
    "fail_if_run_exists",
}


@dataclass(frozen=True)
class ValidatedInferenceConfig:
    values: dict[str, Any]
    config_path: Path
    project_root: Path
    training_run_dir: Path
    eval_cases_config: Path
    output_root: Path
    variants: dict[str, dict[str, Any]]
    cases: tuple[dict[str, Any], ...]
    training_report: dict[str, Any]
    baseline_reuse_report: dict[str, Any]


def is_full_validation(values: Mapping[str, Any]) -> bool:
    return values.get("phase") == FULL_VALIDATION_PHASE


def is_final_test(values: Mapping[str, Any]) -> bool:
    return values.get("phase") == FINAL_TEST_PHASE


def is_fixed_run_profile(values: Mapping[str, Any]) -> bool:
    return is_full_validation(values) or is_final_test(values)


def configured_variant_order(values: Mapping[str, Any]) -> tuple[str, ...]:
    if is_final_test(values):
        return FINAL_TEST_VARIANT_ORDER
    if is_full_validation(values):
        return FULL_VALIDATION_VARIANT_ORDER
    return EXPECTED_VARIANT_ORDER


def configured_split_counts(values: Mapping[str, Any]) -> dict[str, int]:
    if is_final_test(values):
        return {"val": 0, "train_sanity": 0, "test": 11}
    if is_full_validation(values):
        return {"val": 4, "train_sanity": 0, "test": 0}
    return {"val": 6, "train_sanity": 2, "test": 0}


def configured_variant_success_token(values: Mapping[str, Any], variant_id: str) -> str:
    if is_final_test(values):
        tokens = FINAL_TEST_VARIANT_SUCCESS_TOKENS
    elif is_full_validation(values):
        tokens = FULL_VALIDATION_VARIANT_SUCCESS_TOKENS
    else:
        tokens = VARIANT_SUCCESS_TOKENS
    return tokens[variant_id]


def configured_final_success_token(values: Mapping[str, Any]) -> str:
    if is_final_test(values):
        return FINAL_TEST_FINAL_SUCCESS_TOKEN
    if is_full_validation(values):
        return FULL_VALIDATION_FINAL_SUCCESS_TOKEN
    return FINAL_SUCCESS_TOKEN


def configured_test_data_used(values: Mapping[str, Any]) -> bool:
    return is_final_test(values)


def configured_selection_provenance(values: Mapping[str, Any]) -> dict[str, bool]:
    if is_final_test(values):
        return {"test_data_used_for_selection": False}
    return {}


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file does not exist: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return payload


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def require_nonzero_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is zero-size: {resolved}")
    return resolved


def _require_exact(config: Mapping[str, Any], key: str, expected: Any) -> None:
    if config.get(key) != expected:
        raise ValueError(f"Config {key} must be {expected!r}, got {config.get(key)!r}")


def validate_output_root(
    output_root: Path,
    project_root: Path,
    expected_relative: Path = EXPECTED_OUTPUT_RELATIVE,
) -> Path:
    expected = (project_root / expected_relative).resolve()
    resolved = output_root.expanduser().resolve()
    if resolved != expected:
        raise ValueError(f"Output root must be exactly {expected}, got {resolved}")
    if not is_relative_to(resolved, project_root / "outputs" / "phase2n"):
        raise ValueError(f"Output root is outside outputs/phase2n: {resolved}")
    for forbidden in (project_root / "data", project_root / "checkpoints"):
        if is_relative_to(resolved, forbidden):
            raise ValueError(f"Unsafe output root: {resolved}")
    return resolved


def validate_free_disk(output_root: Path, minimum_gib: float) -> dict[str, Any]:
    probe = output_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gib = usage.free / (1024**3)
    if free_gib < minimum_gib:
        raise RuntimeError(f"Only {free_gib:.2f} GiB free at {probe}; {minimum_gib:.2f} GiB required")
    return {"checked_path": str(probe.resolve()), "free_bytes": usage.free, "free_gib": free_gib}


def validate_training_run(training_run_dir: Path) -> dict[str, Any]:
    required = {
        "root_summary": training_run_dir / "summary.json",
        "runtime_manifest": training_run_dir / "00_RUNTIME_MANIFEST.json",
        "root_success": training_run_dir / "_SUCCESS",
        "pc_s1_summary": training_run_dir / "pc_s1" / "summary.json",
        "pc_s1_success": training_run_dir / "pc_s1" / "_SUCCESS",
        "pc_full_summary": training_run_dir / "pc_full" / "summary.json",
        "pc_full_success": training_run_dir / "pc_full" / "_SUCCESS",
    }
    for label, path in required.items():
        require_nonzero_file(path, f"training {label}")

    root_summary = read_json_object(required["root_summary"])
    runtime_manifest = read_json_object(required["runtime_manifest"])
    if root_summary.get("status") != "OK":
        raise ValueError("Week 2 training root summary status is not OK")
    if root_summary.get("scope_order") != ["pc_s1", "pc_full"]:
        raise ValueError("Week 2 training root scope order is invalid")
    if root_summary.get("test_data_used") is not False:
        raise ValueError("Week 2 training summary does not prove test exclusion")
    for scope in ("pc_s1", "pc_full"):
        summary = read_json_object(required[f"{scope}_summary"])
        if summary.get("status") != "OK" or summary.get("scope") != scope:
            raise ValueError(f"Week 2 {scope} summary is not complete")
        if summary.get("test_data_used") is not False:
            raise ValueError(f"Week 2 {scope} summary does not prove test exclusion")
    for marker, token in (
        (required["root_success"], TRAINING_SUCCESS_TOKENS["root"]),
        (required["pc_s1_success"], TRAINING_SUCCESS_TOKENS["pc_s1"]),
        (required["pc_full_success"], TRAINING_SUCCESS_TOKENS["pc_full"]),
    ):
        if token not in marker.read_text(encoding="utf-8"):
            raise ValueError(f"Training success marker is invalid: {marker}")
    return {
        "status": "OK",
        "training_run_dir": str(training_run_dir),
        "root_summary": str(required["root_summary"]),
        "root_success": str(required["root_success"]),
        "pc_s1_success": str(required["pc_s1_success"]),
        "pc_full_success": str(required["pc_full_success"]),
        "runtime_manifest_status": runtime_manifest.get("status"),
        "runtime_manifest_status_is_completion_gate": False,
        "completion_basis": "root and scope summaries status OK plus all _SUCCESS markers",
    }


def validate_frozen_cases(eval_cases_config: Path, project_root: Path) -> tuple[dict[str, Any], ...]:
    payload = read_json_object(eval_cases_config)
    if payload.get("selection_frozen_before_training") is not True:
        raise ValueError("Evaluation case selection was not frozen before training")
    if payload.get("selected_input_view") != "005" or payload.get("reference_lighting") != "AL":
        raise ValueError("Frozen evaluation protocol must use selected view 005 and AL")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 8:
        raise ValueError("Frozen evaluation manifest must contain exactly eight cases")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    split_counts = {"val": 0, "train_sanity": 0, "test": 0}
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"Frozen case {index} is not an object")
        asset_id = raw.get("asset_id")
        split = raw.get("eval_split")
        source_split = raw.get("source_split")
        if not isinstance(asset_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", asset_id):
            raise ValueError(f"Frozen case {index} has an invalid asset ID")
        if asset_id in seen:
            raise ValueError(f"Frozen cases contain duplicate asset {asset_id}")
        seen.add(asset_id)
        if split not in {"val", "train_sanity"}:
            raise ValueError(f"Test or unsupported split appears in frozen cases: {split!r}")
        if source_split == "test" or split == "test":
            raise ValueError(f"Test case is forbidden: {asset_id}")
        if raw.get("selected_input_view") != "005" or raw.get("reference_lighting") != "AL":
            raise ValueError(f"Frozen case {asset_id} does not use view 005 / AL")
        mesh = require_nonzero_file(resolve_project_path(raw.get("mesh_path", ""), project_root), f"{asset_id} mesh")
        reference = require_nonzero_file(
            resolve_project_path(raw.get("reference_image_path", ""), project_root),
            f"{asset_id} AL reference",
        )
        if reference.name != "005_light_AL.png":
            raise ValueError(f"Frozen case {asset_id} reference is not 005_light_AL.png")
        split_counts[split] += 1
        cases.append(
            {
                "asset_id": asset_id,
                "eval_split": split,
                "source_split": source_split,
                "selection_stratum": raw.get("selection_stratum"),
                "selection_rationale": raw.get("selection_rationale"),
                "mesh_path": str(mesh),
                "input_image_path": str(reference),
                "reference_image_path": str(reference),
                "selected_input_view": "005",
                "reference_lighting": "AL",
            }
        )
    if split_counts != {"val": 6, "train_sanity": 2, "test": 0}:
        raise ValueError(f"Frozen split counts must be 6 val / 2 train-sanity / 0 test, got {split_counts}")
    if payload.get("case_count") != 8 or payload.get("split_counts") != split_counts:
        raise ValueError("Frozen manifest summary counts differ from its cases")
    return tuple(cases)


def load_full_validation_builder(project_root: Path) -> Any:
    import importlib.util

    source = project_root / "scripts" / "phase2n_build_full_validation_manifest.py"
    spec = importlib.util.spec_from_file_location("_phase2n_full_validation_manifest", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load full-validation manifest builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_full_validation_cases(
    manifest_config: Path,
    project_root: Path,
) -> tuple[dict[str, Any], ...]:
    builder = load_full_validation_builder(project_root)
    resolved = builder.build_manifest(
        manifest_config,
        project_root,
        require_new_outputs=False,
        validate_files=True,
    )
    remaining = set(resolved["remaining_validation_ids"])
    cases = []
    for row in resolved["cases"]:
        if row["asset_id"] not in remaining:
            continue
        if row["source_split"] != "val" or row["eval_split"] != "val":
            raise ValueError(f"full validation case is not val-only: {row['asset_id']}")
        cases.append(
            {
                "asset_id": row["asset_id"],
                "eval_split": "val",
                "source_split": "val",
                "selection_stratum": "full_validation_remaining",
                "selection_rationale": "Canonical full101 validation asset absent from the six-case pilot.",
                "mesh_path": row["mesh_path"],
                "input_image_path": row["reference_paths"]["005"],
                "reference_image_path": row["reference_paths"]["005"],
                "selected_input_view": "005",
                "reference_lighting": "AL",
            }
        )
    if [case["asset_id"] for case in cases] != resolved["remaining_validation_ids"]:
        raise ValueError("full-validation inference cases do not match the derived remaining four")
    if len(cases) != 4:
        raise ValueError("full-validation inference must contain exactly four cases")
    return tuple(cases)


def load_final_test_builder(project_root: Path) -> Any:
    import importlib.util

    source = project_root / "scripts" / "phase2n_build_final_test_manifest.py"
    spec = importlib.util.spec_from_file_location("_phase2n_final_test_manifest", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load final-test manifest builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_final_test_cases(
    manifest_config: Path,
    project_root: Path,
) -> tuple[dict[str, Any], ...]:
    builder = load_final_test_builder(project_root)
    resolved = builder.build_manifest(
        manifest_config,
        project_root,
        require_new_outputs=False,
        validate_files=True,
    )
    if resolved.get("split_counts") != {"val": 0, "train_sanity": 0, "test": 11}:
        raise ValueError("final-test manifest is not test-only 0/0/11")
    if resolved.get("candidate_variants") != ["pc_full_step320"]:
        raise ValueError("final-test manifest does not freeze PC-Full step 320 alone")
    cases = []
    for row in resolved["cases"]:
        if row["source_split"] != "test" or row["eval_split"] != "test":
            raise ValueError(f"final-test case is not test-only: {row['asset_id']}")
        cases.append(
            {
                "asset_id": row["asset_id"],
                "eval_split": "test",
                "source_split": "test",
                "selection_stratum": "phase2n_final_test",
                "selection_rationale": (
                    "Canonical full101 test split; held out from Phase 2N candidate "
                    "and checkpoint selection."
                ),
                "mesh_path": row["mesh_path"],
                "input_image_path": row["reference_paths"]["005"],
                "reference_image_path": row["reference_paths"]["005"],
                "selected_input_view": "005",
                "reference_lighting": "AL",
            }
        )
    if [case["asset_id"] for case in cases] != resolved["test_ids"] or len(cases) != 11:
        raise ValueError("final-test inference cases do not match the canonical 11 assets")
    return tuple(cases)


def _validate_historical_case(case: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    asset_id = frozen["asset_id"]
    if case.get("eval_split") != frozen["eval_split"] or case.get("source_split") != frozen["source_split"]:
        raise ValueError(f"Historical split mismatch for {asset_id}")
    if case.get("selected_input_view") != "005":
        raise ValueError(f"Historical selected input view is not 005 for {asset_id}")
    expected_paths = {
        "local_mesh_path": frozen["mesh_path"],
        "case_input_mesh": frozen["mesh_path"],
        "selected_input_image": frozen["input_image_path"],
        "case_input_image": frozen["input_image_path"],
    }
    for key, expected in expected_paths.items():
        value = case.get(key)
        if not isinstance(value, str) or Path(value).expanduser().resolve() != Path(expected).resolve():
            raise ValueError(f"Historical {key} mismatch for {asset_id}")


def _validate_baseline_variant(
    label: str,
    config: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    root = resolve_project_path(config.get("root", ""), project_root)
    if not root.is_dir():
        raise FileNotFoundError(f"{label} baseline root is missing: {root}")
    mode = config.get("mode")
    expected_mode = "base" if label == "corrected_input_base" else "finetuned"
    if mode != expected_mode:
        raise ValueError(f"{label} mode must be {expected_mode}")
    filename = config.get("output_filename")
    if not isinstance(filename, str) or not filename.endswith(".glb"):
        raise ValueError(f"{label} output filename must be a GLB")
    checkpoint = None
    if expected_mode == "finetuned":
        checkpoint = require_nonzero_file(
            resolve_project_path(config.get("checkpoint_path", ""), project_root),
            "historical full80 checkpoint",
        )

    records: list[dict[str, Any]] = []
    for case in cases:
        case_dir = root / str(case["eval_split"]) / str(case["asset_id"])
        output = require_nonzero_file(case_dir / filename, f"{label} output for {case['asset_id']}")
        run_plan_path = require_nonzero_file(case_dir / "run_plan.json", f"{label} run plan for {case['asset_id']}")
        run_plan = read_json_object(run_plan_path)
        expected_plan = {
            "mode": expected_mode,
            "max_num_view": 6,
            "resolution": 512,
            "use_remesh": False,
            "dry_run": False,
        }
        for key, expected in expected_plan.items():
            if run_plan.get(key) != expected:
                raise ValueError(f"{label} run plan {key} mismatch for {case['asset_id']}")
        for key, expected in (("input_mesh", case["mesh_path"]), ("input_image", case["input_image_path"])):
            value = run_plan.get(key)
            if not isinstance(value, str) or Path(value).expanduser().resolve() != Path(expected).resolve():
                raise ValueError(f"{label} run plan {key} mismatch for {case['asset_id']}")
        planned_glb = run_plan.get("planned_output_glb")
        if not isinstance(planned_glb, str) or Path(planned_glb).expanduser().resolve() != output:
            raise ValueError(f"{label} planned GLB mismatch for {case['asset_id']}")
        if expected_mode == "base" and run_plan.get("checkpoint") not in {"", None}:
            raise ValueError(f"Corrected-input base unexpectedly records a checkpoint for {case['asset_id']}")
        if checkpoint is not None:
            recorded = run_plan.get("checkpoint")
            if not isinstance(recorded, str) or Path(recorded).expanduser().resolve() != checkpoint:
                raise ValueError(f"Historical full80 checkpoint mismatch for {case['asset_id']}")
        records.append(
            {
                "asset_id": case["asset_id"],
                "eval_split": case["eval_split"],
                "output_glb_path": str(output),
                "output_glb_byte_size": output.stat().st_size,
                "run_plan_path": str(run_plan_path),
            }
        )
    return {
        "status": "OK",
        "label": label,
        "root": str(root),
        "mode": expected_mode,
        "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "covered_case_count": len(records),
        "expected_case_count": len(cases),
        "all_cases_compatible": len(records) == len(cases),
        "cases": records,
    }


def validate_baseline_reuse(
    baseline_config: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    if not isinstance(baseline_config, Mapping):
        raise ValueError("baseline_reuse must be an object")
    historical_path = require_nonzero_file(
        resolve_project_path(baseline_config.get("historical_eval_cases", ""), project_root),
        "historical full80 eval cases",
    )
    historical = read_json_object(historical_path)
    raw_cases = historical.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Historical full80 eval cases are invalid")
    by_id = {case.get("item_id"): case for case in raw_cases if isinstance(case, dict)}
    for frozen in cases:
        old = by_id.get(frozen["asset_id"])
        if not isinstance(old, dict):
            raise ValueError(f"Frozen case is missing from historical eval cases: {frozen['asset_id']}")
        _validate_historical_case(old, frozen)

    base_config = baseline_config.get("corrected_input_base")
    full_config = baseline_config.get("historical_full80_500")
    if not isinstance(base_config, Mapping) or not isinstance(full_config, Mapping):
        raise ValueError("baseline_reuse must define corrected_input_base and historical_full80_500")
    base = _validate_baseline_variant("corrected_input_base", base_config, cases, project_root)
    full = _validate_baseline_variant("historical_full80_500", full_config, cases, project_root)
    return {
        "status": "OK",
        "historical_eval_cases": str(historical_path),
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "fixed_mesh": True,
        "use_remesh": False,
        "case_count": len(cases),
        "corrected_input_base": base,
        "historical_full80_500": full,
        "complete_compatible_coverage": base["covered_case_count"] == full["covered_case_count"] == len(cases),
    }


def _compact_artifact(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "checkpoint_path",
        "manifest_path",
        "format",
        "scope",
        "step",
        "byte_size",
        "sha256",
        "trainable_parameter_tensor_count",
        "trainable_parameter_numel",
    )
    return {key: report[key] for key in keys}


def validate_variant_specs(
    config: Mapping[str, Any],
    project_root: Path,
    training_run_dir: Path,
    *,
    validate_hashes: bool,
    expected_order: Sequence[str] = EXPECTED_VARIANT_ORDER,
) -> dict[str, dict[str, Any]]:
    variants = config.get("variants")
    expected_order = tuple(expected_order)
    if not isinstance(variants, dict) or list(variants) != list(expected_order):
        raise ValueError(f"Config variants must be exactly {list(expected_order)}")
    resolved: dict[str, dict[str, Any]] = {}
    pending_hashes: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for variant_id in expected_order:
        raw = variants[variant_id]
        if not isinstance(raw, dict):
            raise ValueError(f"Variant {variant_id} must be an object")
        expected_keys = {
            "scope",
            "step",
            "checkpoint_path",
            "checkpoint_manifest_path",
            "expected_trainable_tensor_count",
            "expected_trainable_numel",
            "expected_checkpoint_sha256",
            "expected_checkpoint_byte_size",
        }
        if set(raw) != expected_keys:
            raise ValueError(f"Variant {variant_id} fields differ from the frozen schema")
        scope, step, tensor_count, numel = EXPECTED_VARIANTS[variant_id]
        expectations = {
            "scope": scope,
            "step": step,
            "expected_trainable_tensor_count": tensor_count,
            "expected_trainable_numel": numel,
        }
        for key, expected in expectations.items():
            if raw.get(key) != expected:
                raise ValueError(f"Variant {variant_id} {key} must be {expected!r}")
        checkpoint_path = resolve_project_path(raw["checkpoint_path"], project_root)
        manifest_path = resolve_project_path(raw["checkpoint_manifest_path"], project_root)
        expected_checkpoint = training_run_dir / scope / "checkpoints" / f"step_{step}_scope_state.pt"
        expected_manifest = training_run_dir / scope / "checkpoints" / f"step_{step}_manifest.json"
        if checkpoint_path != expected_checkpoint or manifest_path != expected_manifest:
            raise ValueError(f"Variant {variant_id} does not point to its audited training artifact")
        kwargs = {
            "expected_scope": scope,
            "expected_step": step,
            "expected_sha256": raw["expected_checkpoint_sha256"],
            "expected_byte_size": raw["expected_checkpoint_byte_size"],
            "expected_tensor_count": tensor_count,
            "expected_numel": numel,
        }
        if validate_hashes:
            artifact = None
            pending_hashes[variant_id] = (checkpoint_path, manifest_path, kwargs)
        else:
            require_nonzero_file(checkpoint_path, f"{variant_id} checkpoint")
            require_nonzero_file(manifest_path, f"{variant_id} checkpoint manifest")
            artifact = {
                "status": "NOT_HASHED_IN_TEST_MODE",
                "checkpoint_path": str(checkpoint_path),
                "manifest_path": str(manifest_path),
            }
        resolved[variant_id] = {
            "variant_id": variant_id,
            **raw,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_manifest_path": str(manifest_path),
            "artifact_validation": artifact,
        }

    if validate_hashes:
        from concurrent.futures import ThreadPoolExecutor

        # Full byte-wise hashes remain mandatory. Concurrent reads keep the
        # four independent audits practical on cold shared storage.
        with ThreadPoolExecutor(max_workers=len(expected_order)) as executor:
            futures = {
                variant_id: executor.submit(
                    validate_checkpoint_artifact,
                    checkpoint_path,
                    manifest_path,
                    **kwargs,
                )
                for variant_id, (checkpoint_path, manifest_path, kwargs) in pending_hashes.items()
            }
            for variant_id in expected_order:
                resolved[variant_id]["artifact_validation"] = _compact_artifact(
                    futures[variant_id].result()
                )
    return resolved


def validate_config(
    config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    *,
    validate_hashes: bool = True,
    check_free_space: bool = True,
    require_repository_paths: bool = True,
) -> ValidatedInferenceConfig:
    root = Path(project_root).expanduser().resolve()
    path = resolve_project_path(config_path, root)
    config = read_json_object(path)
    phase = config.get("phase")
    full_validation = phase == FULL_VALIDATION_PHASE
    final_test = phase == FINAL_TEST_PHASE
    fixed_profile = full_validation or final_test
    expected_keys = FULL_VALIDATION_CONFIG_KEYS if fixed_profile else EXPECTED_CONFIG_KEYS
    if set(config) != expected_keys:
        raise ValueError("Inference config fields differ from the frozen schema")
    if phase not in {PILOT_PHASE, FULL_VALIDATION_PHASE, FINAL_TEST_PHASE}:
        raise ValueError(f"Unsupported inference phase: {phase!r}")
    variant_order = configured_variant_order(config)
    expected_output = (
        FINAL_TEST_OUTPUT_RELATIVE
        if final_test
        else FULL_VALIDATION_OUTPUT_RELATIVE
        if full_validation
        else EXPECTED_OUTPUT_RELATIVE
    )
    expected_eval = (
        FINAL_TEST_EVAL_CONFIG_RELATIVE
        if final_test
        else FULL_VALIDATION_EVAL_CONFIG_RELATIVE
        if full_validation
        else EXPECTED_EVAL_CONFIG_RELATIVE
    )
    _require_exact(config, "inference_seed", 0)
    _require_exact(config, "selected_input_view", "005")
    _require_exact(config, "reference_lighting", "AL")
    _require_exact(config, "max_num_view", 6)
    _require_exact(config, "resolution", 512)
    _require_exact(config, "fixed_mesh", True)
    _require_exact(config, "use_remesh", False)
    _require_exact(config, "save_glb", True)
    _require_exact(config, "fresh_official_true_pbr_base_per_variant", True)
    _require_exact(config, "variant_order", list(variant_order))
    _require_exact(config, "minimum_gpu_memory_gib", 70)
    _require_exact(config, "minimum_free_disk_gib", 20)
    minimum_disk = 20.0

    training_run_dir = resolve_project_path(config["training_run_dir"], root)
    eval_cases_config = resolve_project_path(config["eval_cases_config"], root)
    output_root = validate_output_root(
        resolve_project_path(config["output_root"], root), root, expected_output
    )
    if require_repository_paths:
        if training_run_dir != (root / EXPECTED_TRAINING_RUN_RELATIVE).resolve():
            raise ValueError("Training run path is not the audited slurm_264123 directory")
        if eval_cases_config != (root / expected_eval).resolve():
            raise ValueError("Evaluation manifest path is not the expected frozen config")
    if fixed_profile:
        expected_run_id = (
            "phase2n_final_test_infer_v1"
            if final_test
            else "phase2n_full_validation_infer_v1"
        )
        _require_exact(config, "fixed_run_id", expected_run_id)
        _require_exact(config, "fail_if_run_exists", True)
        fixed_run = output_root / str(config["fixed_run_id"])
        if final_test and fixed_run.exists():
            raise ValueError(
                f"fixed inference run already exists; refusing overwrite: {fixed_run}"
            )
    training_report = validate_training_run(training_run_dir)
    cases = (
        validate_final_test_cases(eval_cases_config, root)
        if final_test
        else validate_full_validation_cases(eval_cases_config, root)
        if full_validation
        else validate_frozen_cases(eval_cases_config, root)
    )
    variants = validate_variant_specs(
        config,
        root,
        training_run_dir,
        validate_hashes=validate_hashes,
        expected_order=variant_order,
    )
    baseline_report = validate_baseline_reuse(config["baseline_reuse"], cases, root)
    if check_free_space:
        validate_free_disk(output_root, minimum_disk)
    return ValidatedInferenceConfig(
        values=dict(config),
        config_path=path,
        project_root=root,
        training_run_dir=training_run_dir,
        eval_cases_config=eval_cases_config,
        output_root=output_root,
        variants=variants,
        cases=cases,
        training_report=training_report,
        baseline_reuse_report=baseline_report,
    )


def run_check_only(config_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> ValidatedInferenceConfig:
    validated = validate_config(config_path, project_root)
    variant_order = configured_variant_order(validated.values)
    split_counts = configured_split_counts(validated.values)
    for variant_id in variant_order:
        artifact = validated.variants[variant_id]["artifact_validation"]
        variant = validated.variants[variant_id]
        print(
            f"checkpoint={variant_id} scope={artifact.get('scope', variant['scope'])} "
            f"step={artifact.get('step', variant['step'])} "
            f"bytes={artifact.get('byte_size', variant['expected_checkpoint_byte_size'])} "
            f"sha256={artifact.get('sha256', variant['expected_checkpoint_sha256'])} status=OK"
        )
    if is_final_test(validated.values):
        print(
            "final_test_cases=11 val=0 train_sanity=0 test=11 "
            "selected_input_view=005 reference_lighting=AL"
        )
    elif is_full_validation(validated.values):
        print(
            "remaining_cases=4 val=4 train_sanity=0 test=0 "
            "selected_input_view=005 reference_lighting=AL"
        )
    else:
        print("PHASE2N_WEEK2_SCOPE_CHECKPOINTS_OK")
        print("frozen_cases=8 val=6 train_sanity=2 test=0 selected_input_view=005 reference_lighting=AL")
        print("PHASE2N_WEEK2_FROZEN_CASES_OK")
    base = validated.baseline_reuse_report["corrected_input_base"]
    full = validated.baseline_reuse_report["historical_full80_500"]
    case_count = len(validated.cases)
    print(f"corrected_input_base_root={base['root']} coverage={base['covered_case_count']}/{case_count}")
    print(f"historical_full80_root={full['root']} coverage={full['covered_case_count']}/{case_count}")
    if is_final_test(validated.values):
        print(
            f"new_outputs={len(variant_order) * case_count} split_counts={split_counts} "
            f"fixed_run_id={validated.values['fixed_run_id']}"
        )
        print("PHASE2N_FINAL_TEST_INFERENCE_READINESS_OK")
    elif is_full_validation(validated.values):
        print(
            f"new_outputs={len(variant_order) * case_count} split_counts={split_counts} "
            f"fixed_run_id={validated.values['fixed_run_id']}"
        )
        print("PHASE2N_FULL_VALIDATION_INFERENCE_READINESS_OK")
    else:
        print("PHASE2N_WEEK2_BASELINE_REUSE_OK")
        print("PHASE2N_WEEK2_PILOT_INFERENCE_READINESS_OK")
    return validated


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _assert_output_path(path: Path, run_dir: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not is_relative_to(resolved, run_dir):
        raise ValueError(f"Runtime output escapes its run directory: {resolved}")
    return resolved


def write_once_text(path: Path, text: str, run_dir: Path) -> Path:
    destination = _assert_output_path(path, run_dir)
    if destination.exists():
        if not destination.is_file() or destination.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Existing runtime artifact differs; refusing overwrite: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def write_once_json(path: Path, payload: Any, run_dir: Path) -> Path:
    return write_once_text(path, _json_text(payload), run_dir)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"GLB JSON contains non-finite numeric constant: {value}")


def _require_finite_json(value: Any, location: str = "root") -> int:
    checked = 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"GLB JSON contains NaN/Inf at {location}")
        return 1
    if isinstance(value, dict):
        for key, child in value.items():
            checked += _require_finite_json(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            checked += _require_finite_json(child, f"{location}[{index}]")
    return checked


def validate_glb(path: str | Path) -> dict[str, Any]:
    """Validate GLB framing plus finite JSON and float accessor values."""

    source = require_nonzero_file(Path(path), "inference output GLB")
    data = source.read_bytes()
    if len(data) < 20:
        raise ValueError(f"GLB is too small: {source}")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError(f"Invalid GLB header: {source}")

    offset = 12
    json_bytes: bytes | None = None
    binary_chunks: list[bytes] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError(f"Truncated GLB chunk header: {source}")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(data):
            raise ValueError(f"Truncated GLB chunk payload: {source}")
        chunk = data[offset:end]
        offset = end
        if chunk_type == 0x4E4F534A:
            if json_bytes is not None:
                raise ValueError(f"GLB contains duplicate JSON chunks: {source}")
            json_bytes = chunk
        elif chunk_type == 0x004E4942:
            binary_chunks.append(chunk)
    if offset != len(data) or json_bytes is None:
        raise ValueError(f"GLB chunk layout is invalid: {source}")

    document = json.loads(
        json_bytes.rstrip(b" \t\r\n\x00").decode("utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
        raise ValueError(f"GLB JSON is not glTF 2.0: {source}")
    json_float_count = _require_finite_json(document)

    float_accessor_value_count = 0
    accessors = document.get("accessors", [])
    buffer_views = document.get("bufferViews", [])
    type_width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
    if isinstance(accessors, list) and isinstance(buffer_views, list) and binary_chunks:
        binary = binary_chunks[0]
        for index, accessor in enumerate(accessors):
            if not isinstance(accessor, dict) or accessor.get("componentType") != 5126:
                continue
            if "sparse" in accessor:
                raise ValueError(f"Sparse float accessor is unsupported by the strict GLB check: {index}")
            view_index = accessor.get("bufferView")
            count = accessor.get("count")
            width = type_width.get(accessor.get("type"))
            if not isinstance(view_index, int) or not isinstance(count, int) or count < 0 or width is None:
                raise ValueError(f"Invalid float accessor metadata at index {index}")
            try:
                view = buffer_views[view_index]
            except (IndexError, TypeError):
                raise ValueError(f"Float accessor references an invalid bufferView: {index}") from None
            if not isinstance(view, dict) or int(view.get("buffer", 0)) != 0:
                raise ValueError(f"Float accessor does not reference GLB buffer zero: {index}")
            element_size = 4 * width
            stride = int(view.get("byteStride", element_size))
            start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
            for element in range(count):
                element_start = start + element * stride
                element_end = element_start + element_size
                if element_start < 0 or element_end > len(binary):
                    raise ValueError(f"Float accessor exceeds the GLB binary chunk: {index}")
                values = struct.unpack_from(f"<{width}f", binary, element_start)
                if any(not math.isfinite(value) for value in values):
                    raise ValueError(f"GLB float accessor contains NaN/Inf: accessor={index} element={element}")
                float_accessor_value_count += width
    return {
        "status": "OK",
        "path": str(source),
        "byte_size": len(data),
        "sha256": sha256_file(source),
        "glb_version": version,
        "json_float_count_checked": json_float_count,
        "float_accessor_value_count_checked": float_accessor_value_count,
        "nonfinite_value_count": 0,
    }


def _safe_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError(f"Unsafe run ID: {run_id!r}")
    return run_id


def _git_head(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_runtime_device(torch_module: Any, minimum_gpu_memory_gib: float) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("Phase 2N pilot inference requires CUDA")
    device_index = int(torch_module.cuda.current_device())
    properties = torch_module.cuda.get_device_properties(device_index)
    total_bytes = int(properties.total_memory)
    total_gib = total_bytes / (1024**3)
    if "A100" not in str(properties.name):
        raise RuntimeError(f"Phase 2N pilot inference requires an A100, got {properties.name}")
    if total_gib < minimum_gpu_memory_gib:
        raise RuntimeError(f"GPU has {total_gib:.2f} GiB; {minimum_gpu_memory_gib:.2f} GiB required")
    return {
        "device_index": device_index,
        "device_name": str(properties.name),
        "total_memory_bytes": total_bytes,
        "total_memory_gib": total_gib,
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
    }


def reset_inference_seed(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - official runtime dependency.
        raise RuntimeError("NumPy is required by official inference") from exc
    np.random.seed(seed % (2**32))
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def resolved_case_provenance(validated: ValidatedInferenceConfig) -> dict[str, Any]:
    records = []
    for case in validated.cases:
        records.append(
            {
                **case,
                "mesh_sha256": sha256_file(case["mesh_path"]),
                "input_image_sha256": sha256_file(case["input_image_path"]),
                "reference_image_sha256": sha256_file(case["reference_image_path"]),
            }
        )
    return {
        "phase": validated.values["phase"],
        "source_manifest_path": str(validated.eval_cases_config),
        "source_manifest_sha256": sha256_file(validated.eval_cases_config),
        "selection_frozen_before_training": True,
        "case_count": len(records),
        "split_counts": configured_split_counts(validated.values),
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "test_data_used": configured_test_data_used(validated.values),
        **configured_selection_provenance(validated.values),
        "cases": records,
    }


def ensure_runtime_provenance(
    run_dir: Path,
    validated: ValidatedInferenceConfig,
    runtime_device: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_cases = resolved_case_provenance(validated)
    manifest = {
        "phase": validated.values["phase"],
        "status": "RUNNING",
        "run_id": run_dir.name,
        "config_path": str(validated.config_path),
        "config_sha256": sha256_file(validated.config_path),
        "frozen_cases_path": str(validated.eval_cases_config),
        "frozen_cases_sha256": resolved_cases["source_manifest_sha256"],
        "training_run_dir": str(validated.training_run_dir),
        "training_summary_sha256": sha256_file(validated.training_run_dir / "summary.json"),
        "variant_order": list(configured_variant_order(validated.values)),
        "fresh_official_true_pbr_base_per_variant": True,
        "fixed_mesh": True,
        "use_remesh": False,
        "inference_seed": 0,
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "test_data_used": configured_test_data_used(validated.values),
        **configured_selection_provenance(validated.values),
        "runtime_device": dict(runtime_device),
        "project_git_head": _git_head(validated.project_root),
    }
    write_once_json(run_dir / "00_RUNTIME_MANIFEST.json", manifest, run_dir)
    write_once_json(run_dir / "resolved_cases.json", resolved_cases, run_dir)
    write_once_json(run_dir / "baseline_reuse_report.json", validated.baseline_reuse_report, run_dir)
    return resolved_cases


def _target_unet(pipeline: Any) -> Any:
    try:
        target = pipeline.models["multiview_model"].pipeline.unet
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError("Official pipeline no longer exposes the audited multiview UNet target") from exc
    return target


def initialize_variant_pipeline(
    variant: Mapping[str, Any],
    *,
    max_num_view: int,
    resolution: int,
    device: str,
    pipeline_factory: Callable[..., tuple[Any, dict[str, Any]]] = initialize_base_paint_pipeline,
    checkpoint_loader: Callable[..., dict[str, Any]] = load_scope_checkpoint_into_model,
    torch_module: Any | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Create one fresh base and apply exactly one scope checkpoint."""

    pipeline, initialization = pipeline_factory(
        max_num_view=max_num_view,
        resolution=resolution,
        device=device,
    )
    target = _target_unet(pipeline)
    load_report = checkpoint_loader(
        target,
        variant["checkpoint_path"],
        variant["checkpoint_manifest_path"],
        expected_scope=variant["scope"],
        expected_step=int(variant["step"]),
        expected_sha256=variant["expected_checkpoint_sha256"],
        expected_byte_size=int(variant["expected_checkpoint_byte_size"]),
        expected_tensor_count=int(variant["expected_trainable_tensor_count"]),
        expected_numel=int(variant["expected_trainable_numel"]),
        torch_module=torch_module,
    )
    target.eval()
    return pipeline, initialization, load_report


def _case_manifest_expected(
    case: Mapping[str, Any],
    variant: Mapping[str, Any],
    output_glb: Path,
    validated: ValidatedInferenceConfig,
) -> dict[str, Any]:
    return {
        "phase": validated.values["phase"],
        "asset_id": case["asset_id"],
        "eval_split": case["eval_split"],
        "source_split": case["source_split"],
        "mesh_path": case["mesh_path"],
        "mesh_sha256": sha256_file(case["mesh_path"]),
        "input_image_path": case["input_image_path"],
        "input_image_sha256": sha256_file(case["input_image_path"]),
        "reference_image_path": case["reference_image_path"],
        "reference_image_sha256": sha256_file(case["reference_image_path"]),
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "inference_seed": 0,
        "variant": variant["variant_id"],
        "scope": variant["scope"],
        "checkpoint_step": variant["step"],
        "checkpoint_sha256": variant["expected_checkpoint_sha256"],
        "output_glb_path": str(output_glb),
        "exact_inference_settings": {
            "max_num_view": 6,
            "resolution": 512,
            "device": "cuda",
            "fixed_mesh": True,
            "use_remesh": False,
            "save_glb": True,
            "selected_input_view": "005",
            "reference_lighting": "AL",
            "inference_seed": 0,
        },
        "test_data_used": configured_test_data_used(validated.values),
        **configured_selection_provenance(validated.values),
    }


def validate_completed_case(
    case_dir: Path,
    case: Mapping[str, Any],
    variant: Mapping[str, Any],
    validated: ValidatedInferenceConfig,
) -> dict[str, Any]:
    output_glb = require_nonzero_file(case_dir / "textured_mesh.glb", f"{variant['variant_id']} output")
    manifest_path = require_nonzero_file(case_dir / "inference_manifest.json", "case inference manifest")
    if is_final_test(validated.values):
        case_success = require_nonzero_file(case_dir / "_SUCCESS", "case success marker")
        if case_success.read_text(encoding="utf-8") != FINAL_TEST_CASE_SUCCESS_TOKEN + "\n":
            raise RuntimeError(f"Completed case success marker mismatch: {case_success}")
    manifest = read_json_object(manifest_path)
    expected = _case_manifest_expected(case, variant, output_glb, validated)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Completed case manifest mismatch for {variant['variant_id']}/{case['asset_id']}: {key}")
    glb_report = validate_glb(output_glb)
    if manifest.get("output_glb_size") != glb_report["byte_size"]:
        raise RuntimeError(f"Completed case GLB size mismatch: {output_glb}")
    if manifest.get("output_glb_sha256") != glb_report["sha256"]:
        raise RuntimeError(f"Completed case GLB hash mismatch: {output_glb}")
    return manifest


def _variant_summary_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary["split_counts"]
    title = (
        "Phase 2N Final Test"
        if summary.get("phase", PILOT_PHASE) == FINAL_TEST_PHASE
        else "Phase 2N Full Validation"
        if summary.get("phase", PILOT_PHASE) == FULL_VALIDATION_PHASE
        else "Phase 2N Week 2"
    )
    return "\n".join(
        [
            f"# {title} {summary['variant']} Inference",
            "",
            f"- Status: {summary['status']}",
            f"- Scope: {summary['scope']}",
            f"- Checkpoint step: {summary['checkpoint_step']}",
            f"- Cases: {summary['case_count']} "
            f"({counts['val']} val, {counts['train_sanity']} train-sanity, "
            f"{counts['test']} test)",
            "- Every case used selected input view 005, AL lighting, and fixed-mesh inference.",
            "- This variant started from a fresh official true-PBR base.",
            "",
        ]
    )


def validate_completed_variant(
    variant_dir: Path,
    variant: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    validated: ValidatedInferenceConfig,
) -> dict[str, Any]:
    required = (
        variant_dir / "summary.json",
        variant_dir / "summary.md",
        variant_dir / "_SUCCESS",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Existing variant is incomplete; use a new run ID: missing={missing}")
    expected_dirs = {str(case["asset_id"]) for case in cases}
    actual_dirs = {path.name for path in variant_dir.iterdir() if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise RuntimeError(
            f"Existing variant case directories differ: missing={sorted(expected_dirs - actual_dirs)} "
            f"unexpected={sorted(actual_dirs - expected_dirs)}"
        )
    case_manifests = [
        validate_completed_case(variant_dir / str(case["asset_id"]), case, variant, validated)
        for case in cases
    ]
    summary = read_json_object(variant_dir / "summary.json")
    split_counts = configured_split_counts(validated.values)
    expected_summary = {
        "status": "OK",
        "variant": variant["variant_id"],
        "scope": variant["scope"],
        "checkpoint_step": variant["step"],
        "checkpoint_sha256": variant["expected_checkpoint_sha256"],
        "case_count": len(cases),
        "split_counts": split_counts,
        "fresh_official_true_pbr_base": True,
        "test_data_used": configured_test_data_used(validated.values),
        **configured_selection_provenance(validated.values),
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Existing variant summary mismatch at {key}: {variant_dir}")
    if summary.get("case_manifest_paths") != [
        str((variant_dir / str(case["asset_id"]) / "inference_manifest.json").resolve())
        for case in cases
    ]:
        raise RuntimeError(f"Existing variant summary case manifest list differs: {variant_dir}")
    expected_markdown = _variant_summary_markdown(summary)
    if (variant_dir / "summary.md").read_text(encoding="utf-8") != expected_markdown:
        raise RuntimeError(f"Existing variant Markdown summary is invalid: {variant_dir}")
    token = configured_variant_success_token(validated.values, variant["variant_id"])
    if (variant_dir / "_SUCCESS").read_text(encoding="utf-8") != token + "\n":
        raise RuntimeError(f"Existing variant success marker is invalid: {variant_dir}")
    return {"summary": summary, "case_manifests": case_manifests}


def run_variant(
    variant: Mapping[str, Any],
    validated: ValidatedInferenceConfig,
    run_dir: Path,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    variant_id = str(variant["variant_id"])
    variant_dir = run_dir / variant_id
    if variant_dir.exists():
        completed = validate_completed_variant(variant_dir, variant, validated.cases, validated)
        print(f"variant={variant_id} status=SKIPPED_VALID_COMPLETE")
        print(configured_variant_success_token(validated.values, variant_id))
        return completed["summary"]

    variant_dir.mkdir(parents=True)
    pipeline = None
    try:
        pipeline, initialization, load_report = initialize_variant_pipeline(
            variant,
            max_num_view=int(validated.values["max_num_view"]),
            resolution=int(validated.values["resolution"]),
            device="cuda",
            torch_module=torch_module,
        )
        checkpoint_report = {
            "variant": variant_id,
            "artifact_validation": variant["artifact_validation"],
            "initialization": initialization,
            "scope_checkpoint_load": load_report,
            "fresh_official_true_pbr_base": True,
            "checkpoint_state_accumulation": False,
        }
        write_once_json(run_dir / "checkpoint_validation" / f"{variant_id}.json", checkpoint_report, run_dir)

        manifest_paths: list[str] = []
        for case_index, case in enumerate(validated.cases):
            if (
                not is_final_test(validated.values)
                and (case["eval_split"] == "test" or case["source_split"] == "test")
            ):
                raise RuntimeError(f"Test case reached runtime: {case['asset_id']}")
            case_dir = variant_dir / str(case["asset_id"])
            if case_dir.exists():
                raise RuntimeError(f"Incomplete case directory already exists; use a new run ID: {case_dir}")
            case_dir.mkdir()
            output_obj = case_dir / "textured_mesh.obj"
            output_glb = case_dir / "textured_mesh.glb"
            reset_inference_seed(torch_module, int(validated.values["inference_seed"]))
            result = run_fixed_mesh_inference(
                pipeline,
                mesh_path=case["mesh_path"],
                image_path=case["input_image_path"],
                output_mesh_path=output_obj,
            )
            glb_report = validate_glb(output_glb)
            manifest = _case_manifest_expected(case, variant, output_glb.resolve(), validated)
            manifest.update(
                {
                    "status": "OK",
                    "output_mesh_result": str(result),
                    "output_glb_size": glb_report["byte_size"],
                    "output_glb_sha256": glb_report["sha256"],
                    "output_glb_validation": glb_report,
                }
            )
            manifest_path = case_dir / "inference_manifest.json"
            write_once_json(manifest_path, manifest, run_dir)
            if is_final_test(validated.values):
                write_once_text(
                    case_dir / "_SUCCESS",
                    FINAL_TEST_CASE_SUCCESS_TOKEN + "\n",
                    run_dir,
                )
            manifest_paths.append(str(manifest_path.resolve()))
            print(f"variant={variant_id} case={case['asset_id']} split={case['eval_split']} status=OK")
            if case_index == 0:
                print(f"variant={variant_id} first_case_smoke=OK")

        summary = {
            "phase": validated.values["phase"],
            "status": "OK",
            "variant": variant_id,
            "scope": variant["scope"],
            "checkpoint_step": variant["step"],
            "checkpoint_path": variant["checkpoint_path"],
            "checkpoint_sha256": variant["expected_checkpoint_sha256"],
            "case_count": len(validated.cases),
            "split_counts": configured_split_counts(validated.values),
            "case_manifest_paths": manifest_paths,
            "fresh_official_true_pbr_base": True,
            "fixed_mesh": True,
            "use_remesh": False,
            "test_data_used": configured_test_data_used(validated.values),
            **configured_selection_provenance(validated.values),
        }
        write_once_json(variant_dir / "summary.json", summary, run_dir)
        write_once_text(variant_dir / "summary.md", _variant_summary_markdown(summary), run_dir)
        token = configured_variant_success_token(validated.values, variant_id)
        write_once_text(variant_dir / "_SUCCESS", token + "\n", run_dir)
        print(token)
        return summary
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        torch_module.cuda.empty_cache()
        torch_module.cuda.synchronize()


def _final_summary_markdown(summary: Mapping[str, Any]) -> str:
    full_validation = summary["phase"] == FULL_VALIDATION_PHASE
    final_test = summary["phase"] == FINAL_TEST_PHASE
    counts = summary["split_counts_per_variant"]
    variants_line = (
        f"- Variants: {', '.join(summary['variant_order'])}."
        if full_validation or final_test
        else "- Variants: PC-S1 steps 160/320 and PC-Full steps 160/320."
    )
    return "\n".join(
        [
            (
                "# Phase 2N Final-Test Candidate Inference"
                if final_test
                else "# Phase 2N Full Validation Candidate Inference"
                if full_validation
                else "# Phase 2N Week 2 Scope-Checkpoint Pilot Inference"
            ),
            "",
            f"- Status: {summary['status']}",
            f"- Run ID: {summary['run_id']}",
            variants_line,
            f"- Cases per variant: {counts['val']} validation + "
            f"{counts['train_sanity']} train-sanity + {counts['test']} test.",
            "- Every variant began from a newly loaded identical official true-PBR base.",
            "- Existing corrected-input base and historical full80 outputs were reused, not rerun.",
            "",
        ]
    )


def finalize_run(run_dir: Path, validated: ValidatedInferenceConfig) -> dict[str, Any]:
    variant_order = configured_variant_order(validated.values)
    split_counts = configured_split_counts(validated.values)
    results = {
        variant_id: validate_completed_variant(
            run_dir / variant_id,
            validated.variants[variant_id],
            validated.cases,
            validated,
        )["summary"]
        for variant_id in variant_order
    }
    summary = {
        "phase": validated.values["phase"],
        "status": "OK",
        "run_id": run_dir.name,
        "variant_order": list(variant_order),
        "variant_summaries": results,
        "case_count_per_variant": len(validated.cases),
        "total_inference_outputs": len(variant_order) * len(validated.cases),
        "split_counts_per_variant": split_counts,
        "fresh_official_true_pbr_base_per_variant": True,
        "checkpoint_state_accumulation": False,
        "baseline_outputs_rerun": False,
        "test_data_used": configured_test_data_used(validated.values),
        **configured_selection_provenance(validated.values),
    }
    write_once_json(run_dir / "summary.json", summary, run_dir)
    write_once_text(run_dir / "summary.md", _final_summary_markdown(summary), run_dir)
    token = configured_final_success_token(validated.values)
    write_once_text(run_dir / "_SUCCESS", token + "\n", run_dir)
    print(token)
    return summary


def run_inference(
    validated: ValidatedInferenceConfig,
    *,
    run_id: str,
    variants: Sequence[str],
) -> Path:
    requested = tuple(variants)
    variant_order = configured_variant_order(validated.values)
    if not requested or any(variant not in variant_order for variant in requested):
        raise ValueError(f"Runtime variants must be drawn from {variant_order}")
    fixed_profile = is_fixed_run_profile(validated.values)
    if is_full_validation(validated.values) and requested != variant_order:
        raise ValueError("full-validation inference must run both configured candidates together")
    if is_final_test(validated.values) and requested != variant_order:
        raise ValueError("final-test inference must run the frozen candidate exactly once")
    safe_run_id = _safe_run_id(run_id)
    run_dir = (validated.output_root / safe_run_id).resolve()
    if not is_relative_to(run_dir, validated.output_root):
        raise ValueError(f"Run directory escapes the output root: {run_dir}")
    if fixed_profile and safe_run_id != validated.values["fixed_run_id"]:
        raise ValueError(
            f"fixed run ID must be {validated.values['fixed_run_id']!r}"
        )
    if fixed_profile and run_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixed inference run: {run_dir}")

    import torch

    runtime_device = validate_runtime_device(torch, float(validated.values["minimum_gpu_memory_gib"]))
    validate_free_disk(validated.output_root, float(validated.values["minimum_free_disk_gib"]))
    run_dir.mkdir(parents=True, exist_ok=not fixed_profile)
    ensure_runtime_provenance(run_dir, validated, runtime_device)
    for variant_id in requested:
        run_variant(validated.variants[variant_id], validated, run_dir, torch_module=torch)
    if all((run_dir / variant_id).is_dir() for variant_id in variant_order):
        finalize_run(run_dir, validated)
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or run Phase 2N scope-checkpoint inference."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Pilot inference JSON config")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="Validate without Torch, CUDA, or Hunyuan")
    modes.add_argument("--run-all", action="store_true", help="Run all configured variants in order")
    modes.add_argument("--variant", choices=EXPECTED_VARIANT_ORDER, help="Run one variant for staged recovery")
    parser.add_argument("--run-id", help="Required for runtime modes; normally derived from SLURM_JOB_ID")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        if args.run_id:
            raise ValueError("--run-id is only valid with --run-all or --variant")
        run_check_only(args.config)
        return 0
    if not args.run_id:
        raise ValueError("--run-all and --variant require --run-id")
    validated = validate_config(args.config)
    if is_fixed_run_profile(validated.values) and not args.run_all:
        raise ValueError("fixed-profile inference requires --run-all")
    variants = configured_variant_order(validated.values) if args.run_all else (args.variant,)
    run_dir = run_inference(validated, run_id=args.run_id, variants=variants)
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
