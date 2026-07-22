#!/usr/bin/env python3
"""Deterministic Phase 2N Week 2 PC-S1/PC-Full pilot runner.

Import and check-only paths use the Python standard library only. Torch,
Lightning, the protocol reader, and Hunyuan are imported only by an explicit
runtime mode intended for the accompanying A100 Slurm job.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase2n_week2_pilot_training.json"
DAY5_CONFIG_RELATIVE = Path("configs/phase2n_day5_a100_preflight.json")
DAY5_SCRIPT_RELATIVE = Path("scripts/phase2n_day5_a100_preflight.py")
SELECTIVE_SOURCE_RELATIVE = Path("src/hy3dft/selective_training.py")
PROTOCOL_SOURCE_RELATIVE = Path("src/hy3dft/protocol_corrected_dataset.py")
SBATCH_RELATIVE = Path("env/run_phase2n_week2_train_pilots_a100.sbatch")
EVAL_MANIFEST_RELATIVE = Path("configs/phase2n_week2_pilot_eval_cases.json")
AUDITED_CONDITIONING_SOURCE = Path(
    "/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1/"
    "hy3dpaint/hunyuanpaintpbr/unet/model.py"
)
AUDITED_CONDITIONING_SOURCE_SHA256 = "5d135a73c12f0d7765366a72a37264b6bcb15da1551d3e7d60526d04b5eb1dc0"
EXPECTED_TRAIN_JSON_RELATIVE = Path(
    "data/hy3dpaint_train_examples/datav2_frame_panels_full101/examples_train_abs.json"
)
EXPECTED_OUTPUT_RELATIVE = Path("outputs/phase2n/week2_pilot_training")
EXPECTED_TRAIN_COUNT = 80
EXPECTED_MAX_STEPS = 320
EXPECTED_WARMUP_STEPS = 50
EXPECTED_CHECKPOINT_STEPS = (160, 320)
EXPECTED_SCOPE_ORDER = ("pc_s1", "pc_full")
EXPECTED_LEARNING_RATES = {"pc_s1": 1e-6, "pc_full": 5e-7}
EXPECTED_CONDITIONING_DROPOUT_POLICY = {
    "drop_cond_prob": 0.1,
    "require_mva_active": True,
    "maximum_seed_retries": 128,
}
EXPECTED_REFERENCE_WEIGHTS = {
    "005": 0.50,
    "004": 0.30,
    "000": 0.05,
    "001": 0.05,
    "002": 0.05,
    "003": 0.05,
}
EXPECTED_SCOPE_COUNTS = {
    "pc_s1": {"tensor_count": 80, "numel": 49_574_080},
    "pc_full": {"tensor_count": 981, "numel": 1_046_761_668},
}
HISTORICAL_MODEL_KEYS = (
    "images_cond",
    "images_albedo",
    "images_mr",
    "images_normal",
    "images_position",
    "name",
)
REQUIRED_DAY5_APIS = {
    "ValidatedConfiguration",
    "validate_config",
    "initialize_true_pbr_model",
    "run_production_loss",
    "_reset_runtime_seeds",
    "_prepend_runtime_paths",
    "move_model_batch_to_device",
    "optimizer_membership_report",
}
REQUIRED_DAY4_APIS = {
    "ProtocolCorrectedDatasetFromJson",
    "protocol_collate_fn",
    "apply_trainable_scope",
    "build_trainable_adamw",
    "build_warmup_constant_scheduler",
}
HEAVY_IMPORT_PREFIXES = (
    "torch",
    "pytorch_lightning",
    "lightning",
    "hunyuanpaintpbr",
    "omegaconf",
)
READINESS_TOKEN = "PHASE2N_WEEK2_PILOT_TRAINING_READINESS_OK"
MVA_SCHEDULE_TOKEN = "PHASE2N_WEEK2_MVA_ACTIVE_SCHEDULE_OK"
SCOPE_SUCCESS_TOKENS = {
    "pc_s1": "PHASE2N_WEEK2_PC_S1_TRAINING_OK",
    "pc_full": "PHASE2N_WEEK2_PC_FULL_TRAINING_OK",
}
TRACE_SUCCESS_TOKEN = "PHASE2N_WEEK2_TRACE_EQUIVALENCE_OK"
FINAL_SUCCESS_TOKEN = "PHASE2N_WEEK2_PILOT_TRAINING_OK"
CONFIG_KEYS = {
    "phase",
    "train_json",
    "output_root",
    "base_seed",
    "schedule_seed",
    "image_size",
    "batch_size",
    "num_workers",
    "augmentation_mode",
    "conditioning_dropout_policy",
    "max_steps",
    "warmup_steps",
    "gradient_clip_norm",
    "precision",
    "checkpoint_steps",
    "log_every_n_steps",
    "scope_order",
    "scopes",
    "minimum_gpu_memory_gib",
    "minimum_free_disk_gib",
}


@dataclass(frozen=True)
class ValidatedPilotConfiguration:
    values: dict[str, Any]
    config_path: Path
    project_root: Path
    train_json: Path
    output_root: Path
    train_sample_paths: tuple[Path, ...]
    eval_manifest: Path
    sbatch_path: Path


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


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc


def read_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact(config: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config {key!r} must be {expected!r}, got {actual!r}")


def _require_number(config: Mapping[str, Any], key: str, *, positive: bool = True) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config {key!r} must be numeric")
    if positive and value <= 0:
        raise ValueError(f"Config {key!r} must be positive")
    return float(value)


def validate_config_values(config: Mapping[str, Any]) -> None:
    extra = sorted(set(config) - CONFIG_KEYS)
    missing = sorted(CONFIG_KEYS - set(config))
    if extra or missing:
        raise ValueError(f"Training config keys differ from the frozen schema: missing={missing}, extra={extra}")
    _require_exact(config, "phase", "phase2n_week2_pilot_training")
    _require_exact(config, "train_json", EXPECTED_TRAIN_JSON_RELATIVE.as_posix())
    _require_exact(config, "output_root", EXPECTED_OUTPUT_RELATIVE.as_posix())
    _require_exact(config, "base_seed", 42)
    _require_exact(config, "schedule_seed", 42)
    _require_exact(config, "image_size", 512)
    _require_exact(config, "batch_size", 1)
    _require_exact(config, "num_workers", 0)
    _require_exact(config, "augmentation_mode", "none")
    policy = config.get("conditioning_dropout_policy")
    if not isinstance(policy, dict) or set(policy) != set(EXPECTED_CONDITIONING_DROPOUT_POLICY):
        raise ValueError(
            "Config conditioning_dropout_policy must contain exactly "
            f"{sorted(EXPECTED_CONDITIONING_DROPOUT_POLICY)}"
        )
    if (
        isinstance(policy.get("drop_cond_prob"), bool)
        or not isinstance(policy.get("drop_cond_prob"), (int, float))
        or float(policy["drop_cond_prob"]) != 0.1
    ):
        raise ValueError("Config conditioning_dropout_policy.drop_cond_prob must be exactly 0.1")
    if policy.get("require_mva_active") is not True:
        raise ValueError("Config conditioning_dropout_policy.require_mva_active must be exactly true")
    retries = policy.get("maximum_seed_retries")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries != 128:
        raise ValueError("Config conditioning_dropout_policy.maximum_seed_retries must be exactly 128")
    _require_exact(config, "max_steps", EXPECTED_MAX_STEPS)
    _require_exact(config, "warmup_steps", EXPECTED_WARMUP_STEPS)
    _require_exact(config, "gradient_clip_norm", 1.0)
    _require_exact(config, "precision", "bf16")
    _require_exact(config, "checkpoint_steps", list(EXPECTED_CHECKPOINT_STEPS))
    _require_exact(config, "log_every_n_steps", 10)
    _require_exact(config, "scope_order", list(EXPECTED_SCOPE_ORDER))
    _require_exact(config, "minimum_gpu_memory_gib", 70)
    _require_exact(config, "minimum_free_disk_gib", 15)
    scopes = config.get("scopes")
    if not isinstance(scopes, dict) or set(scopes) != set(EXPECTED_SCOPE_ORDER):
        raise ValueError(f"Config scopes must be exactly {list(EXPECTED_SCOPE_ORDER)}")
    for scope, expected_lr in EXPECTED_LEARNING_RATES.items():
        scope_config = scopes.get(scope)
        if not isinstance(scope_config, dict) or set(scope_config) != {"learning_rate"}:
            raise ValueError(f"Config scopes.{scope} must contain only learning_rate")
        actual_lr = scope_config["learning_rate"]
        if isinstance(actual_lr, bool) or not isinstance(actual_lr, (int, float)) or float(actual_lr) != expected_lr:
            raise ValueError(f"Config scopes.{scope}.learning_rate must be {expected_lr}, got {actual_lr!r}")
    for key in ("gradient_clip_norm", "minimum_gpu_memory_gib", "minimum_free_disk_gib"):
        _require_number(config, key)


def validate_train_json(path: Path, expected_count: int = EXPECTED_TRAIN_COUNT) -> tuple[Path, ...]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Training JSON must contain a list: {path}")
    if len(payload) != expected_count:
        raise ValueError(f"Training JSON must contain exactly {expected_count} assets, got {len(payload)}")
    paths: list[Path] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Training JSON item {index} must be a non-empty path string")
        sample_path = Path(value).expanduser()
        if not sample_path.is_absolute():
            raise ValueError(f"Training JSON item {index} is not absolute: {value}")
        sample_path = sample_path.resolve()
        if not sample_path.is_dir():
            raise FileNotFoundError(f"Training sample directory does not exist at index {index}: {sample_path}")
        paths.append(sample_path)
    if len(paths) != len(set(paths)):
        raise ValueError("Training JSON contains duplicate sample directories")
    return tuple(paths)


def _load_path_list_if_present(path: Path) -> set[Path]:
    if not path.is_file():
        return set()
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Split JSON must contain a list: {path}")
    result: set[Path] = set()
    for value in payload:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Split JSON contains a non-path value: {path}")
        result.add(Path(value).expanduser().resolve())
    return result


def reject_test_training_references(train_json: Path, train_paths: Sequence[Path]) -> None:
    if "test" in train_json.name.lower():
        raise ValueError(f"Test JSON is forbidden for Week 2 training: {train_json}")
    test_json = train_json.parent / "examples_test_abs.json"
    test_paths = _load_path_list_if_present(test_json)
    overlap = sorted(set(train_paths) & test_paths)
    if overlap:
        raise ValueError(f"Week 2 training includes test assets: {[str(path) for path in overlap[:20]]}")


def validate_output_root(output_root: Path, project_root: Path) -> Path:
    expected = (project_root / EXPECTED_OUTPUT_RELATIVE).resolve()
    resolved = output_root.expanduser().resolve()
    if resolved != expected:
        raise ValueError(f"Output root must be exactly {expected}, got {resolved}")
    phase_root = (project_root / "outputs" / "phase2n").resolve()
    if not is_relative_to(resolved, phase_root):
        raise ValueError(f"Output root is outside outputs/phase2n: {resolved}")
    forbidden = (
        project_root / "data",
        project_root / "checkpoints",
        Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work"),
    )
    if any(is_relative_to(resolved, path) for path in forbidden):
        raise ValueError(f"Unsafe Week 2 output root: {resolved}")
    return resolved


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"No existing parent for output path: {path}")
        current = current.parent
    return current


def validate_free_disk(output_root: Path, minimum_gib: float) -> dict[str, Any]:
    anchor = _nearest_existing_parent(output_root)
    usage = shutil.disk_usage(anchor)
    free_gib = usage.free / float(1024**3)
    if free_gib < minimum_gib:
        raise RuntimeError(f"Week 2 output filesystem has {free_gib:.2f} GiB free; requires {minimum_gib:.2f} GiB")
    return {"checked_path": str(anchor), "free_bytes": usage.free, "free_gib": free_gib}


def inspect_python_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _literal_mapping_assignment(path: Path, assignment_name: str) -> dict[str, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, dict):
            break
        return {str(key): float(item) for key, item in parsed.items()}
    raise RuntimeError(f"Could not statically read {assignment_name} from {path}")


def validate_week1_api_contracts(project_root: Path) -> dict[str, Any]:
    day5_path = project_root / DAY5_SCRIPT_RELATIVE
    selective_path = project_root / SELECTIVE_SOURCE_RELATIVE
    protocol_path = project_root / PROTOCOL_SOURCE_RELATIVE
    for path in (day5_path, selective_path, protocol_path, project_root / DAY5_CONFIG_RELATIVE):
        if not path.is_file():
            raise FileNotFoundError(f"Required committed Week 1 file is missing: {path}")
    missing_day5 = sorted(REQUIRED_DAY5_APIS - inspect_python_symbols(day5_path))
    missing_day4 = sorted(REQUIRED_DAY4_APIS - inspect_python_symbols(selective_path))
    if missing_day5 or missing_day4:
        raise RuntimeError(f"Committed Week 1 APIs changed: day5_missing={missing_day5}, day4_missing={missing_day4}")
    heavy = sorted(
        name
        for name in top_level_imports(day5_path)
        if name in HEAVY_IMPORT_PREFIXES or name.startswith(tuple(f"{prefix}." for prefix in HEAVY_IMPORT_PREFIXES))
    )
    if heavy:
        raise RuntimeError(f"Day 5 helper module is no longer import-safe: top-level imports={heavy}")
    reference_weights = _literal_mapping_assignment(protocol_path, "REFERENCE_VIEW_WEIGHTS")
    if reference_weights != EXPECTED_REFERENCE_WEIGHTS:
        raise RuntimeError(f"Reference-view probabilities changed: {reference_weights}")
    return {
        "day5_runtime_helpers": sorted(REQUIRED_DAY5_APIS),
        "day4_training_apis": sorted(REQUIRED_DAY4_APIS),
        "reference_view_weights": reference_weights,
        "shared_runtime_extraction_needed": False,
    }


def validate_sbatch_contract(path: Path, project_root: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Week 2 sbatch is missing: {path}")
    text = path.read_text(encoding="utf-8")
    required = (
        "#SBATCH -p a100",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=04:00:00",
        "set -eo pipefail",
        "phase2n_week2_train_pilots.py",
        "--check-only",
        "--run-all",
        "--run-id",
        "===== JOB END: SUCCESS =====",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError(f"Week 2 sbatch is missing required contracts: {missing}")
    forbidden = ("gpgpuC", "--constraint=a100", "set -u", "--test", "examples_test_abs.json")
    present = [token for token in forbidden if token in text]
    if present:
        raise ValueError(f"Week 2 sbatch contains forbidden tokens: {present}")
    if str(project_root / "src") not in text and "$PROJ/src" not in text:
        raise ValueError("Week 2 sbatch must add project src to PYTHONPATH")
    if "$PROJ/scripts" not in text:
        raise ValueError("Week 2 sbatch must add project scripts to PYTHONPATH")


def validate_eval_manifest(path: Path, project_root: Path, *, require_files: bool = True) -> dict[str, Any]:
    payload = read_json_object(path)
    if payload.get("selection_frozen_before_training") is not True:
        raise ValueError("Week 2 evaluation selection must be frozen before training")
    if payload.get("selected_input_view") != "005" or payload.get("reference_lighting") != "AL":
        raise ValueError("Week 2 evaluation must use selected input view 005 and AL lighting")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("Week 2 evaluation manifest must contain exactly eight cases")
    ids: list[str] = []
    split_counts = {"val": 0, "train_sanity": 0, "test": 0}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {index} must be an object")
        asset_id = case.get("asset_id")
        source_split = case.get("source_split")
        eval_split = case.get("eval_split")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError(f"Evaluation case {index} has no asset_id")
        allowed_split_pairs = {("val", "val"), ("train", "train_sanity")}
        if (source_split, eval_split) not in allowed_split_pairs:
            raise ValueError(
                f"Test or unknown split is forbidden in evaluation case {asset_id}: "
                f"source={source_split!r}, eval={eval_split!r}"
            )
        if case.get("selected_input_view") != "005" or case.get("reference_lighting") != "AL":
            raise ValueError(f"Evaluation protocol drift for {asset_id}")
        ids.append(asset_id)
        split_counts[eval_split] += 1
        for key in ("mesh_path", "reference_image_path"):
            value = case.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Evaluation case {asset_id} has no {key}")
            resolved = resolve_project_path(value, project_root)
            if require_files and (not resolved.is_file() or resolved.stat().st_size <= 0):
                raise FileNotFoundError(f"Evaluation case {asset_id} {key} is missing or empty: {resolved}")
    if len(ids) != len(set(ids)):
        raise ValueError("Week 2 evaluation manifest contains duplicate assets")
    if split_counts != {"val": 6, "train_sanity": 2, "test": 0}:
        raise ValueError(f"Week 2 evaluation split counts are wrong: {split_counts}")
    if payload.get("split_counts") != split_counts or payload.get("case_count") != 8:
        raise ValueError("Week 2 evaluation manifest summary counts do not match its cases")
    return {"case_ids": ids, "split_counts": split_counts}


def validate_training_config(
    config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    *,
    require_repository_contracts: bool = True,
    require_eval_files: bool = True,
    check_free_space: bool = True,
) -> ValidatedPilotConfiguration:
    root = Path(project_root).expanduser().resolve()
    path = resolve_project_path(config_path, root)
    config = read_json_object(path)
    validate_config_values(config)
    train_json = resolve_project_path(config["train_json"], root)
    expected_train_json = (root / EXPECTED_TRAIN_JSON_RELATIVE).resolve()
    if train_json != expected_train_json:
        raise ValueError(f"Week 2 train JSON must be exactly {expected_train_json}")
    train_paths = validate_train_json(train_json)
    reject_test_training_references(train_json, train_paths)
    output_root = validate_output_root(resolve_project_path(config["output_root"], root), root)
    if check_free_space:
        validate_free_disk(output_root, float(config["minimum_free_disk_gib"]))
    eval_manifest = (root / EVAL_MANIFEST_RELATIVE).resolve()
    validate_eval_manifest(eval_manifest, root, require_files=require_eval_files)
    sbatch_path = (root / SBATCH_RELATIVE).resolve()
    if require_repository_contracts:
        validate_week1_api_contracts(root)
        validate_sbatch_contract(sbatch_path, root)
    return ValidatedPilotConfiguration(
        values=dict(config),
        config_path=path,
        project_root=root,
        train_json=train_json,
        output_root=output_root,
        train_sample_paths=train_paths,
        eval_manifest=eval_manifest,
        sbatch_path=sbatch_path,
    )


def _derive_uint64(payload: Mapping[str, Any]) -> int:
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big", signed=False)


def validate_conditioning_source_audit() -> dict[str, Any]:
    source_path = AUDITED_CONDITIONING_SOURCE.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Audited Hunyuan conditioning source is missing: {source_path}")
    actual_sha256 = sha256_file(source_path)
    if actual_sha256 != AUDITED_CONDITIONING_SOURCE_SHA256:
        raise RuntimeError(
            "Audited Hunyuan conditioning source SHA-256 changed: "
            f"{actual_sha256} != {AUDITED_CONDITIONING_SOURCE_SHA256}: {source_path}"
        )
    return {
        "source_path": str(source_path),
        "sha256": actual_sha256,
        "batch_size": 1,
        "use_dino": True,
        "draw_order": [
            "normal_drop_draw",
            "position_drop_draw",
            "dino_primary_drop_draw",
            "dino_secondary_drop_draw",
            "mva_ref_branch_draw",
            "mva_ref_choice_draw_if_branch_above_one_minus_drop_prob",
        ],
    }


class _NumpyMT19937:
    """Minimal NumPy RandomState-compatible MT19937 scalar draw generator."""

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError("NumPy-compatible seed must be an unsigned 32-bit integer")
        self._state = [0] * self._N
        self._state[0] = seed
        for index in range(1, self._N):
            previous = self._state[index - 1]
            self._state[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF
        self._index = self._N

    def _twist(self) -> None:
        state = self._state
        for index in range(self._N - self._M):
            value = (state[index] & self._UPPER_MASK) | (state[index + 1] & self._LOWER_MASK)
            state[index] = state[index + self._M] ^ (value >> 1) ^ (self._MATRIX_A if value & 1 else 0)
        for index in range(self._N - self._M, self._N - 1):
            value = (state[index] & self._UPPER_MASK) | (state[index + 1] & self._LOWER_MASK)
            state[index] = state[index + self._M - self._N] ^ (value >> 1) ^ (
                self._MATRIX_A if value & 1 else 0
            )
        value = (state[-1] & self._UPPER_MASK) | (state[0] & self._LOWER_MASK)
        state[-1] = state[self._M - 1] ^ (value >> 1) ^ (self._MATRIX_A if value & 1 else 0)
        self._index = 0

    def _uint32(self) -> int:
        if self._index >= self._N:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF

    def random_sample(self) -> float:
        high = self._uint32() >> 5
        low = self._uint32() >> 6
        return (high * 67108864.0 + low) / 9007199254740992.0


def preview_conditioning_dropout(seed: int, drop_cond_prob: float) -> dict[str, Any]:
    """Mirror audited upstream NumPy draws for batch size 1 and use_dino=True."""

    if isinstance(drop_cond_prob, bool) or not isinstance(drop_cond_prob, (int, float)):
        raise ValueError("drop_cond_prob must be numeric")
    probability = float(drop_cond_prob)
    if not 0.0 <= probability < 0.5:
        raise ValueError("drop_cond_prob must be in [0, 0.5)")
    rng = _NumpyMT19937(seed)
    draws: dict[str, Any] = {
        "normal_drop_draw": rng.random_sample(),
        "position_drop_draw": rng.random_sample(),
        "dino_primary_drop_draw": rng.random_sample(),
        "dino_secondary_drop_draw": rng.random_sample(),
        "mva_ref_branch_draw": rng.random_sample(),
        "mva_ref_choice_draw": None,
    }
    mva_scale = 1.0
    ref_scale = 1.0
    branch_draw = float(draws["mva_ref_branch_draw"])
    if branch_draw < probability:
        mva_scale = 0.0
        ref_scale = 0.0
    elif branch_draw > 1.0 - probability:
        choice_draw = rng.random_sample()
        draws["mva_ref_choice_draw"] = choice_draw
        if choice_draw < 0.5:
            mva_scale = 0.0
        else:
            ref_scale = 0.0
    return {**draws, "expected_mva_scale": mva_scale, "expected_ref_scale": ref_scale}


def derive_mva_retry_seed(schedule_identity: Mapping[str, Any], retry_index: int) -> int:
    if not isinstance(retry_index, int) or isinstance(retry_index, bool) or retry_index <= 0:
        raise ValueError("retry_index must be a positive integer")
    return _derive_uint64(
        {
            "purpose": "phase2n_week2_mva_active_retry_seed_v1",
            "schedule_identity": dict(schedule_identity),
            "retry_index": retry_index,
        }
    ) % (2**32)


def resolve_mva_active_seed(
    original_candidate_seed: int,
    schedule_identity: Mapping[str, Any],
    *,
    drop_cond_prob: float,
    maximum_seed_retries: int,
) -> dict[str, Any]:
    if (
        not isinstance(maximum_seed_retries, int)
        or isinstance(maximum_seed_retries, bool)
        or maximum_seed_retries < 0
    ):
        raise ValueError("maximum_seed_retries must be a non-negative integer")
    for retry_count in range(maximum_seed_retries + 1):
        accepted_seed = (
            original_candidate_seed
            if retry_count == 0
            else derive_mva_retry_seed(schedule_identity, retry_count)
        )
        preview = preview_conditioning_dropout(accepted_seed, drop_cond_prob)
        if preview["expected_mva_scale"] == 1.0:
            return {
                "original_candidate_seed": original_candidate_seed,
                "training_step_seed": accepted_seed,
                "mva_seed_retry_count": retry_count,
                **preview,
            }
    raise RuntimeError(
        "Could not derive an MVA-active training seed within "
        f"{maximum_seed_retries} retries for schedule identity {dict(schedule_identity)}"
    )


def build_training_schedule(
    sample_paths: Sequence[str | Path],
    *,
    conditioning_dropout_policy: Mapping[str, Any],
    conditioning_source_audit: Mapping[str, Any],
    schedule_seed: int = 42,
    max_steps: int = EXPECTED_MAX_STEPS,
) -> dict[str, Any]:
    paths = tuple(Path(path).expanduser().resolve() for path in sample_paths)
    policy = dict(conditioning_dropout_policy)
    source_audit = dict(conditioning_source_audit)
    if policy != EXPECTED_CONDITIONING_DROPOUT_POLICY or policy.get("require_mva_active") is not True:
        raise ValueError("Schedule requires the exact frozen MVA-active conditioning dropout policy")
    if (
        not isinstance(source_audit.get("source_path"), str)
        or not isinstance(source_audit.get("sha256"), str)
        or source_audit.get("batch_size") != 1
        or source_audit.get("use_dino") is not True
        or not isinstance(source_audit.get("draw_order"), list)
    ):
        raise ValueError("Schedule conditioning source audit is incomplete")
    if len(paths) != EXPECTED_TRAIN_COUNT:
        raise ValueError(f"Schedule requires exactly {EXPECTED_TRAIN_COUNT} assets, got {len(paths)}")
    if len(set(paths)) != len(paths):
        raise ValueError("Schedule input contains duplicate assets")
    if max_steps % len(paths) != 0:
        raise ValueError("max_steps must be an exact multiple of the train asset count")
    records: list[dict[str, Any]] = []
    epoch_count = max_steps // len(paths)
    for epoch in range(epoch_count):
        epoch_seed = _derive_uint64({"purpose": "epoch_permutation", "schedule_seed": schedule_seed, "epoch": epoch})
        permutation = list(range(len(paths)))
        random.Random(epoch_seed).shuffle(permutation)
        for within_epoch_position, asset_index in enumerate(permutation):
            asset_path = paths[asset_index]
            global_update = len(records) + 1
            original_candidate_seed = _derive_uint64(
                {
                    "purpose": "training_step",
                    "schedule_seed": schedule_seed,
                    "global_update": global_update,
                    "epoch": epoch,
                    "asset_index": asset_index,
                    "asset_id": asset_path.name,
                }
            ) % (2**32)
            schedule_identity = {
                "schedule_seed": schedule_seed,
                "global_update": global_update,
                "epoch": epoch,
                "within_epoch_position": within_epoch_position,
                "asset_index": asset_index,
                "asset_id": asset_path.name,
                "asset_path": str(asset_path),
                "original_candidate_seed": original_candidate_seed,
            }
            seed_record = resolve_mva_active_seed(
                original_candidate_seed,
                schedule_identity,
                drop_cond_prob=float(policy["drop_cond_prob"]),
                maximum_seed_retries=int(policy["maximum_seed_retries"]),
            )
            records.append(
                {
                    "global_update": global_update,
                    "epoch": epoch,
                    "within_epoch_position": within_epoch_position,
                    "asset_index": asset_index,
                    "asset_id": asset_path.name,
                    "asset_path": str(asset_path),
                    **seed_record,
                }
            )
    mva_active_count = sum(record["expected_mva_scale"] == 1.0 for record in records)
    mva_inactive_count = len(records) - mva_active_count
    retry_record_count = sum(record["mva_seed_retry_count"] > 0 for record in records)
    core = {
        "phase": "phase2n_week2_pilot_training",
        "schedule_seed": schedule_seed,
        "asset_count": len(paths),
        "epoch_count": epoch_count,
        "record_count": len(records),
        "indexing": {"global_update": "one_based", "epoch": "zero_based", "within_epoch_position": "zero_based"},
        "conditioning_dropout_policy": policy,
        "conditioning_source_audit": source_audit,
        "mva_active_record_count": mva_active_count,
        "mva_inactive_record_count": mva_inactive_count,
        "mva_seed_retry_record_count": retry_record_count,
        "records": records,
    }
    return {**core, "schedule_hash": sha256_bytes(canonical_json_bytes(core))}


def _schedule_identity_from_record(record: Mapping[str, Any], schedule_seed: int) -> dict[str, Any]:
    return {
        "schedule_seed": schedule_seed,
        "global_update": record["global_update"],
        "epoch": record["epoch"],
        "within_epoch_position": record["within_epoch_position"],
        "asset_index": record["asset_index"],
        "asset_id": record["asset_id"],
        "asset_path": record["asset_path"],
        "original_candidate_seed": record["original_candidate_seed"],
    }


def mva_schedule_stats(schedule: Mapping[str, Any]) -> dict[str, int]:
    records = schedule.get("records")
    if not isinstance(records, list):
        raise ValueError("Training schedule records must be a list")
    active = sum(record.get("expected_mva_scale") == 1.0 for record in records)
    return {
        "mva_active_records": active,
        "mva_inactive_records": len(records) - active,
        "mva_seed_retry_records": sum(
            isinstance(record.get("mva_seed_retry_count"), int) and record["mva_seed_retry_count"] > 0
            for record in records
        ),
    }


def validate_training_schedule(
    schedule: Mapping[str, Any],
    sample_paths: Sequence[Path],
    *,
    conditioning_dropout_policy: Mapping[str, Any],
    conditioning_source_audit: Mapping[str, Any],
) -> str:
    records = schedule.get("records")
    policy = dict(conditioning_dropout_policy)
    source_audit = dict(conditioning_source_audit)
    if schedule.get("conditioning_dropout_policy") != policy:
        raise ValueError("Training schedule conditioning policy differs from the validated config")
    if schedule.get("conditioning_source_audit") != source_audit:
        raise ValueError("Training schedule conditioning source audit differs from the live audited source")
    if policy != EXPECTED_CONDITIONING_DROPOUT_POLICY or policy.get("require_mva_active") is not True:
        raise ValueError("Training schedule does not require the frozen MVA-active policy")
    if not isinstance(records, list) or len(records) != EXPECTED_MAX_STEPS:
        raise ValueError("Training schedule must contain exactly 320 records")
    if schedule.get("epoch_count") != 4 or schedule.get("asset_count") != EXPECTED_TRAIN_COUNT:
        raise ValueError("Training schedule must contain four complete 80-asset epochs")
    expected_indices = set(range(EXPECTED_TRAIN_COUNT))
    for epoch in range(4):
        epoch_records = [record for record in records if record.get("epoch") == epoch]
        if len(epoch_records) != EXPECTED_TRAIN_COUNT:
            raise ValueError(f"Schedule epoch {epoch} does not contain exactly 80 records")
        if {record.get("asset_index") for record in epoch_records} != expected_indices:
            raise ValueError(f"Schedule epoch {epoch} does not contain each asset exactly once")
        if [record.get("within_epoch_position") for record in epoch_records] != list(range(EXPECTED_TRAIN_COUNT)):
            raise ValueError(f"Schedule epoch {epoch} positions are not canonical")
    seed_fields = (
        "original_candidate_seed",
        "training_step_seed",
        "mva_seed_retry_count",
        "normal_drop_draw",
        "position_drop_draw",
        "dino_primary_drop_draw",
        "dino_secondary_drop_draw",
        "mva_ref_branch_draw",
        "mva_ref_choice_draw",
        "expected_mva_scale",
        "expected_ref_scale",
    )
    for expected_update, record in enumerate(records, start=1):
        if record.get("global_update") != expected_update:
            raise ValueError("Schedule global updates are not contiguous and one-based")
        index = record.get("asset_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(sample_paths):
            raise ValueError(f"Schedule asset index is invalid at update {expected_update}")
        expected_path = str(Path(sample_paths[index]).resolve())
        if record.get("asset_path") != expected_path or record.get("asset_id") != Path(expected_path).name:
            raise ValueError(f"Schedule asset path/index mismatch at update {expected_update}")
        original_seed = _derive_uint64(
            {
                "purpose": "training_step",
                "schedule_seed": schedule["schedule_seed"],
                "global_update": expected_update,
                "epoch": record["epoch"],
                "asset_index": index,
                "asset_id": record["asset_id"],
            }
        ) % (2**32)
        if record.get("original_candidate_seed") != original_seed:
            raise ValueError(f"Schedule original candidate seed drifted at update {expected_update}")
        expected_seed_record = resolve_mva_active_seed(
            original_seed,
            _schedule_identity_from_record(record, int(schedule["schedule_seed"])),
            drop_cond_prob=float(policy["drop_cond_prob"]),
            maximum_seed_retries=int(policy["maximum_seed_retries"]),
        )
        if any(record.get(field) != expected_seed_record.get(field) for field in seed_fields):
            raise ValueError(f"Schedule conditioning preview or retry drifted at update {expected_update}")
        if record.get("expected_mva_scale") != 1.0:
            raise RuntimeError(f"Schedule predicts inactive MVA at update {expected_update}")
    stats = mva_schedule_stats(schedule)
    if stats["mva_active_records"] != EXPECTED_MAX_STEPS or stats["mva_inactive_records"] != 0:
        raise RuntimeError(f"Training schedule is not fully MVA-active: {stats}")
    if schedule.get("mva_active_record_count") != stats["mva_active_records"]:
        raise ValueError("Training schedule MVA-active count metadata is wrong")
    if schedule.get("mva_inactive_record_count") != stats["mva_inactive_records"]:
        raise ValueError("Training schedule MVA-inactive count metadata is wrong")
    if schedule.get("mva_seed_retry_record_count") != stats["mva_seed_retry_records"]:
        raise ValueError("Training schedule MVA retry count metadata is wrong")
    core = {key: value for key, value in schedule.items() if key != "schedule_hash"}
    actual_hash = sha256_bytes(canonical_json_bytes(core))
    if schedule.get("schedule_hash") != actual_hash:
        raise ValueError("Training schedule hash does not match its contents")
    return actual_hash


def sampling_trace_record(schedule_record: Mapping[str, Any], protocol_metadata: Mapping[str, Any]) -> dict[str, Any]:
    required_metadata = {
        "asset_id",
        "selected_reference_view",
        "reference_lighting_pair",
        "reference_image_paths",
        "target_view_order",
        "derived_seed",
        "spatial_augmentation",
        "epoch",
    }
    missing = sorted(required_metadata - set(protocol_metadata))
    if missing:
        raise ValueError(f"Protocol metadata is missing sampling fields: {missing}")
    if protocol_metadata["asset_id"] != schedule_record["asset_id"]:
        raise RuntimeError("Scheduled asset and protocol metadata asset differ")
    if protocol_metadata["epoch"] != schedule_record["epoch"]:
        raise RuntimeError("Scheduled epoch and protocol metadata epoch differ")
    if protocol_metadata["spatial_augmentation"] != "none":
        raise RuntimeError("Week 2 sampling trace detected spatial augmentation")
    if protocol_metadata["target_view_order"] != ["000", "001", "002", "003", "004", "005"]:
        raise RuntimeError("Target-view ordering drifted from 000-005")
    return {
        "global_update": schedule_record["global_update"],
        "epoch": schedule_record["epoch"],
        "within_epoch_position": schedule_record["within_epoch_position"],
        "asset_index": schedule_record["asset_index"],
        "asset_id": schedule_record["asset_id"],
        "asset_path": schedule_record["asset_path"],
        "original_candidate_seed": schedule_record["original_candidate_seed"],
        "training_step_seed": schedule_record["training_step_seed"],
        "expected_mva_scale": schedule_record["expected_mva_scale"],
        "expected_ref_scale": schedule_record["expected_ref_scale"],
        "mva_seed_retry_count": schedule_record["mva_seed_retry_count"],
        "selected_reference_view": protocol_metadata["selected_reference_view"],
        "reference_lighting_pair": list(protocol_metadata["reference_lighting_pair"]),
        "reference_image_paths": list(protocol_metadata["reference_image_paths"]),
        "target_view_order": list(protocol_metadata["target_view_order"]),
        "protocol_derived_seed": protocol_metadata["derived_seed"],
        "spatial_augmentation": protocol_metadata["spatial_augmentation"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
        records.append(payload)
    return records


def sha256_jsonl_prefix(path: Path, record_count: int) -> str:
    if record_count <= 0:
        raise ValueError("JSONL prefix record_count must be positive")
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            if seen >= record_count:
                break
            digest.update(line)
            seen += 1
    if seen != record_count:
        raise RuntimeError(f"JSONL file has only {seen} records; required prefix is {record_count}: {path}")
    return digest.hexdigest()


def compare_sampling_traces(pc_s1_trace: Path, pc_full_trace: Path, expected_count: int = EXPECTED_MAX_STEPS) -> dict[str, Any]:
    left = read_jsonl(pc_s1_trace)
    right = read_jsonl(pc_full_trace)
    if len(left) != expected_count or len(right) != expected_count:
        raise RuntimeError(f"Sampling traces must each contain {expected_count} records")
    asset_mismatches: list[int] = []
    protocol_mismatches: list[int] = []
    for index, (left_row, right_row) in enumerate(zip(left, right), start=1):
        asset_fields = (
            "global_update",
            "epoch",
            "asset_index",
            "asset_id",
            "asset_path",
            "original_candidate_seed",
            "training_step_seed",
            "expected_mva_scale",
            "expected_ref_scale",
            "mva_seed_retry_count",
        )
        protocol_fields = (
            "selected_reference_view",
            "reference_lighting_pair",
            "reference_image_paths",
            "target_view_order",
            "protocol_derived_seed",
            "spatial_augmentation",
        )
        if any(left_row.get(field) != right_row.get(field) for field in asset_fields):
            asset_mismatches.append(index)
        if any(left_row.get(field) != right_row.get(field) for field in protocol_fields):
            protocol_mismatches.append(index)
    left_hash = sha256_file(pc_s1_trace)
    right_hash = sha256_file(pc_full_trace)
    equivalent = not asset_mismatches and not protocol_mismatches and left_hash == right_hash
    report = {
        "status": "OK" if equivalent else "FAIL",
        "record_count": expected_count,
        "pc_s1_trace_sha256": left_hash,
        "pc_full_trace_sha256": right_hash,
        "trace_hashes_equal": left_hash == right_hash,
        "asset_order_equal": not asset_mismatches,
        "reference_view_and_light_equal": not protocol_mismatches,
        "asset_mismatch_updates": asset_mismatches,
        "protocol_mismatch_updates": protocol_mismatches,
    }
    if not equivalent:
        raise RuntimeError(f"PC-S1 and PC-Full sampling traces differ: {report}")
    return report


def require_finite_scalar(value: float, label: str = "value") -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} is NaN or Inf: {numeric}")
    return numeric


def validate_scope_integrity(model: Any, expected_names: Sequence[str]) -> None:
    live_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if live_names != tuple(expected_names):
        missing = sorted(set(expected_names) - set(live_names))
        extra = sorted(set(live_names) - set(expected_names))
        raise RuntimeError(f"Trainable scope drift detected: missing={missing[:20]}, extra={extra[:20]}")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def atomic_write_text(path: Path, text: str, allowed_root: Path, *, overwrite: bool = False) -> Path:
    destination = path.expanduser().resolve()
    root = allowed_root.expanduser().resolve()
    if not is_relative_to(destination, root):
        raise ValueError(f"Output path is outside the allowed run root: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic-write temporary path already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def atomic_write_json(path: Path, payload: Any, allowed_root: Path, *, overwrite: bool = False) -> Path:
    return atomic_write_text(path, _json_text(payload), allowed_root, overwrite=overwrite)


def append_jsonl(path: Path, payload: Mapping[str, Any], allowed_root: Path) -> None:
    destination = path.resolve()
    if not is_relative_to(destination, allowed_root.resolve()):
        raise ValueError(f"JSONL path is outside the run root: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(dict(payload)).decode("ascii") + "\n")
        handle.flush()


def run_check_only(config_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> ValidatedPilotConfiguration:
    validated = validate_training_config(config_path, project_root)
    api_report = validate_week1_api_contracts(validated.project_root)
    disk_report = validate_free_disk(validated.output_root, float(validated.values["minimum_free_disk_gib"]))
    eval_report = validate_eval_manifest(validated.eval_manifest, validated.project_root)
    source_audit = validate_conditioning_source_audit()
    policy = validated.values["conditioning_dropout_policy"]
    schedule = build_training_schedule(
        validated.train_sample_paths,
        conditioning_dropout_policy=policy,
        conditioning_source_audit=source_audit,
        schedule_seed=int(validated.values["schedule_seed"]),
        max_steps=int(validated.values["max_steps"]),
    )
    validate_training_schedule(
        schedule,
        validated.train_sample_paths,
        conditioning_dropout_policy=policy,
        conditioning_source_audit=source_audit,
    )
    stats = mva_schedule_stats(schedule)
    print(f"config={validated.config_path}")
    print(f"train_json={validated.train_json}")
    print(f"train_count={len(validated.train_sample_paths)}")
    print(f"conditioning_source={source_audit['source_path']}")
    print(f"conditioning_source_sha256={source_audit['sha256']}")
    print(f"schedule_records={schedule['record_count']}")
    print(f"schedule_epochs={schedule['epoch_count']}")
    print(f"mva_active_records={stats['mva_active_records']}")
    print(f"mva_inactive_records={stats['mva_inactive_records']}")
    print(f"mva_seed_retry_records={stats['mva_seed_retry_records']}")
    print(f"schedule_hash={schedule['schedule_hash']}")
    print(f"output_root={validated.output_root}")
    print(f"free_disk_gib={disk_report['free_gib']:.2f}")
    print(f"scope_order={','.join(validated.values['scope_order'])}")
    print(f"eval_case_ids={','.join(eval_report['case_ids'])}")
    print(f"day5_runtime_reuse={not api_report['shared_runtime_extraction_needed']}")
    print(MVA_SCHEDULE_TOKEN)
    print(READINESS_TOKEN)
    return validated


def _load_day5_runtime_module():
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import phase2n_day5_a100_preflight as day5

    return day5



def _prepend_runtime_paths(project_root: Path) -> None:
    for path in (project_root / "src", project_root / "scripts"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def validate_runtime_device(torch: Any, minimum_gpu_memory_gib: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 2 runtime requires CUDA inside the A100 sbatch")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Week 2 runtime requires A100-class bf16 support")
    properties = torch.cuda.get_device_properties(0)
    total_bytes = int(properties.total_memory)
    total_gib = total_bytes / float(1024**3)
    name = str(properties.name)
    if "A100" not in name.upper():
        raise RuntimeError(f"Week 2 runtime requires an A100, got {name!r}")
    if total_gib < minimum_gpu_memory_gib:
        raise RuntimeError(f"A100 has {total_gib:.2f} GiB, below required {minimum_gpu_memory_gib:.2f} GiB")
    return {
        "device_index": 0,
        "device_name": name,
        "total_memory_bytes": total_bytes,
        "total_memory_gib": total_gib,
        "bf16_supported": True,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _git_head(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _safe_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id must use only letters, digits, dot, underscore, and hyphen")
    return run_id


def _scope_paths(scope_dir: Path) -> dict[str, Path]:
    return {
        "scope_report": scope_dir / "scope_report.json",
        "metrics": scope_dir / "training_metrics.jsonl",
        "trace": scope_dir / "sampling_trace.jsonl",
        "summary_json": scope_dir / "summary.json",
        "summary_md": scope_dir / "summary.md",
        "success": scope_dir / "_SUCCESS",
        "checkpoints": scope_dir / "checkpoints",
    }


def ensure_schedule_file(run_dir: Path, expected_schedule: Mapping[str, Any], sample_paths: Sequence[Path]) -> Path:
    path = run_dir / "training_schedule.json"
    if path.exists():
        existing = read_json_object(path)
        validate_training_schedule(
            existing,
            sample_paths,
            conditioning_dropout_policy=expected_schedule["conditioning_dropout_policy"],
            conditioning_source_audit=expected_schedule["conditioning_source_audit"],
        )
        if canonical_json_bytes(existing) != canonical_json_bytes(expected_schedule):
            raise RuntimeError("Existing training schedule differs from the deterministic schedule for this config")
        return path
    atomic_write_json(path, expected_schedule, run_dir)
    return path


def ensure_runtime_manifest(
    run_dir: Path,
    validated: ValidatedPilotConfiguration,
    schedule: Mapping[str, Any],
    runtime_device: Mapping[str, Any],
) -> Path:
    path = run_dir / "00_RUNTIME_MANIFEST.json"
    config_hash = sha256_file(validated.config_path)
    expected_identity = {
        "phase": validated.values["phase"],
        "run_id": run_dir.name,
        "config_path": str(validated.config_path),
        "config_sha256": config_hash,
        "schedule_sha256": schedule["schedule_hash"],
        "train_json": str(validated.train_json),
        "train_count": len(validated.train_sample_paths),
        "output_root": str(validated.output_root),
        "conditioning_dropout_policy": schedule["conditioning_dropout_policy"],
        "conditioning_source_audit": schedule["conditioning_source_audit"],
    }
    if path.exists():
        existing = read_json_object(path)
        for key, value in expected_identity.items():
            if existing.get(key) != value:
                raise RuntimeError(f"Existing run manifest differs at {key}: {existing.get(key)!r} != {value!r}")
        return path
    payload = {
        **expected_identity,
        "status": "RUNNING",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "project_git_head": _git_head(validated.project_root),
        "runtime_device": dict(runtime_device),
        "scope_order": list(validated.values["scope_order"]),
        "fresh_true_pbr_base_per_scope": True,
        "formal_training": True,
        "test_data_used": False,
        "spatial_augmentation": "none",
        "checkpoint_format": "model-only trainable-scope state",
    }
    return atomic_write_json(path, payload, run_dir)


def _torch_load_state(path: Path, torch: Any) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older runtime fallback.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Scope checkpoint is not a state mapping: {path}")
    return payload


def atomic_torch_save(payload: Mapping[str, Any], path: Path, allowed_root: Path, torch: Any) -> Path:
    destination = path.resolve()
    if not is_relative_to(destination, allowed_root.resolve()):
        raise ValueError(f"Checkpoint is outside the Week 2 run root: {destination}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Checkpoint temporary path already exists: {temporary}")
    torch.save(dict(payload), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def collect_trainable_scope_state(model: Any, expected_names: Sequence[str]) -> dict[str, Any]:
    named_parameters = dict(model.named_parameters())
    missing = sorted(set(expected_names) - set(named_parameters))
    if missing:
        raise RuntimeError(f"Checkpoint names are missing from model: {missing[:20]}")
    state: dict[str, Any] = {}
    for name in expected_names:
        parameter = named_parameters[name]
        if not parameter.requires_grad:
            raise RuntimeError(f"Refusing to checkpoint frozen parameter: {name}")
        state[name] = parameter.detach().cpu().clone()
    return state


def save_scope_checkpoint(
    *,
    scope_root: Any,
    scope: str,
    scope_report: Mapping[str, Any],
    step: int,
    checkpoints_dir: Path,
    run_dir: Path,
    schedule_hash: str,
    sampling_trace_hash: str,
    learning_rate: float,
    base_identifier: str,
    torch: Any,
) -> dict[str, Any]:
    if step not in EXPECTED_CHECKPOINT_STEPS:
        raise ValueError(f"Checkpoint step must be one of {EXPECTED_CHECKPOINT_STEPS}, got {step}")
    expected_names = tuple(scope_report["trainable_parameter_names"])
    validate_scope_integrity(scope_root, expected_names)
    state = collect_trainable_scope_state(scope_root, expected_names)
    dtype_by_name = {name: str(tensor.dtype) for name, tensor in state.items()}
    numel_by_name = {name: int(tensor.numel()) for name, tensor in state.items()}
    state_path = checkpoints_dir / f"step_{step}_scope_state.pt"
    manifest_path = checkpoints_dir / f"step_{step}_manifest.json"
    atomic_torch_save(state, state_path, run_dir, torch)
    del state
    gc.collect()
    loaded = _torch_load_state(state_path, torch)
    loaded_names = tuple(loaded.keys())
    if loaded_names != expected_names or set(loaded_names) != set(expected_names):
        raise RuntimeError("Reloaded scope checkpoint keys differ from exact trainable names")
    del loaded
    gc.collect()
    manifest = {
        "format": "phase2n_week2_model_only_trainable_scope_v1",
        "scope": scope,
        "global_update": step,
        "checkpoint_path": str(state_path.resolve()),
        "byte_size": state_path.stat().st_size,
        "sha256": sha256_file(state_path),
        "trainable_parameter_names": list(expected_names),
        "trainable_parameter_tensor_count": len(expected_names),
        "trainable_parameter_numel": sum(numel_by_name.values()),
        "dtype_by_name": dtype_by_name,
        "numel_by_name": numel_by_name,
        "scope_report": dict(scope_report),
        "base_identifier": base_identifier,
        "schedule_sha256": schedule_hash,
        "sampling_trace_sha256": sampling_trace_hash,
        "learning_rate_after_update": float(learning_rate),
        "contains_optimizer_state": False,
        "contains_scheduler_state": False,
        "contains_frozen_parameters": False,
        "contains_full_model": False,
        "atomic_write": True,
        "reload_key_verification": "exact",
    }
    atomic_write_json(manifest_path, manifest, run_dir)
    return manifest


def validate_scope_checkpoint(
    state_path: Path,
    manifest_path: Path,
    *,
    expected_scope: str,
    expected_step: int,
    expected_schedule_hash: str,
    torch: Any,
) -> dict[str, Any]:
    if not state_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Scope checkpoint pair is incomplete: {state_path}, {manifest_path}")
    manifest = read_json_object(manifest_path)
    expected_values = {
        "scope": expected_scope,
        "global_update": expected_step,
        "schedule_sha256": expected_schedule_hash,
        "contains_optimizer_state": False,
        "contains_scheduler_state": False,
        "contains_frozen_parameters": False,
        "contains_full_model": False,
    }
    for key, value in expected_values.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Checkpoint manifest mismatch at {key}: {manifest.get(key)!r} != {value!r}")
    if manifest.get("byte_size") != state_path.stat().st_size or manifest.get("sha256") != sha256_file(state_path):
        raise RuntimeError(f"Checkpoint size/hash validation failed: {state_path}")
    names = manifest.get("trainable_parameter_names")
    if not isinstance(names, list) or len(names) != manifest.get("trainable_parameter_tensor_count"):
        raise RuntimeError("Checkpoint manifest trainable-name contract is invalid")
    loaded = _torch_load_state(state_path, torch)
    if tuple(loaded.keys()) != tuple(names) or set(loaded) != set(names):
        raise RuntimeError("Checkpoint loaded keys do not exactly equal manifest trainable names")
    del loaded
    return manifest


def validate_scope_completion(
    scope_dir: Path,
    scope: str,
    schedule_hash: str,
    *,
    torch: Any,
) -> dict[str, Any]:
    paths = _scope_paths(scope_dir)
    required = [
        paths["scope_report"],
        paths["metrics"],
        paths["trace"],
        paths["summary_json"],
        paths["summary_md"],
        paths["success"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Existing scope directory is incomplete; use a new run-id: missing={missing}")
    if SCOPE_SUCCESS_TOKENS[scope] not in paths["success"].read_text(encoding="utf-8"):
        raise RuntimeError(f"Scope success marker is invalid: {paths['success']}")
    summary = read_json_object(paths["summary_json"])
    if summary.get("status") != "OK" or summary.get("scope") != scope:
        raise RuntimeError(f"Scope summary is not complete: {paths['summary_json']}")
    trace_records = read_jsonl(paths["trace"])
    metrics_records = read_jsonl(paths["metrics"])
    if len(trace_records) != EXPECTED_MAX_STEPS or len(metrics_records) != EXPECTED_MAX_STEPS:
        raise RuntimeError(f"Completed scope {scope} does not contain exactly 320 trace/metric records")
    expected_updates = list(range(1, EXPECTED_MAX_STEPS + 1))
    if [row.get("global_update") for row in trace_records] != expected_updates:
        raise RuntimeError(f"Completed scope {scope} sampling trace updates are not contiguous")
    if [row.get("global_update") for row in metrics_records] != expected_updates:
        raise RuntimeError(f"Completed scope {scope} metric updates are not contiguous")
    trace_hash = sha256_file(paths["trace"])
    if summary.get("sampling_trace_sha256") != trace_hash or summary.get("schedule_sha256") != schedule_hash:
        raise RuntimeError(f"Completed scope {scope} summary hashes do not validate")
    manifests = []
    for step in EXPECTED_CHECKPOINT_STEPS:
        manifest = validate_scope_checkpoint(
            paths["checkpoints"] / f"step_{step}_scope_state.pt",
            paths["checkpoints"] / f"step_{step}_manifest.json",
            expected_scope=scope,
            expected_step=step,
            expected_schedule_hash=schedule_hash,
            torch=torch,
        )
        expected_trace_hash = sha256_jsonl_prefix(paths["trace"], step)
        if manifest.get("sampling_trace_sha256") != expected_trace_hash:
            raise RuntimeError(f"Scope {scope} step-{step} manifest does not match its sampling-trace prefix")
        manifests.append(manifest)
    actual_states = set(paths["checkpoints"].glob("*.pt"))
    expected_states = {paths["checkpoints"] / f"step_{step}_scope_state.pt" for step in EXPECTED_CHECKPOINT_STEPS}
    if actual_states != expected_states:
        raise RuntimeError(f"Scope {scope} has unexpected retained checkpoint binaries: {sorted(actual_states)}")
    return {"summary": summary, "checkpoint_manifests": manifests, "sampling_trace_sha256": trace_hash}


def validate_optimizer_membership(day5: Any, scope_root: Any, optimizer: Any) -> None:
    report = day5.optimizer_membership_report(scope_root, optimizer)
    if not report.get("optimizer_matches_trainable_exactly"):
        raise RuntimeError(f"Optimizer no longer exactly matches trainable parameters: {report}")
    if report.get("duplicate_optimizer_identity_count") or report.get("frozen_optimizer_overlap_count"):
        raise RuntimeError(f"Optimizer contains duplicate/frozen parameters: {report}")


def validate_and_clip_gradients(
    scope_root: Any,
    expected_names: Sequence[str],
    max_norm: float,
    torch: Any,
) -> tuple[float, bool]:
    validate_scope_integrity(scope_root, expected_names)
    trainable: list[Any] = []
    missing: list[str] = []
    frozen_with_gradients: list[str] = []
    for name, parameter in scope_root.named_parameters():
        if parameter.requires_grad:
            trainable.append(parameter)
            if parameter.grad is None:
                missing.append(name)
        elif parameter.grad is not None:
            frozen_with_gradients.append(name)
    if not trainable:
        raise RuntimeError("Training step has no trainable parameters")
    if missing:
        raise RuntimeError(f"Training step has empty gradients: {missing[:20]}")
    if frozen_with_gradients:
        raise RuntimeError(f"Frozen parameters received gradients: {frozen_with_gradients[:20]}")
    norm_value = torch.nn.utils.clip_grad_norm_(trainable, max_norm, error_if_nonfinite=True)
    gradient_norm = require_finite_scalar(float(norm_value.item()), "pre-clip gradient norm")
    if gradient_norm <= 0:
        raise RuntimeError("Training step has an all-zero gradient norm")
    return gradient_norm, gradient_norm > max_norm


def _cuda_memory(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _scope_summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# Phase 2N Week 2 {summary['scope']} Pilot",
            "",
            f"- Status: {summary['status']}",
            f"- Optimizer updates: {summary['optimizer_updates']}",
            f"- Peak learning rate: {summary['peak_learning_rate']}",
            f"- Trainable tensors: {summary['trainable_parameter_tensor_count']}",
            f"- Trainable parameters: {summary['trainable_parameter_numel']}",
            f"- Schedule SHA-256: {summary['schedule_sha256']}",
            f"- Sampling trace SHA-256: {summary['sampling_trace_sha256']}",
            "- Retained checkpoints contain trainable-scope model tensors only.",
            "- No validation or test inference ran during training.",
            "",
        ]
    )


def run_scope_training(
    *,
    scope: str,
    validated: ValidatedPilotConfiguration,
    run_dir: Path,
    schedule: Mapping[str, Any],
    day5_validated: Any,
    day5: Any,
    torch: Any,
) -> dict[str, Any]:
    from hy3dft.selective_training import (
        HISTORICAL_MODEL_KEYS as DAY4_MODEL_KEYS,
        ProtocolCorrectedDatasetFromJson,
        apply_trainable_scope,
        build_trainable_adamw,
        build_warmup_constant_scheduler,
        protocol_collate_fn,
    )

    if tuple(DAY4_MODEL_KEYS) != HISTORICAL_MODEL_KEYS:
        raise RuntimeError("Historical model-key contract changed")
    inactive_updates = [
        record["global_update"]
        for record in schedule["records"]
        if record.get("expected_mva_scale") != 1.0
    ]
    if inactive_updates:
        raise RuntimeError(
            f"Refusing model load because the schedule predicts inactive MVA: {inactive_updates[:20]}"
        )
    scope_dir = run_dir / scope
    if scope_dir.exists():
        completed = validate_scope_completion(scope_dir, scope, str(schedule["schedule_hash"]), torch=torch)
        print(f"scope={scope} status=SKIPPED_VALID_COMPLETE")
        print(SCOPE_SUCCESS_TOKENS[scope])
        return completed["summary"]
    scope_dir.mkdir(parents=True)
    paths = _scope_paths(scope_dir)
    paths["checkpoints"].mkdir()

    torch.cuda.reset_peak_memory_stats()
    day5._reset_runtime_seeds(torch, int(validated.values["base_seed"]))
    model, initialization = day5.initialize_true_pbr_model(day5_validated, scope_dir, torch)
    scope_root = model.unet
    scope_report_object = apply_trainable_scope(scope_root, scope)
    scope_report = scope_report_object.to_dict()
    expected_counts = EXPECTED_SCOPE_COUNTS[scope]
    if scope_report["trainable_parameter_tensor_count"] != expected_counts["tensor_count"]:
        raise RuntimeError(f"{scope} trainable tensor count drifted from Day 5")
    if scope_report["trainable_parameter_numel"] != expected_counts["numel"]:
        raise RuntimeError(f"{scope} trainable numel drifted from Day 5")
    expected_names = tuple(scope_report["trainable_parameter_names"])
    validate_scope_integrity(scope_root, expected_names)
    atomic_write_json(paths["scope_report"], scope_report, run_dir)

    peak_lr = float(validated.values["scopes"][scope]["learning_rate"])
    optimizer = build_trainable_adamw(scope_root, peak_lr)
    scheduler = build_warmup_constant_scheduler(optimizer, warmup_steps=int(validated.values["warmup_steps"]))
    validate_optimizer_membership(day5, scope_root, optimizer)

    dataset = ProtocolCorrectedDatasetFromJson(
        json_path=validated.train_json,
        image_size=int(validated.values["image_size"]),
        base_seed=int(validated.values["base_seed"]),
        rank=0,
        augmentation_mode=str(validated.values["augmentation_mode"]),
    )
    if len(dataset) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError(f"Protocol reader returned {len(dataset)} samples instead of 80")

    checkpoint_manifests: list[dict[str, Any]] = []
    loss_values: list[float] = []
    clipped_update_count = 0
    base_identifier = str(initialization.get("pretrained_model_name_or_path"))
    for schedule_record in schedule["records"]:
        update = int(schedule_record["global_update"])
        epoch = int(schedule_record["epoch"])
        dataset.set_epoch(epoch)
        sample = dataset[int(schedule_record["asset_index"])]
        batch = protocol_collate_fn([sample])
        names = list(batch["name"])
        if names != [schedule_record["asset_path"]]:
            raise RuntimeError(f"Scheduled sample mismatch at update {update}: {names}")
        metadata_records = batch.pop("protocol_metadata")
        if not isinstance(metadata_records, list) or len(metadata_records) != 1:
            raise RuntimeError(f"Protocol metadata batch is invalid at update {update}")
        protocol_metadata = metadata_records[0]
        trace_record = sampling_trace_record(schedule_record, protocol_metadata)
        model_batch = day5.move_model_batch_to_device(batch, torch.device("cuda"), torch)

        validate_scope_integrity(scope_root, expected_names)
        validate_optimizer_membership(day5, scope_root, optimizer)
        lr_before = float(optimizer.param_groups[0]["lr"])
        loss_value = day5.run_production_loss(
            model,
            model_batch,
            optimizer,
            seed=int(schedule_record["training_step_seed"]),
            backward=True,
            torch=torch,
        )
        loss_value = require_finite_scalar(loss_value, "production loss")
        gradient_norm, clipped = validate_and_clip_gradients(
            scope_root,
            expected_names,
            float(validated.values["gradient_clip_norm"]),
            torch,
        )
        optimizer.step()
        scheduler.step()
        lr_after = float(optimizer.param_groups[0]["lr"])
        validate_scope_integrity(scope_root, expected_names)
        validate_optimizer_membership(day5, scope_root, optimizer)
        torch.cuda.synchronize()

        loss_values.append(loss_value)
        clipped_update_count += int(clipped)
        append_jsonl(paths["trace"], trace_record, run_dir)
        metric_record = {
            "scope": scope,
            "global_update": update,
            "epoch": epoch,
            "within_epoch_position": schedule_record["within_epoch_position"],
            "asset_index": schedule_record["asset_index"],
            "asset_id": schedule_record["asset_id"],
            "asset_path": schedule_record["asset_path"],
            "original_candidate_seed": schedule_record["original_candidate_seed"],
            "training_step_seed": schedule_record["training_step_seed"],
            "expected_mva_scale": schedule_record["expected_mva_scale"],
            "expected_ref_scale": schedule_record["expected_ref_scale"],
            "mva_seed_retry_count": schedule_record["mva_seed_retry_count"],
            "selected_reference_view": trace_record["selected_reference_view"],
            "reference_lighting_pair": trace_record["reference_lighting_pair"],
            "target_view_order": trace_record["target_view_order"],
            "spatial_augmentation": trace_record["spatial_augmentation"],
            "loss": loss_value,
            "learning_rate_before_optimizer_step": lr_before,
            "learning_rate_after_scheduler_step": lr_after,
            "pre_clip_gradient_norm": gradient_norm,
            "gradient_clip_norm": float(validated.values["gradient_clip_norm"]),
            "gradient_clipping_applied": clipped,
            "cuda_memory": _cuda_memory(torch),
        }
        append_jsonl(paths["metrics"], metric_record, run_dir)

        if update in EXPECTED_CHECKPOINT_STEPS:
            trace_hash = sha256_file(paths["trace"])
            manifest = save_scope_checkpoint(
                scope_root=scope_root,
                scope=scope,
                scope_report=scope_report,
                step=update,
                checkpoints_dir=paths["checkpoints"],
                run_dir=run_dir,
                schedule_hash=str(schedule["schedule_hash"]),
                sampling_trace_hash=trace_hash,
                learning_rate=lr_after,
                base_identifier=base_identifier,
                torch=torch,
            )
            checkpoint_manifests.append(manifest)
        if update % int(validated.values["log_every_n_steps"]) == 0 or update == 1:
            print(
                f"scope={scope} update={update}/{EXPECTED_MAX_STEPS} epoch={epoch} "
                f"asset={schedule_record['asset_id']} loss={loss_value:.8g} lr={lr_before:.8g} "
                f"grad_norm={gradient_norm:.8g}"
            )
        del model_batch, batch, sample

    trace_hash = sha256_file(paths["trace"])
    summary = {
        "phase": validated.values["phase"],
        "status": "OK",
        "scope": scope,
        "optimizer_updates": EXPECTED_MAX_STEPS,
        "epoch_count": 4,
        "peak_learning_rate": peak_lr,
        "warmup_steps": int(validated.values["warmup_steps"]),
        "gradient_clip_norm": float(validated.values["gradient_clip_norm"]),
        "clipped_update_count": clipped_update_count,
        "mean_loss": sum(loss_values) / len(loss_values),
        "minimum_loss": min(loss_values),
        "maximum_loss": max(loss_values),
        "trainable_parameter_tensor_count": scope_report["trainable_parameter_tensor_count"],
        "trainable_parameter_numel": scope_report["trainable_parameter_numel"],
        "base_identifier": base_identifier,
        "fresh_base_loaded": True,
        "schedule_sha256": schedule["schedule_hash"],
        "sampling_trace_sha256": trace_hash,
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
        "checkpoint_paths": [manifest["checkpoint_path"] for manifest in checkpoint_manifests],
        "checkpoint_contains_optimizer": False,
        "checkpoint_contains_full_model": False,
        "test_data_used": False,
        "validation_or_inference_during_training": False,
        "final_cuda_memory": _cuda_memory(torch),
    }
    atomic_write_json(paths["summary_json"], summary, run_dir)
    atomic_write_text(paths["summary_md"], _scope_summary_markdown(summary), run_dir)
    atomic_write_text(paths["success"], SCOPE_SUCCESS_TOKENS[scope] + "\n", run_dir)
    print(SCOPE_SUCCESS_TOKENS[scope])

    del dataset, scheduler, optimizer, scope_root, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return summary


def _final_summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 2N Week 2 Controlled Pilot Training",
            "",
            f"- Status: {summary['status']}",
            f"- Run ID: {summary['run_id']}",
            f"- Scope order: {', '.join(summary['scope_order'])}",
            f"- Optimizer updates per scope: {summary['optimizer_updates_per_scope']}",
            f"- Schedule SHA-256: {summary['schedule_sha256']}",
            f"- Shared sampling trace SHA-256: {summary['sampling_trace_sha256']}",
            "- PC-S1 and PC-Full used identical asset, reference-view, and lighting decisions.",
            "- No validation/test inference ran during training.",
            "",
        ]
    )


def finalize_run(
    run_dir: Path,
    validated: ValidatedPilotConfiguration,
    schedule: Mapping[str, Any],
    *,
    torch: Any,
) -> dict[str, Any]:
    scope_results = {
        scope: validate_scope_completion(run_dir / scope, scope, str(schedule["schedule_hash"]), torch=torch)
        for scope in EXPECTED_SCOPE_ORDER
    }
    comparison_dir = run_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    comparison_path = comparison_dir / "trace_equivalence.json"
    comparison = compare_sampling_traces(
        run_dir / "pc_s1" / "sampling_trace.jsonl",
        run_dir / "pc_full" / "sampling_trace.jsonl",
    )
    if comparison_path.exists():
        if read_json_object(comparison_path) != comparison:
            raise RuntimeError("Existing trace-equivalence report differs from recomputed evidence")
    else:
        atomic_write_json(comparison_path, comparison, run_dir)
    print(TRACE_SUCCESS_TOKEN)

    summary = {
        "phase": validated.values["phase"],
        "status": "OK",
        "run_id": run_dir.name,
        "scope_order": list(EXPECTED_SCOPE_ORDER),
        "optimizer_updates_per_scope": EXPECTED_MAX_STEPS,
        "schedule_sha256": schedule["schedule_hash"],
        "sampling_trace_sha256": comparison["pc_s1_trace_sha256"],
        "trace_equivalence": comparison,
        "scope_summaries": {scope: result["summary"] for scope, result in scope_results.items()},
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
        "test_data_used": False,
        "selection_manifest_frozen_before_training": str(validated.eval_manifest),
    }
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    success = run_dir / "_SUCCESS"
    if summary_json.exists():
        if read_json_object(summary_json) != summary:
            raise RuntimeError("Existing final summary differs from validated completed scopes")
    else:
        atomic_write_json(summary_json, summary, run_dir)
        atomic_write_text(summary_md, _final_summary_markdown(summary), run_dir)
        atomic_write_text(success, FINAL_SUCCESS_TOKEN + "\n", run_dir)
    if not summary_md.is_file() or not success.is_file() or FINAL_SUCCESS_TOKEN not in success.read_text(encoding="utf-8"):
        raise RuntimeError("Final Week 2 success output is incomplete")
    print(FINAL_SUCCESS_TOKEN)
    return summary


def run_training(
    validated: ValidatedPilotConfiguration,
    *,
    run_id: str,
    scopes: Sequence[str],
) -> Path:
    policy = validated.values["conditioning_dropout_policy"]
    source_audit = validate_conditioning_source_audit()
    schedule = build_training_schedule(
        validated.train_sample_paths,
        conditioning_dropout_policy=policy,
        conditioning_source_audit=source_audit,
        schedule_seed=int(validated.values["schedule_seed"]),
        max_steps=int(validated.values["max_steps"]),
    )
    validate_training_schedule(
        schedule,
        validated.train_sample_paths,
        conditioning_dropout_policy=policy,
        conditioning_source_audit=source_audit,
    )

    _prepend_runtime_paths(validated.project_root)
    import torch

    day5 = _load_day5_runtime_module()
    day5_validated = day5.validate_config(validated.project_root / DAY5_CONFIG_RELATIVE, validated.project_root)
    if tuple(day5_validated.train_sample_paths) != tuple(validated.train_sample_paths):
        raise RuntimeError("Day 5 and Week 2 training sample paths differ")
    day5._prepend_runtime_paths(day5_validated)
    runtime_device = validate_runtime_device(torch, float(validated.values["minimum_gpu_memory_gib"]))
    validate_free_disk(validated.output_root, float(validated.values["minimum_free_disk_gib"]))
    torch.set_float32_matmul_precision("medium")

    safe_run_id = _safe_run_id(run_id)
    run_dir = (validated.output_root / safe_run_id).resolve()
    if not is_relative_to(run_dir, validated.output_root):
        raise ValueError(f"Run directory escapes Week 2 output root: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = ensure_schedule_file(run_dir, schedule, validated.train_sample_paths)
    schedule = read_json_object(schedule_path)
    validate_training_schedule(
        schedule,
        validated.train_sample_paths,
        conditioning_dropout_policy=policy,
        conditioning_source_audit=source_audit,
    )
    ensure_runtime_manifest(run_dir, validated, schedule, runtime_device)

    requested = tuple(scopes)
    if not requested or any(scope not in EXPECTED_SCOPE_ORDER for scope in requested):
        raise ValueError(f"Runtime scopes must be drawn from {EXPECTED_SCOPE_ORDER}")
    for scope in requested:
        run_scope_training(
            scope=scope,
            validated=validated,
            run_dir=run_dir,
            schedule=schedule,
            day5_validated=day5_validated,
            day5=day5,
            torch=torch,
        )
    if all((run_dir / scope).is_dir() for scope in EXPECTED_SCOPE_ORDER):
        finalize_run(run_dir, validated, schedule, torch=torch)
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or run deterministic Phase 2N Week 2 PC-S1/PC-Full pilot training."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Week 2 pilot training JSON")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="Validate contracts without Torch, CUDA, or Hunyuan")
    modes.add_argument("--run-all", action="store_true", help="Run PC-S1 then PC-Full on the shared schedule")
    modes.add_argument("--scope", choices=EXPECTED_SCOPE_ORDER, help="Run one scope for staged recovery")
    parser.add_argument("--run-id", help="Required runtime output ID; normally derived from SLURM_JOB_ID")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        if args.run_id:
            raise ValueError("--run-id is only valid with --run-all or --scope")
        run_check_only(args.config)
        return 0
    if not args.run_id:
        raise ValueError("--run-all and --scope require --run-id")
    validated = validate_training_config(args.config)
    scopes = EXPECTED_SCOPE_ORDER if args.run_all else (args.scope,)
    run_dir = run_training(validated, run_id=args.run_id, scopes=scopes)
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
