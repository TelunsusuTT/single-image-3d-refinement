#!/usr/bin/env python3
"""Phase 2N Day 5 real-Hunyuan A100 runtime preflight.

The module is deliberately standard-library-only at import time. Check-only
mode validates paths and contracts without importing Torch, Lightning, or
Hunyuan. Run-audit mode is intended exclusively for the accompanying A100
Slurm job.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase2n_day5_a100_preflight.json"
EXPECTED_OUTPUT_RELATIVE = Path("outputs/phase2n/day5_a100_preflight")
EXPECTED_TRAIN_COUNT = 80
HISTORICAL_MODEL_KEYS = (
    "images_cond",
    "images_albedo",
    "images_mr",
    "images_normal",
    "images_position",
    "name",
)
REQUIRED_DAY4_APIS = {
    "ProtocolCorrectedDataModule",
    "apply_trainable_scope",
    "build_trainable_adamw",
    "build_warmup_constant_scheduler",
    "warmup_constant_multiplier",
}
REQUIRED_RUNTIME_TOKENS = (
    "PHASE2N_DAY5_MODEL_LOAD_OK",
    "PHASE2N_DAY5_REAL_BATCH_LOSS_OK",
    "PHASE2N_DAY5_PC_S1_SCOPE_OK",
    "PHASE2N_DAY5_PC_S1_GRADIENT_OK",
    "PHASE2N_DAY5_PC_S1_UPDATE_OK",
    "PHASE2N_DAY5_PC_S1_CHECKPOINT_RELOAD_OK",
    "PHASE2N_DAY5_PC_FULL_SCOPE_OK",
    "PHASE2N_DAY5_PC_FULL_GRADIENT_OK",
    "PHASE2N_DAY5_PC_FULL_UPDATE_OK",
    "PHASE2N_DAY5_PC_FULL_AUDIT_OK",
    "PHASE2N_DAY5_A100_PREFLIGHT_OK",
)
CHECK_ONLY_TOKEN = "PHASE2N_DAY5_PREFLIGHT_READINESS_OK"


@dataclass(frozen=True)
class ValidatedConfiguration:
    values: dict[str, Any]
    config_path: Path
    project_root: Path
    train_json: Path
    output_root: Path
    historical_train_config: Path
    official_hypaint: Path
    true_pbr_model_dir: Path
    train_sample_paths: tuple[Path, ...]


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


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_exact(config: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config {key!r} must be {expected!r}, got {actual!r}")


def _require_positive_number(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Config {key!r} must be a positive number")
    return float(value)


def validate_output_root(output_root: Path, project_root: Path) -> Path:
    expected = (project_root / EXPECTED_OUTPUT_RELATIVE).resolve()
    resolved = output_root.expanduser().resolve()
    if resolved != expected:
        raise ValueError(f"Output root must be exactly {expected}, got {resolved}")
    if not is_relative_to(resolved, project_root / "outputs" / "phase2n"):
        raise ValueError(f"Output root is outside outputs/phase2n: {resolved}")
    if is_relative_to(resolved, project_root / "data") or is_relative_to(resolved, project_root / "checkpoints"):
        raise ValueError(f"Unsafe output root: {resolved}")
    return resolved


def validate_train_json_contract(path: Path, expected_count: int = EXPECTED_TRAIN_COUNT) -> tuple[Path, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Training examples JSON does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training examples JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Training examples JSON must contain a list")
    if len(payload) != expected_count:
        raise ValueError(f"Training examples JSON must contain exactly {expected_count} paths, got {len(payload)}")
    paths: list[Path] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str) or not item:
            raise ValueError(f"Training examples item {index} must be a non-empty path string")
        sample_path = Path(item).expanduser()
        if not sample_path.is_absolute():
            raise ValueError(f"Training examples item {index} is not absolute: {item}")
        sample_path = sample_path.resolve()
        if not sample_path.is_dir():
            raise FileNotFoundError(f"Training sample directory does not exist at index {index}: {sample_path}")
        paths.append(sample_path)
    if len(set(paths)) != len(paths):
        raise ValueError("Training examples JSON contains duplicate sample directories")
    return tuple(paths)


def _yaml_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    if match is None:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def inspect_historical_initialization_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not re.search(r"^\s*target:\s*hunyuanpaintpbr\.HunyuanPaint\s*$", text, flags=re.MULTILINE):
        raise ValueError(f"Historical config does not target hunyuanpaintpbr.HunyuanPaint: {path}")
    pretrained = _yaml_scalar(text, "pretrained_model_name_or_path")
    custom_pipeline = _yaml_scalar(text, "custom_pipeline")
    noise_channels = _yaml_scalar(text, "noise_in_channels")
    resume_from = _yaml_scalar(text, "resume_from")
    init_control_from = _yaml_scalar(text, "init_control_from")
    if not pretrained:
        raise ValueError("Historical config is missing pretrained_model_name_or_path")
    if custom_pipeline != "./hunyuanpaintpbr":
        raise ValueError(f"Historical custom_pipeline must be './hunyuanpaintpbr', got {custom_pipeline!r}")
    if noise_channels != "12":
        raise ValueError(f"Historical noise_in_channels must be 12, got {noise_channels!r}")
    if resume_from not in {"null", "None", "~"}:
        raise ValueError(f"Historical resume_from must be null, got {resume_from!r}")
    if init_control_from not in {"null", "None", "~"}:
        raise ValueError(f"Historical init_control_from must be null, got {init_control_from!r}")
    model_dir = Path(pretrained).expanduser().resolve()
    if model_dir.name != "hunyuan3d-paintpbr-v2-1":
        raise ValueError(f"Historical base is not the official PBR pipeline directory: {model_dir}")
    required = (
        model_dir / "model_index.json",
        model_dir / "unet" / "config.json",
        model_dir / "scheduler" / "scheduler_config.json",
        model_dir / "vae" / "config.json",
        model_dir / "text_encoder" / "config.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Official true-PBR initialization files are missing: {missing}")
    return {
        "model_target": "hunyuanpaintpbr.HunyuanPaint",
        "pretrained_model_name_or_path": str(model_dir),
        "custom_pipeline": custom_pipeline,
        "noise_in_channels": 12,
        "resume_from": None,
        "init_control_from": None,
        "required_pipeline_files": [str(item) for item in required],
    }


def inspect_python_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def validate_config(config_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> ValidatedConfiguration:
    root = Path(project_root).expanduser().resolve()
    path = resolve_project_path(config_path, root)
    config = read_json_object(path)

    _require_exact(config, "phase", "phase2n_day5_a100_preflight")
    _require_exact(config, "sample_index", 0)
    _require_exact(config, "batch_size", 1)
    _require_exact(config, "num_workers", 0)
    _require_exact(config, "image_size", 512)
    _require_exact(config, "base_seed", 42)
    _require_exact(config, "epoch", 0)
    _require_exact(config, "augmentation_mode", "none")
    _require_exact(config, "warmup_steps", 50)
    _require_exact(config, "scope_order", ["pc_s1", "pc_full"])
    _require_exact(config, "expected_train_count", EXPECTED_TRAIN_COUNT)

    learning_rates = config.get("candidate_peak_learning_rates")
    if not isinstance(learning_rates, dict):
        raise ValueError("candidate_peak_learning_rates must be an object")
    for scope, expected in (("pc_s1", 1e-6), ("pc_full", 5e-7)):
        value = learning_rates.get(scope)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected:
            raise ValueError(f"Candidate peak LR for {scope} must be {expected}, got {value!r}")

    snapshots = config.get("exact_update_snapshot")
    if not isinstance(snapshots, dict):
        raise ValueError("exact_update_snapshot must be an object")
    pc_s1_snapshot = snapshots.get("pc_s1")
    pc_full_snapshot = snapshots.get("pc_full")
    if not isinstance(pc_s1_snapshot, dict) or pc_s1_snapshot.get("mode") != "all_if_cpu_memory_permits":
        raise ValueError("PC-S1 snapshot mode must be all_if_cpu_memory_permits")
    if not isinstance(pc_full_snapshot, dict) or pc_full_snapshot.get("mode") != "deterministic_bounded":
        raise ValueError("PC-Full snapshot mode must be deterministic_bounded")
    if int(pc_full_snapshot.get("max_tensors", 0)) > 16 or int(pc_full_snapshot.get("max_tensors", 0)) <= 0:
        raise ValueError("PC-Full max_tensors must be in [1, 16]")
    if int(pc_s1_snapshot.get("cpu_memory_limit_bytes", 0)) <= 0:
        raise ValueError("PC-S1 cpu_memory_limit_bytes must be positive")
    if int(pc_s1_snapshot.get("fallback_max_tensors", 0)) > 16:
        raise ValueError("PC-S1 fallback_max_tensors must be at most 16")

    checkpoint = config.get("pc_s1_smoke_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("pc_s1_smoke_checkpoint must be an object")
    expected_checkpoint_flags = {
        "save_selected_parameters": True,
        "save_optimizer_state": True,
        "save_scheduler_state": True,
        "delete_binary_after_success": True,
    }
    for key, expected in expected_checkpoint_flags.items():
        if checkpoint.get(key) is not expected:
            raise ValueError(f"pc_s1_smoke_checkpoint.{key} must be true")
    checkpoint_name = checkpoint.get("temporary_filename")
    if checkpoint_name != "_temporary_pc_s1_selective_checkpoint.pt":
        raise ValueError("Unexpected PC-S1 temporary checkpoint filename")
    if "/" in checkpoint_name or "\\" in checkpoint_name:
        raise ValueError("Temporary checkpoint filename must not contain directories")

    _require_positive_number(config, "gradient_clip_norm")
    _require_positive_number(config, "update_ratio_safety_ceiling")
    tolerance = config.get("zero_update_tolerance")
    if not isinstance(tolerance, dict):
        raise ValueError("zero_update_tolerance must be an object")
    for key in ("absolute", "relative"):
        value = tolerance.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"zero_update_tolerance.{key} must be non-negative")
    _require_exact(config, "scheduler_evidence_steps", [0, 25, 50, 160, 320])

    train_json = resolve_project_path(config.get("train_json", ""), root)
    train_paths = validate_train_json_contract(train_json, EXPECTED_TRAIN_COUNT)
    output_root = validate_output_root(resolve_project_path(config.get("output_root", ""), root), root)
    historical_config = resolve_project_path(config.get("historical_train_config", ""), root)
    if not historical_config.is_file():
        raise FileNotFoundError(f"Historical training config is missing: {historical_config}")
    initialization = inspect_historical_initialization_config(historical_config)
    true_pbr_model_dir = Path(initialization["pretrained_model_name_or_path"])

    official_hypaint = Path(str(config.get("official_hypaint", ""))).expanduser().resolve()
    official_files = (
        official_hypaint / "train.py",
        official_hypaint / "hunyuanpaintpbr" / "unet" / "model.py",
        official_hypaint / "hunyuanpaintpbr" / "pipeline.py",
        official_hypaint / "src" / "utils" / "train_util.py",
    )
    missing_official = [str(item) for item in official_files if not item.is_file()]
    if missing_official:
        raise FileNotFoundError(f"Official Hunyuan initialization files are missing: {missing_official}")

    dino_cache = root / "caches" / "hf" / "hub" / "models--facebook--dinov2-giant"
    if not dino_cache.is_dir():
        raise FileNotFoundError(f"Local DINO cache required by HunyuanPaint is missing: {dino_cache}")

    selective_source = root / "src" / "hy3dft" / "selective_training.py"
    protocol_source = root / "src" / "hy3dft" / "protocol_corrected_dataset.py"
    for source in (selective_source, protocol_source):
        if not source.is_file():
            raise FileNotFoundError(f"Required Phase 2N source is missing: {source}")
    symbols = inspect_python_symbols(selective_source)
    missing_apis = sorted(REQUIRED_DAY4_APIS - symbols)
    if missing_apis:
        raise RuntimeError(f"Day 4 APIs are missing: {missing_apis}")

    return ValidatedConfiguration(
        values=config,
        config_path=path,
        project_root=root,
        train_json=train_json,
        output_root=output_root,
        historical_train_config=historical_config,
        official_hypaint=official_hypaint,
        true_pbr_model_dir=true_pbr_model_dir,
        train_sample_paths=train_paths,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_report(path: Path, payload: Mapping[str, Any], allowed_root: Path, *, overwrite: bool = False) -> Path:
    destination = path.expanduser().resolve()
    root = allowed_root.expanduser().resolve()
    if not is_relative_to(destination, root):
        raise ValueError(f"Report path is outside the run directory: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def deterministic_bounded_names(names: Sequence[str], limit: int) -> list[str]:
    ordered = sorted(set(names))
    if limit <= 0:
        raise ValueError("Snapshot limit must be positive")
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indices]


def select_update_snapshot_names(
    named_parameters: Iterable[tuple[str, Any]],
    scope: str,
    *,
    pc_s1_cpu_limit_bytes: int,
    pc_s1_fallback_limit: int,
    pc_full_limit: int,
) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in named_parameters if bool(parameter.requires_grad)]
    ordered_names = sorted(name for name, _parameter in trainable)
    bytes_by_name = {
        name: int(parameter.numel()) * int(parameter.element_size()) for name, parameter in trainable
    }
    total_bytes = sum(bytes_by_name.values())
    if scope == "pc_s1" and total_bytes <= pc_s1_cpu_limit_bytes:
        selected = ordered_names
        rule = "all_trainable_names_sorted_cpu_budget_permitted"
    elif scope == "pc_s1":
        selected = deterministic_bounded_names(ordered_names, pc_s1_fallback_limit)
        rule = "evenly_spaced_over_sorted_trainable_names_cpu_budget_fallback"
    elif scope == "pc_full":
        selected = deterministic_bounded_names(ordered_names, pc_full_limit)
        rule = "evenly_spaced_over_sorted_trainable_names"
    else:
        raise ValueError(f"Unsupported scope for snapshots: {scope}")
    return {
        "scope": scope,
        "sampling_rule": rule,
        "trainable_name_count": len(ordered_names),
        "trainable_cpu_snapshot_bytes": total_bytes,
        "selected_name_count": len(selected),
        "selected_names": selected,
        "all_trainable_selected": selected == ordered_names,
    }


def snapshot_parameters(model: Any, names: Sequence[str]) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise KeyError(f"Snapshot parameter names are missing: {missing[:10]}")
    return {name: parameters[name].detach().cpu().clone() for name in names}


def calculate_update_ratios(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if set(before) != set(after):
        raise ValueError("Before/after snapshot names differ")
    rows: list[dict[str, Any]] = []
    for name in sorted(before):
        old = before[name].detach().float().cpu()
        new = after[name].detach().float().cpu()
        if old.shape != new.shape:
            raise ValueError(f"Snapshot shape changed for {name}: {old.shape} -> {new.shape}")
        delta = new - old
        weight_norm = float(torch.linalg.vector_norm(old).item())
        update_norm = float(torch.linalg.vector_norm(delta).item())
        ratio = update_norm / max(weight_norm, 1e-12)
        finite = all(math.isfinite(value) for value in (weight_norm, update_norm, ratio))
        rows.append(
            {
                "name": name,
                "weight_norm": weight_norm,
                "update_norm": update_norm,
                "update_to_weight_ratio": ratio,
                "finite": finite,
                "nonzero_update": update_norm > 0.0,
            }
        )
    ratios = [row["update_to_weight_ratio"] for row in rows]
    return {
        "sampled_tensor_count": len(rows),
        "finite_tensor_count": sum(row["finite"] for row in rows),
        "nonzero_update_count": sum(row["nonzero_update"] for row in rows),
        "max_update_to_weight_ratio": max(ratios, default=0.0),
        "mean_update_to_weight_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "per_tensor": rows,
    }


def validate_update_report(report: Mapping[str, Any], safety_ceiling: float) -> None:
    if report.get("sampled_tensor_count", 0) <= 0:
        raise RuntimeError("Update audit sampled zero tensors")
    if report.get("finite_tensor_count") != report.get("sampled_tensor_count"):
        raise RuntimeError("Update audit contains NaN or Inf")
    if report.get("nonzero_update_count", 0) <= 0:
        raise RuntimeError("Update audit observed zero updates")
    maximum = float(report.get("max_update_to_weight_ratio", float("inf")))
    if maximum > safety_ceiling:
        raise RuntimeError(f"Update-to-weight ratio {maximum:.6g} exceeds safety ceiling {safety_ceiling:.6g}")


def classify_gradients(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    import torch

    trainable_rows: list[dict[str, Any]] = []
    frozen_with_gradients: list[str] = []
    total_squared = 0.0
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if not parameter.requires_grad:
            if gradient is not None:
                frozen_with_gradients.append(name)
            continue
        if gradient is None:
            row = {"name": name, "has_gradient": False, "finite": True, "nonzero": False, "gradient_norm": 0.0}
        else:
            detached = gradient.detach().float()
            finite = bool(torch.isfinite(detached).all().item())
            norm = float(torch.linalg.vector_norm(detached).item()) if finite else float("inf")
            nonzero = finite and bool(torch.count_nonzero(detached).item())
            row = {
                "name": name,
                "has_gradient": True,
                "finite": finite,
                "nonzero": nonzero,
                "gradient_norm": norm,
            }
            if finite:
                total_squared += norm * norm
        trainable_rows.append(row)
    return {
        "trainable_tensor_count": len(trainable_rows),
        "finite_gradient_count": sum(row["finite"] for row in trainable_rows),
        "nonzero_gradient_count": sum(row["nonzero"] for row in trainable_rows),
        "missing_gradient_count": sum(not row["has_gradient"] for row in trainable_rows),
        "total_gradient_norm": math.sqrt(total_squared),
        "frozen_tensors_with_gradients": sorted(frozen_with_gradients),
        "per_trainable_tensor": trainable_rows,
    }


def validate_gradient_report(report: Mapping[str, Any]) -> None:
    count = int(report.get("trainable_tensor_count", 0))
    if count <= 0:
        raise RuntimeError("Gradient audit found zero trainable tensors")
    if int(report.get("finite_gradient_count", 0)) != count:
        raise RuntimeError("Gradient audit contains NaN or Inf")
    if int(report.get("nonzero_gradient_count", 0)) <= 0:
        raise RuntimeError("All selected gradients are zero")
    frozen = report.get("frozen_tensors_with_gradients", [])
    if frozen:
        raise RuntimeError(f"Frozen tensors received gradients: {list(frozen)[:20]}")


def checkpoint_file_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "temporary_binary_path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "binary_removed_after_success": False,
    }


def cleanup_temporary_checkpoint(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_path = Path(str(manifest.get("temporary_binary_path", ""))).resolve()
    if path.resolve() != expected_path:
        raise ValueError("Checkpoint cleanup path does not match manifest")
    if path.stat().st_size != manifest.get("byte_size") or sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError("Temporary checkpoint changed before cleanup")
    path.unlink()
    result = dict(manifest)
    result["binary_removed_after_success"] = not path.exists()
    if not result["binary_removed_after_success"]:
        raise RuntimeError(f"Temporary checkpoint was not removed: {path}")
    return result


def git_snapshot(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    return {"root": str(repo.resolve()), "head": run("rev-parse", "HEAD"), "status_short": run("status", "--short")}


def _source_hashes(validated: ValidatedConfiguration) -> dict[str, dict[str, Any]]:
    paths = (
        validated.config_path,
        Path(__file__).resolve(),
        validated.historical_train_config,
        validated.project_root / "src" / "hy3dft" / "protocol_corrected_dataset.py",
        validated.project_root / "src" / "hy3dft" / "selective_training.py",
        validated.official_hypaint / "train.py",
        validated.official_hypaint / "hunyuanpaintpbr" / "unet" / "model.py",
        validated.official_hypaint / "hunyuanpaintpbr" / "pipeline.py",
    )
    return {
        str(path): {"sha256": sha256_file(path), "byte_size": path.stat().st_size} for path in paths
    }


def run_check_only(config_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> ValidatedConfiguration:
    validated = validate_config(config_path, project_root)
    print(f"config={validated.config_path}")
    print(f"train_json={validated.train_json}")
    print(f"train_count={len(validated.train_sample_paths)}")
    print(f"sample_index={validated.values['sample_index']}")
    print(f"historical_train_config={validated.historical_train_config}")
    print(f"true_pbr_model_dir={validated.true_pbr_model_dir}")
    print(f"output_root={validated.output_root}")
    print(f"scope_order={','.join(validated.values['scope_order'])}")
    print(CHECK_ONLY_TOKEN)
    return validated


def _prepend_runtime_paths(validated: ValidatedConfiguration) -> None:
    hy21 = validated.official_hypaint.parent
    for path in (validated.project_root / "src", validated.official_hypaint, hy21):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _reset_runtime_seeds(torch: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError as exc:  # pragma: no cover - runtime environment contract.
        raise RuntimeError("NumPy is required by the official production loss") from exc
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _expand_noise_conv_like_official_train(model: Any, config: Any, torch: Any) -> dict[str, Any]:
    model_unet = model.unet.unet
    nested = False
    if hasattr(model_unet, "unet"):
        model_unet = model_unet.unet
        nested = True
    noise_in_channels = config.model.params.get("noise_in_channels", None)
    if noise_in_channels is None:
        raise RuntimeError("Historical config lost noise_in_channels")
    old_conv = model_unet.conv_in
    old_channels = int(old_conv.in_channels)
    with torch.no_grad():
        new_conv = torch.nn.Conv2d(
            int(noise_in_channels),
            old_conv.out_channels,
            old_conv.kernel_size,
            old_conv.stride,
            old_conv.padding,
        )
        new_conv.weight.zero_()
        copy_channels = min(old_channels, int(noise_in_channels))
        new_conv.weight[:, :copy_channels].copy_(old_conv.weight[:, :copy_channels])
        if old_conv.bias is not None:
            new_conv.bias.zero_()
            new_conv.bias.copy_(old_conv.bias)
        model_unet.conv_in = new_conv
    return {
        "nested_unet": nested,
        "original_conv_in_channels": old_channels,
        "configured_noise_in_channels": int(noise_in_channels),
        "final_conv_in_channels": int(model_unet.conv_in.in_channels),
        "copied_input_channels": copy_channels,
        "implementation": "official_train_py_noise_in_channels_lifecycle",
    }


def initialize_true_pbr_model(validated: ValidatedConfiguration, run_dir: Path, torch: Any) -> tuple[Any, dict[str, Any]]:
    from omegaconf import OmegaConf
    from src.utils.train_util import instantiate_from_config

    config = OmegaConf.load(str(validated.historical_train_config))
    for key in ("init_unet_from", "init_vae_from", "init_control_from", "resume_from"):
        if getattr(config, key, None) not in (None, "null"):
            raise RuntimeError(f"Day 5 refuses unexpected historical initialization override {key}")
    previous_cwd = Path.cwd()
    try:
        os.chdir(validated.official_hypaint)
        model = instantiate_from_config(config.model)
    finally:
        os.chdir(previous_cwd)
    conv_metadata = _expand_noise_conv_like_official_train(model, config, torch)
    model.logdir = str(run_dir)
    model.learning_rate = float(config.model.base_learning_rate)
    model.pipeline.to(torch.device("cuda"))
    model.to(torch.device("cuda"))
    model.train()
    torch.cuda.synchronize()
    scope_root = model.unet
    trainable = [(name, parameter) for name, parameter in scope_root.named_parameters() if parameter.requires_grad]
    metadata = {
        "historical_train_config": str(validated.historical_train_config),
        "model_target": str(config.model.target),
        "pretrained_model_name_or_path": str(config.model.params.stable_diffusion_config.pretrained_model_name_or_path),
        "custom_pipeline": str(config.model.params.stable_diffusion_config.custom_pipeline),
        "resume_from": None,
        "init_control_from": None,
        "scope_root": "model.unet",
        "precision": "bf16-mixed-autocast",
        "conv_in": conv_metadata,
        "official_initial_trainable_tensor_count": len(trainable),
        "official_initial_trainable_numel": sum(int(parameter.numel()) for _name, parameter in trainable),
    }
    return model, metadata


def _install_training_step_audit_hooks(model: Any, optimizer: Any) -> None:
    model.log = types.MethodType(lambda _self, *_args, **_kwargs: None, model)
    model.log_dict = types.MethodType(lambda _self, *_args, **_kwargs: None, model)
    model.optimizers = types.MethodType(lambda _self: optimizer, model)
    model._trainer = types.SimpleNamespace(global_step=0)


def run_production_loss(
    model: Any,
    batch: Mapping[str, Any],
    optimizer: Any,
    *,
    seed: int,
    backward: bool,
    torch: Any,
) -> float:
    _install_training_step_audit_hooks(model, optimizer)
    _reset_runtime_seeds(torch, seed)
    if backward:
        optimizer.zero_grad(set_to_none=True)
    gradient_context = torch.enable_grad() if backward else torch.no_grad()
    with gradient_context:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.training_step(dict(batch), 0)
    if not hasattr(loss, "detach") or loss.numel() != 1:
        raise RuntimeError(f"Official training_step did not return a scalar tensor: {type(loss).__name__}")
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise RuntimeError("Official production loss is NaN or Inf")
    if backward:
        loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach().float().cpu().item())


class _OptimizerView:
    def __init__(self, learning_rate: float = 0.0) -> None:
        self.param_groups = [{"lr": float(learning_rate)}]


def make_zero_update_report(before: float, after: float, tolerance: Mapping[str, Any]) -> dict[str, Any]:
    absolute = abs(after - before)
    relative = absolute / max(abs(before), 1e-12)
    allowed = float(tolerance["absolute"]) + float(tolerance["relative"]) * abs(before)
    return {
        "loss_before_scope": before,
        "loss_after_scope": after,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "absolute_tolerance": float(tolerance["absolute"]),
        "relative_tolerance": float(tolerance["relative"]),
        "combined_allowed_difference": allowed,
        "within_tolerance": absolute <= allowed,
        "bitwise_equality_claimed": False,
    }


def _validate_pc_s1_names(names: Sequence[str]) -> None:
    allowed = (
        ".attn_multiview.to_q.",
        ".attn_multiview.to_k.",
        ".attn_multiview.to_v.",
        ".attn_multiview.to_out.0.",
    )
    invalid = [name for name in names if not any(token in f".{name}" for token in allowed)]
    if invalid:
        raise RuntimeError(f"PC-S1 selected names outside the exact multiview projections: {invalid[:20]}")


def optimizer_membership_report(model: Any, optimizer: Any) -> dict[str, Any]:
    named = list(model.named_parameters())
    trainable_ids = {id(parameter) for _name, parameter in named if parameter.requires_grad}
    frozen_ids = {id(parameter) for _name, parameter in named if not parameter.requires_grad}
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    return {
        "trainable_identity_count": len(trainable_ids),
        "optimizer_identity_count": len(optimizer_ids),
        "duplicate_optimizer_identity_count": len(optimizer_parameters) - len(optimizer_ids),
        "optimizer_matches_trainable_exactly": optimizer_ids == trainable_ids,
        "frozen_optimizer_overlap_count": len(optimizer_ids & frozen_ids),
    }


def _batch_summary(batch: Mapping[str, Any]) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for key in HISTORICAL_MODEL_KEYS:
        value = batch[key]
        if hasattr(value, "shape"):
            tensors[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "minimum": float(value.min().item()),
                "maximum": float(value.max().item()),
            }
        else:
            tensors[key] = {"type": type(value).__name__, "value": list(value) if isinstance(value, tuple) else value}
    return {
        "model_input_keys": list(HISTORICAL_MODEL_KEYS),
        "excluded_audit_only_keys": ["protocol_metadata"],
        "production_loss_entrypoint": "HunyuanPaint.training_step(batch, batch_idx)",
        "tensors_and_values": tensors,
    }


def move_model_batch_to_device(batch: Mapping[str, Any], device: Any, torch: Any) -> dict[str, Any]:
    """Mirror Lightning's batch transfer before calling training_step directly."""

    moved: dict[str, Any] = {}
    for key in HISTORICAL_MODEL_KEYS:
        value = batch[key]
        moved[key] = value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _record_cuda_memory(torch: Any, label: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "label": label,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _release_cuda(torch: Any, *objects: Any) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _scheduler_evidence(config: ValidatedConfiguration, torch: Any, build_scheduler: Any) -> dict[str, Any]:
    steps = config.values["scheduler_evidence_steps"]
    peak_rates = config.values["candidate_peak_learning_rates"]
    result: dict[str, Any] = {
        "implementation": "hy3dft.selective_training.build_warmup_constant_scheduler",
        "warmup_steps": config.values["warmup_steps"],
        "steps": steps,
        "scopes": {},
    }
    for scope in config.values["scope_order"]:
        peak = float(peak_rates[scope])
        parameter = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        optimizer = torch.optim.SGD([parameter], lr=peak)
        scheduler = build_scheduler(optimizer, warmup_steps=config.values["warmup_steps"])
        rows = [{"step": 0, "multiplier": optimizer.param_groups[0]["lr"] / peak, "absolute_lr": optimizer.param_groups[0]["lr"]}]
        wanted = set(steps[1:])
        for step in range(1, max(steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()
            if step in wanted:
                rows.append(
                    {
                        "step": step,
                        "multiplier": optimizer.param_groups[0]["lr"] / peak,
                        "absolute_lr": optimizer.param_groups[0]["lr"],
                    }
                )
        expected = {0: 0.0, 25: 0.5, 50: 1.0, 160: 1.0, 320: 1.0}
        if any(not math.isclose(row["multiplier"], expected[row["step"]], abs_tol=1e-12) for row in rows):
            raise RuntimeError(f"Scheduler runtime evidence differs from contract for {scope}: {rows}")
        result["scopes"][scope] = {"candidate_peak_learning_rate": peak, "evidence": rows}
    return result


def _make_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    return f"{timestamp}_{job_id}"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 2N Day 5 A100 Runtime Preflight",
            "",
            f"Status: `{summary['status']}`",
            "",
            "This is a one-batch, one-update-per-scope runtime audit. It is not formal training or quality evaluation.",
            "",
            "| Scope | Loss | Trainable tensors | Trainable parameters | Max update/weight ratio |",
            "|---|---:|---:|---:|---:|",
            f"| PC-S1 | {summary['pc_s1']['loss']:.8g} | {summary['pc_s1']['trainable_tensor_count']} | {summary['pc_s1']['trainable_numel']} | {summary['pc_s1']['max_update_to_weight_ratio']:.6g} |",
            f"| PC-Full | {summary['pc_full']['loss']:.8g} | {summary['pc_full']['trainable_tensor_count']} | {summary['pc_full']['trainable_numel']} | {summary['pc_full']['max_update_to_weight_ratio']:.6g} |",
            "",
            "The PC-S1 checkpoint binary was removed after exact selective reload verification.",
            "Candidate learning rates remain subject to human review; this audit does not authorize Week 2.",
            "",
        ]
    )


def run_audit(validated: ValidatedConfiguration) -> Path:
    _prepend_runtime_paths(validated)
    import torch

    from hy3dft.selective_training import (
        HISTORICAL_MODEL_KEYS as DAY4_MODEL_KEYS,
        ProtocolCorrectedDataModule,
        apply_trainable_scope,
        build_trainable_adamw,
        build_warmup_constant_scheduler,
    )

    if tuple(DAY4_MODEL_KEYS) != HISTORICAL_MODEL_KEYS:
        raise RuntimeError("Day 4 historical model-key contract changed")
    if not torch.cuda.is_available():
        raise RuntimeError("--run-audit requires CUDA inside the A100 sbatch")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Day 5 requires A100-class bf16 support")
    torch.set_float32_matmul_precision("medium")

    run_dir = validated.output_root / _make_run_id()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Day 5 run: {run_dir}")
    run_dir.mkdir(parents=True)
    for relative in ("batch", "pc_s1", "pc_full"):
        (run_dir / relative).mkdir()

    upstream_repo = validated.official_hypaint.parent
    project_git_before = git_snapshot(validated.project_root)
    upstream_git_before = git_snapshot(upstream_repo)
    torch.cuda.reset_peak_memory_stats()
    initial_memory = _record_cuda_memory(torch, "runtime_start")
    manifest: dict[str, Any] = {
        "phase": validated.values["phase"],
        "status": "RUNNING",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "environment": {
            key: os.environ.get(key)
            for key in ("PROJ", "HY21", "HYPAINT", "ENV_NAME", "CUDA_HOME", "HF_HOME", "TMPDIR", "SLURM_JOB_ID")
        },
        "git": {"project_before": project_git_before, "upstream_before": upstream_git_before},
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "config": validated.values,
        "source_hashes": _source_hashes(validated),
        "initial_cuda_memory": initial_memory,
        "formal_training": False,
        "optimizer_updates_per_scope": 1,
    }
    write_json_report(run_dir / "00_RUNTIME_MANIFEST.json", manifest, run_dir)

    data = ProtocolCorrectedDataModule(
        train_json=validated.train_json,
        batch_size=validated.values["batch_size"],
        num_workers=validated.values["num_workers"],
        image_size=validated.values["image_size"],
        base_seed=validated.values["base_seed"],
        rank=0,
        augmentation_mode=validated.values["augmentation_mode"],
        shuffle_train=False,
    )
    data.set_epoch(validated.values["epoch"])
    data.prepare_data()
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    expected_name = str(validated.train_sample_paths[validated.values["sample_index"]])
    if list(batch["name"]) != [expected_name]:
        raise RuntimeError(f"Deterministic sample-index contract failed: {batch['name']} != {[expected_name]}")
    protocol_metadata = batch["protocol_metadata"]
    if len(protocol_metadata) != 1 or protocol_metadata[0].get("spatial_augmentation") != "none":
        raise RuntimeError(f"Protocol metadata does not prove no augmentation: {protocol_metadata}")
    # Lightning normally performs this transfer before HunyuanPaint.training_step.
    # The audit calls training_step directly, so reproduce that lifecycle step;
    # in particular, the official normal/position preparation does not move its
    # tensors itself.
    model_batch = move_model_batch_to_device(batch, torch.device("cuda"), torch)
    write_json_report(
        run_dir / "batch" / "protocol_metadata.json",
        {"sample_index": 0, "records": protocol_metadata},
        run_dir,
    )
    write_json_report(run_dir / "batch" / "batch_summary.json", _batch_summary(model_batch), run_dir)

    memory_stages: list[dict[str, Any]] = [initial_memory]
    seed = int(validated.values["base_seed"])
    peak_rates = validated.values["candidate_peak_learning_rates"]
    tolerance = validated.values["zero_update_tolerance"]
    clip_norm = float(validated.values["gradient_clip_norm"])
    safety_ceiling = float(validated.values["update_ratio_safety_ceiling"])
    snapshot_config = validated.values["exact_update_snapshot"]

    # PC-S1 begins with the stage-1 real batch and production-loss proof.
    torch.cuda.reset_peak_memory_stats()
    _reset_runtime_seeds(torch, seed)
    model, pc_s1_init = initialize_true_pbr_model(validated, run_dir, torch)
    print("PHASE2N_DAY5_MODEL_LOAD_OK")
    baseline_loss = run_production_loss(model, model_batch, _OptimizerView(), seed=seed, backward=False, torch=torch)
    print("PHASE2N_DAY5_REAL_BATCH_LOSS_OK")
    scope_root = model.unet
    scope_report = apply_trainable_scope(scope_root, "pc_s1")
    _validate_pc_s1_names(scope_report.trainable_parameter_names)
    write_json_report(run_dir / "pc_s1" / "scope_report.json", scope_report.to_dict(), run_dir)
    print("PHASE2N_DAY5_PC_S1_SCOPE_OK")
    scoped_loss = run_production_loss(model, model_batch, _OptimizerView(), seed=seed, backward=False, torch=torch)
    zero_report = make_zero_update_report(baseline_loss, scoped_loss, tolerance)
    if not zero_report["within_tolerance"]:
        raise RuntimeError(f"PC-S1 zero-update equivalence failed: {zero_report}")
    write_json_report(run_dir / "pc_s1" / "zero_update_report.json", zero_report, run_dir)

    pc_s1_optimizer = build_trainable_adamw(scope_root, float(peak_rates["pc_s1"]))
    membership = optimizer_membership_report(scope_root, pc_s1_optimizer)
    if not membership["optimizer_matches_trainable_exactly"] or membership["frozen_optimizer_overlap_count"]:
        raise RuntimeError(f"PC-S1 optimizer membership failed: {membership}")
    pc_s1_loss = run_production_loss(
        model, model_batch, pc_s1_optimizer, seed=seed, backward=True, torch=torch
    )
    pc_s1_gradients = classify_gradients(scope_root.named_parameters())
    pc_s1_gradients["loss"] = pc_s1_loss
    pc_s1_gradients["optimizer_membership"] = membership
    validate_gradient_report(pc_s1_gradients)
    write_json_report(run_dir / "pc_s1" / "gradient_report.json", pc_s1_gradients, run_dir)
    print("PHASE2N_DAY5_PC_S1_GRADIENT_OK")

    pc_s1_selection = select_update_snapshot_names(
        scope_root.named_parameters(),
        "pc_s1",
        pc_s1_cpu_limit_bytes=int(snapshot_config["pc_s1"]["cpu_memory_limit_bytes"]),
        pc_s1_fallback_limit=int(snapshot_config["pc_s1"]["fallback_max_tensors"]),
        pc_full_limit=int(snapshot_config["pc_full"]["max_tensors"]),
    )
    pc_s1_before = snapshot_parameters(scope_root, pc_s1_selection["selected_names"])
    clip_before = float(torch.nn.utils.clip_grad_norm_([p for p in scope_root.parameters() if p.requires_grad], clip_norm).item())
    actual_pc_s1_lr = float(pc_s1_optimizer.param_groups[0]["lr"])
    if actual_pc_s1_lr != float(peak_rates["pc_s1"]):
        raise RuntimeError("PC-S1 peak-LR audit accidentally used a scheduler-zero LR")
    pc_s1_optimizer.step()
    pc_s1_after = snapshot_parameters(scope_root, pc_s1_selection["selected_names"])
    pc_s1_updates = calculate_update_ratios(pc_s1_before, pc_s1_after)
    validate_update_report(pc_s1_updates, safety_ceiling)
    pc_s1_update_report = {
        "loss": pc_s1_loss,
        "optimizer_updates": 1,
        "candidate_peak_learning_rate": float(peak_rates["pc_s1"]),
        "actual_optimizer_learning_rate": actual_pc_s1_lr,
        "scheduler_attached_during_peak_update": False,
        "gradient_clip_norm": clip_norm,
        "gradient_norm_before_clip": clip_before,
        "snapshot": pc_s1_selection,
        "ratios": pc_s1_updates,
        "safety_ceiling": safety_ceiling,
        "final_lr_authorized": False,
    }
    write_json_report(run_dir / "pc_s1" / "update_report.json", pc_s1_update_report, run_dir)
    print("PHASE2N_DAY5_PC_S1_UPDATE_OK")

    pc_s1_scheduler = build_warmup_constant_scheduler(
        pc_s1_optimizer, warmup_steps=validated.values["warmup_steps"]
    )
    all_selected_names = list(scope_report.trainable_parameter_names)
    selected_state = snapshot_parameters(scope_root, all_selected_names)
    frozen_names = [name for name, parameter in scope_root.named_parameters() if not parameter.requires_grad]
    frozen_sample_names = deterministic_bounded_names(
        frozen_names, int(validated.values["pc_s1_smoke_checkpoint"]["verify_frozen_sample_count"])
    )
    frozen_base_snapshot = snapshot_parameters(scope_root, frozen_sample_names)
    temporary_checkpoint = run_dir / "pc_s1" / validated.values["pc_s1_smoke_checkpoint"]["temporary_filename"]
    checkpoint_payload = {
        "format": "phase2n_pc_s1_selective_preflight_v1",
        "selected_parameters": selected_state,
        "optimizer_state": pc_s1_optimizer.state_dict(),
        "scheduler_state": pc_s1_scheduler.state_dict(),
        "scope_report": scope_report.to_dict(),
        "initialization_metadata": pc_s1_init,
    }
    torch.save(checkpoint_payload, temporary_checkpoint)
    checkpoint_manifest = checkpoint_file_manifest(temporary_checkpoint)
    checkpoint_manifest.update(
        {
            "format": checkpoint_payload["format"],
            "selected_parameter_count": len(selected_state),
            "selected_parameter_names": all_selected_names,
            "frozen_verification_sampling_rule": "evenly_spaced_over_sorted_frozen_names",
            "frozen_verification_names": frozen_sample_names,
            "contains_full_model": False,
        }
    )
    memory_stages.append(_record_cuda_memory(torch, "pc_s1_after_update_and_checkpoint_save"))
    del checkpoint_payload, selected_state, pc_s1_before, pc_s1_after
    del pc_s1_scheduler, pc_s1_optimizer, scope_root, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Fresh true-PBR load for selective checkpoint restore proof.
    _reset_runtime_seeds(torch, seed)
    reload_model, reload_init = initialize_true_pbr_model(validated, run_dir, torch)
    reload_root = reload_model.unet
    reload_scope = apply_trainable_scope(reload_root, "pc_s1")
    if list(reload_scope.trainable_parameter_names) != all_selected_names:
        raise RuntimeError("Fresh PC-S1 selected names differ from checkpoint names")
    try:
        loaded = torch.load(temporary_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported Torch fallback.
        loaded = torch.load(temporary_checkpoint, map_location="cpu")
    reload_parameters = dict(reload_root.named_parameters())
    with torch.no_grad():
        for name, tensor in loaded["selected_parameters"].items():
            reload_parameters[name].copy_(tensor.to(device=reload_parameters[name].device, dtype=reload_parameters[name].dtype))
    selected_restore_failures = [
        name
        for name, tensor in loaded["selected_parameters"].items()
        if not torch.equal(reload_parameters[name].detach().cpu(), tensor)
    ]
    frozen_restore_failures = [
        name
        for name, tensor in frozen_base_snapshot.items()
        if not torch.equal(reload_parameters[name].detach().cpu(), tensor)
    ]
    reload_optimizer = build_trainable_adamw(reload_root, float(peak_rates["pc_s1"]))
    reload_scheduler = build_warmup_constant_scheduler(
        reload_optimizer, warmup_steps=validated.values["warmup_steps"]
    )
    reload_optimizer.load_state_dict(loaded["optimizer_state"])
    reload_scheduler.load_state_dict(loaded["scheduler_state"])
    reload_report = {
        "status": "OK" if not selected_restore_failures and not frozen_restore_failures else "FAIL",
        "fresh_initialization_metadata": reload_init,
        "selected_parameter_count": len(all_selected_names),
        "selected_restore_exact": not selected_restore_failures,
        "selected_restore_failures": selected_restore_failures,
        "frozen_sample_count": len(frozen_sample_names),
        "frozen_sample_base_equivalent": not frozen_restore_failures,
        "frozen_restore_failures": frozen_restore_failures,
        "optimizer_state_loaded": True,
        "scheduler_state_loaded": True,
    }
    if reload_report["status"] != "OK":
        raise RuntimeError(f"PC-S1 selective checkpoint reload failed: {reload_report}")
    write_json_report(run_dir / "pc_s1" / "checkpoint_reload_report.json", reload_report, run_dir)
    checkpoint_manifest = cleanup_temporary_checkpoint(temporary_checkpoint, checkpoint_manifest)
    write_json_report(run_dir / "pc_s1" / "checkpoint_manifest.json", checkpoint_manifest, run_dir)
    print("PHASE2N_DAY5_PC_S1_CHECKPOINT_RELOAD_OK")
    del loaded, frozen_base_snapshot, reload_scheduler, reload_optimizer, reload_root, reload_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # PC-Full starts from another fresh true-PBR initialization.
    torch.cuda.reset_peak_memory_stats()
    _reset_runtime_seeds(torch, seed)
    full_model, pc_full_init = initialize_true_pbr_model(validated, run_dir, torch)
    full_root = full_model.unet
    requires_grad_before = {name: bool(parameter.requires_grad) for name, parameter in full_root.named_parameters()}
    full_baseline_loss = run_production_loss(
        full_model, model_batch, _OptimizerView(), seed=seed, backward=False, torch=torch
    )
    full_scope_report = apply_trainable_scope(full_root, "pc_full")
    requires_grad_after = {name: bool(parameter.requires_grad) for name, parameter in full_root.named_parameters()}
    if requires_grad_before != requires_grad_after:
        raise RuntimeError("PC-Full changed the official pre-existing requires_grad state")
    write_json_report(run_dir / "pc_full" / "scope_report.json", full_scope_report.to_dict(), run_dir)
    print("PHASE2N_DAY5_PC_FULL_SCOPE_OK")
    full_scoped_loss = run_production_loss(
        full_model, model_batch, _OptimizerView(), seed=seed, backward=False, torch=torch
    )
    full_zero_report = make_zero_update_report(full_baseline_loss, full_scoped_loss, tolerance)
    if not full_zero_report["within_tolerance"]:
        raise RuntimeError(f"PC-Full zero-update equivalence failed: {full_zero_report}")
    write_json_report(run_dir / "pc_full" / "zero_update_report.json", full_zero_report, run_dir)

    full_optimizer = build_trainable_adamw(full_root, float(peak_rates["pc_full"]))
    full_membership = optimizer_membership_report(full_root, full_optimizer)
    if not full_membership["optimizer_matches_trainable_exactly"] or full_membership["frozen_optimizer_overlap_count"]:
        raise RuntimeError(f"PC-Full optimizer membership failed: {full_membership}")
    full_loss = run_production_loss(
        full_model, model_batch, full_optimizer, seed=seed, backward=True, torch=torch
    )
    full_gradients = classify_gradients(full_root.named_parameters())
    full_gradients["loss"] = full_loss
    full_gradients["optimizer_membership"] = full_membership
    validate_gradient_report(full_gradients)
    write_json_report(run_dir / "pc_full" / "gradient_report.json", full_gradients, run_dir)
    print("PHASE2N_DAY5_PC_FULL_GRADIENT_OK")

    full_selection = select_update_snapshot_names(
        full_root.named_parameters(),
        "pc_full",
        pc_s1_cpu_limit_bytes=int(snapshot_config["pc_s1"]["cpu_memory_limit_bytes"]),
        pc_s1_fallback_limit=int(snapshot_config["pc_s1"]["fallback_max_tensors"]),
        pc_full_limit=int(snapshot_config["pc_full"]["max_tensors"]),
    )
    if full_selection["selected_name_count"] > 16:
        raise RuntimeError("PC-Full snapshot exceeded 16 tensors")
    full_before = snapshot_parameters(full_root, full_selection["selected_names"])
    full_clip_before = float(torch.nn.utils.clip_grad_norm_([p for p in full_root.parameters() if p.requires_grad], clip_norm).item())
    actual_full_lr = float(full_optimizer.param_groups[0]["lr"])
    if actual_full_lr != float(peak_rates["pc_full"]):
        raise RuntimeError("PC-Full peak-LR audit accidentally used a scheduler-zero LR")
    full_optimizer.step()
    full_after = snapshot_parameters(full_root, full_selection["selected_names"])
    full_updates = calculate_update_ratios(full_before, full_after)
    validate_update_report(full_updates, safety_ceiling)
    full_update_report = {
        "loss": full_loss,
        "optimizer_updates": 1,
        "candidate_peak_learning_rate": float(peak_rates["pc_full"]),
        "actual_optimizer_learning_rate": actual_full_lr,
        "scheduler_attached_during_peak_update": False,
        "gradient_clip_norm": clip_norm,
        "gradient_norm_before_clip": full_clip_before,
        "snapshot": full_selection,
        "ratios": full_updates,
        "safety_ceiling": safety_ceiling,
        "final_lr_authorized": False,
        "full_checkpoint_saved": False,
    }
    write_json_report(run_dir / "pc_full" / "update_report.json", full_update_report, run_dir)
    print("PHASE2N_DAY5_PC_FULL_UPDATE_OK")
    memory_stages.append(_record_cuda_memory(torch, "pc_full_after_update"))
    print("PHASE2N_DAY5_PC_FULL_AUDIT_OK")
    del full_before, full_after, full_optimizer, full_root, full_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    scheduler_report = _scheduler_evidence(validated, torch, build_warmup_constant_scheduler)
    write_json_report(run_dir / "scheduler_report.json", scheduler_report, run_dir)
    memory_stages.append(_record_cuda_memory(torch, "runtime_end"))
    memory_report = {
        "stages": memory_stages,
        "maximum_observed_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in memory_stages),
        "maximum_observed_peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in memory_stages),
    }
    write_json_report(run_dir / "memory_report.json", memory_report, run_dir)

    upstream_git_after = git_snapshot(upstream_repo)
    if upstream_git_after != upstream_git_before:
        raise RuntimeError("Upstream Hunyuan Git state changed during the Day 5 audit")
    manifest["status"] = "OK"
    manifest["git"]["upstream_after"] = upstream_git_after
    manifest["git"]["upstream_unchanged_during_audit"] = True
    manifest["final_cuda_memory"] = memory_stages[-1]
    write_json_report(run_dir / "00_RUNTIME_MANIFEST.json", manifest, run_dir, overwrite=True)

    total_root_numel_s1 = scope_report.total_parameter_numel
    total_root_numel_full = full_scope_report.total_parameter_numel
    summary = {
        "status": "OK",
        "phase": validated.values["phase"],
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "formal_training": False,
        "quality_conclusion": False,
        "week2_authorized": False,
        "production_loss_entrypoint": "HunyuanPaint.training_step",
        "pc_s1": {
            "loss": pc_s1_loss,
            "trainable_tensor_count": scope_report.trainable_parameter_tensor_count,
            "trainable_numel": scope_report.trainable_parameter_numel,
            "trainable_fraction": scope_report.trainable_parameter_numel / total_root_numel_s1,
            "total_gradient_norm": pc_s1_gradients["total_gradient_norm"],
            "max_update_to_weight_ratio": pc_s1_updates["max_update_to_weight_ratio"],
            "checkpoint_binary_removed": checkpoint_manifest["binary_removed_after_success"],
        },
        "pc_full": {
            "loss": full_loss,
            "trainable_tensor_count": full_scope_report.trainable_parameter_tensor_count,
            "trainable_numel": full_scope_report.trainable_parameter_numel,
            "trainable_fraction": full_scope_report.trainable_parameter_numel / total_root_numel_full,
            "total_gradient_norm": full_gradients["total_gradient_norm"],
            "max_update_to_weight_ratio": full_updates["max_update_to_weight_ratio"],
            "full_checkpoint_saved": False,
            "initialization_metadata": pc_full_init,
        },
        "human_review_required": [
            "candidate peak learning-rate authorization",
            "gradient and update-to-weight magnitude interpretation",
            "CUDA memory margin",
            "normal/position live-batch semantics",
        ],
        "required_success_tokens": list(REQUIRED_RUNTIME_TOKENS),
    }
    write_json_report(run_dir / "summary.json", summary, run_dir)
    (run_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print("PHASE2N_DAY5_A100_PREFLIGHT_OK")
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2N Day 5 A100 runtime preflight.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate contracts without Torch, CUDA, or Hunyuan imports.")
    mode.add_argument("--run-audit", action="store_true", help="Run the real one-batch audit; requires the A100 sbatch.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validated = run_check_only(args.config)
    if args.run_audit:
        run_audit(validated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
