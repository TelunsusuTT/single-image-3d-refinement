"""Strict loading for Phase 2N trainable-scope-only checkpoints.

Static manifest validation uses only the Python standard library. Torch is
imported lazily only by the real checkpoint-loading function.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


CHECKPOINT_FORMAT = "phase2n_week2_model_only_trainable_scope_v1"
ALLOWED_SCOPES = frozenset({"pc_s1", "pc_full"})
UNSAFE_MANIFEST_FLAGS = (
    "contains_optimizer_state",
    "contains_scheduler_state",
    "contains_frozen_parameters",
    "contains_full_model",
)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint manifest does not exist: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint manifest JSON at {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint manifest must be a JSON object: {source}")
    return payload


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 string")
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(f"{label} is not hexadecimal") from None
    return value.lower()


SCOPE_REPORT_KEYS = frozenset(
    {
        "scope",
        "total_parameter_tensor_count",
        "total_parameter_numel",
        "trainable_parameter_tensor_count",
        "trainable_parameter_numel",
        "frozen_parameter_tensor_count",
        "frozen_parameter_numel",
        "trainable_parameter_names",
        "frozen_parameter_names",
    }
)


def _validate_audited_scope_report(
    report: Any,
    *,
    expected_scope: str,
    expected_trainable_names: Sequence[str],
    expected_trainable_numel: int,
) -> dict[str, Any]:
    if not isinstance(report, dict) or set(report) != SCOPE_REPORT_KEYS:
        raise ValueError("Checkpoint scope_report fields differ from the audited schema")
    if report.get("scope") != expected_scope:
        raise ValueError("Checkpoint scope_report scope does not match the manifest")

    trainable_names = report.get("trainable_parameter_names")
    frozen_names = report.get("frozen_parameter_names")
    if trainable_names != list(expected_trainable_names):
        raise ValueError("Checkpoint scope_report trainable names do not match the manifest")
    if not isinstance(frozen_names, list) or any(
        not isinstance(name, str) or not name for name in frozen_names
    ):
        raise ValueError("Checkpoint scope_report frozen names must be non-empty strings")
    if len(frozen_names) != len(set(frozen_names)):
        raise ValueError("Checkpoint scope_report contains duplicate frozen names")
    if set(trainable_names) & set(frozen_names):
        raise ValueError("Checkpoint scope_report trainable and frozen names overlap")

    trainable_count = _require_positive_int(
        report.get("trainable_parameter_tensor_count"),
        "scope_report trainable_parameter_tensor_count",
    )
    trainable_numel = _require_positive_int(
        report.get("trainable_parameter_numel"),
        "scope_report trainable_parameter_numel",
    )
    frozen_count = _require_positive_int(
        report.get("frozen_parameter_tensor_count"),
        "scope_report frozen_parameter_tensor_count",
    )
    frozen_numel = _require_positive_int(
        report.get("frozen_parameter_numel"),
        "scope_report frozen_parameter_numel",
    )
    total_count = _require_positive_int(
        report.get("total_parameter_tensor_count"),
        "scope_report total_parameter_tensor_count",
    )
    total_numel = _require_positive_int(
        report.get("total_parameter_numel"),
        "scope_report total_parameter_numel",
    )
    if trainable_count != len(trainable_names) or trainable_count != len(expected_trainable_names):
        raise ValueError("Checkpoint scope_report trainable tensor count is inconsistent")
    if trainable_numel != expected_trainable_numel:
        raise ValueError("Checkpoint scope_report trainable numel is inconsistent")
    if frozen_count != len(frozen_names):
        raise ValueError("Checkpoint scope_report frozen tensor count is inconsistent")
    if total_count != trainable_count + frozen_count:
        raise ValueError("Checkpoint scope_report total tensor count is inconsistent")
    if total_numel != trainable_numel + frozen_numel:
        raise ValueError("Checkpoint scope_report total numel is inconsistent")
    return dict(report)


def validate_checkpoint_artifact(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_scope: str,
    expected_step: int,
    expected_sha256: str,
    expected_byte_size: int,
    expected_tensor_count: int,
    expected_numel: int,
) -> dict[str, Any]:
    """Validate one audited binary/manifest pair without loading Torch state."""

    state_path = Path(checkpoint_path).expanduser().resolve()
    metadata_path = Path(manifest_path).expanduser().resolve()
    if expected_scope not in ALLOWED_SCOPES:
        raise ValueError(f"Unsupported scope {expected_scope!r}")
    _require_positive_int(expected_step, "expected_step")
    _require_positive_int(expected_byte_size, "expected_byte_size")
    _require_positive_int(expected_tensor_count, "expected_tensor_count")
    _require_positive_int(expected_numel, "expected_numel")
    expected_sha256 = _require_sha256(expected_sha256, "expected_sha256")

    if not state_path.is_file():
        raise FileNotFoundError(f"Scope checkpoint does not exist: {state_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Scope checkpoint manifest does not exist: {metadata_path}")
    manifest = read_json_object(metadata_path)

    expected_values = {
        "format": CHECKPOINT_FORMAT,
        "scope": expected_scope,
        "global_update": expected_step,
        "byte_size": expected_byte_size,
        "sha256": expected_sha256,
        "trainable_parameter_tensor_count": expected_tensor_count,
        "trainable_parameter_numel": expected_numel,
    }
    for key, expected in expected_values.items():
        actual = manifest.get(key)
        if actual != expected:
            raise ValueError(f"Checkpoint manifest {key} mismatch: {actual!r} != {expected!r}")
    for flag in UNSAFE_MANIFEST_FLAGS:
        if manifest.get(flag) is not False:
            raise ValueError(f"Checkpoint manifest must set {flag}=false")

    recorded_path = manifest.get("checkpoint_path")
    if not isinstance(recorded_path, str) or Path(recorded_path).expanduser().resolve() != state_path:
        raise ValueError(f"Checkpoint manifest path does not resolve to {state_path}")

    names = manifest.get("trainable_parameter_names")
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Checkpoint manifest trainable_parameter_names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint manifest contains duplicate trainable parameter names")
    if len(names) != expected_tensor_count:
        raise ValueError("Checkpoint manifest trainable name count does not match expected tensor count")

    numel_by_name = manifest.get("numel_by_name")
    dtype_by_name = manifest.get("dtype_by_name")
    if not isinstance(numel_by_name, dict) or set(numel_by_name) != set(names):
        raise ValueError("Checkpoint manifest numel_by_name keys do not exactly match trainable names")
    if not isinstance(dtype_by_name, dict) or set(dtype_by_name) != set(names):
        raise ValueError("Checkpoint manifest dtype_by_name keys do not exactly match trainable names")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in numel_by_name.values()):
        raise ValueError("Checkpoint manifest contains an invalid parameter numel")
    if sum(numel_by_name.values()) != expected_numel:
        raise ValueError("Checkpoint manifest per-name numel sum does not match expected numel")
    if any(not isinstance(value, str) or not value.startswith("torch.") for value in dtype_by_name.values()):
        raise ValueError("Checkpoint manifest contains an invalid tensor dtype")

    _validate_audited_scope_report(
        manifest.get("scope_report"),
        expected_scope=expected_scope,
        expected_trainable_names=names,
        expected_trainable_numel=expected_numel,
    )

    actual_size = state_path.stat().st_size
    if actual_size != expected_byte_size:
        raise ValueError(f"Checkpoint byte size mismatch: {actual_size} != {expected_byte_size}")
    actual_sha256 = sha256_file(state_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {actual_sha256} != {expected_sha256}")

    return {
        "status": "OK",
        "checkpoint_path": str(state_path),
        "manifest_path": str(metadata_path),
        "format": CHECKPOINT_FORMAT,
        "scope": expected_scope,
        "step": expected_step,
        "byte_size": actual_size,
        "sha256": actual_sha256,
        "trainable_parameter_tensor_count": expected_tensor_count,
        "trainable_parameter_numel": expected_numel,
        "trainable_parameter_names": list(names),
        "dtype_by_name": dict(dtype_by_name),
        "numel_by_name": dict(numel_by_name),
        "manifest": manifest,
    }


def _scope_report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    if not isinstance(report, Mapping):
        raise TypeError("Scope resolver must return a mapping or an object with to_dict()")
    return dict(report)


def derive_expected_scope(
    model: Any,
    scope: str,
    *,
    scope_resolver: Callable[[Any, str], Any] | None = None,
    audited_scope_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive exact scope names while restoring every requires_grad flag.

    PC-Full preserves the official training initializer's trainable subset. An
    inference model does not necessarily retain those training-time flags, so
    its audited training scope report is reconciled against the entire fresh
    live keyspace before selecting that subset.
    """

    if scope not in ALLOWED_SCOPES:
        raise ValueError(f"Unsupported scope {scope!r}")
    parameters = list(model.named_parameters())
    original_flags = {name: bool(parameter.requires_grad) for name, parameter in parameters}
    if len(original_flags) != len(parameters):
        raise ValueError("Live model contains duplicate parameter names")
    live = dict(parameters)

    audited: dict[str, Any] | None = None
    if audited_scope_report is not None:
        audited = dict(audited_scope_report)
        audited_trainable = tuple(audited["trainable_parameter_names"])
        audited_frozen = tuple(audited["frozen_parameter_names"])
        audited_all = audited_trainable + audited_frozen
        if len(audited_all) != len(set(audited_all)):
            raise ValueError("Audited training scope keyspace contains duplicate names")
        missing_live = sorted(set(audited_all) - set(live))
        unexpected_live = sorted(set(live) - set(audited_all))
        if missing_live or unexpected_live:
            raise ValueError(
                "Fresh inference model keyspace differs from the audited training scope: "
                f"missing_live={missing_live[:20]} unexpected_live={unexpected_live[:20]}"
            )
        if len(live) != audited["total_parameter_tensor_count"]:
            raise ValueError("Fresh inference model tensor count differs from the audited training scope")
        live_total_numel = sum(int(parameter.numel()) for parameter in live.values())
        if live_total_numel != audited["total_parameter_numel"]:
            raise ValueError("Fresh inference model numel differs from the audited training scope")

    try:
        if scope == "pc_full" and audited is not None:
            report = dict(audited)
            resolution_basis = "audited_official_training_scope_report"
        else:
            if scope_resolver is None:
                from hy3dft.selective_training import apply_trainable_scope

                scope_resolver = apply_trainable_scope
            report = _scope_report_dict(scope_resolver(model, scope))
            resolution_basis = "semantic_scope_resolver"
    finally:
        for name, parameter in parameters:
            parameter.requires_grad = original_flags[name]

    names = report.get("trainable_parameter_names")
    if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) for name in names):
        raise ValueError("Derived scope has invalid trainable parameter names")
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError("Derived scope contains duplicate parameter names")
    missing = sorted(set(names) - set(live))
    if missing:
        raise ValueError(f"Derived scope names are missing from the live model: {missing[:20]}")
    if audited is not None and names != tuple(audited["trainable_parameter_names"]):
        raise ValueError("Derived scope names differ from the audited official training scope")
    actual_numel = sum(int(live[name].numel()) for name in names)
    if report.get("trainable_parameter_tensor_count") != len(names):
        raise ValueError("Derived scope tensor count is internally inconsistent")
    if report.get("trainable_parameter_numel") != actual_numel:
        raise ValueError("Derived scope numel is internally inconsistent")
    report["trainable_parameter_names"] = list(names)
    report["scope_resolution_basis"] = resolution_basis
    report["audited_live_keyspace_verified"] = audited is not None
    return report


def _deterministic_frozen_sample(
    named_parameters: Mapping[str, Any],
    loaded_names: set[str],
    count: int,
) -> list[str]:
    candidates = [name for name in named_parameters if name not in loaded_names]
    candidates.sort(
        key=lambda name: (
            int(named_parameters[name].numel()),
            hashlib.sha256(name.encode("utf-8")).hexdigest(),
            name,
        )
    )
    return candidates[: max(0, int(count))]


def _load_cpu_state(torch_module: Any, checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        state = torch_module.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - audited runtime is Torch 2.5.
        raise RuntimeError("Torch with weights_only checkpoint loading is required") from exc
    if not isinstance(state, Mapping):
        raise TypeError("Scope checkpoint payload must be a plain tensor mapping")
    return state


def load_scope_checkpoint_into_model(
    model: Any,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_scope: str,
    expected_step: int,
    expected_sha256: str,
    expected_byte_size: int,
    expected_tensor_count: int,
    expected_numel: int,
    torch_module: Any | None = None,
    scope_resolver: Callable[[Any, str], Any] | None = None,
    frozen_sample_count: int = 8,
) -> dict[str, Any]:
    """Strictly reconstruct a scope checkpoint on one fresh base model."""

    if torch_module is None:
        import torch as torch_module  # type: ignore

    static = validate_checkpoint_artifact(
        checkpoint_path,
        manifest_path,
        expected_scope=expected_scope,
        expected_step=expected_step,
        expected_sha256=expected_sha256,
        expected_byte_size=expected_byte_size,
        expected_tensor_count=expected_tensor_count,
        expected_numel=expected_numel,
    )
    scope_report = derive_expected_scope(
        model,
        expected_scope,
        scope_resolver=scope_resolver,
        audited_scope_report=static["manifest"]["scope_report"],
    )
    live_names = tuple(scope_report["trainable_parameter_names"])
    manifest_names = tuple(static["trainable_parameter_names"])
    if live_names != manifest_names or set(live_names) != set(manifest_names):
        missing = sorted(set(manifest_names) - set(live_names))
        unexpected = sorted(set(live_names) - set(manifest_names))
        raise ValueError(
            "Live scope names do not exactly equal checkpoint names: "
            f"missing_live={missing[:20]} unexpected_live={unexpected[:20]}"
        )
    if scope_report["trainable_parameter_tensor_count"] != expected_tensor_count:
        raise ValueError("Live scope tensor count does not match the audited checkpoint")
    if scope_report["trainable_parameter_numel"] != expected_numel:
        raise ValueError("Live scope numel does not match the audited checkpoint")

    named_parameters = dict(model.named_parameters())
    frozen_names = _deterministic_frozen_sample(named_parameters, set(live_names), frozen_sample_count)
    frozen_before = {
        name: named_parameters[name].detach().cpu().clone()
        for name in frozen_names
    }

    state = _load_cpu_state(torch_module, Path(checkpoint_path).expanduser().resolve())
    items = list(state.items())
    state_names = [name for name, _tensor in items]
    if any(not isinstance(name, str) or not name for name in state_names):
        raise TypeError("Scope checkpoint keys must be non-empty strings")
    if len(state_names) != len(set(state_names)):
        raise ValueError("Scope checkpoint payload contains duplicate keys")
    if tuple(state_names) != manifest_names or set(state_names) != set(manifest_names):
        missing = sorted(set(manifest_names) - set(state_names))
        unexpected = sorted(set(state_names) - set(manifest_names))
        raise ValueError(
            "Scope checkpoint keys do not exactly match the manifest: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )

    copied_names: list[str] = []
    with torch_module.no_grad():
        for name, source in items:
            if not torch_module.is_tensor(source):
                raise TypeError(f"Scope checkpoint entry is not a tensor: {name}")
            source_device = getattr(source, "device", None)
            source_device_type = getattr(source_device, "type", str(source_device).split(":", 1)[0])
            if source_device_type != "cpu":
                raise ValueError(f"Scope checkpoint entry was not loaded on CPU: {name}")
            if not bool(torch_module.isfinite(source).all().item()):
                raise ValueError(f"Scope checkpoint entry contains NaN or Inf: {name}")
            target = named_parameters[name]
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"Scope checkpoint shape mismatch for {name}: {tuple(source.shape)} != {tuple(target.shape)}"
                )
            if int(source.numel()) != static["numel_by_name"][name]:
                raise ValueError(f"Scope checkpoint numel mismatch for {name}")
            if str(source.dtype) != static["dtype_by_name"][name]:
                raise ValueError(f"Scope checkpoint dtype mismatch for {name}")
            cast_source = source.to(device=target.device, dtype=target.dtype)
            target.copy_(cast_source)
            if not bool(torch_module.equal(target.detach(), cast_source)):
                raise RuntimeError(f"Loaded target verification failed for {name}")
            copied_names.append(name)

    changed_frozen = [
        name
        for name, before in frozen_before.items()
        if not bool(torch_module.equal(before, named_parameters[name].detach().cpu()))
    ]
    if changed_frozen:
        raise RuntimeError(f"Frozen parameter sample changed during scope loading: {changed_frozen}")
    model.eval()

    return {
        "status": "OK",
        "checkpoint_path": static["checkpoint_path"],
        "manifest_path": static["manifest_path"],
        "checkpoint_sha256": static["sha256"],
        "checkpoint_byte_size": static["byte_size"],
        "scope": expected_scope,
        "step": expected_step,
        "expected_live_scope_derived": True,
        "scope_resolution_basis": scope_report["scope_resolution_basis"],
        "audited_live_keyspace_verified": scope_report["audited_live_keyspace_verified"],
        "exact_key_equality": True,
        "loaded_parameter_tensor_count": len(copied_names),
        "loaded_parameter_numel": sum(int(named_parameters[name].numel()) for name in copied_names),
        "dtype_device_safe_copy": True,
        "checkpoint_loaded_on_cpu": True,
        "loaded_target_equality_verified": True,
        "nonfinite_entry_count": 0,
        "frozen_sample_names": frozen_names,
        "frozen_sample_count": len(frozen_names),
        "frozen_sample_unchanged": True,
        "model_eval_mode": not bool(model.training),
        "requires_grad_flags_restored_after_scope_derivation": True,
    }
