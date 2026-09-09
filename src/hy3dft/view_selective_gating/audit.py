"""Runtime provenance helpers for view-selective conditioning gating."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(label: str, value: Any) -> dict[str, Any]:
    source_class = value if inspect.isclass(value) else type(value)
    source = inspect.getsourcefile(source_class)
    if not source:
        raise RuntimeError(f"Could not resolve live source file for {label}: {value!r}")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Live source file for {label} does not exist: {path}")
    return {
        "label": label,
        "class_name": f"{source_class.__module__}.{source_class.__qualname__}",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
    }


def live_source_inventory(
    paint_pipeline: Any,
    target_inventory: Any,
) -> list[dict[str, Any]]:
    """Resolve live implementation files after exact attention discovery."""

    multiview = paint_pipeline.models["multiview_model"]
    diffusion = multiview.pipeline
    records = [
        source_record("outer_paint_pipeline", paint_pipeline),
        source_record("view_processor", paint_pipeline.view_processor),
        source_record("multiview_model", multiview),
        source_record("diffusion_pipeline", diffusion),
        source_record("multiview_unet", diffusion.unet),
    ]
    for kind in ("attn_refview", "attn_dino", "attn_multiview"):
        try:
            name, attention = target_inventory.targets[kind][0]
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Exact target inventory has no representative {kind} module"
            ) from exc
        processor = getattr(attention, "processor", None)
        if processor is None:
            raise RuntimeError(f"Representative {kind} module has no live processor: {name}")
        attention_record = source_record(f"{kind}_outer_attention", attention)
        attention_record["module_name"] = name
        processor_record = source_record(f"{kind}_processor", processor)
        processor_record["module_name"] = name
        try:
            provenance = target_inventory.records[kind][0].processor_provenance
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Representative {kind} target lacks processor provenance"
            ) from exc
        processor_record["processor_provenance"] = asdict(provenance)
        records.extend((attention_record, processor_record))
    return records


def git_snapshot(repo: str | Path) -> dict[str, Any]:
    root = Path(repo).expanduser().resolve()
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"repo": str(root), "head": head, "status_porcelain_v1": status}


def parameter_sentinel(model: Any, *, sample_values_per_parameter: int = 3) -> dict[str, Any]:
    """Collect sampled parameter-integrity evidence without mutating the model.

    This deliberately does not claim to hash every parameter value. It combines
    stable metadata and deterministic value samples with Python object identity,
    tensor storage identity, and PyTorch's in-place mutation version counter.
    """

    if sample_values_per_parameter <= 0:
        raise ValueError("sample_values_per_parameter must be positive")
    digest = hashlib.sha256()
    tensor_count = 0
    total_numel = 0
    trainable_tensor_count = 0
    sampled_value_count = 0
    parameter_records: list[dict[str, Any]] = []
    parameter_names: set[str] = set()
    for name, parameter in model.named_parameters():
        parameter_name = str(name)
        if parameter_name in parameter_names:
            raise RuntimeError(f"Duplicate parameter name in sentinel: {parameter_name}")
        parameter_names.add(parameter_name)
        tensor_count += 1
        numel = int(parameter.numel())
        if numel < 0:
            raise RuntimeError(f"Parameter {parameter_name} reported negative numel: {numel}")
        total_numel += numel
        if bool(parameter.requires_grad):
            trainable_tensor_count += 1
        data_ptr_method = getattr(parameter, "data_ptr", None)
        if not callable(data_ptr_method):
            raise RuntimeError(f"Parameter {parameter_name} does not expose data_ptr()")
        try:
            data_ptr = int(data_ptr_method())
        except Exception as exc:
            raise RuntimeError(f"Could not read data_ptr() for parameter {parameter_name}") from exc
        try:
            version = int(getattr(parameter, "_version"))
        except Exception as exc:
            raise RuntimeError(f"Could not read _version for parameter {parameter_name}") from exc
        metadata = {
            "name": parameter_name,
            "shape": [int(value) for value in parameter.shape],
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "numel": numel,
            "object_id": id(parameter),
            "data_ptr": data_ptr,
            "_version": version,
        }
        sample_indices: list[int] = []
        sampled_values: list[Any] = []
        if numel:
            count = min(sample_values_per_parameter, numel)
            sample_indices = sorted(
                {
                    round(index * (numel - 1) / max(1, count - 1))
                    for index in range(count)
                }
            )
            flat = parameter.detach().reshape(-1)
            sampled_values = flat[sample_indices].float().cpu().tolist()
            if not isinstance(sampled_values, list):
                sampled_values = [sampled_values]
            sampled_value_count += len(sampled_values)
        record = {
            **metadata,
            "sample_indices": sample_indices,
            "sampled_values": sampled_values,
        }
        parameter_records.append(record)
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(repr(sample_indices).encode("ascii"))
        digest.update(repr(sampled_values).encode("ascii"))
    return {
        "evidence_scope": "sampled_parameter_integrity",
        "full_parameter_values_hashed": False,
        "evidence_claim": (
            "Detects parameter object/storage/version/metadata changes and deterministic sampled-value "
            "changes; it does not prove equality of every unsampled value."
        ),
        "algorithm": (
            "sha256(name,shape,dtype,requires_grad,numel,object_id,data_ptr,_version,"
            "sample_indices,evenly_sampled_values)"
        ),
        "sample_values_per_parameter": sample_values_per_parameter,
        "sampled_value_count": sampled_value_count,
        "sha256": digest.hexdigest(),
        "parameter_tensor_count": tensor_count,
        "parameter_numel": total_numel,
        "trainable_parameter_tensor_count": trainable_tensor_count,
        "parameters": parameter_records,
    }


def write_json_once(path: str | Path, payload: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return destination


def write_text_once(path: str | Path, lines: Iterable[str]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line))
            if not str(line).endswith("\n"):
                handle.write("\n")
    return destination
