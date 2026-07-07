"""Adapter-only save/load helpers for Phase 2M."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


FORBIDDEN_OUTPUT_SUFFIXES = {".ckpt", ".bin"}


def guard_adapter_output_path(output_path: Path) -> None:
    name = output_path.name.lower()
    if output_path.suffix.lower() in FORBIDDEN_OUTPUT_SUFFIXES:
        raise ValueError(f"Refusing adapter save to full-checkpoint-like suffix: {output_path}")
    allowed_training_name = name.startswith("adapter_step_") or name == "adapter_final.pt"
    if "adapter" not in name or ("state" not in name and not allowed_training_name):
        raise ValueError(
            "Adapter filename must include 'adapter' plus either 'state', "
            f"'adapter_step_', or 'adapter_final.pt': {output_path}"
        )
    bad_tokens = ("full_model", "merged", "base_model", "save_pretrained")
    if any(token in name for token in bad_tokens):
        raise ValueError(f"Refusing unsafe adapter output filename: {output_path}")


def config_path_for(output_path: Path) -> Path:
    return output_path.with_name("adapter_config.json")


def save_adapter_state_dict(state_dict: dict[str, Any], output_path: str | Path, config: dict[str, Any] | None = None) -> Path:
    path = Path(output_path)
    guard_adapter_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing adapter state: {path}")
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError("safetensors is not installed; use .pt for adapter save") from exc
        save_file(state_dict, str(path))
    else:
        torch.save(state_dict, path)
    if config is not None:
        config_path_for(path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def load_adapter_state_dict(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is not installed; cannot load .safetensors adapter") from exc
        return dict(load_file(str(path)))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
