"""Exact LoRA target selection for Hunyuan3D-Paint modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TARGET_SUFFIXES = (".to_q", ".to_k", ".to_v", ".to_out.0")
REF_DINO_SEGMENTS = (".attn_refview.", ".attn_dino.")
FORBIDDEN_SUBSTRINGS = (
    "attn_multiview",
    ".attn1.",
    ".attn2.",
    ".ff.",
    ".ff_",
    ".conv",
    ".norm",
    "dino_v2",
    "learned_text_clip",
)


@dataclass(frozen=True)
class LinearModuleInfo:
    name: str
    class_name: str
    shape: list[int] | None
    group: str
    selected_by_ref_dino_preset: bool


def _is_linear(module: object) -> bool:
    try:
        import torch.nn as nn
    except Exception:  # pragma: no cover - torch is required at runtime.
        return False
    return isinstance(module, nn.Linear)


def _weight_shape(module: object) -> list[int] | None:
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    return list(shape) if shape is not None else None


def is_projection_name(name: str) -> bool:
    return name.endswith(TARGET_SUFFIXES)


def classify_linear_name(name: str) -> str:
    if ".attn_refview." in name and is_projection_name(name):
        return "refview_qkv_out"
    if ".attn_dino." in name and is_projection_name(name):
        return "dino_qkv_out"
    if ".attn_multiview." in name and is_projection_name(name):
        return "multiview_qkv_out"
    if any(token in name for token in ("material", "pbr", "roughness", "metallic", "albedo", "normal")):
        return "material_specific"
    if ".attn1." in name or ".attn2." in name:
        return "base_attn"
    return "other_linear"


def selected_by_ref_dino(name: str) -> bool:
    return any(segment in name for segment in REF_DINO_SEGMENTS) and is_projection_name(name)


def inventory_linear_modules(model: object) -> list[LinearModuleInfo]:
    items: list[LinearModuleInfo] = []
    for name, module in model.named_modules():
        if not name or not _is_linear(module):
            continue
        selected = selected_by_ref_dino(name)
        items.append(
            LinearModuleInfo(
                name=name,
                class_name=f"{module.__class__.__module__}.{module.__class__.__name__}",
                shape=_weight_shape(module),
                group=classify_linear_name(name),
                selected_by_ref_dino_preset=selected,
            )
        )
    return items


def select_lora_targets(model: object, preset: str = "ref_dino") -> list[str]:
    if preset != "ref_dino":
        raise ValueError(f"Unsupported LoRA preset: {preset}")
    targets = [item.name for item in inventory_linear_modules(model) if item.selected_by_ref_dino_preset]
    validate_target_names(targets)
    return targets


def validate_target_names(target_names: Iterable[str]) -> list[str]:
    targets = list(target_names)
    if not targets:
        raise ValueError("LoRA target selection is empty for preset ref_dino")
    errors: list[str] = []
    for name in targets:
        if not any(segment in name for segment in REF_DINO_SEGMENTS):
            errors.append(f"{name}: missing .attn_refview. or .attn_dino. segment")
        if not is_projection_name(name):
            errors.append(f"{name}: target must end with .to_q, .to_k, .to_v, or .to_out.0")
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in name:
                errors.append(f"{name}: forbidden target substring {forbidden}")
    if errors:
        raise ValueError("Invalid LoRA targets:\n" + "\n".join(errors))
    return targets
