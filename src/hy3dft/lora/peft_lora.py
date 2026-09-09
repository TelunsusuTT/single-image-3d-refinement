"""LoRA backend selection and injection helpers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .local_linear import adapter_state_dict, wrap_linear_modules


@dataclass
class LoraInjectionResult:
    backend: str
    target_names: list[str]
    wrapped_names: list[str]
    rank: int
    alpha: float
    dropout: float
    trainable_parameter_count: int
    total_parameter_count: int
    trainable_names: list[str]
    backend_note: str


def backend_availability() -> dict[str, bool]:
    return {
        "peft": importlib.util.find_spec("peft") is not None,
        "diffusers": importlib.util.find_spec("diffusers") is not None,
    }


def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def parameter_summary(model: nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_names": trainable_names,
    }


def assert_only_lora_trainable(model: nn.Module) -> None:
    offenders = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" not in name]
    if offenders:
        raise RuntimeError(f"Non-LoRA trainable parameters found: {offenders[:20]}")


def inject_lora(
    model: nn.Module,
    target_names: list[str],
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.0,
    backend: str = "auto",
) -> LoraInjectionResult:
    """Inject LoRA adapters.

    The default `auto` path uses the project's exact Linear implementation.
    PEFT and Diffusers are detected for reporting but are not enabled because
    their generic target matching is too broad for this custom Hunyuan UNet.
    """

    if backend not in {"auto", "local_linear_fallback", "peft", "diffusers"}:
        raise ValueError(f"Unsupported LoRA backend: {backend}")
    availability = backend_availability()
    if backend in {"peft", "diffusers"}:
        raise RuntimeError(
            f"{backend} adapter backend is not enabled for this custom UNet; "
            "use the exact local Linear implementation."
        )
    freeze_all_parameters(model)
    wrapped = wrap_linear_modules(model, target_names, rank=rank, alpha=alpha, dropout=dropout)
    assert_only_lora_trainable(model)
    summary = parameter_summary(model)
    return LoraInjectionResult(
        backend="local_linear_fallback",
        target_names=list(target_names),
        wrapped_names=wrapped,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        trainable_parameter_count=int(summary["trainable_parameter_count"]),
        total_parameter_count=int(summary["total_parameter_count"]),
        trainable_names=list(summary["trainable_names"]),
        backend_note=(
            "PEFT available="
            f"{availability['peft']}, Diffusers available={availability['diffusers']}; "
            "local exact Linear fallback used for safe target specificity."
        ),
    )


def print_trainable_summary(model: nn.Module) -> dict[str, Any]:
    summary = parameter_summary(model)
    total = max(1, int(summary["total_parameter_count"]))
    trainable = int(summary["trainable_parameter_count"])
    pct = 100.0 * trainable / total
    print(f"trainable params: {trainable} / {total} ({pct:.6f}%)")
    for name in summary["trainable_names"][:50]:
        print(f"  trainable: {name}")
    if len(summary["trainable_names"]) > 50:
        print(f"  ... {len(summary['trainable_names']) - 50} more")
    return summary


__all__ = [
    "LoraInjectionResult",
    "adapter_state_dict",
    "assert_only_lora_trainable",
    "backend_availability",
    "inject_lora",
    "parameter_summary",
    "print_trainable_summary",
]
