"""Minimal exact-module LoRA fallback for torch.nn.Linear."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap an existing Linear layer with trainable low-rank adapter weights."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora = torch.matmul(torch.matmul(self.dropout(x), self.lora_A.t()), self.lora_B.t())
        return base_out + lora * self.scale


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    if not parts:
        raise ValueError("module name is empty")
    parent: nn.Module = model
    for part in parts[:-1]:
        child = getattr(parent, part)
        if not isinstance(child, nn.Module):
            raise TypeError(f"{part} in {module_name} is not a Module")
        parent = child
    return parent, parts[-1]


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    module: nn.Module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def wrap_linear_modules(model: nn.Module, target_names: list[str], rank: int = 4, alpha: float = 4.0, dropout: float = 0.0) -> list[str]:
    wrapped: list[str] = []
    for name in target_names:
        module = get_module_by_name(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA fallback can only wrap nn.Linear targets: {name} -> {type(module).__name__}")
        parent, leaf = _get_parent_module(model, name)
        parent.add_module(leaf, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        wrapped.append(name)
    return wrapped


def adapter_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_adapter_state_dict(model: nn.Module, state: dict[str, Any], strict: bool = True) -> dict[str, list[str]]:
    expected = set(adapter_state_dict(model).keys())
    provided = set(state.keys())
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Adapter state mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
    module_map = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    for key, tensor in state.items():
        module_name, param_name = key.rsplit(".", 1)
        module = module_map.get(module_name)
        if module is None:
            continue
        target = getattr(module, param_name)
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
    return {"missing_keys": missing, "unexpected_keys": unexpected}
