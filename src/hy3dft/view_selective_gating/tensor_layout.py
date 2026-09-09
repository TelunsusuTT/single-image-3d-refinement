"""Tensor-layout validation and mask broadcasting for conditioning gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


class TensorLayoutError(ValueError):
    """Raised when a live attention residual does not match the audited layout."""


@dataclass(frozen=True)
class RefviewLayout:
    cfg_batch: int
    materials: int
    views: int
    tokens_per_view: int
    channels: int


@dataclass(frozen=True)
class DinoLayout:
    cfg_batch: int
    materials: int
    views: int
    tokens_per_view: int
    channels: int


@dataclass(frozen=True)
class GateApplication:
    output: Any
    layout: dict[str, int]
    broadcast_shape: tuple[int, ...]


def _shape_tuple(output: Any, label: str) -> tuple[int, ...]:
    shape = getattr(output, "shape", None)
    if shape is None:
        raise TensorLayoutError(f"{label} output has no tensor shape: {type(output)!r}")
    try:
        values = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise TensorLayoutError(f"{label} output has an unreadable shape: {shape!r}") from exc
    if any(value <= 0 for value in values):
        raise TensorLayoutError(f"{label} output has a non-positive dimension: {values}")
    return values


def _validate_common(num_views: int, num_materials: int, expected_cfg_batch: int | None) -> None:
    if num_views <= 0:
        raise TensorLayoutError(f"num_views must be positive, got {num_views}")
    if num_materials <= 0:
        raise TensorLayoutError(f"num_materials must be positive, got {num_materials}")
    if expected_cfg_batch is not None and expected_cfg_batch <= 0:
        raise TensorLayoutError(f"expected_cfg_batch must be positive, got {expected_cfg_batch}")


def resolve_refview_layout(
    shape: Sequence[int],
    *,
    num_views: int,
    num_materials: int,
    expected_cfg_batch: int | None = None,
) -> RefviewLayout:
    """Resolve ``[CFG batch, material, view*sequence, channel]``."""

    _validate_common(num_views, num_materials, expected_cfg_batch)
    dims = tuple(int(value) for value in shape)
    if len(dims) != 4:
        raise TensorLayoutError(f"attn_refview residual must be rank 4, got {dims}")
    cfg_batch, materials, view_tokens, channels = dims
    if materials != num_materials:
        raise TensorLayoutError(f"attn_refview material count {materials} != expected {num_materials}")
    if view_tokens % num_views:
        raise TensorLayoutError(
            f"attn_refview sequence dimension {view_tokens} is not divisible by {num_views} views"
        )
    if expected_cfg_batch is not None and cfg_batch != expected_cfg_batch:
        raise TensorLayoutError(f"attn_refview CFG batch {cfg_batch} != expected {expected_cfg_batch}")
    if min(dims) <= 0:
        raise TensorLayoutError(f"attn_refview residual has a non-positive dimension: {dims}")
    return RefviewLayout(cfg_batch, materials, num_views, view_tokens // num_views, channels)


def resolve_dino_layout(
    shape: Sequence[int],
    *,
    num_views: int,
    num_materials: int,
    expected_cfg_batch: int | None = None,
) -> DinoLayout:
    """Resolve ``[CFG batch*material*view, sequence, channel]``."""

    _validate_common(num_views, num_materials, expected_cfg_batch)
    dims = tuple(int(value) for value in shape)
    if len(dims) != 3:
        raise TensorLayoutError(f"attn_dino residual must be rank 3, got {dims}")
    flattened_batch, tokens, channels = dims
    unit = num_materials * num_views
    if flattened_batch % unit:
        raise TensorLayoutError(
            f"attn_dino leading dimension {flattened_batch} is not divisible by "
            f"materials*views={unit}"
        )
    cfg_batch = flattened_batch // unit
    if expected_cfg_batch is not None and cfg_batch != expected_cfg_batch:
        raise TensorLayoutError(f"attn_dino CFG batch {cfg_batch} != expected {expected_cfg_batch}")
    if min(dims) <= 0:
        raise TensorLayoutError(f"attn_dino residual has a non-positive dimension: {dims}")
    return DinoLayout(cfg_batch, num_materials, num_views, tokens, channels)


def _validated_keep_mask(reference_keep_mask: Sequence[float], num_views: int) -> list[float]:
    if len(reference_keep_mask) != num_views:
        raise TensorLayoutError(
            f"reference_keep_mask length {len(reference_keep_mask)} != view count {num_views}"
        )
    mask = [float(value) for value in reference_keep_mask]
    invalid = [value for value in mask if value not in (0.0, 1.0)]
    if invalid:
        raise TensorLayoutError(f"reference_keep_mask must contain only 0.0/1.0, got {invalid[:5]}")
    return mask


def refview_broadcast_values(
    reference_keep_mask: Sequence[float],
    layout: RefviewLayout,
) -> tuple[list[float], tuple[int, int, int, int]]:
    mask = _validated_keep_mask(reference_keep_mask, layout.views)
    values = [value for value in mask for _ in range(layout.tokens_per_view)]
    return values, (1, 1, layout.views * layout.tokens_per_view, 1)


def dino_broadcast_values(
    reference_keep_mask: Sequence[float],
    layout: DinoLayout,
) -> tuple[list[float], tuple[int, int, int]]:
    mask = _validated_keep_mask(reference_keep_mask, layout.views)
    values = [
        mask[view]
        for _cfg in range(layout.cfg_batch)
        for _material in range(layout.materials)
        for view in range(layout.views)
    ]
    return values, (layout.cfg_batch * layout.materials * layout.views, 1, 1)


def apply_refview_keep_mask(
    output: Any,
    reference_keep_mask: Sequence[float],
    *,
    num_views: int,
    num_materials: int,
    expected_cfg_batch: int | None = None,
) -> GateApplication:
    shape = _shape_tuple(output, "attn_refview")
    layout = resolve_refview_layout(
        shape,
        num_views=num_views,
        num_materials=num_materials,
        expected_cfg_batch=expected_cfg_batch,
    )
    values, broadcast_shape = refview_broadcast_values(reference_keep_mask, layout)
    if all(value == 1.0 for value in values):
        return GateApplication(output, asdict(layout), broadcast_shape)
    mask_tensor = output.new_tensor(values).reshape(broadcast_shape)
    return GateApplication(output.masked_fill(mask_tensor == 0, 0), asdict(layout), broadcast_shape)


def apply_dino_keep_mask(
    output: Any,
    reference_keep_mask: Sequence[float],
    *,
    num_views: int,
    num_materials: int,
    expected_cfg_batch: int | None = None,
) -> GateApplication:
    shape = _shape_tuple(output, "attn_dino")
    layout = resolve_dino_layout(
        shape,
        num_views=num_views,
        num_materials=num_materials,
        expected_cfg_batch=expected_cfg_batch,
    )
    values, broadcast_shape = dino_broadcast_values(reference_keep_mask, layout)
    if all(value == 1.0 for value in values):
        return GateApplication(output, asdict(layout), broadcast_shape)
    mask_tensor = output.new_tensor(values).reshape(broadcast_shape)
    return GateApplication(output.masked_fill(mask_tensor == 0, 0), asdict(layout), broadcast_shape)
