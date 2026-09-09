from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.view_selective_gating.tensor_layout import (  # noqa: E402
    TensorLayoutError,
    apply_dino_keep_mask,
    apply_refview_keep_mask,
    dino_broadcast_values,
    refview_broadcast_values,
    resolve_dino_layout,
    resolve_refview_layout,
)


KEEP_MASK = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def _size(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _strides(shape: Sequence[int]) -> tuple[int, ...]:
    result = []
    for index in range(len(shape)):
        result.append(_size(shape[index + 1 :]))
    return tuple(result)


def _coordinates(shape: Sequence[int]) -> Iterable[tuple[int, ...]]:
    return product(*(range(dimension) for dimension in shape))


def _offset(shape: Sequence[int], coordinates: Sequence[int]) -> int:
    return sum(index * stride for index, stride in zip(coordinates, _strides(shape)))


class FakeTensor:
    """Minimal row-major tensor with the APIs used by tensor_layout.py."""

    def __init__(self, shape: Sequence[int], values: Sequence[float]) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.values = tuple(float(value) for value in values)
        if _size(self.shape) != len(self.values):
            raise ValueError(f"shape {self.shape} does not contain {len(self.values)} values")

    @classmethod
    def sequential(cls, shape: Sequence[int], start: float = 1.0) -> "FakeTensor":
        return cls(shape, [start + index for index in range(_size(shape))])

    def new_tensor(self, values: Sequence[float]) -> "FakeTensor":
        return FakeTensor((len(values),), values)

    def reshape(self, *shape: object) -> "FakeTensor":
        requested = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        dimensions = [int(value) for value in requested]  # type: ignore[arg-type]
        if dimensions.count(-1) > 1:
            raise ValueError("only one inferred dimension is supported")
        if -1 in dimensions:
            known = _size([value for value in dimensions if value != -1])
            dimensions[dimensions.index(-1)] = len(self.values) // known
        return FakeTensor(tuple(dimensions), self.values)

    def __eq__(self, other: object) -> "FakeTensor":  # type: ignore[override]
        if not isinstance(other, (int, float)):
            return NotImplemented
        return FakeTensor(self.shape, [float(value == other) for value in self.values])

    def _broadcast_value(
        self,
        other: "FakeTensor",
        coordinates: Sequence[int],
    ) -> float:
        if len(other.shape) > len(self.shape):
            raise ValueError("mask rank exceeds output rank")
        aligned = (1,) * (len(self.shape) - len(other.shape)) + other.shape
        if any(
            mask_dimension not in (1, output_dimension)
            for output_dimension, mask_dimension in zip(self.shape, aligned)
        ):
            raise ValueError(f"cannot broadcast {other.shape} over {self.shape}")
        mask_coordinates = tuple(
            0 if mask_dimension == 1 else coordinate
            for coordinate, mask_dimension in zip(coordinates, aligned)
        )
        return other.values[_offset(aligned, mask_coordinates)]

    def masked_fill(self, mask: "FakeTensor", value: float) -> "FakeTensor":
        return FakeTensor(
            self.shape,
            [
                float(value) if self._broadcast_value(mask, coordinates) else source
                for coordinates, source in zip(_coordinates(self.shape), self.values)
            ],
        )

    def __mul__(self, other: object) -> "FakeTensor":
        if not isinstance(other, FakeTensor):
            return NotImplemented
        if len(other.shape) > len(self.shape):
            raise ValueError("mask rank exceeds output rank")
        aligned = (1,) * (len(self.shape) - len(other.shape)) + other.shape
        for output_dimension, mask_dimension in zip(self.shape, aligned):
            if mask_dimension not in (1, output_dimension):
                raise ValueError(f"cannot broadcast {other.shape} over {self.shape}")
        mask_strides = _strides(aligned)
        values = []
        for coordinates in _coordinates(self.shape):
            mask_coordinates = tuple(
                0 if mask_dimension == 1 else coordinate
                for coordinate, mask_dimension in zip(coordinates, aligned)
            )
            output_value = self.values[_offset(self.shape, coordinates)]
            mask_value = other.values[
                sum(index * stride for index, stride in zip(mask_coordinates, mask_strides))
            ]
            values.append(output_value * mask_value)
        return FakeTensor(self.shape, values)


def test_refview_layout_resolves_two_materials_six_views_and_cfg_batch() -> None:
    layout = resolve_refview_layout(
        (2, 2, 18, 4),
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    )

    assert layout.cfg_batch == 2
    assert layout.materials == 2
    assert layout.views == 6
    assert layout.tokens_per_view == 3
    assert layout.channels == 4


def test_dino_layout_resolves_two_materials_six_views_and_cfg_batch() -> None:
    layout = resolve_dino_layout(
        (24, 3, 4),
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    )

    assert layout.cfg_batch == 2
    assert layout.materials == 2
    assert layout.views == 6
    assert layout.tokens_per_view == 3
    assert layout.channels == 4


def test_refview_broadcast_repeats_each_view_across_its_tokens() -> None:
    layout = resolve_refview_layout(
        (2, 2, 18, 4),
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    )
    values, shape = refview_broadcast_values(KEEP_MASK, layout)

    assert shape == (1, 1, 18, 1)
    assert values == [1.0] * 9 + [0.0] * 9


def test_dino_broadcast_uses_cfg_material_view_flatten_order() -> None:
    layout = resolve_dino_layout(
        (24, 3, 4),
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    )
    values, shape = dino_broadcast_values(KEEP_MASK, layout)

    assert shape == (24, 1, 1)
    assert values == KEEP_MASK * 4


def test_refview_mask_broadcasts_across_cfg_material_tokens_and_channels() -> None:
    original = FakeTensor.sequential((2, 2, 12, 3))
    application = apply_refview_keep_mask(
        original,
        KEEP_MASK,
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    )

    assert application.layout == {
        "cfg_batch": 2,
        "materials": 2,
        "views": 6,
        "tokens_per_view": 2,
        "channels": 3,
    }
    assert application.broadcast_shape == (1, 1, 12, 1)
    for cfg, material, view_token, channel in _coordinates(original.shape):
        index = _offset(original.shape, (cfg, material, view_token, channel))
        view = view_token // 2
        expected = original.values[index] if KEEP_MASK[view] else 0.0
        assert application.output.values[index] == expected


def test_dino_mask_broadcasts_across_cfg_material_view_sequence_and_channels() -> None:
    original = FakeTensor.sequential((24, 2, 3))
    application = apply_dino_keep_mask(
        original,
        KEEP_MASK,
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    )

    assert application.layout == {
        "cfg_batch": 2,
        "materials": 2,
        "views": 6,
        "tokens_per_view": 2,
        "channels": 3,
    }
    assert application.broadcast_shape == (24, 1, 1)
    for flattened, token, channel in _coordinates(original.shape):
        index = _offset(original.shape, (flattened, token, channel))
        view = flattened % 6
        expected = original.values[index] if KEEP_MASK[view] else 0.0
        assert application.output.values[index] == expected


@pytest.mark.parametrize(
    ("resolver", "shape", "message"),
    [
        (resolve_refview_layout, (2, 2, 18), "rank 4"),
        (resolve_refview_layout, (2, 1, 18, 4), "material count"),
        (resolve_refview_layout, (2, 2, 17, 4), "not divisible"),
        (resolve_refview_layout, (0, 2, 18, 4), "non-positive"),
        (resolve_dino_layout, (24, 3, 4, 1), "rank 3"),
        (resolve_dino_layout, (25, 3, 4), "not divisible"),
        (resolve_dino_layout, (0, 3, 4), "non-positive"),
    ],
)
def test_invalid_live_tensor_shapes_fail_closed(resolver, shape, message: str) -> None:
    with pytest.raises(TensorLayoutError, match=message):
        resolver(shape, num_views=6, num_materials=2)


@pytest.mark.parametrize("resolver,shape", [(resolve_refview_layout, (2, 2, 18, 4)), (resolve_dino_layout, (24, 3, 4))])
def test_cfg_batch_mismatch_fails_closed(resolver, shape) -> None:
    with pytest.raises(TensorLayoutError, match="CFG batch"):
        resolver(
            shape,
            num_views=6,
            num_materials=2,
            expected_cfg_batch=3,
        )


@pytest.mark.parametrize("num_views,num_materials,expected_cfg_batch", [(0, 2, 2), (6, 0, 2), (6, 2, 0)])
def test_invalid_declared_layout_counts_fail_closed(
    num_views: int,
    num_materials: int,
    expected_cfg_batch: int,
) -> None:
    with pytest.raises(TensorLayoutError):
        resolve_dino_layout(
            (24, 3, 4),
            num_views=num_views,
            num_materials=num_materials,
            expected_cfg_batch=expected_cfg_batch,
        )


@pytest.mark.parametrize("mask", [[1.0] * 5, [1.0] * 7, [1.0, 1.0, 1.0, 0.5, 0.0, 0.0], [1.0, 1.0, 1.0, -1.0, 0.0, 0.0]])
def test_invalid_mask_length_or_soft_values_fail_closed(mask: list[float]) -> None:
    layout = resolve_dino_layout((12, 1, 1), num_views=6, num_materials=2)
    with pytest.raises(TensorLayoutError, match="reference_keep_mask"):
        dino_broadcast_values(mask, layout)


def test_non_tensor_output_fails_closed() -> None:
    with pytest.raises(TensorLayoutError, match="no tensor shape"):
        apply_dino_keep_mask(
            object(),
            KEEP_MASK,
            num_views=6,
            num_materials=2,
        )


def test_all_one_masks_are_exact_numeric_identities() -> None:
    refview = FakeTensor.sequential((2, 2, 12, 3), start=-20.0)
    dino = FakeTensor.sequential((24, 2, 3), start=-20.0)

    refview_result = apply_refview_keep_mask(
        refview,
        [1.0] * 6,
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    ).output
    dino_result = apply_dino_keep_mask(
        dino,
        [1.0] * 6,
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
    ).output

    assert refview_result is refview
    assert dino_result is dino
    assert refview_result.values == refview.values
    assert dino_result.values == dino.values
    assert all(math.copysign(1.0, new) == math.copysign(1.0, old) for new, old in zip(refview_result.values, refview.values))
