from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.view_selective_gating.camera_gate import (  # noqa: E402
    _clamped_dot_and_angle,
    _reference_keep_for_angle,
    CameraMetadataError,
    all_one_keep_mask,
    build_reference_keep_mask,
)


OFFICIAL_AZIMUTHS = [0.0, 90.0, 180.0, 270.0, 0.0, 180.0]
OFFICIAL_ELEVATIONS = [0.0, 0.0, 0.0, 0.0, 90.0, -90.0]
EXPECTED_DIRECTIONS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
]


def test_official_six_uses_normalized_3d_angular_separation() -> None:
    mask, records = build_reference_keep_mask(
        OFFICIAL_AZIMUTHS,
        OFFICIAL_ELEVATIONS,
        expected_num_views=6,
    )

    assert mask == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    assert [record["angular_separation_degrees"] for record in records] == pytest.approx(
        [0.0, 90.0, 180.0, 90.0, 90.0, 90.0],
        abs=1e-12,
    )
    assert [record["clamped_direction_dot_product"] for record in records] == pytest.approx(
        [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        abs=1e-12,
    )
    for record, expected_direction in zip(records, EXPECTED_DIRECTIONS):
        assert record["direction_unit_vector"] == pytest.approx(expected_direction, abs=1e-12)
        assert sum(value * value for value in record["direction_unit_vector"]) == pytest.approx(1.0)
        assert record["reference_direction_unit_vector"] == pytest.approx(
            EXPECTED_DIRECTIONS[0],
            abs=1e-12,
        )
        assert record["reference_camera_index"] == 0
        assert record["reference_elevation_degrees"] == 0.0
        assert record["reference_azimuth_degrees"] == 0.0
        assert record["reference_keep"] == mask[record["index"]]


def test_polar_azimuth_does_not_change_3d_separation_from_front() -> None:
    mask, records = build_reference_keep_mask(
        [0.0, 0.0, 137.0],
        [0.0, 90.0, -90.0],
        expected_num_views=3,
    )

    assert mask == [1.0, 1.0, 1.0]
    assert [record["angular_separation_degrees"] for record in records] == pytest.approx(
        [0.0, 90.0, 90.0],
        abs=1e-12,
    )


def test_threshold_is_inclusive_and_has_a_strict_keep_side() -> None:
    below = 120.0 - 1e-6
    above = 120.0 + 1e-6
    mask, records = build_reference_keep_mask(
        [0.0, below, above, -above],
        [0.0] * 4,
        expected_num_views=4,
    )

    assert mask == [1.0, 1.0, 0.0, 0.0]
    assert records[1]["angular_separation_degrees"] < 120.0
    assert records[2]["angular_separation_degrees"] > 120.0
    assert records[3]["angular_separation_degrees"] > 120.0


def test_literal_theta_threshold_does_not_widen_near_boundary() -> None:
    threshold = 120.0
    immediately_below = math.nextafter(threshold, -math.inf)
    immediately_above = math.nextafter(threshold, math.inf)

    assert _reference_keep_for_angle(immediately_below, threshold) == 1.0
    assert _reference_keep_for_angle(threshold, threshold) == 0.0
    assert _reference_keep_for_angle(immediately_above, threshold) == 0.0


def test_dot_product_is_clamped_before_acos() -> None:
    high_dot, high_theta = _clamped_dot_and_angle(
        (1.0000000000000002, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    low_dot, low_theta = _clamped_dot_and_angle(
        (-1.0000000000000002, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )

    assert (high_dot, high_theta) == (1.0, 0.0)
    assert (low_dot, low_theta) == (-1.0, 180.0)


def test_custom_threshold_remains_hard_and_inclusive() -> None:
    mask, _records = build_reference_keep_mask(
        [0.0, 89.999, 90.0, 270.0],
        [0.0] * 4,
        suppression_threshold_degrees=90.0,
        expected_num_views=4,
    )

    assert mask == [1.0, 1.0, 0.0, 0.0]


def test_configured_reference_is_validated_but_direction_comes_from_live_index_zero() -> None:
    mask, records = build_reference_keep_mask(
        [10.0, 100.0, 190.0],
        [0.0, 0.0, 0.0],
        reference_azimuth_degrees=10.0,
        expected_num_views=3,
    )

    assert mask == [1.0, 1.0, 0.0]
    assert records[0]["reference_azimuth_degrees"] == 10.0
    assert records[0]["reference_direction_unit_vector"] == pytest.approx(
        records[0]["direction_unit_vector"],
        abs=1e-12,
    )




@pytest.mark.parametrize(
    ("azimuths", "elevations"),
    [
        ([90.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [45.0, 0.0]),
    ],
)
def test_reference_camera_must_be_selected_index_zero(
    azimuths: list[float],
    elevations: list[float],
) -> None:
    with pytest.raises(CameraMetadataError, match="index 0"):
        build_reference_keep_mask(
            azimuths,
            elevations,
            expected_num_views=2,
        )


def test_reference_camera_tolerance_is_explicit() -> None:
    mask, _records = build_reference_keep_mask(
        [0.0001, 180.0],
        [0.0, 0.0],
        expected_num_views=2,
        reference_camera_tolerance_degrees=0.001,
    )
    assert mask == [1.0, 0.0]

    with pytest.raises(CameraMetadataError, match="index 0"):
        build_reference_keep_mask(
            [0.0001, 180.0],
            [0.0, 0.0],
            expected_num_views=2,
            reference_camera_tolerance_degrees=0.00001,
        )


@pytest.mark.parametrize(
    "bad_value",
    [None, True, "0", float("nan"), float("inf"), -float("inf")],
)
def test_invalid_numeric_metadata_fails_closed(bad_value: object) -> None:
    with pytest.raises(CameraMetadataError):
        build_reference_keep_mask(
            [0.0, bad_value],  # type: ignore[list-item]
            [0.0, 0.0],
            expected_num_views=2,
        )


def test_missing_mismatched_or_unexpected_camera_counts_fail_closed() -> None:
    with pytest.raises(CameraMetadataError, match="at least one"):
        build_reference_keep_mask([], [], expected_num_views=None)
    with pytest.raises(CameraMetadataError, match="lengths differ"):
        build_reference_keep_mask([0.0], [0.0, 0.0])
    with pytest.raises(CameraMetadataError, match="camera count"):
        build_reference_keep_mask([0.0, 90.0], [0.0, 0.0], expected_num_views=6)
    with pytest.raises(CameraMetadataError, match="positive integer"):
        build_reference_keep_mask(
            [0.0], [0.0], expected_num_views=True  # type: ignore[arg-type]
        )
    with pytest.raises(CameraMetadataError, match="positive integer"):
        build_reference_keep_mask([0.0], [0.0], expected_num_views=0)


@pytest.mark.parametrize("threshold", [0.0, -1.0, 180.0001, math.inf, math.nan])
def test_invalid_suppression_threshold_fails_closed(threshold: float) -> None:
    with pytest.raises(CameraMetadataError):
        build_reference_keep_mask(
            [0.0],
            [0.0],
            suppression_threshold_degrees=threshold,
            expected_num_views=1,
        )


@pytest.mark.parametrize("tolerance", [-1.0, math.inf, math.nan])
def test_invalid_reference_tolerance_fails_closed(tolerance: float) -> None:
    with pytest.raises(CameraMetadataError):
        build_reference_keep_mask(
            [0.0],
            [0.0],
            reference_camera_tolerance_degrees=tolerance,
            expected_num_views=1,
        )


def test_noop_mask_is_all_ones_but_audits_3d_theta_and_vectors() -> None:
    mask, records = all_one_keep_mask(
        OFFICIAL_AZIMUTHS,
        OFFICIAL_ELEVATIONS,
        expected_num_views=6,
    )

    assert mask == [1.0] * 6
    assert [record["reference_keep"] for record in records] == mask
    assert [record["angular_separation_degrees"] for record in records] == pytest.approx(
        [0.0, 90.0, 180.0, 90.0, 90.0, 90.0],
        abs=1e-12,
    )
    for record in records:
        assert sum(
            value * value for value in record["direction_unit_vector"]
        ) == pytest.approx(1.0)


def test_noop_still_rejects_bad_shape_values_and_reference_order() -> None:
    with pytest.raises(CameraMetadataError, match="lengths differ"):
        all_one_keep_mask([0.0, 90.0], [0.0], expected_num_views=2)
    with pytest.raises(CameraMetadataError):
        all_one_keep_mask(
            [0.0, float("nan")], [0.0, 0.0], expected_num_views=2
        )
    with pytest.raises(CameraMetadataError, match="index 0"):
        all_one_keep_mask([90.0, 0.0], [0.0, 0.0], expected_num_views=2)
