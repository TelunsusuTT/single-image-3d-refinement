"""Fail-closed 3D camera-direction policy for view-selective gating.

The official six generation cameras are compared with the live front camera in
normalized 3D direction space. This gives top and bottom cameras an unambiguous
90-degree separation from the front without assigning a fictitious polar
azimuth role.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Sequence

class CameraMetadataError(ValueError):
    """Raised when generation-camera metadata is incomplete or inconsistent."""



@dataclass(frozen=True)
class CameraGateRecord:
    index: int
    azimuth_degrees: float
    elevation_degrees: float
    direction_unit_vector: tuple[float, float, float]
    reference_camera_index: int
    reference_azimuth_degrees: float
    reference_elevation_degrees: float
    reference_direction_unit_vector: tuple[float, float, float]
    clamped_direction_dot_product: float
    angular_separation_degrees: float
    reference_keep: float


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CameraMetadataError(f"{label} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise CameraMetadataError(f"{label} must be finite, got {value!r}")
    return result



def _normalized_camera_direction(
    azimuth_degrees: float,
    elevation_degrees: float,
) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    raw = (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )
    norm = math.sqrt(sum(value * value for value in raw))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CameraMetadataError(
            "camera direction could not be normalized for "
            f"elevation={elevation_degrees!r}, azimuth={azimuth_degrees!r}"
        )
    normalized = tuple(value / norm for value in raw)
    snapped = tuple(0.0 if abs(value) < 1e-15 else value for value in normalized)
    snapped_norm = math.sqrt(sum(value * value for value in snapped))
    if not math.isfinite(snapped_norm) or snapped_norm <= 0.0:
        raise CameraMetadataError("normalized camera direction became invalid")
    return tuple(value / snapped_norm for value in snapped)  # type: ignore[return-value]


def _clamped_dot_and_angle(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float]:
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    if not math.isfinite(dot):
        raise CameraMetadataError("camera-direction dot product is not finite")
    clamped = max(-1.0, min(1.0, dot))
    theta = math.degrees(math.acos(clamped))
    if not math.isfinite(theta):
        raise CameraMetadataError("camera angular separation is not finite")
    return clamped, theta


def _reference_keep_for_angle(
    theta_degrees: float,
    suppression_threshold_degrees: float,
) -> float:
    return 0.0 if theta_degrees >= suppression_threshold_degrees else 1.0


def _validated_camera_vectors(
    camera_azimuths: Sequence[float],
    camera_elevations: Sequence[float],
    *,
    expected_num_views: int | None,
) -> tuple[list[float], list[float]]:
    if isinstance(camera_azimuths, (str, bytes)) or isinstance(camera_elevations, (str, bytes)):
        raise CameraMetadataError("camera azimuths/elevations must be numeric sequences")
    azimuths = [
        _finite_number(value, f"camera_azimuths[{index}]")
        for index, value in enumerate(camera_azimuths)
    ]
    elevations = [
        _finite_number(value, f"camera_elevations[{index}]")
        for index, value in enumerate(camera_elevations)
    ]
    if not azimuths:
        raise CameraMetadataError("at least one generation camera is required")
    if len(azimuths) != len(elevations):
        raise CameraMetadataError(
            f"camera azimuth/elevation lengths differ: {len(azimuths)} != {len(elevations)}"
        )
    if expected_num_views is not None:
        if (
            isinstance(expected_num_views, bool)
            or not isinstance(expected_num_views, Integral)
            or int(expected_num_views) <= 0
        ):
            raise CameraMetadataError(
                f"expected_num_views must be a positive integer, got {expected_num_views!r}"
            )
        if len(azimuths) != int(expected_num_views):
            raise CameraMetadataError(
                f"selected generation-camera count {len(azimuths)} != expected {expected_num_views}"
            )
    return azimuths, elevations


def _validated_camera_directions(
    camera_azimuths: Sequence[float],
    camera_elevations: Sequence[float],
    *,
    reference_azimuth_degrees: float,
    reference_elevation_degrees: float,
    expected_num_views: int | None,
    reference_camera_tolerance_degrees: float,
) -> tuple[
    list[float],
    list[float],
    list[tuple[float, float, float]],
    tuple[float, float, float],
]:
    azimuths, elevations = _validated_camera_vectors(
        camera_azimuths,
        camera_elevations,
        expected_num_views=expected_num_views,
    )
    expected_reference_azimuth = _finite_number(
        reference_azimuth_degrees,
        "reference_azimuth_degrees",
    )
    expected_reference_elevation = _finite_number(
        reference_elevation_degrees,
        "reference_elevation_degrees",
    )
    tolerance = _finite_number(
        reference_camera_tolerance_degrees,
        "reference_camera_tolerance_degrees",
    )
    if tolerance < 0.0:
        raise CameraMetadataError(
            f"reference camera tolerance must be non-negative, got {tolerance}"
        )

    directions = [
        _normalized_camera_direction(azimuth, elevation)
        for azimuth, elevation in zip(azimuths, elevations)
    ]
    expected_reference_direction = _normalized_camera_direction(
        expected_reference_azimuth,
        expected_reference_elevation,
    )
    live_reference_direction = directions[0]
    _dot, reference_error = _clamped_dot_and_angle(
        live_reference_direction,
        expected_reference_direction,
    )
    if reference_error > tolerance:
        raise CameraMetadataError(
            "selected generation camera index 0 must be the live reference front "
            f"(elevation={expected_reference_elevation:g}, "
            f"azimuth={expected_reference_azimuth:g}); got "
            f"elevation={elevations[0]:g}, azimuth={azimuths[0]:g}, "
            f"angular_error={reference_error:g}"
        )
    return azimuths, elevations, directions, live_reference_direction


def _camera_gate_records(
    azimuths: Sequence[float],
    elevations: Sequence[float],
    directions: Sequence[tuple[float, float, float]],
    reference_direction: tuple[float, float, float],
    keep_mask: Sequence[float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (azimuth, elevation, direction, keep) in enumerate(
        zip(azimuths, elevations, directions, keep_mask)
    ):
        clamped_dot, theta = _clamped_dot_and_angle(reference_direction, direction)
        records.append(
            asdict(
                CameraGateRecord(
                    index=index,
                    azimuth_degrees=azimuth,
                    elevation_degrees=elevation,
                    direction_unit_vector=direction,
                    reference_camera_index=0,
                    reference_azimuth_degrees=azimuths[0],
                    reference_elevation_degrees=elevations[0],
                    reference_direction_unit_vector=reference_direction,
                    clamped_direction_dot_product=clamped_dot,
                    angular_separation_degrees=theta,
                    reference_keep=float(keep),
                )
            )
        )
    return records


def build_reference_keep_mask(
    camera_azimuths: Sequence[float],
    camera_elevations: Sequence[float],
    *,
    reference_azimuth_degrees: float = 0.0,
    reference_elevation_degrees: float = 0.0,
    suppression_threshold_degrees: float = 120.0,
    expected_num_views: int | None = None,
    reference_camera_tolerance_degrees: float = 1e-6,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Build the hard mask from live normalized 3D camera directions.

    Camera index 0 must be the configured live front. 1.0 keeps the
    reference/DINO residual and 0.0 suppresses it. Angular separation is
    computed with a clamped dot product and acos; the threshold is inclusive.
    """

    azimuths, elevations, directions, reference_direction = _validated_camera_directions(
        camera_azimuths,
        camera_elevations,
        reference_azimuth_degrees=reference_azimuth_degrees,
        reference_elevation_degrees=reference_elevation_degrees,
        expected_num_views=expected_num_views,
        reference_camera_tolerance_degrees=reference_camera_tolerance_degrees,
    )
    threshold = _finite_number(
        suppression_threshold_degrees,
        "suppression_threshold_degrees",
    )
    if not (0.0 < threshold <= 180.0):
        raise CameraMetadataError(
            f"suppression threshold must be in (0, 180], got {threshold}"
        )

    mask: list[float] = []
    for direction in directions:
        _dot, theta = _clamped_dot_and_angle(reference_direction, direction)
        mask.append(_reference_keep_for_angle(theta, threshold))
    records = _camera_gate_records(
        azimuths,
        elevations,
        directions,
        reference_direction,
        mask,
    )
    return mask, records


def all_one_keep_mask(
    camera_azimuths: Sequence[float],
    camera_elevations: Sequence[float],
    *,
    reference_azimuth_degrees: float = 0.0,
    reference_elevation_degrees: float = 0.0,
    expected_num_views: int | None = None,
    reference_camera_tolerance_degrees: float = 1e-6,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Validate and audit live 3D metadata, then return the explicit no-op."""

    azimuths, elevations, directions, reference_direction = _validated_camera_directions(
        camera_azimuths,
        camera_elevations,
        reference_azimuth_degrees=reference_azimuth_degrees,
        reference_elevation_degrees=reference_elevation_degrees,
        expected_num_views=expected_num_views,
        reference_camera_tolerance_degrees=reference_camera_tolerance_degrees,
    )
    mask = [1.0] * len(azimuths)
    records = _camera_gate_records(
        azimuths,
        elevations,
        directions,
        reference_direction,
        mask,
    )
    return mask, records
