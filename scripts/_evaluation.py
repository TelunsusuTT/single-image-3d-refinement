#!/usr/bin/env python3
"""Validation and aggregation helpers for the fixed-view evaluation contract."""

from __future__ import annotations

import json
import math
import statistics
import string
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ID = "fixed_view_rerendering"
VIEW_IDS = ("000", "001", "002", "003", "004", "005")
VIEW_GROUP_NAMES = ("all", "front", "conditioning", "non_front")
METRIC_NAMES = ("mae", "rmse", "psnr", "ssim_like")
PAIR_NAMES = (
    "baseline_vs_candidate",
    "baseline_vs_reference",
    "candidate_vs_reference",
)


class EvaluationContractError(ValueError):
    """Raised when evaluation inputs do not satisfy the public contract."""


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    label: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise EvaluationContractError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise EvaluationContractError(f"{label} has unsupported fields: {sorted(unknown)}")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationContractError(f"{label} must be a non-empty string")
    return value


def _require_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationContractError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationContractError(f"{label} must be finite")
    if positive and result <= 0:
        raise EvaluationContractError(f"{label} must be positive")
    return result


def _require_vector(
    value: Any,
    label: str,
    *,
    length: int,
    unit_interval: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise EvaluationContractError(f"{label} must be a {length}-element list")
    result = [_require_number(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if unit_interval and any(item < 0 or item > 1 for item in result):
        raise EvaluationContractError(f"{label} values must be in [0, 1]")
    return result


def load_json_object(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvaluationContractError(f"{label} does not exist: {source}") from None
    except json.JSONDecodeError as exc:
        raise EvaluationContractError(f"invalid JSON in {label} {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationContractError(f"{label} must contain a JSON object: {source}")
    return source, payload


def validate_evaluation_config_payload(config: Mapping[str, Any]) -> None:
    _require_exact_keys(
        config,
        "evaluation config",
        required={
            "schema_version",
            "evaluation_id",
            "display_name",
            "description",
            "camera",
            "rendering",
            "view_groups",
            "metrics",
            "stages",
        },
    )
    if config["schema_version"] != 1:
        raise EvaluationContractError("evaluation config schema_version must be 1")
    if config["evaluation_id"] != EVALUATION_ID:
        raise EvaluationContractError(f"evaluation_id must be {EVALUATION_ID!r}")
    _require_text(config["display_name"], "display_name")
    _require_text(config["description"], "description")

    camera = _require_object(config["camera"], "camera")
    _require_exact_keys(
        camera,
        "camera",
        required={
            "projection",
            "tracking_axis",
            "up_axis",
            "target",
            "clip_start",
            "clip_end_margin",
            "distance_policy",
            "poses",
        },
    )
    if camera["projection"] != "orthographic":
        raise EvaluationContractError("camera.projection must be orthographic")
    if camera["tracking_axis"] != "-Z" or camera["up_axis"] != "Y":
        raise EvaluationContractError("camera axes must use tracking_axis=-Z and up_axis=Y")
    _require_vector(camera["target"], "camera.target", length=3)
    _require_number(camera["clip_start"], "camera.clip_start", positive=True)
    _require_number(camera["clip_end_margin"], "camera.clip_end_margin", positive=True)

    distance_policy = _require_object(camera["distance_policy"], "camera.distance_policy")
    _require_exact_keys(
        distance_policy,
        "camera.distance_policy",
        required={"minimum", "diagonal_multiplier", "ortho_scale_multiplier"},
    )
    for key in ("minimum", "diagonal_multiplier", "ortho_scale_multiplier"):
        _require_number(distance_policy[key], f"camera.distance_policy.{key}", positive=True)

    poses = camera["poses"]
    if not isinstance(poses, list) or len(poses) != len(VIEW_IDS):
        raise EvaluationContractError("camera.poses must contain exactly six poses")
    pose_ids: list[str] = []
    for index, raw_pose in enumerate(poses):
        pose = _require_object(raw_pose, f"camera.poses[{index}]")
        _require_exact_keys(
            pose,
            f"camera.poses[{index}]",
            required={"view_id", "azimuth_degrees", "elevation_degrees"},
        )
        pose_ids.append(_require_text(pose["view_id"], f"camera.poses[{index}].view_id"))
        _require_number(pose["azimuth_degrees"], f"camera.poses[{index}].azimuth_degrees")
        _require_number(pose["elevation_degrees"], f"camera.poses[{index}].elevation_degrees")
    if tuple(pose_ids) != VIEW_IDS:
        raise EvaluationContractError(f"camera pose IDs must be ordered as {list(VIEW_IDS)}")

    rendering = _require_object(config["rendering"], "rendering")
    _require_exact_keys(
        rendering,
        "rendering",
        required={
            "resolution",
            "resolution_percentage",
            "file_format",
            "color_mode",
            "film_transparent",
            "engine",
            "engine_fallback",
            "view_transform",
            "look",
            "exposure",
            "gamma",
            "background_color",
            "normalization",
            "lights",
        },
    )
    resolution = rendering["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in resolution)
    ):
        raise EvaluationContractError("rendering.resolution must contain two positive integers")
    if rendering["resolution_percentage"] != 100:
        raise EvaluationContractError("rendering.resolution_percentage must be 100")
    if rendering["file_format"] != "PNG" or rendering["color_mode"] != "RGB":
        raise EvaluationContractError("the evaluator supports PNG RGB output")
    if not isinstance(rendering["film_transparent"], bool):
        raise EvaluationContractError("rendering.film_transparent must be a boolean")
    for key in ("engine", "engine_fallback", "view_transform", "look"):
        _require_text(rendering[key], f"rendering.{key}")
    _require_number(rendering["exposure"], "rendering.exposure")
    _require_number(rendering["gamma"], "rendering.gamma", positive=True)
    _require_vector(
        rendering["background_color"],
        "rendering.background_color",
        length=3,
        unit_interval=True,
    )

    normalization = _require_object(rendering["normalization"], "rendering.normalization")
    _require_exact_keys(
        normalization,
        "rendering.normalization",
        required={"target_max_extent", "camera_margin"},
    )
    _require_number(
        normalization["target_max_extent"],
        "rendering.normalization.target_max_extent",
        positive=True,
    )
    _require_number(
        normalization["camera_margin"],
        "rendering.normalization.camera_margin",
        positive=True,
    )

    lights = rendering["lights"]
    if not isinstance(lights, list) or not lights:
        raise EvaluationContractError("rendering.lights must contain at least one light")
    light_names: set[str] = set()
    for index, raw_light in enumerate(lights):
        light = _require_object(raw_light, f"rendering.lights[{index}]")
        _require_exact_keys(
            light,
            f"rendering.lights[{index}]",
            required={"name", "type", "location", "energy", "size"},
        )
        name = _require_text(light["name"], f"rendering.lights[{index}].name")
        if name in light_names:
            raise EvaluationContractError(f"duplicate light name: {name}")
        light_names.add(name)
        _require_text(light["type"], f"rendering.lights[{index}].type")
        _require_vector(light["location"], f"rendering.lights[{index}].location", length=3)
        _require_number(light["energy"], f"rendering.lights[{index}].energy", positive=True)
        _require_number(light["size"], f"rendering.lights[{index}].size", positive=True)

    groups = _require_object(config["view_groups"], "view_groups")
    _require_exact_keys(groups, "view_groups", required=set(VIEW_GROUP_NAMES))
    normalized_groups: dict[str, tuple[str, ...]] = {}
    for name in VIEW_GROUP_NAMES:
        raw_group = groups[name]
        if not isinstance(raw_group, list) or not raw_group:
            raise EvaluationContractError(f"view_groups.{name} must be a non-empty list")
        if not all(isinstance(item, str) for item in raw_group):
            raise EvaluationContractError(f"view_groups.{name} must contain view IDs")
        group = tuple(raw_group)
        if len(group) != len(set(group)):
            raise EvaluationContractError(f"view_groups.{name} contains duplicate view IDs")
        if not set(group).issubset(VIEW_IDS):
            raise EvaluationContractError(f"view_groups.{name} contains an unknown view ID")
        normalized_groups[name] = group
    if normalized_groups["all"] != VIEW_IDS:
        raise EvaluationContractError("view_groups.all must match the ordered camera pose IDs")
    if set(normalized_groups["front"]) & set(normalized_groups["non_front"]):
        raise EvaluationContractError("front and non_front groups must be disjoint")
    if set(normalized_groups["front"]) | set(normalized_groups["non_front"]) != set(VIEW_IDS):
        raise EvaluationContractError("front and non_front groups must partition all views")
    if not set(normalized_groups["conditioning"]).issubset(normalized_groups["front"]):
        raise EvaluationContractError("conditioning views must be a subset of front views")

    metrics = _require_object(config["metrics"], "metrics")
    _require_exact_keys(
        metrics,
        "metrics",
        required={"enabled", "data_range", "ssim_like"},
    )
    if tuple(metrics["enabled"]) != METRIC_NAMES:
        raise EvaluationContractError(f"metrics.enabled must be ordered as {list(METRIC_NAMES)}")
    _require_number(metrics["data_range"], "metrics.data_range", positive=True)
    ssim = _require_object(metrics["ssim_like"], "metrics.ssim_like")
    _require_exact_keys(ssim, "metrics.ssim_like", required={"implementation", "k1", "k2"})
    if ssim["implementation"] != "global_grayscale":
        raise EvaluationContractError(
            "metrics.ssim_like.implementation must be global_grayscale"
        )
    _require_number(ssim["k1"], "metrics.ssim_like.k1", positive=True)
    _require_number(ssim["k2"], "metrics.ssim_like.k2", positive=True)
    _require_object(config["stages"], "stages")


def load_evaluation_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source, config = load_json_object(path, "evaluation config")
    validate_evaluation_config_payload(config)
    return source, config


def _validate_reference_template(template: str) -> None:
    fields: list[str] = []
    try:
        for _literal, field_name, format_spec, conversion in string.Formatter().parse(template):
            if field_name:
                if format_spec or conversion:
                    raise EvaluationContractError(
                        "reference_image_path_template does not support format specs or conversions"
                    )
                fields.append(field_name)
    except ValueError as exc:
        raise EvaluationContractError(
            f"invalid reference_image_path_template: {exc}"
        ) from exc
    if set(fields) != {"asset_id", "view_id"}:
        raise EvaluationContractError(
            "reference_image_path_template must use {asset_id} and {view_id}"
        )


def validate_cases_manifest_payload(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(
        manifest,
        "cases manifest",
        required={
            "schema_version",
            "baseline_label",
            "candidate_label",
            "reference_image_path_template",
            "cases",
        },
    )
    if manifest["schema_version"] != 1:
        raise EvaluationContractError("cases manifest schema_version must be 1")
    baseline_label = _require_text(manifest["baseline_label"], "baseline_label")
    candidate_label = _require_text(manifest["candidate_label"], "candidate_label")
    if baseline_label == candidate_label:
        raise EvaluationContractError("baseline_label and candidate_label must differ")
    template = _require_text(
        manifest["reference_image_path_template"],
        "reference_image_path_template",
    )
    _validate_reference_template(template)

    cases = _require_object(manifest["cases"], "cases")
    if not cases:
        raise EvaluationContractError("cases must not be empty")
    for asset_id, raw_case in cases.items():
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise EvaluationContractError("each case key must be a non-empty asset ID")
        if Path(asset_id).name != asset_id or asset_id in {".", ".."}:
            raise EvaluationContractError(f"asset ID must be one path-safe segment: {asset_id!r}")
        case = _require_object(raw_case, f"cases.{asset_id}")
        _require_exact_keys(
            case,
            f"cases.{asset_id}",
            required={"baseline_glb", "candidate_glb"},
            optional={"asset_label"},
        )
        _require_text(case["baseline_glb"], f"cases.{asset_id}.baseline_glb")
        _require_text(case["candidate_glb"], f"cases.{asset_id}.candidate_glb")
        if "asset_label" in case:
            _require_text(case["asset_label"], f"cases.{asset_id}.asset_label")


def load_cases_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source, manifest = load_json_object(path, "cases manifest")
    validate_cases_manifest_payload(manifest)
    return source, manifest


def resolve_project_path(path_text: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def reference_image_path(
    manifest: Mapping[str, Any],
    asset_id: str,
    view_id: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    template = str(manifest["reference_image_path_template"])
    return resolve_project_path(
        template.format(asset_id=asset_id, view_id=view_id),
        project_root,
    )


def asset_label(manifest: Mapping[str, Any], asset_id: str) -> str:
    case = _require_object(manifest["cases"][asset_id], f"cases.{asset_id}")
    return str(case.get("asset_label", asset_id))


def expected_image_size(config: Mapping[str, Any]) -> tuple[int, int]:
    resolution = config["rendering"]["resolution"]
    return int(resolution[0]), int(resolution[1])


def require_complete_view_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    repetitions: int = 1,
) -> None:
    expected = {
        view_id: repetitions
        for view_id in config["view_groups"]["all"]
    }
    actual = Counter(str(row.get("view_id")) for row in rows)
    if actual != Counter(expected):
        raise EvaluationContractError(
            f"metric rows do not match configured views: expected {expected}, got {dict(actual)}"
        )


def _mean_pair(
    rows: Sequence[Mapping[str, Any]],
    pair_name: str,
    metric_names: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values: list[float] = []
        has_infinite_psnr = False
        for row in rows:
            pair = _require_object(row.get(pair_name), f"view.{pair_name}")
            value = pair.get(metric_name)
            if metric_name == "psnr" and value is None:
                if pair.get("psnr_is_infinite") is not True:
                    raise EvaluationContractError(
                        f"{pair_name}.psnr may be null only when psnr_is_infinite is true"
                    )
                has_infinite_psnr = True
                continue
            values.append(_require_number(value, f"{pair_name}.{metric_name}"))
        if metric_name == "psnr" and has_infinite_psnr:
            summary[metric_name] = None
            summary["psnr_is_infinite"] = True
        else:
            summary[metric_name] = statistics.fmean(values)
            if metric_name == "psnr":
                summary["psnr_is_infinite"] = False
    return summary


def summarize_view_groups(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    metric_names = list(config["metrics"]["enabled"])
    result: dict[str, Any] = {}
    for group_name in VIEW_GROUP_NAMES:
        view_ids = list(config["view_groups"][group_name])
        selected = [row for row in rows if row.get("view_id") in view_ids]
        if not selected:
            raise EvaluationContractError(f"view group {group_name!r} has no metric rows")
        pairs = {
            pair_name: _mean_pair(selected, pair_name, metric_names)
            for pair_name in PAIR_NAMES
        }
        baseline = pairs["baseline_vs_reference"]
        candidate = pairs["candidate_vs_reference"]
        delta = {
            metric_name: (
                None
                if baseline[metric_name] is None or candidate[metric_name] is None
                else float(candidate[metric_name]) - float(baseline[metric_name])
            )
            for metric_name in metric_names
        }
        result[group_name] = {
            "view_ids": view_ids,
            "view_count": len(selected),
            **pairs,
            "candidate_minus_baseline": delta,
        }
    return result
