#!/usr/bin/env python3
"""Render baseline and candidate GLBs under the canonical fixed-view protocol."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _evaluation import (  # noqa: E402
    asset_label,
    load_cases_manifest,
    load_evaluation_config,
    resolve_project_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GLBs using the canonical fixed-view evaluation protocol."
    )
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def clear_scene(bpy: Any) -> None:
    """Remove scene objects and their orphaned data before the next GLB."""

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection_name in (
        "meshes",
        "materials",
        "images",
        "textures",
        "cameras",
        "lights",
        "curves",
    ):
        collection = getattr(bpy.data, collection_name)
        for block in list(collection):
            collection.remove(block, do_unlink=True)
    for _ in range(2):
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except (AttributeError, RuntimeError, TypeError):
            break


def set_render_settings(bpy: Any, rendering: Mapping[str, Any]) -> str:
    scene = bpy.context.scene
    width, height = rendering["resolution"]
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = int(rendering["resolution_percentage"])
    scene.render.image_settings.file_format = str(rendering["file_format"])
    scene.render.image_settings.color_mode = str(rendering["color_mode"])
    scene.render.film_transparent = bool(rendering["film_transparent"])
    scene.view_settings.view_transform = str(rendering["view_transform"])
    scene.view_settings.look = str(rendering["look"])
    scene.view_settings.exposure = float(rendering["exposure"])
    scene.view_settings.gamma = float(rendering["gamma"])

    requested_engine = str(rendering["engine"])
    fallback_engine = str(rendering["engine_fallback"])
    try:
        scene.render.engine = requested_engine
        selected_engine = requested_engine
    except (TypeError, ValueError):
        scene.render.engine = fallback_engine
        selected_engine = fallback_engine

    world = scene.world or bpy.data.worlds.new("fixed_view_world")
    scene.world = world
    world.color = tuple(float(value) for value in rendering["background_color"])
    return selected_engine


def vector_to_list(vector: Any) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def compute_world_bbox(meshes: list[Any]) -> dict[str, Any]:
    from mathutils import Vector

    corners: list[Any] = []
    for obj in meshes:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        raise ValueError("imported GLB contains no mesh geometry")
    bbox_min = Vector(
        (min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners))
    )
    bbox_max = Vector(
        (max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners))
    )
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    return {
        "min": bbox_min,
        "max": bbox_max,
        "center": center,
        "extent": extent,
        "max_extent": float(max(extent.x, extent.y, extent.z)),
        "diagonal": float(extent.length),
    }


def serialize_bbox(bbox: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "min": vector_to_list(bbox["min"]),
        "max": vector_to_list(bbox["max"]),
        "center": vector_to_list(bbox["center"]),
        "extent": vector_to_list(bbox["extent"]),
        "max_extent": float(bbox["max_extent"]),
        "diagonal": float(bbox["diagonal"]),
    }


def imported_root_objects(imported_objects: list[Any]) -> list[Any]:
    imported_set = set(imported_objects)
    return [
        obj
        for obj in imported_objects
        if obj.parent is None or obj.parent not in imported_set
    ]


def normalize_scene(
    bpy: Any,
    imported_objects: list[Any],
    normalization: Mapping[str, Any],
    distance_policy: Mapping[str, Any],
    clip_end_margin: float,
) -> dict[str, Any]:
    meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    original_bbox = compute_world_bbox(meshes)
    center = original_bbox["center"]
    max_extent = original_bbox["max_extent"]
    if max_extent <= 0:
        raise ValueError("imported GLB has zero spatial extent")

    target_max_extent = float(normalization["target_max_extent"])
    camera_margin = float(normalization["camera_margin"])
    applied_scale = target_max_extent / max_extent
    roots = imported_root_objects(imported_objects)
    for obj in roots:
        obj.location = (obj.location - center) * applied_scale
        obj.scale = tuple(component * applied_scale for component in obj.scale)
    bpy.context.view_layer.update()

    normalized_bbox = compute_world_bbox(meshes)
    normalized_max_extent = max(normalized_bbox["max_extent"], target_max_extent)
    normalized_diagonal = max(normalized_bbox["diagonal"], target_max_extent)
    ortho_scale = normalized_max_extent * camera_margin
    camera_distance = max(
        float(distance_policy["minimum"]),
        normalized_diagonal * float(distance_policy["diagonal_multiplier"])
        + ortho_scale * float(distance_policy["ortho_scale_multiplier"]),
    )
    return {
        "original_bbox": serialize_bbox(original_bbox),
        "normalized_bbox": serialize_bbox(normalized_bbox),
        "applied_scale": float(applied_scale),
        "target_max_extent": target_max_extent,
        "camera_margin": camera_margin,
        "ortho_scale": float(ortho_scale),
        "camera_distance": float(camera_distance),
        "camera_clip_end": float(camera_distance + normalized_diagonal + clip_end_margin),
        "root_object_names": [obj.name for obj in roots],
    }


def make_camera(
    bpy: Any,
    camera_config: Mapping[str, Any],
    ortho_scale: float,
    clip_end: float,
) -> Any:
    camera_data = bpy.data.cameras.new("fixed_view_camera")
    camera = bpy.data.objects.new("fixed_view_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_start = float(camera_config["clip_start"])
    camera.data.clip_end = clip_end
    return camera


def camera_pose_for_view(
    camera: Any,
    pose: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    camera_distance: float,
) -> dict[str, Any]:
    from mathutils import Vector

    azimuth = math.radians(float(pose["azimuth_degrees"]))
    elevation = math.radians(float(pose["elevation_degrees"]))
    target = Vector(tuple(float(value) for value in camera_config["target"]))
    horizontal_radius = camera_distance * math.cos(elevation)
    offset = Vector(
        (
            horizontal_radius * math.cos(azimuth),
            horizontal_radius * math.sin(azimuth),
            camera_distance * math.sin(elevation),
        )
    )
    camera.location = target + offset
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat(
        str(camera_config["tracking_axis"]),
        str(camera_config["up_axis"]),
    ).to_euler()
    view_direction = direction.normalized()
    return {
        "view_id": str(pose["view_id"]),
        "azimuth_degrees": float(pose["azimuth_degrees"]),
        "elevation_degrees": float(pose["elevation_degrees"]),
        "camera_location": vector_to_list(camera.location),
        "view_direction": vector_to_list(view_direction),
        "transform_matrix": [
            [float(value) for value in row]
            for row in camera.matrix_world
        ],
    }


def remove_lights(bpy: Any) -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def configure_lighting(
    bpy: Any,
    background_color: list[float],
    light_configs: list[Mapping[str, Any]],
) -> None:
    remove_lights(bpy)
    world = bpy.context.scene.world or bpy.data.worlds.new("fixed_view_world")
    bpy.context.scene.world = world
    world.color = tuple(float(value) for value in background_color)
    for light_config in light_configs:
        light_data = bpy.data.lights.new(
            str(light_config["name"]),
            type=str(light_config["type"]),
        )
        light_data.energy = float(light_config["energy"])
        if hasattr(light_data, "size"):
            light_data.size = float(light_config["size"])
        light = bpy.data.objects.new(str(light_config["name"]), light_data)
        bpy.context.collection.objects.link(light)
        light.location = tuple(float(value) for value in light_config["location"])


def render_png(bpy: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_variant(
    bpy: Any,
    input_glb: Path,
    output_dir: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if not input_glb.is_file():
        raise FileNotFoundError(f"input GLB does not exist: {input_glb}")

    camera_config = evaluation["camera"]
    rendering = evaluation["rendering"]
    clear_scene(bpy)
    selected_engine = set_render_settings(bpy, rendering)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    imported_objects = list(bpy.context.scene.objects)
    normalize_info = normalize_scene(
        bpy,
        imported_objects,
        rendering["normalization"],
        camera_config["distance_policy"],
        float(camera_config["clip_end_margin"]),
    )
    camera = make_camera(
        bpy,
        camera_config,
        normalize_info["ortho_scale"],
        normalize_info["camera_clip_end"],
    )
    configure_lighting(
        bpy,
        list(rendering["background_color"]),
        list(rendering["lights"]),
    )

    frames: list[dict[str, Any]] = []
    for pose in camera_config["poses"]:
        view_id = str(pose["view_id"])
        frame = camera_pose_for_view(
            camera,
            pose,
            camera_config,
            normalize_info["camera_distance"],
        )
        frame["file_path"] = str(output_dir / f"{view_id}.png")
        render_png(bpy, output_dir / f"{view_id}.png")
        frames.append(frame)
    return {
        "input_glb": str(input_glb),
        "output_dir": str(output_dir),
        "selected_render_engine": selected_engine,
        "normalization": normalize_info,
        "frames": frames,
    }


def write_render_manifest(
    output_root: Path,
    *,
    evaluation_path: Path,
    cases_path: Path,
    evaluation: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
    render_reports: Mapping[str, Any],
) -> None:
    data = {
        "schema_version": 1,
        "evaluation_id": evaluation["evaluation_id"],
        "evaluation_config": str(evaluation_path),
        "cases_config": str(cases_path),
        "baseline_label": cases_manifest["baseline_label"],
        "candidate_label": cases_manifest["candidate_label"],
        "camera": evaluation["camera"],
        "rendering": evaluation["rendering"],
        "render_reports": render_reports,
    }
    path = output_root / "render_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)
    evaluation_path, evaluation = load_evaluation_config(args.evaluation_config)
    cases_path, cases_manifest = load_cases_manifest(args.cases_config)
    output_root = resolve_project_path(str(args.output_root))

    import bpy  # type: ignore

    render_reports: dict[str, Any] = {}
    for asset_id, case in cases_manifest["cases"].items():
        render_reports[asset_id] = {
            "asset_label": asset_label(cases_manifest, asset_id),
        }
        for variant, key in (("baseline", "baseline_glb"), ("candidate", "candidate_glb")):
            input_glb = resolve_project_path(str(case[key]))
            output_dir = output_root / "renders" / asset_id / variant
            render_reports[asset_id][variant] = render_variant(
                bpy,
                input_glb,
                output_dir,
                evaluation,
            )
            print(f"rendered {asset_id} {variant}: {output_dir}")

    write_render_manifest(
        output_root,
        evaluation_path=evaluation_path,
        cases_path=cases_path,
        evaluation=evaluation,
        cases_manifest=cases_manifest,
        render_reports=render_reports,
    )
    render_manifest_path = output_root / "render_manifest.json"
    print(f"render_manifest: {render_manifest_path}")
    print("FIXED_VIEW_RENDERING_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
