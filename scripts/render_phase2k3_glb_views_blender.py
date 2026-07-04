#!/usr/bin/env python3
"""Render Phase 2K.3 base/fine-tuned GLBs from fixed views.

Run with Blender:
  blender -b --python scripts/render_phase2k3_glb_views_blender.py -- \
    --cases-config configs/phase2k3_rendered_view_eval_cases.json \
    --output-root outputs/phase2k/rendered_view_eval_truepbr200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_MAX_EXTENT = 2.0
CAMERA_MARGIN = 1.35
DEFAULT_ELEVATION_DEGREES = 18.0


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 2K.3 GLB views.")
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clear_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_render_defaults(bpy: Any, resolution: int, background_color: list[float]) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    world = scene.world or bpy.data.worlds.new("phase2k3_world")
    scene.world = world
    world.color = tuple(float(value) for value in background_color)


def mesh_objects(bpy: Any) -> list[Any]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def vector_to_list(vector: Any) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def compute_world_bbox(meshes: list[Any]) -> dict[str, Any]:
    from mathutils import Vector

    corners: list[Any] = []
    for obj in meshes:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        zero = Vector((0.0, 0.0, 0.0))
        return {
            "min": zero,
            "max": zero,
            "center": zero,
            "extent": zero,
            "max_extent": 0.0,
            "diagonal": 0.0,
        }
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


def serialize_bbox(bbox: dict[str, Any]) -> dict[str, Any]:
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
    target_max_extent: float = NORMALIZED_MAX_EXTENT,
    margin: float = CAMERA_MARGIN,
) -> dict[str, Any]:
    meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    original_bbox = compute_world_bbox(meshes)
    center = original_bbox["center"]
    max_extent = original_bbox["max_extent"]
    applied_scale = target_max_extent / max_extent if max_extent > 0 else 1.0
    roots = imported_root_objects(imported_objects)
    for obj in roots:
        obj.location = (obj.location - center) * applied_scale
        obj.scale = tuple(component * applied_scale for component in obj.scale)
    bpy.context.view_layer.update()
    normalized_bbox = compute_world_bbox(meshes)
    normalized_max_extent = max(normalized_bbox["max_extent"], target_max_extent)
    normalized_diagonal = max(normalized_bbox["diagonal"], target_max_extent)
    ortho_scale = normalized_max_extent * margin
    camera_distance = max(4.0, normalized_diagonal * 2.0 + ortho_scale)
    return {
        "original_bbox": serialize_bbox(original_bbox),
        "normalized_bbox": serialize_bbox(normalized_bbox),
        "applied_scale": float(applied_scale),
        "target_max_extent": float(target_max_extent),
        "ortho_scale": float(ortho_scale),
        "margin": float(margin),
        "camera_distance": float(camera_distance),
        "camera_clip_start": 0.01,
        "camera_clip_end": float(camera_distance + normalized_diagonal + 20.0),
        "root_object_names": [obj.name for obj in roots],
    }


def make_camera(bpy: Any, ortho_scale: float, clip_end: float) -> Any:
    camera_data = bpy.data.cameras.new("phase2k3_camera")
    camera = bpy.data.objects.new("phase2k3_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_start = 0.01
    camera.data.clip_end = clip_end
    return camera


def look_at(obj: Any, target: Any) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def camera_pose_for_view(
    camera: Any,
    view_index: int,
    num_view: int,
    camera_distance: float,
) -> dict[str, Any]:
    from mathutils import Vector

    angle = (2.0 * math.pi * view_index) / num_view
    elevation = math.radians(DEFAULT_ELEVATION_DEGREES)
    horizontal_radius = camera_distance * math.cos(elevation)
    height = camera_distance * math.sin(elevation)
    camera.location = Vector(
        (horizontal_radius * math.cos(angle), horizontal_radius * math.sin(angle), height)
    )
    target = Vector((0.0, 0.0, 0.0))
    look_at(camera, target)
    view_direction = (target - camera.location).normalized()
    return {
        "view_index": view_index,
        "azimuth_degrees": math.degrees(angle),
        "elevation_degrees": math.degrees(elevation),
        "camera_location": vector_to_list(camera.location),
        "view_direction": vector_to_list(view_direction),
        "transform_matrix": [[float(value) for value in row] for row in camera.matrix_world],
    }


def remove_lights(bpy: Any) -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def add_light(
    bpy: Any,
    light_type: str,
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float = 5.0,
) -> Any:
    light_data = bpy.data.lights.new(name, type=light_type)
    light_data.energy = energy
    if hasattr(light_data, "size"):
        light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    return light


def configure_lighting(bpy: Any, background_color: list[float]) -> None:
    remove_lights(bpy)
    world = bpy.context.scene.world or bpy.data.worlds.new("phase2k3_world")
    bpy.context.scene.world = world
    world.color = tuple(float(value) for value in background_color)
    add_light(bpy, "AREA", "phase2k3_area_light", (0.0, -3.0, 4.0), 450.0, size=5.0)


def render_png(bpy: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_variant(
    bpy: Any,
    input_glb: Path,
    output_dir: Path,
    view_ids: list[str],
    resolution: int,
    background_color: list[float],
) -> dict[str, Any]:
    clear_scene(bpy)
    set_render_defaults(bpy, resolution, background_color)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    imported_objects = list(bpy.context.scene.objects)
    normalize_info = normalize_scene(bpy, imported_objects)
    camera = make_camera(bpy, normalize_info["ortho_scale"], normalize_info["camera_clip_end"])
    configure_lighting(bpy, background_color)
    frames = []
    for view_index, view_id in enumerate(view_ids):
        frame = camera_pose_for_view(
            camera,
            view_index,
            len(view_ids),
            normalize_info["camera_distance"],
        )
        frame["view_id"] = view_id
        frame["file_path"] = str(output_dir / f"{view_id}.png")
        render_png(bpy, output_dir / f"{view_id}.png")
        frames.append(frame)
    return {
        "input_glb": str(input_glb),
        "output_dir": str(output_dir),
        "normalize_info": normalize_info,
        "frames": frames,
    }


def write_render_config(
    output_root: Path,
    config: dict[str, Any],
    render_reports: dict[str, Any],
) -> None:
    data = {
        "source_config": config,
        "view_convention": {
            "source": "Phase 1E normalized orthographic six-view turntable",
            "normalized_max_extent": NORMALIZED_MAX_EXTENT,
            "camera_margin": CAMERA_MARGIN,
            "elevation_degrees": DEFAULT_ELEVATION_DEGREES,
            "azimuth_degrees": [
                (360.0 * index) / len(config["view_ids"])
                for index, _ in enumerate(config["view_ids"])
            ],
            "lighting": "single area light at (0, -3, 4), energy 450, size 5",
            "background_color": config.get("background_color", [0.28, 0.28, 0.28]),
        },
        "render_reports": render_reports,
    }
    path = output_root / "render_config_used.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)
    config = load_config(args.cases_config)
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    view_ids = list(config.get("view_ids", []))
    resolution = int(config.get("render_resolution", 512))
    background_color = list(config.get("background_color", [0.28, 0.28, 0.28]))

    import bpy  # type: ignore

    render_reports: dict[str, Any] = {}
    for asset_id, case in config.get("cases", {}).items():
        render_reports[asset_id] = {}
        for variant, key in (("base", "base_glb"), ("finetuned", "finetuned_glb")):
            input_glb = resolve_project_path(str(case[key]))
            output_dir = output_root / "renders" / asset_id / variant
            report = render_variant(
                bpy,
                input_glb,
                output_dir,
                view_ids,
                resolution,
                background_color,
            )
            render_reports[asset_id][variant] = report
            print(f"rendered {asset_id} {variant}: {output_dir}")
    write_render_config(output_root, config, render_reports)
    print(f"render_config_used: {output_root / 'render_config_used.json'}")
    print("PHASE2K3_RENDERED_VIEWS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
