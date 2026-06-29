#!/usr/bin/env python3
"""Render one GLB into a minimal Hunyuan3D-Paint-style training sample.

Run with Blender:
  blender --background --python scripts/blender_render_hy3dpaint_example.py -- \
    --input-glb PATH --sample-name NAME --out-root PATH --num-view 6 --resolution 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


NORMALIZED_MAX_EXTENT = 2.0
CAMERA_MARGIN = 1.35


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one GLB into a minimal Hunyuan3D-Paint sample."
    )
    parser.add_argument("--input-glb", required=True, type=Path)
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--qa-root", type=Path)
    return parser.parse_args(argv)


def clear_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_render_defaults(bpy: Any, resolution: int) -> None:
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


def mesh_objects(bpy: Any) -> list[Any]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def vector_to_list(vector: Any) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def compute_world_bbox(meshes: list[Any]) -> dict[str, Any]:
    from mathutils import Vector

    corners: list[Vector] = []
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
        "view_directions": [],
    }


def make_camera(bpy: Any, ortho_scale: float, clip_end: float) -> Any:
    camera_data = bpy.data.cameras.new("phase1e_camera")
    camera = bpy.data.objects.new("phase1e_camera", camera_data)
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
    camera: Any, view_index: int, num_view: int, camera_distance: float
) -> dict[str, Any]:
    from mathutils import Vector

    angle = (2.0 * math.pi * view_index) / num_view
    elevation = math.radians(18.0)
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
    size: float = 4.0,
) -> Any:
    light_data = bpy.data.lights.new(name, type=light_type)
    light_data.energy = energy
    if hasattr(light_data, "size"):
        light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    return light


def configure_light_mode(bpy: Any, mode: str) -> None:
    remove_lights(bpy)
    world = bpy.context.scene.world or bpy.data.worlds.new("phase1e_world")
    bpy.context.scene.world = world
    world.color = (1.0, 1.0, 1.0)

    if mode == "AL":
        world.color = (0.28, 0.28, 0.28)
        add_light(bpy, "AREA", "phase1e_area_light", (0.0, -3.0, 4.0), 450.0, size=5.0)
    elif mode == "ENVMAP":
        world.color = (0.65, 0.68, 0.75)
        add_light(bpy, "SUN", "phase1e_env_sun", (0.0, 0.0, 4.0), 1.6)
    elif mode == "PL":
        world.color = (0.03, 0.03, 0.03)
        add_light(bpy, "POINT", "phase1e_point_light", (2.5, -2.5, 2.8), 850.0)
    else:
        world.color = (1.0, 1.0, 1.0)
        add_light(bpy, "AREA", "phase1e_beauty_light", (0.0, -3.5, 4.0), 650.0, size=4.5)


def make_emission_material(bpy: Any, name: str, color: tuple[float, float, float, float]) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def make_geometry_material(bpy: Any, name: str, geometry_output: str) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    geometry = nodes.new(type="ShaderNodeNewGeometry")
    material.node_tree.links.new(geometry.outputs[geometry_output], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def save_material_slots(meshes: list[Any]) -> dict[str, list[Any]]:
    return {obj.name: [slot.material for slot in obj.material_slots] for obj in meshes}


def restore_material_slots(meshes: list[Any], saved: dict[str, list[Any]]) -> None:
    for obj in meshes:
        obj.data.materials.clear()
        for material in saved.get(obj.name, []):
            obj.data.materials.append(material)


def override_material(meshes: list[Any], material: Any) -> None:
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def render_png(bpy: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_texture_passes(
    bpy: Any,
    render_tex: Path,
    view_name: str,
    meshes: list[Any],
    saved_materials: dict[str, list[Any]],
) -> None:
    configure_light_mode(bpy, "beauty")
    restore_material_slots(meshes, saved_materials)
    render_png(bpy, render_tex / f"{view_name}.png")

    restore_material_slots(meshes, saved_materials)
    render_png(bpy, render_tex / f"{view_name}_albedo.png")

    mr_material = make_emission_material(bpy, f"phase1e_mr_{view_name}", (0.55, 0.35, 0.0, 1.0))
    override_material(meshes, mr_material)
    render_png(bpy, render_tex / f"{view_name}_mr.png")

    normal_material = make_geometry_material(bpy, f"phase1e_normal_{view_name}", "Normal")
    override_material(meshes, normal_material)
    render_png(bpy, render_tex / f"{view_name}_normal.png")

    pos_material = make_geometry_material(bpy, f"phase1e_pos_{view_name}", "Position")
    override_material(meshes, pos_material)
    render_png(bpy, render_tex / f"{view_name}_pos.png")
    restore_material_slots(meshes, saved_materials)


def render_condition_passes(
    bpy: Any,
    render_cond: Path,
    view_name: str,
    meshes: list[Any],
    saved_materials: dict[str, list[Any]],
) -> None:
    restore_material_slots(meshes, saved_materials)
    for mode in ("AL", "ENVMAP", "PL"):
        configure_light_mode(bpy, mode)
        render_png(bpy, render_cond / f"{view_name}_light_{mode}.png")


def configure_mask_mode(bpy: Any) -> None:
    remove_lights(bpy)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    world = scene.world or bpy.data.worlds.new("phase1e_world")
    scene.world = world
    world.color = (0.0, 0.0, 0.0)


def render_mask_pass(
    bpy: Any,
    qa_sample_dir: Path,
    view_name: str,
    meshes: list[Any],
    saved_materials: dict[str, list[Any]],
) -> None:
    configure_mask_mode(bpy)
    mask_material = make_emission_material(
        bpy, f"phase1e_mask_{view_name}", (1.0, 1.0, 1.0, 1.0)
    )
    override_material(meshes, mask_material)
    render_png(bpy, qa_sample_dir / f"{view_name}_mask.png")
    restore_material_slots(meshes, saved_materials)


def camera_angle_x(camera: Any) -> float:
    return float(camera.data.angle_x)


def write_transforms(path: Path, input_glb: Path, camera: Any, resolution: int, frames: list[dict[str, Any]]) -> None:
    data = {
        "input_glb": str(input_glb),
        "resolution": resolution,
        "camera_model": "orthographic",
        "camera_angle_x": camera_angle_x(camera),
        "ortho_scale": float(camera.data.ortho_scale),
        "frames": frames,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_camera_framing_report(
    path: Path, input_glb: Path, normalize_info: dict[str, Any], frames: list[dict[str, Any]]
) -> None:
    data = {
        "input_glb": str(input_glb),
        "original_bbox": normalize_info["original_bbox"],
        "normalized_bbox": normalize_info["normalized_bbox"],
        "applied_scale": normalize_info["applied_scale"],
        "target_max_extent": normalize_info["target_max_extent"],
        "ortho_scale": normalize_info["ortho_scale"],
        "margin": normalize_info["margin"],
        "camera_distance": normalize_info["camera_distance"],
        "camera_clip_start": normalize_info["camera_clip_start"],
        "camera_clip_end": normalize_info["camera_clip_end"],
        "root_object_names": normalize_info["root_object_names"],
        "view_directions": [frame["view_direction"] for frame in frames],
        "frames": frames,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_summary(
    path: Path,
    sample_dir: Path,
    input_glb: Path,
    num_view: int,
    resolution: int,
    normalize_info: dict[str, Any],
    qa_sample_dir: Path | None,
) -> None:
    data = {
        "sample_dir": str(sample_dir),
        "input_glb": str(input_glb),
        "num_view": num_view,
        "resolution": resolution,
        "render_tex_files_per_view": ["png", "albedo", "mr", "normal", "pos"],
        "render_cond_files_per_view": ["light_AL", "light_ENVMAP", "light_PL"],
        "normalize_info": normalize_info,
        "camera_framing_report": (
            str(qa_sample_dir / "camera_framing_report.json")
            if qa_sample_dir is not None
            else ""
        ),
        "qa_note": "QA artifacts are sidecar files outside the Hunyuan training sample.",
        "caveats": [
            "Phase 1E is a structure/rendering smoke test.",
            "Material passes are approximate Blender overrides, not final official-quality supervision.",
            "No Hunyuan training was run by this renderer.",
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)
    if args.num_view < 1:
        raise SystemExit("--num-view must be >= 1")
    if args.resolution < 1:
        raise SystemExit("--resolution must be >= 1")

    import bpy

    sample_dir = args.out_root / args.sample_name
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    qa_sample_dir = args.qa_root / args.sample_name if args.qa_root is not None else None
    render_tex.mkdir(parents=True, exist_ok=True)
    render_cond.mkdir(parents=True, exist_ok=True)
    if qa_sample_dir is not None:
        qa_sample_dir.mkdir(parents=True, exist_ok=True)

    clear_scene(bpy)
    set_render_defaults(bpy, args.resolution)
    bpy.ops.import_scene.gltf(filepath=str(args.input_glb))
    imported_objects = list(bpy.context.scene.objects)
    normalize_info = normalize_scene(bpy, imported_objects)

    meshes = mesh_objects(bpy)
    saved_materials = save_material_slots(meshes)
    camera = make_camera(bpy, normalize_info["ortho_scale"], normalize_info["camera_clip_end"])
    frames: list[dict[str, Any]] = []

    for view_index in range(args.num_view):
        view_name = f"{view_index:03d}"
        frame = camera_pose_for_view(
            camera, view_index, args.num_view, normalize_info["camera_distance"]
        )
        frame["file_path"] = f"{view_name}.png"
        frames.append(frame)
        render_texture_passes(bpy, render_tex, view_name, meshes, saved_materials)
        render_condition_passes(bpy, render_cond, view_name, meshes, saved_materials)
        if qa_sample_dir is not None:
            render_mask_pass(bpy, qa_sample_dir, view_name, meshes, saved_materials)

    write_transforms(render_tex / "transforms.json", args.input_glb, camera, args.resolution, frames)
    if qa_sample_dir is not None:
        write_camera_framing_report(
            qa_sample_dir / "camera_framing_report.json",
            args.input_glb,
            normalize_info,
            frames,
        )
    write_summary(
        sample_dir / "sample_summary.json",
        sample_dir,
        args.input_glb,
        args.num_view,
        args.resolution,
        normalize_info,
        qa_sample_dir,
    )
    print(f"wrote sample: {sample_dir}")
    print(f"render_tex: {render_tex}")
    print(f"render_cond: {render_cond}")
    if qa_sample_dir is not None:
        print(f"qa_dir: {qa_sample_dir}")
        print(f"camera_framing_report: {qa_sample_dir / 'camera_framing_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
