#!/usr/bin/env python3
"""Inspect a GLB/GLTF asset inside Blender and write lightweight reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a GLB/GLTF in Blender and summarize mesh/material metadata."
    )
    parser.add_argument("--input-glb", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def clear_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def image_names_from_material(material: Any) -> list[str]:
    names: list[str] = []
    if not material.use_nodes or material.node_tree is None:
        return names
    for node in material.node_tree.nodes:
        if getattr(node, "type", "") == "TEX_IMAGE":
            image = getattr(node, "image", None)
            if image is not None:
                names.append(image.name)
    return sorted(set(names))


def material_summary(bpy: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for material in sorted(bpy.data.materials, key=lambda item: item.name):
        summaries.append(
            {
                "name": material.name,
                "use_nodes": bool(material.use_nodes),
                "image_texture_names": image_names_from_material(material),
            }
        )
    return summaries


def mesh_object_summary(objects: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for obj in objects:
        mesh = obj.data if obj.type == "MESH" else None
        summaries.append(
            {
                "name": obj.name,
                "type": obj.type,
                "vertex_count": len(mesh.vertices) if mesh is not None else 0,
                "polygon_count": len(mesh.polygons) if mesh is not None else 0,
                "uv_layer_count": len(mesh.uv_layers) if mesh is not None else 0,
                "material_slot_count": len(obj.material_slots),
            }
        )
    return summaries


def bounding_box(objects: list[Any]) -> tuple[list[float], list[float]]:
    corners: list[tuple[float, float, float]] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ obj.matrix_world.to_translation().__class__(corner)
            corners.append((float(world_corner.x), float(world_corner.y), float(world_corner.z)))

    if not corners:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    mins = [min(corner[index] for corner in corners) for index in range(3)]
    maxs = [max(corner[index] for corner in corners) for index in range(3)]
    return mins, maxs


def build_report(bpy: Any, input_glb: Path, import_status: str, import_error: str) -> dict[str, Any]:
    objects = list(bpy.context.scene.objects)
    mesh_objects = [obj for obj in objects if obj.type == "MESH"]
    materials = list(bpy.data.materials)
    material_summaries = material_summary(bpy)

    total_vertices = sum(len(obj.data.vertices) for obj in mesh_objects)
    total_polygons = sum(len(obj.data.polygons) for obj in mesh_objects)
    total_triangles = sum(
        max(1, len(polygon.vertices) - 2)
        for obj in mesh_objects
        for polygon in obj.data.polygons
    )
    total_uv_layers = sum(len(obj.data.uv_layers) for obj in mesh_objects)
    texture_names = {
        name
        for material in material_summaries
        for name in material["image_texture_names"]
    }
    bbox_min, bbox_max = bounding_box(mesh_objects)

    warnings = {
        "no_mesh": len(mesh_objects) == 0,
        "no_uv": total_uv_layers == 0,
        "no_material": len(materials) == 0,
        "no_texture_image": len(texture_names) == 0,
    }

    return {
        "input_glb": str(input_glb),
        "import_status": import_status,
        "import_error": import_error,
        "object_count": len(objects),
        "mesh_object_count": len(mesh_objects),
        "material_count": len(materials),
        "texture_image_count": len(texture_names),
        "total_vertices": total_vertices,
        "total_polygons": total_polygons,
        "total_triangles_estimate": total_triangles,
        "bounding_box_min": bbox_min,
        "bounding_box_max": bbox_max,
        "per_object": mesh_object_summary(objects),
        "per_material": material_summaries,
        "warnings": warnings,
    }


def write_json(out_dir: Path, report: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "asset_inspection.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_markdown(out_dir: Path, report: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "asset_inspection_summary.md"
    warnings = report["warnings"]
    warning_lines = [
        f"- {name}: {'yes' if value else 'no'}" for name, value in sorted(warnings.items())
    ]
    object_lines = [
        "- {name}: type={type} vertices={vertex_count} polygons={polygon_count} "
        "uv_layers={uv_layer_count} material_slots={material_slot_count}".format(**item)
        for item in report["per_object"]
    ]
    material_lines = [
        "- {name}: use_nodes={use_nodes} images={images}".format(
            name=item["name"],
            use_nodes=item["use_nodes"],
            images=", ".join(item["image_texture_names"]) or "none",
        )
        for item in report["per_material"]
    ]

    lines = [
        "# Asset Inspection Summary",
        "",
        f"- input_glb: `{report['input_glb']}`",
        f"- import_status: `{report['import_status']}`",
        f"- object_count: {report['object_count']}",
        f"- mesh_object_count: {report['mesh_object_count']}",
        f"- material_count: {report['material_count']}",
        f"- texture_image_count: {report['texture_image_count']}",
        f"- total_vertices: {report['total_vertices']}",
        f"- total_polygons: {report['total_polygons']}",
        f"- total_triangles_estimate: {report['total_triangles_estimate']}",
        f"- bounding_box_min: {report['bounding_box_min']}",
        f"- bounding_box_max: {report['bounding_box_max']}",
        "",
        "## Warnings",
        "",
        *(warning_lines or ["- none"]),
        "",
        "## Objects",
        "",
        *(object_lines or ["- none"]),
        "",
        "## Materials",
        "",
        *(material_lines or ["- none"]),
    ]
    if report["import_error"]:
        lines.extend(["", "## Import Error", "", f"```text\n{report['import_error']}\n```"])

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)

    import bpy  # Imported only inside Blender.

    clear_scene(bpy)
    import_status = "OK"
    import_error = ""
    try:
        bpy.ops.import_scene.gltf(filepath=str(args.input_glb))
    except Exception as exc:  # Blender operators raise broad runtime exceptions.
        import_status = "ERROR"
        import_error = repr(exc)

    report = build_report(bpy, args.input_glb, import_status, import_error)
    json_path = write_json(args.out_dir, report)
    markdown_path = write_markdown(args.out_dir, report)
    print(f"wrote inspection JSON: {json_path}")
    print(f"wrote inspection summary: {markdown_path}")
    return 0 if import_status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
