#!/usr/bin/env python3
"""Export one OBJ to GLB in an isolated clean Blender process.

Run with Blender:
  blender -b --python scripts/export_obj_to_glb_blender.py -- \
    --input-obj /path/to/textured_mesh.obj \
    --output-glb /path/to/textured_mesh.glb
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Sequence


def argv_after_double_dash(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export exactly one canonical textured mesh to GLB."
    )
    parser.add_argument("--input-obj", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    return parser.parse_args(argv)


def clear_scene_and_datablocks(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection_name in (
        "objects",
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


def import_obj(bpy: Any, path: Path) -> None:
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:  # Blender 3.6 fallback.
        bpy.ops.import_scene.obj(filepath=str(path))


def canonicalize_scene(bpy: Any) -> Any:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"OBJ-to-GLB export requires exactly one mesh, got {len(meshes)}")
    mesh = meshes[0]
    if len(mesh.material_slots) != 1:
        raise RuntimeError(
            "OBJ-to-GLB export requires exactly one material slot, "
            f"got {len(mesh.material_slots)}"
        )
    mesh.name = "textured_mesh"
    mesh.data.name = "textured_mesh"
    material = mesh.material_slots[0].material
    if material is None:
        raise RuntimeError("Input mesh material slot is empty")
    material.name = "Material"
    for image in bpy.data.images:
        filename = Path(str(image.filepath or image.name)).name.lower()
        if "metallic" in filename:
            image.name = "textured_mesh_metallic"
        elif "roughness" in filename:
            image.name = "textured_mesh_roughness"
        else:
            image.name = "textured_mesh_albedo"
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    return mesh


def primitive_count(path: Path) -> int:
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise RuntimeError(f"Invalid exported GLB: {path}")
    offset = 12
    document: dict[str, Any] | None = None
    while offset < total:
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset : offset + length]
        offset += length
        if kind == 0x4E4F534A:
            document = json.loads(payload.decode("utf-8").rstrip(" \t\r\n\0"))
    if document is None:
        raise RuntimeError(f"Exported GLB has no JSON chunk: {path}")
    return sum(len(mesh.get("primitives", [])) for mesh in document.get("meshes", []))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)
    input_obj = args.input_obj.expanduser().resolve()
    output_glb = args.output_glb.expanduser().resolve()
    if not input_obj.is_file() or input_obj.stat().st_size <= 0:
        raise RuntimeError(f"Missing input OBJ: {input_obj}")
    if output_glb.exists():
        raise FileExistsError(f"Refusing to overwrite output GLB: {output_glb}")
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    import bpy  # type: ignore

    clear_scene_and_datablocks(bpy)
    import_obj(bpy, input_obj)
    canonicalize_scene(bpy)
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
    )
    count = primitive_count(output_glb)
    if count != 1:
        raise RuntimeError(f"output GLB primitive count must be 1, got {count}")
    print(f"GLTF_PRIMITIVE_COUNT={count}")
    print("ISOLATED_GLTF_EXPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
