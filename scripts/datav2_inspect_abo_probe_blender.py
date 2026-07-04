#!/usr/bin/env python3
"""Inspect local ABO probe GLBs in Blender and create contact sheets.

Run manually with Blender:
  blender -b --python scripts/datav2_inspect_abo_probe_blender.py -- \
    --probe-manifest data/candidates/datav2_abo_probe_top40_manifest.csv \
    --output-root outputs/phase2l/abo_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blender_inspect_glb import build_report  # noqa: E402
from render_phase2k3_glb_views_blender import (  # noqa: E402
    add_light,
    camera_pose_for_view,
    clear_scene,
    make_camera,
    normalize_scene,
    remove_lights,
    set_render_defaults,
)


RESULT_COLUMNS = [
    "candidate_rank",
    "asset_id",
    "input_glb",
    "import_ok",
    "render_ok",
    "object_count",
    "mesh_object_count",
    "material_count",
    "texture_image_count",
    "total_vertices",
    "total_polygons",
    "bounding_box_min",
    "bounding_box_max",
    "thumbnail_dir",
    "contact_sheet_path",
    "error",
]


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ABO probe assets in Blender.")
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-view", type=int, default=6)
    return parser.parse_args(argv)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1"}


def mesh_objects(bpy: Any) -> list[Any]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def import_glb(bpy: Any, path: Path) -> tuple[str, str, list[Any]]:
    clear_scene(bpy)
    before_names = {obj.name for obj in bpy.context.scene.objects}
    try:
        bpy.ops.import_scene.gltf(filepath=str(path))
    except Exception as exc:  # Blender import operators raise broad exceptions.
        return "ERROR", repr(exc), []
    imported = [obj for obj in bpy.context.scene.objects if obj.name not in before_names]
    return "OK", "", imported


def setup_scene_for_render(bpy: Any, imported_objects: list[Any], resolution: int) -> dict[str, Any]:
    set_render_defaults(bpy, resolution, [1.0, 1.0, 1.0])
    remove_lights(bpy)
    add_light(bpy, "AREA", "phase2l_key_light", (3.0, -4.0, 5.0), 500.0, size=5.0)
    framing = normalize_scene(bpy, imported_objects)
    make_camera(bpy, framing["ortho_scale"], framing["camera_clip_end"])
    return framing


def render_thumbnails(
    bpy: Any,
    out_dir: Path,
    asset_id: str,
    num_view: int,
    camera_distance: float,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = bpy.context.scene.camera
    paths: list[Path] = []
    for view_index in range(num_view):
        camera_pose_for_view(camera, view_index, num_view, camera_distance)
        path = out_dir / f"{asset_id}_{view_index:03d}.png"
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(path)
    return paths


def copy_image_pixels(source: Any, target: Any, offset_x: int, offset_y: int, sheet_width: int) -> None:
    width, height = int(source.size[0]), int(source.size[1])
    source_pixels = list(source.pixels)
    target_pixels = list(target.pixels)
    for y in range(height):
        for x in range(width):
            source_index = ((y * width) + x) * 4
            target_index = (((offset_y + y) * sheet_width) + (offset_x + x)) * 4
            target_pixels[target_index : target_index + 4] = source_pixels[source_index : source_index + 4]
    target.pixels = target_pixels


def make_contact_sheet(bpy: Any, image_paths: list[Path], out_path: Path, columns: int = 3) -> Path:
    if not image_paths:
        raise ValueError("no thumbnail images to combine")
    loaded = [bpy.data.images.load(str(path)) for path in image_paths]
    width, height = int(loaded[0].size[0]), int(loaded[0].size[1])
    rows = (len(loaded) + columns - 1) // columns
    sheet = bpy.data.images.new("phase2l_contact_sheet", width=width * columns, height=height * rows, alpha=False)
    blank = [1.0, 1.0, 1.0, 1.0] * (sheet.size[0] * sheet.size[1])
    sheet.pixels = blank
    for index, image in enumerate(loaded):
        column = index % columns
        row = rows - 1 - (index // columns)
        copy_image_pixels(image, sheet, column * width, row * height, int(sheet.size[0]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.filepath_raw = str(out_path)
    sheet.file_format = "JPEG"
    sheet.save()
    return out_path


def inspect_asset(bpy: Any, row: dict[str, str], output_root: Path, resolution: int, num_view: int) -> dict[str, str]:
    asset_id = row.get("asset_id", "")
    input_glb = Path(row.get("local_glb_path", ""))
    thumbnail_dir = output_root / "contact_sheets" / asset_id
    contact_sheet_path = output_root / "contact_sheets" / f"{asset_id}_contact_sheet.jpg"
    result = {
        "candidate_rank": row.get("candidate_rank", ""),
        "asset_id": asset_id,
        "input_glb": str(input_glb),
        "import_ok": "no",
        "render_ok": "no",
        "object_count": "0",
        "mesh_object_count": "0",
        "material_count": "0",
        "texture_image_count": "0",
        "total_vertices": "0",
        "total_polygons": "0",
        "bounding_box_min": "",
        "bounding_box_max": "",
        "thumbnail_dir": str(thumbnail_dir),
        "contact_sheet_path": str(contact_sheet_path),
        "error": "",
    }
    if not input_glb.is_file():
        result["error"] = "local_glb_missing"
        return result

    import_status, import_error, imported_objects = import_glb(bpy, input_glb)
    report = build_report(bpy, input_glb, import_status, import_error)
    result.update(
        {
            "import_ok": "yes" if report["import_status"] == "OK" else "no",
            "object_count": str(report["object_count"]),
            "mesh_object_count": str(report["mesh_object_count"]),
            "material_count": str(report["material_count"]),
            "texture_image_count": str(report["texture_image_count"]),
            "total_vertices": str(report["total_vertices"]),
            "total_polygons": str(report["total_polygons"]),
            "bounding_box_min": json.dumps(report["bounding_box_min"]),
            "bounding_box_max": json.dumps(report["bounding_box_max"]),
            "error": import_error,
        }
    )
    if report["import_status"] != "OK" or not mesh_objects(bpy):
        return result

    try:
        framing = setup_scene_for_render(bpy, imported_objects, resolution)
        thumbnails = render_thumbnails(
            bpy,
            thumbnail_dir,
            asset_id,
            num_view,
            float(framing["camera_distance"]),
        )
        make_contact_sheet(bpy, thumbnails, contact_sheet_path)
        result["render_ok"] = "yes"
    except Exception as exc:  # Render/contact-sheet failures should be per-asset.
        result["error"] = repr(exc)
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_root: Path, rows: list[dict[str, str]]) -> None:
    summary = {
        "asset_count": len(rows),
        "import_ok_count": sum(1 for row in rows if row["import_ok"] == "yes"),
        "render_ok_count": sum(1 for row in rows if row["render_ok"] == "yes"),
        "rows": rows,
    }
    json_path = output_root / "blender_inspection_summary.json"
    md_path = output_root / "blender_inspection_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Phase 2L.2A ABO Blender Inspection Summary",
        "",
        f"asset count: `{summary['asset_count']}`",
        f"import OK: `{summary['import_ok_count']}`",
        f"render OK: `{summary['render_ok_count']}`",
        "",
        "| Rank | Asset ID | Import | Render | Meshes | Materials | Textures | Contact Sheet | Error |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{asset}` | `{import_ok}` | `{render_ok}` | {meshes} | {materials} | {textures} | `{sheet}` | {error} |".format(
                rank=row["candidate_rank"],
                asset=row["asset_id"],
                import_ok=row["import_ok"],
                render_ok=row["render_ok"],
                meshes=row["mesh_object_count"],
                materials=row["material_count"],
                textures=row["texture_image_count"],
                sheet=row["contact_sheet_path"],
                error=row["error"].replace("|", "\\|"),
            )
        )
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)

    import bpy  # Imported only in Blender runtime.

    rows = [
        row
        for row in read_manifest(args.probe_manifest)
        if truthy(row.get("local_glb_exists", ""))
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [
        inspect_asset(bpy, row, args.output_root, args.resolution, args.num_view)
        for row in rows
    ]
    out_csv = args.output_root / "blender_inspection_results.csv"
    write_csv(out_csv, results)
    write_summary(args.output_root, results)
    print("Phase 2L.2A ABO Blender inspection")
    print(f"  local assets inspected: {len(results)}")
    print(f"  results_csv: {out_csv}")
    print("PHASE2L2A_ABO_BLENDER_INSPECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
