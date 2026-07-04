#!/usr/bin/env python3
"""Inspect manually downloaded ABO GLBs and render six fixed Blender views.

Run manually with Blender:
  blender -b --python scripts/datav2_inspect_manual_abo_blender.py -- \
    --config configs/datav2_manual_abo_visual_inspection.json
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
    "item_id",
    "relative_path",
    "local_glb_path",
    "import_ok",
    "render_ok",
    "object_count",
    "mesh_count",
    "material_count",
    "texture_image_count",
    "texture_count",
    "face_count",
    "bbox_extent_x",
    "bbox_extent_y",
    "bbox_extent_z",
    "bbox_extents",
    "render_dir",
    "error",
]


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect manually downloaded ABO GLBs in Blender.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--only-missing", action="store_true")
    return parser.parse_args(argv)


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def output_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_root"])


def render_root(config: dict[str, Any]) -> Path:
    return output_root(config) / "renders"


def inspection_csv_path(config: dict[str, Any]) -> Path:
    return output_root(config) / "inspection_results.csv"


def summary_json_path(config: dict[str, Any]) -> Path:
    return output_root(config) / "inspection_summary.json"


def summary_md_path(config: dict[str, Any]) -> Path:
    return output_root(config) / "inspection_summary.md"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def item_id(row: dict[str, str]) -> str:
    return (row.get("item_id") or row.get("asset_id") or "").strip()


def resolve_local_glb(row: dict[str, str]) -> Path:
    return resolve_project_path(row.get("local_glb_path", ""))


def eligible_manifest_rows(config: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    rows = read_csv(resolve_project_path(config["manifest"]))
    eligible: list[dict[str, str]] = []
    skipped = 0
    for row in rows:
        if row.get("status") != "found":
            skipped += 1
            continue
        path = resolve_local_glb(row)
        if not path.is_file() or path.stat().st_size <= 0:
            skipped += 1
            continue
        eligible.append(row)
    return eligible, skipped


def complete_render_set(config: dict[str, Any], item: str) -> bool:
    view_ids = list(config.get("view_ids", ["000", "001", "002", "003", "004", "005"]))
    asset_render_dir = render_root(config) / item
    return all((asset_render_dir / f"{view_id}.png").is_file() for view_id in view_ids)


def bbox_extents(report: dict[str, Any]) -> tuple[float, float, float]:
    bbox_min = report.get("bounding_box_min") or [0.0, 0.0, 0.0]
    bbox_max = report.get("bounding_box_max") or [0.0, 0.0, 0.0]
    return tuple(float(bbox_max[index]) - float(bbox_min[index]) for index in range(3))


def import_glb(bpy: Any, path: Path) -> tuple[str, str, list[Any]]:
    clear_scene(bpy)
    before_names = {obj.name for obj in bpy.context.scene.objects}
    try:
        bpy.ops.import_scene.gltf(filepath=str(path))
    except Exception as exc:  # Blender operators raise broad runtime exceptions.
        return "ERROR", repr(exc), []
    imported = [obj for obj in bpy.context.scene.objects if obj.name not in before_names]
    return "OK", "", imported


def setup_scene(bpy: Any, imported_objects: list[Any], resolution: int) -> dict[str, Any]:
    set_render_defaults(bpy, resolution, [1.0, 1.0, 1.0])
    remove_lights(bpy)
    add_light(bpy, "AREA", "phase2l_manual_key_light", (3.0, -4.0, 5.0), 500.0, size=5.0)
    framing = normalize_scene(bpy, imported_objects)
    make_camera(bpy, framing["ortho_scale"], framing["camera_clip_end"])
    return framing


def render_views(bpy: Any, config: dict[str, Any], item: str, camera_distance: float) -> None:
    asset_render_dir = render_root(config) / item
    asset_render_dir.mkdir(parents=True, exist_ok=True)
    camera = bpy.context.scene.camera
    view_ids = list(config.get("view_ids", ["000", "001", "002", "003", "004", "005"]))
    for view_index, view_id in enumerate(view_ids):
        camera_pose_for_view(camera, view_index, len(view_ids), camera_distance)
        out_path = asset_render_dir / f"{view_id}.png"
        bpy.context.scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)


def inspect_asset(bpy: Any, config: dict[str, Any], row: dict[str, str]) -> dict[str, str]:
    item = item_id(row)
    local_glb = resolve_local_glb(row)
    asset_render_dir = render_root(config) / item
    result = {
        "item_id": item,
        "relative_path": row.get("relative_path", ""),
        "local_glb_path": str(local_glb),
        "import_ok": "no",
        "render_ok": "yes" if complete_render_set(config, item) else "no",
        "object_count": "0",
        "mesh_count": "0",
        "material_count": "0",
        "texture_image_count": "0",
        "texture_count": "0",
        "face_count": "0",
        "bbox_extent_x": "",
        "bbox_extent_y": "",
        "bbox_extent_z": "",
        "bbox_extents": "",
        "render_dir": str(asset_render_dir),
        "error": "",
    }
    import_status, import_error, imported_objects = import_glb(bpy, local_glb)
    report = build_report(bpy, local_glb, import_status, import_error)
    extent_x, extent_y, extent_z = bbox_extents(report)
    texture_count = int(report.get("texture_image_count", 0))
    result.update(
        {
            "import_ok": "yes" if report.get("import_status") == "OK" else "no",
            "object_count": str(report.get("object_count", 0)),
            "mesh_count": str(report.get("mesh_object_count", 0)),
            "material_count": str(report.get("material_count", 0)),
            "texture_image_count": str(texture_count),
            "texture_count": str(texture_count),
            "face_count": str(report.get("total_polygons", 0)),
            "bbox_extent_x": f"{extent_x:.6f}",
            "bbox_extent_y": f"{extent_y:.6f}",
            "bbox_extent_z": f"{extent_z:.6f}",
            "bbox_extents": f"{extent_x:.6f} x {extent_y:.6f} x {extent_z:.6f}",
            "error": import_error,
        }
    )
    if result["import_ok"] != "yes" or int(result["mesh_count"]) <= 0:
        return result
    if result["render_ok"] == "yes":
        return result
    try:
        framing = setup_scene(bpy, imported_objects, int(config.get("render_resolution", 512)))
        render_views(bpy, config, item, float(framing["camera_distance"]))
        result["render_ok"] = "yes" if complete_render_set(config, item) else "no"
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def ordered_results(manifest_rows: list[dict[str, str]], result_by_item: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    emitted: set[str] = set()
    for manifest in manifest_rows:
        item = item_id(manifest)
        if item in result_by_item:
            rows.append(result_by_item[item])
            emitted.add(item)
    for item, row in result_by_item.items():
        if item not in emitted:
            rows.append(row)
    return rows


def write_summary(config: dict[str, Any], rows: list[dict[str, str]], skipped_manifest_rows: int) -> None:
    out_json = summary_json_path(config)
    out_md = summary_md_path(config)
    summary = {
        "asset_count": len(rows),
        "skipped_manifest_rows": skipped_manifest_rows,
        "import_ok_count": sum(1 for row in rows if row["import_ok"] == "yes"),
        "render_ok_count": sum(1 for row in rows if row["render_ok"] == "yes"),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Phase 2L.2B Manual ABO Visual Inspection Summary",
        "",
        f"asset count: `{summary['asset_count']}`",
        f"skipped manifest rows: `{skipped_manifest_rows}`",
        f"import OK: `{summary['import_ok_count']}`",
        f"render OK: `{summary['render_ok_count']}`",
        "",
        "| Item ID | Import | Render | Meshes | Materials | Textures | Faces | BBox Extents | Error |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{item}` | `{import_ok}` | `{render_ok}` | {meshes} | {materials} | {textures} | {faces} | `{bbox}` | {error} |".format(
                item=row["item_id"],
                import_ok=row["import_ok"],
                render_ok=row["render_ok"],
                meshes=row["mesh_count"],
                materials=row["material_count"],
                textures=row["texture_image_count"],
                faces=row["face_count"],
                bbox=row["bbox_extents"],
                error=row["error"].replace("|", "\\|"),
            )
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def selected_rows(rows: list[dict[str, str]], start_index: int, limit: int | None) -> list[dict[str, str]]:
    start = max(start_index, 0)
    if limit is None:
        return rows[start:]
    return rows[start : start + max(limit, 0)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)

    import bpy  # Imported only inside Blender runtime.

    config = load_config(args.config)
    manifest_rows, skipped = eligible_manifest_rows(config)
    existing_rows = read_csv(inspection_csv_path(config))
    result_by_item = {row["item_id"]: row for row in existing_rows if row.get("item_id")}
    to_process = selected_rows(manifest_rows, args.start_index, args.limit)

    processed = 0
    for row in to_process:
        item = item_id(row)
        if args.only_missing and complete_render_set(config, item) and item in result_by_item:
            continue
        result_by_item[item] = inspect_asset(bpy, config, row)
        processed += 1

    results = ordered_results(manifest_rows, result_by_item)
    write_csv(inspection_csv_path(config), results)
    write_summary(config, results, skipped)
    print("Phase 2L.2B manual ABO Blender visual inspection")
    print(f"  eligible assets: {len(manifest_rows)}")
    print(f"  processed this run: {processed}")
    print(f"  inspection_csv: {inspection_csv_path(config)}")
    print("PHASE2L2B_MANUAL_ABO_BLENDER_INSPECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
