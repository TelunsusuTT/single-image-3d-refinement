#!/usr/bin/env python3
"""Inspect downloaded ABO dedup GLBs and render six fixed views in Blender.

Run manually with Blender:
  blender -b --python scripts/datav2_inspect_abo_dedup_blender.py -- \
    --config configs/datav2_abo_visual_inspection.json
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
    "dedup_rank",
    "candidate_rank",
    "asset_id",
    "local_glb_path",
    "import_ok",
    "render_ok",
    "object_count",
    "mesh_count",
    "material_count",
    "texture_count",
    "bbox_extent_x",
    "bbox_extent_y",
    "bbox_extent_z",
    "face_count",
    "render_dir",
    "error",
]


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect downloaded ABO dedup GLBs in Blender.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def bbox_extents(report: dict[str, Any]) -> tuple[float, float, float]:
    bbox_min = report.get("bounding_box_min") or [0.0, 0.0, 0.0]
    bbox_max = report.get("bounding_box_max") or [0.0, 0.0, 0.0]
    return tuple(float(bbox_max[index]) - float(bbox_min[index]) for index in range(3))


def import_glb(bpy: Any, path: Path) -> tuple[str, str, list[Any]]:
    clear_scene(bpy)
    before_names = {obj.name for obj in bpy.context.scene.objects}
    try:
        bpy.ops.import_scene.gltf(filepath=str(path))
    except Exception as exc:  # Blender import operators raise broad exceptions.
        return "ERROR", repr(exc), []
    imported = [obj for obj in bpy.context.scene.objects if obj.name not in before_names]
    return "OK", "", imported


def setup_scene(bpy: Any, imported_objects: list[Any], resolution: int) -> dict[str, Any]:
    set_render_defaults(bpy, resolution, [1.0, 1.0, 1.0])
    remove_lights(bpy)
    add_light(bpy, "AREA", "phase2l_key_light", (3.0, -4.0, 5.0), 500.0, size=5.0)
    framing = normalize_scene(bpy, imported_objects)
    make_camera(bpy, framing["ortho_scale"], framing["camera_clip_end"])
    return framing


def render_views(
    bpy: Any,
    render_dir: Path,
    view_ids: list[str],
    camera_distance: float,
) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    camera = bpy.context.scene.camera
    for view_index, view_id in enumerate(view_ids):
        camera_pose_for_view(camera, view_index, len(view_ids), camera_distance)
        out_path = render_dir / f"{view_id}.png"
        bpy.context.scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)


def inspect_asset(bpy: Any, row: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    asset_id = row.get("asset_id", "")
    local_glb = Path(row.get("local_glb_path", ""))
    render_dir = resolve_project_path(config["render_root"]) / asset_id
    result = {
        "dedup_rank": row.get("dedup_rank", ""),
        "candidate_rank": row.get("candidate_rank", ""),
        "asset_id": asset_id,
        "local_glb_path": str(local_glb),
        "import_ok": "no",
        "render_ok": "no",
        "object_count": "0",
        "mesh_count": "0",
        "material_count": "0",
        "texture_count": "0",
        "bbox_extent_x": "",
        "bbox_extent_y": "",
        "bbox_extent_z": "",
        "face_count": "0",
        "render_dir": str(render_dir),
        "error": "",
    }
    if not local_glb.is_file():
        result["error"] = "local_glb_missing"
        return result

    import_status, import_error, imported_objects = import_glb(bpy, local_glb)
    report = build_report(bpy, local_glb, import_status, import_error)
    extent_x, extent_y, extent_z = bbox_extents(report)
    result.update(
        {
            "import_ok": "yes" if report.get("import_status") == "OK" else "no",
            "object_count": str(report.get("object_count", 0)),
            "mesh_count": str(report.get("mesh_object_count", 0)),
            "material_count": str(report.get("material_count", 0)),
            "texture_count": str(report.get("texture_image_count", 0)),
            "bbox_extent_x": f"{extent_x:.6f}",
            "bbox_extent_y": f"{extent_y:.6f}",
            "bbox_extent_z": f"{extent_z:.6f}",
            "face_count": str(report.get("total_polygons", 0)),
            "error": import_error,
        }
    )
    if result["import_ok"] != "yes" or int(result["mesh_count"]) <= 0:
        return result

    try:
        framing = setup_scene(bpy, imported_objects, int(config.get("render_resolution", 512)))
        render_views(
            bpy,
            render_dir,
            list(config.get("view_ids", ["000", "001", "002", "003", "004", "005"])),
            float(framing["camera_distance"]),
        )
        result["render_ok"] = "yes"
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def write_summary(config: dict[str, Any], rows: list[dict[str, str]]) -> None:
    out_json = resolve_project_path(config["inspection_summary_json"])
    out_md = resolve_project_path(config["inspection_summary_md"])
    summary = {
        "asset_count": len(rows),
        "import_ok_count": sum(1 for row in rows if row["import_ok"] == "yes"),
        "render_ok_count": sum(1 for row in rows if row["render_ok"] == "yes"),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Phase 2L.2C ABO Visual Inspection Summary",
        "",
        f"asset count: `{summary['asset_count']}`",
        f"import OK: `{summary['import_ok_count']}`",
        f"render OK: `{summary['render_ok_count']}`",
        "",
        "| Dedup Rank | Asset ID | Import | Render | Meshes | Materials | Textures | Faces | Error |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dedup} | `{asset}` | `{import_ok}` | `{render_ok}` | {meshes} | {materials} | {textures} | {faces} | {error} |".format(
                dedup=row["dedup_rank"],
                asset=row["asset_id"],
                import_ok=row["import_ok"],
                render_ok=row["render_ok"],
                meshes=row["mesh_count"],
                materials=row["material_count"],
                textures=row["texture_count"],
                faces=row["face_count"],
                error=row["error"].replace("|", "\\|"),
            )
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)

    import bpy  # Imported only inside Blender runtime.

    config = load_config(args.config)
    rows = read_csv(resolve_project_path(config["input_manifest"]))
    results = [inspect_asset(bpy, row, config) for row in rows]
    out_csv = resolve_project_path(config["inspection_csv"])
    write_csv(out_csv, results)
    write_summary(config, results)
    print("Phase 2L.2C ABO Blender visual inspection")
    print(f"  assets inspected: {len(results)}")
    print(f"  inspection_csv: {out_csv}")
    print("PHASE2L2C_ABO_BLENDER_INSPECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
