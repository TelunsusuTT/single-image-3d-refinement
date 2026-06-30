#!/usr/bin/env python3
"""Summarize Phase 2B Blender inspection JSON reports without running Blender."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "candidate_id",
    "source_id",
    "local_glb_path",
    "inspection_json",
    "inspection_status",
    "pass",
    "reject_reason",
    "import_status",
    "mesh_object_count",
    "material_count",
    "texture_image_count",
    "total_uv_layers",
    "total_polygons",
    "total_vertices",
    "name",
    "product_type",
    "selection_reason",
]


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"manifest CSV missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest CSV has no header: {path}")
        return list(reader)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"inspection JSON must contain object: {path}")
    return data


def int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def total_uv_layers(report: dict[str, Any]) -> int:
    if "total_uv_layers" in report:
        return int_value(report.get("total_uv_layers"))
    total = 0
    per_object = report.get("per_object", [])
    if isinstance(per_object, list):
        for item in per_object:
            if isinstance(item, dict):
                total += int_value(item.get("uv_layer_count"))
    return total


def reject_reasons(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["inspection JSON missing"]
    reasons: list[str] = []
    if report.get("import_status") != "OK":
        reasons.append("import_status is not OK")
    if int_value(report.get("mesh_object_count")) <= 0:
        reasons.append("mesh_object_count is 0")
    if int_value(report.get("material_count")) <= 0:
        reasons.append("material_count is 0")
    if int_value(report.get("texture_image_count")) <= 0:
        reasons.append("texture_image_count is 0")
    if total_uv_layers(report) <= 0:
        reasons.append("total_uv_layers is 0")
    return reasons


def summarize_row(row: dict[str, str], inspection_root: Path) -> dict[str, str]:
    source_id = row.get("source_id", "").strip()
    inspection_json = inspection_root / source_id / "asset_inspection.json"
    report = load_json(inspection_json)
    reasons = reject_reasons(report)
    passed = not reasons

    return {
        "candidate_id": row.get("candidate_id", ""),
        "source_id": source_id,
        "local_glb_path": row.get("local_glb_path", ""),
        "inspection_json": str(inspection_json),
        "inspection_status": "found" if report is not None else "missing",
        "pass": "yes" if passed else "no",
        "reject_reason": "; ".join(reasons),
        "import_status": str(report.get("import_status", "")) if report else "",
        "mesh_object_count": str(int_value(report.get("mesh_object_count")) if report else 0),
        "material_count": str(int_value(report.get("material_count")) if report else 0),
        "texture_image_count": str(int_value(report.get("texture_image_count")) if report else 0),
        "total_uv_layers": str(total_uv_layers(report) if report else 0),
        "total_polygons": str(int_value(report.get("total_polygons")) if report else 0),
        "total_vertices": str(int_value(report.get("total_vertices")) if report else 0),
        "name": row.get("name", ""),
        "product_type": row.get("product_type", ""),
        "selection_reason": row.get("selection_reason", ""),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2B Inspection Summary",
        "",
        "| source_id | pass | reject_reason | meshes | materials | textures | uv_layers | polygons |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {source_id} | {passed} | {reason} | {meshes} | {materials} | {textures} | {uv} | {polygons} |".format(
                source_id=row["source_id"],
                passed=row["pass"],
                reason=row["reject_reason"] or "",
                meshes=row["mesh_object_count"],
                materials=row["material_count"],
                textures=row["texture_image_count"],
                uv=row["total_uv_layers"],
                polygons=row["total_polygons"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 2B inspection JSON reports.")
    parser.add_argument("--inspection-root", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest_rows = load_manifest(args.manifest_csv)
        summary_rows = [summarize_row(row, args.inspection_root) for row in manifest_rows]
        write_csv(args.out_csv, summary_rows)
        write_markdown(args.out_md, summary_rows)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    passed = sum(1 for row in summary_rows if row["pass"] == "yes")
    failed = len(summary_rows) - passed
    print(f"wrote CSV summary: {args.out_csv}")
    print(f"wrote Markdown summary: {args.out_md}")
    print(f"Inspection summary: {passed} pass, {failed} fail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
