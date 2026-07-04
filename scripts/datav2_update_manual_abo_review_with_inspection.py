#!/usr/bin/env python3
"""Merge manual ABO manifest, review template, and Blender inspection results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_COLUMNS = [
    "human_decision",
    "reject_reason",
    "subclass",
    "selected_input_view",
    "alternative_input_view",
    "primary_eval_views",
    "front_quality_score",
    "texture_quality_score",
    "leakage_risk",
    "notes",
]
BASE_COLUMNS = [
    "item_id",
    "relative_path",
    "url",
    "local_glb_path",
    "exists_local",
    "status",
]
INSPECTION_COLUMNS = [
    "import_ok",
    "render_ok",
    "contact_sheet_page",
    "contact_sheet_path",
    "face_count",
    "mesh_count",
    "material_count",
    "texture_image_count",
    "bbox_extents",
]
OUTPUT_COLUMNS = BASE_COLUMNS + INSPECTION_COLUMNS + HUMAN_COLUMNS


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_root"])


def contact_sheet_root(config: dict[str, Any]) -> Path:
    return output_root(config) / "contact_sheets"


def inspection_csv_path(config: dict[str, Any]) -> Path:
    return output_root(config) / "inspection_results.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def index_by_item(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("item_id", ""): row for row in rows if row.get("item_id")}


def contact_sheet_lookup(config: dict[str, Any], inspection_rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    assets_per_page = int(config.get("contact_sheet_assets_per_page", 12))
    lookup: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(inspection_rows):
        item_id = row.get("item_id", "")
        if not item_id or row.get("render_ok") != "yes":
            continue
        page = (index // assets_per_page) + 1
        page_name = f"contact_sheet_{page:03d}.jpg"
        lookup[item_id] = (page_name, str(contact_sheet_root(config) / page_name))
    return lookup


def merged_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    manifest_rows = read_csv(resolve_project_path(config["manifest"]))
    review_rows = read_csv(resolve_project_path(config["human_review_csv"]))
    inspection_rows = read_csv(inspection_csv_path(config))
    review_by_item = index_by_item(review_rows)
    inspection_by_item = index_by_item(inspection_rows)
    page_by_item = contact_sheet_lookup(config, inspection_rows)
    rows: list[dict[str, str]] = []
    for manifest in manifest_rows:
        item = manifest.get("item_id", "")
        review = review_by_item.get(item, {})
        inspection = inspection_by_item.get(item, {})
        page_name, page_path = page_by_item.get(item, ("", ""))
        row = {column: "" for column in OUTPUT_COLUMNS}
        for column in BASE_COLUMNS:
            row[column] = manifest.get(column, review.get(column, ""))
        row.update(
            {
                "import_ok": inspection.get("import_ok", ""),
                "render_ok": inspection.get("render_ok", ""),
                "contact_sheet_page": page_name,
                "contact_sheet_path": page_path,
                "face_count": inspection.get("face_count", ""),
                "mesh_count": inspection.get("mesh_count", ""),
                "material_count": inspection.get("material_count", ""),
                "texture_image_count": inspection.get("texture_image_count", inspection.get("texture_count", "")),
                "bbox_extents": inspection.get("bbox_extents", ""),
            }
        )
        for column in HUMAN_COLUMNS:
            row[column] = review.get(column, "")
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge manual ABO review CSV with Blender inspection results.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    rows = merged_rows(config)
    out_csv = resolve_project_path(args.out_csv)
    write_csv(out_csv, rows)
    inspected = sum(1 for row in rows if row["import_ok"])
    rendered = sum(1 for row in rows if row["render_ok"] == "yes")
    print("Phase 2L.2B manual ABO review with inspection")
    print(f"  review rows: {len(rows)}")
    print(f"  inspected rows: {inspected}")
    print(f"  rendered rows: {rendered}")
    print(f"  out_csv: {out_csv}")
    print("PHASE2L2B_MANUAL_ABO_REVIEW_WITH_INSPECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
