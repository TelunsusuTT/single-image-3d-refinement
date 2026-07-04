#!/usr/bin/env python3
"""Resolve manually curated ABO item IDs against local 3D model metadata."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABO_HTTPS_PREFIX = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original"
DEFAULT_SUMMARY_ROOT = "outputs/phase2l/manual_abo_download"
MANIFEST_COLUMNS = [
    "item_id",
    "relative_path",
    "url",
    "local_glb_path",
    "exists_local",
    "status",
    "faces",
    "vertices",
    "meshes",
    "materials",
    "textures",
    "images",
    "image_width_max",
    "image_height_max",
]
METADATA_COLUMNS = [
    "faces",
    "vertices",
    "meshes",
    "materials",
    "textures",
    "images",
    "image_width_max",
    "image_height_max",
]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def clean_abo_path(path_text: str) -> str:
    path = (path_text or "").strip()
    prefixes = (
        "s3://amazon-berkeley-objects/3dmodels/original/",
        "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/",
        "3dmodels/original/",
    )
    for prefix in prefixes:
        if path.startswith(prefix):
            path = path.removeprefix(prefix)
    return path.lstrip("/")


def read_item_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    item_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line or line in seen:
            continue
        seen.add(line)
        item_ids.append(line)
    return item_ids


def metadata_item_id(row: dict[str, str]) -> str:
    for key in ("3dmodel_id", "item_id", "model_id", "id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def metadata_path(row: dict[str, str]) -> str:
    for key in ("path", "abo_path", "relative_path"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def load_metadata_index(path: Path) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        index: dict[str, dict[str, str]] = {}
        for row in rows:
            item_id = metadata_item_id(row)
            if item_id and item_id not in index:
                index[item_id] = row
        return index


def row_from_metadata(item_id: str, row: dict[str, str], raw_asset_root: Path) -> dict[str, str]:
    relative_path = clean_abo_path(metadata_path(row))
    local_path = raw_asset_root / relative_path
    manifest_row = {
        "item_id": item_id,
        "relative_path": relative_path,
        "url": f"{ABO_HTTPS_PREFIX}/{relative_path}" if relative_path else "",
        "local_glb_path": str(local_path) if relative_path else "",
        "exists_local": "true" if relative_path and local_path.is_file() and local_path.stat().st_size > 0 else "false",
        "status": "found",
    }
    for column in METADATA_COLUMNS:
        manifest_row[column] = row.get(column, "")
    return manifest_row


def missing_row(item_id: str) -> dict[str, str]:
    row = {
        "item_id": item_id,
        "relative_path": "",
        "url": "",
        "local_glb_path": "",
        "exists_local": "false",
        "status": "missing_in_metadata",
    }
    for column in METADATA_COLUMNS:
        row[column] = ""
    return row


def resolve_item_ids(item_ids: list[str], metadata_index: dict[str, dict[str, str]], raw_asset_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item_id in item_ids:
        metadata_row = metadata_index.get(item_id)
        if metadata_row is None:
            rows.append(missing_row(item_id))
        else:
            rows.append(row_from_metadata(item_id, metadata_row, raw_asset_root))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summary_payload(item_ids: list[str], rows: list[dict[str, str]], out_csv: Path) -> dict[str, Any]:
    found_rows = [row for row in rows if row["status"] == "found"]
    missing_rows = [row for row in rows if row["status"] == "missing_in_metadata"]
    local_rows = [row for row in found_rows if row["exists_local"] == "true"]
    return {
        "manifest_csv": str(out_csv),
        "total_requested_after_comments": len(item_ids),
        "unique_item_ids": len(item_ids),
        "found_count": len(found_rows),
        "missing_in_metadata_count": len(missing_rows),
        "exists_local_count": len(local_rows),
        "needs_download_count": len(found_rows) - len(local_rows),
        "rows": rows,
    }


def write_summary(summary_json: Path, summary_md: Path, payload: dict[str, Any]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Phase 2L.2A Manual ABO Item-ID Resolution",
        "",
        f"unique item IDs: `{payload['unique_item_ids']}`",
        f"found in metadata: `{payload['found_count']}`",
        f"missing in metadata: `{payload['missing_in_metadata_count']}`",
        f"local GLBs already present: `{payload['exists_local_count']}`",
        f"needs download: `{payload['needs_download_count']}`",
        "",
        "| Item ID | Status | Local | Relative Path | Local GLB Path |",
        "|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{item}` | `{status}` | `{local}` | `{relative}` | `{local_path}` |".format(
                item=row["item_id"],
                status=row["status"],
                local=row["exists_local"],
                relative=row["relative_path"],
                local_path=row["local_glb_path"],
            )
        )
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve manual ABO item IDs into a safe GLB manifest.")
    parser.add_argument("--item-ids", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    item_ids_path = resolve_project_path(args.item_ids, project_root)
    metadata_path = resolve_project_path(args.metadata, project_root)
    out_csv = resolve_project_path(args.out_csv, project_root)
    summary_root = project_root / DEFAULT_SUMMARY_ROOT
    summary_md = resolve_project_path(args.summary_md, project_root) if args.summary_md else summary_root / "manual_abo_resolve_summary.md"
    summary_json = (
        resolve_project_path(args.summary_json, project_root)
        if args.summary_json
        else summary_root / "manual_abo_resolve_summary.json"
    )
    raw_asset_root = project_root / "data" / "raw_assets" / "abo"

    item_ids = read_item_ids(item_ids_path)
    metadata_index = load_metadata_index(metadata_path)
    rows = resolve_item_ids(item_ids, metadata_index, raw_asset_root)
    write_csv(out_csv, rows)
    payload = summary_payload(item_ids, rows, out_csv)
    write_summary(summary_json, summary_md, payload)

    print("Phase 2L.2A manual ABO item-id resolution")
    print(f"  unique item IDs: {len(item_ids)}")
    print(f"  found in metadata: {payload['found_count']}")
    print(f"  missing in metadata: {payload['missing_in_metadata_count']}")
    print(f"  manifest: {out_csv}")
    print("PHASE2L2A_MANUAL_ABO_IDS_RESOLVED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
