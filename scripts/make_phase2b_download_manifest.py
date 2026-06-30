#!/usr/bin/env python3
"""Create a Phase 2B selected ABO GLB download manifest without downloading."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ABO_HTTPS_PREFIX = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original"
OUTPUT_FIELDS = [
    "candidate_id",
    "source_id",
    "abo_path",
    "download_url",
    "local_glb_path",
    "product_type",
    "name",
    "faces",
    "textures",
    "images",
    "selection_reason",
]


def clean_abo_path(value: str) -> str:
    path = value.strip()
    if path.startswith("s3://amazon-berkeley-objects/3dmodels/original/"):
        path = path.removeprefix("s3://amazon-berkeley-objects/3dmodels/original/")
    if path.startswith("https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/"):
        path = path.removeprefix(
            "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/"
        )
    if path.startswith("3dmodels/original/"):
        path = path.removeprefix("3dmodels/original/")
    return path.lstrip("/")


def is_selected(row: dict[str, str]) -> bool:
    status = (row.get("selection_status") or row.get("status") or "").strip().lower()
    return not status or status == "selected"


def load_selected_rows(selected_csv: Path) -> list[dict[str, str]]:
    if not selected_csv.is_file():
        raise ValueError(f"selected CSV missing: {selected_csv}")
    with selected_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"selected CSV has no header: {selected_csv}")
        return [dict(row) for row in reader if is_selected(row)]


def build_manifest_rows(rows: list[dict[str, str]], raw_dir: Path) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in rows:
        source_id = row.get("source_id", "").strip()
        abo_path = clean_abo_path(row.get("abo_path", ""))
        if not source_id:
            raise ValueError(f"selected row missing source_id: {row.get('candidate_id', '')}")
        if not abo_path:
            raise ValueError(f"selected row missing abo_path: {row.get('candidate_id', source_id)}")

        output_rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "source_id": source_id,
                "abo_path": abo_path,
                "download_url": f"{ABO_HTTPS_PREFIX}/{abo_path}",
                "local_glb_path": str(raw_dir / f"{source_id}.glb"),
                "product_type": row.get("product_type", ""),
                "name": row.get("name", ""),
                "faces": row.get("faces", ""),
                "textures": row.get("textures", ""),
                "images": row.get("images", ""),
                "selection_reason": row.get("selection_reason", ""),
            }
        )
    return output_rows


def write_manifest(out_csv: Path, rows: list[dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2B selected GLB manifest.")
    parser.add_argument("--selected-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selected_rows = load_selected_rows(args.selected_csv)
        manifest_rows = build_manifest_rows(selected_rows, args.raw_dir)
        write_manifest(args.out_csv, manifest_rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for row in manifest_rows:
        print(f"{row['candidate_id']}: {row['download_url']} -> {row['local_glb_path']}")
    print(f"wrote manifest: {args.out_csv}")
    print(f"selected assets: {len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
