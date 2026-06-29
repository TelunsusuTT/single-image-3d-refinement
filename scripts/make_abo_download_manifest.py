#!/usr/bin/env python3
"""Create a small ABO GLB download manifest without downloading anything."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


FIELDNAMES = [
    "selected_id",
    "source",
    "source_id",
    "abo_path",
    "s3_uri",
    "local_path",
    "name",
    "category",
    "license",
    "selection_reason",
    "status",
    "notes",
]
OUTPUT_FIELDNAMES = FIELDNAMES + ["resolved_s3_uri", "resolved_local_path"]
ABO_S3_PREFIX = "s3://amazon-berkeley-objects/3dmodels/original"


def is_todo_or_empty(value: str) -> bool:
    stripped = value.strip()
    return not stripped or stripped.upper().startswith("TODO")


def load_rows(input_csv: Path) -> list[dict[str, str]]:
    if not input_csv.exists():
        raise ValueError(f"input CSV missing: {input_csv}")
    if not input_csv.is_file():
        raise ValueError(f"input CSV is not a file: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"input CSV has no header: {input_csv}")

        missing = [field for field in FIELDNAMES if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"input CSV missing required columns: {', '.join(missing)}")

        return [{field: row.get(field, "") for field in FIELDNAMES} for row in reader]


def resolve_row(row: dict[str, str], asset_root: Path) -> dict[str, str]:
    output = {field: row.get(field, "") for field in FIELDNAMES}
    output["resolved_s3_uri"] = ""
    output["resolved_local_path"] = ""

    source = row.get("source", "").strip().upper()
    abo_path = row.get("abo_path", "").strip().lstrip("/")
    if source != "ABO" or is_todo_or_empty(abo_path):
        return output

    source_id = row.get("source_id", "").strip()
    if is_todo_or_empty(source_id):
        raise ValueError(
            f"source_id is required for selected_id {row.get('selected_id', '')}"
        )

    output["resolved_s3_uri"] = f"{ABO_S3_PREFIX}/{abo_path}"
    local_path = row.get("local_path", "").strip()
    if is_todo_or_empty(local_path):
        local_path = str(asset_root / f"{source_id}.glb")
    output["resolved_local_path"] = local_path
    return output


def write_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a one-asset ABO download manifest without downloading."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        rows = load_rows(args.input_csv)
        resolved_rows = [resolve_row(row, args.asset_root) for row in rows]
        write_rows(args.output_csv, resolved_rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for row in resolved_rows:
        if row["resolved_s3_uri"]:
            print(
                "resolved asset: "
                f"{row['selected_id']} {row['resolved_s3_uri']} -> "
                f"{row['resolved_local_path']}"
            )
    print(f"wrote manifest: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
