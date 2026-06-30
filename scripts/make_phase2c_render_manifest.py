#!/usr/bin/env python3
"""Create a Phase 2C render manifest from Phase 2B passed assets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


OUTPUT_FIELDS = [
    "source_id",
    "candidate_id",
    "input_glb",
    "sample_name",
    "sample_dir",
    "dataset_root",
    "qa_dir",
    "product_type",
    "name",
    "faces",
    "textures",
    "images",
]


def truthy(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1", "pass", "passed"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"CSV missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def build_rows(
    download_rows: list[dict[str, str]],
    inspection_rows: list[dict[str, str]],
    dataset_root: Path,
    qa_root: Path,
) -> list[dict[str, str]]:
    passed_source_ids = {
        row.get("source_id", "").strip()
        for row in inspection_rows
        if truthy(row.get("pass", ""))
    }
    output_rows: list[dict[str, str]] = []
    for row in download_rows:
        source_id = row.get("source_id", "").strip()
        if not source_id or source_id not in passed_source_ids:
            continue
        sample_name = source_id
        output_rows.append(
            {
                "source_id": source_id,
                "candidate_id": row.get("candidate_id", ""),
                "input_glb": row.get("local_glb_path", ""),
                "sample_name": sample_name,
                "sample_dir": str(dataset_root / sample_name),
                "dataset_root": str(dataset_root),
                "qa_dir": str(qa_root / sample_name),
                "product_type": row.get("product_type", ""),
                "name": row.get("name", ""),
                "faces": row.get("faces", ""),
                "textures": row.get("textures", ""),
                "images": row.get("images", ""),
            }
        )
    return output_rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2C render manifest.")
    parser.add_argument("--download-manifest", required=True, type=Path)
    parser.add_argument("--inspection-summary", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--qa-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        download_rows = read_csv_rows(args.download_manifest)
        inspection_rows = read_csv_rows(args.inspection_summary)
        rows = build_rows(download_rows, inspection_rows, args.dataset_root, args.qa_root)
        write_rows(args.out_csv, rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"wrote render manifest: {args.out_csv}")
    print(f"assets to render: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
