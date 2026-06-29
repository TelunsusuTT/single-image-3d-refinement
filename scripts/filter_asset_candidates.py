#!/usr/bin/env python3
"""Filter Phase 1B asset candidates to texture-heavy, non-rejected rows."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


FIELDNAMES = [
    "candidate_id",
    "source",
    "source_id",
    "name",
    "category",
    "asset_uri",
    "metadata_uri",
    "license",
    "expected_texture_heavy",
    "notes",
    "status",
]
TRUE_VALUES = {"yes", "true", "1"}


def is_texture_heavy(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def is_rejected(value: str) -> bool:
    return value.strip().lower() == "rejected"


def load_candidates(input_csv: Path) -> list[dict[str, str]]:
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


def filter_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if is_texture_heavy(row["expected_texture_heavy"]) and not is_rejected(row["status"])
    ]


def write_candidates(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_counts(label: str, counts: Counter[str]) -> None:
    print(f"{label}:")
    if not counts:
        print("  none=0")
        return
    for key in sorted(counts):
        print(f"  {key or '(blank)'}={counts[key]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Phase 1B candidate assets for review."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        rows = load_candidates(args.input_csv)
        kept_rows = filter_candidates(rows)
        write_candidates(args.output_csv, kept_rows)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"total rows: {len(rows)}")
    print(f"kept rows: {len(kept_rows)}")
    print(f"rejected rows: {len(rows) - len(kept_rows)}")
    print_counts("counts by source", Counter(row["source"] for row in kept_rows))
    print_counts("counts by category", Counter(row["category"] for row in kept_rows))
    print(f"wrote filtered CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
