#!/usr/bin/env python3
"""Append or update selected Phase 2A candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ALLOWED_STATUS = {"selected", "rejected", "needs_review"}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def mark_candidate(
    candidates_csv: Path,
    selected_csv: Path,
    candidate_id: str,
    status: str,
    reason: str,
    notes: str,
) -> dict[str, str]:
    candidate_fields, candidate_rows = read_rows(candidates_csv)
    matching = [row for row in candidate_rows if row.get("candidate_id") == candidate_id]
    if not matching:
        raise ValueError(f"candidate_id not found: {candidate_id}")

    selected_row = dict(matching[0])
    selected_row["selection_status"] = status
    selected_row["selection_reason"] = reason
    if notes:
        selected_row["notes"] = notes

    fieldnames = list(candidate_fields)
    for field in ("selection_status", "selection_reason", "notes"):
        if field not in fieldnames:
            fieldnames.append(field)

    if selected_csv.is_file():
        existing_fields, selected_rows = read_rows(selected_csv)
        for field in existing_fields:
            if field not in fieldnames:
                fieldnames.append(field)
    else:
        selected_rows = []

    updated = False
    for index, row in enumerate(selected_rows):
        if row.get("candidate_id") == candidate_id:
            selected_rows[index] = {**row, **selected_row}
            updated = True
            break
    if not updated:
        selected_rows.append(selected_row)

    write_rows(selected_csv, fieldnames, selected_rows)
    return selected_row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark a Phase 2A candidate selection.")
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--selected-csv", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--status", required=True, choices=sorted(ALLOWED_STATUS))
    parser.add_argument("--reason", required=True)
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        row = mark_candidate(
            args.candidates_csv,
            args.selected_csv,
            args.candidate_id,
            args.status,
            args.reason,
            args.notes,
        )
    except (OSError, csv.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"selected CSV: {args.selected_csv}")
    print(f"updated row: {json.dumps(row, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
