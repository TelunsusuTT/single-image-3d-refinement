#!/usr/bin/env python3
"""Create a human curation template from a manual ABO download manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_COLUMNS = [
    "item_id",
    "relative_path",
    "url",
    "local_glb_path",
    "exists_local",
    "status",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_review_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        review = {column: "" for column in REVIEW_COLUMNS}
        for column in ("item_id", "relative_path", "url", "local_glb_path", "exists_local", "status"):
            review[column] = row.get(column, "")
        rows.append(review)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manual ABO human review CSV.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_rows = read_csv(args.manifest)
    review_rows = make_review_rows(manifest_rows)
    write_csv(args.out_csv, review_rows)
    print("Phase 2L.2A manual ABO human review template")
    print(f"  manifest rows: {len(manifest_rows)}")
    print(f"  review rows: {len(review_rows)}")
    print(f"  out_csv: {args.out_csv}")
    print("PHASE2L2A_MANUAL_ABO_REVIEW_TEMPLATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
