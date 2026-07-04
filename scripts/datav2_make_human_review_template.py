#!/usr/bin/env python3
"""Create a human curation CSV template from ranked Data v2 candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_COLUMNS = [
    "candidate_rank",
    "source",
    "asset_id",
    "title",
    "tags",
    "category",
    "file_format",
    "rank_score",
    "auto_reasons",
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def auto_reasons(row: dict[str, str]) -> str:
    if row.get("auto_reasons"):
        return row["auto_reasons"]
    parts = []
    if row.get("positive_keyword_hits"):
        parts.append(f"positive:{row['positive_keyword_hits']}")
    if row.get("negative_keyword_hits"):
        parts.append(f"negative:{row['negative_keyword_hits']}")
    if row.get("technical_flags"):
        parts.append(f"technical:{row['technical_flags']}")
    return "; ".join(parts)


def make_review_rows(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    selected = rows[:top_k] if top_k > 0 else rows
    review_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        review_rows.append(
            {
                "candidate_rank": row.get("candidate_rank") or str(index),
                "source": row.get("source", ""),
                "asset_id": row.get("asset_id", ""),
                "title": row.get("title", ""),
                "tags": row.get("tags", ""),
                "category": row.get("category", ""),
                "file_format": row.get("file_format", ""),
                "rank_score": row.get("rank_score", ""),
                "auto_reasons": auto_reasons(row),
                "human_decision": "",
                "reject_reason": "",
                "subclass": "",
                "selected_input_view": "",
                "alternative_input_view": "",
                "primary_eval_views": "",
                "front_quality_score": "",
                "texture_quality_score": "",
                "leakage_risk": "",
                "notes": "",
            }
        )
    return review_rows


def write_review_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Data v2 human-review template.")
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.candidates_csv)
    review_rows = make_review_rows(rows, args.top_k)
    write_review_csv(review_rows, args.out_csv)
    print("Phase 2L.1 human review template")
    print(f"  input candidates: {len(rows)}")
    print(f"  review rows: {len(review_rows)}")
    print(f"  out_csv: {args.out_csv}")
    print("PHASE2L1_HUMAN_REVIEW_TEMPLATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
