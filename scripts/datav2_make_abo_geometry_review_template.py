#!/usr/bin/env python3
"""Create a human review template for ABO geometry-ranked candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_COLUMNS = [
    "candidate_rank",
    "asset_id",
    "path",
    "rank_score",
    "flatness_ratio",
    "panel_aspect_ratio",
    "extent_x",
    "extent_y",
    "extent_z",
    "faces",
    "vertices",
    "materials",
    "textures",
    "images",
    "geometry_flags",
    "technical_flags",
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


def make_review_rows(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    selected = rows[:top_k] if top_k > 0 else rows
    review_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        review_rows.append(
            {
                "candidate_rank": row.get("candidate_rank") or str(index),
                "asset_id": row.get("asset_id", ""),
                "path": row.get("path", ""),
                "rank_score": row.get("rank_score", ""),
                "flatness_ratio": row.get("flatness_ratio", ""),
                "panel_aspect_ratio": row.get("panel_aspect_ratio", ""),
                "extent_x": row.get("extent_x", ""),
                "extent_y": row.get("extent_y", ""),
                "extent_z": row.get("extent_z", ""),
                "faces": row.get("faces", ""),
                "vertices": row.get("vertices", ""),
                "materials": row.get("materials", ""),
                "textures": row.get("textures", ""),
                "images": row.get("images", ""),
                "geometry_flags": row.get("geometry_flags", ""),
                "technical_flags": row.get("technical_flags", ""),
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
    parser = argparse.ArgumentParser(description="Create an ABO geometry human-review template.")
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.candidates_csv)
    review_rows = make_review_rows(rows, args.top_k)
    write_review_csv(review_rows, args.out_csv)
    print("Phase 2L.1B ABO geometry human review template")
    print(f"  input candidates: {len(rows)}")
    print(f"  review rows: {len(review_rows)}")
    print(f"  out_csv: {args.out_csv}")
    print("PHASE2L1B_ABO_GEOMETRY_REVIEW_TEMPLATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
