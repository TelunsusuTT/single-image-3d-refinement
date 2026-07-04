#!/usr/bin/env python3
"""Create a human curation template for the Phase 2L.2A ABO probe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_COLUMNS = [
    "candidate_rank",
    "asset_id",
    "local_glb_path",
    "contact_sheet_path",
    "flatness_ratio",
    "panel_aspect_ratio",
    "faces",
    "textures",
    "materials",
    "import_ok",
    "render_ok",
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
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def index_by_asset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("asset_id", ""): row for row in rows if row.get("asset_id")}


def make_review_rows(manifest_rows: list[dict[str, str]], inspection_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    inspection_by_asset = index_by_asset(inspection_rows)
    review_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        inspection = inspection_by_asset.get(row.get("asset_id", ""), {})
        review_rows.append(
            {
                "candidate_rank": row.get("candidate_rank", ""),
                "asset_id": row.get("asset_id", ""),
                "local_glb_path": row.get("local_glb_path", ""),
                "contact_sheet_path": inspection.get("contact_sheet_path", ""),
                "flatness_ratio": row.get("flatness_ratio", ""),
                "panel_aspect_ratio": row.get("panel_aspect_ratio", ""),
                "faces": row.get("faces", ""),
                "textures": row.get("textures", ""),
                "materials": row.get("materials", ""),
                "import_ok": inspection.get("import_ok", ""),
                "render_ok": inspection.get("render_ok", ""),
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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an ABO probe human-review template.")
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--inspection-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_rows = read_csv(args.probe_manifest)
    inspection_rows = read_csv(args.inspection_csv)
    review_rows = make_review_rows(manifest_rows, inspection_rows)
    write_csv(args.out_csv, review_rows)
    print("Phase 2L.2A ABO probe human review template")
    print(f"  manifest rows: {len(manifest_rows)}")
    print(f"  inspection rows: {len(inspection_rows)}")
    print(f"  review rows: {len(review_rows)}")
    print(f"  out_csv: {args.out_csv}")
    print("PHASE2L2A_ABO_PROBE_REVIEW_TEMPLATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
