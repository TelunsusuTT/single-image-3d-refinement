#!/usr/bin/env python3
"""Create a human review CSV for Phase 2L.2C ABO visual inspection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_PER_PAGE = 10
REVIEW_COLUMNS = [
    "dedup_rank",
    "asset_id",
    "local_glb_path",
    "contact_sheet_page",
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


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def index_by_asset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("asset_id", ""): row for row in rows if row.get("asset_id")}


def page_for_rank(rank_text: str) -> str:
    try:
        rank = int(rank_text)
    except ValueError:
        rank = 1
    page = ((rank - 1) // ASSETS_PER_PAGE) + 1
    return f"contact_sheet_{page:03d}.jpg"


def make_review_rows(manifest_rows: list[dict[str, str]], inspection_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    inspection_by_asset = index_by_asset(inspection_rows)
    rows: list[dict[str, str]] = []
    for manifest in manifest_rows:
        inspection = inspection_by_asset.get(manifest.get("asset_id", ""), {})
        dedup_rank = manifest.get("dedup_rank", "")
        rows.append(
            {
                "dedup_rank": dedup_rank,
                "asset_id": manifest.get("asset_id", ""),
                "local_glb_path": manifest.get("local_glb_path", ""),
                "contact_sheet_page": page_for_rank(dedup_rank),
                "flatness_ratio": manifest.get("flatness_ratio", ""),
                "panel_aspect_ratio": manifest.get("panel_aspect_ratio", ""),
                "faces": manifest.get("faces", ""),
                "textures": manifest.get("textures", ""),
                "materials": manifest.get("materials", ""),
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
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2L.2C ABO human review CSV.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    manifest_rows = read_csv(resolve_project_path(config["input_manifest"]))
    inspection_rows = read_csv(resolve_project_path(config["inspection_csv"]))
    review_rows = make_review_rows(manifest_rows, inspection_rows)
    out_csv = resolve_project_path(config["human_review_csv"])
    write_csv(out_csv, review_rows)
    print("Phase 2L.2C ABO visual human review")
    print(f"  manifest rows: {len(manifest_rows)}")
    print(f"  inspection rows: {len(inspection_rows)}")
    print(f"  review rows: {len(review_rows)}")
    print(f"  out_csv: {out_csv}")
    print("PHASE2L2C_ABO_VISUAL_HUMAN_REVIEW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
