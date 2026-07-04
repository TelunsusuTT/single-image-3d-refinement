#!/usr/bin/env python3
"""Create a download-ready manifest from deduplicated ABO candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABO_HTTPS_PREFIX = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original"
MANIFEST_COLUMNS = [
    "dedup_rank",
    "candidate_rank",
    "asset_id",
    "abo_path",
    "download_url",
    "local_glb_path",
    "rank_score",
    "flatness_ratio",
    "panel_aspect_ratio",
    "faces",
    "textures",
    "materials",
    "images",
    "dedupe_key",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def clean_abo_path(path_text: str) -> str:
    path = (path_text or "").strip()
    prefixes = (
        "s3://amazon-berkeley-objects/3dmodels/original/",
        "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/",
        "3dmodels/original/",
    )
    for prefix in prefixes:
        if path.startswith(prefix):
            path = path.removeprefix(prefix)
    return path.lstrip("/")


def manifest_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    input_csv = resolve_project_path(config["dedup_candidates_csv"])
    raw_asset_dir = resolve_project_path(config.get("raw_asset_dir", "data/raw_assets/abo"))
    rows: list[dict[str, str]] = []
    for row in read_csv(input_csv):
        abo_path = clean_abo_path(row.get("path", ""))
        asset_id = row.get("asset_id", "")
        rows.append(
            {
                "dedup_rank": row.get("dedup_rank", ""),
                "candidate_rank": row.get("candidate_rank", ""),
                "asset_id": asset_id,
                "abo_path": abo_path,
                "download_url": f"{ABO_HTTPS_PREFIX}/{abo_path}",
                "local_glb_path": str(raw_asset_dir / abo_path),
                "rank_score": row.get("rank_score", ""),
                "flatness_ratio": row.get("flatness_ratio", ""),
                "panel_aspect_ratio": row.get("panel_aspect_ratio", ""),
                "faces": row.get("faces", ""),
                "textures": row.get("textures", ""),
                "materials": row.get("materials", ""),
                "images": row.get("images", ""),
                "dedupe_key": row.get("dedupe_key", ""),
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ABO dedup download manifest.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    rows = manifest_rows(config)
    out_csv = resolve_project_path(config["download_manifest_csv"])
    write_csv(out_csv, rows)
    print("Phase 2L.2B ABO dedup download manifest")
    print(f"  manifest rows: {len(rows)}")
    print(f"  out_csv: {out_csv}")
    print("PHASE2L2B_ABO_DEDUP_DOWNLOAD_MANIFEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
