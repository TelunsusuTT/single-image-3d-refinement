#!/usr/bin/env python3
"""Deduplicate ABO geometry-ranked candidates for a small acquisition probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_COLUMNS = [
    "dedup_rank",
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
    "meshes",
    "materials",
    "textures",
    "images",
    "image_width_max",
    "image_height_max",
    "dedupe_key",
    "duplicate_group_size",
    "dedupe_choice_reason",
    "geometry_flags",
    "technical_flags",
    "reject_flags",
    "raw_metadata_json",
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


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(str(value).replace(",", ""))
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def parse_int(value: str | None, default: int = 0) -> int:
    return int(round(parse_float(value, float(default))))


def bucket(value: float, bucket_size: float) -> int:
    if bucket_size <= 0:
        return int(round(value))
    return int(round(value / bucket_size))


def dedupe_key(row: dict[str, str], settings: dict[str, Any]) -> str:
    extent_digits = int(settings.get("extent_round_digits", 2))
    ratio_digits = int(settings.get("ratio_round_digits", 2))
    face_bucket_size = float(settings.get("face_bucket_size", 5000))
    texture_bucket_size = float(settings.get("texture_bucket_size", 1))
    material_bucket_size = float(settings.get("material_bucket_size", 1))
    extents = sorted(
        round(parse_float(row.get(field)), extent_digits)
        for field in ("extent_x", "extent_y", "extent_z")
    )
    parts = [
        f"ext:{extents[0]:.{extent_digits}f},{extents[1]:.{extent_digits}f},{extents[2]:.{extent_digits}f}",
        f"flat:{round(parse_float(row.get('flatness_ratio')), ratio_digits):.{ratio_digits}f}",
        f"aspect:{round(parse_float(row.get('panel_aspect_ratio')), ratio_digits):.{ratio_digits}f}",
        f"faces:{bucket(parse_float(row.get('faces')), face_bucket_size)}",
        f"textures:{bucket(parse_float(row.get('textures')), texture_bucket_size)}",
        f"materials:{bucket(parse_float(row.get('materials')), material_bucket_size)}",
    ]
    return "|".join(parts)


def candidate_sort_key(row: dict[str, str], settings: dict[str, Any]) -> tuple[float, float, float, str]:
    rank = parse_float(row.get("candidate_rank"), 1_000_000.0)
    width = parse_float(row.get("image_width_max"))
    height = parse_float(row.get("image_height_max"))
    resolution = width * height
    face_target = float(settings.get("moderate_face_target", 25000))
    face_distance = abs(parse_float(row.get("faces")) - face_target)
    return (rank, -resolution, face_distance, row.get("asset_id", ""))


def group_candidates(rows: list[dict[str, str]], settings: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = dedupe_key(row, settings)
        row["_dedupe_key"] = key
        groups.setdefault(key, []).append(row)
    return groups


def choose_representatives(rows: list[dict[str, str]], settings: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    groups = group_candidates(rows, settings)
    chosen: list[dict[str, str]] = []
    duplicate_rows_removed = 0
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: candidate_sort_key(row, settings))
        winner = dict(ordered[0])
        winner["dedupe_key"] = key
        winner["duplicate_group_size"] = str(len(group))
        winner["dedupe_choice_reason"] = "earlier_rank_then_higher_resolution_then_moderate_faces"
        chosen.append(winner)
        duplicate_rows_removed += max(0, len(group) - 1)
    chosen.sort(key=lambda row: candidate_sort_key(row, settings))
    return chosen, {
        "input_row_count": len(rows),
        "unique_group_count": len(groups),
        "duplicate_rows_removed": duplicate_rows_removed,
        "max_duplicate_group_size": max((len(group) for group in groups.values()), default=0),
    }


def output_row(row: dict[str, str], dedup_rank: int) -> dict[str, str]:
    output = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
    output["dedup_rank"] = str(dedup_rank)
    output["dedupe_key"] = row.get("dedupe_key", row.get("_dedupe_key", ""))
    return output


def write_summary(summary: dict[str, Any], selected: list[dict[str, str]], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Phase 2L.2B ABO Dedup Summary",
        "",
        f"input rows considered: `{summary['input_row_count']}`",
        f"unique groups: `{summary['unique_group_count']}`",
        f"selected rows: `{summary['selected_count']}`",
        f"duplicate rows removed: `{summary['duplicate_rows_removed']}`",
        f"max duplicate group size: `{summary['max_duplicate_group_size']}`",
        "",
        "| Dedup Rank | Original Rank | Asset ID | Score | Faces | Textures | Materials | Dedupe Key |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            "| {dedup} | {rank} | `{asset}` | {score} | {faces} | {textures} | {materials} | `{key}` |".format(
                dedup=row["dedup_rank"],
                rank=row["candidate_rank"],
                asset=row["asset_id"],
                score=row["rank_score"],
                faces=row["faces"],
                textures=row["textures"],
                materials=row["materials"],
                key=row["dedupe_key"],
            )
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def dedupe_from_config(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    input_csv = resolve_project_path(config["input_candidates_csv"])
    max_input_rows = int(config.get("max_input_rows", 300))
    target_count = int(config.get("target_count", 30))
    settings = config.get("dedupe", {})
    rows = read_csv(input_csv)
    if max_input_rows > 0:
        rows = rows[:max_input_rows]
    chosen, summary = choose_representatives(rows, settings)
    selected = [output_row(row, index) for index, row in enumerate(chosen[:target_count], start=1)]
    summary.update(
        {
            "input_candidates_csv": str(input_csv),
            "target_count": target_count,
            "max_input_rows": max_input_rows,
            "selected_count": len(selected),
            "dedupe_settings": settings,
            "selected_asset_ids": [row["asset_id"] for row in selected],
        }
    )
    return selected, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate ABO geometry candidates.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    selected, summary = dedupe_from_config(config)
    out_csv = resolve_project_path(config["dedup_candidates_csv"])
    out_md = resolve_project_path(config["summary_md"])
    out_json = resolve_project_path(config["summary_json"])
    write_csv(out_csv, selected, OUTPUT_COLUMNS)
    write_summary(summary, selected, out_json, out_md)
    print("Phase 2L.2B ABO dedupe")
    print(f"  selected rows: {len(selected)}")
    print(f"  duplicate rows removed: {summary['duplicate_rows_removed']}")
    print(f"  out_csv: {out_csv}")
    print("PHASE2L2B_ABO_DEDUPE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
