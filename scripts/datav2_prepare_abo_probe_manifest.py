#!/usr/bin/env python3
"""Prepare a top-k ABO probe manifest without downloading assets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_COLUMNS = [
    "candidate_rank",
    "asset_id",
    "metadata_path",
    "expected_glb_relative_path",
    "rank_score",
    "flatness_ratio",
    "panel_aspect_ratio",
    "extents",
    "extent_x",
    "extent_y",
    "extent_z",
    "faces",
    "textures",
    "materials",
    "images",
    "local_glb_exists",
    "local_glb_path",
    "needs_download",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def project_root_from_config(config: dict[str, Any]) -> Path:
    root = config.get("project_root")
    if root:
        return Path(str(root)).expanduser().resolve()
    return PROJECT_ROOT


def resolve_project_path(path_text: str | Path, project_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_abo_path(value: str) -> str:
    path = (value or "").strip()
    prefixes = (
        "s3://amazon-berkeley-objects/3dmodels/original/",
        "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/",
        "3dmodels/original/",
    )
    for prefix in prefixes:
        if path.startswith(prefix):
            path = path.removeprefix(prefix)
    return path.lstrip("/")


def extents_text(row: dict[str, str]) -> str:
    values = [row.get("extent_x", ""), row.get("extent_y", ""), row.get("extent_z", "")]
    return " x ".join(values)


def candidate_local_paths(project_root: Path, asset_id: str, relative_path: str) -> list[Path]:
    paths = [
        project_root / "data" / "raw_assets" / "phase2b_abo_selected" / f"{asset_id}.glb",
        project_root / "data" / "raw_assets" / "abo" / relative_path,
        project_root / "data" / "raw_assets" / "abo" / f"{asset_id}.glb",
        project_root / "data" / "raw_assets" / f"{asset_id}.glb",
        project_root / "caches" / "abo" / relative_path,
        project_root / "caches" / "abo" / f"{asset_id}.glb",
        project_root / "caches" / f"{asset_id}.glb",
    ]
    return paths


def find_local_glb(project_root: Path, asset_id: str, relative_path: str) -> Path | None:
    for path in candidate_local_paths(project_root, asset_id, relative_path):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def build_manifest_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    project_root = project_root_from_config(config)
    input_csv = resolve_project_path(config["input_candidates_csv"], project_root)
    top_k = int(config.get("top_k", 40))
    rows = read_csv(input_csv)[:top_k]
    manifest: list[dict[str, str]] = []
    for row in rows:
        asset_id = (row.get("asset_id") or "").strip()
        metadata_path = (row.get("path") or row.get("metadata_path") or "").strip()
        relative_path = clean_abo_path(metadata_path)
        local_path = find_local_glb(project_root, asset_id, relative_path)
        local_exists = local_path is not None
        manifest.append(
            {
                "candidate_rank": row.get("candidate_rank", ""),
                "asset_id": asset_id,
                "metadata_path": metadata_path,
                "expected_glb_relative_path": relative_path,
                "rank_score": row.get("rank_score", ""),
                "flatness_ratio": row.get("flatness_ratio", ""),
                "panel_aspect_ratio": row.get("panel_aspect_ratio", ""),
                "extents": extents_text(row),
                "extent_x": row.get("extent_x", ""),
                "extent_y": row.get("extent_y", ""),
                "extent_z": row.get("extent_z", ""),
                "faces": row.get("faces", ""),
                "textures": row.get("textures", ""),
                "materials": row.get("materials", ""),
                "images": row.get("images", ""),
                "local_glb_exists": "yes" if local_exists else "no",
                "local_glb_path": str(local_path) if local_path else "",
                "needs_download": "no" if local_exists else "yes",
            }
        )
    return manifest


def availability_paths(config: dict[str, Any], project_root: Path) -> tuple[Path, Path]:
    report = config.get("availability_report", {})
    out_md = resolve_project_path(report.get("md", "outputs/phase2l/abo_probe/availability_report.md"), project_root)
    out_json = resolve_project_path(
        report.get("json", "outputs/phase2l/abo_probe/availability_report.json"),
        project_root,
    )
    return out_md, out_json


def write_reports(config: dict[str, Any], rows: list[dict[str, str]]) -> None:
    project_root = project_root_from_config(config)
    out_md, out_json = availability_paths(config, project_root)
    local_count = sum(1 for row in rows if row["local_glb_exists"] == "yes")
    missing_count = len(rows) - local_count
    report = {
        "input_candidates_csv": str(resolve_project_path(config["input_candidates_csv"], project_root)),
        "top_k": int(config.get("top_k", 40)),
        "candidate_count": len(rows),
        "local_glb_count": local_count,
        "needs_download_count": missing_count,
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Phase 2L.2A ABO Probe Availability",
        "",
        f"candidate count: `{len(rows)}`",
        f"local GLBs found: `{local_count}`",
        f"needs download: `{missing_count}`",
        "",
        "| Rank | Asset ID | Local | Needs Download | Expected GLB | Local Path |",
        "|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{asset}` | `{local}` | `{needs}` | `{expected}` | `{path}` |".format(
                rank=row["candidate_rank"],
                asset=row["asset_id"],
                local=row["local_glb_exists"],
                needs=row["needs_download"],
                expected=row["expected_glb_relative_path"],
                path=row["local_glb_path"],
            )
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a top-k ABO probe manifest.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    project_root = project_root_from_config(config)
    manifest_path = resolve_project_path(config["candidate_manifest"], project_root)
    rows = build_manifest_rows(config)
    write_csv(manifest_path, rows, MANIFEST_COLUMNS)
    write_reports(config, rows)
    print("Phase 2L.2A ABO probe manifest")
    print(f"  candidates: {len(rows)}")
    print(f"  local GLBs found: {sum(1 for row in rows if row['local_glb_exists'] == 'yes')}")
    print(f"  needs download: {sum(1 for row in rows if row['needs_download'] == 'yes')}")
    print(f"  manifest: {manifest_path}")
    print("PHASE2L2A_ABO_PROBE_MANIFEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
