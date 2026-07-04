#!/usr/bin/env python3
"""Mine ABO flat-panel candidates from asset-level geometry metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from datav2_inventory_metadata_sources import load_config, open_text_file, resolve_project_path


EPS = 1e-9
DEFAULT_OUT_CSV = Path("data/candidates/datav2_abo_geometry_flat_panel_candidates.csv")
DEFAULT_OUT_MD = Path("data/candidates/datav2_abo_geometry_flat_panel_candidates.md")
DEFAULT_OUT_JSON = Path("data/candidates/datav2_abo_geometry_flat_panel_summary.json")

MODEL_ID_ALIASES = ("3dmodel_id", "model_id", "modelId", "asset_id", "uid", "id")
PATH_ALIASES = ("path", "abo_path", "asset_path", "file_path", "glb_path", "uri")
MESH_ALIASES = ("meshes", "mesh_count", "num_meshes", "object_count", "objects")
MATERIAL_ALIASES = ("materials", "material_count", "num_materials")
TEXTURE_ALIASES = ("textures", "texture_count", "num_textures", "num_texture_files")
IMAGE_ALIASES = ("images", "image_count", "num_images", "image_file_count")
WIDTH_MAX_ALIASES = ("image_width_max", "max_width", "width_max", "max_texture_width", "texture_width")
WIDTH_MIN_ALIASES = ("image_width_min", "min_width", "width_min", "min_texture_width")
HEIGHT_MAX_ALIASES = ("image_height_max", "max_height", "height_max", "max_texture_height", "texture_height")
HEIGHT_MIN_ALIASES = ("image_height_min", "min_height", "height_min", "min_texture_height")
VERTEX_ALIASES = ("vertices", "vertex_count", "num_vertices")
FACE_ALIASES = ("faces", "face_count", "num_faces", "triangles", "triangle_count", "polygons")
EXTENT_X_ALIASES = ("extent_x", "size_x", "x_extent", "dim_x")
EXTENT_Y_ALIASES = ("extent_y", "size_y", "y_extent", "dim_y")
EXTENT_Z_ALIASES = ("extent_z", "size_z", "z_extent", "dim_z")
SEMANTIC_ALIASES = ("title", "name", "description", "tags", "keywords", "category", "product_type")

CANDIDATE_COLUMNS = [
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
    "geometry_flags",
    "technical_flags",
    "reject_flags",
    "raw_metadata_json",
]


def row_get(row: dict[str, Any], aliases: Iterable[str]) -> Any:
    lower_map = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
        value = lower_map.get(alias.lower())
        if value not in (None, ""):
            return value
    return None


def parse_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def number_text(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def compact_raw_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def keyword_hits(text: str, keywords: Iterable[str]) -> list[str]:
    hits: list[str] = []
    lower_text = text.lower()
    for keyword in keywords:
        key = str(keyword).strip().lower()
        if not key:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
        if re.search(pattern, lower_text):
            hits.append(str(keyword))
    return sorted(set(hits))


def semantic_text(row: dict[str, Any]) -> str:
    values = []
    for alias in SEMANTIC_ALIASES:
        value = row_get(row, (alias,))
        if value not in (None, ""):
            values.append(str(value))
    return " ".join(values)


def geometry_components(
    extent_x: float | None,
    extent_y: float | None,
    extent_z: float | None,
    settings: dict[str, Any],
) -> tuple[float, float, str, str, list[str], list[str]]:
    geometry_flags: list[str] = []
    reject_flags: list[str] = []
    extents = [extent_x, extent_y, extent_z]
    if any(value is None or value <= 0 for value in extents if value is not None) or any(value is None for value in extents):
        reject_flags.append("missing_extents")
        return 0.0, 0.0, "", "", geometry_flags, reject_flags

    small, mid, large = sorted(float(value) for value in extents if value is not None)
    flatness_ratio = small / max(mid, large, EPS)
    panel_aspect_ratio = large / max(mid, EPS)

    good = float(settings.get("flatness_good_threshold", 0.12))
    ok = float(settings.get("flatness_ok_threshold", 0.20))
    aspect_min = float(settings.get("aspect_min", 0.6))
    aspect_max = float(settings.get("aspect_max", 3.5))
    rod_aspect = float(settings.get("thin_rod_penalty_aspect", 5.0))

    if flatness_ratio <= good:
        flatness_score = 1.0
        geometry_flags.append("flatness_good")
    elif flatness_ratio <= ok:
        flatness_score = 0.65
        geometry_flags.append("flatness_ok")
    else:
        flatness_score = max(0.0, 1.0 - ((flatness_ratio - ok) / max(1.0 - ok, EPS)))
        reject_flags.append("not_flat")

    if flatness_ratio < 0.003:
        flatness_score *= 0.5
        reject_flags.append("possibly_degenerate_thickness")

    if aspect_min <= panel_aspect_ratio <= aspect_max:
        aspect_score = 1.0
        geometry_flags.append("aspect_panel_like")
    elif panel_aspect_ratio > rod_aspect:
        aspect_score = -1.0
        reject_flags.append("rod_like_aspect")
    else:
        aspect_score = 0.2
        reject_flags.append("aspect_outside_target")

    return (
        flatness_score,
        aspect_score,
        f"{flatness_ratio:.6f}",
        f"{panel_aspect_ratio:.6f}",
        geometry_flags,
        reject_flags,
    )


def count_score(value: float | None, minimum: float, missing_flag: str, ok_flag: str) -> tuple[float, list[str], list[str]]:
    technical_flags: list[str] = []
    reject_flags: list[str] = []
    if value is None:
        reject_flags.append(missing_flag)
        return 0.0, technical_flags, reject_flags
    if value >= minimum:
        technical_flags.append(ok_flag)
        return 1.0, technical_flags, reject_flags
    reject_flags.append(missing_flag)
    return 0.0, technical_flags, reject_flags


def mesh_score(meshes: float | None, max_meshes: float) -> tuple[float, list[str], list[str]]:
    if meshes is None:
        return 0.5, [], ["missing_mesh_count"]
    if meshes <= max_meshes:
        return 1.0, [f"meshes:{int(meshes)}"], []
    return -0.25, [], ["too_many_meshes"]


def face_score(faces: float | None, min_faces: float, max_faces: float) -> tuple[float, list[str], list[str]]:
    if faces is None:
        return 0.5, [], ["missing_face_count"]
    if faces < min_faces:
        return -0.25, [], ["too_few_faces"]
    if faces > max_faces:
        return -0.25, [], ["too_many_faces"]
    return 1.0, [f"faces:{int(faces)}"], []


def image_resolution_score(width: float | None, height: float | None, images: float | None) -> tuple[float, list[str]]:
    if width is not None and height is not None:
        smaller = min(width, height)
        if smaller >= 1024:
            return 1.0, [f"image_res:{int(width)}x{int(height)}"]
        if smaller >= 512:
            return 0.5, [f"image_res:{int(width)}x{int(height)}"]
        return 0.0, ["low_image_resolution"]
    if images is not None and images > 0:
        return 0.25, ["images_present_no_resolution"]
    return 0.0, []


def score_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    settings = config.get("abo_geometry_scoring", {})
    weights = settings.get("weights", {})
    extent_x = parse_number(row_get(row, EXTENT_X_ALIASES))
    extent_y = parse_number(row_get(row, EXTENT_Y_ALIASES))
    extent_z = parse_number(row_get(row, EXTENT_Z_ALIASES))
    faces = parse_number(row_get(row, FACE_ALIASES))
    vertices = parse_number(row_get(row, VERTEX_ALIASES))
    meshes = parse_number(row_get(row, MESH_ALIASES))
    materials = parse_number(row_get(row, MATERIAL_ALIASES))
    textures = parse_number(row_get(row, TEXTURE_ALIASES))
    images = parse_number(row_get(row, IMAGE_ALIASES))
    image_width_max = parse_number(row_get(row, WIDTH_MAX_ALIASES))
    image_width_min = parse_number(row_get(row, WIDTH_MIN_ALIASES))
    image_height_max = parse_number(row_get(row, HEIGHT_MAX_ALIASES))
    image_height_min = parse_number(row_get(row, HEIGHT_MIN_ALIASES))
    width_for_score = image_width_max if image_width_max is not None else image_width_min
    height_for_score = image_height_max if image_height_max is not None else image_height_min

    flat_score, aspect_score, flatness_ratio, panel_aspect_ratio, geometry_flags, reject_flags = geometry_components(
        extent_x, extent_y, extent_z, settings
    )
    technical_flags: list[str] = []

    texture_score, flags, rejects = count_score(
        textures, float(settings.get("min_textures", 1)), "missing_textures", f"textures:{int(textures or 0)}"
    )
    technical_flags.extend(flags)
    reject_flags.extend(rejects)

    material_score, flags, rejects = count_score(
        materials, float(settings.get("min_materials", 1)), "missing_materials", f"materials:{int(materials or 0)}"
    )
    technical_flags.extend(flags)
    reject_flags.extend(rejects)

    mesh_count_score, flags, rejects = mesh_score(meshes, float(settings.get("max_meshes", 3)))
    technical_flags.extend(flags)
    reject_flags.extend(rejects)

    face_count_score, flags, rejects = face_score(
        faces,
        float(settings.get("min_faces", 500)),
        float(settings.get("max_faces", 150000)),
    )
    technical_flags.extend(flags)
    reject_flags.extend(rejects)

    image_score, flags = image_resolution_score(width_for_score, height_for_score, images)
    technical_flags.extend(flags)

    text = semantic_text(row)
    positive_hits = keyword_hits(text, config.get("positive_keywords", []))
    negative_hits = keyword_hits(text, config.get("negative_keywords", []))
    if positive_hits:
        technical_flags.append("semantic_positive:" + ",".join(positive_hits))
    if negative_hits:
        reject_flags.append("semantic_negative:" + ",".join(negative_hits))

    rank_score = (
        flat_score * float(weights.get("flatness_score", 4.0))
        + aspect_score * float(weights.get("rectangular_aspect_score", 3.0))
        + texture_score * float(weights.get("texture_score", 2.0))
        + material_score * float(weights.get("material_score", 1.5))
        + mesh_count_score * float(weights.get("mesh_count_score", 1.0))
        + face_count_score * float(weights.get("face_count_score", 1.0))
        + image_score * float(weights.get("image_resolution_score", 0.5))
        + len(positive_hits) * float(weights.get("semantic_keyword_score", 1.0))
        - len(negative_hits) * float(weights.get("negative_keyword_penalty", 2.0))
    )

    asset_id = str(row_get(row, MODEL_ID_ALIASES) or "").strip()
    path = str(row_get(row, PATH_ALIASES) or "").strip()
    if not asset_id:
        asset_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", path).strip("_")[:96] or "unknown_asset"

    return {
        "candidate_rank": "",
        "asset_id": asset_id,
        "path": path,
        "rank_score": f"{rank_score:.4f}",
        "flatness_ratio": flatness_ratio,
        "panel_aspect_ratio": panel_aspect_ratio,
        "extent_x": number_text(extent_x),
        "extent_y": number_text(extent_y),
        "extent_z": number_text(extent_z),
        "faces": number_text(faces),
        "vertices": number_text(vertices),
        "meshes": number_text(meshes),
        "materials": number_text(materials),
        "textures": number_text(textures),
        "images": number_text(images),
        "image_width_max": number_text(image_width_max),
        "image_height_max": number_text(image_height_max),
        "geometry_flags": ";".join(sorted(set(geometry_flags))),
        "technical_flags": ";".join(sorted(set(technical_flags))),
        "reject_flags": ";".join(sorted(set(reject_flags))),
        "raw_metadata_json": compact_raw_json(row),
    }


def candidate_source_files(config: dict[str, Any]) -> list[Path]:
    settings = config.get("abo_geometry_scoring", {})
    paths = settings.get("candidate_source_files") or ["data/metadata/abo/3dmodels.csv.gz"]
    return [resolve_project_path(str(path)) for path in paths]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with open_text_file(path) as handle:
        return list(csv.DictReader(handle))


def mine_candidates(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    for source_file in candidate_source_files(config):
        if not source_file.exists():
            continue
        source_rows = read_csv_rows(source_file)
        source_counts[str(source_file)] = len(source_rows)
        rows.extend(score_row(row, config) for row in source_rows)

    rows.sort(
        key=lambda row: (
            -float(row["rank_score"]),
            row.get("reject_flags", ""),
            row.get("asset_id", ""),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["candidate_rank"] = str(index)

    scores = [float(row["rank_score"]) for row in rows]
    summary = {
        "candidate_source_files": [str(path) for path in candidate_source_files(config)],
        "source_row_counts": source_counts,
        "candidate_count": len(rows),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "reject_flag_counts": {},
        "top_candidates": [
            {
                "candidate_rank": row["candidate_rank"],
                "asset_id": row["asset_id"],
                "path": row["path"],
                "rank_score": row["rank_score"],
                "flatness_ratio": row["flatness_ratio"],
                "panel_aspect_ratio": row["panel_aspect_ratio"],
                "reject_flags": row["reject_flags"],
            }
            for row in rows[:25]
        ],
    }
    reject_counts: dict[str, int] = {}
    for row in rows:
        for flag in row["reject_flags"].split(";"):
            if flag:
                reject_counts[flag] = reject_counts.get(flag, 0) + 1
    summary["reject_flag_counts"] = dict(sorted(reject_counts.items()))
    return rows, summary


def filter_rows(rows: list[dict[str, str]], min_score: float | None, top_k: int) -> list[dict[str, str]]:
    filtered = rows
    if min_score is not None:
        filtered = [row for row in filtered if float(row["rank_score"]) >= min_score]
    if top_k > 0:
        filtered = filtered[:top_k]
    for index, row in enumerate(filtered, start=1):
        row["candidate_rank"] = str(index)
    return filtered


def write_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: str) -> str:
    text = (value or "").replace("\n", " ").replace("|", "\\|")
    return text[:160] + "..." if len(text) > 160 else text


def write_markdown(rows: list[dict[str, str]], summary: dict[str, Any], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data v2 ABO Geometry Flat-Panel Candidates",
        "",
        f"candidate count: `{summary['candidate_count']}`",
        f"written rows: `{len(rows)}`",
        "",
        "| Rank | Score | Asset ID | Path | Flatness | Aspect | Extents | Tech | Reject |",
        "|---:|---:|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        extents = f"{row['extent_x']} x {row['extent_y']} x {row['extent_z']}"
        lines.append(
            "| {rank} | {score} | `{asset}` | {path} | {flat} | {aspect} | {extents} | {tech} | {reject} |".format(
                rank=row["candidate_rank"],
                score=row["rank_score"],
                asset=md_cell(row["asset_id"]),
                path=md_cell(row["path"]),
                flat=row["flatness_ratio"],
                aspect=row["panel_aspect_ratio"],
                extents=md_cell(extents),
                tech=md_cell(row["technical_flags"]),
                reject=md_cell(row["reject_flags"]),
            )
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine ABO flat-panel candidates from geometry metadata.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    rows, summary = mine_candidates(config)
    filtered = filter_rows(rows, args.min_score, args.top_k)
    summary["rows_after_filters"] = len(filtered)
    summary["top_k"] = args.top_k
    summary["min_score"] = args.min_score

    print("Phase 2L.1B ABO geometry candidate mining")
    print(f"  candidate rows scored: {summary['candidate_count']}")
    print(f"  rows after filters: {len(filtered)}")
    if args.dry_run:
        print("  dry-run: no candidate files written")
        print("PHASE2L1B_ABO_GEOMETRY_CANDIDATES_OK")
        return 0

    out_csv = resolve_project_path(DEFAULT_OUT_CSV)
    out_md = resolve_project_path(DEFAULT_OUT_MD)
    out_json = resolve_project_path(DEFAULT_OUT_JSON)
    write_csv(filtered, out_csv)
    write_markdown(filtered, summary, out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  out_csv: {out_csv}")
    print(f"  out_md: {out_md}")
    print(f"  out_json: {out_json}")
    print("PHASE2L1B_ABO_GEOMETRY_CANDIDATES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
