#!/usr/bin/env python3
"""Mine ranked flat-panel candidates from local metadata only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from datav2_inventory_metadata_sources import (
    PROJECT_ROOT,
    detect_format,
    load_config,
    open_text_file,
    resolve_metadata_files,
    resolve_output_path,
    rows_from_json_value,
)


TITLE_ALIASES = ("title", "name", "object_name", "objectName", "item_name", "product_name", "display_name")
DESCRIPTION_ALIASES = ("description", "desc", "caption", "summary", "product_description")
TAGS_ALIASES = ("tags", "tag", "categories", "keywords", "item_keywords", "labels")
CATEGORY_ALIASES = ("category", "product_type", "class", "class_name", "node", "node_text")
ID_ALIASES = ("asset_id", "uid", "uuid", "id", "source_id", "3dmodel_id", "model_id", "modelId", "item_id")
LICENSE_ALIASES = ("license", "licence", "license_name", "rights")
FORMAT_ALIASES = ("file_format", "format", "extension", "ext", "file_extension")
SIZE_ALIASES = ("file_size_bytes", "size_bytes", "file_size", "filesize", "size")
FACE_ALIASES = ("face_count", "faces", "num_faces", "poly_count", "polygons", "triangles", "triangle_count")
VERTEX_ALIASES = ("vertex_count", "vertices", "num_vertices")
OBJECT_ALIASES = ("object_count", "objects", "num_objects", "mesh_count", "mesh_object_count")
MATERIAL_ALIASES = ("material_count", "materials", "num_materials")
TEXTURE_ALIASES = ("texture_count", "textures", "num_textures", "texture_file_count", "image_texture_count")
MISSING_TEXTURE_ALIASES = ("missing_textures", "missing_texture_files", "has_missing_textures")
URL_ALIASES = ("url", "uri", "path", "file_path", "asset_path", "abo_path", "glb_path", "download_url", "s3_uri")
ENGLISH_KEYS = ("en_us", "en_gb", "en", "en-us", "en-gb")
LANGUAGE_KEYS = ("language", "locale", "language_tag", "languageTag", "lang")
TEXT_VALUE_KEYS = ("value", "text", "name", "display_name", "displayName")

CANDIDATE_COLUMNS = [
    "candidate_rank",
    "source",
    "source_file",
    "asset_id",
    "title",
    "description",
    "tags",
    "category",
    "license",
    "file_format",
    "file_size_bytes",
    "face_count",
    "vertex_count",
    "object_count",
    "material_count",
    "texture_count",
    "missing_textures",
    "url_or_path",
    "rank_score",
    "positive_keyword_hits",
    "negative_keyword_hits",
    "technical_flags",
    "auto_reasons",
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


def dict_english_value(value: dict[str, Any]) -> Any:
    lower_keys = {str(key).lower(): key for key in value.keys()}
    for english_key in ENGLISH_KEYS:
        original_key = lower_keys.get(english_key)
        if original_key is not None:
            return value[original_key]
    language_value = ""
    for key in LANGUAGE_KEYS:
        if key in value:
            language_value = str(value[key]).lower()
            break
    if language_value.startswith("en"):
        for key in TEXT_VALUE_KEYS:
            if key in value:
                return value[key]
    return None


def flatten_text(value: Any, prefer_english: bool = True) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if prefer_english:
            english_parts = []
            for item in value:
                if isinstance(item, dict):
                    english_value = dict_english_value(item)
                    if english_value is not None:
                        english_parts.append(flatten_text(english_value, prefer_english=True))
            english_text = " ".join(part for part in english_parts if part)
            if english_text:
                return english_text
        return " ".join(flatten_text(item, prefer_english=prefer_english) for item in value if item is not None)
    if isinstance(value, dict):
        if prefer_english:
            english_value = dict_english_value(value)
            if english_value is not None:
                return flatten_text(english_value, prefer_english=prefer_english)
        return " ".join(flatten_text(item, prefer_english=prefer_english) for item in value.values() if item is not None)
    return str(value)


def parse_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, list):
        for item in value:
            parsed = parse_number(item)
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, dict):
        for item in value.values():
            parsed = parse_number(item)
            if parsed is not None:
                return parsed
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def parse_bool_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "yes"
    if text in {"0", "false", "no", "n"}:
        return "no"
    return text


def number_text(value: Any) -> str:
    parsed = parse_number(value)
    if parsed is None:
        return ""
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def infer_source(path: Path) -> str:
    text = str(path).lower()
    if "objaverse_xl" in text or "objaverse-xl" in text:
        return "objaverse_xl"
    if "objaverse" in text:
        return "objaverse"
    if "/abo/" in text or "\\abo\\" in text or text.endswith("abo"):
        return "abo"
    return "unknown"


def infer_file_format(row: dict[str, Any], url_or_path: str) -> str:
    explicit = flatten_text(row_get(row, FORMAT_ALIASES)).strip().lower().lstrip(".")
    if explicit:
        return explicit
    suffix = Path(url_or_path.split("?", 1)[0]).suffix.lower().lstrip(".")
    return suffix


def compact_raw_json(row: dict[str, Any]) -> str:
    try:
        return json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return json.dumps({str(key): str(value) for key, value in row.items()}, ensure_ascii=True, sort_keys=True)


def normalize_record(row: dict[str, Any], source_file: Path) -> dict[str, str]:
    url_or_path = flatten_text(row_get(row, URL_ALIASES))
    normalized = {
        "candidate_rank": "",
        "source": infer_source(source_file),
        "source_file": str(source_file),
        "asset_id": flatten_text(row_get(row, ID_ALIASES)),
        "title": flatten_text(row_get(row, TITLE_ALIASES)),
        "description": flatten_text(row_get(row, DESCRIPTION_ALIASES)),
        "tags": flatten_text(row_get(row, TAGS_ALIASES)),
        "category": flatten_text(row_get(row, CATEGORY_ALIASES)),
        "license": flatten_text(row_get(row, LICENSE_ALIASES)),
        "file_format": infer_file_format(row, url_or_path),
        "file_size_bytes": number_text(row_get(row, SIZE_ALIASES)),
        "face_count": number_text(row_get(row, FACE_ALIASES)),
        "vertex_count": number_text(row_get(row, VERTEX_ALIASES)),
        "object_count": number_text(row_get(row, OBJECT_ALIASES)),
        "material_count": number_text(row_get(row, MATERIAL_ALIASES)),
        "texture_count": number_text(row_get(row, TEXTURE_ALIASES)),
        "missing_textures": parse_bool_text(row_get(row, MISSING_TEXTURE_ALIASES)),
        "url_or_path": url_or_path,
        "rank_score": "",
        "positive_keyword_hits": "",
        "negative_keyword_hits": "",
        "technical_flags": "",
        "auto_reasons": "",
        "raw_metadata_json": compact_raw_json(row),
    }
    if not normalized["asset_id"]:
        stable_text = normalized["url_or_path"] or normalized["title"] or compact_raw_json(row)[:80]
        normalized["asset_id"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", stable_text).strip("_")[:96] or "unknown_asset"
    return normalized


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


def numeric_field(candidate: dict[str, str], field: str) -> float | None:
    return parse_number(candidate.get(field, ""))


def technical_scores(candidate: dict[str, str], config: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    filters = config.get("technical_filters", {})
    preferred_formats = {str(item).lower().lstrip(".") for item in filters.get("prefer_file_extensions", [])}
    flags: list[str] = []

    file_format = candidate["file_format"].lower().lstrip(".")
    file_format_score = 1.0 if file_format in preferred_formats else 0.0
    if file_format_score:
        flags.append(f"preferred_format:{file_format}")

    texture_count = numeric_field(candidate, "texture_count")
    material_count = numeric_field(candidate, "material_count")
    object_count = numeric_field(candidate, "object_count")
    face_count = numeric_field(candidate, "face_count")
    file_size = numeric_field(candidate, "file_size_bytes")

    texture_score = 1.0 if texture_count is not None and texture_count > 0 else 0.0
    if texture_score:
        flags.append(f"textures:{int(texture_count)}")
    if candidate.get("missing_textures") == "yes":
        texture_score -= 1.0
        flags.append("missing_textures")

    material_score = 1.0 if material_count is not None and material_count > 0 else 0.0
    if material_score:
        flags.append(f"materials:{int(material_count)}")

    geometry_score = 0.5
    if object_count is not None:
        geometry_score = 1.0 if object_count <= 3 else 0.0
        flags.append(f"object_count:{int(object_count)}")
    if face_count is not None:
        min_faces = filters.get("optional_min_face_count")
        max_faces = filters.get("optional_max_face_count")
        if min_faces is not None and face_count < float(min_faces):
            geometry_score -= 0.5
            flags.append("below_min_faces")
        if max_faces is not None and face_count > float(max_faces):
            geometry_score -= 0.5
            flags.append("above_max_faces")
    if file_size is not None:
        min_size = filters.get("optional_min_file_size_bytes")
        max_size = filters.get("optional_max_file_size_bytes")
        if min_size is not None and file_size < float(min_size):
            geometry_score -= 0.5
            flags.append("below_min_size")
        if max_size is not None and file_size > float(max_size):
            geometry_score -= 0.5
            flags.append("above_max_size")

    return (
        {
            "texture_indicator_score": max(texture_score, -1.0),
            "material_indicator_score": material_score,
            "file_format_score": file_format_score,
            "geometry_sanity_score": max(geometry_score, -1.0),
        },
        flags,
    )


def score_candidate(candidate: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    weights = config.get("rank_weights", {})
    combined_text = " ".join(
        candidate.get(field, "")
        for field in ("title", "description", "tags", "category", "url_or_path")
    )
    positives = keyword_hits(combined_text, config.get("positive_keywords", []))
    negatives = keyword_hits(combined_text, config.get("negative_keywords", []))
    tech_scores, flags = technical_scores(candidate, config)
    source_priority = float(config.get("source_priority", {}).get(candidate["source"], 0.2))

    score = (
        len(positives) * float(weights.get("positive_keyword_score", 1.0))
        - len(negatives) * float(weights.get("negative_keyword_penalty", 1.0))
        + tech_scores["texture_indicator_score"] * float(weights.get("texture_indicator_score", 1.0))
        + tech_scores["material_indicator_score"] * float(weights.get("material_indicator_score", 1.0))
        + tech_scores["file_format_score"] * float(weights.get("file_format_score", 1.0))
        + tech_scores["geometry_sanity_score"] * float(weights.get("geometry_sanity_score", 1.0))
        + source_priority * float(weights.get("source_priority_score", 1.0))
    )

    candidate["rank_score"] = f"{score:.4f}"
    candidate["positive_keyword_hits"] = ";".join(positives)
    candidate["negative_keyword_hits"] = ";".join(negatives)
    candidate["technical_flags"] = ";".join(flags)
    candidate["auto_reasons"] = "; ".join(
        part
        for part in (
            f"positive:{','.join(positives)}" if positives else "",
            f"negative:{','.join(negatives)}" if negatives else "",
            f"technical:{','.join(flags)}" if flags else "",
        )
        if part
    )
    return candidate


def iter_metadata_records(path: Path, max_rows: int = 0) -> Iterator[dict[str, Any]]:
    file_format = detect_format(path)
    count = 0
    if file_format == "csv":
        with open_text_file(path) as handle:
            for row in csv.DictReader(handle):
                yield dict(row)
                count += 1
                if max_rows and count >= max_rows:
                    return
    elif file_format == "jsonl":
        with open_text_file(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
                    count += 1
                    if max_rows and count >= max_rows:
                        return
    elif file_format == "json":
        with open_text_file(path) as handle:
            rows = rows_from_json_value(json.load(handle))
        for row in rows:
            if isinstance(row, dict):
                yield row
                count += 1
                if max_rows and count >= max_rows:
                    return


def mine_candidates(config: dict[str, Any], max_rows_per_source: int = 0) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows_seen = 0
    parse_errors: list[str] = []
    candidates: list[dict[str, str]] = []
    for source_file in resolve_metadata_files(config):
        if detect_format(source_file) not in {"csv", "json", "jsonl"}:
            continue
        try:
            for raw_row in iter_metadata_records(source_file, max_rows=max_rows_per_source):
                rows_seen += 1
                normalized = normalize_record(raw_row, source_file)
                candidates.append(score_candidate(normalized, config))
        except Exception as exc:  # noqa: BLE001 - keep mining other sources.
            parse_errors.append(f"{source_file}: {type(exc).__name__}: {exc}")

    candidates.sort(key=lambda row: (-float(row.get("rank_score", "0") or 0), row["source"], row["asset_id"]))
    for rank, row in enumerate(candidates, start=1):
        row["candidate_rank"] = str(rank)

    summary = {
        "target_subclass": config.get("target_subclass", ""),
        "metadata_file_count": len(resolve_metadata_files(config)),
        "rows_seen": rows_seen,
        "candidates_scored": len(candidates),
        "parse_errors": parse_errors,
        "source_counts": dict(Counter(row["source"] for row in candidates)),
    }
    if candidates:
        scores = [float(row["rank_score"]) for row in candidates]
        summary["score_min"] = min(scores)
        summary["score_max"] = max(scores)
        summary["top_candidates"] = [
            {
                "candidate_rank": row["candidate_rank"],
                "source": row["source"],
                "asset_id": row["asset_id"],
                "title": row["title"],
                "rank_score": row["rank_score"],
                "positive_keyword_hits": row["positive_keyword_hits"],
                "negative_keyword_hits": row["negative_keyword_hits"],
            }
            for row in candidates[:25]
        ]
    return candidates, summary


def filtered_candidates(
    candidates: list[dict[str, str]],
    min_score: float | None,
    top_k: int,
) -> list[dict[str, str]]:
    rows = candidates
    if min_score is not None:
        rows = [row for row in rows if float(row["rank_score"]) >= min_score]
    if top_k > 0:
        rows = rows[:top_k]
    for rank, row in enumerate(rows, start=1):
        row["candidate_rank"] = str(rank)
    return rows


def write_candidates_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: str) -> str:
    text = (value or "").replace("\n", " ").replace("|", "\\|")
    return text[:180] + "..." if len(text) > 180 else text


def write_candidates_md(rows: list[dict[str, str]], summary: dict[str, Any], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data v2 Flat Panel Metadata Candidates",
        "",
        f"target subclass: `{summary.get('target_subclass', '')}`",
        f"rows seen: `{summary.get('rows_seen', 0)}`",
        f"candidates written: `{len(rows)}`",
        "",
        "| Rank | Score | Source | Asset ID | Title | Positive hits | Negative hits | Technical flags |",
        "|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {score} | {source} | `{asset}` | {title} | {pos} | {neg} | {tech} |".format(
                rank=row["candidate_rank"],
                score=row["rank_score"],
                source=row["source"],
                asset=md_cell(row["asset_id"]),
                title=md_cell(row["title"]),
                pos=md_cell(row["positive_keyword_hits"]),
                neg=md_cell(row["negative_keyword_hits"]),
                tech=md_cell(row["technical_flags"]),
            )
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def output_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    outputs = config.get("outputs", {})
    return (
        resolve_output_path(outputs.get("candidates_csv", "data/candidates/datav2_flat_panel_metadata_candidates.csv")),
        resolve_output_path(outputs.get("candidates_md", "data/candidates/datav2_flat_panel_metadata_candidates.md")),
        resolve_output_path(outputs.get("candidate_summary_json", "data/candidates/datav2_flat_panel_candidate_summary.json")),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine flat-panel candidates from metadata only.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    candidates, summary = mine_candidates(config, max_rows_per_source=args.max_rows_per_source)
    rows = filtered_candidates(candidates, min_score=args.min_score, top_k=args.top_k)
    summary["candidates_after_filters"] = len(rows)
    summary["min_score_filter"] = args.min_score
    summary["top_k"] = args.top_k

    out_csv, out_md, out_json = output_paths(config)
    print("Phase 2L.1 flat-panel candidate mining")
    print(f"  rows seen: {summary['rows_seen']}")
    print(f"  candidates scored: {summary['candidates_scored']}")
    print(f"  candidates after filters: {len(rows)}")
    if args.dry_run:
        print("  dry-run: no candidate files written")
        print("PHASE2L1_FLAT_PANEL_CANDIDATES_MINED_OK")
        return 0

    write_candidates_csv(rows, out_csv)
    write_candidates_md(rows, summary, out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  out_csv: {out_csv}")
    print(f"  out_md: {out_md}")
    print(f"  out_json: {out_json}")
    print("PHASE2L1_FLAT_PANEL_CANDIDATES_MINED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
