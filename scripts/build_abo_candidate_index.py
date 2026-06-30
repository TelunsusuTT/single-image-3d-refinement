#!/usr/bin/env python3
"""Build a metadata-scored ABO candidate index without downloading assets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Iterable


ABO_BUCKET = "amazon-berkeley-objects"
ABO_HTTPS_BASE = f"https://{ABO_BUCKET}.s3.amazonaws.com"
ABO_S3_BASE = f"s3://{ABO_BUCKET}"

POSITIVE_KEYWORDS = (
    "package",
    "packaging",
    "box",
    "carton",
    "label",
    "bottle",
    "can",
    "book",
    "cover",
    "poster",
    "sign",
    "sticker",
    "logo",
    "mug",
    "container",
    "tin",
    "snack",
    "cereal",
    "coffee",
    "tea",
    "case",
)

NEGATIVE_KEYWORDS = (
    "shelf",
    "shelves",
    "bookcase",
    "cabinet",
    "furniture",
    "sofa",
    "chair",
    "table",
    "bed",
    "rack",
    "planter",
    "mattress",
    "curtain",
    "carpet",
    "rug",
    "pillow",
    "blanket",
    "plain",
    "solid",
    "animal",
    "human",
    "character",
    "plush",
    "hair",
    "fur",
)

CANDIDATE_COLUMNS = [
    "candidate_id",
    "source",
    "source_id",
    "abo_path",
    "item_id",
    "name",
    "product_type",
    "node_text",
    "brand",
    "material",
    "main_image_id",
    "image_path",
    "thumbnail_url",
    "original_image_url",
    "asset_s3_uri",
    "textures",
    "images",
    "image_width_max",
    "image_height_max",
    "materials",
    "faces",
    "vertices",
    "extent_x",
    "extent_y",
    "extent_z",
    "texture_score",
    "resolution_score",
    "semantic_score",
    "geometry_score",
    "texture_density_proxy",
    "final_candidate_score",
    "selection_status",
    "selection_reason",
    "notes",
]

MODEL_ID_ALIASES = ("3dmodel_id", "model_id", "modelId", "id")
MODEL_PATH_ALIASES = (
    "abo_path",
    "path",
    "model_path",
    "glb_path",
    "file_path",
    "asset_path",
    "s3_path",
    "uri",
)
TEXTURE_ALIASES = (
    "textures",
    "texture_count",
    "num_textures",
    "num_texture_files",
    "texture_file_count",
    "image_texture_count",
)
IMAGE_COUNT_ALIASES = ("images", "image_count", "num_images", "image_file_count")
WIDTH_ALIASES = ("image_width_max", "max_width", "width", "max_texture_width", "texture_width")
HEIGHT_ALIASES = (
    "image_height_max",
    "max_height",
    "height",
    "max_texture_height",
    "texture_height",
)
MATERIAL_ALIASES = ("materials", "material_count", "num_materials")
FACE_ALIASES = ("faces", "face_count", "num_faces", "triangles", "triangle_count", "polygons")
VERTEX_ALIASES = ("vertices", "vertex_count", "num_vertices")
EXTENT_X_ALIASES = ("extent_x", "size_x", "x_extent", "dim_x")
EXTENT_Y_ALIASES = ("extent_y", "size_y", "y_extent", "dim_y")
EXTENT_Z_ALIASES = ("extent_z", "size_z", "z_extent", "dim_z")
IMAGE_ID_ALIASES = ("image_id", "id")
IMAGE_PATH_ALIASES = ("image_path", "path", "s3_path", "uri", "url")
LISTING_MODEL_ID_ALIASES = ("3dmodel_id", "model_id", "modelId")
MAIN_IMAGE_ID_ALIASES = ("main_image_id", "mainImageId", "image_id", "primary_image_id")
ITEM_ID_ALIASES = ("item_id", "asin", "listing_id", "id")
ENGLISH_KEYS = ("en_us", "en_gb", "en", "en-us", "en-gb")
LANGUAGE_KEYS = ("language", "locale", "language_tag", "languageTag", "lang")
TEXT_VALUE_KEYS = ("value", "text", "name", "display_name", "displayName")


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
    if isinstance(value, (int, float)):
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
    text = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def parse_count(row: dict[str, Any], aliases: Iterable[str]) -> float | None:
    value = row_get(row, aliases)
    parsed = parse_number(value)
    if parsed is not None:
        return parsed
    if isinstance(value, (list, tuple, dict)):
        return float(len(value))
    return None


def format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def keyword_hits(text: str, keywords: Iterable[str]) -> int:
    text_lower = text.lower()
    hits = 0
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword.lower())}\b", text_lower):
            hits += 1
    return hits


def semantic_score(text: str) -> float:
    positive = keyword_hits(text, POSITIVE_KEYWORDS)
    negative = keyword_hits(text, NEGATIVE_KEYWORDS)
    return round(clamp(4.0 + positive * 2.0 - negative * 2.2), 4)


def texture_score(textures: float | None, min_textures: int) -> float:
    if textures is None:
        return 0.0
    return round(clamp((textures / max(min_textures, 1)) * 10.0), 4)


def resolution_score(width: float | None, height: float | None, min_resolution: int) -> float:
    max_resolution = max(width or 0.0, height or 0.0)
    if max_resolution <= 0:
        return 0.0
    return round(clamp((max_resolution / max(min_resolution, 1)) * 10.0), 4)


def geometry_score(faces: float | None, max_faces: int) -> float:
    if faces is None or faces <= 0:
        return 6.0
    if faces <= max_faces:
        return 10.0
    return round(clamp((max_faces / faces) * 10.0), 4)


def texture_density_proxy(textures: float | None, faces: float | None) -> float:
    if textures is None or not faces or faces <= 0:
        return 0.0
    return round((textures * 100000.0) / faces, 4)


def final_score(
    texture: float,
    resolution: float,
    semantic: float,
    geometry: float,
    density_proxy: float,
) -> float:
    density_score = clamp(density_proxy)
    score = (
        texture * 0.20
        + resolution * 0.15
        + semantic * 0.50
        + geometry * 0.10
        + density_score * 0.05
    )
    return round(score, 4)


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_json_objects(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        for key in ("listings", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                yield from iter_json_objects(value)
                return
        yield data


def read_listings(metadata_dir: Path) -> list[dict[str, Any]]:
    listings: list[dict[str, Any]] = []
    for path in sorted(metadata_dir.glob("listings_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read().strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    listings.append(item)
        else:
            listings.extend(iter_json_objects(data))
    return listings


def normalize_path_for_s3(abo_path: str) -> str:
    path = abo_path.strip().lstrip("/")
    if not path:
        return ""
    if path.startswith("s3://"):
        return path
    if path.startswith("3dmodels/original/"):
        return f"{ABO_S3_BASE}/{path}"
    return f"{ABO_S3_BASE}/3dmodels/original/{path}"


def image_key_from_path(value: str) -> str:
    path = value.strip()
    if not path:
        return ""
    if path.startswith("s3://"):
        without_scheme = path[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        path = key if bucket == ABO_BUCKET else ""
    elif path.startswith("http://") or path.startswith("https://"):
        parsed = urllib.parse.urlparse(path)
        path = parsed.path.lstrip("/")

    path = path.lstrip("/")
    for prefix in ("images/small/", "images/original/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def catalog_image_url(image_path: str, size: str) -> str:
    image_key = image_key_from_path(image_path)
    if not image_key:
        return ""
    return f"{ABO_HTTPS_BASE}/images/{size}/{image_key}"


def sanitize_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("_") or "unknown"


def text_fields(listing: dict[str, Any]) -> dict[str, str]:
    return {
        "name": flatten_text(row_get(listing, ("item_name", "name", "title"))),
        "product_type": flatten_text(row_get(listing, ("product_type", "productType"))),
        "node_text": flatten_text(row_get(listing, ("node", "node_text", "browse_node", "category"))),
        "brand": flatten_text(row_get(listing, ("brand", "manufacturer"))),
        "material": flatten_text(row_get(listing, ("material", "materials"))),
        "keywords": flatten_text(row_get(listing, ("item_keywords", "keywords", "bullet_point"))),
        "pattern": flatten_text(row_get(listing, ("pattern",))),
        "style": flatten_text(row_get(listing, ("style",))),
    }


def passes_filter(
    textures: float | None,
    images: float | None,
    width: float | None,
    height: float | None,
    faces: float | None,
    positive_hits: int,
    args: argparse.Namespace,
) -> bool:
    if args.require_positive_keyword and positive_hits == 0:
        return False
    if textures is not None and textures < args.min_textures:
        return False
    if images is not None and images < args.min_images:
        return False
    max_resolution = max(width or 0.0, height or 0.0)
    if max_resolution > 0 and max_resolution < args.min_resolution:
        return False
    if faces is not None and faces > args.max_faces:
        return False
    return True


def build_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    metadata_dir = args.metadata_dir
    models_path = metadata_dir / "3dmodels.csv.gz"
    images_path = metadata_dir / "images.csv.gz"
    if not models_path.is_file():
        raise FileNotFoundError(f"missing metadata file: {models_path}")
    if not images_path.is_file():
        raise FileNotFoundError(f"missing metadata file: {images_path}")

    model_rows = read_csv_gz(models_path)
    image_rows = read_csv_gz(images_path)
    listings = read_listings(metadata_dir)

    models_by_id = {
        str(model_id): row
        for row in model_rows
        if (model_id := row_get(row, MODEL_ID_ALIASES)) not in (None, "")
    }
    images_by_id = {
        str(image_id): row
        for row in image_rows
        if (image_id := row_get(row, IMAGE_ID_ALIASES)) not in (None, "")
    }

    candidates: list[dict[str, str]] = []
    seen_ids: dict[str, int] = {}
    for listing in listings:
        model_id = row_get(listing, LISTING_MODEL_ID_ALIASES)
        if model_id in (None, ""):
            continue
        model_id_text = str(model_id)
        model_row = models_by_id.get(model_id_text)
        if model_row is None:
            continue

        main_image_id = row_get(listing, MAIN_IMAGE_ID_ALIASES)
        image_row = images_by_id.get(str(main_image_id), {}) if main_image_id not in (None, "") else {}
        item_id = flatten_text(row_get(listing, ITEM_ID_ALIASES))
        fields = text_fields(listing)
        text_blob = " ".join(fields.values())
        positive_hits = keyword_hits(text_blob, POSITIVE_KEYWORDS)

        abo_path = flatten_text(row_get(model_row, MODEL_PATH_ALIASES))
        image_path = flatten_text(row_get(image_row, IMAGE_PATH_ALIASES))
        thumbnail_url = catalog_image_url(image_path, "small")
        original_image_url = catalog_image_url(image_path, "original")

        textures = parse_count(model_row, TEXTURE_ALIASES)
        images = parse_count(model_row, IMAGE_COUNT_ALIASES)
        if images is None:
            images = parse_count(listing, IMAGE_COUNT_ALIASES)
        width = parse_count(image_row, WIDTH_ALIASES)
        if width is None:
            width = parse_count(model_row, WIDTH_ALIASES)
        height = parse_count(image_row, HEIGHT_ALIASES)
        if height is None:
            height = parse_count(model_row, HEIGHT_ALIASES)
        materials = parse_count(model_row, MATERIAL_ALIASES)
        faces = parse_count(model_row, FACE_ALIASES)
        vertices = parse_count(model_row, VERTEX_ALIASES)
        extent_x = parse_count(model_row, EXTENT_X_ALIASES)
        extent_y = parse_count(model_row, EXTENT_Y_ALIASES)
        extent_z = parse_count(model_row, EXTENT_Z_ALIASES)

        if not passes_filter(textures, images, width, height, faces, positive_hits, args):
            continue

        tex_score = texture_score(textures, args.min_textures)
        res_score = resolution_score(width, height, args.min_resolution)
        sem_score = semantic_score(text_blob)
        geo_score = geometry_score(faces, args.max_faces)
        density = texture_density_proxy(textures, faces)
        final = final_score(tex_score, res_score, sem_score, geo_score, density)

        base_id = sanitize_id(f"abo_{model_id_text}_{item_id}" if item_id else f"abo_{model_id_text}")
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        candidate_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"

        candidates.append(
            {
                "candidate_id": candidate_id,
                "source": "ABO",
                "source_id": model_id_text,
                "abo_path": abo_path,
                "item_id": item_id,
                "name": fields["name"],
                "product_type": fields["product_type"],
                "node_text": fields["node_text"],
                "brand": fields["brand"],
                "material": fields["material"],
                "main_image_id": str(main_image_id or ""),
                "image_path": image_path,
                "thumbnail_url": thumbnail_url,
                "original_image_url": original_image_url,
                "asset_s3_uri": normalize_path_for_s3(abo_path),
                "textures": format_number(textures),
                "images": format_number(images),
                "image_width_max": format_number(width),
                "image_height_max": format_number(height),
                "materials": format_number(materials),
                "faces": format_number(faces),
                "vertices": format_number(vertices),
                "extent_x": format_number(extent_x),
                "extent_y": format_number(extent_y),
                "extent_z": format_number(extent_z),
                "texture_score": format_number(tex_score),
                "resolution_score": format_number(res_score),
                "semantic_score": format_number(sem_score),
                "geometry_score": format_number(geo_score),
                "texture_density_proxy": format_number(density),
                "final_candidate_score": format_number(final),
                "selection_status": "candidate",
                "selection_reason": "",
                "notes": "",
            }
        )

    candidates.sort(key=lambda row: float(row["final_candidate_score"] or 0.0), reverse=True)
    return candidates


def write_candidates(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CANDIDATE_COLUMNS})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a scored ABO candidate CSV.")
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--min-textures", type=int, default=3)
    parser.add_argument("--min-images", type=int, default=3)
    parser.add_argument("--min-resolution", type=int, default=2048)
    parser.add_argument("--max-faces", type=int, default=150000)
    parser.add_argument("--require-positive-keyword", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = build_candidates(args)
    except (OSError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    write_candidates(rows, args.out_csv)
    print(f"wrote candidates CSV: {args.out_csv}")
    print(f"candidate rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
