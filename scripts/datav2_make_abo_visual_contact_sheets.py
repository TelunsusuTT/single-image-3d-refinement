#!/usr/bin/env python3
"""Create paginated contact sheets from Phase 2L.2C rendered thumbnails."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_PER_PAGE = 10


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def page_path(contact_sheet_root: Path, page_index: int) -> Path:
    return contact_sheet_root / f"contact_sheet_{page_index:03d}.jpg"


def contact_sheet_page_for_rank(dedup_rank: str) -> str:
    try:
        rank = int(dedup_rank)
    except ValueError:
        rank = 1
    page = ((rank - 1) // ASSETS_PER_PAGE) + 1
    return f"contact_sheet_{page:03d}.jpg"


def expected_render_paths(config: dict[str, Any], row: dict[str, str]) -> list[Path]:
    render_root = resolve_project_path(config["render_root"])
    view_ids = list(config.get("view_ids", ["000", "001", "002", "003", "004", "005"]))
    asset_id = row.get("asset_id", "")
    return [render_root / asset_id / f"{view_id}.png" for view_id in view_ids]


def missing_render_paths(config: dict[str, Any], rows: list[dict[str, str]]) -> list[Path]:
    missing: list[Path] = []
    for row in rows:
        for path in expected_render_paths(config, row):
            if not path.is_file():
                missing.append(path)
    return missing


def text_for_row(row: dict[str, str]) -> list[str]:
    return [
        f"{row.get('asset_id', '')}",
        f"dedup {row.get('dedup_rank', '')} / cand {row.get('candidate_rank', '')}",
        f"flat {row.get('flatness_ratio', '')} aspect {row.get('panel_aspect_ratio', '')}",
        f"faces {row.get('faces', '')} tex {row.get('textures', '')} mat {row.get('materials', '')}",
        f"import {row.get('import_ok', '')} render {row.get('render_ok', '')}",
    ]


def make_pages(config: dict[str, Any], rows: list[dict[str, str]]) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    contact_root = resolve_project_path(config["contact_sheet_root"])
    contact_root.mkdir(parents=True, exist_ok=True)
    resolution = int(config.get("render_resolution", 512))
    thumb_size = 132
    label_width = 360
    row_height = 164
    page_width = label_width + thumb_size * len(config.get("view_ids", []))
    page_paths: list[Path] = []
    font = ImageFont.load_default()

    for page_number, start in enumerate(range(0, len(rows), ASSETS_PER_PAGE), start=1):
        page_rows = rows[start : start + ASSETS_PER_PAGE]
        image = Image.new("RGB", (page_width, row_height * len(page_rows)), "white")
        draw = ImageDraw.Draw(image)
        for row_index, row in enumerate(page_rows):
            y0 = row_index * row_height
            draw.rectangle((0, y0, page_width - 1, y0 + row_height - 1), outline=(200, 200, 200))
            for line_index, line in enumerate(text_for_row(row)):
                draw.text((8, y0 + 8 + line_index * 18), line, fill=(20, 20, 20), font=font)
            for view_index, path in enumerate(expected_render_paths(config, row)):
                with Image.open(path) as thumb:
                    thumb = thumb.convert("RGB")
                    thumb.thumbnail((thumb_size, thumb_size))
                    x = label_width + view_index * thumb_size + (thumb_size - thumb.width) // 2
                    y = y0 + (row_height - thumb.height) // 2
                    image.paste(thumb, (x, y))
        out_path = page_path(contact_root, page_number)
        image.save(out_path, quality=92)
        page_paths.append(out_path)
    return page_paths


def write_index(config: dict[str, Any], rows: list[dict[str, str]], pages: list[Path]) -> Path:
    contact_root = resolve_project_path(config["contact_sheet_root"])
    index_path = contact_root / "contact_sheet_index.md"
    lines = [
        "# Phase 2L.2C ABO Visual Contact Sheets",
        "",
        f"assets: `{len(rows)}`",
        f"pages: `{len(pages)}`",
        "",
    ]
    for index, path in enumerate(pages, start=1):
        lines.append(f"- page {index}: `{path}`")
    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return index_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2L.2C ABO contact sheets.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    inspection_csv = resolve_project_path(config["inspection_csv"])
    rows = read_csv(inspection_csv)
    missing = missing_render_paths(config, rows)
    if missing:
        print(
            f"ERROR: missing {len(missing)} rendered thumbnail(s); first missing: {missing[0]}",
            file=sys.stderr,
        )
        return 1
    try:
        pages = make_pages(config, rows)
    except ImportError as exc:
        print(f"ERROR: Pillow is required to create contact sheets: {exc}", file=sys.stderr)
        return 1
    index_path = write_index(config, rows, pages)
    print("Phase 2L.2C ABO contact sheets")
    print(f"  assets: {len(rows)}")
    print(f"  pages: {len(pages)}")
    print(f"  index: {index_path}")
    print("PHASE2L2C_ABO_CONTACT_SHEETS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
