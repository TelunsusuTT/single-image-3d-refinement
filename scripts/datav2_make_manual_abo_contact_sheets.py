#!/usr/bin/env python3
"""Create contact sheets from Phase 2L.2B manual ABO preview renders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_root"])


def render_root(config: dict[str, Any]) -> Path:
    return output_root(config) / "renders"


def contact_sheet_root(config: dict[str, Any]) -> Path:
    return output_root(config) / "contact_sheets"


def inspection_csv_path(config: dict[str, Any]) -> Path:
    return output_root(config) / "inspection_results.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def view_ids(config: dict[str, Any]) -> list[str]:
    return list(config.get("view_ids", ["000", "001", "002", "003", "004", "005"]))


def render_paths_for_row(config: dict[str, Any], row: dict[str, str]) -> list[Path]:
    item_id = row.get("item_id", "")
    return [render_root(config) / item_id / f"{view_id}.png" for view_id in view_ids(config)]


def row_has_all_renders(config: dict[str, Any], row: dict[str, str]) -> bool:
    return all(path.is_file() for path in render_paths_for_row(config, row))


def rows_with_renders(config: dict[str, Any], rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    complete: list[dict[str, str]] = []
    missing: list[str] = []
    for row in rows:
        missing_paths = [str(path) for path in render_paths_for_row(config, row) if not path.is_file()]
        if missing_paths:
            missing.append(f"{row.get('item_id', '')}: {missing_paths[0]}")
        else:
            complete.append(row)
    return complete, missing


def page_path(root: Path, page_index: int) -> Path:
    return root / f"contact_sheet_{page_index:03d}.jpg"


def text_for_row(row: dict[str, str]) -> list[str]:
    return [
        row.get("item_id", ""),
        f"import {row.get('import_ok', '')} render {row.get('render_ok', '')}",
        f"faces {row.get('face_count', '')} mesh {row.get('mesh_count', '')}",
        f"mat {row.get('material_count', '')} tex {row.get('texture_count') or row.get('texture_image_count', '')}",
        f"bbox {row.get('bbox_extents', '')}",
    ]


def make_pages(config: dict[str, Any], rows: list[dict[str, str]]) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    root = contact_sheet_root(config)
    root.mkdir(parents=True, exist_ok=True)
    assets_per_page = int(config.get("contact_sheet_assets_per_page", 12))
    thumb_size = 116
    label_width = 390
    row_height = 150
    page_width = label_width + thumb_size * len(view_ids(config))
    font = ImageFont.load_default()
    pages: list[Path] = []

    for page_number, start in enumerate(range(0, len(rows), assets_per_page), start=1):
        page_rows = rows[start : start + assets_per_page]
        image = Image.new("RGB", (page_width, row_height * max(len(page_rows), 1)), "white")
        draw = ImageDraw.Draw(image)
        for row_index, row in enumerate(page_rows):
            y0 = row_index * row_height
            draw.rectangle((0, y0, page_width - 1, y0 + row_height - 1), outline=(205, 205, 205))
            for line_index, line in enumerate(text_for_row(row)):
                draw.text((8, y0 + 8 + line_index * 18), line, fill=(20, 20, 20), font=font)
            for view_index, path in enumerate(render_paths_for_row(config, row)):
                with Image.open(path) as thumb:
                    thumb = thumb.convert("RGB")
                    thumb.thumbnail((thumb_size, thumb_size))
                    x = label_width + view_index * thumb_size + (thumb_size - thumb.width) // 2
                    y = y0 + (row_height - thumb.height) // 2
                    image.paste(thumb, (x, y))
        out_path = page_path(root, page_number)
        image.save(out_path, quality=92)
        pages.append(out_path)
    return pages


def write_index(config: dict[str, Any], rows: list[dict[str, str]], pages: list[Path], missing: list[str]) -> Path:
    root = contact_sheet_root(config)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "contact_sheet_index.md"
    assets_per_page = int(config.get("contact_sheet_assets_per_page", 12))
    lines = [
        "# Phase 2L.2B Manual ABO Contact Sheets",
        "",
        f"assets with complete renders: `{len(rows)}`",
        f"pages: `{len(pages)}`",
        f"missing render sets: `{len(missing)}`",
        "",
        "## Pages",
        "",
    ]
    for index, path in enumerate(pages, start=1):
        lines.append(f"- page {index}: `{path}`")
    lines.extend(
        [
            "",
            "## Assets",
            "",
            "| Item ID | Page | Import | Render | Faces | Materials | Textures | BBox Extents |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for index, row in enumerate(rows):
        page = (index // assets_per_page) + 1
        lines.append(
            "| `{item}` | `contact_sheet_{page:03d}.jpg` | `{import_ok}` | `{render_ok}` | {faces} | {materials} | {textures} | `{bbox}` |".format(
                item=row.get("item_id", ""),
                page=page,
                import_ok=row.get("import_ok", ""),
                render_ok=row.get("render_ok", ""),
                faces=row.get("face_count", ""),
                materials=row.get("material_count", ""),
                textures=row.get("texture_count") or row.get("texture_image_count", ""),
                bbox=row.get("bbox_extents", ""),
            )
        )
    if missing:
        lines.extend(["", "## Missing Renders", ""])
        lines.extend(f"- `{entry}`" for entry in missing)
    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return index_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2L.2B manual ABO contact sheets.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    rows = read_csv(inspection_csv_path(config))
    complete_rows, missing = rows_with_renders(config, rows)
    try:
        pages = make_pages(config, complete_rows)
    except ImportError as exc:
        print(f"ERROR: Pillow is required to create contact sheets: {exc}", file=sys.stderr)
        return 1
    index_path = write_index(config, complete_rows, pages, missing)
    print("Phase 2L.2B manual ABO contact sheets")
    print(f"  assets with complete renders: {len(complete_rows)}")
    print(f"  missing render sets: {len(missing)}")
    print(f"  pages: {len(pages)}")
    print(f"  index: {index_path}")
    print("PHASE2L2B_MANUAL_ABO_CONTACT_SHEETS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
