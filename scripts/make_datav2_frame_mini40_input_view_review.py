#!/usr/bin/env python3
"""Build mini40 input-view review board and override CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CONFIG = "mini40"
OVERRIDE_FIELDS = [
    "item_id",
    "source_split",
    "eval_split",
    "selected_input_view",
    "alternative_input_view",
    "primary_eval_views",
    "review_status",
    "notes",
]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def output_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_root"])


def select_assets(config: dict[str, Any], split_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    split_config = str(config.get("split_config", DEFAULT_SPLIT_CONFIG))
    rows = [row for row in split_rows if row.get("split_config", split_config) == split_config]
    eval_splits = set(config.get("eval_splits", ["val", "test"]))
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") in eval_splits:
            selected.append({**row, "source_split": row.get("split", ""), "eval_split": row.get("split", "")})
    train_count = int(config.get("optional_train_sanity_count", 0))
    for row in [row for row in rows if row.get("split") == "train"][:train_count]:
        selected.append({**row, "source_split": "train", "eval_split": "train_sanity"})
    return selected


def render_cond_path(config: dict[str, Any], item_id: str, view_id: str) -> Path:
    root = resolve_project_path(config["train_examples_root"])
    return root / item_id / "render_cond" / f"{view_id}_light_AL.png"


def create_override_csv(config: dict[str, Any], assets: list[dict[str, str]], overwrite: bool) -> dict[str, Any]:
    path = resolve_project_path(config["input_view_override_csv"])
    if path.exists() and not overwrite:
        return {"path": str(path), "created": False, "row_count": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    default_view = str(config.get("default_selected_input_view", "005"))
    alternative = str(config.get("alternative_input_view", "004"))
    primary = ";".join(config.get("primary_front_views", ["004", "005"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERRIDE_FIELDS)
        writer.writeheader()
        for row in assets:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "source_split": row.get("source_split", row.get("split", "")),
                    "eval_split": row.get("eval_split", row.get("split", "")),
                    "selected_input_view": default_view,
                    "alternative_input_view": alternative,
                    "primary_eval_views": primary,
                    "review_status": "needs_review",
                    "notes": "",
                }
            )
    return {"path": str(path), "created": True, "row_count": len(assets)}


def make_review_board(config: dict[str, Any], assets: list[dict[str, str]], board_path: Path) -> list[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on runtime image stack.
        raise RuntimeError("Pillow is required to create the input-view review board") from exc

    view_ids = list(config.get("eval_view_ids", ["000", "001", "002", "003", "004", "005"]))
    label_width = 210
    cell = 128
    header_height = 32
    row_height = cell + 28
    width = label_width + len(view_ids) * cell
    height = header_height + max(1, len(assets)) * row_height
    board = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(board)
    draw.text((8, 8), "item / split", fill=(0, 0, 0))
    for col, view_id in enumerate(view_ids):
        draw.text((label_width + col * cell + 8, 8), view_id, fill=(0, 0, 0))

    missing: list[str] = []
    for row_index, asset in enumerate(assets):
        y = header_height + row_index * row_height
        item_id = asset["item_id"]
        draw.text((8, y + 8), f"{item_id} / {asset.get('eval_split', asset.get('split', ''))}", fill=(0, 0, 0))
        for col, view_id in enumerate(view_ids):
            path = render_cond_path(config, item_id, view_id)
            x = label_width + col * cell
            if path.is_file():
                image = Image.open(path).convert("RGB").resize((cell, cell), Image.Resampling.BICUBIC)
            else:
                missing.append(str(path))
                image = Image.new("RGB", (cell, cell), (120, 30, 30))
                ImageDraw.Draw(image).text((8, 54), "missing", fill=(255, 255, 255))
            board.paste(image, (x, y + 24))
            draw.text((x + 4, y + 4), view_id, fill=(0, 0, 0))
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(board_path, quality=92)
    return missing


def write_summary(
    config: dict[str, Any],
    assets: list[dict[str, str]],
    override_info: dict[str, Any],
    missing_images: list[str],
) -> dict[str, Any]:
    root = output_root(config) / "input_view_review"
    board_path = root / "input_view_review_board.jpg"
    summary = {
        "experiment_name": config.get("experiment_name", ""),
        "asset_count": len(assets),
        "eval_splits": config.get("eval_splits", []),
        "optional_train_sanity_count": int(config.get("optional_train_sanity_count", 0)),
        "default_selected_input_view": config.get("default_selected_input_view", ""),
        "alternative_input_view": config.get("alternative_input_view", ""),
        "primary_front_views": config.get("primary_front_views", []),
        "board_path": str(board_path),
        "override_csv": override_info,
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "assets": [
            {
                "item_id": row["item_id"],
                "source_split": row.get("source_split", row.get("split", "")),
                "eval_split": row.get("eval_split", row.get("split", "")),
            }
            for row in assets
        ],
        "ok": not missing_images,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "input_view_review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.5A Mini40 Input-View Review",
        "",
        f"asset_count: `{summary['asset_count']}`",
        f"default selected input view: `{summary['default_selected_input_view']}`",
        f"alternative input view: `{summary['alternative_input_view']}`",
        f"override CSV: `{override_info['path']}`",
        f"review board: `{board_path}`",
        f"missing images: `{len(missing_images)}`",
        "",
        "| Item ID | Source Split | Eval Split |",
        "|---|---|---|",
    ]
    for row in summary["assets"]:
        lines.append(f"| `{row['item_id']}` | `{row['source_split']}` | `{row['eval_split']}` |")
    if missing_images:
        lines.extend(["", "## Missing Images", ""])
        lines.extend(f"- `{path}`" for path in missing_images)
    (root / "input_view_review_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mini40 input-view review board and override CSV.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    split_rows = read_csv(resolve_project_path(config["split_membership_csv"]))
    assets = select_assets(config, split_rows)
    review_dir = output_root(config) / "input_view_review"
    board_path = review_dir / "input_view_review_board.jpg"
    missing_images = make_review_board(config, assets, board_path)
    override_info = create_override_csv(config, assets, args.overwrite)
    summary = write_summary(config, assets, override_info, missing_images)
    print("Phase 2L.5A mini40 input-view review")
    print(f"  asset_count: {summary['asset_count']}")
    print(f"  board: {summary['board_path']}")
    print(f"  override_csv: {override_info['path']}")
    print(f"  missing_images: {summary['missing_image_count']}")
    if summary["ok"]:
        print("PHASE2L5A_MINI40_INPUT_VIEW_REVIEW_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
