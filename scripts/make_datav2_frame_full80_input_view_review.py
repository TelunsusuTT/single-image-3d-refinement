#!/usr/bin/env python3
"""Build full80 input-view review board and override CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts import make_datav2_frame_mini40_input_view_review as base
except ImportError:  # pragma: no cover - supports direct script execution.
    import make_datav2_frame_mini40_input_view_review as base  # type: ignore


def write_summary(
    config: dict[str, Any],
    assets: list[dict[str, str]],
    override_info: dict[str, Any],
    missing_images: list[str],
) -> dict[str, Any]:
    root = base.output_root(config) / "input_view_review"
    board_path = root / "input_view_review_board.jpg"
    summary = {
        "experiment_name": config.get("experiment_name", ""),
        "asset_count": len(assets),
        "primary_eval_asset_count": sum(1 for row in assets if row.get("eval_split") in set(config.get("eval_splits", []))),
        "train_sanity_count": sum(1 for row in assets if row.get("eval_split") == "train_sanity"),
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
        "# Phase 2L.7A Full80 Input-View Review",
        "",
        f"asset_count: `{summary['asset_count']}`",
        f"primary eval assets: `{summary['primary_eval_asset_count']}`",
        f"train sanity assets: `{summary['train_sanity_count']}`",
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
    parser = argparse.ArgumentParser(description="Create full80 input-view review board and override CSV.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = base.load_config(args.config)
    split_rows = base.read_csv(base.resolve_project_path(config["split_membership_csv"]))
    assets = base.select_assets(config, split_rows)
    review_dir = base.output_root(config) / "input_view_review"
    board_path = review_dir / "input_view_review_board.jpg"
    missing_images = base.make_review_board(config, assets, board_path)
    override_info = base.create_override_csv(config, assets, args.overwrite)
    summary = write_summary(config, assets, override_info, missing_images)
    print("Phase 2L.7A full80 input-view review")
    print(f"  asset_count: {summary['asset_count']}")
    print(f"  board: {summary['board_path']}")
    print(f"  override_csv: {override_info['path']}")
    print(f"  missing_images: {summary['missing_image_count']}")
    if summary["ok"]:
        print("PHASE2L7A_FULL80_INPUT_VIEW_REVIEW_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
