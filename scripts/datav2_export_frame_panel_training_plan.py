#!/usr/bin/env python3
"""Export the Phase 2L.2C Data v2 frame-panel training plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_dir"])


def report_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["report_dir"])


def split_path(config: dict[str, Any], split_name: str) -> Path:
    dataset = config["dataset_name"]
    return output_dir(config) / f"{dataset}_{split_name}_split.json"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required file missing: {path}")


def write_plan(config: dict[str, Any]) -> Path:
    curated = output_dir(config) / f"{config['dataset_name']}_curated_manifest.csv"
    mini = split_path(config, "mini40")
    full = split_path(config, "full101")
    require_file(curated)
    require_file(mini)
    require_file(full)
    path = report_dir(config) / "training_plan.md"
    lines = [
        "# Data v2 Frame Panels Training Plan",
        "",
        "## Checkpoint Names",
        "",
        "- mini40: `datav2_frame_mini40_truepbr_500_lr1e6`",
        "- full80: `datav2_frame_full80_truepbr_500_lr1e6`",
        "",
        "## Recommended First Run",
        "",
        "Run `mini40` first with 500 steps and learning rate `1e-6`.",
        "",
        "## Warning",
        "",
        "Do not submit `full80` until mini40 train/eval succeeds.",
        "",
        "## Inputs",
        "",
        f"- curated manifest: `{curated}`",
        f"- mini40 split: `{mini}`",
        f"- full101 split: `{full}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Data v2 frame-panel training plan.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    path = write_plan(config)
    print("Phase 2L.2C frame-panel training plan")
    print(f"  training_plan: {path}")
    print("PHASE2L2C_FRAME_PANEL_TRAINING_PLAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
