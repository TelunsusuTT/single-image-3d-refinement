#!/usr/bin/env python3
"""Create Phase 2K.4 rendered-view eval configs for local GPU rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def resolve_project_path(path_text: str, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def join_config_path(root_text: str, *parts: str) -> str:
    return "/".join([root_text.rstrip("/"), *parts])


def render_eval_config(config: dict[str, Any], input_view: str) -> dict[str, Any]:
    output_root_text = str(config["output_root"])
    cases = {}
    for asset_id in config["asset_ids"]:
        cases[asset_id] = {
            "base_glb": join_config_path(
                output_root_text,
                "infer",
                "base",
                asset_id,
                f"input_{input_view}",
                "base_textured_mesh.glb",
            ),
            "finetuned_glb": join_config_path(
                output_root_text,
                "infer",
                "finetuned",
                asset_id,
                f"input_{input_view}",
                "finetuned_textured_mesh.glb",
            ),
        }
    return {
        "output_root": join_config_path(output_root_text, "rendered_eval", f"input_{input_view}"),
        "view_ids": ALL_VIEW_IDS,
        "render_resolution": 512,
        "background_color": [0.28, 0.28, 0.28],
        "reference_image_path_template": "data/hy3dpaint_train_examples/pilot_v1/{asset_id}/render_cond/{view_id}_light_AL.png",
        "cases": cases,
        "phase2k4_input_view": input_view,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2K.4 render eval configs.")
    parser.add_argument("--cases-config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.cases_config)
    output_root = resolve_project_path(config["output_root"])
    config_dir = output_root / "render_eval_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for input_view in config["input_views"]:
        data = render_eval_config(config, input_view)
        path = config_dir / f"render_eval_input_{input_view}.json"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        created.append(path)
    print("Phase 2K.4 render eval configs")
    for path in created:
        print(f"  {path}")
    print("PHASE2K4_RENDER_EVAL_CONFIGS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
