#!/usr/bin/env python3
"""Create Phase 2L.5B render-eval config from mini40 inference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_cases(config: dict[str, Any]) -> dict[str, Any]:
    return load_config(resolve_project_path(config["output_root"]) / "eval_cases.json")


def make_render_eval_config(config: dict[str, Any], eval_cases: dict[str, Any]) -> dict[str, Any]:
    root = resolve_project_path(config["output_root"])
    sample_root = resolve_project_path(config["train_examples_root"])
    view_ids = list(config.get("eval_view_ids", DEFAULT_VIEW_IDS))
    cases: dict[str, Any] = {}
    for case in eval_cases.get("cases", []):
        item_id = case["item_id"]
        eval_split = case["eval_split"]
        cases[item_id] = {
            "base_glb": str(root / "infer" / "base" / eval_split / item_id / "base_textured_mesh.glb"),
            "finetuned_glb": str(root / "infer" / "finetuned" / eval_split / item_id / "finetuned_textured_mesh.glb"),
            "eval_split": eval_split,
            "source_split": case.get("source_split", ""),
            "selected_input_view": case.get("selected_input_view", ""),
            "primary_front_views": case.get("primary_eval_views", config.get("primary_front_views", ["004", "005"])),
            "reference_images": {
                view_id: str(sample_root / item_id / "render_cond" / f"{view_id}_light_AL.png")
                for view_id in view_ids
            },
        }
    return {
        "experiment_name": config.get("experiment_name", ""),
        "output_root": str(root / "render_eval"),
        "view_ids": view_ids,
        "render_resolution": int(config.get("resolution", 512)),
        "background_color": [0.28, 0.28, 0.28],
        "reference_image_path_template": str(sample_root / "{asset_id}" / "render_cond" / "{view_id}_light_AL.png"),
        "primary_front_views": list(config.get("primary_front_views", ["004", "005"])),
        "cases": cases,
    }


def write_summary(render_config: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Phase 2L.5B Mini40 Render-Eval Cases",
        "",
        f"case_count: `{len(render_config['cases'])}`",
        f"output_root: `{render_config['output_root']}`",
        "",
        "| Item ID | Eval Split | Selected Input View | Base GLB | Fine GLB |",
        "|---|---|---|---|---|",
    ]
    for item_id, case in render_config["cases"].items():
        lines.append(
            f"| `{item_id}` | `{case['eval_split']}` | `{case['selected_input_view']}` | "
            f"`{case['base_glb']}` | `{case['finetuned_glb']}` |"
        )
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mini40 render-eval config.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    eval_cases = load_eval_cases(config)
    render_config = make_render_eval_config(config, eval_cases)
    out_path = resolve_project_path(config["output_root"]) / "render_eval" / "render_eval_cases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(render_config, indent=2) + "\n", encoding="utf-8")
    write_summary(render_config, out_path.parent / "render_eval_cases_summary.md")
    print("Phase 2L.5B mini40 render-eval config")
    print(f"  cases: {len(render_config['cases'])}")
    print(f"  output: {out_path}")
    print("PHASE2L5B_MINI40_RENDER_EVAL_CONFIG_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
