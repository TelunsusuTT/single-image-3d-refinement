#!/usr/bin/env python3
"""Create Phase 2L.7A render-eval config from full80 inference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import make_datav2_frame_mini40_render_eval_configs as base
except ImportError:  # pragma: no cover - supports direct script execution.
    import make_datav2_frame_mini40_render_eval_configs as base  # type: ignore


def write_summary(render_config: dict, out_path: Path) -> None:
    lines = [
        "# Phase 2L.7A Full80 Render-Eval Cases",
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
    parser = argparse.ArgumentParser(description="Create full80 render-eval config.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = base.load_config(args.config)
    eval_cases = base.load_eval_cases(config)
    render_config = base.make_render_eval_config(config, eval_cases)
    out_path = base.resolve_project_path(config["output_root"]) / "render_eval" / "render_eval_cases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(render_config, indent=2) + "\n", encoding="utf-8")
    write_summary(render_config, out_path.parent / "render_eval_cases_summary.md")
    print("Phase 2L.7A full80 render-eval config")
    print(f"  cases: {len(render_config['cases'])}")
    print(f"  output: {out_path}")
    print("PHASE2L7A_FULL80_RENDER_EVAL_CONFIG_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
