#!/usr/bin/env python3
"""Compare full80 rendered base/fine views against render_cond references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import compare_datav2_frame_mini40_rendered_views as base
except ImportError:  # pragma: no cover - supports direct script execution.
    import compare_datav2_frame_mini40_rendered_views as base  # type: ignore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full80 rendered views against references.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_config = json.loads(args.config.read_text(encoding="utf-8"))
    render_root = base.resolve_project_path(eval_config["output_root"]) / "render_eval"
    render_config = base.load_json(render_root / "render_eval_cases.json")
    results = base.compare_all(render_config, render_root)
    print("Phase 2L.7A full80 rendered-view comparison")
    print(f"  cases: {len(results)}")
    print(f"  render_root: {render_root}")
    print("PHASE2L7A_FULL80_RENDERED_VIEW_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
