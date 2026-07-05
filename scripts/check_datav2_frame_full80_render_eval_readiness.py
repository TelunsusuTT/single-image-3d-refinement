#!/usr/bin/env python3
"""Static readiness checks for full80 rendered-view evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import check_datav2_frame_mini40_render_eval_readiness as base
except ImportError:  # pragma: no cover - supports direct script execution.
    import check_datav2_frame_mini40_render_eval_readiness as base  # type: ignore


def write_report(report: dict) -> None:
    root = Path(report["render_eval_root"])
    out_json = root / "render_eval_readiness_summary.json"
    out_md = root / "render_eval_readiness_summary.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.7A Full80 Render-Eval Readiness",
        "",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        f"case_count: `{report['case_count']}`",
        f"blender_bin: `{report['blender_bin']['path']}`",
        "",
        "| Item ID | Eval Split | Selected Input View |",
        "|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(f"| `{case['item_id']}` | `{case['eval_split']}` | `{case['selected_input_view']}` |")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in report["errors"])
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check full80 render-eval readiness.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = base.readiness_report(args.config)
    write_report(report)
    print("Phase 2L.7A full80 render-eval readiness")
    print(f"  case_count: {report['case_count']}")
    print(f"  errors: {len(report['errors'])}")
    if report["ok"]:
        print("PHASE2L7A_FULL80_RENDER_EVAL_READINESS_OK")
        return 0
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
