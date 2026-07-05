#!/usr/bin/env python3
"""Create full80 evaluation cases with explicit selected input views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts import make_datav2_frame_mini40_eval_cases as base
except ImportError:  # pragma: no cover - supports direct script execution.
    import make_datav2_frame_mini40_eval_cases as base  # type: ignore


def write_outputs(config: dict[str, Any], data: dict[str, Any]) -> None:
    root = base.resolve_project_path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "eval_cases.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    summary = {
        "experiment_name": data["experiment_name"],
        "case_count": data["case_count"],
        "primary_eval_case_count": sum(1 for case in data["cases"] if case["eval_split"] in set(config.get("eval_splits", []))),
        "train_sanity_count": sum(1 for case in data["cases"] if case["eval_split"] == "train_sanity"),
        "cases": [
            {
                "item_id": case["item_id"],
                "source_split": case["source_split"],
                "eval_split": case["eval_split"],
                "selected_input_view": case["selected_input_view"],
                "selected_input_view_source": case["selected_input_view_source"],
                "selected_input_image": case["selected_input_image"],
            }
            for case in data["cases"]
        ],
    }
    (root / "eval_cases_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.7A Full80 Eval Cases",
        "",
        f"case_count: `{data['case_count']}`",
        f"primary_eval_case_count: `{summary['primary_eval_case_count']}`",
        f"train_sanity_count: `{summary['train_sanity_count']}`",
        "",
        "| Item ID | Source Split | Eval Split | Selected Input View | Source | Selected Input Image |",
        "|---|---|---|---|---|---|",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| `{case['item_id']}` | `{case['source_split']}` | `{case['eval_split']}` | "
            f"`{case['selected_input_view']}` | `{case['selected_input_view_source']}` | `{case['selected_input_image']}` |"
        )
    (root / "eval_cases_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create full80 eval cases.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = base.load_config(args.config)
    data = base.build_cases(config)
    write_outputs(config, data)
    print("Phase 2L.7A full80 eval cases")
    print(f"  case_count: {data['case_count']}")
    print(f"  eval_cases: {base.resolve_project_path(config['output_root']) / 'eval_cases.json'}")
    if data["case_count"] > 0:
        print("PHASE2L7A_FULL80_EVAL_CASES_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
