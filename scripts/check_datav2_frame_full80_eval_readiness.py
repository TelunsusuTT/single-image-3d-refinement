#!/usr/bin/env python3
"""Static readiness checks for full80 true-PBR corrected-input evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts import check_datav2_frame_mini40_eval_readiness as base
except ImportError:  # pragma: no cover - supports direct script execution.
    import check_datav2_frame_mini40_eval_readiness as base  # type: ignore


def add_full80_count_checks(report: dict[str, Any], config: dict[str, Any]) -> None:
    cases = report.get("cases", [])
    primary_splits = set(config.get("eval_splits", ["val", "test"]))
    primary_count = sum(1 for case in cases if case.get("eval_split") in primary_splits)
    train_sanity_count = sum(1 for case in cases if case.get("eval_split") == "train_sanity")
    report["primary_eval_case_count"] = primary_count
    report["train_sanity_count"] = train_sanity_count
    expected_primary = config.get("expected_primary_eval_case_count")
    expected_total = config.get("expected_total_case_count")
    if expected_primary is not None and primary_count != int(expected_primary):
        report["errors"].append(f"primary eval case count {primary_count} != expected {expected_primary}")
    if expected_total is not None and report.get("case_count", 0) != int(expected_total):
        report["errors"].append(f"total eval case count {report.get('case_count', 0)} != expected {expected_total}")
    report["ok"] = not report["errors"]


def readiness_report(config_path: Path) -> dict[str, Any]:
    report = base.readiness_report(config_path)
    config = base.load_json(config_path)
    add_full80_count_checks(report, config)
    return report


def write_report(report: dict[str, Any]) -> None:
    root = Path(report["output_root"])
    out_json = root / "readiness_summary.json"
    out_md = root / "readiness_summary.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.7A Full80 Eval Readiness",
        "",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        f"case_count: `{report['case_count']}`",
        f"primary_eval_case_count: `{report.get('primary_eval_case_count', 0)}`",
        f"train_sanity_count: `{report.get('train_sanity_count', 0)}`",
        f"checkpoint: `{report['checkpoint']['path']}`",
        f"override CSV: `{report['override_csv']['path']}`",
        "",
        "| Item ID | Eval Split | Selected Input View |",
        "|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(f"| `{case['item_id']}` | `{case['eval_split']}` | `{case['selected_input_view']}` |")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in report["warnings"])
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in report["errors"])
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check full80 eval readiness.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = readiness_report(args.config)
    write_report(report)
    print("Phase 2L.7A full80 eval readiness")
    print(f"  output_root: {report['output_root']}")
    print(f"  case_count: {report['case_count']}")
    print(f"  primary_eval_case_count: {report.get('primary_eval_case_count', 0)}")
    print(f"  train_sanity_count: {report.get('train_sanity_count', 0)}")
    print(f"  warnings: {len(report['warnings'])}")
    print(f"  errors: {len(report['errors'])}")
    if report["ok"]:
        print("PHASE2L7A_FULL80_EVAL_READINESS_OK")
        return 0
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
