#!/usr/bin/env python3
"""Aggregate full80 true-PBR rendered-view evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts import aggregate_datav2_frame_mini40_eval as base
except ImportError:  # pragma: no cover - supports direct script execution.
    import aggregate_datav2_frame_mini40_eval as base  # type: ignore


def mini40_note() -> dict[str, Any]:
    path = base.PROJECT_ROOT / "outputs" / "phase2l" / "datav2_frame_panels" / "mini40_eval_truepbr500" / "summary" / "mini40_eval_summary.json"
    return {
        "mini40_summary_path": str(path),
        "mini40_summary_exists": path.is_file(),
        "note": "Mini40 comparison is optional; interpret full80 metrics directly if the mini40 summary is unavailable.",
    }


def write_report(config: dict[str, Any], report: dict[str, Any]) -> None:
    out_dir = base.resolve_project_path(config["output_root"]) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    report["mini40_comparison_note"] = mini40_note()
    (out_dir / "full80_eval_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.7A Full80 Rendered-View Evaluation Summary",
        "",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        f"case_count: `{report['case_count']}`",
        f"row_count: `{report['row_count']}`",
        f"mini40 summary available: `{report['mini40_comparison_note']['mini40_summary_exists']}`",
        "",
        "## View Groups",
    ]
    for group, summary in report["by_view_group"].items():
        lines.extend(
            [
                f"### {group}",
                f"- view count: `{summary['view_count']}`",
                f"- base->ref MAE: `{summary['base_vs_reference_mean_mae']}`",
                f"- fine->ref MAE: `{summary['fine_vs_reference_mean_mae']}`",
                f"- base->ref SSIM-like: `{summary['base_vs_reference_mean_ssim']}`",
                f"- fine->ref SSIM-like: `{summary['fine_vs_reference_mean_ssim']}`",
                f"- base-vs-fine MAE: `{summary['base_vs_fine_mean_mae']}`",
                f"- fine improves MAE views: `{summary['views_where_fine_improves_mae']}`",
                f"- fine improves SSIM-like views: `{summary['views_where_fine_improves_ssim']}`",
                "",
            ]
        )
    lines.append("## Eval Sets")
    for split, summary in report["by_eval_set"].items():
        lines.append(
            f"- `{split}`: views=`{summary['view_count']}`, base_mae=`{summary['base_vs_reference_mean_mae']}`, fine_mae=`{summary['fine_vs_reference_mean_mae']}`, base_vs_fine_mae=`{summary['base_vs_fine_mean_mae']}`"
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in report["errors"])
    (out_dir / "full80_eval_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate full80 evaluation metrics.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = base.load_json(args.config)
    report = base.aggregate(config)
    write_report(config, report)
    print("Phase 2L.7A full80 eval aggregation")
    print(f"  case_count: {report['case_count']}")
    print(f"  row_count: {report['row_count']}")
    print(f"  errors: {len(report['errors'])}")
    if report["ok"]:
        print("PHASE2L7A_FULL80_EVAL_AGGREGATED_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
