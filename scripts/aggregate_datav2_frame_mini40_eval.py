#!/usr/bin/env python3
"""Aggregate mini40 true-PBR rendered-view evaluation metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def metric_value(row: dict[str, Any], pair: str, *names: str) -> float | None:
    data = row.get(pair, {})
    for name in names:
        value = data.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_mae = [value for row in rows if (value := metric_value(row, "base_vs_reference", "mae")) is not None]
    fine_mae = [value for row in rows if (value := metric_value(row, "fine_vs_reference", "mae")) is not None]
    base_ssim = [value for row in rows if (value := metric_value(row, "base_vs_reference", "ssim_like", "ssim")) is not None]
    fine_ssim = [value for row in rows if (value := metric_value(row, "fine_vs_reference", "ssim_like", "ssim")) is not None]
    improves_mae = 0
    improves_ssim = 0
    for row in rows:
        b_mae = metric_value(row, "base_vs_reference", "mae")
        f_mae = metric_value(row, "fine_vs_reference", "mae")
        b_ssim = metric_value(row, "base_vs_reference", "ssim_like", "ssim")
        f_ssim = metric_value(row, "fine_vs_reference", "ssim_like", "ssim")
        improves_mae += int(b_mae is not None and f_mae is not None and f_mae < b_mae)
        improves_ssim += int(b_ssim is not None and f_ssim is not None and f_ssim > b_ssim)
    return {
        "view_count": len(rows),
        "base_vs_reference_mean_mae": mean(base_mae),
        "fine_vs_reference_mean_mae": mean(fine_mae),
        "base_vs_reference_mean_ssim": mean(base_ssim),
        "fine_vs_reference_mean_ssim": mean(fine_ssim),
        "views_where_fine_improves_mae": improves_mae,
        "views_where_fine_improves_ssim": improves_ssim,
    }


def view_bucket(view_id: str, selected_input_view: str, primary_front_views: set[str]) -> list[str]:
    groups = ["all_views"]
    if view_id == selected_input_view:
        groups.append("input_view_only")
    if view_id in primary_front_views:
        groups.append("front_views")
    else:
        groups.append("non_front_back_views")
    return groups


def collect_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    root = resolve_project_path(config["output_root"])
    eval_cases = load_json(root / "eval_cases.json")
    primary_default = set(config.get("primary_front_views", ["004", "005"]))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for case in eval_cases.get("cases", []):
        item_id = case["item_id"]
        metrics_path = root / "render_eval" / "metrics" / item_id / "rendered_view_metrics.json"
        if not metrics_path.is_file():
            errors.append(f"missing rendered metrics for {item_id}: {metrics_path}")
            continue
        metrics = load_json(metrics_path)
        selected_input_view = str(case.get("selected_input_view", ""))
        primary_front_views = set(case.get("primary_eval_views") or primary_default)
        for view in metrics.get("views", []):
            view_id = str(view.get("view_id", ""))
            rows.append(
                {
                    **view,
                    "item_id": item_id,
                    "eval_split": case.get("eval_split", ""),
                    "source_split": case.get("source_split", ""),
                    "selected_input_view": selected_input_view,
                    "view_groups": view_bucket(view_id, selected_input_view, primary_front_views),
                }
            )
    return rows, errors


def aggregate(config: dict[str, Any]) -> dict[str, Any]:
    rows, errors = collect_rows(config)
    groups = ["all_views", "input_view_only", "front_views", "non_front_back_views"]
    by_group = {group: summarize_rows([row for row in rows if group in row["view_groups"]]) for group in groups}
    split_names = sorted({row["eval_split"] for row in rows})
    by_split = {split: summarize_rows([row for row in rows if row["eval_split"] == split]) for split in split_names}
    report = {
        "experiment_name": config.get("experiment_name", ""),
        "output_root": str(resolve_project_path(config["output_root"])),
        "row_count": len(rows),
        "case_count": len({row["item_id"] for row in rows}),
        "by_view_group": by_group,
        "by_split": by_split,
        "errors": errors,
        "ok": not errors and bool(rows),
    }
    return report


def write_report(config: dict[str, Any], report: dict[str, Any]) -> None:
    out_dir = resolve_project_path(config["output_root"]) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mini40_eval_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.5A Mini40 Evaluation Summary",
        "",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        f"case_count: `{report['case_count']}`",
        f"row_count: `{report['row_count']}`",
        "",
        "## View Groups",
    ]
    for group, summary in report["by_view_group"].items():
        lines.extend(
            [
                f"### {group}",
                f"- view count: `{summary['view_count']}`",
                f"- base MAE: `{summary['base_vs_reference_mean_mae']}`",
                f"- fine MAE: `{summary['fine_vs_reference_mean_mae']}`",
                f"- base SSIM: `{summary['base_vs_reference_mean_ssim']}`",
                f"- fine SSIM: `{summary['fine_vs_reference_mean_ssim']}`",
                f"- fine improves MAE views: `{summary['views_where_fine_improves_mae']}`",
                f"- fine improves SSIM views: `{summary['views_where_fine_improves_ssim']}`",
                "",
            ]
        )
    lines.append("## Splits")
    for split, summary in report["by_split"].items():
        lines.append(
            f"- `{split}`: views=`{summary['view_count']}`, base_mae=`{summary['base_vs_reference_mean_mae']}`, fine_mae=`{summary['fine_vs_reference_mean_mae']}`"
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in report["errors"])
    (out_dir / "mini40_eval_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate mini40 evaluation metrics.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    report = aggregate(config)
    write_report(config, report)
    print("Phase 2L.5A mini40 eval aggregation")
    print(f"  case_count: {report['case_count']}")
    print(f"  row_count: {report['row_count']}")
    print(f"  errors: {len(report['errors'])}")
    if report["ok"]:
        print("PHASE2L5A_MINI40_EVAL_AGGREGATED_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
