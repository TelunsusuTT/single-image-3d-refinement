#!/usr/bin/env python3
"""Aggregate mini40 true-PBR rendered-view evaluation metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONT_VIEWS = {"004", "005"}
DEFAULT_NON_FRONT_VIEWS = {"000", "001", "002", "003"}


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
    def collect(pair: str, *names: str) -> list[float]:
        return [value for row in rows if (value := metric_value(row, pair, *names)) is not None]

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
        "base_vs_fine_mean_mae": mean(collect("base_vs_fine", "mae")),
        "base_vs_fine_mean_rmse": mean(collect("base_vs_fine", "rmse")),
        "base_vs_fine_mean_ssim": mean(collect("base_vs_fine", "ssim_like", "ssim")),
        "base_vs_reference_mean_mae": mean(collect("base_vs_reference", "mae")),
        "fine_vs_reference_mean_mae": mean(collect("fine_vs_reference", "mae")),
        "base_vs_reference_mean_rmse": mean(collect("base_vs_reference", "rmse")),
        "fine_vs_reference_mean_rmse": mean(collect("fine_vs_reference", "rmse")),
        "base_vs_reference_mean_ssim": mean(collect("base_vs_reference", "ssim_like", "ssim")),
        "fine_vs_reference_mean_ssim": mean(collect("fine_vs_reference", "ssim_like", "ssim")),
        "base_vs_reference_mean_histogram_l1": mean(collect("base_vs_reference", "histogram_l1")),
        "fine_vs_reference_mean_histogram_l1": mean(collect("fine_vs_reference", "histogram_l1")),
        "base_vs_reference_mean_edge_difference": mean(collect("base_vs_reference", "edge_difference")),
        "fine_vs_reference_mean_edge_difference": mean(collect("fine_vs_reference", "edge_difference")),
        "views_where_fine_improves_mae": improves_mae,
        "views_where_fine_improves_ssim": improves_ssim,
    }


def view_groups(view_id: str, selected_input_view: str, front_views: set[str]) -> list[str]:
    groups = ["all_views"]
    if view_id == "005":
        groups.append("input_view_005")
    if view_id == selected_input_view:
        groups.append("selected_input_view")
    if view_id in front_views:
        groups.append("front_views_004_005")
    if view_id in DEFAULT_NON_FRONT_VIEWS:
        groups.append("non_front_views_000_003")
    return groups


def metric_path_for(root: Path, case: dict[str, Any]) -> Path:
    item_id = case["item_id"]
    eval_split = case.get("eval_split", "")
    new_path = root / "render_eval" / "metrics" / eval_split / f"{item_id}_rendered_view_metrics.json"
    if new_path.is_file():
        return new_path
    return root / "render_eval" / "metrics" / item_id / "rendered_view_metrics.json"


def collect_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    root = resolve_project_path(config["output_root"])
    eval_cases = load_json(root / "eval_cases.json")
    default_front = set(config.get("primary_front_views", sorted(DEFAULT_FRONT_VIEWS)))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for case in eval_cases.get("cases", []):
        item_id = case["item_id"]
        metrics_path = metric_path_for(root, case)
        if not metrics_path.is_file():
            errors.append(f"missing rendered metrics for {item_id}: {metrics_path}")
            continue
        metrics = load_json(metrics_path)
        selected = str(case.get("selected_input_view") or metrics.get("selected_input_view", ""))
        front_views = set(case.get("primary_eval_views") or metrics.get("primary_front_views") or default_front)
        for view in metrics.get("views", []):
            view_id = str(view.get("view_id", ""))
            rows.append(
                {
                    **view,
                    "item_id": item_id,
                    "eval_split": case.get("eval_split", metrics.get("eval_split", "")),
                    "source_split": case.get("source_split", metrics.get("source_split", "")),
                    "selected_input_view": selected,
                    "view_groups": view_groups(view_id, selected, front_views),
                }
            )
    return rows, errors


def aggregate(config: dict[str, Any]) -> dict[str, Any]:
    rows, errors = collect_rows(config)
    groups = ["all_views", "input_view_005", "selected_input_view", "front_views_004_005", "non_front_views_000_003"]
    by_view_group = {group: summarize_rows([row for row in rows if group in row["view_groups"]]) for group in groups}
    by_split = {split: summarize_rows([row for row in rows if row["eval_split"] == split]) for split in sorted({row["eval_split"] for row in rows})}
    val_test_rows = [row for row in rows if row["eval_split"] in {"val", "test"}]
    by_eval_set = {
        "val": summarize_rows([row for row in rows if row["eval_split"] == "val"]),
        "test": summarize_rows([row for row in rows if row["eval_split"] == "test"]),
        "val_test": summarize_rows(val_test_rows),
        "train_sanity": summarize_rows([row for row in rows if row["eval_split"] == "train_sanity"]),
    }
    report = {
        "experiment_name": config.get("experiment_name", ""),
        "output_root": str(resolve_project_path(config["output_root"])),
        "row_count": len(rows),
        "case_count": len({row["item_id"] for row in rows}),
        "by_view_group": by_view_group,
        "by_split": by_split,
        "by_eval_set": by_eval_set,
        "errors": errors,
        "ok": not errors and bool(rows),
    }
    return report


def write_report(config: dict[str, Any], report: dict[str, Any]) -> None:
    out_dir = resolve_project_path(config["output_root"]) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mini40_eval_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.5B Mini40 Rendered-View Evaluation Summary",
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
    print("Phase 2L.5B mini40 eval aggregation")
    print(f"  case_count: {report['case_count']}")
    print(f"  row_count: {report['row_count']}")
    print(f"  errors: {len(report['errors'])}")
    if report["ok"]:
        print("PHASE2L5B_MINI40_EVAL_AGGREGATED_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
