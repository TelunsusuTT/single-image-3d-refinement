#!/usr/bin/env python3
"""Aggregate Phase 2K.3 rendered-view metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def mean_present(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.fmean(numbers) if numbers else None


def collect_view_values(case_metrics: dict[str, Any], pair_name: str, metric_name: str) -> list[float]:
    values = []
    for row in case_metrics.get("views", []):
        value = row.get(pair_name, {}).get(metric_name)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize_case(asset_id: str, metrics_path: Path) -> dict[str, Any]:
    data = load_json(metrics_path)
    views = data.get("views", [])
    return {
        "asset_id": asset_id,
        "metrics_path": str(metrics_path),
        "view_count": len(views),
        "base_vs_fine_mean_mae": mean_present(
            collect_view_values(data, "base_vs_fine", "mae")
        ),
        "base_vs_fine_mean_rmse": mean_present(
            collect_view_values(data, "base_vs_fine", "rmse")
        ),
        "base_vs_fine_mean_ssim_like": mean_present(
            collect_view_values(data, "base_vs_fine", "ssim_like")
        ),
        "base_vs_reference_mean_mae": mean_present(
            collect_view_values(data, "base_vs_reference", "mae")
        ),
        "base_vs_reference_mean_rmse": mean_present(
            collect_view_values(data, "base_vs_reference", "rmse")
        ),
        "base_vs_reference_mean_ssim_like": mean_present(
            collect_view_values(data, "base_vs_reference", "ssim_like")
        ),
        "fine_vs_reference_mean_mae": mean_present(
            collect_view_values(data, "fine_vs_reference", "mae")
        ),
        "fine_vs_reference_mean_rmse": mean_present(
            collect_view_values(data, "fine_vs_reference", "rmse")
        ),
        "fine_vs_reference_mean_ssim_like": mean_present(
            collect_view_values(data, "fine_vs_reference", "ssim_like")
        ),
        "views_fine_improves_mae": sum(
            1
            for row in views
            if numeric(row.get("improvement", {}).get("fine_minus_base_mae")) is not None
            and float(row["improvement"]["fine_minus_base_mae"]) < 0
        ),
        "views_fine_improves_ssim": sum(
            1
            for row in views
            if numeric(row.get("improvement", {}).get("fine_minus_base_ssim_like")) is not None
            and float(row["improvement"]["fine_minus_base_ssim_like"]) > 0
        ),
    }


def sum_key(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0)) for row in rows))


def interpret(summary: dict[str, Any]) -> str:
    means = summary["aggregate_means"]
    total_views = summary["total_views"]
    mae_improve = summary["views_fine_improves_mae"]
    ssim_improve = summary["views_fine_improves_ssim"]
    base_fine_mae = means.get("base_vs_fine_mean_mae")
    if isinstance(base_fine_mae, (int, float)) and base_fine_mae > 80:
        return "possible instability: base-vs-fine rendered changes are very large"
    if total_views and mae_improve > total_views / 2 and ssim_improve > total_views / 2:
        return "possible improvement: fine-tuned renders improve MAE and SSIM-like score in most views"
    if isinstance(base_fine_mae, (int, float)) and base_fine_mae < 25:
        return "stable no-collapse: base-vs-fine rendered changes are moderate"
    return "inconclusive: rendered-view metrics are mixed"


def aggregate(cases_config: Path, output_root: Path) -> dict[str, Any]:
    config = load_json(cases_config)
    asset_ids = list(config.get("cases", {}).keys())
    rows = []
    for asset_id in asset_ids:
        metrics_path = output_root / "metrics" / asset_id / "rendered_view_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing rendered metrics for {asset_id}: {metrics_path}")
        rows.append(summarize_case(asset_id, metrics_path))

    aggregate_means = {
        "base_vs_fine_mean_mae": mean_present([row["base_vs_fine_mean_mae"] for row in rows]),
        "base_vs_fine_mean_rmse": mean_present([row["base_vs_fine_mean_rmse"] for row in rows]),
        "base_vs_fine_mean_ssim_like": mean_present([row["base_vs_fine_mean_ssim_like"] for row in rows]),
        "base_vs_reference_mean_mae": mean_present([row["base_vs_reference_mean_mae"] for row in rows]),
        "base_vs_reference_mean_rmse": mean_present([row["base_vs_reference_mean_rmse"] for row in rows]),
        "base_vs_reference_mean_ssim_like": mean_present([row["base_vs_reference_mean_ssim_like"] for row in rows]),
        "fine_vs_reference_mean_mae": mean_present([row["fine_vs_reference_mean_mae"] for row in rows]),
        "fine_vs_reference_mean_rmse": mean_present([row["fine_vs_reference_mean_rmse"] for row in rows]),
        "fine_vs_reference_mean_ssim_like": mean_present([row["fine_vs_reference_mean_ssim_like"] for row in rows]),
    }
    summary = {
        "cases_config": str(cases_config.resolve()),
        "output_root": str(output_root.resolve()),
        "asset_ids": asset_ids,
        "expected_asset_ids": REQUIRED_ASSET_IDS,
        "case_count": len(rows),
        "total_views": sum_key(rows, "view_count"),
        "views_fine_improves_mae": sum_key(rows, "views_fine_improves_mae"),
        "views_fine_improves_ssim": sum_key(rows, "views_fine_improves_ssim"),
        "aggregate_means": aggregate_means,
        "cases": rows,
    }
    summary["interpretation"] = interpret(summary)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_json = summary_dir / "rendered_view_metrics_summary.json"
    out_md = summary_dir / "rendered_view_metrics_summary.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_md)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 2K.3 Rendered-View Metrics Summary",
        "",
        f"case_count: `{summary['case_count']}`",
        f"total_views: `{summary['total_views']}`",
        f"views fine improves MAE: `{summary['views_fine_improves_mae']}`",
        f"views fine improves SSIM-like: `{summary['views_fine_improves_ssim']}`",
        f"interpretation: `{summary['interpretation']}`",
        "",
        "## Aggregate Means",
    ]
    for key, value in summary["aggregate_means"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Per Asset"])
    for row in summary["cases"]:
        lines.extend(
            [
                f"### {row['asset_id']}",
                f"- view count: `{row['view_count']}`",
                f"- base vs fine MAE: `{row['base_vs_fine_mean_mae']}`",
                f"- base vs reference MAE: `{row['base_vs_reference_mean_mae']}`",
                f"- fine vs reference MAE: `{row['fine_vs_reference_mean_mae']}`",
                f"- views fine improves MAE: `{row['views_fine_improves_mae']}`",
                f"- views fine improves SSIM-like: `{row['views_fine_improves_ssim']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2K.3 rendered metrics.")
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases_config = resolve_project_path(str(args.cases_config))
    output_root = resolve_project_path(str(args.output_root))
    summary = aggregate(cases_config, output_root)
    print("Phase 2K.3 rendered metrics aggregation")
    print(f"  output_root: {summary['output_root']}")
    print(f"  total_views: {summary['total_views']}")
    print(f"  summary_json: {output_root / 'summary' / 'rendered_view_metrics_summary.json'}")
    print(f"  summary_md: {output_root / 'summary' / 'rendered_view_metrics_summary.md'}")
    print("PHASE2K3_RENDERED_METRICS_AGGREGATED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
