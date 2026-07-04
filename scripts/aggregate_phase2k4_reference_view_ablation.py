#!/usr/bin/env python3
"""Aggregate Phase 2K.4 reference-view ablation rendered metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FRONT_VIEWS = ["004", "005"]


def resolve_project_path(path_text: str, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_present(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.fmean(numbers) if numbers else None


def summary_path(root: Path) -> Path:
    return root / "summary" / "rendered_view_metrics_summary.json"


def extract_summary(label: str, root: Path) -> dict[str, Any]:
    path = summary_path(root)
    if not path.is_file():
        return {"label": label, "root": str(root), "exists": False}
    data = load_json(path)
    means = data.get("aggregate_means", {})
    cases = data.get("cases", [])
    return {
        "label": label,
        "root": str(root),
        "exists": True,
        "summary_path": str(path),
        "aggregate": {
            "base_vs_reference_mean_mae": means.get("base_vs_reference_mean_mae"),
            "fine_vs_reference_mean_mae": means.get("fine_vs_reference_mean_mae"),
            "base_vs_reference_mean_ssim_like": means.get("base_vs_reference_mean_ssim_like"),
            "fine_vs_reference_mean_ssim_like": means.get("fine_vs_reference_mean_ssim_like"),
            "views_fine_improves_mae": data.get("views_fine_improves_mae"),
            "views_fine_improves_ssim": data.get("views_fine_improves_ssim"),
        },
        "per_asset": cases,
    }


def front_metrics_for_case(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in metrics.get("views", []) if row.get("view_id") in PRIMARY_FRONT_VIEWS]
    return {
        "view_count": len(rows),
        "base_front_mae_mean": mean_present(
            [row.get("base_vs_reference", {}).get("mae") for row in rows]
        ),
        "fine_front_mae_mean": mean_present(
            [row.get("fine_vs_reference", {}).get("mae") for row in rows]
        ),
        "base_front_ssim_mean": mean_present(
            [row.get("base_vs_reference", {}).get("ssim_like") for row in rows]
        ),
        "fine_front_ssim_mean": mean_present(
            [row.get("fine_vs_reference", {}).get("ssim_like") for row in rows]
        ),
    }


def extract_front_metrics(root: Path, asset_ids: list[str]) -> dict[str, Any]:
    per_asset = {}
    for asset_id in asset_ids:
        path = root / "metrics" / asset_id / "rendered_view_metrics.json"
        if path.is_file():
            per_asset[asset_id] = {
                "exists": True,
                "metrics_path": str(path),
                **front_metrics_for_case(load_json(path)),
            }
        else:
            per_asset[asset_id] = {"exists": False, "metrics_path": str(path)}
    existing = [item for item in per_asset.values() if item.get("exists")]
    aggregate = {
        "asset_count": len(existing),
        "base_front_mae_mean": mean_present([item.get("base_front_mae_mean") for item in existing]),
        "fine_front_mae_mean": mean_present([item.get("fine_front_mae_mean") for item in existing]),
        "base_front_ssim_mean": mean_present([item.get("base_front_ssim_mean") for item in existing]),
        "fine_front_ssim_mean": mean_present([item.get("fine_front_ssim_mean") for item in existing]),
    }
    return {"aggregate": aggregate, "per_asset": per_asset}


def interpretation(conditions: dict[str, Any]) -> str:
    baseline = conditions.get("previous_baseline", {})
    input_004 = conditions.get("input_004", {})
    input_005 = conditions.get("input_005", {})
    if not baseline.get("exists"):
        return "baseline unavailable; compare input_004 and input_005 manually"
    baseline_mae = baseline.get("front_metrics", {}).get("aggregate", {}).get("base_front_mae_mean")
    candidates = []
    for label, condition in (("input_004", input_004), ("input_005", input_005)):
        mae = condition.get("front_metrics", {}).get("aggregate", {}).get("base_front_mae_mean")
        if isinstance(mae, (int, float)) and isinstance(baseline_mae, (int, float)):
            candidates.append((label, mae - baseline_mae))
    improved = [label for label, delta in candidates if delta < 0]
    if improved:
        return f"front input may help base rendered match on: {', '.join(improved)}"
    if candidates:
        return "front input did not improve base front-view MAE versus baseline"
    return "insufficient front-view metrics for interpretation"


def aggregate(config: dict[str, Any]) -> dict[str, Any]:
    output_root = resolve_project_path(config["output_root"])
    baseline_root = resolve_project_path(config["previous_baseline_render_eval_root"])
    asset_ids = list(config["asset_ids"])
    conditions = {
        "previous_baseline": extract_summary("previous_baseline", baseline_root),
        "input_004": extract_summary("input_004", output_root / "rendered_eval" / "input_004"),
        "input_005": extract_summary("input_005", output_root / "rendered_eval" / "input_005"),
    }
    conditions["previous_baseline"]["front_metrics"] = extract_front_metrics(baseline_root, asset_ids)
    conditions["input_004"]["front_metrics"] = extract_front_metrics(
        output_root / "rendered_eval" / "input_004",
        asset_ids,
    )
    conditions["input_005"]["front_metrics"] = extract_front_metrics(
        output_root / "rendered_eval" / "input_005",
        asset_ids,
    )
    report = {
        "output_root": str(output_root),
        "asset_ids": asset_ids,
        "input_views": list(config["input_views"]),
        "primary_front_views": PRIMARY_FRONT_VIEWS,
        "conditions": conditions,
    }
    report["interpretation"] = interpretation(conditions)
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_json = summary_dir / "reference_view_ablation_summary.json"
    out_md = summary_dir / "reference_view_ablation_summary.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_md)
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 2K.4 Reference-View Ablation Summary",
        "",
        f"interpretation: `{report['interpretation']}`",
        "",
        "## Conditions",
    ]
    for label, condition in report["conditions"].items():
        lines.extend(
            [
                f"### {label}",
                f"- exists: `{condition.get('exists')}`",
                f"- root: `{condition.get('root')}`",
            ]
        )
        aggregate_metrics = condition.get("aggregate", {})
        for key, value in aggregate_metrics.items():
            lines.append(f"- `{key}`: `{value}`")
        front = condition.get("front_metrics", {}).get("aggregate", {})
        if front:
            lines.append("- front metrics:")
            for key, value in front.items():
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2K.4 ablation metrics.")
    parser.add_argument("--cases-config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.cases_config)
    report = aggregate(config)
    output_root = resolve_project_path(config["output_root"])
    print("Phase 2K.4 reference-view ablation aggregation")
    print(f"  output_root: {output_root}")
    print(f"  summary_json: {output_root / 'summary' / 'reference_view_ablation_summary.json'}")
    print(f"  summary_md: {output_root / 'summary' / 'reference_view_ablation_summary.md'}")
    print("PHASE2K4_REFERENCE_VIEW_AGGREGATED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
