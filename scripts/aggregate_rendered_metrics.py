#!/usr/bin/env python3
"""Aggregate canonical fixed-view metrics across assets and view groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from _evaluation import (
    EvaluationContractError,
    METRIC_NAMES,
    PAIR_NAMES,
    VIEW_GROUP_NAMES,
    asset_label,
    load_cases_manifest,
    load_evaluation_config,
    require_complete_view_rows,
    resolve_project_path,
    summarize_view_groups,
)


OUTPUT_JSON = "rendered_view_metrics_summary.json"
OUTPUT_MD = "rendered_view_metrics_summary.md"


def load_case_metrics(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise EvaluationContractError(f"invalid case metrics JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvaluationContractError(f"case metrics must contain a JSON object: {path}")
    return data


def summarize_case(
    asset_id: str,
    metrics_path: Path,
    evaluation: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_case_metrics(metrics_path)
    if data.get("evaluation_id") != evaluation["evaluation_id"]:
        raise EvaluationContractError(
            f"case metrics use the wrong evaluation_id for {asset_id}: {metrics_path}"
        )
    if data.get("asset_id") != asset_id:
        raise EvaluationContractError(
            f"case metrics asset_id does not match {asset_id}: {metrics_path}"
        )
    if data.get("baseline_label") != cases_manifest["baseline_label"]:
        raise EvaluationContractError(f"baseline label mismatch in {metrics_path}")
    if data.get("candidate_label") != cases_manifest["candidate_label"]:
        raise EvaluationContractError(f"candidate label mismatch in {metrics_path}")
    if data.get("view_groups") != evaluation["view_groups"]:
        raise EvaluationContractError(f"view group contract mismatch in {metrics_path}")
    if data.get("metrics") != evaluation["metrics"]:
        raise EvaluationContractError(f"metric contract mismatch in {metrics_path}")

    raw_views = data.get("views")
    if not isinstance(raw_views, list) or not all(isinstance(row, dict) for row in raw_views):
        raise EvaluationContractError(f"views must be a list of objects in {metrics_path}")
    views: list[dict[str, Any]] = list(raw_views)
    require_complete_view_rows(views, evaluation)
    return (
        {
            "asset_id": asset_id,
            "asset_label": asset_label(cases_manifest, asset_id),
            "metrics_path": str(metrics_path),
            "view_count": len(views),
            "groups": summarize_view_groups(views, evaluation),
        },
        views,
    )


def aggregate(
    evaluation_config: Path,
    cases_config: Path,
    output_root: Path,
) -> dict[str, Any]:
    evaluation_path, evaluation = load_evaluation_config(evaluation_config)
    cases_path, cases_manifest = load_cases_manifest(cases_config)
    output_root = resolve_project_path(str(output_root))
    asset_ids = list(cases_manifest["cases"])
    case_rows: list[dict[str, Any]] = []
    all_views: list[dict[str, Any]] = []

    for asset_id in asset_ids:
        metrics_path = output_root / "metrics" / asset_id / "rendered_view_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing rendered metrics for {asset_id}: {metrics_path}")
        case_summary, views = summarize_case(
            asset_id,
            metrics_path,
            evaluation,
            cases_manifest,
        )
        case_rows.append(case_summary)
        all_views.extend(views)

    require_complete_view_rows(
        all_views,
        evaluation,
        repetitions=len(asset_ids),
    )
    summary = {
        "schema_version": 1,
        "evaluation_id": evaluation["evaluation_id"],
        "evaluation_config": str(evaluation_path),
        "cases_config": str(cases_path),
        "output_root": str(output_root),
        "baseline_label": cases_manifest["baseline_label"],
        "candidate_label": cases_manifest["candidate_label"],
        "asset_ids": asset_ids,
        "case_count": len(case_rows),
        "total_views": len(all_views),
        "view_groups": evaluation["view_groups"],
        "metrics": evaluation["metrics"],
        "groups": summarize_view_groups(all_views, evaluation),
        "cases": case_rows,
    }
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / OUTPUT_JSON).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, summary_dir / OUTPUT_MD)
    return summary


def _metric_text(pair: Mapping[str, Any], metric_name: str) -> str:
    value = pair.get(metric_name)
    if metric_name == "psnr" and value is None and pair.get("psnr_is_infinite") is True:
        return "infinite"
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _comparison_labels(summary: Mapping[str, Any]) -> dict[str, str]:
    baseline_label = str(summary["baseline_label"])
    candidate_label = str(summary["candidate_label"])
    return {
        "baseline_vs_candidate": f"{baseline_label} vs {candidate_label}",
        "baseline_vs_reference": f"{baseline_label} vs Reference",
        "candidate_vs_reference": f"{candidate_label} vs Reference",
        "candidate_minus_baseline": (
            f"{candidate_label} minus {baseline_label} against Reference"
        ),
    }


def _append_group_table(
    lines: list[str],
    group: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    labels = _comparison_labels(summary)
    lines.extend(
        [
            f"Views: {', '.join(group['view_ids'])}",
            "",
            "| Comparison | MAE | RMSE | PSNR | SSIM-like |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair_name in (*PAIR_NAMES, "candidate_minus_baseline"):
        pair = group[pair_name]
        lines.append(
            "| "
            + labels[pair_name]
            + " | "
            + " | ".join(_metric_text(pair, metric) for metric in METRIC_NAMES)
            + " |"
        )


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Fixed-View Metrics Summary",
        "",
        f"- Baseline: {summary['baseline_label']}",
        f"- Candidate: {summary['candidate_label']}",
        f"- Cases: {summary['case_count']}",
        f"- Rendered views: {summary['total_views']}",
        "",
        "## Aggregate View Groups",
    ]
    for group_name in VIEW_GROUP_NAMES:
        lines.extend(["", f"### {group_name}", ""])
        _append_group_table(lines, summary["groups"][group_name], summary)

    lines.extend(["", "## Per Asset"])
    for case in summary["cases"]:
        lines.extend(["", f"### {case['asset_label']}", ""])
        for group_name in VIEW_GROUP_NAMES:
            lines.extend(["", f"#### {group_name}", ""])
            _append_group_table(lines, case["groups"][group_name], summary)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics under the canonical fixed-view protocol."
    )
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = aggregate(
        args.evaluation_config,
        args.cases_config,
        args.output_root,
    )
    output_root = Path(summary["output_root"])
    print("Fixed-View rendered metrics aggregation")
    print(f"  baseline: {summary['baseline_label']}")
    print(f"  candidate: {summary['candidate_label']}")
    print(f"  output_root: {summary['output_root']}")
    print(f"  total_views: {summary['total_views']}")
    print(f"  summary_json: {output_root / 'summary' / OUTPUT_JSON}")
    print(f"  summary_md: {output_root / 'summary' / OUTPUT_MD}")
    print("RENDERED_METRICS_AGGREGATED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
