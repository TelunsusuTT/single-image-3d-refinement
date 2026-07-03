#!/usr/bin/env python3
"""Aggregate Phase 2K.2 multi-case texture comparison metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSET_IDS = ["B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def resolve_project_path(path_text: str, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def stat_mean(stats: dict[str, Any], name: str) -> float | None:
    item = stats.get(name, {})
    return number(item.get("grayscale_mean")) if isinstance(item, dict) else None


def mesh_size(mesh_inventory: dict[str, Any], name: str) -> int | None:
    item = mesh_inventory.get(name, {})
    value = item.get("size_bytes") if isinstance(item, dict) else None
    return int(value) if isinstance(value, int) else None


def extract_case_metrics(asset_id: str, metrics_path: Path) -> dict[str, Any]:
    data = load_json(metrics_path)
    pair = data.get("pair_metrics", {})
    stats = data.get("map_stats", {})
    interpretation = data.get("interpretation", {})
    mesh_inventory = data.get("mesh_inventory", {})

    base_metallic = stat_mean(stats, "base_metallic")
    finetuned_metallic = stat_mean(stats, "finetuned_metallic")
    base_roughness = stat_mean(stats, "base_roughness")
    finetuned_roughness = stat_mean(stats, "finetuned_roughness")

    return {
        "asset_id": asset_id,
        "metrics_path": str(metrics_path),
        "albedo_mean_abs_diff": number(pair.get("albedo", {}).get("mean_abs_diff")),
        "metallic_mean_abs_diff": number(pair.get("metallic", {}).get("mean_abs_diff")),
        "roughness_mean_abs_diff": number(pair.get("roughness", {}).get("mean_abs_diff")),
        "metallic_mean_shift": (
            finetuned_metallic - base_metallic
            if base_metallic is not None and finetuned_metallic is not None
            else number(interpretation.get("metallic_mean_shift"))
        ),
        "roughness_mean_shift": (
            finetuned_roughness - base_roughness
            if base_roughness is not None and finetuned_roughness is not None
            else number(interpretation.get("roughness_mean_shift"))
        ),
        "albedo_differs_strongly": bool(interpretation.get("albedo_differs_strongly", False)),
        "metallic_differs_strongly": bool(interpretation.get("metallic_differs_strongly", False)),
        "roughness_differs_strongly": bool(interpretation.get("roughness_differs_strongly", False)),
        "base_glb_size_bytes": mesh_size(mesh_inventory, "base_glb"),
        "finetuned_glb_size_bytes": mesh_size(mesh_inventory, "finetuned_glb"),
    }


def mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.fmean(values) if values else None


def write_markdown(summary: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2K.2 Multi-Case Metrics Summary",
        "",
        f"output_root: `{summary['output_root']}`",
        f"case_count: `{summary['case_count']}`",
        "",
        "## Aggregate Means",
    ]
    for key, value in summary["aggregate_means"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Per Case"])
    for row in summary["cases"]:
        lines.extend(
            [
                f"### {row['asset_id']}",
                f"- albedo MAE: `{row['albedo_mean_abs_diff']}`",
                f"- metallic MAE: `{row['metallic_mean_abs_diff']}`",
                f"- roughness MAE: `{row['roughness_mean_abs_diff']}`",
                f"- metallic mean shift: `{row['metallic_mean_shift']}`",
                f"- roughness mean shift: `{row['roughness_mean_shift']}`",
                f"- albedo differs strongly: `{row['albedo_differs_strongly']}`",
                f"- metallic differs strongly: `{row['metallic_differs_strongly']}`",
                f"- roughness differs strongly: `{row['roughness_differs_strongly']}`",
                f"- base GLB bytes: `{row['base_glb_size_bytes']}`",
                f"- fine-tuned GLB bytes: `{row['finetuned_glb_size_bytes']}`",
                "",
            ]
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def aggregate(cases_config: Path, output_root: Path) -> dict[str, Any]:
    config = load_json(cases_config)
    asset_ids = list(config.get("selected_asset_ids", REQUIRED_ASSET_IDS))
    rows = []
    for asset_id in asset_ids:
        metrics_path = output_root / "compare" / asset_id / "base_vs_finetuned_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics JSON for {asset_id}: {metrics_path}")
        rows.append(extract_case_metrics(asset_id, metrics_path))

    summary = {
        "cases_config": str(cases_config.resolve()),
        "output_root": str(output_root.resolve()),
        "asset_ids": asset_ids,
        "case_count": len(rows),
        "cases": rows,
        "aggregate_means": {
            "albedo_mean_abs_diff": mean_present(rows, "albedo_mean_abs_diff"),
            "metallic_mean_abs_diff": mean_present(rows, "metallic_mean_abs_diff"),
            "roughness_mean_abs_diff": mean_present(rows, "roughness_mean_abs_diff"),
            "metallic_mean_shift": mean_present(rows, "metallic_mean_shift"),
            "roughness_mean_shift": mean_present(rows, "roughness_mean_shift"),
        },
    }
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_json = summary_dir / "multicase_metrics_summary.json"
    out_md = summary_dir / "multicase_metrics_summary.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_md)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 2K.2 multicase comparison metrics."
    )
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolve_project_path(str(args.output_root))
    cases_config = resolve_project_path(str(args.cases_config))
    summary = aggregate(cases_config, output_root)
    print("Phase 2K.2 multi-case metrics aggregation")
    print(f"  output_root: {summary['output_root']}")
    print(f"  case_count: {summary['case_count']}")
    print(f"  summary_json: {output_root / 'summary' / 'multicase_metrics_summary.json'}")
    print(f"  summary_md: {output_root / 'summary' / 'multicase_metrics_summary.md'}")
    print("PHASE2K2_MULTICASE_METRICS_AGGREGATED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
