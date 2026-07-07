#!/usr/bin/env python3
"""Aggregate rendered-view metrics for Phase 2M.3C LoRA multi-scale pilot."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_rendered"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_eval"
DEFAULT_VARIANTS = ["base", "lora_scale050", "lora_scale075", "lora_scale100"]
DEFAULT_FRONT_VIEWS = {"004", "005"}
DEFAULT_NON_FRONT_VIEWS = {"000", "001", "002", "003"}

scripts_root = PROJECT_ROOT / "scripts"
if str(scripts_root) not in sys.path:
    sys.path.insert(0, str(scripts_root))
try:
    from compare_phase2k3_rendered_views import pair_metrics, rgb255
except ImportError:  # pragma: no cover
    from scripts.compare_phase2k3_rendered_views import pair_metrics, rgb255  # type: ignore


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def view_groups(view_id: str, selected_input_view: str, front_views: set[str]) -> list[str]:
    groups = ["all_views"]
    if view_id == "005":
        groups.append("input_view_005")
    if view_id == selected_input_view:
        groups.append("selected_input_view")
    if view_id in front_views:
        groups.extend(["front_views_004_005", "front_views"])
    if view_id in DEFAULT_NON_FRONT_VIEWS:
        groups.extend(["non_front_views_000_003", "non_front_back_views"])
    return groups


def metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(name: str) -> list[float]:
        return [value for row in rows if (value := metric(row, name)) is not None]

    improves_mae = sum(1 for row in rows if metric(row, "variant_minus_base_mae") is not None and float(row["variant_minus_base_mae"]) < 0)
    improves_ssim = sum(1 for row in rows if metric(row, "variant_minus_base_ssim_like") is not None and float(row["variant_minus_base_ssim_like"]) > 0)
    return {
        "view_count": len(rows),
        "base_mean_mae": mean(collect("base_mae")),
        "variant_mean_mae": mean(collect("variant_mae")),
        "base_mean_rmse": mean(collect("base_rmse")),
        "variant_mean_rmse": mean(collect("variant_rmse")),
        "base_mean_ssim_like": mean(collect("base_ssim_like")),
        "variant_mean_ssim_like": mean(collect("variant_ssim_like")),
        "base_vs_variant_mean_mae": mean(collect("base_vs_variant_mae")),
        "variant_minus_base_mean_mae": mean(collect("variant_minus_base_mae")),
        "variant_minus_base_mean_rmse": mean(collect("variant_minus_base_rmse")),
        "variant_minus_base_mean_ssim_like": mean(collect("variant_minus_base_ssim_like")),
        "views_where_variant_improves_mae": improves_mae,
        "views_where_variant_improves_ssim_like": improves_ssim,
    }


def image_pair_metrics(reference_path: Path, candidate_path: Path, background_rgb: tuple[int, int, int]) -> tuple[dict[str, Any], bool]:
    from PIL import Image

    reference = Image.open(reference_path).convert("RGB")
    candidate_raw = Image.open(candidate_path).convert("RGB")
    resized = candidate_raw.size != reference.size
    candidate = candidate_raw.resize(reference.size, Image.Resampling.BICUBIC) if resized else candidate_raw
    return pair_metrics(reference, candidate, background_rgb), resized


def candidate_pair_metrics(a_path: Path, b_path: Path, target_size: tuple[int, int], background_rgb: tuple[int, int, int]) -> dict[str, Any]:
    from PIL import Image

    image_a = Image.open(a_path).convert("RGB")
    image_b = Image.open(b_path).convert("RGB")
    if image_a.size != target_size:
        image_a = image_a.resize(target_size, Image.Resampling.BICUBIC)
    if image_b.size != target_size:
        image_b = image_b.resize(target_size, Image.Resampling.BICUBIC)
    return pair_metrics(image_a, image_b, background_rgb)


def collect_metric_rows(render_config: dict[str, Any], render_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    from PIL import Image

    background_rgb = rgb255(list(render_config.get("background_color", [0.28, 0.28, 0.28])))
    variants = list(render_config.get("variants", DEFAULT_VARIANTS))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for case_id, case in render_config.get("cases", {}).items():
        split = str(case.get("eval_split", ""))
        selected_input_view = str(case.get("selected_input_view", ""))
        front_views = set(case.get("primary_front_views") or DEFAULT_FRONT_VIEWS)
        for view_id in render_config.get("view_ids", []):
            view_id = str(view_id)
            reference_path = resolve_project_path(case["reference_images"][view_id])
            base_path = render_root / "renders" / split / case_id / "base" / f"{view_id}.png"
            if not reference_path.is_file():
                errors.append(f"missing reference image: {reference_path}")
                continue
            if not base_path.is_file():
                errors.append(f"missing base rendered image: {base_path}")
                continue
            reference_size = Image.open(reference_path).size
            try:
                base_metrics, base_resized = image_pair_metrics(reference_path, base_path, background_rgb)
            except Exception as exc:
                errors.append(f"failed base metrics for {case_id}/{view_id}: {type(exc).__name__}: {exc}")
                continue
            for variant in variants:
                variant_path = render_root / "renders" / split / case_id / variant / f"{view_id}.png"
                if not variant_path.is_file():
                    errors.append(f"missing rendered image: {variant_path}")
                    continue
                try:
                    variant_metrics, variant_resized = image_pair_metrics(reference_path, variant_path, background_rgb)
                    base_vs_variant = candidate_pair_metrics(base_path, variant_path, reference_size, background_rgb)
                except Exception as exc:
                    errors.append(f"failed metrics for {case_id}/{variant}/{view_id}: {type(exc).__name__}: {exc}")
                    continue
                groups = view_groups(view_id, selected_input_view, front_views)
                rows.append(
                    {
                        "case_id": case_id,
                        "eval_split": split,
                        "source_split": case.get("source_split", split),
                        "variant": variant,
                        "view_id": view_id,
                        "selected_input_view": selected_input_view,
                        "view_groups": groups,
                        "reference_path": str(reference_path),
                        "rendered_path": str(variant_path),
                        "base_rendered_path": str(base_path),
                        "base_resized": base_resized,
                        "variant_resized": variant_resized,
                        "base_mae": base_metrics["mae"],
                        "variant_mae": variant_metrics["mae"],
                        "base_rmse": base_metrics["rmse"],
                        "variant_rmse": variant_metrics["rmse"],
                        "base_ssim_like": base_metrics["ssim_like"],
                        "variant_ssim_like": variant_metrics["ssim_like"],
                        "base_histogram_l1": base_metrics["histogram_l1"],
                        "variant_histogram_l1": variant_metrics["histogram_l1"],
                        "base_edge_difference": base_metrics["edge_difference"],
                        "variant_edge_difference": variant_metrics["edge_difference"],
                        "base_vs_variant_mae": base_vs_variant["mae"],
                        "base_vs_variant_rmse": base_vs_variant["rmse"],
                        "base_vs_variant_ssim_like": base_vs_variant["ssim_like"],
                        "variant_minus_base_mae": variant_metrics["mae"] - base_metrics["mae"],
                        "variant_minus_base_rmse": variant_metrics["rmse"] - base_metrics["rmse"],
                        "variant_minus_base_ssim_like": variant_metrics["ssim_like"] - base_metrics["ssim_like"],
                    }
                )
    return rows, errors


def rows_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "case_id",
        "eval_split",
        "variant",
        "view_id",
        "selected_input_view",
        "view_groups",
        "reference_path",
        "rendered_path",
        "base_rendered_path",
        "base_mae",
        "variant_mae",
        "base_rmse",
        "variant_rmse",
        "base_ssim_like",
        "variant_ssim_like",
        "base_vs_variant_mae",
        "variant_minus_base_mae",
        "variant_minus_base_rmse",
        "variant_minus_base_ssim_like",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fields}
            out["view_groups"] = ";".join(row.get("view_groups", []))
            writer.writerow(out)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def build_aggregate_summary(render_config: dict[str, Any], rows: list[dict[str, Any]], errors: list[str], render_root: Path, eval_root: Path) -> dict[str, Any]:
    variants = list(render_config.get("variants", DEFAULT_VARIANTS))
    view_group_names = [
        "all_views",
        "input_view_005",
        "selected_input_view",
        "front_views_004_005",
        "front_views",
        "non_front_views_000_003",
        "non_front_back_views",
    ]
    by_variant = {variant: summarize_rows([row for row in rows if row["variant"] == variant]) for variant in variants}
    by_split = {
        split: {variant: summarize_rows([row for row in rows if row["eval_split"] == split and row["variant"] == variant]) for variant in variants}
        for split in sorted({row["eval_split"] for row in rows})
    }
    by_case = {
        case_id: {variant: summarize_rows([row for row in rows if row["case_id"] == case_id and row["variant"] == variant]) for variant in variants}
        for case_id in sorted({row["case_id"] for row in rows})
    }
    by_view = {
        view_id: {variant: summarize_rows([row for row in rows if row["view_id"] == view_id and row["variant"] == variant]) for variant in variants}
        for view_id in sorted({row["view_id"] for row in rows})
    }
    by_view_group = {
        group: {variant: summarize_rows([row for row in rows if group in row.get("view_groups", []) and row["variant"] == variant]) for variant in variants}
        for group in view_group_names
    }
    return {
        "phase": "2M.3C",
        "status": "OK" if not errors and rows else "FAIL",
        "ok": not errors and bool(rows),
        "render_root": str(render_root),
        "eval_root": str(eval_root),
        "case_count": len({row["case_id"] for row in rows}),
        "row_count": len(rows),
        "variants": variants,
        "view_ids": list(render_config.get("view_ids", [])),
        "by_variant": by_variant,
        "by_split": by_split,
        "by_case": by_case,
        "by_view": by_view,
        "by_view_group": by_view_group,
        "errors": errors,
    }


def build_scale_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    variants = [variant for variant in summary["variants"] if variant != "base"]
    view_groups = ["all_views", "front_views_004_005", "non_front_views_000_003", "input_view_005", "selected_input_view"]
    comparisons: dict[str, Any] = {}
    for variant in variants:
        comparisons[variant] = {
            "overall": summary["by_variant"].get(variant, {}),
            "view_groups": {group: summary["by_view_group"].get(group, {}).get(variant, {}) for group in view_groups},
            "splits": {split: per_variant.get(variant, {}) for split, per_variant in summary["by_split"].items()},
        }
    candidates = [
        (variant, comparisons[variant]["overall"].get("variant_minus_base_mean_mae"))
        for variant in variants
        if isinstance(comparisons[variant]["overall"].get("variant_minus_base_mean_mae"), (int, float))
    ]
    best = min(candidates, key=lambda item: item[1])[0] if candidates else None
    return {
        "phase": "2M.3C",
        "base_variant": "base",
        "compared_variants": variants,
        "comparison_note": "negative variant_minus_base_mean_mae means LoRA is closer to reference than corrected-input base",
        "comparisons": comparisons,
        "best_variant_by_all_views_mae_delta": best,
    }


def write_summary_md(path: Path, summary: dict[str, Any], scale_comparison: dict[str, Any]) -> None:
    lines = [
        "# Phase 2M.3C LoRA Pilot Rendered-View Summary",
        "",
        f"status: `{summary['status']}`",
        f"case_count: `{summary['case_count']}`",
        f"row_count: `{summary['row_count']}`",
        f"best all-view MAE delta: `{scale_comparison['best_variant_by_all_views_mae_delta']}`",
        "",
        "## Variants",
        "",
        "| Variant | Views | Base MAE | Variant MAE | MAE Delta | Base SSIM-like | Variant SSIM-like | SSIM Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, data in summary["by_variant"].items():
        lines.append(
            f"| `{variant}` | `{data['view_count']}` | `{data['base_mean_mae']}` | `{data['variant_mean_mae']}` | `{data['variant_minus_base_mean_mae']}` | `{data['base_mean_ssim_like']}` | `{data['variant_mean_ssim_like']}` | `{data['variant_minus_base_mean_ssim_like']}` |"
        )
    lines.extend(["", "## View Groups", ""])
    for group, per_variant in summary["by_view_group"].items():
        lines.append(f"### {group}")
        for variant, data in per_variant.items():
            lines.append(f"- `{variant}`: views=`{data['view_count']}`, mae_delta=`{data['variant_minus_base_mean_mae']}`, ssim_delta=`{data['variant_minus_base_mean_ssim_like']}`")
        lines.append("")
    if summary["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- `{error}`" for error in summary["errors"])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2M.3C LoRA rendered-view metrics.")
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_root = resolve_project_path(args.render_root)
    eval_root = resolve_project_path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    render_config = load_json(render_root / "render_eval_cases.json")
    rows, errors = collect_metric_rows(render_config, render_root)
    rows_to_csv(rows, eval_root / "metrics_rows.csv")
    summary = build_aggregate_summary(render_config, rows, errors, render_root, eval_root)
    scale_comparison = build_scale_comparison(summary)
    write_json(eval_root / "aggregate_summary.json", summary)
    write_json(eval_root / "scale_comparison.json", scale_comparison)
    write_summary_md(eval_root / "aggregate_summary.md", summary, scale_comparison)
    print("Phase 2M.3C LoRA multi-scale pilot aggregation")
    print(f"  row_count: {summary['row_count']}")
    print(f"  case_count: {summary['case_count']}")
    print(f"  errors: {len(errors)}")
    if summary["ok"]:
        print("PHASE2M_M3C_AGGREGATE_LORA_PILOT_OK")
        return 0
    for error in errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
