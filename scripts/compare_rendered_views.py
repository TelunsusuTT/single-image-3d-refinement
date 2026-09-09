#!/usr/bin/env python3
"""Compare canonical fixed-view renders against their reference images."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Mapping

from _evaluation import (
    EvaluationContractError,
    METRIC_NAMES,
    PAIR_NAMES,
    VIEW_GROUP_NAMES,
    asset_label,
    expected_image_size,
    load_cases_manifest,
    load_evaluation_config,
    reference_image_path,
    require_complete_view_rows,
    resolve_project_path,
    summarize_view_groups,
)


OUTPUT_JSON = "rendered_view_metrics.json"
OUTPUT_MD = "rendered_view_report.md"
OUTPUT_BOARD = "rendered_view_board.jpg"


def diff_stats(
    image_a: Any,
    image_b: Any,
    *,
    data_range: float = 255.0,
) -> dict[str, Any]:
    from PIL import ImageChops, ImageStat

    if image_a.size != image_b.size:
        raise EvaluationContractError(
            f"metric images must have identical sizes: {image_a.size} != {image_b.size}"
        )
    diff = ImageChops.difference(image_a.convert("RGB"), image_b.convert("RGB"))
    stat = ImageStat.Stat(diff)
    channel_mae = [float(value) for value in stat.mean[:3]]
    channel_rmse = [float(value) for value in stat.rms[:3]]
    mae = float(sum(channel_mae) / len(channel_mae))
    rmse = math.sqrt(sum(value * value for value in channel_rmse) / len(channel_rmse))
    psnr_is_infinite = rmse == 0
    psnr = None if psnr_is_infinite else 20.0 * math.log10(data_range / rmse)
    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "psnr_is_infinite": psnr_is_infinite,
    }


def global_ssim_like(
    image_a: Any,
    image_b: Any,
    *,
    data_range: float,
    k1: float,
    k2: float,
) -> float:
    if image_a.size != image_b.size:
        raise EvaluationContractError(
            f"metric images must have identical sizes: {image_a.size} != {image_b.size}"
        )
    gray_a = image_a.convert("L")
    gray_b = image_b.convert("L")
    data_a = [float(value) for value in gray_a.getdata()]
    data_b = [float(value) for value in gray_b.getdata()]
    count = len(data_a)
    if count == 0:
        raise EvaluationContractError("metric images must not be empty")
    mean_a = sum(data_a) / count
    mean_b = sum(data_b) / count
    variance_a = sum((value - mean_a) ** 2 for value in data_a) / count
    variance_b = sum((value - mean_b) ** 2 for value in data_b) / count
    covariance = sum(
        (value_a - mean_a) * (value_b - mean_b)
        for value_a, value_b in zip(data_a, data_b)
    ) / count
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    denominator = (
        (mean_a * mean_a + mean_b * mean_b + c1)
        * (variance_a + variance_b + c2)
    )
    if denominator == 0:
        return 1.0
    return float(
        ((2 * mean_a * mean_b + c1) * (2 * covariance + c2))
        / denominator
    )


def pair_metrics(
    image_a: Any,
    image_b: Any,
    metric_config: Mapping[str, Any],
) -> dict[str, Any]:
    enabled = list(metric_config["enabled"])
    data_range = float(metric_config["data_range"])
    base = diff_stats(image_a, image_b, data_range=data_range)
    ssim_config = metric_config["ssim_like"]
    computed = {
        **base,
        "ssim_like": global_ssim_like(
            image_a,
            image_b,
            data_range=data_range,
            k1=float(ssim_config["k1"]),
            k2=float(ssim_config["k2"]),
        ),
    }
    result = {name: computed[name] for name in enabled}
    if "psnr" in enabled:
        result["psnr_is_infinite"] = computed["psnr_is_infinite"]
    return result


def _load_rgb(path: Path) -> tuple[Any, tuple[int, int]]:
    from PIL import Image

    with Image.open(path) as image:
        size = tuple(image.size)
        rgb = image.convert("RGB").copy()
    return rgb, size


def load_view_images(
    evaluation: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
    output_root: Path,
    asset_id: str,
    view_id: str,
) -> dict[str, Any]:
    reference_path = reference_image_path(cases_manifest, asset_id, view_id)
    baseline_path = output_root / "renders" / asset_id / "baseline" / f"{view_id}.png"
    candidate_path = output_root / "renders" / asset_id / "candidate" / f"{view_id}.png"
    paths = {
        "reference": reference_path,
        "baseline": baseline_path,
        "candidate": candidate_path,
    }
    images: dict[str, Any] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {role} image for {asset_id} view {view_id}: {path}")
        images[role], sizes[role] = _load_rgb(path)

    expected = expected_image_size(evaluation)
    mismatches = {
        role: size
        for role, size in sizes.items()
        if size != expected
    }
    if mismatches:
        observed = ", ".join(f"{role}={size}" for role, size in sizes.items())
        raise EvaluationContractError(
            f"image size mismatch for {asset_id} view {view_id}: "
            f"expected {expected}; {observed}. Images are never resized for metrics."
        )
    images["paths"] = {role: str(path) for role, path in paths.items()}
    images["size"] = list(expected)
    return images


def compare_view(
    evaluation: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
    output_root: Path,
    asset_id: str,
    view_id: str,
) -> dict[str, Any]:
    images = load_view_images(
        evaluation,
        cases_manifest,
        output_root,
        asset_id,
        view_id,
    )
    metric_config = evaluation["metrics"]
    baseline_vs_candidate = pair_metrics(
        images["baseline"],
        images["candidate"],
        metric_config,
    )
    baseline_vs_reference = pair_metrics(
        images["reference"],
        images["baseline"],
        metric_config,
    )
    candidate_vs_reference = pair_metrics(
        images["reference"],
        images["candidate"],
        metric_config,
    )
    candidate_minus_baseline = {
        metric_name: (
            None
            if baseline_vs_reference[metric_name] is None
            or candidate_vs_reference[metric_name] is None
            else float(candidate_vs_reference[metric_name])
            - float(baseline_vs_reference[metric_name])
        )
        for metric_name in metric_config["enabled"]
    }
    return {
        "view_id": view_id,
        "image_size": images["size"],
        "paths": images["paths"],
        "baseline_vs_candidate": baseline_vs_candidate,
        "baseline_vs_reference": baseline_vs_reference,
        "candidate_vs_reference": candidate_vs_reference,
        "candidate_minus_baseline": candidate_minus_baseline,
    }


def diff_image(image_a: Any, image_b: Any) -> Any:
    from PIL import ImageChops

    return ImageChops.difference(image_a.convert("RGB"), image_b.convert("RGB"))


def board_headers(cases_manifest: Mapping[str, Any]) -> list[str]:
    baseline_label = str(cases_manifest["baseline_label"])
    candidate_label = str(cases_manifest["candidate_label"])
    return [
        "Reference",
        baseline_label,
        candidate_label,
        f"Difference: {baseline_label} vs {candidate_label}",
        f"Difference: Reference vs {baseline_label}",
        f"Difference: Reference vs {candidate_label}",
    ]


def make_case_board(
    evaluation: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
    output_root: Path,
    asset_id: str,
    board_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    view_ids = list(evaluation["view_groups"]["all"])
    headers = board_headers(cases_manifest)
    cell_size = (200, 200)
    column_width = 240
    row_label_height = 26
    header_height = 76
    title_height = 34
    board = Image.new(
        "RGB",
        (
            len(headers) * column_width,
            title_height + header_height + len(view_ids) * (cell_size[1] + row_label_height),
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(board)
    draw.text(
        (8, 8),
        f"Asset: {asset_label(cases_manifest, asset_id)}",
        fill=(0, 0, 0),
    )
    for column, header in enumerate(headers):
        wrapped = "\n".join(textwrap.wrap(header, width=30))
        draw.multiline_text(
            (column * column_width + 8, title_height + 6),
            wrapped,
            fill=(0, 0, 0),
            spacing=3,
        )

    for row_index, view_id in enumerate(view_ids):
        images = load_view_images(
            evaluation,
            cases_manifest,
            output_root,
            asset_id,
            view_id,
        )
        cells = [
            images["reference"],
            images["baseline"],
            images["candidate"],
            diff_image(images["baseline"], images["candidate"]),
            diff_image(images["reference"], images["baseline"]),
            diff_image(images["reference"], images["candidate"]),
        ]
        row_top = title_height + header_height + row_index * (
            cell_size[1] + row_label_height
        )
        draw.text((8, row_top + 6), f"View {view_id}", fill=(0, 0, 0))
        for column, image in enumerate(cells):
            thumbnail = image.copy()
            thumbnail.thumbnail(cell_size, Image.Resampling.LANCZOS)
            x = column * column_width + (column_width - thumbnail.width) // 2
            y = row_top + row_label_height + (cell_size[1] - thumbnail.height) // 2
            board.paste(thumbnail, (x, y))

    board_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(board_path, quality=92)


def _metric_text(pair: Mapping[str, Any], metric_name: str) -> str:
    value = pair.get(metric_name)
    if metric_name == "psnr" and value is None and pair.get("psnr_is_infinite") is True:
        return "infinite"
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _comparison_labels(cases_manifest: Mapping[str, Any]) -> dict[str, str]:
    baseline_label = str(cases_manifest["baseline_label"])
    candidate_label = str(cases_manifest["candidate_label"])
    return {
        "baseline_vs_candidate": f"{baseline_label} vs {candidate_label}",
        "baseline_vs_reference": f"{baseline_label} vs Reference",
        "candidate_vs_reference": f"{candidate_label} vs Reference",
        "candidate_minus_baseline": (
            f"{candidate_label} minus {baseline_label} against Reference"
        ),
    }


def _append_metric_table(
    lines: list[str],
    summary: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
) -> None:
    labels = _comparison_labels(cases_manifest)
    lines.extend(
        [
            "| Comparison | MAE | RMSE | PSNR | SSIM-like |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair_name in (*PAIR_NAMES, "candidate_minus_baseline"):
        pair = summary[pair_name]
        lines.append(
            "| "
            + labels[pair_name]
            + " | "
            + " | ".join(_metric_text(pair, metric) for metric in METRIC_NAMES)
            + " |"
        )


def write_case_report(
    asset_id: str,
    metrics: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
    path: Path,
) -> None:
    lines = [
        f"# Fixed-View Report: {asset_label(cases_manifest, asset_id)}",
        "",
        f"- Baseline: {cases_manifest['baseline_label']}",
        f"- Candidate: {cases_manifest['candidate_label']}",
        "",
        "## View Groups",
    ]
    for group_name in VIEW_GROUP_NAMES:
        group = metrics["groups"][group_name]
        lines.extend(
            [
                "",
                f"### {group_name}",
                "",
                f"Views: {', '.join(group['view_ids'])}",
                "",
            ]
        )
        _append_metric_table(lines, group, cases_manifest)

    lines.extend(["", "## Per View"])
    for row in metrics["views"]:
        lines.extend(["", f"### View {row['view_id']}", ""])
        _append_metric_table(
            lines,
            {
                **{pair_name: row[pair_name] for pair_name in PAIR_NAMES},
                "candidate_minus_baseline": row["candidate_minus_baseline"],
            },
            cases_manifest,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compare_all(
    evaluation: Mapping[str, Any],
    cases_manifest: Mapping[str, Any],
    output_root: Path,
    *,
    evaluation_path: Path | None = None,
    cases_path: Path | None = None,
) -> dict[str, Any]:
    all_metrics: dict[str, Any] = {}
    view_ids = list(evaluation["view_groups"]["all"])
    for asset_id in cases_manifest["cases"]:
        views = [
            compare_view(
                evaluation,
                cases_manifest,
                output_root,
                asset_id,
                view_id,
            )
            for view_id in view_ids
        ]
        require_complete_view_rows(views, evaluation)
        case_metrics = {
            "schema_version": 1,
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation_config": str(evaluation_path) if evaluation_path else None,
            "cases_config": str(cases_path) if cases_path else None,
            "asset_id": asset_id,
            "asset_label": asset_label(cases_manifest, asset_id),
            "baseline_label": cases_manifest["baseline_label"],
            "candidate_label": cases_manifest["candidate_label"],
            "view_groups": evaluation["view_groups"],
            "metrics": evaluation["metrics"],
            "views": views,
            "groups": summarize_view_groups(views, evaluation),
        }
        metrics_dir = output_root / "metrics" / asset_id
        report_dir = output_root / "reports" / asset_id
        board_dir = output_root / "boards" / asset_id
        metrics_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        board_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / OUTPUT_JSON).write_text(
            json.dumps(case_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        write_case_report(
            asset_id,
            case_metrics,
            cases_manifest,
            report_dir / OUTPUT_MD,
        )
        make_case_board(
            evaluation,
            cases_manifest,
            output_root,
            asset_id,
            board_dir / OUTPUT_BOARD,
        )
        all_metrics[asset_id] = case_metrics
    return all_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare renders under the canonical fixed-view protocol."
    )
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluation_path, evaluation = load_evaluation_config(args.evaluation_config)
    cases_path, cases_manifest = load_cases_manifest(args.cases_config)
    output_root = resolve_project_path(str(args.output_root))
    compare_all(
        evaluation,
        cases_manifest,
        output_root,
        evaluation_path=evaluation_path,
        cases_path=cases_path,
    )
    print("Fixed-View rendered-view comparison")
    print(f"  baseline: {cases_manifest['baseline_label']}")
    print(f"  candidate: {cases_manifest['candidate_label']}")
    print(f"  output_root: {output_root}")
    for asset_id in cases_manifest["cases"]:
        print(f"  metrics: {output_root / 'metrics' / asset_id / OUTPUT_JSON}")
        print(f"  report: {output_root / 'reports' / asset_id / OUTPUT_MD}")
        print(f"  board: {output_root / 'boards' / asset_id / OUTPUT_BOARD}")
    print("RENDERED_VIEW_COMPARISON_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
