#!/usr/bin/env python3
"""Compare Phase 2K.3 rendered views against references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = "rendered_view_metrics.json"
OUTPUT_MD = "rendered_view_report.md"
OUTPUT_BOARD = "rendered_view_board.jpg"


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rgb255(values: list[float]) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(float(value) * 255.0)))) for value in values[:3])


def ensure_size(image: Any, target_size: tuple[int, int], resample: Any) -> tuple[Any, bool]:
    if image.size == target_size:
        return image, False
    return image.resize(target_size, resample), True


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def image_mean_rgb(image: Any) -> list[float]:
    from PIL import ImageStat

    stat = ImageStat.Stat(image.convert("RGB"))
    return [float(value) for value in stat.mean[:3]]


def make_foreground_mask(image: Any, background_rgb: tuple[int, int, int]) -> tuple[Any | None, str]:
    from PIL import Image

    if "A" in image.getbands():
        alpha = image.getchannel("A")
        extrema = alpha.getextrema()
        if extrema[0] < 250:
            return alpha.point(lambda value: 255 if value > 8 else 0, mode="L"), "alpha"

    rgb = image.convert("RGB")
    pixels = rgb.getdata()
    mask_data = []
    threshold = 18
    for red, green, blue in pixels:
        distance = abs(red - background_rgb[0]) + abs(green - background_rgb[1]) + abs(blue - background_rgb[2])
        mask_data.append(255 if distance > threshold else 0)
    mask = Image.new("L", rgb.size)
    mask.putdata(mask_data)
    if mask.getbbox() is None:
        return None, "none"
    return mask, "background_distance"


def union_masks(*masks: Any | None) -> Any | None:
    from PIL import ImageChops

    valid = [mask for mask in masks if mask is not None]
    if not valid:
        return None
    result = valid[0]
    for mask in valid[1:]:
        result = ImageChops.lighter(result, mask)
    return result if result.getbbox() is not None else None


def mask_pixel_count(mask: Any | None) -> int:
    if mask is None:
        return 0
    hist = mask.histogram()
    return int(sum(count for value, count in enumerate(hist) if value > 0))


def diff_stats(image_a: Any, image_b: Any, mask: Any | None = None) -> dict[str, Any]:
    from PIL import ImageChops, ImageStat

    diff = ImageChops.difference(image_a.convert("RGB"), image_b.convert("RGB"))
    stat = ImageStat.Stat(diff, mask) if mask is not None else ImageStat.Stat(diff)
    means = [float(value) for value in stat.mean[:3]]
    rms_values = [float(value) for value in stat.rms[:3]]
    mae = float(sum(means) / len(means))
    rmse = math.sqrt(sum(value * value for value in rms_values) / len(rms_values))
    psnr_is_infinite = rmse == 0
    psnr = None if psnr_is_infinite else 20.0 * math.log10(255.0 / rmse)
    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "psnr_is_infinite": psnr_is_infinite,
        "per_channel_mae": means,
    }


def global_ssim_like(image_a: Any, image_b: Any) -> float:
    gray_a = image_a.convert("L")
    gray_b = image_b.convert("L")
    data_a = [float(value) for value in gray_a.getdata()]
    data_b = [float(value) for value in gray_b.getdata()]
    n = len(data_a)
    if n == 0:
        return 0.0
    mean_a = sum(data_a) / n
    mean_b = sum(data_b) / n
    var_a = sum((value - mean_a) ** 2 for value in data_a) / n
    var_b = sum((value - mean_b) ** 2 for value in data_b) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(data_a, data_b)) / n
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0
    return float(((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / denominator)


def histogram_l1(image_a: Any, image_b: Any) -> float:
    rgb_a = image_a.convert("RGB")
    rgb_b = image_b.convert("RGB")
    hist_a = rgb_a.histogram()
    hist_b = rgb_b.histogram()
    total = rgb_a.size[0] * rgb_a.size[1] * 3
    if total == 0:
        return 0.0
    return float(sum(abs(a - b) for a, b in zip(hist_a, hist_b)) / (2.0 * total))


def edge_difference(image_a: Any, image_b: Any) -> float:
    from PIL import ImageChops, ImageFilter, ImageStat

    edge_a = image_a.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_b = image_b.convert("L").filter(ImageFilter.FIND_EDGES)
    diff = ImageChops.difference(edge_a, edge_b)
    return float(ImageStat.Stat(diff).mean[0])


def pair_metrics(
    image_a: Any,
    image_b: Any,
    background_rgb: tuple[int, int, int],
) -> dict[str, Any]:
    mask_a, mask_source_a = make_foreground_mask(image_a, background_rgb)
    mask_b, mask_source_b = make_foreground_mask(image_b, background_rgb)
    mask = union_masks(mask_a, mask_b)
    full = diff_stats(image_a, image_b)
    foreground = diff_stats(image_a, image_b, mask) if mask is not None else None
    mean_a = image_mean_rgb(image_a)
    mean_b = image_mean_rgb(image_b)
    color_shift = [b - a for a, b in zip(mean_a, mean_b)]
    return {
        "mae": full["mae"],
        "rmse": full["rmse"],
        "psnr": full["psnr"],
        "ssim_like": global_ssim_like(image_a, image_b),
        "histogram_l1": histogram_l1(image_a, image_b),
        "edge_difference": edge_difference(image_a, image_b),
        "color_mean_shift": color_shift,
        "mean_abs_color_shift": float(sum(abs(value) for value in color_shift) / len(color_shift)),
        "foreground_mask": {
            "source_a": mask_source_a,
            "source_b": mask_source_b,
            "pixel_count": mask_pixel_count(mask),
        },
        "foreground_mae": foreground["mae"] if foreground else None,
        "foreground_rmse": foreground["rmse"] if foreground else None,
        "per_channel_mae": full["per_channel_mae"],
    }


def diff_image(image_a: Any, image_b: Any) -> Any:
    from PIL import ImageChops, ImageOps

    return ImageOps.autocontrast(ImageChops.difference(image_a.convert("RGB"), image_b.convert("RGB")))


def load_view_images(
    config: dict[str, Any],
    output_root: Path,
    asset_id: str,
    view_id: str,
) -> tuple[dict[str, Any], list[str]]:
    from PIL import Image

    reference_template = config["reference_image_path_template"]
    reference_path = resolve_project_path(
        reference_template.format(asset_id=asset_id, view_id=view_id)
    )
    base_path = output_root / "renders" / asset_id / "base" / f"{view_id}.png"
    finetuned_path = output_root / "renders" / asset_id / "finetuned" / f"{view_id}.png"
    reference = Image.open(reference_path).convert("RGB")
    base_raw = Image.open(base_path).convert("RGB")
    finetuned_raw = Image.open(finetuned_path).convert("RGB")
    target_size = reference.size
    base, base_resized = ensure_size(base_raw, target_size, Image.Resampling.BICUBIC)
    finetuned, finetuned_resized = ensure_size(finetuned_raw, target_size, Image.Resampling.BICUBIC)
    resize_events = []
    if base_resized:
        resize_events.append(f"base {base_raw.size} -> {target_size}")
    if finetuned_resized:
        resize_events.append(f"finetuned {finetuned_raw.size} -> {target_size}")
    paths = {
        "reference": str(reference_path),
        "base": str(base_path),
        "finetuned": str(finetuned_path),
    }
    return {"reference": reference, "base": base, "finetuned": finetuned, "paths": paths}, resize_events


def compare_view(
    config: dict[str, Any],
    output_root: Path,
    asset_id: str,
    view_id: str,
    background_rgb: tuple[int, int, int],
) -> dict[str, Any]:
    images, resize_events = load_view_images(config, output_root, asset_id, view_id)
    base_vs_fine = pair_metrics(images["base"], images["finetuned"], background_rgb)
    base_vs_reference = pair_metrics(images["reference"], images["base"], background_rgb)
    fine_vs_reference = pair_metrics(images["reference"], images["finetuned"], background_rgb)
    improvement = {
        "fine_minus_base_mae": fine_vs_reference["mae"] - base_vs_reference["mae"],
        "fine_minus_base_rmse": fine_vs_reference["rmse"] - base_vs_reference["rmse"],
        "fine_minus_base_histogram_l1": fine_vs_reference["histogram_l1"] - base_vs_reference["histogram_l1"],
        "fine_minus_base_edge_difference": fine_vs_reference["edge_difference"] - base_vs_reference["edge_difference"],
        "fine_minus_base_ssim_like": fine_vs_reference["ssim_like"] - base_vs_reference["ssim_like"],
    }
    return {
        "view_id": view_id,
        "paths": images["paths"],
        "resize_events": resize_events,
        "base_vs_fine": base_vs_fine,
        "base_vs_reference": base_vs_reference,
        "fine_vs_reference": fine_vs_reference,
        "improvement": improvement,
    }


def mean_metric(view_rows: list[dict[str, Any]], pair_name: str, metric_name: str) -> float | None:
    values = [
        row[pair_name][metric_name]
        for row in view_rows
        if isinstance(row.get(pair_name, {}).get(metric_name), (int, float))
    ]
    return mean([float(value) for value in values])


def aggregate_case(view_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "view_count": len(view_rows),
        "base_vs_fine_mean_mae": mean_metric(view_rows, "base_vs_fine", "mae"),
        "base_vs_fine_mean_rmse": mean_metric(view_rows, "base_vs_fine", "rmse"),
        "base_vs_fine_mean_ssim_like": mean_metric(view_rows, "base_vs_fine", "ssim_like"),
        "base_vs_reference_mean_mae": mean_metric(view_rows, "base_vs_reference", "mae"),
        "fine_vs_reference_mean_mae": mean_metric(view_rows, "fine_vs_reference", "mae"),
        "base_vs_reference_mean_ssim_like": mean_metric(view_rows, "base_vs_reference", "ssim_like"),
        "fine_vs_reference_mean_ssim_like": mean_metric(view_rows, "fine_vs_reference", "ssim_like"),
        "views_fine_improves_mae": sum(1 for row in view_rows if row["improvement"]["fine_minus_base_mae"] < 0),
        "views_fine_improves_ssim": sum(1 for row in view_rows if row["improvement"]["fine_minus_base_ssim_like"] > 0),
    }


def make_case_board(
    config: dict[str, Any],
    output_root: Path,
    asset_id: str,
    board_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    view_ids = list(config["view_ids"])
    target_size = (160, 160)
    label_height = 28
    headers = [
        "reference",
        "base",
        "fine-tuned",
        "base/fine diff",
        "ref/base diff",
        "ref/fine diff",
    ]
    columns = len(headers)
    rows = len(view_ids)
    board = Image.new(
        "RGB",
        (columns * target_size[0], label_height + rows * (target_size[1] + label_height)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(board)
    for col, header in enumerate(headers):
        draw.text((col * target_size[0] + 6, 8), header, fill=(0, 0, 0))
    for row_index, view_id in enumerate(view_ids):
        images, _ = load_view_images(config, output_root, asset_id, view_id)
        cells = [
            images["reference"],
            images["base"],
            images["finetuned"],
            diff_image(images["base"], images["finetuned"]),
            diff_image(images["reference"], images["base"]),
            diff_image(images["reference"], images["finetuned"]),
        ]
        y = label_height + row_index * (target_size[1] + label_height)
        draw.text((4, y + 6), view_id, fill=(0, 0, 0))
        for col, image in enumerate(cells):
            thumb = image.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
            board.paste(thumb, (col * target_size[0], y + label_height))
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(board_path, quality=92)


def write_case_report(asset_id: str, metrics: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Phase 2K.3 Rendered-View Report: {asset_id}",
        "",
        f"view_count: `{metrics['aggregate']['view_count']}`",
        "",
        "## Aggregate",
    ]
    for key, value in metrics["aggregate"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Views"])
    for row in metrics["views"]:
        lines.extend(
            [
                f"### View {row['view_id']}",
                f"- base vs fine MAE: `{row['base_vs_fine']['mae']}`",
                f"- base vs reference MAE: `{row['base_vs_reference']['mae']}`",
                f"- fine vs reference MAE: `{row['fine_vs_reference']['mae']}`",
                f"- fine minus base MAE: `{row['improvement']['fine_minus_base_mae']}`",
                f"- fine minus base SSIM-like: `{row['improvement']['fine_minus_base_ssim_like']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compare_all(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    background_rgb = rgb255(list(config.get("background_color", [0.28, 0.28, 0.28])))
    all_metrics: dict[str, Any] = {}
    for asset_id in config["cases"]:
        views = [
            compare_view(config, output_root, asset_id, view_id, background_rgb)
            for view_id in config["view_ids"]
        ]
        case_metrics = {
            "asset_id": asset_id,
            "view_ids": list(config["view_ids"]),
            "background_rgb": background_rgb,
            "views": views,
            "aggregate": aggregate_case(views),
        }
        metrics_dir = output_root / "metrics" / asset_id
        report_dir = output_root / "reports" / asset_id
        board_dir = output_root / "boards" / asset_id
        metrics_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        board_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / OUTPUT_JSON).write_text(
            json.dumps(case_metrics, indent=2),
            encoding="utf-8",
        )
        write_case_report(asset_id, case_metrics, report_dir / OUTPUT_MD)
        make_case_board(config, output_root, asset_id, board_dir / OUTPUT_BOARD)
        all_metrics[asset_id] = case_metrics
    return all_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 2K.3 rendered views.")
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.cases_config)
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    compare_all(config, output_root)
    print("Phase 2K.3 rendered-view comparison")
    print(f"  output_root: {output_root}")
    for asset_id in config["cases"]:
        print(f"  metrics: {output_root / 'metrics' / asset_id / OUTPUT_JSON}")
        print(f"  report: {output_root / 'reports' / asset_id / OUTPUT_MD}")
        print(f"  board: {output_root / 'boards' / asset_id / OUTPUT_BOARD}")
    print("PHASE2K3_RENDERED_VIEW_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
