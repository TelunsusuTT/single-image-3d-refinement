#!/usr/bin/env python3
"""Create Phase 2G.6 base-vs-fine-tuned texture diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Number
from pathlib import Path
from typing import Any


OUTPUT_BOARD = "base_vs_finetuned_texture_board.jpg"
OUTPUT_JSON = "base_vs_finetuned_metrics.json"
OUTPUT_MD = "base_vs_finetuned_report.md"
MAX_CELL_WIDTH = 384
LABEL_HEIGHT = 24


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make Phase 2G.6 texture comparison board and metrics.")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--finetuned-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def image_paths(case_dir: Path, base_dir: Path, finetuned_dir: Path) -> dict[str, Path]:
    return {
        "reference": case_dir / "input" / "image.png",
        "base_albedo": base_dir / "base_textured_mesh.jpg",
        "base_metallic": base_dir / "base_textured_mesh_metallic.jpg",
        "base_roughness": base_dir / "base_textured_mesh_roughness.jpg",
        "finetuned_albedo": finetuned_dir / "finetuned_textured_mesh.jpg",
        "finetuned_metallic": finetuned_dir / "finetuned_textured_mesh_metallic.jpg",
        "finetuned_roughness": finetuned_dir / "finetuned_textured_mesh_roughness.jpg",
    }


def mesh_inventory(base_dir: Path, finetuned_dir: Path) -> dict[str, Any]:
    paths = {
        "base_obj": base_dir / "base_textured_mesh.obj",
        "base_glb": base_dir / "base_textured_mesh.glb",
        "base_mtl": base_dir / "base_textured_mesh.mtl",
        "finetuned_obj": finetuned_dir / "finetuned_textured_mesh.obj",
        "finetuned_glb": finetuned_dir / "finetuned_textured_mesh.glb",
        "finetuned_mtl": finetuned_dir / "finetuned_textured_mesh.mtl",
    }
    return {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
        for name, path in paths.items()
    }


def channel_values(pixel: Any) -> tuple[int, ...]:
    if isinstance(pixel, int):
        return (pixel,)
    return tuple(int(value) for value in pixel[:3])


def normalize_extrema(extrema: Any) -> list[tuple[int, int]]:
    """Normalize PIL extrema from grayscale or multi-channel images."""
    if (
        isinstance(extrema, (tuple, list))
        and len(extrema) == 2
        and all(isinstance(value, Number) for value in extrema)
    ):
        return [(int(extrema[0]), int(extrema[1]))]
    return [(int(pair[0]), int(pair[1])) for pair in extrema]


def numeric_list(values: Any) -> list[float]:
    if isinstance(values, Number):
        return [float(values)]
    return [float(value) for value in values]


def image_stats(image: Any) -> dict[str, Any]:
    from PIL import ImageStat

    stat_image = image.convert("RGB") if image.mode not in ("L", "RGB") else image
    stat = ImageStat.Stat(stat_image)
    gray = image.convert("L")
    gray_stat = ImageStat.Stat(gray)
    extrema = normalize_extrema(stat_image.getextrema())
    gray_data = list(gray.getdata())
    total = len(gray_data) or 1
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "per_channel_mean": numeric_list(stat.mean),
        "per_channel_std": numeric_list(stat.stddev),
        "per_channel_min": [int(pair[0]) for pair in extrema],
        "per_channel_max": [int(pair[1]) for pair in extrema],
        "grayscale_mean": float(gray_stat.mean[0]),
        "grayscale_std": float(gray_stat.stddev[0]),
        "dark_pixel_ratio_threshold_30": sum(1 for value in gray_data if value < 30) / total,
        "bright_pixel_ratio_threshold_225": sum(1 for value in gray_data if value > 225) / total,
    }


def pair_metrics(base: Any, finetuned: Any, compare_mode: str = "RGB") -> dict[str, Any]:
    from PIL import ImageChops

    if base.size != finetuned.size:
        raise ValueError(f"image dimensions differ: base={base.size} finetuned={finetuned.size}")
    base_compare = base.convert(compare_mode)
    finetuned_compare = finetuned.convert(compare_mode)
    diff = ImageChops.difference(base_compare, finetuned_compare)
    pixels = list(diff.getdata())
    abs_sum = 0
    square_sum = 0
    max_abs = 0
    changed_5 = 0
    changed_15 = 0
    total_values = 0
    for pixel in pixels:
        values = channel_values(pixel)
        pixel_max = max(values)
        changed_5 += int(pixel_max > 5)
        changed_15 += int(pixel_max > 15)
        total_values += len(values)
        for value in values:
            abs_sum += value
            square_sum += value * value
            max_abs = max(max_abs, value)
    total_values = max(1, total_values)
    total_pixels = len(pixels) or 1
    return {
        "dimensions_match": True,
        "compare_mode": compare_mode,
        "width": base.width,
        "height": base.height,
        "mean_abs_diff": abs_sum / total_values,
        "rmse": math.sqrt(square_sum / total_values),
        "max_abs_diff": max_abs,
        "percent_pixels_changed_threshold_5": 100.0 * changed_5 / total_pixels,
        "percent_pixels_changed_threshold_15": 100.0 * changed_15 / total_pixels,
    }


def abs_diff_image(base: Any, finetuned: Any) -> Any:
    from PIL import ImageChops

    return ImageChops.difference(base.convert("RGB"), finetuned.convert("RGB"))


def resize_for_cell(image: Any, max_width: int = MAX_CELL_WIDTH) -> Any:
    if image.width <= max_width:
        return image.copy()
    height = max(1, round(image.height * (max_width / image.width)))
    return image.resize((max_width, height))


def make_cell(image: Any, label: str, width: int, height: int) -> Any:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0))
    resized = resize_for_cell(image.convert("RGB"))
    x = (width - resized.width) // 2
    y = LABEL_HEIGHT + (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def make_board(images: dict[str, Any], out_path: Path) -> None:
    from PIL import Image

    rows = [
        [
            ("Reference", images["reference"]),
            ("Base albedo", images["base_albedo"]),
            ("Fine-tuned albedo", images["finetuned_albedo"]),
            ("Albedo abs diff", abs_diff_image(images["base_albedo"], images["finetuned_albedo"])),
        ],
        [
            ("Base metallic", images["base_metallic"]),
            ("Fine-tuned metallic", images["finetuned_metallic"]),
            ("Metallic abs diff", abs_diff_image(images["base_metallic"], images["finetuned_metallic"])),
        ],
        [
            ("Base roughness", images["base_roughness"]),
            ("Fine-tuned roughness", images["finetuned_roughness"]),
            ("Roughness abs diff", abs_diff_image(images["base_roughness"], images["finetuned_roughness"])),
        ],
    ]
    cell_width = MAX_CELL_WIDTH
    resized_heights = [
        resize_for_cell(image.convert("RGB")).height
        for row in rows
        for _, image in row
    ]
    cell_height = max(resized_heights) if resized_heights else MAX_CELL_WIDTH
    row_widths = [len(row) * cell_width for row in rows]
    board = Image.new("RGB", (max(row_widths), len(rows) * (cell_height + LABEL_HEIGHT)), "white")
    y = 0
    for row in rows:
        x = 0
        for label, image in row:
            cell = make_cell(image, label, cell_width, cell_height)
            board.paste(cell, (x, y))
            x += cell_width
        y += cell_height + LABEL_HEIGHT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path, quality=92)


def shift_label(shift: float, threshold: float = 20.0) -> str:
    if shift > threshold:
        return "fine-tuned much higher than base"
    if shift < -threshold:
        return "fine-tuned much lower than base"
    return "fine-tuned close to base"


def interpretation(metrics: dict[str, Any]) -> dict[str, Any]:
    pair = metrics["pair_metrics"]
    map_stats = metrics["map_stats"]
    albedo = pair["albedo"]
    metallic = pair["metallic"]
    roughness = pair["roughness"]
    metallic_shift = map_stats["finetuned_metallic"]["grayscale_mean"] - map_stats["base_metallic"]["grayscale_mean"]
    roughness_shift = map_stats["finetuned_roughness"]["grayscale_mean"] - map_stats["base_roughness"]["grayscale_mean"]
    return {
        "albedo_differs_strongly": albedo["mean_abs_diff"] > 20.0 or albedo["percent_pixels_changed_threshold_15"] > 25.0,
        "metallic_differs_strongly": metallic["mean_abs_diff"] > 20.0 or abs(metallic_shift) > 20.0,
        "roughness_differs_strongly": roughness["mean_abs_diff"] > 20.0 or abs(roughness_shift) > 20.0,
        "albedo_exactly_identical": albedo["max_abs_diff"] == 0,
        "metallic_exactly_identical": metallic["max_abs_diff"] == 0,
        "roughness_exactly_identical": roughness["max_abs_diff"] == 0,
        "metallic_mean_shift": metallic_shift,
        "metallic_shift_label": shift_label(metallic_shift),
        "roughness_mean_shift": roughness_shift,
        "roughness_shift_label": shift_label(roughness_shift),
        "note": "Diagnostic only; no final quality claim.",
    }


def write_report(metrics: dict[str, Any], out_md: Path) -> None:
    interp = metrics["interpretation"]
    lines = [
        "# Phase 2G.6 Base vs Fine-Tuned Diagnostic Report",
        "",
        "## Diagnostic Interpretation",
        f"- albedo differs strongly: `{interp['albedo_differs_strongly']}`",
        f"- metallic differs strongly: `{interp['metallic_differs_strongly']}`",
        f"- roughness differs strongly: `{interp['roughness_differs_strongly']}`",
        f"- albedo exactly identical: `{interp['albedo_exactly_identical']}`",
        f"- metallic exactly identical: `{interp['metallic_exactly_identical']}`",
        f"- roughness exactly identical: `{interp['roughness_exactly_identical']}`",
        f"- metallic mean shift: `{interp['metallic_mean_shift']:.3f}` ({interp['metallic_shift_label']})",
        f"- roughness mean shift: `{interp['roughness_mean_shift']:.3f}` ({interp['roughness_shift_label']})",
        "- no final quality claim",
        "",
        "## Pair Metrics",
    ]
    for name, item in metrics["pair_metrics"].items():
        lines.extend(
            [
                f"### {name}",
                f"- mean abs diff: `{item['mean_abs_diff']:.3f}`",
                f"- rmse: `{item['rmse']:.3f}`",
                f"- max abs diff: `{item['max_abs_diff']}`",
                f"- pixels changed > 5: `{item['percent_pixels_changed_threshold_5']:.3f}%`",
                f"- pixels changed > 15: `{item['percent_pixels_changed_threshold_15']:.3f}%`",
            ]
        )
    lines.extend(["", "## Metallic And Roughness Means"])
    for map_name in ("metallic", "roughness"):
        base = metrics["map_stats"][f"base_{map_name}"]["grayscale_mean"]
        ft = metrics["map_stats"][f"finetuned_{map_name}"]["grayscale_mean"]
        lines.append(f"- {map_name}: base=`{base:.3f}` fine-tuned=`{ft:.3f}` shift=`{ft - base:.3f}`")
    lines.extend(["", "## GLB/OBJ Inventory"])
    for name, item in metrics["mesh_inventory"].items():
        lines.append(f"- {name}: exists=`{item['exists']}` size_bytes=`{item['size_bytes']}` path=`{item['path']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from PIL import Image

    case_dir = args.case_dir.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    finetuned_dir = args.finetuned_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = image_paths(case_dir, base_dir, finetuned_dir)
    images = {name: Image.open(path).copy() for name, path in paths.items()}

    pair = {
        "albedo": pair_metrics(images["base_albedo"], images["finetuned_albedo"], "RGB"),
        "metallic": pair_metrics(images["base_metallic"], images["finetuned_metallic"], "L"),
        "roughness": pair_metrics(images["base_roughness"], images["finetuned_roughness"], "L"),
    }
    stats = {name: image_stats(image) for name, image in images.items()}
    metrics: dict[str, Any] = {
        "case_dir": str(case_dir),
        "base_dir": str(base_dir),
        "finetuned_dir": str(finetuned_dir),
        "output_dir": str(output_dir),
        "image_paths": {name: str(path) for name, path in paths.items()},
        "map_stats": stats,
        "pair_metrics": pair,
        "mesh_inventory": mesh_inventory(base_dir, finetuned_dir),
    }
    metrics["interpretation"] = interpretation(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    board_path = output_dir / OUTPUT_BOARD
    json_path = output_dir / OUTPUT_JSON
    md_path = output_dir / OUTPUT_MD
    make_board(images, board_path)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, md_path)

    print("Phase 2G.6 texture comparison")
    print(f"  board: {board_path}")
    print(f"  metrics_json: {json_path}")
    print(f"  report_md: {md_path}")
    print("PHASE2G6_TEXTURE_COMPARISON_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
