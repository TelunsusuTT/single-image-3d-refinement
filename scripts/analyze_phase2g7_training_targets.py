#!/usr/bin/env python3
"""Analyze pilot_v1 training targets for Phase 2G.7 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TARGET_ASSET_ID = "B075YLTF7Q"
OUTPUT_JSON = "phase2g7_training_target_stats.json"
OUTPUT_CSV = "phase2g7_training_target_stats.csv"
OUTPUT_MD = "phase2g7_training_target_report.md"
OUTPUT_BOARD = "phase2g7_mr_target_channel_board.jpg"
MAX_CELL_WIDTH = 256
LABEL_HEIGHT = 24


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 2G.7 training targets.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--finetuned-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_sample_dir(row: dict[str, str], dataset_root: Path) -> Path:
    raw = row.get("sample_dir") or row.get("sample_path") or ""
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    sample_name = row.get("sample_name") or row.get("source_id") or row.get("candidate_id") or ""
    return (dataset_root / sample_name).resolve()


def channel_values(pixel: Any) -> tuple[int, ...]:
    if isinstance(pixel, int):
        return (pixel,)
    return tuple(int(value) for value in pixel[:3])


def scalar_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def image_channel_stats(image: Any, mode: str = "RGB") -> dict[str, Any]:
    converted = image.convert(mode)
    channels = len(converted.getbands())
    values = [[] for _ in range(channels)]
    for pixel in converted.getdata():
        for index, value in enumerate(channel_values(pixel)):
            if index < channels:
                values[index].append(float(value))
    return {
        "width": converted.width,
        "height": converted.height,
        "mode": converted.mode,
        "bands": list(converted.getbands()),
        "channels": [
            {
                "band": converted.getbands()[index],
                **scalar_summary(channel),
            }
            for index, channel in enumerate(values)
        ],
    }


def albedo_stats(image: Any) -> dict[str, Any]:
    rgb = image_channel_stats(image, "RGB")
    gray = image.convert("L")
    gray_values = [float(value) for value in gray.getdata()]
    total = len(gray_values) or 1
    rgb["grayscale"] = scalar_summary(gray_values)
    rgb["dark_pixel_ratio_threshold_30"] = sum(1 for value in gray_values if value < 30) / total
    rgb["bright_pixel_ratio_threshold_225"] = sum(1 for value in gray_values if value > 225) / total
    return rgb


def mr_stats(image: Any) -> dict[str, Any]:
    return image_channel_stats(image, "RGB")


def load_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.copy()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_asset_stats(image_stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not image_stats_list:
        return {}
    band_names = [channel["band"] for channel in image_stats_list[0]["channels"]]
    return {
        "image_count": len(image_stats_list),
        "channels": [
            {
                "band": band,
                "mean_of_means": mean([stats["channels"][index]["mean"] for stats in image_stats_list]),
                "mean_of_stds": mean([stats["channels"][index]["std"] for stats in image_stats_list]),
                "min": min(stats["channels"][index]["min"] for stats in image_stats_list),
                "max": max(stats["channels"][index]["max"] for stats in image_stats_list),
            }
            for index, band in enumerate(band_names)
        ],
    }


def output_map_paths(base_dir: Path, finetuned_dir: Path) -> dict[str, Path]:
    return {
        "base_albedo": base_dir / "base_textured_mesh.jpg",
        "base_metallic": base_dir / "base_textured_mesh_metallic.jpg",
        "base_roughness": base_dir / "base_textured_mesh_roughness.jpg",
        "finetuned_albedo": finetuned_dir / "finetuned_textured_mesh.jpg",
        "finetuned_metallic": finetuned_dir / "finetuned_textured_mesh_metallic.jpg",
        "finetuned_roughness": finetuned_dir / "finetuned_textured_mesh_roughness.jpg",
    }


def analyze_sample(row: dict[str, str], dataset_root: Path) -> dict[str, Any]:
    sample_dir = resolve_sample_dir(row, dataset_root)
    render_tex = sample_dir / "render_tex"
    mr_files = sorted(render_tex.glob("*_mr.png"))
    albedo_files = sorted(render_tex.glob("*_albedo.png"))
    mr_per_view = [{"path": str(path), **mr_stats(load_image(path))} for path in mr_files]
    albedo_per_view = [{"path": str(path), **albedo_stats(load_image(path))} for path in albedo_files]
    return {
        "source_id": row.get("source_id", ""),
        "sample_name": row.get("sample_name", ""),
        "candidate_id": row.get("candidate_id", ""),
        "name": row.get("name", ""),
        "sample_dir": str(sample_dir),
        "mr_count": len(mr_files),
        "albedo_count": len(albedo_files),
        "mr_per_view": mr_per_view,
        "albedo_per_view": albedo_per_view,
        "mr_asset_aggregate": aggregate_asset_stats(mr_per_view),
        "albedo_asset_aggregate": aggregate_asset_stats(albedo_per_view),
    }


def global_channel_aggregate(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    per_view = [view for sample in samples for view in sample[key]]
    return aggregate_asset_stats(per_view)


def output_stats(base_dir: Path, finetuned_dir: Path) -> dict[str, Any]:
    stats = {}
    for name, path in output_map_paths(base_dir, finetuned_dir).items():
        image = load_image(path)
        stats[name] = albedo_stats(image) if name.endswith("albedo") else image_channel_stats(image, "L")
        stats[name]["path"] = str(path)
    return stats


def channel_mean(aggregate: dict[str, Any], band: str) -> float:
    for channel in aggregate.get("channels", []):
        if channel.get("band") == band:
            return float(channel.get("mean_of_means", channel.get("mean", 0.0)))
    return 0.0


def output_gray_mean(stats: dict[str, Any], name: str) -> float:
    channels = stats[name].get("channels", [])
    return float(channels[0]["mean"]) if channels else 0.0


def diagnostic_interpretation(samples: list[dict[str, Any]], outputs: dict[str, Any]) -> dict[str, Any]:
    global_mr = global_channel_aggregate(samples, "mr_per_view")
    global_albedo = global_channel_aggregate(samples, "albedo_per_view")
    mr_r_mean = channel_mean(global_mr, "R")
    mr_g_mean = channel_mean(global_mr, "G")
    mr_b_mean = channel_mean(global_mr, "B")
    ft_metallic = output_gray_mean(outputs, "finetuned_metallic")
    base_metallic = output_gray_mean(outputs, "base_metallic")
    ft_roughness = output_gray_mean(outputs, "finetuned_roughness")
    base_roughness = output_gray_mean(outputs, "base_roughness")
    albedo_gray_means = [
        view["grayscale"]["mean"]
        for sample in samples
        for view in sample["albedo_per_view"]
        if "grayscale" in view
    ]
    target_albedo_mean = mean(albedo_gray_means)
    ft_albedo_mean = outputs["finetuned_albedo"]["grayscale"]["mean"]
    return {
        "global_mr_r_mean": mr_r_mean,
        "global_mr_g_mean": mr_g_mean,
        "global_mr_b_mean": mr_b_mean,
        "base_metallic_mean": base_metallic,
        "finetuned_metallic_mean": ft_metallic,
        "base_roughness_mean": base_roughness,
        "finetuned_roughness_mean": ft_roughness,
        "target_albedo_grayscale_mean": target_albedo_mean,
        "finetuned_albedo_grayscale_mean": ft_albedo_mean,
        "possible_metallic_target_issue": mr_r_mean > 96.0,
        "possible_training_collapse_not_target_metallic": mr_r_mean <= 64.0 and ft_metallic - base_metallic > 50.0,
        "possible_target_albedo_issue": target_albedo_mean < 45.0,
        "possible_overfit_or_catastrophic_forgetting": target_albedo_mean >= 45.0 and abs(ft_albedo_mean - target_albedo_mean) > 50.0,
        "note": "Diagnostic only; no retraining decision or quality claim.",
    }


def write_csv(samples: list[dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "source_id",
        "sample_name",
        "mr_count",
        "albedo_count",
        "mr_r_mean",
        "mr_g_mean",
        "mr_b_mean",
        "albedo_r_mean",
        "albedo_g_mean",
        "albedo_b_mean",
        "albedo_gray_mean",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            mr_agg = sample["mr_asset_aggregate"]
            albedo_agg = sample["albedo_asset_aggregate"]
            gray_means = [
                view["grayscale"]["mean"]
                for view in sample["albedo_per_view"]
                if "grayscale" in view
            ]
            writer.writerow(
                {
                    "source_id": sample["source_id"],
                    "sample_name": sample["sample_name"],
                    "mr_count": sample["mr_count"],
                    "albedo_count": sample["albedo_count"],
                    "mr_r_mean": channel_mean(mr_agg, "R"),
                    "mr_g_mean": channel_mean(mr_agg, "G"),
                    "mr_b_mean": channel_mean(mr_agg, "B"),
                    "albedo_r_mean": channel_mean(albedo_agg, "R"),
                    "albedo_g_mean": channel_mean(albedo_agg, "G"),
                    "albedo_b_mean": channel_mean(albedo_agg, "B"),
                    "albedo_gray_mean": mean(gray_means),
                }
            )


def resize_for_cell(image: Any) -> Any:
    if image.width <= MAX_CELL_WIDTH:
        return image.convert("RGB")
    height = max(1, round(image.height * (MAX_CELL_WIDTH / image.width)))
    return image.convert("RGB").resize((MAX_CELL_WIDTH, height))


def make_cell(image: Any, label: str, width: int, height: int) -> Any:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(0, 0, 0))
    resized = resize_for_cell(image)
    canvas.paste(resized, ((width - resized.width) // 2, LABEL_HEIGHT + (height - resized.height) // 2))
    return canvas


def channel_panel(path: Path, channel: str) -> Any:
    image = load_image(path).convert("RGB")
    index = {"R": 0, "G": 1, "B": 2}[channel]
    return image.getchannel(index).convert("RGB")


def make_board(samples: list[dict[str, Any]], base_dir: Path, finetuned_dir: Path, out_path: Path) -> None:
    from PIL import Image

    target = next((sample for sample in samples if sample["source_id"] == TARGET_ASSET_ID), None)
    if target is None:
        target = next((sample for sample in samples if sample["sample_name"] == TARGET_ASSET_ID), samples[0])
    mr_paths = [Path(view["path"]) for view in target["mr_per_view"][:2]]
    albedo_paths = [Path(view["path"]) for view in target["albedo_per_view"][:2]]
    cells: list[tuple[str, Any]] = []
    for path in mr_paths:
        stem = path.stem.split("_")[0]
        for channel in ("R", "G", "B"):
            cells.append((f"{stem} MR {channel}", channel_panel(path, channel)))
    for path in albedo_paths:
        stem = path.stem.split("_")[0]
        cells.append((f"{stem} target albedo", load_image(path)))
    outputs = output_map_paths(base_dir, finetuned_dir)
    for label, key in (
        ("Base metallic", "base_metallic"),
        ("Fine-tuned metallic", "finetuned_metallic"),
        ("Base roughness", "base_roughness"),
        ("Fine-tuned roughness", "finetuned_roughness"),
        ("Base albedo", "base_albedo"),
        ("Fine-tuned albedo", "finetuned_albedo"),
    ):
        cells.append((label, load_image(outputs[key])))
    resized_heights = [resize_for_cell(image).height for _, image in cells]
    cell_width = MAX_CELL_WIDTH
    cell_height = max(resized_heights) if resized_heights else MAX_CELL_WIDTH
    columns = 4
    rows = math.ceil(len(cells) / columns)
    board = Image.new("RGB", (columns * cell_width, rows * (cell_height + LABEL_HEIGHT)), "white")
    for index, (label, image) in enumerate(cells):
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + LABEL_HEIGHT)
        board.paste(make_cell(image, label, cell_width, cell_height), (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path, quality=92)


def write_report(report: dict[str, Any], out_md: Path) -> None:
    interp = report["diagnostic_interpretation"]
    lines = [
        "# Phase 2G.7 Training Target Diagnostic Report",
        "",
        "## Interpretation",
        f"- possible metallic target issue: `{interp['possible_metallic_target_issue']}`",
        f"- possible training collapse not target metallic: `{interp['possible_training_collapse_not_target_metallic']}`",
        f"- possible target albedo issue: `{interp['possible_target_albedo_issue']}`",
        f"- possible overfit or catastrophic forgetting: `{interp['possible_overfit_or_catastrophic_forgetting']}`",
        "- diagnostic only; no retraining decision",
        "",
        "## Global MR Means",
        f"- R: `{interp['global_mr_r_mean']:.3f}`",
        f"- G: `{interp['global_mr_g_mean']:.3f}`",
        f"- B: `{interp['global_mr_b_mean']:.3f}`",
        "",
        "## Output Map Means",
        f"- base metallic: `{interp['base_metallic_mean']:.3f}`",
        f"- fine-tuned metallic: `{interp['finetuned_metallic_mean']:.3f}`",
        f"- base roughness: `{interp['base_roughness_mean']:.3f}`",
        f"- fine-tuned roughness: `{interp['finetuned_roughness_mean']:.3f}`",
        f"- target albedo gray mean: `{interp['target_albedo_grayscale_mean']:.3f}`",
        f"- fine-tuned albedo gray mean: `{interp['finetuned_albedo_grayscale_mean']:.3f}`",
        "",
        "## Per-Asset Summary",
    ]
    for sample in report["samples"]:
        lines.append(
            f"- `{sample['source_id'] or sample['sample_name']}`: "
            f"MR R/G/B means = "
            f"`{channel_mean(sample['mr_asset_aggregate'], 'R'):.3f}`, "
            f"`{channel_mean(sample['mr_asset_aggregate'], 'G'):.3f}`, "
            f"`{channel_mean(sample['mr_asset_aggregate'], 'B'):.3f}`; "
            f"albedo gray mean = "
            f"`{mean([view['grayscale']['mean'] for view in sample['albedo_per_view']]):.3f}`"
        )
    target = next((sample for sample in report["samples"] if sample["source_id"] == TARGET_ASSET_ID), None)
    if target:
        lines.extend(["", f"## {TARGET_ASSET_ID} Per-View Detail"])
        for view in target["mr_per_view"]:
            means = ", ".join(f"{channel['band']}={channel['mean']:.3f}" for channel in view["channels"])
            lines.append(f"- MR `{Path(view['path']).name}`: {means}")
        for view in target["albedo_per_view"]:
            means = ", ".join(f"{channel['band']}={channel['mean']:.3f}" for channel in view["channels"])
            lines.append(f"- Albedo `{Path(view['path']).name}`: {means}; gray={view['grayscale']['mean']:.3f}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_csv = args.manifest_csv.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    finetuned_dir = args.finetuned_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    rows = read_manifest(manifest_csv)
    samples = [analyze_sample(row, dataset_root) for row in rows]
    outputs = output_stats(base_dir, finetuned_dir)
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "manifest_csv": str(manifest_csv),
        "base_dir": str(base_dir),
        "finetuned_dir": str(finetuned_dir),
        "sample_count": len(samples),
        "global_mr_aggregate": global_channel_aggregate(samples, "mr_per_view"),
        "global_albedo_aggregate": global_channel_aggregate(samples, "albedo_per_view"),
        "output_map_stats": outputs,
        "samples": samples,
    }
    report["diagnostic_interpretation"] = diagnostic_interpretation(samples, outputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / OUTPUT_JSON
    out_csv = output_dir / OUTPUT_CSV
    out_md = output_dir / OUTPUT_MD
    out_board = output_dir / OUTPUT_BOARD
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(samples, out_csv)
    write_report(report, out_md)
    make_board(samples, base_dir, finetuned_dir, out_board)

    print("Phase 2G.7 training-target diagnostic")
    print(f"  stats_json: {out_json}")
    print(f"  stats_csv: {out_csv}")
    print(f"  report_md: {out_md}")
    print(f"  board: {out_board}")
    print("PHASE2G7_TRAINING_TARGET_DIAGNOSTIC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
