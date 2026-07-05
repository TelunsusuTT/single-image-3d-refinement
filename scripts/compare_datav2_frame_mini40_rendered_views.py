#!/usr/bin/env python3
"""Compare mini40 rendered base/fine views against render_cond references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.compare_phase2k3_rendered_views import aggregate_case, diff_image, pair_metrics, rgb255
except ImportError:  # pragma: no cover - supports direct script execution.
    from compare_phase2k3_rendered_views import aggregate_case, diff_image, pair_metrics, rgb255


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON_SUFFIX = "_rendered_view_metrics.json"
OUTPUT_MD_SUFFIX = "_rendered_view_report.md"
OUTPUT_BOARD_SUFFIX = "_rendered_view_board.jpg"


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_size(image: Any, target_size: tuple[int, int]) -> tuple[Any, bool]:
    from PIL import Image

    if image.size == target_size:
        return image, False
    return image.resize(target_size, Image.Resampling.BICUBIC), True


def load_view_images(render_root: Path, case: dict[str, Any], item_id: str, view_id: str) -> tuple[dict[str, Any], list[str]]:
    from PIL import Image

    eval_split = case["eval_split"]
    reference_path = resolve_project_path(case.get("reference_images", {}).get(view_id, ""))
    base_path = render_root / "renders" / eval_split / item_id / "base" / f"{view_id}.png"
    finetuned_path = render_root / "renders" / eval_split / item_id / "finetuned" / f"{view_id}.png"
    reference = Image.open(reference_path).convert("RGB")
    base_raw = Image.open(base_path).convert("RGB")
    fine_raw = Image.open(finetuned_path).convert("RGB")
    base, base_resized = ensure_size(base_raw, reference.size)
    fine, fine_resized = ensure_size(fine_raw, reference.size)
    resize_events = []
    if base_resized:
        resize_events.append(f"base {base_raw.size} -> {reference.size}")
    if fine_resized:
        resize_events.append(f"finetuned {fine_raw.size} -> {reference.size}")
    return {
        "reference": reference,
        "base": base,
        "finetuned": fine,
        "paths": {
            "reference": str(reference_path),
            "base": str(base_path),
            "finetuned": str(finetuned_path),
        },
    }, resize_events


def compare_view(render_root: Path, case: dict[str, Any], item_id: str, view_id: str, background_rgb: tuple[int, int, int]) -> dict[str, Any]:
    images, resize_events = load_view_images(render_root, case, item_id, view_id)
    base_vs_fine = pair_metrics(images["base"], images["finetuned"], background_rgb)
    base_vs_reference = pair_metrics(images["reference"], images["base"], background_rgb)
    fine_vs_reference = pair_metrics(images["reference"], images["finetuned"], background_rgb)
    return {
        "view_id": view_id,
        "paths": images["paths"],
        "resize_events": resize_events,
        "base_vs_fine": base_vs_fine,
        "base_vs_reference": base_vs_reference,
        "fine_vs_reference": fine_vs_reference,
        "improvement": {
            "fine_minus_base_mae": fine_vs_reference["mae"] - base_vs_reference["mae"],
            "fine_minus_base_rmse": fine_vs_reference["rmse"] - base_vs_reference["rmse"],
            "fine_minus_base_histogram_l1": fine_vs_reference["histogram_l1"] - base_vs_reference["histogram_l1"],
            "fine_minus_base_edge_difference": fine_vs_reference["edge_difference"] - base_vs_reference["edge_difference"],
            "fine_minus_base_ssim_like": fine_vs_reference["ssim_like"] - base_vs_reference["ssim_like"],
        },
    }


def make_board(render_root: Path, config: dict[str, Any], item_id: str, case: dict[str, Any], metrics: dict[str, Any], board_path: Path) -> None:
    from PIL import Image, ImageDraw

    target_size = (160, 160)
    label_height = 28
    headers = ["reference", "base", "fine-tuned", "base/fine diff", "ref/base diff", "ref/fine diff"]
    view_ids = list(config["view_ids"])
    board = Image.new(
        "RGB",
        (len(headers) * target_size[0], label_height + len(view_ids) * (target_size[1] + label_height)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(board)
    for col, header in enumerate(headers):
        draw.text((col * target_size[0] + 6, 8), header, fill=(0, 0, 0))
    for row_index, view_id in enumerate(view_ids):
        images, _ = load_view_images(render_root, case, item_id, view_id)
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
            board.paste(image.convert("RGB").resize(target_size, Image.Resampling.BICUBIC), (col * target_size[0], y + label_height))
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(board_path, quality=92)


def write_report(item_id: str, metrics: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Phase 2L.5B Rendered-View Report: {item_id}",
        "",
        f"eval_split: `{metrics['eval_split']}`",
        f"selected_input_view: `{metrics['selected_input_view']}`",
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


def compare_all(config: dict[str, Any], render_root: Path) -> dict[str, Any]:
    background_rgb = rgb255(list(config.get("background_color", [0.28, 0.28, 0.28])))
    results: dict[str, Any] = {}
    for item_id, case in config.get("cases", {}).items():
        eval_split = case["eval_split"]
        views = [compare_view(render_root, case, item_id, view_id, background_rgb) for view_id in config["view_ids"]]
        metrics = {
            "item_id": item_id,
            "eval_split": eval_split,
            "source_split": case.get("source_split", ""),
            "selected_input_view": case.get("selected_input_view", ""),
            "primary_front_views": case.get("primary_front_views", config.get("primary_front_views", ["004", "005"])),
            "view_ids": list(config["view_ids"]),
            "background_rgb": background_rgb,
            "views": views,
            "aggregate": aggregate_case(views),
        }
        metrics_dir = render_root / "metrics" / eval_split
        reports_dir = render_root / "reports" / eval_split
        boards_dir = render_root / "boards" / eval_split
        metrics_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        boards_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{item_id}{OUTPUT_JSON_SUFFIX}").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        write_report(item_id, metrics, reports_dir / f"{item_id}{OUTPUT_MD_SUFFIX}")
        make_board(render_root, config, item_id, case, metrics, boards_dir / f"{item_id}{OUTPUT_BOARD_SUFFIX}")
        results[item_id] = metrics
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mini40 rendered views against references.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_config = load_json(args.config)
    render_root = resolve_project_path(eval_config["output_root"]) / "render_eval"
    render_config = load_json(render_root / "render_eval_cases.json")
    results = compare_all(render_config, render_root)
    print("Phase 2L.5B mini40 rendered-view comparison")
    print(f"  cases: {len(results)}")
    print(f"  render_root: {render_root}")
    print("PHASE2L5B_MINI40_RENDERED_VIEW_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
