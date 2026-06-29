#!/usr/bin/env python3
"""Validate lightweight Hunyuan3D-Paint training example packaging."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAP_PATTERNS = {
    "albedo": ("*_albedo.*",),
    "metallic_roughness": ("*_mr.*", "*_metallic_roughness.*"),
    "normal": ("*_normal.*",),
    "position": ("*_pos.*", "*_position.*"),
}


def image_files(path: Path) -> list[Path]:
    """Return supported image files directly under path."""
    if not path.is_dir():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_matching_images(path: Path, patterns: tuple[str, ...]) -> int:
    count = 0
    for image_path in image_files(path):
        name = image_path.name.lower()
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            count += 1
    return count


def load_examples_json(examples_json: Path) -> list[Path]:
    if not examples_json.exists():
        raise ValueError(f"examples-json missing: {examples_json}")
    if not examples_json.is_file():
        raise ValueError(f"examples-json is not a file: {examples_json}")

    try:
        with examples_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"examples-json is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("examples-json must contain a JSON list of sample directories")
    if not data:
        raise ValueError("examples-json is empty")

    sample_dirs: list[Path] = []
    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, str):
            raise ValueError(f"examples-json entry {index} is not a string path")
        sample_dirs.append(Path(entry).expanduser())
    return sample_dirs


def check_sample(sample_dir: Path, num_view: int, strict: bool = False) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    render_cond = sample_dir / "render_cond"
    render_tex = sample_dir / "render_tex"
    transforms_candidates = [sample_dir / "transforms.json", render_tex / "transforms.json"]
    transforms_path = next((path for path in transforms_candidates if path.is_file()), None)

    sample_exists = sample_dir.is_dir()
    render_cond_exists = render_cond.is_dir()
    render_tex_exists = render_tex.is_dir()

    condition_count = len(image_files(render_cond))
    counts = {
        "condition": condition_count,
        "albedo": count_matching_images(render_tex, MAP_PATTERNS["albedo"]),
        "metallic_roughness": count_matching_images(
            render_tex, MAP_PATTERNS["metallic_roughness"]
        ),
        "normal": count_matching_images(render_tex, MAP_PATTERNS["normal"]),
        "position": count_matching_images(render_tex, MAP_PATTERNS["position"]),
    }

    errors: list[str] = []
    if not sample_exists:
        errors.append("sample directory missing")
    if not render_cond_exists:
        errors.append("render_cond/ missing")
    if not render_tex_exists:
        errors.append("render_tex/ missing")
    if counts["condition"] == 0:
        errors.append("condition image count is 0")
    for key in ("albedo", "normal", "position"):
        if counts[key] < num_view:
            errors.append(f"{key} image count {counts[key]} < num-view {num_view}")
    if strict and counts["metallic_roughness"] < num_view:
        errors.append(
            "metallic-roughness image count "
            f"{counts['metallic_roughness']} < num-view {num_view}"
        )
    if strict and transforms_path is None:
        errors.append("transforms.json missing from sample dir or render_tex/")

    return {
        "sample_dir": str(sample_dir),
        "sample_exists": sample_exists,
        "render_cond_exists": render_cond_exists,
        "render_tex_exists": render_tex_exists,
        "transforms_json_exists": transforms_path is not None,
        "transforms_json_path": str(transforms_path) if transforms_path else "",
        "counts": counts,
        "num_view": num_view,
        "strict": strict,
        "ok": not errors,
        "errors": errors,
    }


def check_examples(
    examples_json: Path, num_view: int, strict: bool = False
) -> list[dict[str, Any]]:
    return [
        check_sample(sample_dir, num_view=num_view, strict=strict)
        for sample_dir in load_examples_json(examples_json)
    ]


def print_sample_summary(result: dict[str, Any], index: int, total: int) -> None:
    counts = result["counts"]
    status = "OK" if result["ok"] else "FAIL"
    transforms = result["transforms_json_path"] or "missing"

    print(f"[{index}/{total}] {result['sample_dir']}")
    print(
        "  dirs: "
        f"sample={'OK' if result['sample_exists'] else 'MISSING'} "
        f"render_cond={'OK' if result['render_cond_exists'] else 'MISSING'} "
        f"render_tex={'OK' if result['render_tex_exists'] else 'MISSING'}"
    )
    print(f"  transforms.json: {transforms}")
    print(
        "  counts: "
        f"condition={counts['condition']} "
        f"albedo={counts['albedo']} "
        f"metallic_roughness={counts['metallic_roughness']} "
        f"normal={counts['normal']} "
        f"position={counts['position']}"
    )
    print(f"  status: {status}")
    if result["errors"]:
        print("  errors:")
        for error in result["errors"]:
            print(f"    - {error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Hunyuan3D-Paint training example directories."
    )
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--num-view", type=int, default=6)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_view < 1:
        print("ERROR: --num-view must be >= 1", file=sys.stderr)
        return 2

    try:
        results = check_examples(args.examples_json, args.num_view, strict=args.strict)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total = len(results)
    for index, result in enumerate(results, start=1):
        print_sample_summary(result, index, total)

    failed = [result for result in results if not result["ok"]]
    print(f"Checked {total} sample(s): {total - len(failed)} OK, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
