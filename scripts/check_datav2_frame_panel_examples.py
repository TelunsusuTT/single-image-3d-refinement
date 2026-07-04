#!/usr/bin/env python3
"""Check rendered Data v2 frame-panel mini40 examples without running Hunyuan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEX_SUFFIXES = [".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"]
COND_SUFFIXES = ["_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def report_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["report_root"])


def dataset_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_dataset_root"])


def examples_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = dataset_root(config)
    return {
        "train": root / "examples_train_abs.json",
        "val": root / "examples_val_abs.json",
        "test": root / "examples_test_abs.json",
        "all": root / "examples_all_abs.json",
    }


def expected_counts(config: dict[str, Any]) -> dict[str, int]:
    split_payload = load_json(resolve_project_path(config["split_file"]))
    counts = split_payload.get("counts", {})
    return {split: int(counts.get(split, 0)) for split in ("train", "val", "test")}


def expected_files(sample_dir: Path, view_ids: list[str]) -> list[Path]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    files = [render_tex / "transforms.json"]
    for view_id in view_ids:
        files.extend(render_tex / f"{view_id}{suffix}" for suffix in TEX_SUFFIXES)
        files.extend(render_cond / f"{view_id}{suffix}" for suffix in COND_SUFFIXES)
    return files


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def check_sample(sample_dir: Path, view_ids: list[str], expected_size: int) -> dict[str, Any]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    missing = [str(path) for path in expected_files(sample_dir, view_ids) if not path.is_file()]
    if not render_tex.is_dir():
        missing.append(str(render_tex))
    if not render_cond.is_dir():
        missing.append(str(render_cond))
    bad_sizes: list[str] = []
    for path in expected_files(sample_dir, view_ids):
        if path.suffix.lower() != ".png" or not path.is_file():
            continue
        size = image_size(path)
        if size is not None and size != (expected_size, expected_size):
            bad_sizes.append(f"{path}: {size[0]}x{size[1]}")
    return {
        "sample_dir": str(sample_dir),
        "render_tex_exists": render_tex.is_dir(),
        "render_cond_exists": render_cond.is_dir(),
        "missing_count": len(missing),
        "missing_files": missing,
        "bad_image_size_count": len(bad_sizes),
        "bad_image_sizes": bad_sizes,
        "status": "pass" if not missing and not bad_sizes else "fail",
    }


def check_examples(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = examples_paths(config)
    counts = expected_counts(config)
    view_ids = list(config.get("view_ids", ["000", "001", "002", "003", "004", "005"]))
    expected_size = int(config.get("render_resolution", 512))
    split_summaries: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    all_values: list[str] = []

    for split in ("train", "val", "test"):
        path = paths[split]
        if not path.is_file():
            failures.append(f"missing examples JSON: {path}")
            values: list[str] = []
        else:
            values = list(load_json(path))
        all_values.extend(values)
        if len(values) != counts[split]:
            failures.append(f"{split} count mismatch: expected {counts[split]}, found {len(values)}")
        split_summaries[split] = {"path": str(path), "expected_count": counts[split], "actual_count": len(values)}
        for sample in values:
            row = check_sample(Path(sample), view_ids, expected_size)
            row["split"] = split
            sample_rows.append(row)

    all_path = paths["all"]
    all_json_values = list(load_json(all_path)) if all_path.is_file() else []
    if not all_path.is_file():
        failures.append(f"missing examples JSON: {all_path}")
    if sorted(set(all_values)) != sorted(all_json_values):
        failures.append("examples_all_abs.json does not match train+val+test union")
    for row in sample_rows:
        if row["status"] != "pass":
            failures.append(f"{row['split']} sample failed: {row['sample_dir']}")
    payload = {
        "dataset_name": config["dataset_name"],
        "split_summaries": split_summaries,
        "all_examples_path": str(all_path),
        "all_count": len(all_json_values),
        "sample_count": len(sample_rows),
        "failure_count": len(failures),
        "failures": failures,
    }
    return payload, sample_rows


def write_reports(config: dict[str, Any], payload: dict[str, Any], sample_rows: list[dict[str, Any]]) -> None:
    out_json = report_root(config) / "check_summary.json"
    out_md = report_root(config) / "check_summary.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": payload, "samples": sample_rows}, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Data v2 Frame Panels Mini40 Check Summary",
        "",
        f"dataset: `{payload['dataset_name']}`",
        f"samples checked: `{payload['sample_count']}`",
        f"failures: `{payload['failure_count']}`",
        "",
        "| Split | Expected | Actual |",
        "|---|---:|---:|",
    ]
    for split, summary in payload["split_summaries"].items():
        lines.append(f"| `{split}` | {summary['expected_count']} | {summary['actual_count']} |")
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in payload["failures"])
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Data v2 frame-panel mini40 rendered examples.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    payload, sample_rows = check_examples(config)
    write_reports(config, payload, sample_rows)
    print("Phase 2L.3A frame-panel examples check")
    print(f"  samples checked: {payload['sample_count']}")
    print(f"  failures: {payload['failure_count']}")
    print(f"  check_summary: {report_root(config) / 'check_summary.json'}")
    if payload["failure_count"]:
        return 1
    print("PHASE2L3A_FRAME_PANEL_EXAMPLES_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
