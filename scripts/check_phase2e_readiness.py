#!/usr/bin/env python3
"""Check Phase 2E pilot_v1 training smoke readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_examples(examples_json: Path, errors: list[str]) -> list[Path]:
    if not examples_json.is_file():
        errors.append(f"examples JSON missing: {examples_json}")
        return []

    try:
        data = json.loads(examples_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"examples JSON is not valid JSON: {exc}")
        return []

    if not isinstance(data, list):
        errors.append("examples JSON must contain a list")
        return []

    sample_dirs: list[Path] = []
    for index, value in enumerate(data, start=1):
        if not isinstance(value, str):
            errors.append(f"examples JSON entry {index} is not a string")
            continue
        sample_dir = Path(value).expanduser()
        if not sample_dir.is_absolute():
            errors.append(f"sample path is not absolute: {sample_dir}")
        sample_dirs.append(sample_dir)
    return sample_dirs


def check_sample(sample_dir: Path) -> dict[str, Any]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    transforms = render_tex / "transforms.json"
    errors: list[str] = []
    if not sample_dir.is_dir():
        errors.append("sample directory missing")
    if not render_tex.is_dir():
        errors.append("render_tex missing")
    if not render_cond.is_dir():
        errors.append("render_cond missing")
    if not transforms.is_file():
        errors.append("render_tex/transforms.json missing")
    return {
        "sample_dir": str(sample_dir),
        "sample_dir_exists": sample_dir.is_dir(),
        "render_tex_exists": render_tex.is_dir(),
        "render_cond_exists": render_cond.is_dir(),
        "transforms_json_exists": transforms.is_file(),
        "ok": not errors,
        "errors": errors,
    }


def check_readiness(examples_json: Path, config: Path, expected_count: int) -> dict[str, Any]:
    errors: list[str] = []
    resolved_examples_json = examples_json.expanduser().resolve()
    resolved_config = config.expanduser().resolve()
    sample_dirs = load_examples(resolved_examples_json, errors)
    sample_results = [check_sample(sample_dir) for sample_dir in sample_dirs]

    if len(sample_dirs) != expected_count:
        errors.append(f"sample count {len(sample_dirs)} != expected-count {expected_count}")
    for result in sample_results:
        errors.extend(
            f"{result['sample_dir']}: {error}"
            for error in result["errors"]
        )

    config_contains_examples_json = False
    if not resolved_config.is_file():
        errors.append(f"config missing: {resolved_config}")
    else:
        config_text = resolved_config.read_text(encoding="utf-8")
        config_contains_examples_json = str(resolved_examples_json) in config_text
        if not config_contains_examples_json:
            errors.append(
                "config does not contain examples JSON path: "
                f"{resolved_examples_json}"
            )

    return {
        "examples_json": str(resolved_examples_json),
        "config": str(resolved_config),
        "expected_count": expected_count,
        "sample_count": len(sample_dirs),
        "samples": sample_results,
        "config_exists": resolved_config.is_file(),
        "config_contains_examples_json": config_contains_examples_json,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2E readiness")
    print(f"  examples_json: {report['examples_json']}")
    print(f"  config: {report['config']}")
    print(f"  expected_count: {report['expected_count']}")
    print(f"  sample_count: {report['sample_count']}")
    print(f"  config_exists: {report['config_exists']}")
    print(f"  config_contains_examples_json: {report['config_contains_examples_json']}")
    for index, sample in enumerate(report["samples"], start=1):
        print(f"  [{index}] {sample['sample_dir']}")
        print(f"      sample_dir_exists: {sample['sample_dir_exists']}")
        print(f"      render_tex_exists: {sample['render_tex_exists']}")
        print(f"      render_cond_exists: {sample['render_cond_exists']}")
        print(f"      transforms_json_exists: {sample['transforms_json_exists']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print(f"dataset length = {report['sample_count']}")
        print("DATASET_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2E smoke readiness.")
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_count < 1:
        print("ERROR: --expected-count must be >= 1", file=sys.stderr)
        return 2
    report = check_readiness(args.examples_json, args.config, args.expected_count)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
