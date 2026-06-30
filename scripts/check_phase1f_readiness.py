#!/usr/bin/env python3
"""Check that the Phase 1F one-asset smoke config points at valid local data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_single_sample_path(examples_json: Path, errors: list[str]) -> Path | None:
    if not examples_json.is_file():
        errors.append(f"examples JSON missing: {examples_json}")
        return None

    try:
        data = json.loads(examples_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"examples JSON is not valid JSON: {exc}")
        return None

    if not isinstance(data, list):
        errors.append("examples JSON must contain a list")
        return None
    if len(data) != 1:
        errors.append(f"examples JSON must contain exactly one sample path, found {len(data)}")
        return None
    if not isinstance(data[0], str):
        errors.append("examples JSON sample path must be a string")
        return None

    sample_dir = Path(data[0]).expanduser()
    if not sample_dir.is_absolute():
        errors.append(f"sample path is not absolute: {sample_dir}")
    return sample_dir


def check_readiness(examples_json: Path, config: Path) -> dict[str, Any]:
    errors: list[str] = []
    resolved_examples_json = examples_json.expanduser().resolve()
    resolved_config = config.expanduser().resolve()
    sample_dir = load_single_sample_path(resolved_examples_json, errors)

    render_tex = sample_dir / "render_tex" if sample_dir is not None else None
    render_cond = sample_dir / "render_cond" if sample_dir is not None else None
    transforms = render_tex / "transforms.json" if render_tex is not None else None

    if sample_dir is not None and not sample_dir.is_dir():
        errors.append(f"sample directory missing: {sample_dir}")
    if render_tex is not None and not render_tex.is_dir():
        errors.append(f"render_tex missing: {render_tex}")
    if render_cond is not None and not render_cond.is_dir():
        errors.append(f"render_cond missing: {render_cond}")
    if transforms is not None and not transforms.is_file():
        errors.append(f"render_tex/transforms.json missing: {transforms}")

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
        "sample_dir": str(sample_dir) if sample_dir is not None else "",
        "sample_dir_exists": sample_dir.is_dir() if sample_dir is not None else False,
        "render_tex_exists": render_tex.is_dir() if render_tex is not None else False,
        "render_cond_exists": render_cond.is_dir() if render_cond is not None else False,
        "transforms_json_exists": transforms.is_file() if transforms is not None else False,
        "config_exists": resolved_config.is_file(),
        "config_contains_examples_json": config_contains_examples_json,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 1F readiness")
    print(f"  examples_json: {report['examples_json']}")
    print(f"  config: {report['config']}")
    print(f"  sample_dir: {report['sample_dir'] or 'missing'}")
    print(f"  sample_dir_exists: {report['sample_dir_exists']}")
    print(f"  render_tex_exists: {report['render_tex_exists']}")
    print(f"  render_cond_exists: {report['render_cond_exists']}")
    print(f"  transforms_json_exists: {report['transforms_json_exists']}")
    print(f"  config_exists: {report['config_exists']}")
    print(f"  config_contains_examples_json: {report['config_contains_examples_json']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("DATASET_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 1F smoke readiness.")
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.examples_json, args.config)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
