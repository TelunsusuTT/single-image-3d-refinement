#!/usr/bin/env python3
"""Check Phase 2J.1 true-PBR initialization probe readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


FORBIDDEN_CONFIG_TOKENS = [
    "sd2-community/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1",
    "pilot_v1_overfit_500",
]
REQUIRED_HYPAINT_FILES = [
    "train.py",
    "textureGenPipeline.py",
    "hunyuanpaintpbr/unet/model.py",
    "hunyuanpaintpbr/pipeline.py",
]
REQUIRED_PBR_ENTRIES = [
    "model_index.json",
    "unet",
    "scheduler",
    "tokenizer",
    "text_encoder",
]


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    size = resolved.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {resolved}")
    return {"path": str(resolved), "exists": exists, "size_bytes": size}


def check_dir(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_dir()
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    return {"path": str(resolved), "exists": exists}


def check_output_dir(output_dir: Path, errors: list[str]) -> dict[str, Any]:
    resolved = output_dir.expanduser().resolve()
    parent = resolved.parent
    ok = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        ok = parent.is_dir()
    except OSError as exc:
        errors.append(f"output-dir parent cannot be created: {parent}: {exc}")
    return {"path": str(resolved), "parent": str(parent), "parent_exists_or_created": ok}


def check_config(config: Path, pbr_dir: Path, errors: list[str]) -> dict[str, Any]:
    result = check_file(config, "config", errors)
    contains_pbr_dir = False
    forbidden_hits: list[str] = []
    if result["exists"]:
        text = config.expanduser().resolve().read_text(encoding="utf-8")
        contains_pbr_dir = str(pbr_dir.expanduser().resolve()) in text
        forbidden_hits = [token for token in FORBIDDEN_CONFIG_TOKENS if token in text]
        if not contains_pbr_dir:
            errors.append(f"config does not contain pbr-dir: {pbr_dir.expanduser().resolve()}")
        for token in forbidden_hits:
            errors.append(f"config contains forbidden token: {token}")
    result["contains_pbr_dir"] = contains_pbr_dir
    result["forbidden_hits"] = forbidden_hits
    return result


def check_readiness(
    hypaint: Path,
    config: Path,
    pbr_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    resolved_hypaint = hypaint.expanduser().resolve()
    resolved_pbr_dir = pbr_dir.expanduser().resolve()

    hypaint_dir = check_dir(resolved_hypaint, "hypaint", errors)
    hypaint_files = {
        rel: check_file(resolved_hypaint / rel, f"official {rel}", errors)
        for rel in REQUIRED_HYPAINT_FILES
    }
    pbr_root = check_dir(resolved_pbr_dir, "pbr-dir", errors)
    pbr_entries: dict[str, Any] = {}
    for rel in REQUIRED_PBR_ENTRIES:
        path = resolved_pbr_dir / rel
        if rel == "model_index.json":
            pbr_entries[rel] = check_file(path, f"pbr-dir/{rel}", errors)
        else:
            pbr_entries[rel] = check_dir(path, f"pbr-dir/{rel}", errors)
    config_check = check_config(config, resolved_pbr_dir, errors)
    output = check_output_dir(output_dir, errors)
    return {
        "hypaint": hypaint_dir,
        "hypaint_files": hypaint_files,
        "pbr_dir": pbr_root,
        "pbr_entries": pbr_entries,
        "config": config_check,
        "output_dir": output,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2J.1 true-PBR init readiness")
    print(f"  hypaint: exists={report['hypaint']['exists']} path={report['hypaint']['path']}")
    for rel, item in report["hypaint_files"].items():
        print(f"  {rel}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    print(f"  pbr_dir: exists={report['pbr_dir']['exists']} path={report['pbr_dir']['path']}")
    for rel, item in report["pbr_entries"].items():
        if "size_bytes" in item:
            print(f"  pbr/{rel}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
        else:
            print(f"  pbr/{rel}: exists={item['exists']} path={item['path']}")
    config = report["config"]
    print(f"  config: exists={config['exists']} size_bytes={config['size_bytes']} path={config['path']}")
    print(f"  config_contains_pbr_dir: {config['contains_pbr_dir']}")
    print(f"  config_forbidden_hits: {config['forbidden_hits']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2J1_TRUEPBR_INIT_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2J.1 init probe readiness.")
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pbr-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.hypaint, args.config, args.pbr_dir, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
