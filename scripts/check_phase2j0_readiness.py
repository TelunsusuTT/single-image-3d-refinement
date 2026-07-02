#!/usr/bin/env python3
"""Check Phase 2J.0 planning readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


REQUIRED_HYPAINT_FILES = [
    "train.py",
    "hunyuanpaintpbr/pipeline.py",
    "hunyuanpaintpbr/unet/model.py",
    "cfgs/hunyuan-paint-pbr.yaml",
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


def check_search_root(path: Path, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.exists()
    is_dir = resolved.is_dir()
    if not exists:
        errors.append(f"search root missing: {resolved}")
    elif not is_dir:
        errors.append(f"search root is not a directory: {resolved}")
    return {"path": str(resolved), "exists": exists, "is_dir": is_dir}


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


def check_readiness(
    hypaint: Path,
    search_roots: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    resolved_hypaint = hypaint.expanduser().resolve()
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")
    hypaint_files = {
        rel: check_file(resolved_hypaint / rel, f"official {rel}", errors)
        for rel in REQUIRED_HYPAINT_FILES
    }
    roots = [check_search_root(path, errors) for path in search_roots]
    output = check_output_dir(output_dir, errors)
    return {
        "hypaint": str(resolved_hypaint),
        "hypaint_files": hypaint_files,
        "search_roots": roots,
        "output_dir": output,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2J.0 readiness")
    print(f"  hypaint: {report['hypaint']}")
    for rel, item in report["hypaint_files"].items():
        print(f"  {rel}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    for index, item in enumerate(report["search_roots"], start=1):
        print(f"  search_root[{index}]: exists={item['exists']} is_dir={item['is_dir']} path={item['path']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2J0_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2J.0 readiness.")
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--search-root", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.hypaint, args.search_root, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
