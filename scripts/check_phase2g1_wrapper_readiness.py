#!/usr/bin/env python3
"""Check Phase 2G.1 wrapper readiness without importing Hunyuan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    size = resolved.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {resolved}")
    return {"path": str(resolved), "exists": exists, "size_bytes": size}


def check_readiness(case_dir: Path, checkpoint: Path, wrapper: Path, hypaint: Path) -> dict[str, Any]:
    errors: list[str] = []
    resolved_case = case_dir.expanduser().resolve()
    resolved_hypaint = hypaint.expanduser().resolve()
    if not resolved_case.is_dir():
        errors.append(f"case-dir missing: {resolved_case}")
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")

    checks = {
        "input_mesh": check_file(resolved_case / "input" / "mesh.glb", "input mesh", errors),
        "input_image": check_file(resolved_case / "input" / "image.png", "input image", errors),
        "checkpoint": check_file(checkpoint, "checkpoint", errors),
        "wrapper": check_file(wrapper, "wrapper", errors),
        "demo_py": check_file(resolved_hypaint / "demo.py", "official demo.py", errors),
        "textureGenPipeline_py": check_file(
            resolved_hypaint / "textureGenPipeline.py",
            "official textureGenPipeline.py",
            errors,
        ),
        "hunyuanpaintpbr_pipeline_py": check_file(
            resolved_hypaint / "hunyuanpaintpbr" / "pipeline.py",
            "official hunyuanpaintpbr/pipeline.py",
            errors,
        ),
    }
    return {
        "case_dir": str(resolved_case),
        "hypaint": str(resolved_hypaint),
        "checks": checks,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2G.1 wrapper readiness")
    print(f"  case_dir: {report['case_dir']}")
    print(f"  hypaint: {report['hypaint']}")
    for name, item in report["checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2G1_WRAPPER_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2G.1 wrapper readiness.")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.case_dir, args.checkpoint, args.wrapper, args.hypaint)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
