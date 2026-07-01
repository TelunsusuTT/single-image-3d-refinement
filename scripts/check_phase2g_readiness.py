#!/usr/bin/env python3
"""Check Phase 2G inference case readiness without importing Hunyuan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def check_nonzero_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    ok = exists and size > 0
    if not exists:
        errors.append(f"{label} missing: {path}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {path}")
    return {"path": str(path), "exists": exists, "size_bytes": size, "ok": ok}


def check_readiness(case_dir: Path, checkpoint: Path, hypaint: Path) -> dict[str, Any]:
    errors: list[str] = []
    resolved_case_dir = case_dir.expanduser().resolve()
    resolved_checkpoint = checkpoint.expanduser().resolve()
    resolved_hypaint = hypaint.expanduser().resolve()

    if not resolved_case_dir.is_dir():
        errors.append(f"case_dir missing: {resolved_case_dir}")
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")

    mesh = check_nonzero_file(resolved_case_dir / "input" / "mesh.glb", "input mesh", errors)
    image = check_nonzero_file(resolved_case_dir / "input" / "image.png", "input image", errors)
    ckpt = check_nonzero_file(resolved_checkpoint, "checkpoint", errors)
    demo = check_nonzero_file(resolved_hypaint / "demo.py", "demo.py", errors)
    train = check_nonzero_file(resolved_hypaint / "train.py", "train.py", errors)

    return {
        "case_dir": str(resolved_case_dir),
        "checkpoint": str(resolved_checkpoint),
        "hypaint": str(resolved_hypaint),
        "mesh": mesh,
        "image": image,
        "checkpoint_file": ckpt,
        "demo_py": demo,
        "train_py": train,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2G readiness")
    print(f"  case_dir: {report['case_dir']}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  hypaint: {report['hypaint']}")
    for key in ("mesh", "image", "checkpoint_file", "demo_py", "train_py"):
        item = report[key]
        print(f"  {key}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2G_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2G readiness.")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.case_dir, args.checkpoint, args.hypaint)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
