#!/usr/bin/env python3
"""Check Phase 2I pretrained initialization audit readiness."""

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
    configs: list[Path],
    checkpoints: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    resolved_hypaint = hypaint.expanduser().resolve()
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")

    official_checks = {
        "textureGenPipeline_py": check_file(
            resolved_hypaint / "textureGenPipeline.py",
            "official textureGenPipeline.py",
            errors,
        ),
        "config_yaml": check_file(
            resolved_hypaint / "cfgs" / "hunyuan-paint-pbr.yaml",
            "official cfgs/hunyuan-paint-pbr.yaml",
            errors,
        ),
        "realesrgan_ckpt": check_file(
            resolved_hypaint / "ckpt" / "RealESRGAN_x4plus.pth",
            "official ckpt/RealESRGAN_x4plus.pth",
            errors,
        ),
    }
    config_checks = [
        check_file(path, "training config", errors)
        for path in configs
    ]
    checkpoint_checks = [
        check_file(path, "checkpoint", errors)
        for path in checkpoints
    ]
    output = check_output_dir(output_dir, errors)
    return {
        "hypaint": str(resolved_hypaint),
        "official_checks": official_checks,
        "configs": config_checks,
        "checkpoints": checkpoint_checks,
        "output_dir": output,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2I pretrained initialization audit readiness")
    print(f"  hypaint: {report['hypaint']}")
    for name, item in report["official_checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    for index, item in enumerate(report["configs"], start=1):
        print(f"  config[{index}]: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    for index, item in enumerate(report["checkpoints"], start=1):
        print(f"  checkpoint[{index}]: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2I_AUDIT_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2I audit readiness.")
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--config", nargs="+", required=True, type=Path)
    parser.add_argument("--checkpoint", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.hypaint, args.config, args.checkpoint, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
