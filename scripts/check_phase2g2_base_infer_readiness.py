#!/usr/bin/env python3
"""Check Phase 2G.2 base inference smoke readiness without importing Hunyuan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


OUTPUT_SUFFIXES = {".glb", ".obj"}


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    size = resolved.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {resolved}")
    return {"path": str(resolved), "exists": exists, "size_bytes": size}


def existing_mesh_outputs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in OUTPUT_SUFFIXES
    )


def check_output_dir(output_dir: Path, errors: list[str]) -> dict[str, Any]:
    resolved = output_dir.expanduser().resolve()
    parent = resolved.parent
    parent_created_or_exists = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_created_or_exists = parent.is_dir()
    except OSError as exc:
        errors.append(f"output-dir parent cannot be created: {parent}: {exc}")
    outputs = existing_mesh_outputs(resolved)
    if outputs:
        errors.append(
            "output-dir already contains mesh outputs: "
            + ", ".join(str(path) for path in outputs)
        )
    return {
        "path": str(resolved),
        "parent": str(parent),
        "parent_exists_or_created": parent_created_or_exists,
        "existing_outputs": [str(path) for path in outputs],
    }


def check_readiness(case_dir: Path, wrapper: Path, hypaint: Path, output_dir: Path) -> dict[str, Any]:
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
        "wrapper": check_file(wrapper, "wrapper", errors),
        "demo_py": check_file(resolved_hypaint / "demo.py", "official demo.py", errors),
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
    output = check_output_dir(output_dir, errors)
    return {
        "case_dir": str(resolved_case),
        "hypaint": str(resolved_hypaint),
        "output_dir": output,
        "checks": checks,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2G.2 base inference readiness")
    print(f"  case_dir: {report['case_dir']}")
    print(f"  hypaint: {report['hypaint']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    print(f"  existing_mesh_outputs: {len(output['existing_outputs'])}")
    for name, item in report["checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2G2_BASE_INFER_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2G.2 base inference readiness.")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.case_dir, args.wrapper, args.hypaint, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
