#!/usr/bin/env python3
"""Check Phase 2G.6 base-vs-fine-tuned comparison readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


BASE_REQUIRED = {
    "base_obj": "base_textured_mesh.obj",
    "base_glb": "base_textured_mesh.glb",
    "base_albedo": "base_textured_mesh.jpg",
    "base_metallic": "base_textured_mesh_metallic.jpg",
    "base_roughness": "base_textured_mesh_roughness.jpg",
}
FINETUNED_REQUIRED = {
    "finetuned_obj": "finetuned_textured_mesh.obj",
    "finetuned_glb": "finetuned_textured_mesh.glb",
    "finetuned_albedo": "finetuned_textured_mesh.jpg",
    "finetuned_metallic": "finetuned_textured_mesh_metallic.jpg",
    "finetuned_roughness": "finetuned_textured_mesh_roughness.jpg",
}


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


def check_readiness(case_dir: Path, base_dir: Path, finetuned_dir: Path, output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    resolved_case = case_dir.expanduser().resolve()
    resolved_base = base_dir.expanduser().resolve()
    resolved_finetuned = finetuned_dir.expanduser().resolve()

    checks: dict[str, Any] = {
        "case_dir": check_dir(resolved_case, "case-dir", errors),
        "base_dir": check_dir(resolved_base, "base-dir", errors),
        "finetuned_dir": check_dir(resolved_finetuned, "finetuned-dir", errors),
        "reference_image": check_file(resolved_case / "input" / "image.png", "case input image", errors),
    }
    for label, filename in BASE_REQUIRED.items():
        checks[label] = check_file(resolved_base / filename, label, errors)
    for label, filename in FINETUNED_REQUIRED.items():
        checks[label] = check_file(resolved_finetuned / filename, label, errors)

    output = check_output_dir(output_dir, errors)
    return {
        "case_dir": str(resolved_case),
        "base_dir": str(resolved_base),
        "finetuned_dir": str(resolved_finetuned),
        "output_dir": output,
        "checks": checks,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2G.6 comparison readiness")
    print(f"  case_dir: {report['case_dir']}")
    print(f"  base_dir: {report['base_dir']}")
    print(f"  finetuned_dir: {report['finetuned_dir']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    for name, item in report["checks"].items():
        if "size_bytes" in item:
            print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
        else:
            print(f"  {name}: exists={item['exists']} path={item['path']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2G6_COMPARE_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2G.6 comparison readiness.")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--finetuned-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.case_dir, args.base_dir, args.finetuned_dir, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
