#!/usr/bin/env python3
"""Check Phase 2G.4 load-only readiness without importing torch or Hunyuan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


REQUIRED_RECOMMENDATION_SNIPPETS = {
    "recommendation_status": "recommendation status: `RECOMMENDED`",
    "recommended_transform": "recommended transform: `strip:unet.`",
    "recommended_target_path": "recommended target path: `paint_pipeline.models['multiview_model'].pipeline.unet`",
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


def candidate_keyspace_paths(output_dir: Path) -> list[Path]:
    resolved = output_dir.expanduser().resolve()
    candidates = [resolved / "keyspace_compare_v2.md"]
    if resolved.parent.name == "load_only":
        candidates.append(resolved.parent.parent / "key_inspection" / resolved.name / "keyspace_compare_v2.md")
    candidates.append(Path.cwd() / "outputs" / "phase2g" / "key_inspection" / resolved.name / "keyspace_compare_v2.md")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def check_keyspace_report(output_dir: Path, errors: list[str]) -> dict[str, Any]:
    checked_paths = candidate_keyspace_paths(output_dir)
    existing = next((path for path in checked_paths if path.is_file()), None)
    if existing is None:
        errors.append(
            "keyspace_compare_v2.md missing; checked: "
            + ", ".join(str(path) for path in checked_paths)
        )
        return {
            "path": "",
            "exists": False,
            "checked_paths": [str(path) for path in checked_paths],
            "required_snippets": REQUIRED_RECOMMENDATION_SNIPPETS,
            "snippet_checks": {key: False for key in REQUIRED_RECOMMENDATION_SNIPPETS},
        }

    text = existing.read_text(encoding="utf-8", errors="replace")
    snippet_checks = {
        key: snippet in text
        for key, snippet in REQUIRED_RECOMMENDATION_SNIPPETS.items()
    }
    for key, ok in snippet_checks.items():
        if not ok:
            errors.append(f"keyspace report missing required recommendation snippet: {key}")
    return {
        "path": str(existing),
        "exists": True,
        "checked_paths": [str(path) for path in checked_paths],
        "required_snippets": REQUIRED_RECOMMENDATION_SNIPPETS,
        "snippet_checks": snippet_checks,
    }


def check_readiness(checkpoint: Path, hypaint: Path, output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    resolved_checkpoint = checkpoint.expanduser().resolve()
    resolved_hypaint = hypaint.expanduser().resolve()
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")

    checks = {
        "checkpoint": check_file(resolved_checkpoint, "checkpoint", errors),
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
    keyspace_report = check_keyspace_report(output_dir, errors)
    return {
        "checkpoint": str(resolved_checkpoint),
        "hypaint": str(resolved_hypaint),
        "output_dir": output,
        "checks": checks,
        "keyspace_report": keyspace_report,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2G.4 load-only readiness")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  hypaint: {report['hypaint']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    for name, item in report["checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    keyspace = report["keyspace_report"]
    print(f"  keyspace_compare_v2_md: exists={keyspace['exists']} path={keyspace['path']}")
    for name, ok in keyspace["snippet_checks"].items():
        print(f"  keyspace_{name}: {ok}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2G4_LOAD_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2G.4 load-only readiness.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.checkpoint, args.hypaint, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
