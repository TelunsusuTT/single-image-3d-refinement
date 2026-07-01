#!/usr/bin/env python3
"""Check Phase 2H.1 conservative checkpoint evaluation readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSERVATIVE_TOKEN = "pilot_v1_conservative_50_lr1e6"
COLLAPSED_TOKEN = "pilot_v1_overfit_500"
OUTPUT_SUFFIXES = {".glb", ".obj"}
BASE_OUTPUT_FILES = [
    "base_textured_mesh.obj",
    "base_textured_mesh.glb",
    "base_textured_mesh.jpg",
    "base_textured_mesh_metallic.jpg",
    "base_textured_mesh_roughness.jpg",
]
PROJECT_SCRIPTS = [
    "load_phase2g4_finetuned_checkpoint_only.py",
    "run_phase2g_paint_infer.py",
    "make_phase2g6_texture_comparison.py",
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


def check_parent(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    parent = resolved.parent
    ok = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        ok = parent.is_dir()
    except OSError as exc:
        errors.append(f"{label} parent cannot be created: {parent}: {exc}")
    return {"path": str(resolved), "parent": str(parent), "parent_exists_or_created": ok}


def existing_mesh_outputs(output_dir: Path) -> list[Path]:
    resolved = output_dir.expanduser().resolve()
    if not resolved.exists():
        return []
    return sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in OUTPUT_SUFFIXES
    )


def check_infer_output_dir(output_dir: Path, errors: list[str]) -> dict[str, Any]:
    result = check_parent(output_dir, "infer-output-dir", errors)
    outputs = existing_mesh_outputs(output_dir)
    if outputs:
        errors.append(
            "infer-output-dir already contains mesh outputs: "
            + ", ".join(str(path) for path in outputs)
        )
    result["existing_mesh_outputs"] = [str(path) for path in outputs]
    return result


def check_readiness(
    case_dir: Path,
    base_dir: Path,
    checkpoint: Path,
    hypaint: Path,
    load_output_dir: Path,
    infer_output_dir: Path,
    compare_output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    scripts_dir = PROJECT_ROOT / "scripts"
    resolved_case = case_dir.expanduser().resolve()
    resolved_base = base_dir.expanduser().resolve()
    resolved_checkpoint = checkpoint.expanduser().resolve()
    resolved_hypaint = hypaint.expanduser().resolve()

    if not resolved_case.is_dir():
        errors.append(f"case-dir missing: {resolved_case}")
    if not resolved_base.is_dir():
        errors.append(f"base-dir missing: {resolved_base}")
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")

    checkpoint_text = str(resolved_checkpoint)
    checkpoint_has_conservative_token = CONSERVATIVE_TOKEN in checkpoint_text
    checkpoint_has_collapsed_token = COLLAPSED_TOKEN in checkpoint_text
    if not checkpoint_has_conservative_token:
        errors.append(f"checkpoint path does not contain {CONSERVATIVE_TOKEN}")
    if checkpoint_has_collapsed_token:
        errors.append(f"checkpoint path contains forbidden {COLLAPSED_TOKEN}")

    checks: dict[str, Any] = {
        "input_mesh": check_file(resolved_case / "input" / "mesh.glb", "input mesh", errors),
        "input_image": check_file(resolved_case / "input" / "image.png", "input image", errors),
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
    for filename in BASE_OUTPUT_FILES:
        checks[f"base_{filename}"] = check_file(
            resolved_base / filename,
            f"base output {filename}",
            errors,
        )
    for filename in PROJECT_SCRIPTS:
        checks[f"script_{filename}"] = check_file(
            scripts_dir / filename,
            f"project script {filename}",
            errors,
        )

    outputs = {
        "load_output_dir": check_parent(load_output_dir, "load-output-dir", errors),
        "infer_output_dir": check_infer_output_dir(infer_output_dir, errors),
        "compare_output_dir": check_parent(compare_output_dir, "compare-output-dir", errors),
    }

    return {
        "case_dir": str(resolved_case),
        "base_dir": str(resolved_base),
        "checkpoint": str(resolved_checkpoint),
        "hypaint": str(resolved_hypaint),
        "checkpoint_has_conservative_token": checkpoint_has_conservative_token,
        "checkpoint_has_collapsed_token": checkpoint_has_collapsed_token,
        "checks": checks,
        "outputs": outputs,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2H.1 conservative evaluation readiness")
    print(f"  case_dir: {report['case_dir']}")
    print(f"  base_dir: {report['base_dir']}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  hypaint: {report['hypaint']}")
    print(f"  checkpoint_has_conservative_token: {report['checkpoint_has_conservative_token']}")
    print(f"  checkpoint_has_collapsed_token: {report['checkpoint_has_collapsed_token']}")
    for name, item in report["checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    for name, item in report["outputs"].items():
        print(f"  {name}: {item['path']}")
        print(f"      parent_exists_or_created: {item['parent_exists_or_created']}")
        if "existing_mesh_outputs" in item:
            print(f"      existing_mesh_outputs: {len(item['existing_mesh_outputs'])}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2H1_EVAL_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase 2H.1 conservative checkpoint evaluation readiness."
    )
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--load-output-dir", required=True, type=Path)
    parser.add_argument("--infer-output-dir", required=True, type=Path)
    parser.add_argument("--compare-output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(
        args.case_dir,
        args.base_dir,
        args.checkpoint,
        args.hypaint,
        args.load_output_dir,
        args.infer_output_dir,
        args.compare_output_dir,
    )
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
