#!/usr/bin/env python3
"""Check Phase 2G.5 fine-tuned inference readiness without torch/Hunyuan imports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


OUTPUT_SUFFIXES = {".glb", ".obj"}
KEYSPACE_SNIPPETS = {
    "recommendation_status": "recommendation status: `RECOMMENDED`",
    "recommended_transform": "recommended transform: `strip:unet.`",
    "recommended_target_path": "recommended target path: `paint_pipeline.models['multiview_model'].pipeline.unet`",
}
LOAD_ONLY_SNIPPETS = {
    "strict_load_state_dict": "strict_load_state_dict: `OK`",
    "ok_token": "PHASE2G4_CHECKPOINT_LOAD_ONLY_OK",
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
    parent_ok = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_ok = parent.is_dir()
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
        "parent_exists_or_created": parent_ok,
        "existing_outputs": [str(path) for path in outputs],
    }


def project_root_from_output(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    parts = resolved.parts
    for index in range(len(parts) - 1):
        if parts[index] == "outputs" and index + 1 < len(parts) and parts[index + 1] == "phase2g":
            return Path(*parts[:index])
    return Path.cwd()


def phase_name_from_checkpoint(checkpoint: Path) -> str:
    parent_name = checkpoint.expanduser().resolve().parent.name
    return parent_name or "pilot_v1_overfit_500"


def candidate_report_paths(output_dir: Path, checkpoint: Path, kind: str) -> list[Path]:
    project_root = project_root_from_output(output_dir)
    phase_name = phase_name_from_checkpoint(checkpoint)
    if kind == "keyspace":
        rel = Path("outputs") / "phase2g" / "key_inspection" / phase_name / "keyspace_compare_v2.md"
    elif kind == "load_only":
        rel = Path("outputs") / "phase2g" / "load_only" / phase_name / "load_only_report.md"
    else:
        raise ValueError(kind)
    candidates = [
        project_root / rel,
        Path.cwd() / rel,
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def check_text_report(
    output_dir: Path,
    checkpoint: Path,
    kind: str,
    label: str,
    snippets: dict[str, str],
    errors: list[str],
) -> dict[str, Any]:
    checked_paths = candidate_report_paths(output_dir, checkpoint, kind)
    existing = next((path for path in checked_paths if path.is_file()), None)
    if existing is None:
        errors.append(f"{label} missing; checked: " + ", ".join(str(path) for path in checked_paths))
        return {
            "path": "",
            "exists": False,
            "checked_paths": [str(path) for path in checked_paths],
            "snippet_checks": {key: False for key in snippets},
        }
    text = existing.read_text(encoding="utf-8", errors="replace")
    snippet_checks = {key: snippet in text for key, snippet in snippets.items()}
    for key, ok in snippet_checks.items():
        if not ok:
            errors.append(f"{label} missing required snippet: {key}")
    return {
        "path": str(existing),
        "exists": True,
        "checked_paths": [str(path) for path in checked_paths],
        "snippet_checks": snippet_checks,
    }


def check_readiness(
    case_dir: Path,
    checkpoint: Path,
    wrapper: Path,
    hypaint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    resolved_case = case_dir.expanduser().resolve()
    resolved_checkpoint = checkpoint.expanduser().resolve()
    resolved_hypaint = hypaint.expanduser().resolve()
    if not resolved_case.is_dir():
        errors.append(f"case-dir missing: {resolved_case}")
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")

    checks = {
        "input_mesh": check_file(resolved_case / "input" / "mesh.glb", "input mesh", errors),
        "input_image": check_file(resolved_case / "input" / "image.png", "input image", errors),
        "checkpoint": check_file(resolved_checkpoint, "checkpoint", errors),
        "wrapper": check_file(wrapper, "wrapper", errors),
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
    keyspace = check_text_report(output_dir, checkpoint, "keyspace", "keyspace_compare_v2.md", KEYSPACE_SNIPPETS, errors)
    load_only = check_text_report(output_dir, checkpoint, "load_only", "load_only_report.md", LOAD_ONLY_SNIPPETS, errors)
    return {
        "case_dir": str(resolved_case),
        "checkpoint": str(resolved_checkpoint),
        "hypaint": str(resolved_hypaint),
        "output_dir": output,
        "checks": checks,
        "keyspace_report": keyspace,
        "load_only_report": load_only,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2G.5 fine-tuned inference readiness")
    print(f"  case_dir: {report['case_dir']}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  hypaint: {report['hypaint']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    print(f"  existing_mesh_outputs: {len(output['existing_outputs'])}")
    for name, item in report["checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    for report_name in ("keyspace_report", "load_only_report"):
        item = report[report_name]
        print(f"  {report_name}: exists={item['exists']} path={item['path']}")
        for name, ok in item["snippet_checks"].items():
            print(f"  {report_name}_{name}: {ok}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2G5_FINETUNED_INFER_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2G.5 fine-tuned inference readiness.")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.case_dir, args.checkpoint, args.wrapper, args.hypaint, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
