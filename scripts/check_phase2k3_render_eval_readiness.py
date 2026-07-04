#!/usr/bin/env python3
"""Check Phase 2K.3 rendered-view evaluation readiness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]
REQUIRED_SCRIPTS = [
    "render_phase2k3_glb_views_blender.py",
    "compare_phase2k3_rendered_views.py",
    "aggregate_phase2k3_rendered_metrics.py",
]


def resolve_project_path(path_text: str, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


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


def load_config(path: Path, errors: list[str]) -> dict[str, Any]:
    result = check_file(path, "cases config", errors)
    if not result["exists"]:
        return {}
    try:
        data = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"cases config is not valid JSON: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append("cases config must contain a JSON object")
        return {}
    return data


def available_blender() -> dict[str, Any]:
    env_value = os.environ.get("BLENDER_BIN", "").strip()
    if env_value:
        path = Path(env_value).expanduser().resolve()
        return {
            "source": "BLENDER_BIN",
            "path": str(path),
            "available": path.is_file() and os.access(path, os.X_OK),
        }
    found = shutil.which("blender")
    return {
        "source": "PATH",
        "path": found or "",
        "available": bool(found),
    }


def rendered_pngs(output_root: Path, asset_id: str) -> list[Path]:
    render_root = output_root / "renders" / asset_id
    if not render_root.exists():
        return []
    return sorted(path for path in render_root.rglob("*.png") if path.is_file())


def check_readiness(cases_config: Path, output_root_arg: Path) -> dict[str, Any]:
    errors: list[str] = []
    config = load_config(cases_config, errors)
    output_root = output_root_arg.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    else:
        output_root = output_root.resolve()

    view_ids = list(config.get("view_ids", []))
    cases = config.get("cases", {}) if isinstance(config.get("cases", {}), dict) else {}
    asset_ids = sorted(cases.keys())
    if asset_ids != sorted(REQUIRED_ASSET_IDS):
        errors.append(f"asset ids {asset_ids} != expected {sorted(REQUIRED_ASSET_IDS)}")

    reference_template = str(config.get("reference_image_path_template", ""))
    per_asset: dict[str, Any] = {}
    for asset_id in asset_ids:
        case = cases.get(asset_id, {})
        base_glb = resolve_project_path(str(case.get("base_glb", "")))
        finetuned_glb = resolve_project_path(str(case.get("finetuned_glb", "")))
        refs = {
            view_id: check_file(
                resolve_project_path(
                    reference_template.format(asset_id=asset_id, view_id=view_id)
                ),
                f"{asset_id} reference view {view_id}",
                errors,
            )
            for view_id in view_ids
        }
        existing_renders = rendered_pngs(output_root, asset_id)
        if existing_renders:
            errors.append(
                f"render output dir already contains PNGs for {asset_id}: "
                + ", ".join(str(path) for path in existing_renders[:10])
            )
        per_asset[asset_id] = {
            "base_glb": check_file(base_glb, f"{asset_id} base GLB", errors),
            "finetuned_glb": check_file(
                finetuned_glb,
                f"{asset_id} fine-tuned GLB",
                errors,
            ),
            "reference_views": refs,
            "existing_render_pngs": [str(path) for path in existing_renders],
        }

    scripts = {
        filename: check_file(
            PROJECT_ROOT / "scripts" / filename,
            f"required script {filename}",
            errors,
        )
        for filename in REQUIRED_SCRIPTS
    }
    blender = available_blender()
    if not blender["available"]:
        errors.append("Blender executable not available via BLENDER_BIN or PATH")

    report = {
        "cases_config": str(cases_config.expanduser().resolve()),
        "output_root": str(output_root),
        "view_ids": view_ids,
        "asset_ids": asset_ids,
        "expected_asset_ids": REQUIRED_ASSET_IDS,
        "output_parent": check_parent(output_root, "output-root", errors),
        "per_asset": per_asset,
        "scripts": scripts,
        "blender": blender,
    }
    report["ok"] = not errors
    report["errors"] = errors
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2K.3 rendered-view evaluation readiness")
    print(f"  cases_config: {report['cases_config']}")
    print(f"  output_root: {report['output_root']}")
    print(f"  view_ids: {report['view_ids']}")
    print(f"  asset_ids: {report['asset_ids']}")
    print(f"  blender_available: {report['blender']['available']}")
    print(f"  blender_source: {report['blender']['source']}")
    print(f"  blender_path: {report['blender']['path']}")
    for filename, item in report["scripts"].items():
        print(f"  script {filename}: exists={item['exists']} path={item['path']}")
    for asset_id, item in report["per_asset"].items():
        print(f"  asset {asset_id}:")
        print(f"    base_glb: exists={item['base_glb']['exists']} path={item['base_glb']['path']}")
        print(
            "    finetuned_glb: "
            f"exists={item['finetuned_glb']['exists']} path={item['finetuned_glb']['path']}"
        )
        print(f"    reference_views: {len(item['reference_views'])}")
        print(f"    existing_render_pngs: {len(item['existing_render_pngs'])}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2K3_RENDER_EVAL_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase 2K.3 rendered-view evaluation readiness."
    )
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.cases_config, args.output_root)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
