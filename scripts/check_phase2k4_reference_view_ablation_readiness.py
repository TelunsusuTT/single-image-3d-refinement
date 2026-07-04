#!/usr/bin/env python3
"""Check Phase 2K.4 reference-view ablation readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]
REQUIRED_INPUT_VIEWS = ["004", "005"]
CHECKPOINT_TOKEN = "pilot_v1_truepbr_200_lr1e6"
FORBIDDEN_CHECKPOINT_TOKENS = [
    "pilot_v1_overfit_500",
    "pilot_v1_conservative_50_lr1e6",
    "pilot_v1_truepbr_50_lr1e6",
]
OUTPUT_SUFFIXES = {".glb", ".obj"}
PROJECT_SCRIPTS = [
    "run_phase2g_paint_infer.py",
    "make_phase2g6_texture_comparison.py",
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


def existing_mesh_outputs(output_dir: Path) -> list[Path]:
    resolved = output_dir.expanduser().resolve()
    if not resolved.exists():
        return []
    return sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in OUTPUT_SUFFIXES
    )


def load_cases_config(path: Path, errors: list[str]) -> dict[str, Any]:
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


def check_output_dirs(output_root: Path, asset_id: str, view_id: str, errors: list[str]) -> dict[str, Any]:
    base_dir = output_root / "infer" / "base" / asset_id / f"input_{view_id}"
    finetuned_dir = output_root / "infer" / "finetuned" / asset_id / f"input_{view_id}"
    base_outputs = existing_mesh_outputs(base_dir)
    finetuned_outputs = existing_mesh_outputs(finetuned_dir)
    if base_outputs:
        errors.append(f"base output dir already contains mesh outputs for {asset_id} input {view_id}")
    if finetuned_outputs:
        errors.append(f"fine-tuned output dir already contains mesh outputs for {asset_id} input {view_id}")
    return {
        "base_dir": str(base_dir),
        "finetuned_dir": str(finetuned_dir),
        "base_existing_mesh_outputs": [str(path) for path in base_outputs],
        "finetuned_existing_mesh_outputs": [str(path) for path in finetuned_outputs],
    }


def check_readiness(cases_config: Path, hypaint: Path) -> dict[str, Any]:
    errors: list[str] = []
    config = load_cases_config(cases_config, errors)
    asset_ids = list(config.get("asset_ids", []))
    input_views = list(config.get("input_views", []))
    primary_eval_views = list(config.get("primary_eval_views", []))
    if asset_ids != REQUIRED_ASSET_IDS:
        errors.append(f"asset ids {asset_ids} != expected {REQUIRED_ASSET_IDS}")
    if input_views != REQUIRED_INPUT_VIEWS:
        errors.append(f"input views {input_views} != expected {REQUIRED_INPUT_VIEWS}")
    if primary_eval_views != REQUIRED_INPUT_VIEWS:
        errors.append(f"primary eval views {primary_eval_views} != expected {REQUIRED_INPUT_VIEWS}")

    checkpoint = resolve_project_path(config.get("checkpoint", "")) if config else Path("")
    checkpoint_text = str(checkpoint)
    checkpoint_has_token = CHECKPOINT_TOKEN in checkpoint_text
    forbidden_checkpoint_hits = [
        token for token in FORBIDDEN_CHECKPOINT_TOKENS if token in checkpoint_text
    ]
    if config and not checkpoint_has_token:
        errors.append(f"checkpoint path does not contain {CHECKPOINT_TOKEN}")
    for token in forbidden_checkpoint_hits:
        errors.append(f"checkpoint path contains forbidden token: {token}")

    output_root = resolve_project_path(config.get("output_root", "")) if config else Path("")
    resolved_hypaint = hypaint.expanduser().resolve()
    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")

    checks: dict[str, Any] = {
        "checkpoint": check_file(checkpoint, "checkpoint", errors),
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
    for filename in PROJECT_SCRIPTS:
        checks[f"script_{filename}"] = check_file(
            PROJECT_ROOT / "scripts" / filename,
            f"project script {filename}",
            errors,
        )

    assets = config.get("assets", {}) if isinstance(config.get("assets", {}), dict) else {}
    per_asset: dict[str, Any] = {}
    for asset_id in asset_ids:
        asset = assets.get(asset_id, {})
        mesh = resolve_project_path(str(asset.get("mesh_path", "")))
        template = str(asset.get("reference_image_path_template", ""))
        view_checks = {}
        for view_id in input_views:
            view_checks[view_id] = {
                "reference_image": check_file(
                    resolve_project_path(template.format(view_id=view_id)),
                    f"{asset_id} render_cond {view_id}",
                    errors,
                ),
                "outputs": check_output_dirs(output_root, asset_id, view_id, errors),
            }
        per_asset[asset_id] = {
            "mesh": check_file(mesh, f"{asset_id} mesh", errors),
            "views": view_checks,
        }

    outputs = {
        "output_root": check_parent(output_root, "output-root", errors),
    }
    return {
        "cases_config": str(cases_config.expanduser().resolve()),
        "asset_ids": asset_ids,
        "input_views": input_views,
        "primary_eval_views": primary_eval_views,
        "checkpoint": str(checkpoint),
        "checkpoint_has_token": checkpoint_has_token,
        "forbidden_checkpoint_hits": forbidden_checkpoint_hits,
        "hypaint": str(resolved_hypaint),
        "output_root": str(output_root),
        "checks": checks,
        "per_asset": per_asset,
        "outputs": outputs,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2K.4 reference-view ablation readiness")
    print(f"  cases_config: {report['cases_config']}")
    print(f"  asset_ids: {report['asset_ids']}")
    print(f"  input_views: {report['input_views']}")
    print(f"  primary_eval_views: {report['primary_eval_views']}")
    print(f"  checkpoint: {report['checkpoint']}")
    print(f"  checkpoint_has_token: {report['checkpoint_has_token']}")
    print(f"  forbidden_checkpoint_hits: {report['forbidden_checkpoint_hits']}")
    print(f"  hypaint: {report['hypaint']}")
    print(f"  output_root: {report['output_root']}")
    for name, item in report["checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    for asset_id, item in report["per_asset"].items():
        print(f"  asset {asset_id}: mesh_exists={item['mesh']['exists']} path={item['mesh']['path']}")
        for view_id, view_item in item["views"].items():
            print(
                f"    view {view_id}: ref_exists={view_item['reference_image']['exists']} "
                f"base_existing={len(view_item['outputs']['base_existing_mesh_outputs'])} "
                f"fine_existing={len(view_item['outputs']['finetuned_existing_mesh_outputs'])}"
            )
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2K4_REFERENCE_VIEW_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase 2K.4 reference-view ablation readiness."
    )
    parser.add_argument("--cases-config", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.cases_config, args.hypaint)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
