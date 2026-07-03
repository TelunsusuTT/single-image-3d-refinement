#!/usr/bin/env python3
"""Prepare Phase 2K.2 multi-case inference directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSET_IDS = ["B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def template_path(template: str, asset_id: str) -> Path:
    return resolve_project_path(template.format(asset_id=asset_id))


def link_or_copy(source: Path, target: Path) -> str:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target)
        return "symlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def prepare_case(config: dict[str, Any], asset_id: str) -> dict[str, Any]:
    output_root = resolve_project_path(config["output_root"])
    checkpoint = resolve_project_path(config["checkpoint"])
    mesh = template_path(config["raw_mesh_path_template"], asset_id)
    image = template_path(config["reference_image_path_template"], asset_id)
    case_dir = output_root / "cases" / asset_id
    input_dir = case_dir / "input"
    input_mesh = input_dir / "mesh.glb"
    input_image = input_dir / "image.png"

    mesh_mode = link_or_copy(mesh, input_mesh)
    image_mode = link_or_copy(image, input_image)
    manifest = {
        "asset_id": asset_id,
        "source_mesh_path": str(mesh),
        "source_reference_path": str(image),
        "checkpoint_path": str(checkpoint),
        "case_dir": str(case_dir),
        "input_mesh_path": str(input_mesh),
        "input_image_path": str(input_image),
        "mesh_link_mode": mesh_mode,
        "image_link_mode": image_mode,
        "ok": True,
    }
    (case_dir / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 2K.2 multi-case dirs.")
    parser.add_argument(
        "--cases-config",
        default="configs/phase2k2_truepbr200_cases.json",
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.cases_config)
    reference_case = resolve_project_path(config["reference_case_layout_path"])
    if not (reference_case / "input" / "mesh.glb").exists():
        raise SystemExit(f"reference case mesh missing: {reference_case / 'input' / 'mesh.glb'}")
    if not (reference_case / "input" / "image.png").exists():
        raise SystemExit(f"reference case image missing: {reference_case / 'input' / 'image.png'}")

    asset_ids = list(config.get("selected_asset_ids", []))
    if asset_ids != REQUIRED_ASSET_IDS:
        raise SystemExit(f"selected_asset_ids mismatch: {asset_ids} != {REQUIRED_ASSET_IDS}")

    manifests = [prepare_case(config, asset_id) for asset_id in asset_ids]
    print("Phase 2K.2 prepared case directories")
    for manifest in manifests:
        print(f"  {manifest['asset_id']}: {manifest['case_dir']}")
        print(f"    mesh: {manifest['source_mesh_path']}")
        print(f"    image: {manifest['source_reference_path']}")
    print("PHASE2K2_MULTICASE_CASES_PREPARED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
