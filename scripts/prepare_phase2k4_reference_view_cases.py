#!/usr/bin/env python3
"""Prepare Phase 2K.4 reference-view ablation case directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_text: str, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


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


def prepare_case(config: dict[str, Any], asset_id: str, input_view: str) -> dict[str, Any]:
    output_root = resolve_project_path(config["output_root"])
    asset = config["assets"][asset_id]
    mesh = resolve_project_path(asset["mesh_path"])
    image = resolve_project_path(asset["reference_image_path_template"].format(view_id=input_view))
    case_dir = output_root / "cases" / asset_id / f"input_{input_view}"
    input_dir = case_dir / "input"
    input_mesh = input_dir / "mesh.glb"
    input_image = input_dir / "image.png"
    mesh_mode = link_or_copy(mesh, input_mesh)
    image_mode = link_or_copy(image, input_image)
    manifest = {
        "asset_id": asset_id,
        "input_view": input_view,
        "mesh_source": str(mesh),
        "image_source": str(image),
        "case_dir": str(case_dir),
        "input_mesh_path": str(input_mesh),
        "input_image_path": str(input_image),
        "mesh_link_mode": mesh_mode,
        "image_link_mode": image_mode,
        "human_confirmed_front_view": True,
        "ok": True,
    }
    (case_dir / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 2K.4 reference-view cases.")
    parser.add_argument("--cases-config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.cases_config)
    manifests = []
    for asset_id in config.get("asset_ids", []):
        for input_view in config.get("input_views", []):
            manifests.append(prepare_case(config, asset_id, input_view))
    print("Phase 2K.4 prepared reference-view case directories")
    for manifest in manifests:
        print(f"  {manifest['asset_id']} input_{manifest['input_view']}: {manifest['case_dir']}")
        print(f"    mesh: {manifest['mesh_source']}")
        print(f"    image: {manifest['image_source']}")
    print("PHASE2K4_REFERENCE_VIEW_CASES_PREPARED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
