#!/usr/bin/env python3
"""Prepare a small Phase 2G inference case without loading model weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def require_file(path: Path, label: str, errors: list[str]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        errors.append(f"{label} missing: {resolved}")
    return resolved


def link_or_copy(source: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(source)
        return "symlink"
    except OSError:
        shutil.copy2(source, dest)
        return "copy"


def prepare_case(
    asset_id: str,
    mesh: Path,
    reference_image: Path,
    checkpoint: Path,
    out_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    mesh_path = require_file(mesh, "mesh", errors)
    reference_path = require_file(reference_image, "reference image", errors)
    checkpoint_path = require_file(checkpoint, "checkpoint", errors)
    if errors:
        return {"ok": False, "errors": errors}

    case_dir = out_dir.expanduser().resolve()
    input_dir = case_dir / "input"
    input_mesh = input_dir / "mesh.glb"
    input_image = input_dir / "image.png"
    mesh_mode = link_or_copy(mesh_path, input_mesh)
    image_mode = link_or_copy(reference_path, input_image)

    manifest = {
        "asset_id": asset_id,
        "source_mesh_path": str(mesh_path),
        "source_reference_path": str(reference_path),
        "checkpoint_path": str(checkpoint_path),
        "input_mesh_path": str(input_mesh),
        "input_image_path": str(input_image),
        "mesh_link_mode": mesh_mode,
        "image_link_mode": image_mode,
        "ok": True,
    }
    (case_dir / "case_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 2G inference case inputs.")
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_case(
        asset_id=args.asset_id,
        mesh=args.mesh,
        reference_image=args.reference_image,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
    )
    if not manifest.get("ok"):
        for error in manifest["errors"]:
            print(f"ERROR: {error}")
        return 1
    print(f"prepared case: {args.out_dir}")
    print(f"input mesh: {manifest['input_mesh_path']}")
    print(f"input image: {manifest['input_image_path']}")
    print(f"checkpoint: {manifest['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
