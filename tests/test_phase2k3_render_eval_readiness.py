from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_phase2k3_render_eval_readiness as readiness  # noqa: E402


ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]
VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_fake_project_scripts(project_root: Path) -> Path:
    scripts_dir = project_root / "scripts"
    for filename in (
        "render_phase2k3_glb_views_blender.py",
        "compare_phase2k3_rendered_views.py",
        "aggregate_phase2k3_rendered_metrics.py",
    ):
        write_file(scripts_dir / filename, b"script")
    return project_root


def make_fake_blender(root: Path) -> Path:
    blender = write_file(root / "bin" / "blender", b"#!/bin/sh\nexit 0\n")
    mode = blender.stat().st_mode
    blender.chmod(mode | stat.S_IXUSR)
    return blender


def write_cases_config(path: Path, root: Path, asset_ids: list[str] | None = None) -> Path:
    asset_ids = asset_ids or ASSET_IDS
    cases = {}
    for asset_id in asset_ids:
        base = write_file(root / "glbs" / asset_id / "base.glb", b"base")
        fine = write_file(root / "glbs" / asset_id / "fine.glb", b"fine")
        cases[asset_id] = {
            "base_glb": str(base),
            "finetuned_glb": str(fine),
        }
        for view_id in VIEW_IDS:
            write_file(root / "refs" / asset_id / "render_cond" / f"{view_id}_light_AL.png", b"image")
    data = {
        "output_root": str(root / "outputs" / "rendered"),
        "view_ids": VIEW_IDS,
        "render_resolution": 512,
        "background_color": [0.28, 0.28, 0.28],
        "reference_image_path_template": str(root / "refs" / "{asset_id}" / "render_cond" / "{view_id}_light_AL.png"),
        "cases": cases,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_readiness(config: Path, output_root: Path) -> int:
    return readiness.main(
        [
            "--cases-config",
            str(config),
            "--output-root",
            str(output_root),
        ]
    )


def test_readiness_passes_with_fake_setup(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root)
        project = make_fake_project_scripts(root / "project")
        blender = make_fake_blender(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", project)
        monkeypatch.setenv("BLENDER_BIN", str(blender))

        assert run_readiness(config, root / "outputs" / "rendered") == 0


def test_readiness_fails_if_one_reference_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root)
        (root / "refs" / "B073NZS57V" / "render_cond" / "003_light_AL.png").unlink()
        project = make_fake_project_scripts(root / "project")
        blender = make_fake_blender(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", project)
        monkeypatch.setenv("BLENDER_BIN", str(blender))

        assert run_readiness(config, root / "outputs" / "rendered") == 1


def test_readiness_fails_if_finetuned_glb_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root)
        (root / "glbs" / "B07B8MWCR8" / "fine.glb").unlink()
        project = make_fake_project_scripts(root / "project")
        blender = make_fake_blender(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", project)
        monkeypatch.setenv("BLENDER_BIN", str(blender))

        assert run_readiness(config, root / "outputs" / "rendered") == 1


def test_readiness_fails_if_asset_set_differs(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root, asset_ids=["B075YLTF7Q"])
        project = make_fake_project_scripts(root / "project")
        blender = make_fake_blender(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", project)
        monkeypatch.setenv("BLENDER_BIN", str(blender))

        assert run_readiness(config, root / "outputs" / "rendered") == 1


def test_readiness_fails_if_output_render_dir_has_png(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root)
        output_root = root / "outputs" / "rendered"
        write_file(output_root / "renders" / "B075YLTF7Q" / "base" / "000.png", b"old")
        project = make_fake_project_scripts(root / "project")
        blender = make_fake_blender(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", project)
        monkeypatch.setenv("BLENDER_BIN", str(blender))

        assert run_readiness(config, output_root) == 1
