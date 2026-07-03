from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_phase2k2_truepbr200_multicase_readiness as readiness  # noqa: E402


ASSET_IDS = ["B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_config(path: Path, root: Path, asset_ids: list[str] | None = None, checkpoint: Path | None = None) -> Path:
    checkpoint = checkpoint or (
        root
        / "checkpoints"
        / "pilot_v1_truepbr_200_lr1e6"
        / "pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt"
    )
    data = {
        "checkpoint": str(checkpoint),
        "output_root": str(root / "outputs" / "phase2k" / "multicase_truepbr200"),
        "selected_asset_ids": asset_ids or ASSET_IDS,
        "raw_mesh_path_template": str(root / "raw" / "{asset_id}.glb"),
        "reference_image_path_template": str(root / "refs" / "{asset_id}" / "render_cond" / "001_light_AL.png"),
        "reference_case_layout_path": str(root / "reference_case"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def make_project_scripts(project_root: Path) -> Path:
    scripts_dir = project_root / "scripts"
    for filename in (
        "run_phase2g_paint_infer.py",
        "make_phase2g6_texture_comparison.py",
        "load_phase2g4_finetuned_checkpoint_only.py",
    ):
        write_file(scripts_dir / filename, b"script")
    return project_root


def make_valid_setup(root: Path) -> tuple[Path, Path]:
    checkpoint = write_file(
        root
        / "checkpoints"
        / "pilot_v1_truepbr_200_lr1e6"
        / "pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt",
        b"checkpoint",
    )
    for asset_id in ASSET_IDS:
        write_file(root / "raw" / f"{asset_id}.glb", b"mesh")
        write_file(root / "refs" / asset_id / "render_cond" / "001_light_AL.png", b"image")
    config = write_config(root / "cases.json", root, checkpoint=checkpoint)
    hypaint = make_hypaint(root)
    return config, hypaint


def run_readiness(config: Path, hypaint: Path) -> int:
    return readiness.main(
        [
            "--cases-config",
            str(config),
            "--hypaint",
            str(hypaint),
        ]
    )


def test_readiness_passes_with_valid_fake_setup(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, hypaint = make_valid_setup(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))

        assert run_readiness(config, hypaint) == 0


def test_readiness_fails_if_selected_assets_differ(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, hypaint = make_valid_setup(root)
        write_config(config, root, asset_ids=["B07HSK7MXZ"])
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))

        assert run_readiness(config, hypaint) == 1


def test_readiness_fails_if_checkpoint_contains_overfit_token(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _, hypaint = make_valid_setup(root)
        checkpoint = write_file(
            root
            / "checkpoints"
            / "pilot_v1_truepbr_200_lr1e6"
            / "pilot_v1_overfit_500"
            / "pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt",
            b"checkpoint",
        )
        config = write_config(root / "cases.json", root, checkpoint=checkpoint)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))

        assert run_readiness(config, hypaint) == 1


def test_readiness_fails_if_checkpoint_contains_truepbr50_token(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _, hypaint = make_valid_setup(root)
        checkpoint = write_file(
            root
            / "checkpoints"
            / "pilot_v1_truepbr_200_lr1e6"
            / "pilot_v1_truepbr_50_lr1e6"
            / "pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt",
            b"checkpoint",
        )
        config = write_config(root / "cases.json", root, checkpoint=checkpoint)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))

        assert run_readiness(config, hypaint) == 1


def test_readiness_fails_if_one_raw_mesh_is_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, hypaint = make_valid_setup(root)
        (root / "raw" / "B073NZS57V.glb").unlink()
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))

        assert run_readiness(config, hypaint) == 1


def test_readiness_fails_if_finetuned_output_dir_already_contains_glb(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, hypaint = make_valid_setup(root)
        write_file(
            root
            / "outputs"
            / "phase2k"
            / "multicase_truepbr200"
            / "finetuned"
            / "B073NZS57V"
            / "old.glb",
            b"old",
        )
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))

        assert run_readiness(config, hypaint) == 1
