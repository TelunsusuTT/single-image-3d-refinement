from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_phase2k4_reference_view_ablation_readiness as readiness  # noqa: E402


ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]
INPUT_VIEWS = ["004", "005"]


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def make_project_scripts(project_root: Path) -> Path:
    scripts_dir = project_root / "scripts"
    for filename in ("run_phase2g_paint_infer.py", "make_phase2g6_texture_comparison.py"):
        write_file(scripts_dir / filename, b"script")
    return project_root


def write_cases_config(
    path: Path,
    root: Path,
    input_views: list[str] | None = None,
    checkpoint: Path | None = None,
) -> Path:
    input_views = input_views or INPUT_VIEWS
    checkpoint = checkpoint or (
        root
        / "checkpoints"
        / "pilot_v1_truepbr_200_lr1e6"
        / "pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt"
    )
    write_file(checkpoint, b"checkpoint")
    assets = {}
    for asset_id in ASSET_IDS:
        mesh = write_file(root / "meshes" / f"{asset_id}.glb", b"mesh")
        refs = root / "refs" / asset_id / "render_cond"
        for view_id in INPUT_VIEWS:
            write_file(refs / f"{view_id}_light_AL.png", b"image")
        assets[asset_id] = {
            "mesh_path": str(mesh),
            "reference_image_path_template": str(refs / "{view_id}_light_AL.png"),
        }
    data = {
        "checkpoint": str(checkpoint),
        "output_root": str(root / "outputs" / "phase2k4"),
        "asset_ids": ASSET_IDS,
        "input_views": input_views,
        "primary_eval_views": input_views,
        "previous_baseline_render_eval_root": str(root / "baseline"),
        "assets": assets,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


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
        config = write_cases_config(root / "cases.json", root)
        hypaint = make_hypaint(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))
        assert run_readiness(config, hypaint) == 0


def test_readiness_fails_if_input_views_not_exact(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root, input_views=["005", "004"])
        hypaint = make_hypaint(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))
        assert run_readiness(config, hypaint) == 1


def test_readiness_fails_if_checkpoint_contains_truepbr50(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint = (
            root
            / "checkpoints"
            / "pilot_v1_truepbr_200_lr1e6"
            / "pilot_v1_truepbr_50_lr1e6"
            / "pilot_v1_truepbr_200_lr1e6-stepstep=200.ckpt"
        )
        config = write_cases_config(root / "cases.json", root, checkpoint=checkpoint)
        hypaint = make_hypaint(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))
        assert run_readiness(config, hypaint) == 1


def test_readiness_fails_if_render_cond_004_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root)
        (root / "refs" / "B073NZS57V" / "render_cond" / "004_light_AL.png").unlink()
        hypaint = make_hypaint(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))
        assert run_readiness(config, hypaint) == 1


def test_readiness_fails_if_output_dir_already_contains_glb(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_cases_config(root / "cases.json", root)
        write_file(
            root
            / "outputs"
            / "phase2k4"
            / "infer"
            / "finetuned"
            / "B075YLTF7Q"
            / "input_004"
            / "old.glb",
            b"old",
        )
        hypaint = make_hypaint(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_project_scripts(root / "project"))
        assert run_readiness(config, hypaint) == 1
