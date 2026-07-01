from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g4_load_readiness import main as readiness_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def keyspace_text(
    status: str = "RECOMMENDED",
    transform: str = "strip:unet.",
    target_path: str = "paint_pipeline.models['multiview_model'].pipeline.unet",
) -> str:
    return "\n".join(
        [
            "# Phase 2G.3 Keyspace Compare",
            "",
            "## Recommendation",
            f"- recommendation status: `{status}`",
            f"- recommended target path: `{target_path}`",
            "- recommended target class: `UNet2p5DConditionModel`",
            f"- recommended transform: `{transform}`",
        ]
    )


def make_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = write_file(root / "checkpoints" / "model.ckpt", b"checkpoint")
    hypaint = make_hypaint(root)
    output_dir = root / "outputs" / "phase2g" / "load_only" / "pilot_v1_overfit_500"
    keyspace = root / "outputs" / "phase2g" / "key_inspection" / "pilot_v1_overfit_500" / "keyspace_compare_v2.md"
    return checkpoint, hypaint, output_dir, keyspace


def run_readiness(checkpoint: Path, hypaint: Path, output_dir: Path) -> int:
    return readiness_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--hypaint",
            str(hypaint),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_readiness_passes_with_required_recommendation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint, hypaint, output_dir, keyspace = make_paths(root)
        write_text(keyspace, keyspace_text())

        assert run_readiness(checkpoint, hypaint, output_dir) == 0
        assert output_dir.parent.is_dir()


def test_readiness_fails_if_recommendation_is_unknown() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint, hypaint, output_dir, keyspace = make_paths(root)
        write_text(keyspace, keyspace_text(status="UNKNOWN"))

        assert run_readiness(checkpoint, hypaint, output_dir) == 1


def test_readiness_fails_if_transform_is_not_strip_unet() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint, hypaint, output_dir, keyspace = make_paths(root)
        write_text(keyspace, keyspace_text(transform="strip:unet.unet."))

        assert run_readiness(checkpoint, hypaint, output_dir) == 1


def test_readiness_fails_if_checkpoint_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint, hypaint, output_dir, keyspace = make_paths(root)
        write_text(keyspace, keyspace_text())

        assert run_readiness(checkpoint.parent / "missing.ckpt", hypaint, output_dir) == 1
