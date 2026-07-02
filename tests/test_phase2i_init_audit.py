from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_phase2i_training_initialization import main as inspect_main  # noqa: E402
from check_phase2i_audit_readiness import main as readiness_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_hypaint(root: Path, include_realesrgan: bool = True) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    if include_realesrgan:
        write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def write_sd2_config(path: Path) -> Path:
    return write_text(
        path,
        "\n".join(
            [
                "model:",
                "  base_learning_rate: 1e-6",
                "  params:",
                "    stable_diffusion_config:",
                "      pretrained_model_name_or_path: sd2-community/stable-diffusion-2-1",
                "      custom_pipeline: ./hunyuanpaintpbr",
                "lightning:",
                "  trainer:",
                "    max_steps: 50",
                "init_control_from: null",
                "resume_from: null",
            ]
        )
        + "\n",
    )


def run_inspector(config: Path, out_json: Path, out_md: Path) -> int:
    return inspect_main(
        [
            "--configs",
            str(config),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )


def run_readiness(
    hypaint: Path,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
) -> int:
    return readiness_main(
        [
            "--hypaint",
            str(hypaint),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_config_inspector_flags_sd2_resume_null_as_not_confirmed_hunyuan_finetune() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = write_sd2_config(root / "config.yaml")
        out_json = root / "report.json"
        out_md = root / "report.md"

        assert run_inspector(config, out_json, out_md) == 0
        report = json.loads(out_json.read_text(encoding="utf-8"))
        item = report["configs"][0]
        assert item["stable_diffusion_config.pretrained_model_name_or_path"] == (
            "sd2-community/stable-diffusion-2-1"
        )
        assert item["resume_from_is_null_or_missing"] is True
        assert item["not_confirmed_hunyuan_pbr_finetune"] is True
        assert "not confirmed as Hunyuan3D-Paint PBR fine-tune" in item["interpretation"]


def test_readiness_passes_with_fake_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        config = write_sd2_config(root / "config.yaml")
        checkpoint = write_file(root / "model.ckpt", b"checkpoint")
        output_dir = root / "outputs" / "phase2i" / "audit"

        assert run_readiness(hypaint, config, checkpoint, output_dir) == 0
        assert output_dir.parent.is_dir()


def test_readiness_fails_if_checkpoint_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        config = write_sd2_config(root / "config.yaml")
        missing_checkpoint = root / "missing.ckpt"
        output_dir = root / "outputs" / "phase2i" / "audit"

        assert run_readiness(hypaint, config, missing_checkpoint, output_dir) == 1


def test_readiness_fails_if_realesrgan_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root, include_realesrgan=False)
        config = write_sd2_config(root / "config.yaml")
        checkpoint = write_file(root / "model.ckpt", b"checkpoint")
        output_dir = root / "outputs" / "phase2i" / "audit"

        assert run_readiness(hypaint, config, checkpoint, output_dir) == 1
