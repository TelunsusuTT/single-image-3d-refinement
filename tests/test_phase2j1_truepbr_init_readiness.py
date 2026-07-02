from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2j1_truepbr_init_readiness import main as readiness_main  # noqa: E402


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
    write_file(hypaint / "train.py", b"train")
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "hunyuanpaintpbr" / "unet" / "model.py", b"model")
    write_file(hypaint / "hunyuanpaintpbr" / "pipeline.py", b"pipeline")
    return hypaint


def make_pbr_dir(root: Path, include_model_index: bool = True) -> Path:
    pbr_dir = root / "hunyuan3d-paintpbr-v2-1"
    if include_model_index:
        write_file(pbr_dir / "model_index.json", b"{}")
    for dirname in ("unet", "scheduler", "tokenizer", "text_encoder"):
        (pbr_dir / dirname).mkdir(parents=True, exist_ok=True)
    return pbr_dir


def write_config(path: Path, pbr_dir: Path, extra_text: str = "") -> Path:
    return write_text(
        path,
        "\n".join(
            [
                "model:",
                "  params:",
                "    stable_diffusion_config:",
                f"      pretrained_model_name_or_path: {pbr_dir.resolve()}",
                "      custom_pipeline: ./hunyuanpaintpbr",
                "resume_from: null",
                "init_control_from: null",
                extra_text,
            ]
        )
        + "\n",
    )


def run_readiness(hypaint: Path, config: Path, pbr_dir: Path, output_dir: Path) -> int:
    return readiness_main(
        [
            "--hypaint",
            str(hypaint),
            "--config",
            str(config),
            "--pbr-dir",
            str(pbr_dir),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_readiness_passes_with_fake_hunyuan_pbr_and_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        pbr_dir = make_pbr_dir(root)
        config = write_config(root / "config.yaml", pbr_dir)
        output_dir = root / "outputs" / "phase2j1"

        assert run_readiness(hypaint, config, pbr_dir, output_dir) == 0
        assert output_dir.parent.is_dir()


def test_readiness_fails_if_config_contains_sd2_community() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        pbr_dir = make_pbr_dir(root)
        config = write_config(
            root / "config.yaml",
            pbr_dir,
            "old: sd2-community/stable-diffusion-2-1",
        )
        output_dir = root / "outputs" / "phase2j1"

        assert run_readiness(hypaint, config, pbr_dir, output_dir) == 1


def test_readiness_fails_if_config_contains_stabilityai_sd21() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        pbr_dir = make_pbr_dir(root)
        config = write_config(
            root / "config.yaml",
            pbr_dir,
            "old: stabilityai/stable-diffusion-2-1",
        )
        output_dir = root / "outputs" / "phase2j1"

        assert run_readiness(hypaint, config, pbr_dir, output_dir) == 1


def test_readiness_fails_if_pbr_dir_lacks_model_index() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        pbr_dir = make_pbr_dir(root, include_model_index=False)
        config = write_config(root / "config.yaml", pbr_dir)
        output_dir = root / "outputs" / "phase2j1"

        assert run_readiness(hypaint, config, pbr_dir, output_dir) == 1


def test_readiness_fails_if_config_contains_overfit500_reference() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        pbr_dir = make_pbr_dir(root)
        config = write_config(
            root / "config.yaml",
            pbr_dir,
            "resume_from: checkpoints/pilot_v1_overfit_500/bad.ckpt",
        )
        output_dir = root / "outputs" / "phase2j1"

        assert run_readiness(hypaint, config, pbr_dir, output_dir) == 1
