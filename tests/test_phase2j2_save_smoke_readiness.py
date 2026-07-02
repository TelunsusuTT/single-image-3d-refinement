from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2j2_save_smoke_readiness import main as readiness_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_samples(root: Path, count: int = 7) -> list[Path]:
    samples = []
    for index in range(count):
        sample = root / f"sample_{index}"
        (sample / "render_tex").mkdir(parents=True)
        (sample / "render_cond").mkdir(parents=True)
        samples.append(sample)
    return samples


def write_examples(path: Path, samples: list[Path]) -> Path:
    return write_text(path, json.dumps([str(sample.resolve()) for sample in samples]))


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "train.py", b"train")
    return hypaint


def write_config(
    path: Path,
    learning_rate: str = "0.0",
    max_steps: int = 1,
    extra_text: str = "",
) -> Path:
    return write_text(
        path,
        "\n".join(
            [
                "model:",
                f"  base_learning_rate: {learning_rate}",
                "  params:",
                "    stable_diffusion_config:",
                "      pretrained_model_name_or_path: /tmp/hunyuan3d-paintpbr-v2-1",
                "lightning:",
                "  modelcheckpoint:",
                "    params:",
                "      every_n_train_steps: 1",
                "      save_weights_only: true",
                "  trainer:",
                f"    max_steps: {max_steps}",
                extra_text,
            ]
        )
        + "\n",
    )


def run_readiness(
    examples_json: Path,
    config: Path,
    hypaint: Path,
    checkpoint_root: Path,
    output_dir: Path,
) -> int:
    return readiness_main(
        [
            "--examples-json",
            str(examples_json),
            "--config",
            str(config),
            "--hypaint",
            str(hypaint),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-dir",
            str(output_dir),
        ]
    )


def make_valid_setup(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    samples = make_samples(root)
    examples_json = write_examples(root / "examples.json", samples)
    config = write_config(root / "config.yaml")
    hypaint = make_hypaint(root)
    checkpoint_root = root / "checkpoints" / "pilot_v1_truepbr_1step_lr0"
    output_dir = root / "outputs" / "phase2j2"
    return examples_json, config, hypaint, checkpoint_root, output_dir


def test_readiness_passes_with_valid_fake_setup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        assert run_readiness(*make_valid_setup(root)) == 0


def test_readiness_fails_if_config_contains_sd2_community() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, output_dir = make_valid_setup(root)
        write_config(config, extra_text="old: sd2-community/stable-diffusion-2-1")
        assert run_readiness(examples_json, config, hypaint, checkpoint_root, output_dir) == 1


def test_readiness_fails_if_learning_rate_is_not_zero() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, output_dir = make_valid_setup(root)
        write_config(config, learning_rate="1e-6")
        assert run_readiness(examples_json, config, hypaint, checkpoint_root, output_dir) == 1


def test_readiness_fails_if_max_steps_is_not_one() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, output_dir = make_valid_setup(root)
        write_config(config, max_steps=50)
        assert run_readiness(examples_json, config, hypaint, checkpoint_root, output_dir) == 1


def test_readiness_fails_if_checkpoint_root_already_contains_ckpt() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, output_dir = make_valid_setup(root)
        write_file(checkpoint_root / "old.ckpt", b"old")
        assert run_readiness(examples_json, config, hypaint, checkpoint_root, output_dir) == 1
