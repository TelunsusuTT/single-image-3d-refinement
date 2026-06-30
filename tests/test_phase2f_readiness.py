from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2f_readiness import main as readiness_main  # noqa: E402


def make_sample(root: Path, name: str) -> Path:
    sample_dir = root / name
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    render_tex.mkdir(parents=True)
    render_cond.mkdir(parents=True)
    (render_tex / "transforms.json").write_text("{}", encoding="utf-8")
    return sample_dir


def write_examples(path: Path, sample_dirs: list[str]) -> None:
    path.write_text(json.dumps(sample_dirs), encoding="utf-8")


def write_config(
    path: Path,
    examples_json: Path,
    max_steps: int = 500,
    every_n_train_steps: int = 500,
    save_top_k: int = -1,
    save_weights_only: bool = True,
) -> None:
    save_weights_only_value = "true" if save_weights_only else "false"
    path.write_text(
        "\n".join(
            [
                f"json_path: {examples_json.resolve()}",
                f"max_steps: {max_steps}",
                "dirpath: /vol/bitbucket/ct1022/hy3dpaint_finetune/checkpoints/pilot_v1_overfit_500",
                f"every_n_train_steps: {every_n_train_steps}",
                f"save_top_k: {save_top_k}",
                "save_last: false",
                f"save_weights_only: {save_weights_only_value}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_readiness(root: Path, examples_json: Path, config: Path, expected_count: int = 2) -> int:
    return readiness_main(
        [
            "--examples-json",
            str(examples_json),
            "--config",
            str(config),
            "--expected-count",
            str(expected_count),
            "--checkpoint-root",
            str(root / "logs" / "train"),
        ]
    )


def test_phase2f_readiness_passes_for_valid_checkpoint_setup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        samples = [make_sample(root, f"sample_{index}") for index in range(2)]
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve()) for sample in samples])
        config = root / "config.yaml"
        write_config(config, examples_json)

        assert run_readiness(root, examples_json, config) == 0
        assert (root / "logs" / "train").is_dir()


def test_phase2f_readiness_fails_if_max_steps_is_not_500() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        write_config(config, examples_json, max_steps=50)

        assert run_readiness(root, examples_json, config, expected_count=1) == 1


def test_phase2f_readiness_fails_if_every_n_train_steps_is_too_high() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        write_config(config, examples_json, every_n_train_steps=1000000)

        assert run_readiness(root, examples_json, config, expected_count=1) == 1


def test_phase2f_readiness_fails_if_save_top_k_disables_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        write_config(config, examples_json, save_top_k=0)

        assert run_readiness(root, examples_json, config, expected_count=1) == 1


def test_phase2f_readiness_fails_on_missing_sample_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        missing_sample = root / "missing_sample"
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(missing_sample.resolve())])
        config = root / "config.yaml"
        write_config(config, examples_json)

        assert run_readiness(root, examples_json, config, expected_count=1) == 1
