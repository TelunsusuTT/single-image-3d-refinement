from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2h1_readiness import main as readiness_main  # noqa: E402


CHECKPOINT_DIRPATH = (
    "/vol/bitbucket/ct1022/hy3dpaint_finetune/"
    "checkpoints/pilot_v1_conservative_50_lr1e6"
)


def make_sample(root: Path, name: str) -> Path:
    sample_dir = root / name
    (sample_dir / "render_tex").mkdir(parents=True)
    (sample_dir / "render_cond").mkdir(parents=True)
    return sample_dir


def write_examples(path: Path, sample_dirs: list[str]) -> None:
    path.write_text(json.dumps(sample_dirs), encoding="utf-8")


def write_config(
    path: Path,
    examples_json: Path,
    max_steps: int = 50,
    learning_rate: str = "1e-6",
    every_n_train_steps: int = 50,
    save_top_k: int = -1,
    save_weights_only: bool = True,
    extra_lines: list[str] | None = None,
) -> None:
    save_weights_only_value = "true" if save_weights_only else "false"
    lines = [
        f"base_learning_rate: {learning_rate}",
        f"json_path: {examples_json.resolve()}",
        f"max_steps: {max_steps}",
        f"dirpath: {CHECKPOINT_DIRPATH}",
        f"every_n_train_steps: {every_n_train_steps}",
        f"save_top_k: {save_top_k}",
        "save_last: false",
        f"save_weights_only: {save_weights_only_value}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_readiness(
    root: Path,
    examples_json: Path,
    config: Path,
    expected_count: int = 2,
) -> int:
    return readiness_main(
        [
            "--examples-json",
            str(examples_json),
            "--config",
            str(config),
            "--expected-count",
            str(expected_count),
            "--checkpoint-root",
            str(root / "checkpoints" / "pilot_v1_conservative_50_lr1e6"),
        ]
    )


def test_phase2h1_readiness_passes_for_valid_conservative_setup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        samples = [make_sample(root, f"sample_{index}") for index in range(2)]
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve()) for sample in samples])
        config = root / "config.yaml"
        write_config(config, examples_json)

        assert run_readiness(root, examples_json, config) == 0
        assert (root / "checkpoints" / "pilot_v1_conservative_50_lr1e6").is_dir()


def test_phase2h1_readiness_fails_if_max_steps_is_not_50() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        write_config(config, examples_json, max_steps=500)

        assert run_readiness(root, examples_json, config, expected_count=1) == 1


def test_phase2h1_readiness_fails_if_learning_rate_is_not_1e_minus_6() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        write_config(config, examples_json, learning_rate="5.0e-05")

        assert run_readiness(root, examples_json, config, expected_count=1) == 1


def test_phase2h1_readiness_fails_if_checkpoint_root_already_has_ckpt() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        write_config(config, examples_json)
        checkpoint_root = root / "checkpoints" / "pilot_v1_conservative_50_lr1e6"
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / "old.ckpt").write_bytes(b"old")

        assert run_readiness(root, examples_json, config, expected_count=1) == 1


def test_phase2h1_readiness_fails_if_collapsed_checkpoint_path_is_present() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        write_config(
            config,
            examples_json,
            extra_lines=[
                "resume_from: /vol/bitbucket/ct1022/hy3dpaint_finetune/"
                "checkpoints/pilot_v1_overfit_500/bad.ckpt"
            ],
        )

        assert run_readiness(root, examples_json, config, expected_count=1) == 1
