from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2e_readiness import main as readiness_main  # noqa: E402


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


def test_phase2e_readiness_passes_for_expected_count_and_absolute_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        samples = [make_sample(root, f"sample_{index}") for index in range(3)]
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve()) for sample in samples])
        config = root / "config.yaml"
        config.write_text(f"json_path: {examples_json.resolve()}\n", encoding="utf-8")

        assert (
            readiness_main(
                [
                    "--examples-json",
                    str(examples_json),
                    "--config",
                    str(config),
                    "--expected-count",
                    "3",
                ]
            )
            == 0
        )


def test_phase2e_readiness_fails_for_relative_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, ["relative/sample"])
        config = root / "config.yaml"
        config.write_text(f"json_path: {examples_json.resolve()}\n", encoding="utf-8")

        assert (
            readiness_main(
                [
                    "--examples-json",
                    str(examples_json),
                    "--config",
                    str(config),
                    "--expected-count",
                    "1",
                ]
            )
            == 1
        )


def test_phase2e_readiness_fails_for_wrong_count() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        config.write_text(f"json_path: {examples_json.resolve()}\n", encoding="utf-8")

        assert (
            readiness_main(
                [
                    "--examples-json",
                    str(examples_json),
                    "--config",
                    str(config),
                    "--expected-count",
                    "2",
                ]
            )
            == 1
        )


def test_phase2e_readiness_fails_when_config_omits_examples_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample = make_sample(root, "sample")
        examples_json = root / "examples_train_abs.json"
        write_examples(examples_json, [str(sample.resolve())])
        config = root / "config.yaml"
        config.write_text("json_path: /tmp/other.json\n", encoding="utf-8")

        assert (
            readiness_main(
                [
                    "--examples-json",
                    str(examples_json),
                    "--config",
                    str(config),
                    "--expected-count",
                    "1",
                ]
            )
            == 1
        )
