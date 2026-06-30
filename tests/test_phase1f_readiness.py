from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase1f_readiness import main as readiness_main  # noqa: E402
from make_phase1f_train_json import main as make_json_main  # noqa: E402


def make_fake_sample(root: Path) -> Path:
    sample_dir = root / "sample"
    (sample_dir / "render_tex").mkdir(parents=True)
    (sample_dir / "render_cond").mkdir(parents=True)
    (sample_dir / "render_tex" / "transforms.json").write_text("{}", encoding="utf-8")
    return sample_dir


def test_make_phase1f_train_json_writes_absolute_sample_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_fake_sample(root)
        out_json = root / "examples_train_abs.json"

        assert make_json_main(["--sample-dir", str(sample_dir), "--out-json", str(out_json)]) == 0

        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data == [str(sample_dir.resolve())]
        assert Path(data[0]).is_absolute()


def test_phase1f_readiness_passes_for_valid_fake_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_fake_sample(root)
        examples_json = root / "examples_train_abs.json"
        examples_json.write_text(json.dumps([str(sample_dir.resolve())]), encoding="utf-8")
        config = root / "config.yaml"
        config.write_text(f"json_path: {examples_json.resolve()}\n", encoding="utf-8")

        assert readiness_main(["--examples-json", str(examples_json), "--config", str(config)]) == 0


def test_phase1f_readiness_fails_for_relative_sample_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json = root / "examples_train_abs.json"
        examples_json.write_text(json.dumps(["relative/sample"]), encoding="utf-8")
        config = root / "config.yaml"
        config.write_text(f"json_path: {examples_json.resolve()}\n", encoding="utf-8")

        assert readiness_main(["--examples-json", str(examples_json), "--config", str(config)]) == 1


def test_phase1f_readiness_fails_when_config_omits_examples_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_fake_sample(root)
        examples_json = root / "examples_train_abs.json"
        examples_json.write_text(json.dumps([str(sample_dir.resolve())]), encoding="utf-8")
        config = root / "config.yaml"
        config.write_text("json_path: /tmp/other.json\n", encoding="utf-8")

        assert readiness_main(["--examples-json", str(examples_json), "--config", str(config)]) == 1
