import json
from pathlib import Path

from scripts import inspect_datav2_frame_mini40_checkpoint as inspect_checkpoint


def write_config(tmp_path: Path, checkpoint_dir: Path) -> Path:
    config = {
        "experiment_name": "datav2_frame_mini40_truepbr_500_lr1e6",
        "steps": 500,
        "output_checkpoint_dir": str(checkpoint_dir),
        "readiness_report_dir": str(tmp_path / "reports"),
    }
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_json


def test_checkpoint_inspection_reports_expected_ckpt(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    ckpt = checkpoint_dir / "datav2_frame_mini40_truepbr_500_lr1e6-step500.ckpt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    config_json = write_config(tmp_path, checkpoint_dir)

    assert inspect_checkpoint.main(["--config", str(config_json)]) == 0
    report = json.loads((tmp_path / "reports" / "checkpoint_inspection.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["expected_checkpoint_count"] == 1
    assert report["checkpoints"][0]["name"] == ckpt.name


def test_checkpoint_inspection_fails_without_expected_step_marker(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "datav2_frame_mini40_truepbr_500_lr1e6-step499.ckpt").write_bytes(b"x")
    config_json = write_config(tmp_path, checkpoint_dir)

    assert inspect_checkpoint.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "checkpoint_inspection.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["expected_checkpoint_count"] == 0
