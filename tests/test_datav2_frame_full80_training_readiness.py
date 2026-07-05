import json
import shutil
from pathlib import Path

from scripts import check_datav2_frame_full80_training_readiness as readiness


def make_sample(sample_dir: Path) -> None:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    render_tex.mkdir(parents=True)
    render_cond.mkdir(parents=True)
    (render_tex / "transforms.json").write_text('{"frames": []}\n', encoding="utf-8")
    for view_id in readiness.VIEW_IDS:
        for suffix in readiness.TEX_SUFFIXES:
            (render_tex / f"{view_id}{suffix}").write_bytes(b"x")
        for suffix in readiness.COND_SUFFIXES:
            (render_cond / f"{view_id}{suffix}").write_bytes(b"x")


def write_examples(root: Path, split: str, count: int) -> tuple[Path, list[Path]]:
    samples = []
    for index in range(count):
        sample_dir = root / f"{split}_{index:03d}"
        make_sample(sample_dir)
        samples.append(sample_dir)
    examples_json = root / f"{split}_examples.json"
    examples_json.write_text(json.dumps([str(path) for path in samples]) + "\n", encoding="utf-8")
    return examples_json, samples


def write_training_yaml(path: Path, config: dict[str, object]) -> None:
    text = f"""
model:
  base_learning_rate: {config["learning_rate"]}
  params:
    stable_diffusion_config:
      pretrained_model_name_or_path: {config["official_pbr_source_dir"]}
data:
  params:
    batch_size: 1
    train:
    - params:
        json_path: {config["train_examples_json"]}
    validation:
    - params:
        json_path: {config["val_examples_json"]}
lightning:
  modelcheckpoint:
    params:
      dirpath: {config["output_checkpoint_dir"]}
      filename: datav2_frame_full80_truepbr_500_lr1e6-step{{step}}
      every_n_train_steps: {config["steps"]}
      save_top_k: -1
      save_last: false
      save_weights_only: true
  trainer:
    max_steps: {config["steps"]}
resume_from: null
"""
    path.write_text(text.lstrip(), encoding="utf-8")


def make_fake_setup(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, list[Path]]]:
    samples_root = tmp_path / "samples"
    train_json, train_samples = write_examples(samples_root, "train", 2)
    val_json, val_samples = write_examples(samples_root, "val", 1)
    test_json, test_samples = write_examples(samples_root, "test", 1)
    pbr_dir = tmp_path / "hunyuan3d-paintpbr-v2-1"
    pbr_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints" / "datav2_frame_full80_truepbr_500_lr1e6"
    split_file = tmp_path / "full101_split.json"
    split_file.write_text(json.dumps({"train": ["a", "b"], "val": ["c"], "test": ["d"]}) + "\n", encoding="utf-8")

    config = {
        "experiment_name": "datav2_frame_full80_truepbr_500_lr1e6",
        "dataset_name": "datav2_frame_panels_full101",
        "training_yaml": str(tmp_path / "train.yaml"),
        "train_examples_json": str(train_json),
        "val_examples_json": str(val_json),
        "test_examples_json": str(test_json),
        "split_file": str(split_file),
        "train_count_expected": 2,
        "val_count_expected": 1,
        "test_count_expected": 1,
        "steps": 500,
        "learning_rate": "1e-6",
        "output_checkpoint_dir": str(checkpoint_dir),
        "official_pbr_source_dir": str(pbr_dir),
        "forbidden_checkpoint_markers": ["pilot_v1_overfit_500", "pilot_v1_truepbr", "sd2-community"],
        "readiness_report_dir": str(tmp_path / "reports"),
        "allow_existing_checkpoints": False,
    }
    write_training_yaml(Path(config["training_yaml"]), config)
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_json, config, {"train": train_samples, "val": val_samples, "test": test_samples}


def test_full80_training_readiness_passes_for_valid_fake_setup(tmp_path):
    config_json, _config, _samples = make_fake_setup(tmp_path)

    assert readiness.main(["--config", str(config_json)]) == 0
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["examples"]["train"]["actual_count"] == 2


def test_full80_training_readiness_fails_on_count_mismatch(tmp_path):
    config_json, config, _samples = make_fake_setup(tmp_path)
    config["train_count_expected"] = 3
    config_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    assert readiness.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("examples count 2 != expected 3" in error for error in report["errors"])


def test_full80_training_readiness_fails_on_missing_render_cond(tmp_path):
    config_json, _config, samples = make_fake_setup(tmp_path)
    shutil.rmtree(samples["train"][0] / "render_cond")

    assert readiness.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("render_cond missing" in error for error in report["errors"])


def test_full80_training_readiness_rejects_forbidden_training_yaml_marker(tmp_path):
    config_json, config, _samples = make_fake_setup(tmp_path)
    yaml_path = Path(config["training_yaml"])
    yaml_path.write_text(yaml_path.read_text(encoding="utf-8") + "wrong_init: sd2-community\n", encoding="utf-8")

    assert readiness.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("forbidden marker" in error for error in report["errors"])


def test_full80_training_readiness_fails_when_checkpoint_dir_has_ckpt(tmp_path):
    config_json, config, _samples = make_fake_setup(tmp_path)
    checkpoint_dir = Path(config["output_checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "old.ckpt").write_bytes(b"x")

    assert readiness.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("already contains .ckpt" in error for error in report["errors"])
