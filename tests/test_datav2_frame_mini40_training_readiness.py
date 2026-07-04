import csv
import json
import shutil
from pathlib import Path

from scripts import check_datav2_frame_mini40_training_readiness as readiness


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
    train:
      params:
        json_path: {config["train_examples_json"]}
    validation:
      params:
        json_path: {config["val_examples_json"]}
lightning:
  modelcheckpoint:
    params:
      dirpath: {config["output_checkpoint_dir"]}
      filename: datav2_frame_mini40_truepbr_500_lr1e6-step{{step}}
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
    train_json, train_samples = write_examples(samples_root, "train", 32)
    val_json, val_samples = write_examples(samples_root, "val", 4)
    test_json, test_samples = write_examples(samples_root, "test", 4)
    hypaint = tmp_path / "hypaint"
    hypaint.mkdir()
    (hypaint / "train.py").write_text("# fake train entry\n", encoding="utf-8")
    pbr_dir = tmp_path / "hunyuan3d-paintpbr-v2-1"
    pbr_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints" / "datav2_frame_mini40_truepbr_500_lr1e6"
    split_csv = tmp_path / "split_membership.csv"
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split_config", "split", "item_id"])
        writer.writeheader()
        for split, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
            for sample in samples:
                writer.writerow({"split_config": "mini40", "split": split, "item_id": sample.name})

    config = {
        "experiment_name": "datav2_frame_mini40_truepbr_500_lr1e6",
        "dataset_name": "datav2_frame_panels_mini40",
        "training_yaml": str(tmp_path / "train.yaml"),
        "train_examples_json": str(train_json),
        "val_examples_json": str(val_json),
        "test_examples_json": str(test_json),
        "train_count_expected": 32,
        "val_count_expected": 4,
        "test_count_expected": 4,
        "steps": 500,
        "learning_rate": "1e-6",
        "output_checkpoint_dir": str(checkpoint_dir),
        "official_pbr_source_dir": str(pbr_dir),
        "forbidden_checkpoint_markers": ["pilot_v1_overfit_500", "pilot_v1_truepbr"],
        "hypaint_dir": str(hypaint),
        "split_membership_csv": str(split_csv),
        "readiness_report_dir": str(tmp_path / "reports"),
    }
    write_training_yaml(Path(config["training_yaml"]), config)
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_json, config, {"train": train_samples, "val": val_samples, "test": test_samples}


def test_mini40_training_readiness_passes_for_valid_fake_setup(tmp_path):
    config_json, _config, _samples = make_fake_setup(tmp_path)

    assert readiness.main(["--config", str(config_json)]) == 0
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["examples"]["train"]["actual_count"] == 32


def test_mini40_training_readiness_fails_on_count_mismatch(tmp_path):
    config_json, config, _samples = make_fake_setup(tmp_path)
    config["train_count_expected"] = 33
    config_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    assert readiness.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("examples count 32 != expected 33" in error for error in report["errors"])


def test_mini40_training_readiness_fails_on_missing_render_cond(tmp_path):
    config_json, _config, samples = make_fake_setup(tmp_path)
    shutil.rmtree(samples["train"][0] / "render_cond")

    assert readiness.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("render_cond missing" in error for error in report["errors"])


def test_mini40_training_readiness_rejects_forbidden_checkpoint_marker(tmp_path):
    config_json, config, _samples = make_fake_setup(tmp_path)
    config["output_checkpoint_dir"] = str(tmp_path / "checkpoints" / "pilot_v1_truepbr")
    write_training_yaml(Path(config["training_yaml"]), config)
    config_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    assert readiness.main(["--config", str(config_json)]) == 1
    report = json.loads((tmp_path / "reports" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("forbidden marker" in error for error in report["errors"])
