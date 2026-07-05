import json
from pathlib import Path

from scripts import make_datav2_frame_mini40_render_eval_configs as make_config


VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def make_setup(tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"
    sample_root = tmp_path / "samples"
    for item_id in ("VAL1", "TEST1"):
        render_cond = sample_root / item_id / "render_cond"
        render_cond.mkdir(parents=True)
        for view_id in VIEW_IDS:
            (render_cond / f"{view_id}_light_AL.png").write_bytes(b"ref")
    eval_cases = {
        "cases": [
            {"item_id": "VAL1", "eval_split": "val", "source_split": "val", "selected_input_view": "005", "primary_eval_views": ["004", "005"]},
            {"item_id": "TEST1", "eval_split": "test", "source_split": "test", "selected_input_view": "004", "primary_eval_views": ["004", "005"]},
        ]
    }
    output_root.mkdir(parents=True)
    (output_root / "eval_cases.json").write_text(json.dumps(eval_cases, indent=2) + "\n", encoding="utf-8")
    config = {
        "experiment_name": "test_eval",
        "output_root": str(output_root),
        "train_examples_root": str(sample_root),
        "eval_view_ids": VIEW_IDS,
        "primary_front_views": ["004", "005"],
        "resolution": 512,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_render_eval_cases_include_split_paths_and_references(tmp_path):
    config_path = make_setup(tmp_path)

    assert make_config.main(["--config", str(config_path)]) == 0

    out = tmp_path / "outputs" / "render_eval" / "render_eval_cases.json"
    summary = tmp_path / "outputs" / "render_eval" / "render_eval_cases_summary.md"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert summary.is_file()
    assert data["cases"]["VAL1"]["base_glb"].endswith("infer/base/val/VAL1/base_textured_mesh.glb")
    assert data["cases"]["TEST1"]["finetuned_glb"].endswith("infer/finetuned/test/TEST1/finetuned_textured_mesh.glb")
    assert data["cases"]["VAL1"]["reference_images"]["005"].endswith("samples/VAL1/render_cond/005_light_AL.png")
