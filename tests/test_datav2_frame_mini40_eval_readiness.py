import json
from pathlib import Path

from scripts import check_datav2_frame_mini40_eval_readiness as readiness


VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def make_readiness_setup(tmp_path: Path, include_override: bool = True) -> Path:
    output_root = tmp_path / "outputs"
    sample_root = tmp_path / "samples"
    case_dir = output_root / "cases" / "val" / "VAL1"
    checkpoint = tmp_path / "datav2_frame_mini40_truepbr_500_lr1e6-stepstep=500.ckpt"
    mesh = tmp_path / "VAL1.glb"
    checkpoint.write_bytes(b"ckpt")
    mesh.write_bytes(b"mesh")
    render_cond = sample_root / "VAL1" / "render_cond"
    render_cond.mkdir(parents=True)
    for view_id in VIEW_IDS:
        (render_cond / f"{view_id}_light_AL.png").write_bytes(b"image")
    case_input = case_dir / "input"
    case_input.mkdir(parents=True)
    (case_input / "mesh.glb").write_bytes(b"mesh")
    (case_input / "image.png").write_bytes(b"image")
    override_csv = tmp_path / "overrides.csv"
    if include_override:
        override_csv.write_text("item_id,selected_input_view\nVAL1,005\n", encoding="utf-8")
    eval_cases = {
        "cases": [
            {
                "item_id": "VAL1",
                "eval_split": "val",
                "source_split": "val",
                "local_mesh_path": str(mesh),
                "case_input_mesh": str(case_input / "mesh.glb"),
                "case_input_image": str(case_input / "image.png"),
                "selected_input_view": "005",
                "selected_input_image": str(render_cond / "005_light_AL.png"),
                "reference_images": {view_id: str(render_cond / f"{view_id}_light_AL.png") for view_id in VIEW_IDS},
            }
        ]
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "eval_cases.json").write_text(json.dumps(eval_cases, indent=2) + "\n", encoding="utf-8")
    config = {
        "experiment_name": "test_eval",
        "checkpoint_path": str(checkpoint),
        "checkpoint_expected_step": 500,
        "input_view_override_csv": str(override_csv),
        "output_root": str(output_root),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_eval_readiness_passes_for_valid_fake_case(tmp_path):
    config_path = make_readiness_setup(tmp_path)

    assert readiness.main(["--config", str(config_path)]) == 0
    report = json.loads((tmp_path / "outputs" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["case_count"] == 1


def test_eval_readiness_fails_when_override_csv_missing(tmp_path):
    config_path = make_readiness_setup(tmp_path, include_override=False)

    assert readiness.main(["--config", str(config_path)]) == 1
    report = json.loads((tmp_path / "outputs" / "readiness_summary.json").read_text(encoding="utf-8"))
    assert any("input view override CSV missing" in error for error in report["errors"])
