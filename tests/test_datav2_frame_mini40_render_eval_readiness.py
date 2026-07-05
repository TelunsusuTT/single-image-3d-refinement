import json
from pathlib import Path

from scripts import check_datav2_frame_mini40_render_eval_readiness as readiness


VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def make_setup(tmp_path: Path, missing_fine: bool = False) -> Path:
    output_root = tmp_path / "outputs"
    sample_root = tmp_path / "samples"
    blender = tmp_path / "blender"
    blender.write_bytes(b"fake blender")
    base_glb = output_root / "infer" / "base" / "val" / "VAL1" / "base_textured_mesh.glb"
    fine_glb = output_root / "infer" / "finetuned" / "val" / "VAL1" / "finetuned_textured_mesh.glb"
    base_glb.parent.mkdir(parents=True)
    fine_glb.parent.mkdir(parents=True)
    base_glb.write_bytes(b"base")
    if not missing_fine:
        fine_glb.write_bytes(b"fine")
    refs = {}
    for view_id in VIEW_IDS:
        ref = sample_root / "VAL1" / "render_cond" / f"{view_id}_light_AL.png"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_bytes(b"ref")
        refs[view_id] = str(ref)
    render_cases = {
        "cases": {
            "VAL1": {
                "eval_split": "val",
                "selected_input_view": "005",
                "base_glb": str(base_glb),
                "finetuned_glb": str(fine_glb),
                "reference_images": refs,
            }
        }
    }
    render_root = output_root / "render_eval"
    render_root.mkdir(parents=True)
    (render_root / "render_eval_cases.json").write_text(json.dumps(render_cases, indent=2) + "\n", encoding="utf-8")
    config = {"output_root": str(output_root), "blender_bin": str(blender)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_render_eval_readiness_passes_for_fake_outputs(tmp_path):
    config_path = make_setup(tmp_path)

    assert readiness.main(["--config", str(config_path)]) == 0
    report = json.loads((tmp_path / "outputs" / "render_eval" / "render_eval_readiness_summary.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["case_count"] == 1


def test_render_eval_readiness_fails_missing_finetuned_glb(tmp_path):
    config_path = make_setup(tmp_path, missing_fine=True)

    assert readiness.main(["--config", str(config_path)]) == 1
    report = json.loads((tmp_path / "outputs" / "render_eval" / "render_eval_readiness_summary.json").read_text(encoding="utf-8"))
    assert any("finetuned GLB missing" in error for error in report["errors"])
