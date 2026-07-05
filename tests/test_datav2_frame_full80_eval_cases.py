import csv
import json
from pathlib import Path

from scripts import make_datav2_frame_full80_eval_cases as cases_script


VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_case_setup(tmp_path: Path) -> Path:
    sample_root = tmp_path / "samples"
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    for item_id in ("VAL1", "TEST1", "TRAIN1"):
        (mesh_dir / f"{item_id}.glb").write_bytes(b"mesh")
        render_cond = sample_root / item_id / "render_cond"
        render_cond.mkdir(parents=True)
        for view_id in VIEW_IDS:
            (render_cond / f"{view_id}_light_AL.png").write_bytes(b"image")
    split_csv = tmp_path / "split.csv"
    curated_csv = tmp_path / "curated.csv"
    override_csv = tmp_path / "overrides.csv"
    split_rows = [
        {"split_config": "full101", "split": "val", "item_id": "VAL1", "local_glb_path": str(mesh_dir / "VAL1.glb")},
        {"split_config": "full101", "split": "test", "item_id": "TEST1", "local_glb_path": str(mesh_dir / "TEST1.glb")},
        {"split_config": "full101", "split": "train", "item_id": "TRAIN1", "local_glb_path": str(mesh_dir / "TRAIN1.glb")},
        {"split_config": "mini40", "split": "val", "item_id": "MINI_VAL", "local_glb_path": str(mesh_dir / "VAL1.glb")},
    ]
    curated_rows = [
        {"item_id": "VAL1", "local_glb_path": str(mesh_dir / "VAL1.glb"), "selected_input_view": "005", "primary_eval_views": "004;005"},
        {"item_id": "TEST1", "local_glb_path": str(mesh_dir / "TEST1.glb"), "selected_input_view": "005", "primary_eval_views": "004;005"},
        {"item_id": "TRAIN1", "local_glb_path": str(mesh_dir / "TRAIN1.glb"), "selected_input_view": "005", "primary_eval_views": "004;005"},
    ]
    override_rows = [{"item_id": "VAL1", "selected_input_view": "004", "primary_eval_views": "004;005"}]
    write_csv(split_csv, split_rows, ["split_config", "split", "item_id", "local_glb_path"])
    write_csv(curated_csv, curated_rows, ["item_id", "local_glb_path", "selected_input_view", "primary_eval_views"])
    write_csv(override_csv, override_rows, ["item_id", "selected_input_view", "primary_eval_views"])
    config = {
        "experiment_name": "full80_eval",
        "split_config": "full101",
        "curated_manifest_csv": str(curated_csv),
        "split_membership_csv": str(split_csv),
        "input_view_override_csv": str(override_csv),
        "train_examples_root": str(sample_root),
        "output_root": str(tmp_path / "outputs"),
        "eval_splits": ["val", "test"],
        "optional_train_sanity_count": 1,
        "default_selected_input_view": "005",
        "eval_view_ids": VIEW_IDS,
        "primary_front_views": ["004", "005"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_eval_cases_use_override_then_curated_manifest_and_full101_split(tmp_path):
    config_path = make_case_setup(tmp_path)

    assert cases_script.main(["--config", str(config_path)]) == 0

    data = json.loads((tmp_path / "outputs" / "eval_cases.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "outputs" / "eval_cases_summary.json").read_text(encoding="utf-8"))
    by_item = {case["item_id"]: case for case in data["cases"]}
    assert data["case_count"] == 3
    assert summary["primary_eval_case_count"] == 2
    assert "MINI_VAL" not in by_item
    assert by_item["VAL1"]["selected_input_view"] == "004"
    assert by_item["VAL1"]["selected_input_view_source"] == "override_csv"
    assert by_item["TEST1"]["selected_input_view_source"] == "curated_manifest"
    assert Path(by_item["VAL1"]["case_input_mesh"]).is_file()
    assert Path(by_item["VAL1"]["case_input_image"]).is_file()
    assert by_item["TRAIN1"]["eval_split"] == "train_sanity"
