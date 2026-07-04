import csv
import json
from pathlib import Path

import pytest

from scripts import make_datav2_frame_mini40_input_view_review as review


VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def write_image(path: Path, color=(128, 128, 128)) -> None:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def write_split_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"split_config": "mini40", "split": "val", "item_id": "VAL1", "local_glb_path": "/tmp/VAL1.glb", "selected_input_view": "005"},
        {"split_config": "mini40", "split": "test", "item_id": "TEST1", "local_glb_path": "/tmp/TEST1.glb", "selected_input_view": "005"},
        {"split_config": "mini40", "split": "train", "item_id": "TRAIN1", "local_glb_path": "/tmp/TRAIN1.glb", "selected_input_view": "005"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split_config", "split", "item_id", "local_glb_path", "selected_input_view"])
        writer.writeheader()
        writer.writerows(rows)


def make_config(tmp_path: Path) -> Path:
    split_csv = tmp_path / "split_membership.csv"
    sample_root = tmp_path / "samples"
    write_split_csv(split_csv)
    for item_id in ("VAL1", "TEST1", "TRAIN1"):
        for view_id in VIEW_IDS:
            write_image(sample_root / item_id / "render_cond" / f"{view_id}_light_AL.png")
    config = {
        "experiment_name": "test_eval",
        "split_membership_csv": str(split_csv),
        "train_examples_root": str(sample_root),
        "output_root": str(tmp_path / "outputs"),
        "eval_splits": ["val", "test"],
        "optional_train_sanity_count": 1,
        "default_selected_input_view": "005",
        "alternative_input_view": "004",
        "primary_front_views": ["004", "005"],
        "input_view_override_csv": str(tmp_path / "overrides.csv"),
        "eval_view_ids": VIEW_IDS,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_input_view_review_creates_board_and_default_override_csv(tmp_path):
    config_path = make_config(tmp_path)

    assert review.main(["--config", str(config_path)]) == 0

    board = tmp_path / "outputs" / "input_view_review" / "input_view_review_board.jpg"
    summary = tmp_path / "outputs" / "input_view_review" / "input_view_review_summary.json"
    override_csv = tmp_path / "overrides.csv"
    assert board.is_file()
    assert summary.is_file()
    rows = list(csv.DictReader(override_csv.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 3
    assert {row["eval_split"] for row in rows} == {"val", "test", "train_sanity"}
    assert all(row["selected_input_view"] == "005" for row in rows)


def test_input_view_review_does_not_overwrite_existing_override_without_flag(tmp_path):
    config_path = make_config(tmp_path)
    override_csv = tmp_path / "overrides.csv"
    override_csv.write_text("item_id,selected_input_view\nVAL1,004\n", encoding="utf-8")

    assert review.main(["--config", str(config_path)]) == 0
    assert override_csv.read_text(encoding="utf-8") == "item_id,selected_input_view\nVAL1,004\n"
