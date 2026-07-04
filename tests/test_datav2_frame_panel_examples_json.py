from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_build_frame_panel_examples_json import main as examples_main  # noqa: E402


RESULT_COLUMNS = ["split", "item_id", "sample_name", "sample_dir", "qa_dir", "selected_input_view", "status", "error"]


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_config(root: Path) -> Path:
    config = {
        "dataset_name": "datav2_frame_panels_mini40",
        "split_file": str(root / "split.json"),
        "split_membership_csv": str(root / "membership.csv"),
        "curated_manifest_csv": str(root / "curated.csv"),
        "output_dataset_root": str(root / "dataset"),
        "report_root": str(root / "reports"),
        "view_ids": ["000", "001", "002", "003", "004", "005"],
        "light_conditions": ["AL", "ENVMAP", "PL"],
        "render_resolution": 512,
        "selected_input_view_field": "selected_input_view",
        "background_rgb": [71, 71, 71],
        "blender_bin_default": "/vol/bitbucket/ct1022/tools/bin/blender",
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_examples_json_uses_successful_render_results_only() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_train = root / "dataset" / "ITEM_A"
        sample_val = root / "dataset" / "ITEM_B"
        sample_fail = root / "dataset" / "ITEM_C"
        write_results(
            root / "reports" / "render_results.csv",
            [
                {
                    "split": "train",
                    "item_id": "ITEM_A",
                    "sample_name": "ITEM_A",
                    "sample_dir": str(sample_train),
                    "qa_dir": "",
                    "selected_input_view": "005",
                    "status": "rendered",
                    "error": "",
                },
                {
                    "split": "val",
                    "item_id": "ITEM_B",
                    "sample_name": "ITEM_B",
                    "sample_dir": str(sample_val),
                    "qa_dir": "",
                    "selected_input_view": "005",
                    "status": "skipped_complete",
                    "error": "",
                },
                {
                    "split": "test",
                    "item_id": "ITEM_C",
                    "sample_name": "ITEM_C",
                    "sample_dir": str(sample_fail),
                    "qa_dir": "",
                    "selected_input_view": "005",
                    "status": "failed",
                    "error": "boom",
                },
            ],
        )
        config = write_config(root)

        assert examples_main(["--config", str(config)]) == 0
        train = json.loads((root / "dataset" / "examples_train_abs.json").read_text(encoding="utf-8"))
        val = json.loads((root / "dataset" / "examples_val_abs.json").read_text(encoding="utf-8"))
        test = json.loads((root / "dataset" / "examples_test_abs.json").read_text(encoding="utf-8"))
        all_examples = json.loads((root / "dataset" / "examples_all_abs.json").read_text(encoding="utf-8"))

        assert train == [str(sample_train.resolve())]
        assert val == [str(sample_val.resolve())]
        assert test == []
        assert all_examples == sorted([str(sample_train.resolve()), str(sample_val.resolve())])
