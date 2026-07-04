from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_build_frame_panel_render_plan import main as plan_main  # noqa: E402


CURATED_COLUMNS = ["item_id", "local_glb_path", "selected_input_view"]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURATED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_config(root: Path, split_file: Path, curated_csv: Path) -> Path:
    config = {
        "dataset_name": "datav2_frame_panels_mini40",
        "split_file": str(split_file),
        "split_membership_csv": str(root / "membership.csv"),
        "curated_manifest_csv": str(curated_csv),
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


def test_render_plan_joins_split_and_curated_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        glb_a = root / "raw" / "ITEM_A.glb"
        glb_b = root / "raw" / "ITEM_B.glb"
        glb_a.parent.mkdir(parents=True, exist_ok=True)
        glb_a.write_bytes(b"fake glb")
        glb_b.write_bytes(b"fake glb")
        curated_csv = root / "curated.csv"
        write_csv(
            curated_csv,
            [
                {"item_id": "ITEM_A", "local_glb_path": str(glb_a), "selected_input_view": "005"},
                {"item_id": "ITEM_B", "local_glb_path": str(glb_b), "selected_input_view": "004"},
            ],
        )
        split_file = root / "split.json"
        split_file.write_text(
            json.dumps(
                {
                    "counts": {"train": 1, "val": 1, "test": 0},
                    "splits": {
                        "train": [{"item_id": "ITEM_A", "local_glb_path": str(glb_a), "selected_input_view": "005"}],
                        "val": [{"item_id": "ITEM_B", "local_glb_path": str(glb_b), "selected_input_view": "004"}],
                        "test": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        config = write_config(root, split_file, curated_csv)

        assert plan_main(["--config", str(config)]) == 0
        rows = read_csv(root / "reports" / "render_plan.csv")
        summary = json.loads((root / "reports" / "render_plan_summary.json").read_text(encoding="utf-8"))

        assert [row["item_id"] for row in rows] == ["ITEM_A", "ITEM_B"]
        assert rows[0]["split"] == "train"
        assert rows[0]["selected_input_view"] == "005"
        assert rows[0]["output_example_dir"] == str(root / "dataset" / "ITEM_A")
        assert rows[0]["qa_dir"] == str(root / "reports" / "qa" / "ITEM_A")
        assert summary["asset_count"] == 2
        assert summary["split_counts"] == {"train": 1, "val": 1, "test": 0}
