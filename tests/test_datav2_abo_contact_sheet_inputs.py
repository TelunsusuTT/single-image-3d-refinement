from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_make_abo_visual_contact_sheets import main as contact_main  # noqa: E402


INSPECTION_COLUMNS = [
    "dedup_rank",
    "candidate_rank",
    "asset_id",
    "local_glb_path",
    "import_ok",
    "render_ok",
    "object_count",
    "mesh_count",
    "material_count",
    "texture_count",
    "bbox_extent_x",
    "bbox_extent_y",
    "bbox_extent_z",
    "face_count",
    "render_dir",
    "error",
]


def write_inspection(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INSPECTION_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "dedup_rank": "1",
                "candidate_rank": "2",
                "asset_id": "asset_a",
                "local_glb_path": "/tmp/asset_a.glb",
                "import_ok": "yes",
                "render_ok": "yes",
                "object_count": "1",
                "mesh_count": "1",
                "material_count": "1",
                "texture_count": "3",
                "bbox_extent_x": "1",
                "bbox_extent_y": "1",
                "bbox_extent_z": "0.1",
                "face_count": "12000",
                "render_dir": "",
                "error": "",
            }
        )


def write_config(root: Path, inspection: Path) -> Path:
    config = {
        "input_manifest": str(root / "manifest.csv"),
        "output_root": str(root / "outputs"),
        "render_root": str(root / "outputs" / "renders"),
        "contact_sheet_root": str(root / "outputs" / "contact_sheets"),
        "inspection_csv": str(inspection),
        "inspection_summary_md": str(root / "outputs" / "inspection_summary.md"),
        "inspection_summary_json": str(root / "outputs" / "inspection_summary.json"),
        "human_review_csv": str(root / "review.csv"),
        "render_resolution": 512,
        "view_ids": ["000", "001", "002", "003", "004", "005"],
        "preferred_input_views": ["004", "005"],
        "blender_bin": "/vol/bitbucket/ct1022/tools/bin/blender",
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_contact_sheet_reports_missing_renders_cleanly(capsys) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        inspection = root / "inspection_results.csv"
        write_inspection(inspection)
        config = write_config(root, inspection)

        assert contact_main(["--config", str(config)]) == 1
        captured = capsys.readouterr()
        assert "ERROR: missing" in captured.err
        assert "rendered thumbnail" in captured.err
        assert "asset_a/000.png" in captured.err
