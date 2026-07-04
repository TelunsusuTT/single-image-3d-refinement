from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_make_manual_abo_contact_sheets import main as contact_main  # noqa: E402


INSPECTION_COLUMNS = [
    "item_id",
    "relative_path",
    "local_glb_path",
    "import_ok",
    "render_ok",
    "object_count",
    "mesh_count",
    "material_count",
    "texture_image_count",
    "texture_count",
    "face_count",
    "bbox_extent_x",
    "bbox_extent_y",
    "bbox_extent_z",
    "bbox_extents",
    "render_dir",
    "error",
]


def write_inspection(path: Path, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INSPECTION_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "ITEM_A",
                "relative_path": "models/ITEM_A.glb",
                "local_glb_path": "/tmp/ITEM_A.glb",
                "import_ok": "yes",
                "render_ok": "yes",
                "object_count": "1",
                "mesh_count": "1",
                "material_count": "2",
                "texture_image_count": "3",
                "texture_count": "3",
                "face_count": "1200",
                "bbox_extent_x": "1.0",
                "bbox_extent_y": "2.0",
                "bbox_extent_z": "0.1",
                "bbox_extents": "1.0 x 2.0 x 0.1",
                "render_dir": str(output_root / "renders" / "ITEM_A"),
                "error": "",
            }
        )


def write_config(root: Path) -> Path:
    config = {
        "manifest": str(root / "manifest.csv"),
        "human_review_csv": str(root / "review.csv"),
        "output_root": str(root / "outputs"),
        "render_resolution": 512,
        "view_ids": ["000", "001", "002", "003", "004", "005"],
        "preferred_input_views": ["004", "005"],
        "contact_sheet_assets_per_page": 12,
        "blender_bin_default": "/vol/bitbucket/ct1022/tools/bin/blender",
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_contact_sheet_index_generation_handles_fake_thumbnails() -> None:
    Image = pytest.importorskip("PIL.Image")
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        output_root = root / "outputs"
        config = write_config(root)
        write_inspection(output_root / "inspection_results.csv", output_root)
        render_dir = output_root / "renders" / "ITEM_A"
        render_dir.mkdir(parents=True, exist_ok=True)
        for index, view_id in enumerate(["000", "001", "002", "003", "004", "005"]):
            image = Image.new("RGB", (24, 24), (index * 30, 40, 120))
            image.save(render_dir / f"{view_id}.png")

        assert contact_main(["--config", str(config)]) == 0
        index_path = output_root / "contact_sheets" / "contact_sheet_index.md"
        page_path = output_root / "contact_sheets" / "contact_sheet_001.jpg"
        index_text = index_path.read_text(encoding="utf-8")

        assert page_path.is_file()
        assert "ITEM_A" in index_text
        assert "contact_sheet_001.jpg" in index_text
        assert "assets with complete renders: `1`" in index_text
