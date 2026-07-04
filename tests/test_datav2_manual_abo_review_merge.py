from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_update_manual_abo_review_with_inspection import (  # noqa: E402
    HUMAN_COLUMNS,
    OUTPUT_COLUMNS,
    main as merge_main,
)


MANIFEST_COLUMNS = ["item_id", "relative_path", "url", "local_glb_path", "exists_local", "status"]
INSPECTION_COLUMNS = [
    "item_id",
    "import_ok",
    "render_ok",
    "face_count",
    "mesh_count",
    "material_count",
    "texture_image_count",
    "bbox_extents",
]
REVIEW_COLUMNS = MANIFEST_COLUMNS + HUMAN_COLUMNS


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_config(root: Path, manifest: Path, review: Path) -> Path:
    config = {
        "manifest": str(manifest),
        "human_review_csv": str(review),
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


def test_review_merge_preserves_human_columns_and_missing_inspection_rows() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = root / "manifest.csv"
        review = root / "review.csv"
        inspection = root / "outputs" / "inspection_results.csv"
        out_csv = root / "review_with_inspection.csv"
        write_csv(
            manifest,
            MANIFEST_COLUMNS,
            [
                {
                    "item_id": "ITEM_A",
                    "relative_path": "models/ITEM_A.glb",
                    "url": "https://example.invalid/ITEM_A.glb",
                    "local_glb_path": "/tmp/ITEM_A.glb",
                    "exists_local": "true",
                    "status": "found",
                },
                {
                    "item_id": "ITEM_B",
                    "relative_path": "models/ITEM_B.glb",
                    "url": "https://example.invalid/ITEM_B.glb",
                    "local_glb_path": "/tmp/ITEM_B.glb",
                    "exists_local": "true",
                    "status": "found",
                },
            ],
        )
        write_csv(
            review,
            REVIEW_COLUMNS,
            [
                {
                    "item_id": "ITEM_A",
                    "relative_path": "models/ITEM_A.glb",
                    "url": "https://example.invalid/ITEM_A.glb",
                    "local_glb_path": "/tmp/ITEM_A.glb",
                    "exists_local": "true",
                    "status": "found",
                    "human_decision": "accept",
                    "reject_reason": "",
                    "subclass": "flat_rectangular_graphic_panel",
                    "selected_input_view": "004",
                    "alternative_input_view": "005",
                    "primary_eval_views": "000,001,002,003,004,005",
                    "front_quality_score": "5",
                    "texture_quality_score": "4",
                    "leakage_risk": "low",
                    "notes": "strong frame",
                },
                {
                    "item_id": "ITEM_B",
                    "relative_path": "models/ITEM_B.glb",
                    "url": "https://example.invalid/ITEM_B.glb",
                    "local_glb_path": "/tmp/ITEM_B.glb",
                    "exists_local": "true",
                    "status": "found",
                    "human_decision": "",
                    "reject_reason": "",
                    "subclass": "",
                    "selected_input_view": "",
                    "alternative_input_view": "",
                    "primary_eval_views": "",
                    "front_quality_score": "",
                    "texture_quality_score": "",
                    "leakage_risk": "",
                    "notes": "",
                },
            ],
        )
        write_csv(
            inspection,
            INSPECTION_COLUMNS,
            [
                {
                    "item_id": "ITEM_A",
                    "import_ok": "yes",
                    "render_ok": "yes",
                    "face_count": "1234",
                    "mesh_count": "1",
                    "material_count": "2",
                    "texture_image_count": "3",
                    "bbox_extents": "1.0 x 2.0 x 0.1",
                }
            ],
        )
        config = write_config(root, manifest, review)

        assert merge_main(["--config", str(config), "--out-csv", str(out_csv)]) == 0
        rows = read_csv(out_csv)

        assert list(rows[0].keys()) == OUTPUT_COLUMNS
        assert rows[0]["item_id"] == "ITEM_A"
        assert rows[0]["import_ok"] == "yes"
        assert rows[0]["render_ok"] == "yes"
        assert rows[0]["contact_sheet_page"] == "contact_sheet_001.jpg"
        assert rows[0]["face_count"] == "1234"
        assert rows[0]["human_decision"] == "accept"
        assert rows[0]["selected_input_view"] == "004"
        assert rows[0]["notes"] == "strong frame"
        assert rows[1]["item_id"] == "ITEM_B"
        assert rows[1]["import_ok"] == ""
        assert rows[1]["render_ok"] == ""
        assert rows[1]["human_decision"] == ""
