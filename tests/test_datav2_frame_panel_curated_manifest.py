from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_build_frame_panel_curated_manifest import main as curated_main  # noqa: E402


REVIEW_COLUMNS = [
    "item_id",
    "relative_path",
    "url",
    "local_glb_path",
    "exists_local",
    "status",
    "import_ok",
    "render_ok",
    "contact_sheet_page",
    "contact_sheet_path",
    "face_count",
    "mesh_count",
    "material_count",
    "texture_image_count",
    "bbox_extents",
    "human_decision",
    "reject_reason",
    "subclass",
    "selected_input_view",
    "alternative_input_view",
    "primary_eval_views",
    "front_quality_score",
    "texture_quality_score",
    "leakage_risk",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row(item_id: str, **updates: str) -> dict[str, str]:
    data = {
        "item_id": item_id,
        "relative_path": f"models/{item_id}.glb",
        "url": f"https://example.invalid/{item_id}.glb",
        "local_glb_path": f"/tmp/{item_id}.glb",
        "exists_local": "true",
        "status": "found",
        "import_ok": "yes",
        "render_ok": "yes",
        "contact_sheet_page": "contact_sheet_001.jpg",
        "contact_sheet_path": "/tmp/contact_sheet_001.jpg",
        "face_count": "1200",
        "mesh_count": "1",
        "material_count": "2",
        "texture_image_count": "3",
        "bbox_extents": "1.0 x 2.0 x 0.1",
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
    }
    data.update(updates)
    return data


def write_config(root: Path, review_csv: Path, reject_list: Path) -> Path:
    config = {
        "input_review_csv": str(review_csv),
        "optional_reject_ids": str(reject_list),
        "output_dir": str(root / "data" / "manifests" / "datav2_frame_panels"),
        "report_dir": str(root / "outputs" / "phase2l" / "datav2_frame_panels"),
        "dataset_name": "datav2_frame_panels",
        "default_subclass": "framed_wall_art",
        "default_selected_input_view": "005",
        "default_alternative_input_view": "004",
        "default_primary_eval_views": ["004", "005"],
        "random_seed": 260704,
        "mini_split": {"total": 40, "train": 32, "val": 4, "test": 4},
        "full_split": {"train": 80, "val": 10, "test": 11},
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_curated_manifest_filters_and_fills_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        review_csv = root / "review.csv"
        reject_list = root / "rejects.txt"
        reject_list.write_text("REJECTED_BY_LIST\n", encoding="utf-8")
        write_csv(
            review_csv,
            [
                row("GOOD"),
                row("MISSING", status="missing_in_metadata"),
                row("NOT_LOCAL", exists_local="false"),
                row("IMPORT_FAIL", import_ok="no"),
                row("RENDER_FAIL", render_ok="no"),
                row("REJECTED_BY_LIST"),
            ],
        )
        config = write_config(root, review_csv, reject_list)

        assert curated_main(["--config", str(config)]) == 0
        out_csv = root / "data" / "manifests" / "datav2_frame_panels" / "datav2_frame_panels_curated_manifest.csv"
        rows = read_csv(out_csv)
        summary = json.loads(
            (
                root
                / "outputs"
                / "phase2l"
                / "datav2_frame_panels"
                / "curated_manifest_summary.json"
            ).read_text(encoding="utf-8")
        )

        assert [item["item_id"] for item in rows] == ["GOOD"]
        assert rows[0]["human_decision"] == "accept"
        assert rows[0]["subclass"] == "framed_wall_art"
        assert rows[0]["selected_input_view"] == "005"
        assert rows[0]["alternative_input_view"] == "004"
        assert rows[0]["primary_eval_views"] == "004;005"
        assert rows[0]["group_key"]
        assert summary["counts"]["technical_fail_rows"] == 4
        assert summary["counts"]["optional_reject_rows"] == 1
