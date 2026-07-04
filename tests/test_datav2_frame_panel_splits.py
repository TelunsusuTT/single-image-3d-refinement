from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_make_frame_panel_splits import main as splits_main  # noqa: E402


CURATED_COLUMNS = [
    "dataset_name",
    "item_id",
    "source",
    "relative_path",
    "url",
    "local_glb_path",
    "exists_local",
    "status",
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
    "contact_sheet_page",
    "contact_sheet_path",
    "face_count",
    "mesh_count",
    "material_count",
    "texture_image_count",
    "bbox_extents",
    "group_key",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURATED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def curated_row(index: int) -> dict[str, str]:
    item_id = f"ITEM_{index:03d}"
    return {
        "dataset_name": "datav2_frame_panels",
        "item_id": item_id,
        "source": "ABO",
        "relative_path": f"models/{item_id}.glb",
        "url": f"https://example.invalid/{item_id}.glb",
        "local_glb_path": f"/tmp/{item_id}.glb",
        "exists_local": "true",
        "status": "found",
        "human_decision": "accept",
        "reject_reason": "",
        "subclass": "framed_wall_art",
        "selected_input_view": "005",
        "alternative_input_view": "004",
        "primary_eval_views": "004;005",
        "front_quality_score": "",
        "texture_quality_score": "",
        "leakage_risk": "",
        "notes": "",
        "contact_sheet_page": "contact_sheet_001.jpg",
        "contact_sheet_path": "/tmp/contact_sheet_001.jpg",
        "face_count": str(1000 + index),
        "mesh_count": "1",
        "material_count": "2",
        "texture_image_count": "3",
        "bbox_extents": "1.0 x 2.0 x 0.1",
        "group_key": f"group_{index:03d}",
    }


def write_config(root: Path) -> Path:
    config = {
        "input_review_csv": str(root / "unused_review.csv"),
        "optional_reject_ids": str(root / "rejects.txt"),
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


def split_item_ids(payload: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for split_rows in payload["splits"].values():  # type: ignore[union-attr]
        for row in split_rows:
            ids.add(row["item_id"])
    return ids


def assert_no_overlap(payload: dict[str, object]) -> None:
    seen: dict[str, str] = {}
    for split, split_rows in payload["splits"].items():  # type: ignore[union-attr]
        for row in split_rows:
            item_id = row["item_id"]
            assert item_id not in seen
            seen[item_id] = split


def test_frame_panel_splits_are_reproducible_and_sized() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_dir = root / "data" / "manifests" / "datav2_frame_panels"
        curated_csv = manifest_dir / "datav2_frame_panels_curated_manifest.csv"
        write_csv(curated_csv, [curated_row(index) for index in range(101)])
        config = write_config(root)

        assert splits_main(["--config", str(config)]) == 0
        mini_path = manifest_dir / "datav2_frame_panels_mini40_split.json"
        full_path = manifest_dir / "datav2_frame_panels_full101_split.json"
        membership_path = manifest_dir / "datav2_frame_panels_split_membership.csv"
        mini_first = json.loads(mini_path.read_text(encoding="utf-8"))
        full_first = json.loads(full_path.read_text(encoding="utf-8"))
        membership_first = membership_path.read_text(encoding="utf-8")

        assert splits_main(["--config", str(config)]) == 0
        mini_second = json.loads(mini_path.read_text(encoding="utf-8"))
        full_second = json.loads(full_path.read_text(encoding="utf-8"))
        membership_second = membership_path.read_text(encoding="utf-8")

        assert mini_first == mini_second
        assert full_first == full_second
        assert membership_first == membership_second
        assert mini_first["counts"] == {"train": 32, "val": 4, "test": 4}
        assert full_first["counts"] == {"train": 80, "val": 10, "test": 11}
        assert len(split_item_ids(mini_first)) == 40
        assert len(split_item_ids(full_first)) == 101
        assert_no_overlap(mini_first)
        assert_no_overlap(full_first)
