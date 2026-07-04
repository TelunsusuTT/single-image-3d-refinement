from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_dedupe_abo_probe_candidates import main as dedupe_main  # noqa: E402
from datav2_make_abo_dedup_download_manifest import main as manifest_main  # noqa: E402


INPUT_COLUMNS = [
    "candidate_rank",
    "asset_id",
    "path",
    "rank_score",
    "flatness_ratio",
    "panel_aspect_ratio",
    "extent_x",
    "extent_y",
    "extent_z",
    "faces",
    "vertices",
    "meshes",
    "materials",
    "textures",
    "images",
    "image_width_max",
    "image_height_max",
    "geometry_flags",
    "technical_flags",
    "reject_flags",
    "raw_metadata_json",
]


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_row(
    rank: int,
    asset_id: str,
    path: str,
    faces: str,
    width: str,
    extent_x: str = "3.001",
    extent_y: str = "2.002",
    extent_z: str = "0.101",
) -> dict[str, str]:
    return {
        "candidate_rank": str(rank),
        "asset_id": asset_id,
        "path": path,
        "rank_score": "13.0",
        "flatness_ratio": "0.034",
        "panel_aspect_ratio": "1.499",
        "extent_x": extent_x,
        "extent_y": extent_y,
        "extent_z": extent_z,
        "faces": faces,
        "vertices": "6000",
        "meshes": "1",
        "materials": "1",
        "textures": "3",
        "images": "3",
        "image_width_max": width,
        "image_height_max": width,
        "geometry_flags": "aspect_panel_like;flatness_good",
        "technical_flags": "textures:3;materials:1",
        "reject_flags": "",
        "raw_metadata_json": "{}",
    }


def write_config(root: Path, input_csv: Path) -> Path:
    config = {
        "input_candidates_csv": str(input_csv),
        "target_count": 2,
        "max_input_rows": 10,
        "dedup_candidates_csv": str(root / "dedup.csv"),
        "download_manifest_csv": str(root / "download_manifest.csv"),
        "summary_md": str(root / "summary.md"),
        "summary_json": str(root / "summary.json"),
        "raw_asset_dir": str(root / "raw_assets" / "abo"),
        "dedupe": {
            "extent_round_digits": 1,
            "ratio_round_digits": 1,
            "face_bucket_size": 5000,
            "texture_bucket_size": 1,
            "material_bucket_size": 1,
            "moderate_face_target": 25000,
        },
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_dedup_prefers_earlier_rank_and_outputs_summary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_csv = root / "input.csv"
        write_csv(
            input_csv,
            [
                make_row(1, "first", "A/first.glb", "24000", "1024"),
                make_row(2, "duplicate_later", "A/duplicate.glb", "25000", "4096"),
                make_row(3, "unique", "B/unique.glb", "30000", "4096", extent_x="4.4"),
            ],
            INPUT_COLUMNS,
        )
        config_path = write_config(root, input_csv)

        assert dedupe_main(["--config", str(config_path)]) == 0
        rows = read_csv(root / "dedup.csv")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        assert [row["asset_id"] for row in rows] == ["first", "unique"]
        assert rows[0]["duplicate_group_size"] == "2"
        assert summary["duplicate_rows_removed"] == 1
        assert summary["selected_count"] == 2
        assert (root / "summary.md").is_file()


def test_download_manifest_has_abo_url_and_local_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_csv = root / "input.csv"
        write_csv(input_csv, [make_row(1, "first", "A/first.glb", "24000", "1024")], INPUT_COLUMNS)
        config_path = write_config(root, input_csv)

        assert dedupe_main(["--config", str(config_path)]) == 0
        assert manifest_main(["--config", str(config_path)]) == 0
        rows = read_csv(root / "download_manifest.csv")

        assert len(rows) == 1
        assert rows[0]["asset_id"] == "first"
        assert rows[0]["abo_path"] == "A/first.glb"
        assert rows[0]["download_url"].endswith("/3dmodels/original/A/first.glb")
        assert rows[0]["local_glb_path"] == str(root / "raw_assets" / "abo" / "A" / "first.glb")
