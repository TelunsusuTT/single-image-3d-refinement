from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_prepare_abo_probe_manifest import main as manifest_main  # noqa: E402


GEOMETRY_COLUMNS = [
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


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_config(root: Path, candidates_csv: Path, manifest_csv: Path) -> Path:
    config = {
        "project_root": str(root),
        "input_candidates_csv": str(candidates_csv),
        "top_k": 2,
        "output_root": str(root / "outputs" / "phase2l" / "abo_probe"),
        "candidate_manifest": str(manifest_csv),
        "availability_report": {
            "md": str(root / "outputs" / "phase2l" / "abo_probe" / "availability_report.md"),
            "json": str(root / "outputs" / "phase2l" / "abo_probe" / "availability_report.json"),
        },
        "contact_sheet_root": str(root / "outputs" / "phase2l" / "abo_probe" / "contact_sheets"),
        "human_review_template": str(root / "data" / "candidates" / "review.csv"),
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_probe_manifest_top_k_and_local_asset_detection() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        candidates_csv = root / "data" / "candidates" / "geometry.csv"
        manifest_csv = root / "data" / "candidates" / "probe_manifest.csv"
        local_glb = root / "data" / "raw_assets" / "phase2b_abo_selected" / "panel.glb"
        local_glb.parent.mkdir(parents=True, exist_ok=True)
        local_glb.write_bytes(b"fake glb")

        write_csv(
            candidates_csv,
            [
                {
                    "candidate_rank": "1",
                    "asset_id": "panel",
                    "path": "models/panel.glb",
                    "rank_score": "13.0",
                    "flatness_ratio": "0.03",
                    "panel_aspect_ratio": "1.5",
                    "extent_x": "3",
                    "extent_y": "2",
                    "extent_z": "0.1",
                    "faces": "9000",
                    "vertices": "6000",
                    "meshes": "1",
                    "materials": "2",
                    "textures": "3",
                    "images": "2",
                    "image_width_max": "2048",
                    "image_height_max": "2048",
                    "geometry_flags": "",
                    "technical_flags": "",
                    "reject_flags": "",
                    "raw_metadata_json": "{}",
                },
                {
                    "candidate_rank": "2",
                    "asset_id": "missing",
                    "path": "models/missing.glb",
                    "rank_score": "12.0",
                    "flatness_ratio": "0.04",
                    "panel_aspect_ratio": "1.4",
                    "extent_x": "2.8",
                    "extent_y": "2",
                    "extent_z": "0.1",
                    "faces": "8000",
                    "vertices": "5000",
                    "meshes": "1",
                    "materials": "2",
                    "textures": "3",
                    "images": "2",
                    "image_width_max": "1024",
                    "image_height_max": "1024",
                    "geometry_flags": "",
                    "technical_flags": "",
                    "reject_flags": "",
                    "raw_metadata_json": "{}",
                },
                {
                    "candidate_rank": "3",
                    "asset_id": "not_in_top_k",
                    "path": "models/third.glb",
                    "rank_score": "11.0",
                    "flatness_ratio": "0.05",
                    "panel_aspect_ratio": "1.3",
                    "extent_x": "2.6",
                    "extent_y": "2",
                    "extent_z": "0.1",
                    "faces": "7000",
                    "vertices": "4000",
                    "meshes": "1",
                    "materials": "2",
                    "textures": "3",
                    "images": "2",
                    "image_width_max": "1024",
                    "image_height_max": "1024",
                    "geometry_flags": "",
                    "technical_flags": "",
                    "reject_flags": "",
                    "raw_metadata_json": "{}",
                },
            ],
            GEOMETRY_COLUMNS,
        )
        config = write_config(root, candidates_csv, manifest_csv)

        assert manifest_main(["--config", str(config)]) == 0
        rows = read_csv(manifest_csv)

        assert [row["asset_id"] for row in rows] == ["panel", "missing"]
        assert rows[0]["local_glb_exists"] == "yes"
        assert rows[0]["local_glb_path"] == str(local_glb)
        assert rows[0]["needs_download"] == "no"
        assert rows[1]["local_glb_exists"] == "no"
        assert rows[1]["needs_download"] == "yes"
        report = json.loads(
            (root / "outputs" / "phase2l" / "abo_probe" / "availability_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["local_glb_count"] == 1
        assert report["needs_download_count"] == 1
