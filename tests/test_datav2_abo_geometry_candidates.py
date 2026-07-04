from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import datav2_mine_abo_geometry_candidates as miner  # noqa: E402


FIELDNAMES = [
    "3dmodel_id",
    "path",
    "meshes",
    "materials",
    "textures",
    "images",
    "image_width_max",
    "image_height_max",
    "vertices",
    "faces",
    "extent_x",
    "extent_y",
    "extent_z",
]


def write_abo_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_config(root: Path, metadata_path: Path) -> Path:
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_subclass": "flat_rectangular_graphic_panels",
                "positive_keywords": ["poster", "sign", "panel"],
                "negative_keywords": ["bottle", "chair"],
                "abo_geometry_scoring": {
                    "enabled": True,
                    "candidate_source_files": [str(metadata_path)],
                    "auxiliary_source_files": [str(root / "images.csv.gz")],
                    "min_textures": 1,
                    "min_materials": 1,
                    "max_meshes": 3,
                    "min_faces": 500,
                    "max_faces": 150000,
                    "flatness_good_threshold": 0.12,
                    "flatness_ok_threshold": 0.20,
                    "aspect_min": 0.6,
                    "aspect_max": 3.5,
                    "thin_rod_penalty_aspect": 5.0,
                    "weights": {
                        "flatness_score": 4.0,
                        "rectangular_aspect_score": 3.0,
                        "texture_score": 2.0,
                        "material_score": 1.5,
                        "mesh_count_score": 1.0,
                        "face_count_score": 1.0,
                        "image_resolution_score": 0.5,
                        "semantic_keyword_score": 1.0,
                        "negative_keyword_penalty": 2.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def patch_outputs(monkeypatch, root: Path) -> tuple[Path, Path, Path]:
    out_csv = root / "out.csv"
    out_md = root / "out.md"
    out_json = root / "summary.json"
    monkeypatch.setattr(miner, "DEFAULT_OUT_CSV", out_csv)
    monkeypatch.setattr(miner, "DEFAULT_OUT_MD", out_md)
    monkeypatch.setattr(miner, "DEFAULT_OUT_JSON", out_json)
    return out_csv, out_md, out_json


def test_flat_rectangular_panel_ranks_above_cube_and_long_rod(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata = root / "3dmodels.csv.gz"
        write_abo_metadata(
            metadata,
            [
                {
                    "3dmodel_id": "panel",
                    "path": "panel.glb",
                    "meshes": 1,
                    "materials": 2,
                    "textures": 3,
                    "images": 2,
                    "image_width_max": 2048,
                    "image_height_max": 2048,
                    "vertices": 6000,
                    "faces": 9000,
                    "extent_x": 3.0,
                    "extent_y": 2.0,
                    "extent_z": 0.1,
                },
                {
                    "3dmodel_id": "cube",
                    "path": "cube.glb",
                    "meshes": 1,
                    "materials": 2,
                    "textures": 3,
                    "images": 2,
                    "image_width_max": 2048,
                    "image_height_max": 2048,
                    "vertices": 6000,
                    "faces": 9000,
                    "extent_x": 1.0,
                    "extent_y": 1.0,
                    "extent_z": 1.0,
                },
                {
                    "3dmodel_id": "rod",
                    "path": "rod.glb",
                    "meshes": 1,
                    "materials": 2,
                    "textures": 3,
                    "images": 2,
                    "image_width_max": 2048,
                    "image_height_max": 2048,
                    "vertices": 6000,
                    "faces": 9000,
                    "extent_x": 10.0,
                    "extent_y": 0.2,
                    "extent_z": 0.1,
                },
            ],
        )
        config = write_config(root, metadata)
        out_csv, _out_md, _out_json = patch_outputs(monkeypatch, root)

        assert miner.main(["--config", str(config), "--top-k", "10"]) == 0
        rows = read_rows(out_csv)

        assert rows[0]["asset_id"] == "panel"
        by_id = {row["asset_id"]: row for row in rows}
        assert float(by_id["panel"]["rank_score"]) > float(by_id["cube"]["rank_score"])
        assert float(by_id["panel"]["rank_score"]) > float(by_id["rod"]["rank_score"])
        assert "rod_like_aspect" in by_id["rod"]["reject_flags"]


def test_missing_extents_and_missing_materials_do_not_crash_and_lower_score(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata = root / "3dmodels.csv.gz"
        write_abo_metadata(
            metadata,
            [
                {
                    "3dmodel_id": "panel",
                    "path": "panel.glb",
                    "meshes": 1,
                    "materials": 2,
                    "textures": 3,
                    "images": 2,
                    "image_width_max": 2048,
                    "image_height_max": 2048,
                    "vertices": 6000,
                    "faces": 9000,
                    "extent_x": 3.0,
                    "extent_y": 2.0,
                    "extent_z": 0.1,
                },
                {
                    "3dmodel_id": "missing_extents",
                    "path": "missing.glb",
                    "meshes": 1,
                    "materials": 2,
                    "textures": 3,
                    "images": 2,
                    "image_width_max": 2048,
                    "image_height_max": 2048,
                    "vertices": 6000,
                    "faces": 9000,
                    "extent_x": "",
                    "extent_y": "",
                    "extent_z": "",
                },
                {
                    "3dmodel_id": "no_textures_or_materials",
                    "path": "no_tex.glb",
                    "meshes": 1,
                    "materials": 0,
                    "textures": 0,
                    "images": 0,
                    "image_width_max": "",
                    "image_height_max": "",
                    "vertices": 6000,
                    "faces": 9000,
                    "extent_x": 3.0,
                    "extent_y": 2.0,
                    "extent_z": 0.1,
                },
            ],
        )
        config = write_config(root, metadata)
        out_csv, _out_md, _out_json = patch_outputs(monkeypatch, root)

        assert miner.main(["--config", str(config), "--top-k", "10"]) == 0
        by_id = {row["asset_id"]: row for row in read_rows(out_csv)}

        assert "missing_extents" in by_id["missing_extents"]["reject_flags"]
        assert "missing_textures" in by_id["no_textures_or_materials"]["reject_flags"]
        assert "missing_materials" in by_id["no_textures_or_materials"]["reject_flags"]
        assert float(by_id["panel"]["rank_score"]) > float(by_id["no_textures_or_materials"]["rank_score"])


def test_dry_run_does_not_write_final_outputs(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata = root / "3dmodels.csv.gz"
        write_abo_metadata(
            metadata,
            [
                {
                    "3dmodel_id": "panel",
                    "path": "panel.glb",
                    "meshes": 1,
                    "materials": 2,
                    "textures": 3,
                    "images": 2,
                    "image_width_max": 2048,
                    "image_height_max": 2048,
                    "vertices": 6000,
                    "faces": 9000,
                    "extent_x": 3.0,
                    "extent_y": 2.0,
                    "extent_z": 0.1,
                }
            ],
        )
        config = write_config(root, metadata)
        out_csv, out_md, out_json = patch_outputs(monkeypatch, root)

        assert miner.main(["--config", str(config), "--dry-run"]) == 0
        assert not out_csv.exists()
        assert not out_md.exists()
        assert not out_json.exists()
