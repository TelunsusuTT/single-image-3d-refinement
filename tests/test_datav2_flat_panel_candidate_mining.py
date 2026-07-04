from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_mine_flat_panel_candidates import main as mine_main  # noqa: E402


def write_config(root: Path, metadata_paths: list[Path]) -> Path:
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_subclass": "flat_rectangular_graphic_panels",
                "metadata_paths": [str(item) for item in metadata_paths],
                "outputs": {
                    "candidates_csv": str(root / "candidates.csv"),
                    "candidates_md": str(root / "candidates.md"),
                    "candidate_summary_json": str(root / "summary.json"),
                },
                "positive_keywords": [
                    "poster",
                    "sign",
                    "plaque",
                    "wall art",
                    "canvas",
                    "framed",
                    "frame",
                    "print",
                    "painting",
                    "picture",
                    "artwork",
                    "board",
                    "panel",
                    "tile",
                    "label",
                    "package",
                    "packaging",
                    "box",
                    "cover",
                    "book cover",
                    "decorative",
                    "graphic",
                    "logo",
                ],
                "negative_keywords": ["bottle", "can", "mug", "chair", "sofa", "table", "vehicle"],
                "technical_filters": {
                    "prefer_file_extensions": ["glb", "gltf", "obj"],
                    "optional_min_file_size_bytes": None,
                    "optional_max_file_size_bytes": None,
                    "optional_min_face_count": None,
                    "optional_max_face_count": None,
                },
                "rank_weights": {
                    "positive_keyword_score": 4.0,
                    "negative_keyword_penalty": 5.0,
                    "texture_indicator_score": 2.0,
                    "material_indicator_score": 1.5,
                    "file_format_score": 1.0,
                    "geometry_sanity_score": 1.0,
                    "source_priority_score": 0.5,
                },
                "source_priority": {"unknown": 0.2},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_candidate_mining_ranks_flat_panel_above_negative_examples() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata = root / "metadata.csv"
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "uid",
                    "title",
                    "description",
                    "tags",
                    "format",
                    "texture_count",
                    "material_count",
                    "object_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "uid": "flat_1",
                    "title": "Framed wall art poster panel",
                    "description": "Decorative graphic print with label-like front",
                    "tags": "poster;wall art;panel",
                    "format": "glb",
                    "texture_count": "3",
                    "material_count": "2",
                    "object_count": "1",
                }
            )
            writer.writerow(
                {
                    "uid": "chair_1",
                    "title": "Plain chair and table set",
                    "description": "Furniture object",
                    "tags": "chair;table",
                    "format": "glb",
                    "texture_count": "3",
                    "material_count": "2",
                    "object_count": "1",
                }
            )
            writer.writerow({"uid": "minimal_1"})

        config_path = write_config(root, [metadata])
        assert mine_main(["--config", str(config_path), "--top-k", "10"]) == 0
        rows = read_rows(root / "candidates.csv")

        assert rows[0]["asset_id"] == "flat_1"
        by_id = {row["asset_id"]: row for row in rows}
        assert float(by_id["flat_1"]["rank_score"]) > float(by_id["chair_1"]["rank_score"])
        assert "chair" in by_id["chair_1"]["negative_keyword_hits"]
        assert "minimal_1" in by_id


def test_candidate_mining_dry_run_does_not_write_final_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata = root / "metadata.jsonl"
        metadata.write_text(
            json.dumps({"uid": "flat_1", "title": "Poster sign panel", "format": "glb"}) + "\n",
            encoding="utf-8",
        )
        config_path = write_config(root, [metadata])

        assert mine_main(["--config", str(config_path), "--dry-run"]) == 0
        assert not (root / "candidates.csv").exists()
        assert not (root / "candidates.md").exists()
        assert not (root / "summary.json").exists()
