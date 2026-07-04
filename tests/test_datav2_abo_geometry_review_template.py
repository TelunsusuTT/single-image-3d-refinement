from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_make_abo_geometry_review_template import REVIEW_COLUMNS, main as review_main  # noqa: E402
from datav2_mine_abo_geometry_candidates import CANDIDATE_COLUMNS  # noqa: E402


def test_abo_geometry_review_template_includes_required_curation_columns() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        candidates_csv = root / "candidates.csv"
        out_csv = root / "review.csv"

        with candidates_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "candidate_rank": "1",
                    "asset_id": "panel",
                    "path": "panel.glb",
                    "rank_score": "13.0",
                    "flatness_ratio": "0.033333",
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
                    "geometry_flags": "flatness_good;aspect_panel_like",
                    "technical_flags": "textures:3;materials:2",
                    "reject_flags": "",
                    "raw_metadata_json": "{}",
                }
            )

        assert (
            review_main(
                [
                    "--candidates-csv",
                    str(candidates_csv),
                    "--out-csv",
                    str(out_csv),
                    "--top-k",
                    "1",
                ]
            )
            == 0
        )
        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert rows
        assert list(rows[0].keys()) == REVIEW_COLUMNS
        assert rows[0]["asset_id"] == "panel"
        assert rows[0]["human_decision"] == ""
        assert rows[0]["selected_input_view"] == ""
        assert rows[0]["primary_eval_views"] == ""
