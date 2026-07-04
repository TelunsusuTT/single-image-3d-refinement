from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_make_abo_probe_human_review_template import REVIEW_COLUMNS, main as review_main  # noqa: E402
from datav2_prepare_abo_probe_manifest import MANIFEST_COLUMNS  # noqa: E402


def write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "candidate_rank": "1",
                "asset_id": "panel",
                "metadata_path": "models/panel.glb",
                "expected_glb_relative_path": "models/panel.glb",
                "rank_score": "13.0",
                "flatness_ratio": "0.03",
                "panel_aspect_ratio": "1.5",
                "extents": "3 x 2 x 0.1",
                "extent_x": "3",
                "extent_y": "2",
                "extent_z": "0.1",
                "faces": "9000",
                "textures": "3",
                "materials": "2",
                "images": "2",
                "local_glb_exists": "yes",
                "local_glb_path": "/tmp/panel.glb",
                "needs_download": "no",
            }
        )


def test_probe_review_template_without_inspection_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_csv = root / "manifest.csv"
        inspection_csv = root / "missing_inspection.csv"
        out_csv = root / "review.csv"
        write_manifest(manifest_csv)

        assert (
            review_main(
                [
                    "--probe-manifest",
                    str(manifest_csv),
                    "--inspection-csv",
                    str(inspection_csv),
                    "--out-csv",
                    str(out_csv),
                ]
            )
            == 0
        )
        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert rows
        assert list(rows[0].keys()) == REVIEW_COLUMNS
        assert rows[0]["asset_id"] == "panel"
        assert rows[0]["contact_sheet_path"] == ""
        assert rows[0]["import_ok"] == ""
        assert rows[0]["human_decision"] == ""
        assert rows[0]["selected_input_view"] == ""
