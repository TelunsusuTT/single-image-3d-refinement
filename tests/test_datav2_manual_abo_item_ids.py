from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_download_manual_abo_glbs import main as download_main  # noqa: E402
from datav2_make_manual_abo_review_template import REVIEW_COLUMNS, main as review_main  # noqa: E402
from datav2_resolve_manual_abo_item_ids import main as resolve_main  # noqa: E402


METADATA_COLUMNS = [
    "3dmodel_id",
    "path",
    "faces",
    "vertices",
    "meshes",
    "materials",
    "textures",
    "images",
    "image_width_max",
    "image_height_max",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "3dmodel_id": "ITEM_A",
                    "path": "models/ITEM_A.glb",
                    "faces": "1000",
                    "vertices": "700",
                    "meshes": "1",
                    "materials": "2",
                    "textures": "3",
                    "images": "4",
                    "image_width_max": "1024",
                    "image_height_max": "1024",
                },
                {
                    "3dmodel_id": "ITEM_B",
                    "path": "3dmodels/original/nested/ITEM_B.glb",
                    "faces": "2000",
                    "vertices": "1200",
                    "meshes": "1",
                    "materials": "1",
                    "textures": "2",
                    "images": "2",
                    "image_width_max": "2048",
                    "image_height_max": "1024",
                },
            ]
        )


def test_manual_item_ids_resolve_comments_missing_and_dedup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        item_ids = root / "item_ids.txt"
        metadata = root / "3dmodels.csv.gz"
        manifest = root / "manifest.csv"
        summary_json = root / "summary.json"
        summary_md = root / "summary.md"
        item_ids.write_text(
            "\n".join(
                [
                    "# comment",
                    "",
                    "ITEM_A # useful panel",
                    "ITEM_A",
                    "MISSING_ID",
                    "ITEM_B",
                ]
            ),
            encoding="utf-8",
        )
        write_metadata(metadata)

        assert (
            resolve_main(
                [
                    "--item-ids",
                    str(item_ids),
                    "--metadata",
                    str(metadata),
                    "--out-csv",
                    str(manifest),
                    "--project-root",
                    str(root),
                    "--summary-json",
                    str(summary_json),
                    "--summary-md",
                    str(summary_md),
                ]
            )
            == 0
        )
        rows = read_csv(manifest)
        summary = json.loads(summary_json.read_text(encoding="utf-8"))

        assert [row["item_id"] for row in rows] == ["ITEM_A", "MISSING_ID", "ITEM_B"]
        assert rows[0]["status"] == "found"
        assert rows[0]["relative_path"] == "models/ITEM_A.glb"
        assert rows[0]["url"].endswith("/3dmodels/original/models/ITEM_A.glb")
        assert rows[0]["local_glb_path"] == str(root / "data" / "raw_assets" / "abo" / "models" / "ITEM_A.glb")
        assert rows[1]["status"] == "missing_in_metadata"
        assert rows[1]["url"] == ""
        assert rows[2]["relative_path"] == "nested/ITEM_B.glb"
        assert summary["unique_item_ids"] == 3
        assert summary["found_count"] == 2
        assert summary["missing_in_metadata_count"] == 1
        assert summary_md.is_file()


def test_download_dry_run_does_not_write_glb() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = root / "manifest.csv"
        out_root = root / "data" / "raw_assets" / "abo"
        summary_json = root / "download_summary.json"
        summary_md = root / "download_summary.md"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "relative_path", "url", "local_glb_path", "exists_local", "status"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "item_id": "ITEM_A",
                    "relative_path": "models/ITEM_A.glb",
                    "url": "https://example.invalid/models/ITEM_A.glb",
                    "local_glb_path": str(out_root / "models" / "ITEM_A.glb"),
                    "exists_local": "false",
                    "status": "found",
                }
            )
            writer.writerow(
                {
                    "item_id": "MISSING_ID",
                    "relative_path": "",
                    "url": "",
                    "local_glb_path": "",
                    "exists_local": "false",
                    "status": "missing_in_metadata",
                }
            )

        assert (
            download_main(
                [
                    "--manifest",
                    str(manifest),
                    "--out-root",
                    str(out_root),
                    "--dry-run",
                    "--project-root",
                    str(root),
                    "--summary-json",
                    str(summary_json),
                    "--summary-md",
                    str(summary_md),
                ]
            )
            == 0
        )
        assert not (out_root / "models" / "ITEM_A.glb").exists()
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert payload["counts"]["planned_download_count"] == 1
        assert payload["counts"]["downloaded_count"] == 0
        assert payload["counts"]["missing_in_metadata_rows"] == 1
        assert summary_md.is_file()


def test_review_template_contains_required_curation_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = root / "manifest.csv"
        review_csv = root / "review.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "relative_path", "url", "local_glb_path", "exists_local", "status"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "item_id": "ITEM_A",
                    "relative_path": "models/ITEM_A.glb",
                    "url": "https://example.invalid/models/ITEM_A.glb",
                    "local_glb_path": str(root / "ITEM_A.glb"),
                    "exists_local": "false",
                    "status": "found",
                }
            )

        assert review_main(["--manifest", str(manifest), "--out-csv", str(review_csv)]) == 0
        rows = read_csv(review_csv)

        assert rows[0]["item_id"] == "ITEM_A"
        assert rows[0]["human_decision"] == ""
        assert rows[0]["selected_input_view"] == ""
        assert rows[0]["front_quality_score"] == ""
        assert list(rows[0].keys()) == REVIEW_COLUMNS
