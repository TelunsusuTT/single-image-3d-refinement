from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2b_local_assets import main as check_assets_main  # noqa: E402
from make_phase2b_download_manifest import main as manifest_main  # noqa: E402
from summarize_phase2b_inspections import main as summarize_main  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_selected_csv(path: Path) -> None:
    write_csv(
        path,
        [
            "candidate_id",
            "source_id",
            "abo_path",
            "product_type",
            "name",
            "faces",
            "textures",
            "images",
            "selection_status",
            "selection_reason",
        ],
        [
            {
                "candidate_id": "abo_a",
                "source_id": "B000A",
                "abo_path": "A/B000A.glb",
                "product_type": "SIGN",
                "name": "Printed sign",
                "faces": "1000",
                "textures": "3",
                "images": "3",
                "selection_status": "selected",
                "selection_reason": "printed label",
            },
            {
                "candidate_id": "abo_b",
                "source_id": "B000B",
                "abo_path": "s3://amazon-berkeley-objects/3dmodels/original/B/B000B.glb",
                "product_type": "BOX",
                "name": "Package box",
                "faces": "2000",
                "textures": "4",
                "images": "4",
                "selection_status": "selected",
                "selection_reason": "packaging",
            },
        ],
    )


def test_make_phase2b_download_manifest_paths_and_urls() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        selected_csv = root / "selected.csv"
        out_csv = root / "manifest.csv"
        raw_dir = root / "raw"
        make_selected_csv(selected_csv)

        assert (
            manifest_main(
                [
                    "--selected-csv",
                    str(selected_csv),
                    "--out-csv",
                    str(out_csv),
                    "--raw-dir",
                    str(raw_dir),
                ]
            )
            == 0
        )

        rows = read_csv(out_csv)
        assert len(rows) == 2
        assert rows[0]["download_url"] == (
            "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/A/B000A.glb"
        )
        assert rows[0]["local_glb_path"] == str(raw_dir / "B000A.glb")
        assert rows[1]["download_url"] == (
            "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/B/B000B.glb"
        )
        assert rows[1]["local_glb_path"] == str(raw_dir / "B000B.glb")


def test_check_phase2b_local_assets_detects_existing_and_missing_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_csv = root / "manifest.csv"
        existing = root / "raw" / "exists.glb"
        missing = root / "raw" / "missing.glb"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"glb")
        write_csv(
            manifest_csv,
            ["candidate_id", "source_id", "local_glb_path"],
            [
                {"candidate_id": "ok", "source_id": "ok", "local_glb_path": str(existing)},
                {"candidate_id": "bad", "source_id": "bad", "local_glb_path": str(missing)},
            ],
        )

        assert check_assets_main(["--manifest-csv", str(manifest_csv)]) == 1

        missing.write_bytes(b"glb")
        assert check_assets_main(["--manifest-csv", str(manifest_csv)]) == 0


def write_inspection(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_summarize_phase2b_inspections_marks_pass_and_fail() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_csv = root / "manifest.csv"
        inspection_root = root / "inspections"
        out_csv = root / "summary.csv"
        out_md = root / "summary.md"
        write_csv(
            manifest_csv,
            ["candidate_id", "source_id", "local_glb_path", "name", "product_type", "selection_reason"],
            [
                {
                    "candidate_id": "pass",
                    "source_id": "PASS",
                    "local_glb_path": "PASS.glb",
                    "name": "good",
                    "product_type": "BOX",
                    "selection_reason": "label",
                },
                {
                    "candidate_id": "fail",
                    "source_id": "FAIL",
                    "local_glb_path": "FAIL.glb",
                    "name": "bad",
                    "product_type": "BOX",
                    "selection_reason": "label",
                },
            ],
        )
        write_inspection(
            inspection_root / "PASS" / "asset_inspection.json",
            {
                "import_status": "OK",
                "mesh_object_count": 1,
                "material_count": 2,
                "texture_image_count": 1,
                "total_polygons": 100,
                "total_vertices": 80,
                "per_object": [{"uv_layer_count": 1}],
            },
        )
        write_inspection(
            inspection_root / "FAIL" / "asset_inspection.json",
            {
                "import_status": "OK",
                "mesh_object_count": 1,
                "material_count": 0,
                "texture_image_count": 0,
                "total_uv_layers": 0,
            },
        )

        assert (
            summarize_main(
                [
                    "--inspection-root",
                    str(inspection_root),
                    "--manifest-csv",
                    str(manifest_csv),
                    "--out-csv",
                    str(out_csv),
                    "--out-md",
                    str(out_md),
                ]
            )
            == 0
        )

        rows = {row["source_id"]: row for row in read_csv(out_csv)}
        assert rows["PASS"]["pass"] == "yes"
        assert rows["PASS"]["total_uv_layers"] == "1"
        assert rows["FAIL"]["pass"] == "no"
        assert "material_count is 0" in rows["FAIL"]["reject_reason"]
        assert "texture_image_count is 0" in rows["FAIL"]["reject_reason"]
        assert "total_uv_layers is 0" in rows["FAIL"]["reject_reason"]
        assert "PASS" in out_md.read_text(encoding="utf-8")
