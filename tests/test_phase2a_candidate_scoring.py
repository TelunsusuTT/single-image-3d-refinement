from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import download_abo_candidate_thumbnails as thumbnail_downloader  # noqa: E402
from build_abo_candidate_index import CANDIDATE_COLUMNS, main as build_main  # noqa: E402
from mark_phase2a_candidates import main as mark_main  # noqa: E402


def write_csv_gz(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def make_fake_metadata(root: Path) -> Path:
    metadata_dir = root / "metadata"
    write_csv_gz(
        metadata_dir / "3dmodels.csv.gz",
        [
            "3dmodel_id",
            "path",
            "texture_count",
            "image_count",
            "max_texture_width",
            "max_texture_height",
            "material_count",
            "face_count",
            "vertex_count",
            "extent_x",
            "extent_y",
            "extent_z",
        ],
        [
            {
                "3dmodel_id": "m_box",
                "path": "m_box.glb",
                "texture_count": 6,
                "image_count": 5,
                "max_texture_width": 4096,
                "max_texture_height": 4096,
                "material_count": 3,
                "face_count": 12000,
                "vertex_count": 7000,
                "extent_x": 1.0,
                "extent_y": 1.2,
                "extent_z": 0.4,
            },
            {
                "3dmodel_id": "m_shelf",
                "path": "m_shelf.glb",
                "texture_count": 6,
                "image_count": 5,
                "max_texture_width": 4096,
                "max_texture_height": 4096,
                "material_count": 3,
                "face_count": 12000,
                "vertex_count": 7000,
                "extent_x": 1.0,
                "extent_y": 1.0,
                "extent_z": 1.0,
            },
        ],
    )
    write_csv_gz(
        metadata_dir / "images.csv.gz",
        ["image_id", "path", "width", "height"],
        [
            {"image_id": "img_box", "path": "box.jpg", "width": 4096, "height": 4096},
            {
                "image_id": "img_shelf",
                "path": "shelf.jpg",
                "width": 4096,
                "height": 4096,
            },
        ],
    )
    write_jsonl_gz(
        metadata_dir / "listings_0.json.gz",
        [
            {
                "item_id": "item_box",
                "3dmodel_id": "m_box",
                "main_image_id": "img_box",
                "item_name": {
                    "fr_FR": "etagere simple",
                    "en_US": "Coffee packaging box with printed label",
                },
                "product_type": [
                    {"language_tag": "fr_FR", "value": "Meuble"},
                    {"language_tag": "en_GB", "value": "Package Box"},
                ],
                "item_keywords": ["label", "logo", "carton", "coffee"],
                "node": {"category": {"en_US": "Grocery packaging"}},
                "brand": "Demo Brand",
                "material": "cardboard",
            },
            {
                "item_id": "item_shelf",
                "3dmodel_id": "m_shelf",
                "main_image_id": "img_shelf",
                "item_name": "Plain solid wall shelf",
                "product_type": "Furniture",
                "item_keywords": ["shelf", "furniture", "plain", "solid"],
                "node": ["Home", "Furniture"],
                "brand": "Furniture Brand",
                "material": "wood",
            },
        ],
    )
    return metadata_dir


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_fake_candidates(root: Path, *extra_args: str) -> list[dict[str, str]]:
    metadata_dir = make_fake_metadata(root)
    out_csv = root / "candidates.csv"
    args = ["--metadata-dir", str(metadata_dir), "--out-csv", str(out_csv), *extra_args]
    assert build_main(args) == 0
    return read_rows(out_csv)


def test_build_candidate_index_joins_metadata_and_scores_semantics() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rows = build_fake_candidates(root)

        assert len(rows) == 2
        assert set(CANDIDATE_COLUMNS).issubset(rows[0].keys())

        by_source = {row["source_id"]: row for row in rows}
        box = by_source["m_box"]
        shelf = by_source["m_shelf"]
        assert box["main_image_id"] == "img_box"
        assert box["image_path"] == "box.jpg"
        assert box["thumbnail_url"].endswith("/images/small/box.jpg")
        assert box["original_image_url"].endswith("/images/original/box.jpg")
        assert box["asset_s3_uri"].endswith("/3dmodels/original/m_box.glb")
        assert box["name"] == "Coffee packaging box with printed label"
        assert "etagere" not in box["name"]
        assert float(box["semantic_score"]) > float(shelf["semantic_score"])
        assert float(box["final_candidate_score"]) > float(shelf["final_candidate_score"])


def test_require_positive_keyword_filters_plain_furniture() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = build_fake_candidates(Path(tmpdir), "--require-positive-keyword")
        assert [row["source_id"] for row in rows] == ["m_box"]


def test_thumbnail_downloader_uses_csv_thumbnail_url(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        candidates_csv = root / "candidates.csv"
        thumb_dir = root / "thumbs"
        url_seen: list[str] = []
        fieldnames = ["candidate_id", "image_path", "thumbnail_url"]
        with candidates_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "candidate_id": "c1",
                    "image_path": "this/path/would/be/broken.jpg",
                    "thumbnail_url": "https://example.invalid/images/small/good.jpg",
                }
            )

        def fake_download(url: str, out_path: Path, skip_existing: bool, timeout: int) -> str:
            url_seen.append(url)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake image")
            return "downloaded"

        monkeypatch.setattr(thumbnail_downloader, "download_thumbnail", fake_download)
        assert (
            thumbnail_downloader.main(
                [
                    "--candidates-csv",
                    str(candidates_csv),
                    "--thumb-dir",
                    str(thumb_dir),
                    "--top-k",
                    "1",
                ]
            )
            == 0
        )
        assert url_seen == ["https://example.invalid/images/small/good.jpg"]
        assert (thumb_dir / "c1.jpg").read_bytes() == b"fake image"


def test_mark_phase2a_candidates_appends_and_updates_selected_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata_dir = make_fake_metadata(root)
        candidates_csv = root / "candidates.csv"
        selected_csv = root / "selected.csv"
        assert build_main(["--metadata-dir", str(metadata_dir), "--out-csv", str(candidates_csv)]) == 0

        rows = read_rows(candidates_csv)
        candidate_id = next(row["candidate_id"] for row in rows if row["source_id"] == "m_box")

        assert (
            mark_main(
                [
                    "--candidates-csv",
                    str(candidates_csv),
                    "--selected-csv",
                    str(selected_csv),
                    "--candidate-id",
                    candidate_id,
                    "--status",
                    "selected",
                    "--reason",
                    "strong packaging label",
                ]
            )
            == 0
        )
        assert (
            mark_main(
                [
                    "--candidates-csv",
                    str(candidates_csv),
                    "--selected-csv",
                    str(selected_csv),
                    "--candidate-id",
                    candidate_id,
                    "--status",
                    "needs_review",
                    "--reason",
                    "check texture files first",
                    "--notes",
                    "updated after second pass",
                ]
            )
            == 0
        )

        selected_rows = read_rows(selected_csv)
        assert len(selected_rows) == 1
        assert selected_rows[0]["candidate_id"] == candidate_id
        assert selected_rows[0]["selection_status"] == "needs_review"
        assert selected_rows[0]["selection_reason"] == "check texture files first"
        assert selected_rows[0]["notes"] == "updated after second pass"
