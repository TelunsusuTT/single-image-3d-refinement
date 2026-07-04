from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_make_abo_visual_human_review import REVIEW_COLUMNS, main as review_main  # noqa: E402


MANIFEST_COLUMNS = [
    "dedup_rank",
    "candidate_rank",
    "asset_id",
    "abo_path",
    "download_url",
    "local_glb_path",
    "rank_score",
    "flatness_ratio",
    "panel_aspect_ratio",
    "faces",
    "textures",
    "materials",
    "images",
    "dedupe_key",
]


def write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "dedup_rank": "1",
                "candidate_rank": "7",
                "asset_id": "asset_a",
                "abo_path": "A/asset_a.glb",
                "download_url": "https://example.invalid/A/asset_a.glb",
                "local_glb_path": "/tmp/asset_a.glb",
                "rank_score": "13.0",
                "flatness_ratio": "0.04",
                "panel_aspect_ratio": "1.5",
                "faces": "12000",
                "textures": "3",
                "materials": "1",
                "images": "3",
                "dedupe_key": "demo",
            }
        )


def write_config(root: Path, manifest: Path, inspection: Path, review: Path) -> Path:
    config = {
        "input_manifest": str(manifest),
        "output_root": str(root / "outputs"),
        "render_root": str(root / "outputs" / "renders"),
        "contact_sheet_root": str(root / "outputs" / "contact_sheets"),
        "inspection_csv": str(inspection),
        "inspection_summary_md": str(root / "outputs" / "inspection_summary.md"),
        "inspection_summary_json": str(root / "outputs" / "inspection_summary.json"),
        "human_review_csv": str(review),
        "render_resolution": 512,
        "view_ids": ["000", "001", "002", "003", "004", "005"],
        "preferred_input_views": ["004", "005"],
        "blender_bin": "/vol/bitbucket/ct1022/tools/bin/blender",
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_visual_human_review_without_inspection_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = root / "manifest.csv"
        inspection = root / "missing_inspection.csv"
        review = root / "review.csv"
        write_manifest(manifest)
        config = write_config(root, manifest, inspection, review)

        assert review_main(["--config", str(config)]) == 0
        rows = read_rows(review)

        assert rows
        assert list(rows[0].keys()) == REVIEW_COLUMNS
        assert rows[0]["dedup_rank"] == "1"
        assert rows[0]["asset_id"] == "asset_a"
        assert rows[0]["contact_sheet_page"] == "contact_sheet_001.jpg"
        assert rows[0]["import_ok"] == ""
        assert rows[0]["render_ok"] == ""
        assert rows[0]["human_decision"] == ""
        assert rows[0]["selected_input_view"] == ""
