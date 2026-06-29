from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_local_glb_assets import main  # noqa: E402
from make_abo_download_manifest import OUTPUT_FIELDNAMES  # noqa: E402


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def base_row(local_path: str) -> dict[str, str]:
    return {
        "selected_id": "sel_001",
        "source": "ABO",
        "source_id": "abo_001",
        "abo_path": "asset/path/model.glb",
        "s3_uri": "",
        "local_path": "",
        "name": "fake asset",
        "category": "package_box",
        "license": "TODO",
        "selection_reason": "fake test row",
        "status": "selected",
        "notes": "",
        "resolved_s3_uri": "s3://example/not-used.glb",
        "resolved_local_path": local_path,
    }


def test_check_local_glb_assets_passes_for_existing_tiny_glb() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = Path(tmpdir) / "asset.glb"
        glb_path.write_bytes(b"glTF fake test bytes")
        manifest_csv = Path(tmpdir) / "manifest.csv"
        write_manifest(manifest_csv, [base_row(str(glb_path))])

        exit_code = main(
            [
                "--manifest-csv",
                str(manifest_csv),
                "--min-size-mb",
                "0.000001",
                "--max-size-mb",
                "1",
            ]
        )

        assert exit_code == 0


def test_check_local_glb_assets_fails_for_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_csv = Path(tmpdir) / "manifest.csv"
        write_manifest(manifest_csv, [base_row(str(Path(tmpdir) / "missing.glb"))])

        exit_code = main(
            [
                "--manifest-csv",
                str(manifest_csv),
                "--min-size-mb",
                "0.000001",
                "--max-size-mb",
                "1",
            ]
        )

        assert exit_code == 1


def test_check_local_glb_assets_fails_for_wrong_suffix() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = Path(tmpdir) / "asset.txt"
        txt_path.write_bytes(b"fake test bytes")
        manifest_csv = Path(tmpdir) / "manifest.csv"
        write_manifest(manifest_csv, [base_row(str(txt_path))])

        exit_code = main(
            [
                "--manifest-csv",
                str(manifest_csv),
                "--min-size-mb",
                "0.000001",
                "--max-size-mb",
                "1",
            ]
        )

        assert exit_code == 1
