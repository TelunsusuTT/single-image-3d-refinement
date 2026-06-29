from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_abo_download_manifest import FIELDNAMES, OUTPUT_FIELDNAMES, main  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def base_row(**overrides: str) -> dict[str, str]:
    row = {
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
    }
    row.update(overrides)
    return row


def test_make_manifest_resolves_abo_s3_and_default_local_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "selected.csv"
        output_csv = Path(tmpdir) / "manifest.csv"
        asset_root = Path(tmpdir) / "assets"
        write_rows(input_csv, [base_row()])

        exit_code = main(
            [
                "--input-csv",
                str(input_csv),
                "--output-csv",
                str(output_csv),
                "--asset-root",
                str(asset_root),
            ]
        )

        assert exit_code == 0
        output_rows = read_rows(output_csv)
        assert output_rows[0].keys() == set(OUTPUT_FIELDNAMES)
        assert (
            output_rows[0]["resolved_s3_uri"]
            == "s3://amazon-berkeley-objects/3dmodels/original/asset/path/model.glb"
        )
        assert output_rows[0]["resolved_local_path"] == str(asset_root / "abo_001.glb")


def test_make_manifest_keeps_custom_local_path_and_skips_todo_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = str(Path(tmpdir) / "custom.glb")
        input_csv = Path(tmpdir) / "selected.csv"
        output_csv = Path(tmpdir) / "manifest.csv"
        rows = [
            base_row(selected_id="custom", source_id="abo_custom", local_path=custom_path),
            base_row(selected_id="todo", source_id="TODO_ABO_ID", abo_path="TODO_PATH.glb"),
        ]
        write_rows(input_csv, rows)

        exit_code = main(
            [
                "--input-csv",
                str(input_csv),
                "--output-csv",
                str(output_csv),
                "--asset-root",
                str(Path(tmpdir) / "assets"),
            ]
        )

        assert exit_code == 0
        output_rows = read_rows(output_csv)
        assert output_rows[0]["resolved_local_path"] == custom_path
        assert output_rows[1]["resolved_s3_uri"] == ""
        assert output_rows[1]["resolved_local_path"] == ""
