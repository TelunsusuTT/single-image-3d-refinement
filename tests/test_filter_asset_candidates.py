from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_asset_candidates import FIELDNAMES, main  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def base_row(candidate_id: str, expected_texture_heavy: str, status: str) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "source": "ABO",
        "source_id": f"TODO_{candidate_id}",
        "name": f"candidate {candidate_id}",
        "category": "package_box",
        "asset_uri": "",
        "metadata_uri": "",
        "license": "TODO",
        "expected_texture_heavy": expected_texture_heavy,
        "notes": "fake test row",
        "status": status,
    }


def test_filter_keeps_texture_heavy_candidate_and_writes_expected_rows() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "input.csv"
        output_csv = Path(tmpdir) / "filtered.csv"
        rows = [
            base_row("keep_yes_candidate", "yes", "candidate"),
            base_row("drop_not_texture_heavy", "no", "candidate"),
            base_row("drop_rejected", "yes", "rejected"),
        ]
        write_rows(input_csv, rows)

        exit_code = main(["--input-csv", str(input_csv), "--output-csv", str(output_csv)])

        assert exit_code == 0
        output_rows = read_rows(output_csv)
        assert [row["candidate_id"] for row in output_rows] == ["keep_yes_candidate"]
        assert output_rows[0]["expected_texture_heavy"] == "yes"
        assert output_rows[0]["status"] == "candidate"
