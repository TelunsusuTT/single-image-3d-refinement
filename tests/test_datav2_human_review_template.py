from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_make_human_review_template import REVIEW_COLUMNS, main as review_main  # noqa: E402


def test_human_review_template_includes_required_curation_columns() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        candidates_csv = root / "candidates.csv"
        out_csv = root / "review.csv"

        with candidates_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "candidate_rank",
                    "source",
                    "asset_id",
                    "title",
                    "tags",
                    "category",
                    "file_format",
                    "rank_score",
                    "positive_keyword_hits",
                    "negative_keyword_hits",
                    "technical_flags",
                    "auto_reasons",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "candidate_rank": "1",
                    "source": "abo",
                    "asset_id": "flat_1",
                    "title": "Framed poster panel",
                    "tags": "poster;panel",
                    "category": "wall art",
                    "file_format": "glb",
                    "rank_score": "22.0",
                    "positive_keyword_hits": "poster;panel",
                    "negative_keyword_hits": "",
                    "technical_flags": "preferred_format:glb",
                    "auto_reasons": "",
                }
            )

        assert review_main(["--candidates-csv", str(candidates_csv), "--out-csv", str(out_csv), "--top-k", "1"]) == 0
        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert rows
        assert list(rows[0].keys()) == REVIEW_COLUMNS
        assert rows[0]["asset_id"] == "flat_1"
        assert rows[0]["human_decision"] == ""
        assert rows[0]["selected_input_view"] == ""
        assert "positive:poster;panel" in rows[0]["auto_reasons"]
