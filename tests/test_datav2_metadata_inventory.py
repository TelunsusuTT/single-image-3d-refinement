from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from datav2_inventory_metadata_sources import main as inventory_main  # noqa: E402


def write_config(path: Path, metadata_paths: list[Path]) -> None:
    path.write_text(
        json.dumps(
            {
                "target_subclass": "flat_rectangular_graphic_panels",
                "metadata_paths": [str(item) for item in metadata_paths],
            }
        ),
        encoding="utf-8",
    )


def test_inventory_handles_csv_and_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_path = root / "metadata.csv"
        jsonl_path = root / "metadata.jsonl"
        config_path = root / "config.json"
        out_dir = root / "inventory"

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "name", "format"])
            writer.writeheader()
            writer.writerow({"id": "a1", "name": "Framed poster", "format": "glb"})
            writer.writerow({"id": "a2", "name": "Chair", "format": "glb"})

        jsonl_path.write_text(
            "\n".join(
                [
                    json.dumps({"uid": "j1", "title": "Wall art panel"}),
                    json.dumps({"uid": "j2", "title": "Bottle"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        write_config(config_path, [csv_path, jsonl_path])

        assert inventory_main(["--config", str(config_path), "--out-dir", str(out_dir)]) == 0
        report = json.loads((out_dir / "metadata_sources_report.json").read_text(encoding="utf-8"))

        assert report["metadata_file_count"] == 2
        assert report["parseable_file_count"] == 2
        formats = {source["format"] for source in report["sources"]}
        assert formats == {"csv", "jsonl"}
        assert (out_dir / "metadata_sources_report.md").is_file()
