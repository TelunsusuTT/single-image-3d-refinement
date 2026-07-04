from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_phase2k3_rendered_metrics import main as aggregate_main  # noqa: E402


ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def write_cases_config(path: Path) -> Path:
    path.write_text(
        json.dumps({"cases": {asset_id: {} for asset_id in ASSET_IDS}}),
        encoding="utf-8",
    )
    return path


def view_row(view_id: str, fine_improves: bool) -> dict[str, object]:
    base_mae = 10.0
    fine_mae = 8.0 if fine_improves else 12.0
    base_ssim = 0.50
    fine_ssim = 0.60 if fine_improves else 0.40
    return {
        "view_id": view_id,
        "base_vs_fine": {"mae": 3.0, "rmse": 4.0, "ssim_like": 0.9},
        "base_vs_reference": {"mae": base_mae, "rmse": 12.0, "ssim_like": base_ssim},
        "fine_vs_reference": {"mae": fine_mae, "rmse": 10.0, "ssim_like": fine_ssim},
        "improvement": {
            "fine_minus_base_mae": fine_mae - base_mae,
            "fine_minus_base_ssim_like": fine_ssim - base_ssim,
        },
    }


def write_metrics(output_root: Path, asset_id: str, improved_views: int) -> None:
    rows = [view_row(f"{index:03d}", index < improved_views) for index in range(6)]
    path = output_root / "metrics" / asset_id / "rendered_view_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"asset_id": asset_id, "views": rows}),
        encoding="utf-8",
    )


def test_aggregate_writes_outputs_and_counts_improvements() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cases_config = write_cases_config(root / "cases.json")
        output_root = root / "rendered"
        for index, asset_id in enumerate(ASSET_IDS):
            write_metrics(output_root, asset_id, improved_views=index + 1)

        assert aggregate_main(
            [
                "--cases-config",
                str(cases_config),
                "--output-root",
                str(output_root),
            ]
        ) == 0

        summary_json = output_root / "summary" / "rendered_view_metrics_summary.json"
        summary_md = output_root / "summary" / "rendered_view_metrics_summary.md"
        assert summary_json.is_file()
        assert summary_md.is_file()
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert summary["asset_ids"] == ASSET_IDS
        assert [row["asset_id"] for row in summary["cases"]] == ASSET_IDS
        assert summary["total_views"] == 24
        assert summary["views_fine_improves_mae"] == 10
        assert summary["views_fine_improves_ssim"] == 10
        assert "B075YLTF7Q" in summary_md.read_text(encoding="utf-8")
