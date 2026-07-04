from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_phase2k4_reference_view_ablation import main as aggregate_main  # noqa: E402


ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def write_summary(root: Path, fine_mae: float) -> None:
    path = root / "summary" / "rendered_view_metrics_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "aggregate_means": {
            "base_vs_reference_mean_mae": 20.0,
            "fine_vs_reference_mean_mae": fine_mae,
            "base_vs_reference_mean_ssim_like": 0.40,
            "fine_vs_reference_mean_ssim_like": 0.45,
        },
        "views_fine_improves_mae": 10,
        "views_fine_improves_ssim": 12,
        "cases": [{"asset_id": asset_id} for asset_id in ASSET_IDS],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def write_per_view_metrics(root: Path, asset_id: str, base_mae: float, fine_mae: float) -> None:
    path = root / "metrics" / asset_id / "rendered_view_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    views = []
    for view_id in ["000", "004", "005"]:
        views.append(
            {
                "view_id": view_id,
                "base_vs_reference": {"mae": base_mae, "ssim_like": 0.40},
                "fine_vs_reference": {"mae": fine_mae, "ssim_like": 0.50},
            }
        )
    path.write_text(json.dumps({"views": views}), encoding="utf-8")


def write_cases_config(path: Path, root: Path, baseline: Path) -> Path:
    data = {
        "output_root": str(root / "outputs" / "phase2k4"),
        "asset_ids": ASSET_IDS,
        "input_views": ["004", "005"],
        "previous_baseline_render_eval_root": str(baseline),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_reference_view_aggregate_writes_outputs_and_front_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        baseline = root / "baseline"
        output_root = root / "outputs" / "phase2k4"
        write_summary(baseline, fine_mae=19.0)
        write_summary(output_root / "rendered_eval" / "input_004", fine_mae=15.0)
        write_summary(output_root / "rendered_eval" / "input_005", fine_mae=16.0)
        for asset_id in ASSET_IDS:
            write_per_view_metrics(baseline, asset_id, base_mae=30.0, fine_mae=29.0)
            write_per_view_metrics(output_root / "rendered_eval" / "input_004", asset_id, base_mae=20.0, fine_mae=18.0)
            write_per_view_metrics(output_root / "rendered_eval" / "input_005", asset_id, base_mae=22.0, fine_mae=19.0)
        cases_config = write_cases_config(root / "cases.json", root, baseline)

        assert aggregate_main(["--cases-config", str(cases_config)]) == 0

        summary_json = output_root / "summary" / "reference_view_ablation_summary.json"
        summary_md = output_root / "summary" / "reference_view_ablation_summary.md"
        assert summary_json.is_file()
        assert summary_md.is_file()
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert set(summary["conditions"].keys()) == {"previous_baseline", "input_004", "input_005"}
        assert summary["conditions"]["input_004"]["exists"] is True
        assert summary["conditions"]["input_005"]["exists"] is True
        front = summary["conditions"]["input_004"]["front_metrics"]["aggregate"]
        assert front["base_front_mae_mean"] == 20.0
        assert front["fine_front_mae_mean"] == 18.0
        assert "input_004" in summary_md.read_text(encoding="utf-8")
