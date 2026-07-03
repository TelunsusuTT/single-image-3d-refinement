from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_phase2k2_multicase_metrics import main as aggregate_main  # noqa: E402


ASSET_IDS = ["B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def write_metrics(path: Path, albedo: float, metallic: float, roughness: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pair_metrics": {
            "albedo": {"mean_abs_diff": albedo},
            "metallic": {"mean_abs_diff": metallic},
            "roughness": {"mean_abs_diff": roughness},
        },
        "map_stats": {
            "base_metallic": {"grayscale_mean": 10.0},
            "finetuned_metallic": {"grayscale_mean": 12.0},
            "base_roughness": {"grayscale_mean": 80.0},
            "finetuned_roughness": {"grayscale_mean": 70.0},
        },
        "interpretation": {
            "albedo_differs_strongly": False,
            "metallic_differs_strongly": False,
            "roughness_differs_strongly": False,
        },
        "mesh_inventory": {
            "base_glb": {"size_bytes": 100},
            "finetuned_glb": {"size_bytes": 110},
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def write_cases_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "selected_asset_ids": ASSET_IDS,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_writes_json_md_and_means() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cases_config = root / "cases.json"
        output_root = root / "outputs"
        write_cases_config(cases_config)
        write_metrics(output_root / "compare" / ASSET_IDS[0] / "base_vs_finetuned_metrics.json", 1.0, 2.0, 3.0)
        write_metrics(output_root / "compare" / ASSET_IDS[1] / "base_vs_finetuned_metrics.json", 2.0, 4.0, 6.0)
        write_metrics(output_root / "compare" / ASSET_IDS[2] / "base_vs_finetuned_metrics.json", 3.0, 6.0, 9.0)

        assert aggregate_main(
            [
                "--cases-config",
                str(cases_config),
                "--output-root",
                str(output_root),
            ]
        ) == 0

        summary_json = output_root / "summary" / "multicase_metrics_summary.json"
        summary_md = output_root / "summary" / "multicase_metrics_summary.md"
        assert summary_json.is_file()
        assert summary_md.is_file()
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert summary["asset_ids"] == ASSET_IDS
        assert [row["asset_id"] for row in summary["cases"]] == ASSET_IDS
        assert summary["aggregate_means"]["albedo_mean_abs_diff"] == 2.0
        assert summary["aggregate_means"]["metallic_mean_abs_diff"] == 4.0
        assert summary["aggregate_means"]["roughness_mean_abs_diff"] == 6.0
        assert "B07HSK7MXZ" in summary_md.read_text(encoding="utf-8")
