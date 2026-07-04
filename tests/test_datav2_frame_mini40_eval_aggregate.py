import json
from pathlib import Path

from scripts import aggregate_datav2_frame_mini40_eval as aggregate


def view_row(view_id: str, base_mae: float, fine_mae: float, base_ssim: float, fine_ssim: float) -> dict:
    return {
        "view_id": view_id,
        "base_vs_reference": {"mae": base_mae, "ssim_like": base_ssim},
        "fine_vs_reference": {"mae": fine_mae, "ssim_like": fine_ssim},
    }


def make_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "views": [
            view_row("005", 20.0, 10.0, 0.40, 0.60),
            view_row("004", 30.0, 25.0, 0.30, 0.35),
            view_row("000", 15.0, 18.0, 0.70, 0.60),
        ]
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_aggregate_setup(tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"
    eval_cases = {
        "cases": [
            {
                "item_id": "VAL1",
                "eval_split": "val",
                "source_split": "val",
                "selected_input_view": "005",
                "primary_eval_views": ["004", "005"],
            },
            {
                "item_id": "TRAIN1",
                "eval_split": "train_sanity",
                "source_split": "train",
                "selected_input_view": "004",
                "primary_eval_views": ["004", "005"],
            },
        ]
    }
    output_root.mkdir(parents=True)
    (output_root / "eval_cases.json").write_text(json.dumps(eval_cases, indent=2) + "\n", encoding="utf-8")
    make_metrics(output_root / "render_eval" / "metrics" / "VAL1" / "rendered_view_metrics.json")
    make_metrics(output_root / "render_eval" / "metrics" / "TRAIN1" / "rendered_view_metrics.json")
    config = {
        "experiment_name": "test_eval",
        "output_root": str(output_root),
        "primary_front_views": ["004", "005"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_eval_aggregate_summarizes_view_groups_and_splits(tmp_path):
    config_path = make_aggregate_setup(tmp_path)

    assert aggregate.main(["--config", str(config_path)]) == 0
    report = json.loads((tmp_path / "outputs" / "summary" / "mini40_eval_summary.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["by_view_group"]["all_views"]["view_count"] == 6
    assert report["by_view_group"]["front_views"]["view_count"] == 4
    assert report["by_view_group"]["non_front_back_views"]["view_count"] == 2
    assert report["by_split"]["train_sanity"]["view_count"] == 3
    assert report["by_view_group"]["all_views"]["views_where_fine_improves_mae"] == 4
