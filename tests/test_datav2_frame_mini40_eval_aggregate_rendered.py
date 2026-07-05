import json
from pathlib import Path

from scripts import aggregate_datav2_frame_mini40_eval as aggregate


def view_row(view_id: str, base_mae: float, fine_mae: float, base_ssim: float, fine_ssim: float) -> dict:
    return {
        "view_id": view_id,
        "base_vs_fine": {"mae": abs(base_mae - fine_mae), "rmse": abs(base_mae - fine_mae), "ssim_like": 0.9},
        "base_vs_reference": {"mae": base_mae, "rmse": base_mae, "ssim_like": base_ssim, "histogram_l1": base_mae / 100.0, "edge_difference": base_mae / 10.0},
        "fine_vs_reference": {"mae": fine_mae, "rmse": fine_mae, "ssim_like": fine_ssim, "histogram_l1": fine_mae / 100.0, "edge_difference": fine_mae / 10.0},
    }


def write_metrics(path: Path, eval_split: str, selected_input_view: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "eval_split": eval_split,
        "selected_input_view": selected_input_view,
        "primary_front_views": ["004", "005"],
        "views": [
            view_row("005", 20.0, 10.0, 0.40, 0.60),
            view_row("004", 30.0, 25.0, 0.30, 0.35),
            view_row("000", 15.0, 18.0, 0.70, 0.60),
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_setup(tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"
    eval_cases = {
        "cases": [
            {"item_id": "VAL1", "eval_split": "val", "source_split": "val", "selected_input_view": "005", "primary_eval_views": ["004", "005"]},
            {"item_id": "TEST1", "eval_split": "test", "source_split": "test", "selected_input_view": "005", "primary_eval_views": ["004", "005"]},
            {"item_id": "TRAIN1", "eval_split": "train_sanity", "source_split": "train", "selected_input_view": "004", "primary_eval_views": ["004", "005"]},
        ]
    }
    output_root.mkdir(parents=True)
    (output_root / "eval_cases.json").write_text(json.dumps(eval_cases, indent=2) + "\n", encoding="utf-8")
    write_metrics(output_root / "render_eval" / "metrics" / "val" / "VAL1_rendered_view_metrics.json", "val", "005")
    write_metrics(output_root / "render_eval" / "metrics" / "test" / "TEST1_rendered_view_metrics.json", "test", "005")
    write_metrics(output_root / "render_eval" / "metrics" / "train_sanity" / "TRAIN1_rendered_view_metrics.json", "train_sanity", "004")
    config = {"experiment_name": "test_eval", "output_root": str(output_root), "primary_front_views": ["004", "005"]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_aggregate_rendered_metrics_reports_split_and_view_groups(tmp_path):
    config_path = make_setup(tmp_path)

    assert aggregate.main(["--config", str(config_path)]) == 0
    report = json.loads((tmp_path / "outputs" / "summary" / "mini40_eval_summary.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["by_eval_set"]["val_test"]["view_count"] == 6
    assert report["by_eval_set"]["train_sanity"]["view_count"] == 3
    assert report["by_view_group"]["input_view_005"]["view_count"] == 3
    assert report["by_view_group"]["front_views_004_005"]["view_count"] == 6
    assert report["by_view_group"]["all_views"]["views_where_fine_improves_mae"] == 6
