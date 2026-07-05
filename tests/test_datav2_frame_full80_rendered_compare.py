import json
from pathlib import Path

import pytest

from scripts import compare_datav2_frame_full80_rendered_views as compare


VIEW_IDS = ["000", "001", "002", "003", "004", "005"]


def write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def make_setup(tmp_path: Path) -> Path:
    output_root = tmp_path / "outputs"
    render_root = output_root / "render_eval"
    refs = {}
    for view_id in VIEW_IDS:
        ref = tmp_path / "refs" / "VAL1" / f"{view_id}.png"
        write_png(ref, (100, 100, 100))
        refs[view_id] = str(ref)
        write_png(render_root / "renders" / "val" / "VAL1" / "base" / f"{view_id}.png", (120, 120, 120))
        write_png(render_root / "renders" / "val" / "VAL1" / "finetuned" / f"{view_id}.png", (105, 105, 105))
    render_cases = {
        "view_ids": VIEW_IDS,
        "background_color": [0.28, 0.28, 0.28],
        "primary_front_views": ["004", "005"],
        "cases": {
            "VAL1": {
                "eval_split": "val",
                "source_split": "val",
                "selected_input_view": "005",
                "primary_front_views": ["004", "005"],
                "reference_images": refs,
            }
        },
    }
    render_root.mkdir(parents=True, exist_ok=True)
    (render_root / "render_eval_cases.json").write_text(json.dumps(render_cases, indent=2) + "\n", encoding="utf-8")
    config = {"output_root": str(output_root)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def test_rendered_compare_writes_metrics_reports_and_board(tmp_path):
    config_path = make_setup(tmp_path)

    assert compare.main(["--config", str(config_path)]) == 0

    metrics_path = tmp_path / "outputs" / "render_eval" / "metrics" / "val" / "VAL1_rendered_view_metrics.json"
    board_path = tmp_path / "outputs" / "render_eval" / "boards" / "val" / "VAL1_rendered_view_board.jpg"
    report_path = tmp_path / "outputs" / "render_eval" / "reports" / "val" / "VAL1_rendered_view_report.md"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert board_path.is_file()
    assert report_path.is_file()
    assert metrics["aggregate"]["views_fine_improves_mae"] == 6
    assert metrics["views"][0]["base_vs_reference"]["mae"] > metrics["views"][0]["fine_vs_reference"]["mae"]
