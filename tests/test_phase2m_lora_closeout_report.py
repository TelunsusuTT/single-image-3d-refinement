from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import make_phase2m_lora_closeout_report as report  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def metric(delta: float) -> dict:
    return {
        "view_count": 6,
        "base_mean_mae": 5.0,
        "variant_mean_mae": 5.0 + delta,
        "base_mean_rmse": 10.0,
        "variant_mean_rmse": 10.0 + delta,
        "base_mean_ssim_like": 0.9,
        "variant_mean_ssim_like": 0.9 - delta / 100.0,
        "base_vs_variant_mean_mae": abs(delta),
        "variant_minus_base_mean_mae": delta,
        "variant_minus_base_mean_rmse": delta,
        "variant_minus_base_mean_ssim_like": -delta / 100.0,
        "views_where_variant_improves_mae": 6 if delta < 0 else 0,
        "views_where_variant_improves_ssim_like": 6 if delta < 0 else 0,
    }


def fake_aggregate() -> dict:
    variants = ["base", "lora_scale050", "lora_scale075", "lora_scale100"]
    by_variant = {"base": metric(0.0), "lora_scale050": metric(0.1), "lora_scale075": metric(0.2), "lora_scale100": metric(0.3)}
    by_view_group = {
        "all_views": by_variant,
        "front_views_004_005": {"base": metric(0.0), "lora_scale050": metric(0.3), "lora_scale075": metric(0.4), "lora_scale100": metric(0.5)},
        "non_front_views_000_003": {"base": metric(0.0), "lora_scale050": metric(-0.1), "lora_scale075": metric(-0.1), "lora_scale100": metric(-0.1)},
    }
    by_split = {
        "val": {"base": metric(0.0), "lora_scale050": metric(0.1), "lora_scale075": metric(0.2), "lora_scale100": metric(0.3)},
        "test": {"base": metric(0.0), "lora_scale050": metric(0.1), "lora_scale075": metric(0.2), "lora_scale100": metric(0.3)},
        "train_sanity": {"base": metric(0.0), "lora_scale050": metric(-0.2), "lora_scale075": metric(-0.3), "lora_scale100": metric(-0.1)},
    }
    return {"status": "OK", "variants": variants, "by_variant": by_variant, "by_view_group": by_view_group, "by_split": by_split}


def test_decision_stops_when_heldout_and_front_degrade() -> None:
    decision = report.make_decision(fake_aggregate())

    assert decision["decision"] == "stop_full_lora_eval"
    assert decision["heldout_val_test_and_front_degrade"] is True
    assert decision["train_sanity_improves"] is True


def test_summary_tables_parse_aggregate() -> None:
    decision = report.make_decision(fake_aggregate())
    text = report.make_summary_tables_md(fake_aggregate(), decision)

    assert "Overall All Views" in text
    assert "Front Views 004/005" in text
    assert "Non-Front Views 000-003" in text
    assert "stop_full_lora_eval" in text


def make_fake_inputs(root: Path) -> argparse.Namespace:
    training = root / "outputs" / "phase2m" / "train" / "training_summary.json"
    adapter_config = training.parent / "adapter_config.json"
    pilot = root / "outputs" / "phase2m" / "pilot" / "pilot_summary.json"
    eval_root = root / "outputs" / "phase2m" / "eval"
    aggregate = eval_root / "aggregate_summary.json"
    scale = eval_root / "scale_comparison.json"
    metrics = eval_root / "metrics_rows.csv"
    boards = eval_root / "boards" / "val"
    write_json(training, {"trainable_parameter_count": 829952, "total_parameter_count": 3099557192, "final_adapter_bytes_estimate": 3319808, "max_train_steps": 300})
    write_json(adapter_config, {"target_names": ["unet.attn_refview.to_q", "unet.attn_dino.to_q"]})
    write_json(pilot, {"cases": [{"case_id": "VAL1", "eval_split": "val"}, {"case_id": "TEST1", "eval_split": "test"}, {"case_id": "TRAIN1", "eval_split": "train_sanity"}]})
    write_json(aggregate, fake_aggregate())
    write_json(scale, {"best_variant_by_all_views_mae_delta": "lora_scale050"})
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("case_id,variant\nVAL1,base\n", encoding="utf-8")
    boards.mkdir(parents=True, exist_ok=True)
    (boards / "board.jpg").write_bytes(b"jpg")
    return argparse.Namespace(
        training_summary=training,
        adapter_config=adapter_config,
        pilot_summary=pilot,
        aggregate_summary=aggregate,
        scale_comparison=scale,
        metrics_rows=metrics,
        boards_root=eval_root / "boards",
        report_packet=root / "outputs" / "phase2m" / "eval" / "report_packet",
        doc_report=root / "docs" / "phase2m_lora_negative_result_report.md",
        project_root=root,
    )


def test_generate_report_packet_stays_under_outputs_phase2m(tmp_path: Path) -> None:
    args = make_fake_inputs(tmp_path)

    manifest = report.generate_report(args)

    packet = Path(manifest["report_packet"])
    assert packet.is_dir()
    assert tmp_path / "outputs" / "phase2m" in packet.parents
    assert (packet / "summary_tables.md").is_file()
    assert (packet / "boards" / "val" / "board.jpg").is_file()
    assert manifest["decision"] == "stop_full_lora_eval"


def test_generate_report_rejects_packet_outside_outputs_phase2m(tmp_path: Path) -> None:
    args = make_fake_inputs(tmp_path)
    args.report_packet = tmp_path / "elsewhere" / "packet"

    try:
        report.generate_report(args)
    except ValueError as exc:
        assert "report packet" in str(exc)
    else:
        raise AssertionError("expected ValueError")
