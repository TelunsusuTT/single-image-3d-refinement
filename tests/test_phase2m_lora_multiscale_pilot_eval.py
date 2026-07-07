from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import aggregate_phase2m_lora_multiscale_pilot_eval as aggregate  # noqa: E402
import check_phase2m_lora_multiscale_pilot_eval_ready as ready  # noqa: E402
import phase2m_render_lora_multiscale_pilot as render_pilot  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_fake_case(root: Path, case_id: str, split: str) -> dict:
    render_cond = root / "data" / "hy3dpaint_train_examples" / "datav2_frame_panels_full101" / case_id / "render_cond"
    render_cond.mkdir(parents=True, exist_ok=True)
    for view_id in render_pilot.DEFAULT_VIEW_IDS:
        (render_cond / f"{view_id}_light_AL.png").write_bytes(b"png")
    return {
        "case_id": case_id,
        "eval_split": split,
        "selected_input_view": "005",
        "input_image_path": str(render_cond / "005_light_AL.png"),
        "mesh_path": str(root / "data" / "raw_assets" / f"{case_id}.glb"),
    }


def make_fake_m3b_outputs(root: Path) -> tuple[Path, Path]:
    cases = [
        make_fake_case(root, "VAL1", "val"),
        make_fake_case(root, "TEST1", "test"),
        make_fake_case(root, "TRAIN1", "train_sanity"),
    ]
    summary = {
        "status": "OK",
        "success": True,
        "case_count": 3,
        "cases": cases,
    }
    variants = []
    for case in cases:
        for variant in render_pilot.DEFAULT_VARIANTS:
            glb = root / "outputs" / "phase2m" / "lora_multiscale_pilot" / variant / case["eval_split"] / case["case_id"] / f"{variant}.glb"
            glb.parent.mkdir(parents=True, exist_ok=True)
            glb.write_bytes(b"glb")
            variants.append({"case_id": case["case_id"], "eval_split": case["eval_split"], "variant": variant, "output_glb_path": str(glb)})
    pilot_summary = root / "outputs" / "phase2m" / "lora_multiscale_pilot" / "pilot_summary.json"
    per_variant = pilot_summary.parent / "per_variant_outputs.json"
    write_json(pilot_summary, summary)
    write_json(per_variant, {"variants": variants})
    return pilot_summary, per_variant


def test_build_render_config_maps_cases_variants_and_references(tmp_path: Path) -> None:
    pilot_summary_path, per_variant_path = make_fake_m3b_outputs(tmp_path)

    config = render_pilot.build_render_config(
        json.loads(pilot_summary_path.read_text(encoding="utf-8")),
        json.loads(per_variant_path.read_text(encoding="utf-8")),
        tmp_path / "outputs" / "phase2m" / "lora_multiscale_pilot_rendered",
    )

    assert config["case_count"] == 3
    assert config["variants"] == ["base", "lora_scale050", "lora_scale075", "lora_scale100"]
    assert config["cases"]["VAL1"]["variant_glbs"]["lora_scale075"].endswith("lora_scale075.glb")
    assert config["cases"]["VAL1"]["reference_images"]["004"].endswith("004_light_AL.png")


def make_ready_args(root: Path, missing_glb: bool = False, bad_sbatch: bool = False) -> argparse.Namespace:
    pilot_summary, per_variant = make_fake_m3b_outputs(root)
    if missing_glb:
        data = json.loads(per_variant.read_text(encoding="utf-8"))
        Path(data["variants"][0]["output_glb_path"]).unlink()
        per_variant.write_text(json.dumps(data), encoding="utf-8")
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in [
        "phase2m_render_lora_multiscale_pilot.py",
        "aggregate_phase2m_lora_multiscale_pilot_eval.py",
        "make_phase2m_lora_multiscale_pilot_boards.py",
        "render_phase2k3_glb_views_blender.py",
    ]:
        (scripts / name).write_text("script\n", encoding="utf-8")
    sbatch = root / "env" / "run_phase2m_lora_multiscale_pilot_eval_a100.sbatch"
    sbatch.parent.mkdir(parents=True, exist_ok=True)
    sbatch.write_text(
        "#SBATCH -p a100\n#SBATCH --gres=gpu:1\npython scripts/check_phase2m_lora_multiscale_pilot_eval_ready.py\npython scripts/phase2m_render_lora_multiscale_pilot.py\npython scripts/aggregate_phase2m_lora_multiscale_pilot_eval.py\npython scripts/make_phase2m_lora_multiscale_pilot_boards.py\n"
        if not bad_sbatch
        else "#SBATCH -p gpgpuC\n#SBATCH --constraint=a100\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        pilot_summary=pilot_summary,
        per_variant_outputs=per_variant,
        render_root=root / "outputs" / "phase2m" / "lora_multiscale_pilot_rendered",
        eval_root=root / "outputs" / "phase2m" / "lora_multiscale_pilot_eval",
        render_script=scripts / "phase2m_render_lora_multiscale_pilot.py",
        aggregate_script=scripts / "aggregate_phase2m_lora_multiscale_pilot_eval.py",
        board_script=scripts / "make_phase2m_lora_multiscale_pilot_boards.py",
        blender_wrapper=scripts / "render_phase2k3_glb_views_blender.py",
        sbatch=sbatch,
        project_root=root,
        dry_run=False,
    )


def test_m3c_readiness_passes_with_fake_outputs(tmp_path: Path) -> None:
    report = ready.check_readiness(make_ready_args(tmp_path))

    assert report.errors == []


def test_m3c_readiness_fails_when_glb_missing(tmp_path: Path) -> None:
    report = ready.check_readiness(make_ready_args(tmp_path, missing_glb=True))

    assert any("GLB missing" in error for error in report.errors)


def test_m3c_readiness_rejects_old_sbatch_tokens(tmp_path: Path) -> None:
    report = ready.check_readiness(make_ready_args(tmp_path, bad_sbatch=True))

    assert any("banned token" in error for error in report.errors)


def test_aggregate_summary_selects_best_scale_by_mae_delta() -> None:
    rows = [
        {"variant": "base", "variant_mae": 10.0, "base_mae": 10.0, "variant_rmse": 2.0, "base_rmse": 2.0, "variant_ssim_like": 0.5, "base_ssim_like": 0.5, "base_vs_variant_mae": 0.0, "variant_minus_base_mae": 0.0, "variant_minus_base_rmse": 0.0, "variant_minus_base_ssim_like": 0.0, "case_id": "A", "eval_split": "val", "view_id": "004", "view_groups": ["all_views", "front_views_004_005"]},
        {"variant": "lora_scale050", "variant_mae": 9.0, "base_mae": 10.0, "variant_rmse": 1.8, "base_rmse": 2.0, "variant_ssim_like": 0.6, "base_ssim_like": 0.5, "base_vs_variant_mae": 1.0, "variant_minus_base_mae": -1.0, "variant_minus_base_rmse": -0.2, "variant_minus_base_ssim_like": 0.1, "case_id": "A", "eval_split": "val", "view_id": "004", "view_groups": ["all_views", "front_views_004_005"]},
        {"variant": "lora_scale075", "variant_mae": 8.0, "base_mae": 10.0, "variant_rmse": 1.5, "base_rmse": 2.0, "variant_ssim_like": 0.7, "base_ssim_like": 0.5, "base_vs_variant_mae": 1.2, "variant_minus_base_mae": -2.0, "variant_minus_base_rmse": -0.5, "variant_minus_base_ssim_like": 0.2, "case_id": "A", "eval_split": "val", "view_id": "004", "view_groups": ["all_views", "front_views_004_005"]},
    ]
    config = {"variants": ["base", "lora_scale050", "lora_scale075"], "view_ids": ["004"]}

    summary = aggregate.build_aggregate_summary(config, rows, [], Path("render"), Path("eval"))
    comparison = aggregate.build_scale_comparison(summary)

    assert comparison["best_variant_by_all_views_mae_delta"] == "lora_scale075"
    assert summary["by_view_group"]["front_views_004_005"]["lora_scale075"]["views_where_variant_improves_mae"] == 1
