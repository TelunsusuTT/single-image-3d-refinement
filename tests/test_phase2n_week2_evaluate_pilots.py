from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import phase2n_week2_evaluate_pilots as evaluate  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def make_png_bytes(size: int = 512) -> bytes:
    row = b"\x00" + (b"\x47\x47\x47" * size)
    raw = row * size
    return (
        evaluate.PNG_SIGNATURE
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path
    inference = root / "outputs" / "phase2n" / "week2_pilot_inference" / "slurm_264581"
    inference.mkdir(parents=True)
    (inference / "_SUCCESS").write_text("OK\n", encoding="utf-8")
    write_json(
        inference / "summary.json",
        {
            "status": "OK",
            "test_data_used": False,
            "case_count_per_variant": 8,
            "total_inference_outputs": 32,
            "variant_order": list(evaluate.EXPECTED_VARIANTS[2:]),
            "split_counts_per_variant": {
                "val": 6,
                "train_sanity": 2,
                "test": 0,
            },
        },
    )

    png = make_png_bytes()
    frozen_cases = []
    for index, asset_id in enumerate(evaluate.EXPECTED_CASE_IDS):
        split = "val" if index < 6 else "train_sanity"
        render_cond = (
            root
            / "data"
            / "hy3dpaint_train_examples"
            / "datav2_frame_panels_full101"
            / asset_id
            / "render_cond"
        )
        render_cond.mkdir(parents=True)
        for view_id in evaluate.EXPECTED_VIEW_IDS:
            (render_cond / f"{view_id}_light_AL.png").write_bytes(png)
        frozen_cases.append(
            {
                "asset_id": asset_id,
                "source_split": "val" if split == "val" else "train",
                "eval_split": split,
                "selection_stratum": (
                    "non_front_leakage_risk"
                    if index < 2
                    else "front_reference_complexity"
                    if index < 4
                    else "historical_median"
                    if index < 6
                    else "deterministic_train_sanity"
                ),
                "selection_rationale": "fixture",
                "mesh_path": f"data/raw_assets/{asset_id}.glb",
                "reference_image_path": str(
                    render_cond / "005_light_AL.png"
                ),
                "selected_input_view": "005",
                "reference_lighting": "AL",
            }
        )
    frozen_path = root / "configs" / "phase2n_week2_pilot_eval_cases.json"
    write_json(
        frozen_path,
        {
            "phase": "phase2n_week2_pilot_evaluation",
            "selection_frozen_before_training": True,
            "selected_input_view": "005",
            "reference_lighting": "AL",
            "case_count": 8,
            "split_counts": {"val": 6, "train_sanity": 2, "test": 0},
            "cases": frozen_cases,
        },
    )

    baseline_report = {
        "status": "OK",
        "complete_compatible_coverage": True,
    }
    for variant in evaluate.EXPECTED_VARIANTS[:2]:
        rows = []
        for asset_id in evaluate.EXPECTED_CASE_IDS:
            glb = (
                root
                / "outputs"
                / "historical"
                / variant
                / asset_id
                / "textured_mesh.glb"
            )
            glb.parent.mkdir(parents=True, exist_ok=True)
            glb.write_bytes(b"glTF-fixture")
            rows.append(
                {
                    "asset_id": asset_id,
                    "output_glb_path": str(glb),
                }
            )
        baseline_report[variant] = {
            "status": "OK",
            "all_cases_compatible": True,
            "cases": rows,
        }
    write_json(inference / "baseline_reuse_report.json", baseline_report)

    for variant in evaluate.EXPECTED_VARIANTS[2:]:
        variant_root = inference / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        (variant_root / "_SUCCESS").write_text("OK\n", encoding="utf-8")
        for asset_id in evaluate.EXPECTED_CASE_IDS:
            case_root = inference / variant / asset_id
            case_root.mkdir(parents=True)
            glb = case_root / "textured_mesh.glb"
            glb.write_bytes(b"glTF-fixture")
            write_json(
                case_root / "inference_manifest.json",
                {
                    "status": "OK",
                    "variant": variant,
                    "asset_id": asset_id,
                    "test_data_used": False,
                    "output_glb_path": str(glb),
                },
            )

    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    renderer = scripts / "render_phase2k3_glb_views_blender.py"
    renderer.write_text("def render_variant(*args, **kwargs):\n    return {}\n", encoding="utf-8")
    metrics = scripts / "compare_phase2k3_rendered_views.py"
    metrics.write_text(
        "def pair_metrics(*args, **kwargs):\n    return {}\n"
        "def rgb255(values):\n    return (0, 0, 0)\n"
        "def diff_image(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    blender = root / "tools" / "blender"
    blender.parent.mkdir(parents=True)
    blender.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    blender.chmod(0o755)

    config_path = root / "configs" / "phase2n_week2_pilot_rendered_eval.json"
    write_json(
        config_path,
        {
            "phase": evaluate.EXPECTED_PHASE,
            "inference_run_dir": str(inference),
            "frozen_cases_config": str(frozen_path),
            "output_root": "outputs/phase2n/week2_pilot_rendered_eval",
            "render_resolution": 512,
            "blender_bin": str(blender),
            "stable_renderer_script": str(renderer),
            "stable_metrics_script": str(metrics),
            "variant_order": list(evaluate.EXPECTED_VARIANTS),
            "variant_sources": {
                "corrected_input_base": {
                    "kind": "baseline_reuse_report",
                    "report_key": "corrected_input_base",
                },
                "historical_full80_500": {
                    "kind": "baseline_reuse_report",
                    "report_key": "historical_full80_500",
                },
                **{
                    variant: {
                        "kind": "stage4_inference",
                        "manifest_template": (
                            f"{variant}/{{asset_id}}/inference_manifest.json"
                        ),
                    }
                    for variant in evaluate.EXPECTED_VARIANTS[2:]
                },
            },
            "view_ids": list(evaluate.EXPECTED_VIEW_IDS),
            "view_groups": dict(evaluate.EXPECTED_VIEW_GROUPS),
            "selected_input_view": "005",
            "reference_lighting": "AL",
            "background_color": [0.28, 0.28, 0.28],
            "isolated_blender_process_per_glb": True,
            "expected_counts": {
                "cases": 8,
                "val": 6,
                "train_sanity": 2,
                "test": 0,
                "variants": 6,
                "source_glbs": 48,
                "references": 48,
                "planned_renders": 288,
            },
            "leakage_risk_assets": ["B073P1D981", "B075YLXSJC"],
            "high_front_complexity_assets": ["B075YLQTNP", "B075YM2VXJ"],
            "historical_median_assets": ["B073P52NDX", "B073P1S8VZ"],
            "train_sanity_assets": ["B073P16J7Y", "B073P1H786"],
        },
    )
    return root, config_path


def test_resolves_exact_variants_cases_glbs_and_render_plan(tmp_path: Path) -> None:
    root, config_path = make_fixture(tmp_path)

    protocol = evaluate.resolve_protocol(config_path, project_root=root)

    assert [row["variant"] for row in protocol["variants"]] == evaluate.EXPECTED_VARIANTS
    assert [row["asset_id"] for row in protocol["cases"]] == evaluate.EXPECTED_CASE_IDS
    assert protocol["counts"] == {
        "cases": 8,
        "val": 6,
        "train_sanity": 2,
        "test": 0,
        "variants": 6,
        "source_glbs": 48,
        "references": 48,
        "planned_renders": 288,
    }
    assert len(protocol["plan"]) == 288
    assert len(
        {
            (row["asset_id"], row["variant"], row["source_glb_path"])
            for row in protocol["plan"]
        }
    ) == 48


def test_rejects_any_test_case(tmp_path: Path) -> None:
    root, config_path = make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_path = Path(config["frozen_cases_config"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["cases"][0]["eval_split"] = "test"
    write_json(frozen_path, frozen)

    with pytest.raises(evaluate.EvaluationError, match="test case is forbidden"):
        evaluate.resolve_protocol(config_path, project_root=root)


def test_check_only_never_executes_blender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, config_path = make_fixture(tmp_path)

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("check-only reached subprocess.run")

    monkeypatch.setattr(evaluate.subprocess, "run", forbidden_subprocess)
    protocol = evaluate.run_check_only(config_path, project_root=root)
    output = capsys.readouterr().out

    assert protocol["counts"]["planned_renders"] == 288
    assert "PHASE2N_WEEK2_RENDER_INPUTS_OK" in output
    assert "PHASE2N_WEEK2_RENDER_PROTOCOL_OK" in output
    assert "PHASE2N_WEEK2_NO_TEST_OK" in output
    assert "PHASE2N_WEEK2_RENDERED_EVAL_READINESS_OK" in output


def synthetic_metric_rows(protocol: dict) -> list[dict]:
    rows = []
    leakage = set(protocol["config"]["leakage_risk_assets"])
    for case in protocol["cases"]:
        for view_id in evaluate.EXPECTED_VIEW_IDS:
            for variant in evaluate.EXPECTED_VARIANTS:
                delta = 0.0
                if variant == "pc_s1_step160":
                    if case["eval_split"] == "train_sanity":
                        delta = -3.0
                    elif view_id in {"004", "005"}:
                        delta = -1.0
                    elif case["asset_id"] in leakage:
                        delta = 2.0
                    else:
                        delta = -0.5
                elif variant != evaluate.BASE_VARIANT:
                    delta = 0.25
                rows.append(
                    {
                        "asset_id": case["asset_id"],
                        "eval_split": case["eval_split"],
                        "selection_stratum": case["selection_stratum"],
                        "variant": variant,
                        "view_id": view_id,
                        "view_groups": evaluate.row_view_groups(
                            view_id, protocol["config"]
                        ),
                        "candidate_mae": 10.0 + delta,
                        "candidate_rmse": 12.0 + delta,
                        "candidate_psnr": 20.0 - delta,
                        "candidate_ssim_like": 0.8 - delta / 100.0,
                        "candidate_histogram_l1": 0.1 + delta / 100.0,
                        "candidate_edge_difference": 5.0 + delta,
                        "candidate_mean_abs_color_shift": 2.0 + delta,
                        "delta_mae": delta,
                        "delta_rmse": delta,
                        "delta_psnr": -delta,
                        "delta_ssim_like": -delta / 100.0,
                        "delta_histogram_l1": delta / 100.0,
                        "delta_edge_difference": delta,
                        "delta_mean_abs_color_shift": delta,
                        "direct_visual_change_mae": abs(delta),
                    }
                )
    return rows


def test_aggregation_preserves_groups_splits_wins_and_leakage(tmp_path: Path) -> None:
    root, config_path = make_fixture(tmp_path)
    protocol = evaluate.resolve_protocol(config_path, project_root=root)
    rows = synthetic_metric_rows(protocol)

    aggregate, per_asset = evaluate.build_aggregate_artifacts(rows, protocol)
    evidence = aggregate["candidate_evidence_relative_to_corrected_input_base"][
        "pc_s1_step160"
    ]

    assert aggregate["row_count"] == 288
    assert aggregate["by_split"]["val"]["pc_s1_step160"]["row_count"] == 36
    assert (
        aggregate["by_split"]["train_sanity"]["pc_s1_step160"]["row_count"]
        == 12
    )
    assert aggregate["by_view"]["005"]["pc_s1_step160"]["improved_view_count"] == 8
    assert (
        per_asset["assets"]["B073P1D981"]["front_004_005"]["pc_s1_step160"][
            "improved_view_count"
        ]
        == 2
    )
    assert (
        per_asset["assets"]["B073P1D981"]["nonfront_000_003"][
            "pc_s1_step160"
        ]["worsened_view_count"]
        == 4
    )
    assert evidence["val_front_mean_better_than_base"] is True
    assert evidence["val_front_view_win_count"] == 12
    assert evidence["val_input_mean_better_than_base"] is True
    assert evidence["val_nonfront_mean_better_than_base"] is False
    assert evidence["leakage_risk_regression_count"] == 2
    assert evidence["train_sanity_mean_better_than_base"] is True
    assert aggregate["automatic_winner"] is None


def populate_complete_run(
    protocol: dict,
    run_root: Path,
    png: bytes,
) -> None:
    for case in protocol["cases"]:
        for variant in evaluate.EXPECTED_VARIANTS:
            output_dir = evaluate.render_output_dir(run_root, case, variant)
            output_dir.mkdir(parents=True, exist_ok=True)
            for view_id in evaluate.EXPECTED_VIEW_IDS:
                (output_dir / f"{view_id}.png").write_bytes(png)
    required = evaluate.required_final_paths(run_root, protocol)
    for path in required:
        if path.exists():
            continue
        if path.name == "aggregate.json":
            write_json(
                path,
                {"status": "OK", "row_count": 288, "test_data_used": False},
            )
        elif path.suffix == ".json":
            write_json(path, {"status": "OK"})
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"artifact\n")
    (run_root / "_SUCCESS").write_text("OK\n", encoding="utf-8")


def test_complete_run_is_idempotent_and_missing_png_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path = make_fixture(tmp_path)
    protocol = evaluate.resolve_protocol(config_path, project_root=root)
    run_root = evaluate.prepare_run(protocol, "completed_fixture", create=True)
    populate_complete_run(protocol, run_root, make_png_bytes())

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("complete idempotent run attempted Blender")

    monkeypatch.setattr(evaluate.subprocess, "run", forbidden_subprocess)
    assert evaluate.execute_runtime(protocol, "completed_fixture", "run_all") == 0

    missing = (
        evaluate.render_output_dir(
            run_root, protocol["cases"][0], evaluate.BASE_VARIANT
        )
        / "000.png"
    )
    missing.unlink()
    with pytest.raises(evaluate.EvaluationError, match="incomplete"):
        evaluate.execute_runtime(protocol, "completed_fixture", "run_all")


def test_metric_formulas_are_loaded_from_stable_phase2k_helper() -> None:
    metrics_path = PROJECT_ROOT / "scripts" / "compare_phase2k3_rendered_views.py"

    pair_metrics, rgb255, diff_image = evaluate.load_stable_metric_helpers(
        metrics_path
    )

    assert Path(pair_metrics.__code__.co_filename).resolve() == metrics_path.resolve()
    assert Path(rgb255.__code__.co_filename).resolve() == metrics_path.resolve()
    assert Path(diff_image.__code__.co_filename).resolve() == metrics_path.resolve()
