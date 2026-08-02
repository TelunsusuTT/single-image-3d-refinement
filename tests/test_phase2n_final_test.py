from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CONFIG = PROJECT_ROOT / "configs/phase2n_final_test_manifest.json"
INFERENCE_CONFIG = PROJECT_ROOT / "configs/phase2n_final_test_inference.json"
EVAL_CONFIG = PROJECT_ROOT / "configs/phase2n_final_test_rendered_eval.json"
FREEZE_CONFIG = PROJECT_ROOT / "configs/phase2n_final_test_freeze.json"
SBATCH = PROJECT_ROOT / "env/run_phase2n_final_test_infer_a100.sbatch"


def load_script(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return load_script(
        "phase2n_final_test_builder_test",
        "scripts/phase2n_build_final_test_manifest.py",
    )


@pytest.fixture(scope="module")
def infer():
    return load_script(
        "phase2n_final_test_infer_test",
        "scripts/phase2n_week2_infer_pilots.py",
    )


@pytest.fixture(scope="module")
def evaluate():
    return load_script(
        "phase2n_final_test_eval_test",
        "scripts/phase2n_week2_evaluate_pilots.py",
    )


def validate_final_inference_without_persistent_output(
    infer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs/phase2n/final_test_inference"
    monkeypatch.setattr(
        infer,
        "validate_output_root",
        lambda output, project, expected: output_root.resolve(),
    )
    return infer.validate_config(
        INFERENCE_CONFIG,
        validate_hashes=False,
        check_free_space=False,
    )


def test_manifest_exact_test_split_freeze_and_reuse(builder) -> None:
    manifest = builder.build_manifest(MANIFEST_CONFIG)

    assert manifest["test_ids"] == list(builder.EXPECTED_TEST_IDS)
    assert len(manifest["test_ids"]) == 11
    assert manifest["split_counts"] == {"val": 0, "train_sanity": 0, "test": 11}
    assert set(manifest["test_ids"]).isdisjoint(manifest["canonical_train_ids"])
    assert set(manifest["test_ids"]).isdisjoint(
        manifest["canonical_validation_ids"]
    )
    assert set(manifest["canonical_train_ids"]).isdisjoint(
        manifest["canonical_validation_ids"]
    )
    assert all(case["selected_input_view"] == "005" for case in manifest["cases"])
    assert all(case["eval_split"] == "test" for case in manifest["cases"])
    assert all(case["source_split"] == "test" for case in manifest["cases"])
    assert all(list(case["reference_paths"]) == list(builder.VIEW_IDS) for case in manifest["cases"])
    assert all(case["reference_lighting"] == "AL" for case in manifest["cases"])
    assert all(case["fixed_mesh"] is True for case in manifest["cases"])
    assert all(case["use_remesh"] is False for case in manifest["cases"])

    statuses = Counter(row["reuse_status"] for row in manifest["source_records"])
    assert statuses == {"reused_phase2l": 22, "new_inference_required": 11}
    new_rows = [
        row
        for row in manifest["source_records"]
        if row["reuse_status"] == "new_inference_required"
    ]
    assert len(new_rows) == 11
    assert {row["variant_id"] for row in new_rows} == {"pc_full_step320"}
    assert "pc_s1_step160" not in manifest["variant_order"]
    assert manifest["excluded_variants"] == ["pc_s1_step160"]
    assert manifest["counts"] == builder.EXPECTED_COUNTS
    assert len(manifest["source_records"]) == 33
    assert manifest["counts"]["planned_renders"] == 198

    freeze = manifest["freeze"]
    assert freeze["selection_source_split"] == "validation"
    assert freeze["validation_asset_count"] == 10
    assert freeze["test_data_used_for_selection"] is False
    assert freeze["selected_candidate"]["variant_id"] == "pc_full_step320"
    assert freeze["selected_candidate"]["checkpoint_sha256"] == (
        builder.EXPECTED_CHECKPOINT_SHA256
    )
    assert freeze["validation_evidence"] == {
        "val_mae_delta": -0.44887108272976345,
        "val_ssim_delta": 0.024508462795182862,
        "front_better": True,
        "input_better": True,
        "nonfront_better": True,
        "leakage_regressions": 4,
    }


def test_final_inference_plans_only_pc_full_and_refuses_existing_run(
    infer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validated = validate_final_inference_without_persistent_output(
        infer, monkeypatch, tmp_path
    )
    assert [case["asset_id"] for case in validated.cases] == list(
        load_script(
            "phase2n_final_builder_ids_test",
            "scripts/phase2n_build_final_test_manifest.py",
        ).EXPECTED_TEST_IDS
    )
    assert list(validated.variants) == ["pc_full_step320"]
    assert infer.configured_split_counts(validated.values) == {
        "val": 0,
        "train_sanity": 0,
        "test": 11,
    }
    assert validated.baseline_reuse_report["corrected_input_base"][
        "covered_case_count"
    ] == 11
    assert validated.baseline_reuse_report["historical_full80_500"][
        "covered_case_count"
    ] == 11

    case = validated.cases[0]
    variant = validated.variants["pc_full_step320"]
    case_dir = tmp_path / "case_marker_gate"
    case_dir.mkdir()
    output_glb = case_dir / "textured_mesh.glb"
    output_glb.write_bytes(b"synthetic-test-glb")
    manifest = infer._case_manifest_expected(
        case,
        variant,
        output_glb.resolve(),
        validated,
    )
    manifest.update(
        {
            "status": "OK",
            "output_glb_size": output_glb.stat().st_size,
            "output_glb_sha256": "synthetic-sha256",
        }
    )
    (case_dir / "inference_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        infer,
        "validate_glb",
        lambda path: {
            "byte_size": output_glb.stat().st_size,
            "sha256": "synthetic-sha256",
        },
    )
    with pytest.raises(FileNotFoundError, match="case success marker"):
        infer.validate_completed_case(case_dir, case, variant, validated)
    (case_dir / "_SUCCESS").write_text(
        infer.FINAL_TEST_CASE_SUCCESS_TOKEN + "\n", encoding="utf-8"
    )
    assert infer.validate_completed_case(case_dir, case, variant, validated)[
        "status"
    ] == "OK"

    run_dir = validated.output_root / validated.values["fixed_run_id"]
    run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        infer.run_inference(
            validated,
            run_id="phase2n_final_test_infer_v1",
            variants=("pc_full_step320",),
        )


def test_rendered_eval_plan_is_exact_test_only_matrix(evaluate) -> None:
    protocol = evaluate.resolve_protocol(EVAL_CONFIG, validate_files=False)
    assert protocol["counts"] == {
        "cases": 11,
        "val": 0,
        "train_sanity": 0,
        "test": 11,
        "variants": 3,
        "source_glbs": 33,
        "references": 66,
        "planned_renders": 198,
    }
    assert [row["variant"] for row in protocol["variants"]] == [
        "corrected_input_base",
        "historical_full80_500",
        "pc_full_step320",
    ]
    assert len(protocol["plan"]) == 198
    assert {row["eval_split"] for row in protocol["plan"]} == {"test"}
    assert {row["view_id"] for row in protocol["plan"]} == set(
        evaluate.EXPECTED_VIEW_IDS
    )

    rows = []
    for plan_row in protocol["plan"]:
        delta = 0.0 if plan_row["variant"] == "corrected_input_base" else -0.25
        rows.append(
            {
                "asset_id": plan_row["asset_id"],
                "eval_split": "test",
                "selection_stratum": "phase2n_final_test",
                "variant": plan_row["variant"],
                "view_id": plan_row["view_id"],
                "view_groups": evaluate.row_view_groups(
                    plan_row["view_id"], protocol["config"]
                ),
                "candidate_mae": 1.0 + delta,
                "candidate_rmse": 2.0 + delta,
                "candidate_ssim_like": 0.8 - delta,
                "delta_mae": delta,
                "delta_rmse": delta,
                "delta_ssim_like": -delta,
                "direct_visual_change_mae": abs(delta),
            }
        )
    aggregate, _ = evaluate.build_aggregate_artifacts(rows, protocol)
    assert aggregate["status"] == "OK"
    assert aggregate["row_count"] == 198
    assert aggregate["test_data_used"] is True
    assert aggregate["test_data_used_for_selection"] is False
    assert aggregate["automatic_winner"] is None
    assert aggregate["frozen_candidate_evaluation_only"] is True
    assert aggregate["checkpoint_replacement_allowed"] is False
    assert "leakage_risk_test" in aggregate
    assert "leakage_risk_validation" not in aggregate
    for variant in ("historical_full80_500", "pc_full_step320"):
        evidence = aggregate["candidate_evidence_relative_to_corrected_input_base"][
            variant
        ]
        assert evidence["test_all_views"]["row_count"] == 66
        assert evidence["test_front_004_005"]["row_count"] == 22
        assert evidence["test_nonfront_000_003"]["row_count"] == 44


def test_check_only_routes_do_not_call_runtime_or_modify_manifest(
    builder,
    infer,
    evaluate,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured_output = Path(
        builder.build_manifest(MANIFEST_CONFIG)["output_manifest_path"]
    )
    before = configured_output.read_bytes() if configured_output.exists() else None
    assert builder.main(["--config", str(MANIFEST_CONFIG), "--check-only"]) == 0
    after = configured_output.read_bytes() if configured_output.exists() else None
    assert after == before

    validated = validate_final_inference_without_persistent_output(
        infer, monkeypatch, tmp_path
    )
    monkeypatch.setattr(infer, "validate_config", lambda *args, **kwargs: validated)
    monkeypatch.setattr(
        infer,
        "run_inference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Hunyuan runtime called")
        ),
    )
    assert infer.main(["--config", str(INFERENCE_CONFIG), "--check-only"]) == 0

    protocol = evaluate.resolve_protocol(EVAL_CONFIG, validate_files=False)
    monkeypatch.setattr(evaluate, "resolve_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Blender called")
        ),
    )
    assert evaluate.main(["--config", str(EVAL_CONFIG), "--check-only"]) == 0
    stdout = capsys.readouterr().out
    assert "PHASE2N_FINAL_TEST_FREEZE_OK" in stdout
    assert "PHASE2N_FINAL_TEST_SPLIT_OK" in stdout
    assert "PHASE2N_FINAL_TEST_REUSE_OK" in stdout
    assert "PHASE2N_FINAL_TEST_INFERENCE_READINESS_OK" in stdout
    assert "PHASE2N_FINAL_TEST_RENDERED_EVAL_READINESS_OK" in stdout


def test_invalid_split_missing_baseline_and_protocol_mismatch_fail_closed(
    builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_path = (
        PROJECT_ROOT
        / "data/manifests/datav2_frame_panels/datav2_frame_panels_full101_split.json"
    ).resolve()
    original_load_json = builder.load_json

    def invalid_split(path):
        payload = original_load_json(path)
        if Path(path).resolve() == split_path:
            payload = copy.deepcopy(payload)
            payload["splits"]["test"][0]["item_id"] = "NOT_THE_FROZEN_TEST_ASSET"
        return payload

    monkeypatch.setattr(builder, "load_json", invalid_split)
    with pytest.raises(builder.ManifestError, match="canonical test IDs/order"):
        builder.build_manifest(MANIFEST_CONFIG)
    monkeypatch.setattr(builder, "load_json", original_load_json)

    good = builder.build_manifest(MANIFEST_CONFIG)
    missing_path = Path(
        next(
            row["source_path"]
            for row in good["source_records"]
            if row["variant_id"] == "corrected_input_base"
        )
    ).resolve()
    original_require = builder.require_nonempty

    def missing_baseline(path, label):
        if Path(path).resolve() == missing_path:
            raise builder.ManifestError(f"missing {label}: {path}")
        return original_require(path, label)

    monkeypatch.setattr(builder, "require_nonempty", missing_baseline)
    with pytest.raises(builder.ManifestError, match="missing base GLB"):
        builder.build_manifest(MANIFEST_CONFIG)
    monkeypatch.setattr(builder, "require_nonempty", original_require)

    shared = builder.load_shared_builder(PROJECT_ROOT)
    monkeypatch.setattr(
        shared,
        "_validate_historical_run_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced protocol mismatch")
        ),
    )
    monkeypatch.setattr(builder, "load_shared_builder", lambda root: shared)
    with pytest.raises(builder.ManifestError, match="base protocol mismatch"):
        builder.build_manifest(MANIFEST_CONFIG)


def test_existing_output_and_manifest_overwrite_fail_closed(
    builder,
    infer,
    evaluate,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs/phase2n/final_test_inference"
    (output_root / "phase2n_final_test_infer_v1").mkdir(parents=True)
    monkeypatch.setattr(
        infer,
        "validate_output_root",
        lambda output, project, expected: output_root.resolve(),
    )
    with pytest.raises(ValueError, match="refusing overwrite"):
        infer.validate_config(
            INFERENCE_CONFIG,
            validate_hashes=False,
            check_free_space=False,
        )

    output = tmp_path / "already_exists.json"
    output.write_text("sentinel\n", encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.write_manifest({"output_manifest_path": str(output)})
    assert output.read_bytes() == before

    protocol = evaluate.resolve_protocol(EVAL_CONFIG, validate_files=False)
    protocol = {**protocol, "output_root": str(tmp_path / "rendered_eval")}
    existing_eval = Path(protocol["output_root"]) / "existing_final_test"
    existing_eval.mkdir(parents=True)
    with pytest.raises(evaluate.EvaluationError, match="use a new run ID"):
        evaluate.prepare_run(protocol, "existing_final_test", create=True)


def test_frozen_configs_and_sbatch_are_exact() -> None:
    freeze = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))
    inference = json.loads(INFERENCE_CONFIG.read_text(encoding="utf-8"))
    rendered = json.loads(EVAL_CONFIG.read_text(encoding="utf-8"))
    sbatch = SBATCH.read_text(encoding="utf-8")

    assert freeze["selected_candidate"]["variant_id"] == "pc_full_step320"
    assert freeze["test_data_used_for_selection"] is False
    assert freeze["excluded_candidate"]["variant_id"] == "pc_s1_step160"
    assert inference["variant_order"] == ["pc_full_step320"]
    assert list(inference["variants"]) == ["pc_full_step320"]
    assert inference["fixed_run_id"] == "phase2n_final_test_infer_v1"
    assert inference["fail_if_run_exists"] is True
    assert inference["use_remesh"] is False
    assert rendered["expected_counts"]["planned_renders"] == 198
    assert rendered["expected_counts"]["test"] == 11
    assert rendered["expected_counts"]["val"] == 0
    assert rendered["expected_counts"]["train_sanity"] == 0

    assert "#SBATCH -p a100" in sbatch
    assert "#SBATCH --gres=gpu:1" in sbatch
    assert "gpgpuC" not in sbatch
    assert "--constraint=a100" not in sbatch
    assert "phase2n_week2_evaluate_pilots.py" not in sbatch
    assert "PHASE2N_FINAL_TEST_INFERENCE_OK" in sbatch

    old_pilot = json.loads(
        (PROJECT_ROOT / "configs/phase2n_week2_pilot_inference.json").read_text(
            encoding="utf-8"
        )
    )
    old_full = json.loads(
        (PROJECT_ROOT / "configs/phase2n_full_validation_inference.json").read_text(
            encoding="utf-8"
        )
    )
    assert old_pilot["variant_order"] == [
        "pc_s1_step160",
        "pc_s1_step320",
        "pc_full_step160",
        "pc_full_step320",
    ]
    assert old_full["variant_order"] == ["pc_s1_step160", "pc_full_step320"]
