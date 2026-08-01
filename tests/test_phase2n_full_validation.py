from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CONFIG = PROJECT_ROOT / "configs/phase2n_full_validation_manifest.json"
INFERENCE_CONFIG = PROJECT_ROOT / "configs/phase2n_full_validation_inference.json"
EVAL_CONFIG = PROJECT_ROOT / "configs/phase2n_full_validation_rendered_eval.json"


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
        "phase2n_full_validation_builder_test",
        "scripts/phase2n_build_full_validation_manifest.py",
    )


@pytest.fixture(scope="module")
def infer():
    return load_script(
        "phase2n_full_validation_infer_test",
        "scripts/phase2n_week2_infer_pilots.py",
    )


@pytest.fixture(scope="module")
def evaluate():
    return load_script(
        "phase2n_full_validation_eval_test",
        "scripts/phase2n_week2_evaluate_pilots.py",
    )


def test_manifest_exact_10_6_4_relationship_and_reuse(builder) -> None:
    manifest = builder.build_manifest(MANIFEST_CONFIG)
    validation = manifest["validation_ids"]
    pilot = manifest["pilot_validation_ids"]
    remaining = manifest["remaining_validation_ids"]

    assert len(validation) == 10
    assert len(pilot) == 6
    assert len(remaining) == 4
    assert set(pilot).isdisjoint(remaining)
    assert set(pilot) | set(remaining) == set(validation)
    assert remaining == ["B075YLG1B5", "B075YLQTJ3", "B073P51233", "B077X6WSH1"]
    assert manifest["split_counts"] == {"val": 10, "train_sanity": 0, "test": 0}
    assert manifest["test_data_used"] is False
    assert all(case["selected_input_view"] == "005" for case in manifest["cases"])
    assert all(list(case["reference_paths"]) == list(builder.VIEW_IDS) for case in manifest["cases"])

    statuses = Counter(row["reuse_status"] for row in manifest["source_records"])
    assert statuses == {
        "reused_phase2l": 20,
        "reused_phase2n_pilot": 12,
        "new_inference_required": 8,
    }
    new_rows = [
        row for row in manifest["source_records"] if row["reuse_status"] == "new_inference_required"
    ]
    assert {row["asset_id"] for row in new_rows} == set(remaining)
    assert {row["variant_id"] for row in new_rows} == {
        "pc_s1_step160",
        "pc_full_step320",
    }
    assert len(manifest["source_records"]) == 40
    assert manifest["counts"]["planned_renders"] == 240


def test_inference_profile_plans_only_eight_new_glbs(infer) -> None:
    validated = infer.validate_config(
        INFERENCE_CONFIG,
        validate_hashes=False,
        check_free_space=False,
    )
    assert [case["asset_id"] for case in validated.cases] == [
        "B075YLG1B5",
        "B075YLQTJ3",
        "B073P51233",
        "B077X6WSH1",
    ]
    assert list(validated.variants) == ["pc_s1_step160", "pc_full_step320"]
    assert infer.configured_split_counts(validated.values) == {
        "val": 4,
        "train_sanity": 0,
        "test": 0,
    }
    assert len(validated.cases) * len(validated.variants) == 8
    with pytest.raises(ValueError, match="both configured candidates"):
        infer.run_inference(
            validated,
            run_id="phase2n_full_validation_infer_v1",
            variants=("pc_s1_step160",),
        )


def test_rendered_eval_plan_is_exact_40_glbs_and_240_rows(evaluate) -> None:
    protocol = evaluate.resolve_protocol(EVAL_CONFIG, validate_files=False)
    assert protocol["counts"] == {
        "cases": 10,
        "val": 10,
        "train_sanity": 0,
        "test": 0,
        "variants": 4,
        "source_glbs": 40,
        "references": 60,
        "planned_renders": 240,
    }
    assert [row["variant"] for row in protocol["variants"]] == [
        "corrected_input_base",
        "historical_full80_500",
        "pc_s1_step160",
        "pc_full_step320",
    ]
    assert len(protocol["plan"]) == 240
    assert {row["eval_split"] for row in protocol["plan"]} == {"val"}


def test_old_pilot_configs_still_validate_unchanged(infer, evaluate) -> None:
    pilot_infer = infer.validate_config(
        PROJECT_ROOT / "configs/phase2n_week2_pilot_inference.json",
        validate_hashes=False,
        check_free_space=False,
    )
    assert len(pilot_infer.cases) == 8
    assert list(pilot_infer.variants) == list(infer.EXPECTED_VARIANT_ORDER)

    pilot_eval = evaluate.resolve_protocol(
        PROJECT_ROOT / "configs/phase2n_week2_pilot_rendered_eval.json",
        validate_files=False,
    )
    assert pilot_eval["counts"]["source_glbs"] == 48
    assert pilot_eval["counts"]["planned_renders"] == 288


def test_check_only_routes_never_call_runtime(
    builder, infer, evaluate, monkeypatch, capsys, tmp_path: Path
) -> None:
    config = json.loads(MANIFEST_CONFIG.read_text(encoding="utf-8"))
    for field in (
        "full101_split",
        "pilot_cases_config",
        "historical_eval_cases",
        "historical_base_root",
        "historical_full80_root",
        "historical_full80_checkpoint",
        "pilot_inference_run",
        "training_run_dir",
    ):
        config[field] = str((PROJECT_ROOT / config[field]).resolve())
    for candidate in config["candidate_variants"].values():
        candidate["checkpoint_manifest_path"] = str(
            (PROJECT_ROOT / candidate["checkpoint_manifest_path"]).resolve()
        )

    config["new_inference_run"] = (
        "outputs/phase2n/full_validation_inference/phase2n_full_validation_infer_v1"
    )
    config["output_manifest"] = (
        "outputs/phase2n/full_validation_manifest/check_only_manifest.json"
    )
    temporary_config = tmp_path / "phase2n_full_validation_manifest.json"
    temporary_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    original_build_manifest = builder.build_manifest

    def build_temporary_manifest(config_path):
        return original_build_manifest(config_path, project_root=tmp_path)

    monkeypatch.setattr(builder, "build_manifest", build_temporary_manifest)
    monkeypatch.setattr(
        builder,
        "write_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("write called")),
    )

    output = tmp_path / config["output_manifest"]
    sentinel = tmp_path / "check_only_sentinel.bin"
    sentinel_contents = b"phase2n-check-only-sentinel\x00"
    sentinel.write_bytes(sentinel_contents)
    assert not output.exists()
    assert builder.main(["--config", str(temporary_config), "--check-only"]) == 0
    assert not output.exists()
    assert sentinel.read_bytes() == sentinel_contents

    validated = infer.validate_config(
        INFERENCE_CONFIG,
        validate_hashes=False,
        check_free_space=False,
    )
    monkeypatch.setattr(infer, "validate_config", lambda *args, **kwargs: validated)
    monkeypatch.setattr(
        infer,
        "run_inference",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runtime called")),
    )
    assert infer.main(["--config", str(INFERENCE_CONFIG), "--check-only"]) == 0

    planned = evaluate.resolve_protocol(EVAL_CONFIG, validate_files=False)
    monkeypatch.setattr(evaluate, "resolve_protocol", lambda *args, **kwargs: planned)
    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Blender called")),
    )
    assert evaluate.main(["--config", str(EVAL_CONFIG), "--check-only"]) == 0
    stdout = capsys.readouterr().out
    assert "PHASE2N_FULL_VALIDATION_SPLIT_OK" in stdout
    assert "PHASE2N_FULL_VALIDATION_INFERENCE_READINESS_OK" in stdout
    assert "PHASE2N_FULL_VALIDATION_RENDERED_EVAL_READINESS_OK" in stdout


def test_invalid_invocation_and_manifest_overwrite_are_rejected(builder, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        builder.parse_args([])

    output = tmp_path / "already_exists.json"
    output.write_text("{}\n", encoding="utf-8")
    manifest = {"output_manifest_path": str(output)}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.write_manifest(manifest)


def test_configs_record_fixed_output_and_no_test() -> None:
    inference = json.loads(INFERENCE_CONFIG.read_text(encoding="utf-8"))
    assert inference["fixed_run_id"] == "phase2n_full_validation_infer_v1"
    assert inference["fail_if_run_exists"] is True
    assert inference["use_remesh"] is False

    rendered = json.loads(EVAL_CONFIG.read_text(encoding="utf-8"))
    assert rendered["expected_counts"]["test"] == 0
    assert rendered["expected_counts"]["train_sanity"] == 0
    assert rendered["expected_counts"]["planned_renders"] == 240
