from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/phase2n_closeout.json"


def load_script():
    path = PROJECT_ROOT / "scripts/check_phase2n_closeout.py"
    spec = importlib.util.spec_from_file_location("phase2n_closeout_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return load_script()


def load_record() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_record(
    tmp_path: Path,
    mutate: Callable[[dict], None] | None = None,
) -> Path:
    record = copy.deepcopy(load_record())
    if mutate is not None:
        mutate(record)
    path = tmp_path / "phase2n_closeout.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def copy_file(source_root: Path, target_root: Path, relative: str) -> None:
    source = source_root / relative
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def clone_validation_evidence(target_root: Path, *, marker: bool) -> None:
    source = (
        PROJECT_ROOT
        / "outputs/phase2n/full_validation_rendered_eval/"
        "phase2n_full_validation_eval_v1"
    )
    files = [
        "summary.json",
        "metrics/aggregate.json",
        "metrics/per_view.jsonl",
    ]
    if marker:
        files.append("_SUCCESS")
    for relative in files:
        copy_file(source, target_root, relative)


def clone_final_eval_evidence(target_root: Path, *, marker: bool) -> None:
    source = (
        PROJECT_ROOT
        / "outputs/phase2n/final_test_rendered_eval/"
        "phase2n_final_test_eval_v1"
    )
    files = [
        "summary.json",
        "metrics/aggregate.json",
        "metrics/per_view.jsonl",
        "00_RUNTIME_MANIFEST.json",
        "checkpoint_manifest.json",
    ]
    if marker:
        files.append("_SUCCESS")
    for relative in files:
        copy_file(source, target_root, relative)


def test_valid_completed_closeout_passes(checker) -> None:
    result = checker.check_closeout(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert result == {
        "status": "OK",
        "phase_status": "CLOSED",
        "validation_assets": 10,
        "test_assets": 11,
        "metric_rows": 198,
        "final_candidate": "pc_full_step320",
        "default_recommendation": "corrected_input_base",
    }


def test_missing_validation_success_marker_fails(
    checker, tmp_path: Path
) -> None:
    temporary_root = tmp_path / "validation_without_marker"
    clone_validation_evidence(temporary_root, marker=False)

    config = write_record(
        tmp_path,
        lambda record: record["runs"]["full_validation"].update(
            {"root": str(temporary_root)}
        ),
    )
    with pytest.raises(checker.CloseoutError, match="validation success marker"):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_missing_final_test_success_marker_fails(
    checker, tmp_path: Path
) -> None:
    temporary_root = tmp_path / "final_eval_without_marker"
    clone_final_eval_evidence(temporary_root, marker=False)

    config = write_record(
        tmp_path,
        lambda record: record["runs"]["final_test_evaluation"].update(
            {"root": str(temporary_root)}
        ),
    )
    with pytest.raises(
        checker.CloseoutError, match="final_test_evaluation success marker"
    ):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_incorrect_fixed_split_counts_fail(checker, tmp_path: Path) -> None:
    config = write_record(
        tmp_path,
        lambda record: record["fixed_split_counts"].update({"train": 79}),
    )
    with pytest.raises(checker.CloseoutError, match="fixed split counts"):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_validation_test_contamination_fails(
    checker, tmp_path: Path
) -> None:
    temporary_root = tmp_path / "contaminated_validation"
    clone_validation_evidence(temporary_root, marker=True)

    summary_path = temporary_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["split_counts"] = {"val": 9, "train_sanity": 0, "test": 1}
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    rows_path = temporary_root / "metrics/per_view.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["eval_split"] = "test"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    config = write_record(
        tmp_path,
        lambda record: record["runs"]["full_validation"].update(
            {"root": str(temporary_root)}
        ),
    )
    with pytest.raises(checker.CloseoutError) as exc_info:
        checker.check_closeout(config, project_root=PROJECT_ROOT)
    message = str(exc_info.value)
    assert "validation split counts" in message
    assert "validation row split coverage" in message


def test_changed_checkpoint_hash_fails(checker, tmp_path: Path) -> None:
    config = write_record(
        tmp_path,
        lambda record: record["model_selection"]["final_candidate"].update(
            {"checkpoint_sha256": "0" * 64}
        ),
    )
    with pytest.raises(checker.CloseoutError, match="checkpoint SHA-256"):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_wrong_candidate_fails(checker, tmp_path: Path) -> None:
    config = write_record(
        tmp_path,
        lambda record: record["model_selection"]["final_candidate"].update(
            {"variant_id": "pc_s1_step160"}
        ),
    )
    with pytest.raises(checker.CloseoutError, match="final candidate variant_id"):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_incorrect_metric_value_fails(checker, tmp_path: Path) -> None:
    def mutate(record: dict) -> None:
        record["final_test_metrics"]["all_views"]["pc_full_step320"][
            "mean_mae"
        ] += 1.0

    config = write_record(tmp_path, mutate)
    with pytest.raises(
        checker.CloseoutError, match="final-test metric projection"
    ):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_incorrect_leakage_count_fails(checker, tmp_path: Path) -> None:
    config = write_record(
        tmp_path,
        lambda record: record["leakage_regressions_relative_to_base"][
            "pc_full_step320"
        ].update({"count": 6}),
    )
    with pytest.raises(
        checker.CloseoutError, match="leakage regression evidence"
    ):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_analysis_packet_hash_mismatch_fails(
    checker, tmp_path: Path
) -> None:
    config = write_record(
        tmp_path,
        lambda record: record["analysis_packets"][0].update(
            {"sha256": "f" * 64}
        ),
    )
    with pytest.raises(checker.CloseoutError, match="packet SHA-256"):
        checker.check_closeout(config, project_root=PROJECT_ROOT)


def test_test_based_model_reselection_fails(checker, tmp_path: Path) -> None:
    def mutate(record: dict) -> None:
        selection = record["model_selection"]
        selection["basis"] = "test"
        selection["selection_source_split"] = "test"
        selection["test_data_used_for_selection"] = True
        selection["new_candidate_selected_from_test"] = True

    config = write_record(tmp_path, mutate)
    with pytest.raises(checker.CloseoutError) as exc_info:
        checker.check_closeout(config, project_root=PROJECT_ROOT)
    message = str(exc_info.value)
    assert "selection basis" in message
    assert "no candidate may be selected from final-test evidence" in message


def test_checker_is_read_only(checker) -> None:
    record = load_record()
    evidence = [CONFIG_PATH, *checker.collect_evidence_paths(record, PROJECT_ROOT)]

    def snapshot(path: Path) -> tuple[int, int, str]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, digest

    before = {path: snapshot(path) for path in evidence}
    checker.check_closeout(CONFIG_PATH, project_root=PROJECT_ROOT)
    after = {path: snapshot(path) for path in evidence}
    assert after == before

