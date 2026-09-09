from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _evaluation import (  # noqa: E402
    EvaluationContractError,
    METRIC_NAMES,
    PAIR_NAMES,
    VIEW_GROUP_NAMES,
    load_cases_manifest,
    load_evaluation_config,
    summarize_view_groups,
    validate_cases_manifest_payload,
)
from aggregate_rendered_metrics import aggregate  # noqa: E402
from compare_rendered_views import (  # noqa: E402
    board_headers,
    diff_stats,
    load_view_images,
    pair_metrics,
)
from render_fixed_views_blender import (  # noqa: E402
    argv_after_double_dash,
    clear_scene,
    parse_args as parse_render_args,
)


EVALUATION_PATH = PROJECT_ROOT / "configs/evaluation/fixed_view_rerendering.json"


def test_blender_renderer_resolves_local_helpers_in_isolated_python() -> None:
    renderer = PROJECT_ROOT / "scripts/render_fixed_views_blender.py"
    code = (
        "import runpy; "
        f"runpy.run_path({str(renderer)!r}, run_name='renderer_import_check')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_renderer_clears_orphaned_data_between_glbs() -> None:
    events: list[tuple[object, ...]] = []

    class FakeCollection(list[object]):
        def __init__(self, name: str) -> None:
            super().__init__([f"{name}_block"])
            self.name = name

        def remove(self, block: object, *, do_unlink: bool) -> None:
            events.append(("remove", self.name, block, do_unlink))
            super().remove(block)

    class ObjectOps:
        def select_all(self, *, action: str) -> None:
            events.append(("select_all", action))

        def delete(self, *, use_global: bool) -> None:
            events.append(("delete", use_global))

    class OutlinerOps:
        def orphans_purge(self, *, do_recursive: bool) -> None:
            events.append(("orphans_purge", do_recursive))

    collection_names = (
        "meshes",
        "materials",
        "images",
        "textures",
        "cameras",
        "lights",
        "curves",
    )
    collections = {name: FakeCollection(name) for name in collection_names}
    bpy = SimpleNamespace(
        ops=SimpleNamespace(object=ObjectOps(), outliner=OutlinerOps()),
        data=SimpleNamespace(**collections),
    )

    clear_scene(bpy)

    assert events[:2] == [("select_all", "SELECT"), ("delete", False)]
    assert all(not collection for collection in collections.values())
    assert events.count(("orphans_purge", True)) == 2


def canonical_evaluation() -> dict[str, object]:
    _path, evaluation = load_evaluation_config(EVALUATION_PATH)
    return evaluation


def cases_payload(tmp_path: Path, asset_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "baseline_label": "Corrected-Conditioning Baseline",
        "candidate_label": "Protocol-Corrected Broad-Scope Fine-Tuning",
        "reference_image_path_template": str(
            tmp_path / "references" / "{asset_id}" / "{view_id}.png"
        ),
        "cases": {
            asset_id: {
                "asset_label": f"Panel {asset_id}",
                "baseline_glb": str(tmp_path / "baseline" / f"{asset_id}.glb"),
                "candidate_glb": str(tmp_path / "candidate" / f"{asset_id}.glb"),
            }
            for asset_id in asset_ids
        },
    }


def metric_pair(value: float) -> dict[str, object]:
    return {
        "mae": value,
        "rmse": value + 1.0,
        "psnr": 30.0 - value,
        "psnr_is_infinite": False,
        "ssim_like": 1.0 - value / 100.0,
    }


def view_row(view_id: str, index: int) -> dict[str, object]:
    baseline = metric_pair(10.0 + index)
    candidate = metric_pair(8.0 + index)
    return {
        "view_id": view_id,
        "baseline_vs_candidate": metric_pair(3.0 + index),
        "baseline_vs_reference": baseline,
        "candidate_vs_reference": candidate,
        "candidate_minus_baseline": {
            metric: float(candidate[metric]) - float(baseline[metric])
            for metric in METRIC_NAMES
        },
    }


def write_case_metrics(
    output_root: Path,
    asset_id: str,
    evaluation: dict[str, object],
    manifest: dict[str, object],
) -> None:
    views = [
        view_row(view_id, index)
        for index, view_id in enumerate(evaluation["view_groups"]["all"])
    ]
    path = output_root / "metrics" / asset_id / "rendered_view_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_id": evaluation["evaluation_id"],
                "asset_id": asset_id,
                "baseline_label": manifest["baseline_label"],
                "candidate_label": manifest["candidate_label"],
                "view_groups": evaluation["view_groups"],
                "metrics": evaluation["metrics"],
                "views": views,
            }
        ),
        encoding="utf-8",
    )


def test_canonical_evaluation_owns_camera_render_groups_and_metrics() -> None:
    evaluation = canonical_evaluation()

    assert [pose["view_id"] for pose in evaluation["camera"]["poses"]] == [
        "000",
        "001",
        "002",
        "003",
        "004",
        "005",
    ]
    assert [pose["azimuth_degrees"] for pose in evaluation["camera"]["poses"]] == [
        0.0,
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
    ]
    assert evaluation["rendering"]["resolution"] == [512, 512]
    assert tuple(evaluation["view_groups"]) == VIEW_GROUP_NAMES
    assert tuple(evaluation["metrics"]["enabled"]) == METRIC_NAMES


def test_example_cases_manifest_contains_only_paths_labels_and_reference_template() -> None:
    _path, manifest = load_cases_manifest(
        PROJECT_ROOT / "data/manifests/evaluation_cases.example.json"
    )

    assert set(manifest) == {
        "schema_version",
        "baseline_label",
        "candidate_label",
        "reference_image_path_template",
        "cases",
    }
    assert "view_ids" not in manifest
    assert "render_resolution" not in manifest
    assert "background_color" not in manifest


def test_cases_manifest_rejects_evaluation_settings(tmp_path: Path) -> None:
    manifest = cases_payload(tmp_path, ["DYNAMIC_ASSET"])
    manifest["view_ids"] = ["000"]

    with pytest.raises(EvaluationContractError, match="unsupported fields"):
        validate_cases_manifest_payload(manifest)


def test_size_mismatch_is_rejected_without_resize(tmp_path: Path) -> None:
    evaluation = copy.deepcopy(canonical_evaluation())
    evaluation["rendering"]["resolution"] = [4, 4]
    asset_id = "DYNAMIC_ASSET"
    manifest = cases_payload(tmp_path, [asset_id])
    view_id = "000"
    reference_path = tmp_path / "references" / asset_id / f"{view_id}.png"
    baseline_path = tmp_path / "evaluation" / "renders" / asset_id / "baseline" / f"{view_id}.png"
    candidate_path = tmp_path / "evaluation" / "renders" / asset_id / "candidate" / f"{view_id}.png"
    for path, size in (
        (reference_path, (4, 4)),
        (baseline_path, (5, 4)),
        (candidate_path, (4, 4)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (20, 40, 60)).save(path)

    with pytest.raises(EvaluationContractError, match="never resized"):
        load_view_images(
            evaluation,
            manifest,
            tmp_path / "evaluation",
            asset_id,
            view_id,
        )


def test_identical_images_have_json_safe_infinite_psnr() -> None:
    image = Image.new("RGB", (4, 4), (20, 40, 60))

    stats = diff_stats(image, image.copy())
    metrics = pair_metrics(image, image.copy(), canonical_evaluation()["metrics"])

    assert stats["mae"] == 0
    assert stats["rmse"] == 0
    assert stats["psnr"] is None
    assert stats["psnr_is_infinite"] is True
    assert metrics["ssim_like"] == pytest.approx(1.0)


def test_group_summaries_include_every_required_metric() -> None:
    evaluation = canonical_evaluation()
    rows = [
        view_row(view_id, index)
        for index, view_id in enumerate(evaluation["view_groups"]["all"])
    ]

    groups = summarize_view_groups(rows, evaluation)

    assert tuple(groups) == VIEW_GROUP_NAMES
    assert groups["all"]["view_count"] == 6
    assert groups["front"]["view_count"] == 2
    assert groups["conditioning"]["view_count"] == 1
    assert groups["non_front"]["view_count"] == 4
    for group in groups.values():
        for pair_name in PAIR_NAMES:
            assert all(metric in group[pair_name] for metric in METRIC_NAMES)


def test_aggregate_uses_manifest_assets_labels_and_configured_groups(
    tmp_path: Path,
) -> None:
    evaluation = canonical_evaluation()
    asset_ids = ["DYNAMIC_A", "DYNAMIC_B"]
    manifest = cases_payload(tmp_path, asset_ids)
    cases_config = tmp_path / "cases.json"
    cases_config.write_text(json.dumps(manifest), encoding="utf-8")
    output_root = tmp_path / "evaluation"
    for asset_id in asset_ids:
        write_case_metrics(output_root, asset_id, evaluation, manifest)

    summary = aggregate(EVALUATION_PATH, cases_config, output_root)

    assert summary["asset_ids"] == asset_ids
    assert summary["case_count"] == 2
    assert summary["total_views"] == 12
    assert summary["groups"]["all"]["view_count"] == 12
    assert summary["groups"]["front"]["view_count"] == 4
    assert summary["groups"]["conditioning"]["view_count"] == 2
    assert summary["groups"]["non_front"]["view_count"] == 8
    report = (
        output_root / "summary" / "rendered_view_metrics_summary.md"
    ).read_text(encoding="utf-8")
    assert manifest["baseline_label"] in report
    assert manifest["candidate_label"] in report
    for group_name in VIEW_GROUP_NAMES:
        assert f"### {group_name}" in report


def test_aggregate_fails_when_case_metrics_are_missing(tmp_path: Path) -> None:
    manifest = cases_payload(tmp_path, ["DYNAMIC_ASSET"])
    cases_config = tmp_path / "cases.json"
    cases_config.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="DYNAMIC_ASSET"):
        aggregate(EVALUATION_PATH, cases_config, tmp_path / "evaluation")


def test_board_headers_use_manifest_labels(tmp_path: Path) -> None:
    manifest = cases_payload(tmp_path, ["DYNAMIC_ASSET"])

    headers = board_headers(manifest)

    assert any(manifest["baseline_label"] in header for header in headers)
    assert any(manifest["candidate_label"] in header for header in headers)

def test_renderer_argument_parsing_does_not_import_blender() -> None:
    evaluation_path = Path("evaluation.json")
    cases_path = Path("cases.json")
    output_root = Path("evaluation-output")

    args = parse_render_args(
        [
            "--evaluation-config",
            str(evaluation_path),
            "--cases-config",
            str(cases_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert args.evaluation_config == evaluation_path
    assert args.cases_config == cases_path
    assert args.output_root == output_root
    assert argv_after_double_dash(["blender", "--", "--output-root", "out"]) == [
        "--output-root",
        "out",
    ]
