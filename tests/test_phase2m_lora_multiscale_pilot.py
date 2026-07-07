from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import check_phase2m_lora_multiscale_pilot_ready as ready  # noqa: E402
import phase2m_lora_multiscale_pilot as pilot  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fake_adapter_config() -> dict:
    return {
        "backend": "local_linear_fallback",
        "target_count": 128,
        "target_names": [f"target.{idx}" for idx in range(128)],
        "rank": 4,
        "alpha": 4.0,
        "dropout": 0.05,
        "safety": {"merge_into_base": False, "save_pretrained_full_model": False},
    }


def make_case(root: Path, item_id: str, split: str, selected_view: str = "005") -> dict:
    case_dir = root / "cases" / split / item_id
    mesh = case_dir / "input" / "mesh.glb"
    image = case_dir / "input" / "image.png"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(f"mesh-{item_id}".encode())
    image.write_bytes(f"image-{item_id}".encode())
    return {
        "item_id": item_id,
        "eval_split": split,
        "case_dir": str(case_dir),
        "case_input_mesh": str(mesh),
        "case_input_image": str(image),
        "selected_input_view": selected_view,
    }


def fake_eval_cases(root: Path) -> Path:
    data = {
        "cases": [
            make_case(root, "VAL1", "val"),
            make_case(root, "TEST1", "test"),
            make_case(root, "TRAIN1", "train_sanity"),
        ]
    }
    path = root / "eval_cases.json"
    write_json(path, data)
    return path


def test_select_pilot_cases_chooses_val_test_train_sanity(tmp_path: Path) -> None:
    cases = pilot.smoke.load_eval_cases(fake_eval_cases(tmp_path))

    selected = pilot.select_pilot_cases(cases)

    assert [case["_phase2m_case_info"]["eval_split"] for case in selected] == ["val", "test", "train_sanity"]
    assert [case["_phase2m_case_info"]["item_id"] for case in selected] == ["VAL1", "TEST1", "TRAIN1"]


def test_multiscale_output_paths_use_expected_labels() -> None:
    paths = pilot.lora_output_paths(Path("out"), "test", "CASE", 0.5)
    paths_100 = pilot.lora_output_paths(Path("out"), "test", "CASE", 1.0)

    assert paths["glb"] == Path("out/lora_scale050/test/CASE/lora_scale050_textured_mesh.glb")
    assert paths_100["obj"] == Path("out/lora_scale100/test/CASE/lora_scale100_textured_mesh.obj")
    assert pilot.base_output_paths(Path("out"), "val", "CASE")["glb"] == Path("out/base/val/CASE/base_textured_mesh.glb")


def make_ready_args(root: Path, scales: list[float] | None = None, sbatch_text: str | None = None, m3a_status: str = "OK") -> argparse.Namespace:
    adapter = root / "outputs" / "phase2m" / "lora_train" / "adapter_final.pt"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_bytes(b"adapter-bytes")
    adapter_config = adapter.parent / "adapter_config.json"
    write_json(adapter_config, fake_adapter_config())
    eval_cases = fake_eval_cases(root)
    m3a_summary = root / "outputs" / "phase2m" / "lora_infer_smoke_scale075" / "smoke_summary.json"
    write_json(m3a_summary, {"status": m3a_status, "success": m3a_status == "OK"})
    script = root / "scripts" / "phase2m_lora_multiscale_pilot.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("script\n", encoding="utf-8")
    sbatch = root / "env" / "run_phase2m_lora_multiscale_pilot_a100.sbatch"
    sbatch.parent.mkdir(parents=True, exist_ok=True)
    sbatch.write_text(
        sbatch_text
        or "#SBATCH -p a100\n#SBATCH --gres=gpu:1\npython scripts/check_phase2m_lora_multiscale_pilot_ready.py\npython scripts/phase2m_lora_multiscale_pilot.py --scales 0.5 0.75 1.0\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        adapter_path=adapter,
        adapter_config=adapter_config,
        eval_cases_json=eval_cases,
        m3a_summary=m3a_summary,
        output_dir=root / "outputs" / "phase2m" / "lora_multiscale_pilot",
        script=script,
        sbatch=sbatch,
        scales=scales if scales is not None else [0.5, 0.75, 1.0],
        min_adapter_mb=0.000001,
        project_root=root,
        dry_run=False,
    )


def test_multiscale_readiness_passes_with_fake_files(tmp_path: Path) -> None:
    args = make_ready_args(tmp_path)

    report = ready.check_readiness(args)

    assert report.errors == []


def test_multiscale_readiness_rejects_wrong_scales(tmp_path: Path) -> None:
    args = make_ready_args(tmp_path, scales=[0.75])

    report = ready.check_readiness(args)

    assert any("scale list" in error for error in report.errors)


def test_multiscale_readiness_requires_m3a_success(tmp_path: Path) -> None:
    args = make_ready_args(tmp_path, m3a_status="FAIL")

    report = ready.check_readiness(args)

    assert any("M3A smoke summary" in error for error in report.errors)


def test_real_m3b_sbatch_uses_expected_scale_and_partition_tokens() -> None:
    text = (PROJECT_ROOT / "env" / "run_phase2m_lora_multiscale_pilot_a100.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH -p a100" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "--scales 0.5 0.75 1.0" in text
    assert "gpgpuC" not in text
    assert "--constraint=a100" not in text


def test_lora_adapter_report_summary_includes_keys_and_scaled_counts() -> None:
    report = pilot.summarize_lora_adapter_reports(
        [
            {
                "variant": "lora_scale050",
                "case_id": "CASE1",
                "eval_split": "val",
                "lora_scale": 0.5,
                "missing_adapter_keys": ["missing.a"],
                "unexpected_adapter_keys": [],
                "scaled_module_count": 128,
            },
            {
                "variant": "lora_scale075",
                "case_id": "CASE1",
                "eval_split": "val",
                "lora_scale": 0.75,
                "missing_adapter_keys": [],
                "unexpected_adapter_keys": ["extra.b"],
                "scaled_module_count": 128,
            },
            {"variant": "base", "case_id": "CASE1"},
        ]
    )

    assert report["missing_adapter_keys"] == ["missing.a"]
    assert report["unexpected_adapter_keys"] == ["extra.b"]
    assert report["scaled_module_count_per_scale"] == {"0.5": [128], "0.75": [128]}
    assert len(report["adapter_key_report_by_variant"]) == 2
