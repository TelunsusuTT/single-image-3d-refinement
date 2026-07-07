from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import check_phase2m_lora_infer_smoke_ready as ready  # noqa: E402
import phase2m_lora_infer_smoke as smoke  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fake_adapter_config(backend: str = "local_linear_fallback") -> dict:
    return {
        "backend": backend,
        "target_count": 128,
        "target_names": [f"layer.{idx}" for idx in range(128)],
        "rank": 4,
        "alpha": 4.0,
        "dropout": 0.05,
        "safety": {"merge_into_base": False, "save_pretrained_full_model": False},
    }


def fake_eval_cases(root: Path) -> Path:
    case_dir = root / "outputs" / "cases" / "val" / "CASE005"
    mesh = case_dir / "input" / "mesh.glb"
    image = case_dir / "input" / "image.png"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(b"mesh")
    image.write_bytes(b"image")
    cases = {
        "cases": [
            {
                "item_id": "CASE000",
                "eval_split": "val",
                "case_dir": str(root / "missing"),
                "selected_input_view": "000",
            },
            {
                "item_id": "CASE005",
                "eval_split": "val",
                "case_dir": str(case_dir),
                "case_input_mesh": str(mesh),
                "case_input_image": str(image),
                "selected_input_view": "005",
            },
        ]
    }
    path = root / "outputs" / "phase2l" / "eval_cases.json"
    write_json(path, cases)
    return path


def test_scale_label_and_output_paths() -> None:
    paths = smoke.output_paths(Path("out"), "val", "CASE", 0.75)

    assert smoke.format_scale_label(0.75) == "scale075"
    assert paths["base_glb"] == Path("out/base/val/CASE/base_textured_mesh.glb")
    assert paths["lora_glb"] == Path("out/lora_scale075/val/CASE/lora_scale075_textured_mesh.glb")


def test_select_eval_case_prefers_selected_view_005(tmp_path: Path) -> None:
    cases_path = fake_eval_cases(tmp_path)
    cases = smoke.load_eval_cases(cases_path)
    case = smoke.select_eval_case(cases)
    info = smoke.validate_case(case)

    assert info["item_id"] == "CASE005"
    assert info["selected_input_view"] == "005"


def test_validate_adapter_config_rejects_peft_backend() -> None:
    try:
        smoke.validate_adapter_config(fake_adapter_config(backend="peft"))
    except RuntimeError as exc:
        assert "local_linear_fallback" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("PEFT backend was accepted")


def test_lora_runtime_scale_sets_absolute_multiplier() -> None:
    class FakeLora:
        def __init__(self) -> None:
            self.lora_A = object()
            self.lora_B = object()
            self.scale = 1.0

    class FakeModel:
        def __init__(self) -> None:
            self.mods = [object(), FakeLora(), FakeLora()]

        def modules(self):
            return list(self.mods)

    model = FakeModel()
    first_count = smoke.set_lora_runtime_scale(model, 0.5)
    second_count = smoke.set_lora_runtime_scale(model, 0.75)

    assert first_count == 2
    assert second_count == 2
    assert model.mods[1].scale == 0.75
    assert model.mods[2].scale == 0.75
    assert model.mods[1]._phase2m_base_lora_scale == 1.0


def test_lora_runtime_scale_preserves_non_unit_base_scale() -> None:
    class FakeLora:
        def __init__(self) -> None:
            self.lora_A = object()
            self.lora_B = object()
            self.scale = 2.0

    class FakeModel:
        def __init__(self) -> None:
            self.mods = [FakeLora()]

        def modules(self):
            return list(self.mods)

    model = FakeModel()
    smoke.set_lora_runtime_scale(model, 0.5)
    smoke.set_lora_runtime_scale(model, 0.75)

    assert model.mods[0].scale == 1.5
    assert model.mods[0]._phase2m_base_lora_scale == 2.0


def make_ready_args(root: Path, backend: str = "local_linear_fallback", sbatch_text: str | None = None) -> argparse.Namespace:
    adapter = root / "outputs" / "phase2m" / "lora_train" / "adapter_final.pt"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_bytes(b"adapter-bytes")
    adapter_config = adapter.parent / "adapter_config.json"
    write_json(adapter_config, fake_adapter_config(backend=backend))
    eval_config = root / "configs" / "datav2_frame_full80_eval.json"
    eval_config.parent.mkdir(parents=True, exist_ok=True)
    eval_config.write_text("{}\n", encoding="utf-8")
    eval_cases = fake_eval_cases(root)
    script = root / "scripts" / "phase2m_lora_infer_smoke.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("script\n", encoding="utf-8")
    sbatch = root / "env" / "run_phase2m_lora_infer_smoke_scale075_a100.sbatch"
    sbatch.parent.mkdir(parents=True, exist_ok=True)
    sbatch.write_text(
        sbatch_text
        or "#SBATCH -p a100\n#SBATCH --gres=gpu:1\npython scripts/check_phase2m_lora_infer_smoke_ready.py\npython scripts/phase2m_lora_infer_smoke.py --adapter-path a --lora-scale 0.75\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        adapter_path=adapter,
        adapter_config=adapter_config,
        eval_config=eval_config,
        eval_cases_json=eval_cases,
        output_dir=root / "outputs" / "phase2m" / "lora_infer_smoke_scale075",
        script=script,
        sbatch=sbatch,
        min_adapter_mb=0.000001,
        project_root=root,
        dry_run=False,
    )


def test_lora_infer_readiness_passes_with_fake_files(tmp_path: Path) -> None:
    args = make_ready_args(tmp_path)

    report = ready.check_readiness(args)

    assert report.errors == []


def test_lora_infer_readiness_rejects_wrong_backend(tmp_path: Path) -> None:
    args = make_ready_args(tmp_path, backend="peft")

    report = ready.check_readiness(args)

    assert any("local_linear_fallback" in error for error in report.errors)


def test_lora_infer_readiness_rejects_old_slurm_syntax(tmp_path: Path) -> None:
    args = make_ready_args(
        tmp_path,
        sbatch_text="#SBATCH --partition=gpgpuC\n#SBATCH --constraint=a100\npython scripts/phase2m_lora_infer_smoke.py --lora-scale 0.75\n",
    )

    report = ready.check_readiness(args)

    assert any("gpgpuC" in error for error in report.errors)
    assert any("--constraint=a100" in error for error in report.errors)
    assert any("partition must be exactly" in error for error in report.errors)


def test_real_m3a_sbatch_uses_literal_scale_and_new_partition() -> None:
    text = (PROJECT_ROOT / "env" / "run_phase2m_lora_infer_smoke_scale075_a100.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH -p a100" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "--lora-scale 0.75" in text
    assert "gpgpuC" not in text
    assert "--constraint=a100" not in text
