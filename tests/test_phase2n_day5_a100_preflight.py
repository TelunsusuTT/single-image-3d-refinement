from __future__ import annotations

import builtins
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "phase2n_day5_a100_preflight.py"


def load_script():
    spec = importlib.util.spec_from_file_location("phase2n_day5_a100_preflight_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preflight_module():
    return load_script()


def write_text(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_fake_project(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "project"
    root.mkdir()

    samples: list[Path] = []
    for index in range(80):
        sample = root / "samples" / f"sample_{index:03d}"
        sample.mkdir(parents=True)
        samples.append(sample.resolve())
    train_json = write_text(
        root / "data" / "train.json",
        json.dumps([str(path) for path in samples]) + "\n",
    )

    model_dir = root / "fixtures" / "hunyuan3d-paintpbr-v2-1"
    for relative in (
        "model_index.json",
        "unet/config.json",
        "scheduler/scheduler_config.json",
        "vae/config.json",
        "text_encoder/config.json",
    ):
        write_text(model_dir / relative, "{}\n")

    historical = write_text(
        root / "configs" / "historical.yaml",
        "\n".join(
            [
                "model:",
                "  target: hunyuanpaintpbr.HunyuanPaint",
                "  params:",
                "    noise_in_channels: 12",
                "    stable_diffusion_config:",
                f"      pretrained_model_name_or_path: {model_dir}",
                "      custom_pipeline: ./hunyuanpaintpbr",
                "init_control_from: null",
                "resume_from: null",
                "",
            ]
        ),
    )

    hypaint = root / "official_fixture" / "hy3dpaint"
    for relative in (
        "train.py",
        "hunyuanpaintpbr/unet/model.py",
        "hunyuanpaintpbr/pipeline.py",
        "src/utils/train_util.py",
    ):
        write_text(hypaint / relative)
    (root / "caches" / "hf" / "hub" / "models--facebook--dinov2-giant").mkdir(parents=True)

    write_text(
        root / "src" / "hy3dft" / "selective_training.py",
        "\n".join(
            [
                "class ProtocolCorrectedDataModule: pass",
                "def apply_trainable_scope(): pass",
                "def build_trainable_adamw(): pass",
                "def build_warmup_constant_scheduler(): pass",
                "def warmup_constant_multiplier(): pass",
                "",
            ]
        ),
    )
    write_text(root / "src" / "hy3dft" / "protocol_corrected_dataset.py")

    config = {
        "phase": "phase2n_day5_a100_preflight",
        "train_json": str(train_json.relative_to(root)),
        "sample_index": 0,
        "batch_size": 1,
        "num_workers": 0,
        "image_size": 512,
        "base_seed": 42,
        "epoch": 0,
        "augmentation_mode": "none",
        "warmup_steps": 50,
        "candidate_peak_learning_rates": {"pc_s1": 1e-6, "pc_full": 5e-7},
        "output_root": "outputs/phase2n/day5_a100_preflight",
        "scope_order": ["pc_s1", "pc_full"],
        "historical_train_config": str(historical.relative_to(root)),
        "official_hypaint": str(hypaint),
        "expected_train_count": 80,
        "exact_update_snapshot": {
            "pc_s1": {
                "mode": "all_if_cpu_memory_permits",
                "cpu_memory_limit_bytes": 1024,
                "fallback_max_tensors": 4,
            },
            "pc_full": {"mode": "deterministic_bounded", "max_tensors": 4},
        },
        "pc_s1_smoke_checkpoint": {
            "temporary_filename": "_temporary_pc_s1_selective_checkpoint.pt",
            "save_selected_parameters": True,
            "save_optimizer_state": True,
            "save_scheduler_state": True,
            "verify_frozen_sample_count": 4,
            "delete_binary_after_success": True,
        },
        "gradient_clip_norm": 1.0,
        "zero_update_tolerance": {"absolute": 1e-5, "relative": 1e-4},
        "update_ratio_safety_ceiling": 0.01,
        "scheduler_evidence_steps": [0, 25, 50, 160, 320],
    }
    config_path = write_text(root / "configs" / "day5.json", json.dumps(config) + "\n")
    return root, config_path, samples


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_config_validation_accepts_exact_contract(preflight_module, tmp_path):
    root, config_path, samples = make_fake_project(tmp_path)
    validated = preflight_module.validate_config(config_path, root)
    assert validated.values["augmentation_mode"] == "none"
    assert validated.values["candidate_peak_learning_rates"] == {"pc_s1": 1e-6, "pc_full": 5e-7}
    assert validated.train_sample_paths == tuple(samples)
    assert validated.output_root == (root / "outputs/phase2n/day5_a100_preflight").resolve()
    assert validated.true_pbr_model_dir.name == "hunyuan3d-paintpbr-v2-1"


def test_config_validation_rejects_unsafe_output_root(preflight_module, tmp_path):
    root, config_path, _samples = make_fake_project(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output_root"] = "data/day5"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Output root must be exactly"):
        preflight_module.validate_config(config_path, root)


def test_exact_train_json_contract_rejects_count_and_relative_paths(preflight_module, tmp_path):
    only_sample = tmp_path / "sample"
    only_sample.mkdir()
    wrong_count = write_text(tmp_path / "wrong_count.json", json.dumps([str(only_sample.resolve())]))
    with pytest.raises(ValueError, match="exactly 80"):
        preflight_module.validate_train_json_contract(wrong_count)

    relative = write_text(tmp_path / "relative.json", json.dumps(["relative"] * 80))
    with pytest.raises(ValueError, match="not absolute"):
        preflight_module.validate_train_json_contract(relative)


def test_stage_report_serialization_is_confined_and_non_overwriting(preflight_module, tmp_path):
    run_dir = tmp_path / "run"
    report = preflight_module.write_json_report(
        run_dir / "pc_s1" / "scope_report.json", {"scope": "pc_s1", "status": "OK"}, run_dir
    )
    assert json.loads(report.read_text(encoding="utf-8")) == {"scope": "pc_s1", "status": "OK"}
    with pytest.raises(FileExistsError):
        preflight_module.write_json_report(report, {"status": "replacement"}, run_dir)
    with pytest.raises(ValueError, match="outside"):
        preflight_module.write_json_report(tmp_path / "escape.json", {"status": "bad"}, run_dir)


class FakeParameter:
    def __init__(self, *, requires_grad: bool = True, numel: int = 4, element_size: int = 4):
        self.requires_grad = requires_grad
        self._numel = numel
        self._element_size = element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def all(self):
        return self


class FakeTensor:
    def __init__(self, values):
        self.values = tuple(float(value) for value in values)
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def __sub__(self, other):
        return FakeTensor(left - right for left, right in zip(self.values, other.values))


class FakeLinalg:

    def vector_norm(self, tensor):
        return FakeScalar(math.sqrt(sum(value * value for value in tensor.values)))


class FakeTorch:
    linalg = FakeLinalg()


    def isfinite(self, tensor):
        return FakeScalar(all(math.isfinite(value) for value in tensor.values))


    def count_nonzero(self, tensor):
        return FakeScalar(sum(value != 0.0 for value in tensor.values))


class FakeGradientParameter:
    def __init__(self, *, requires_grad, gradient):
        self.requires_grad = requires_grad
        self.grad = gradient


def test_deterministic_parameter_snapshot_selection(preflight_module):
    parameters = [(f"layer.{index:02d}.weight", FakeParameter()) for index in range(40)]
    first = preflight_module.select_update_snapshot_names(
        parameters,
        "pc_full",
        pc_s1_cpu_limit_bytes=1,
        pc_s1_fallback_limit=4,
        pc_full_limit=16,
    )
    second = preflight_module.select_update_snapshot_names(
        reversed(parameters),
        "pc_full",
        pc_s1_cpu_limit_bytes=1,
        pc_s1_fallback_limit=4,
        pc_full_limit=16,
    )
    assert first == second
    assert first["selected_name_count"] == 16
    assert first["sampling_rule"] == "evenly_spaced_over_sorted_trainable_names"
    assert first["selected_names"][0] == "layer.00.weight"
    assert first["selected_names"][-1] == "layer.39.weight"


def test_sampled_update_ratio_calculation(preflight_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    before = {"weight": FakeTensor([3.0, 4.0]), "bias": FakeTensor([2.0])}
    after = {"weight": FakeTensor([3.3, 4.4]), "bias": FakeTensor([2.0])}
    report = preflight_module.calculate_update_ratios(before, after)
    rows = {row["name"]: row for row in report["per_tensor"]}
    assert rows["weight"]["weight_norm"] == pytest.approx(5.0)
    assert rows["weight"]["update_norm"] == pytest.approx(0.5)
    assert rows["weight"]["update_to_weight_ratio"] == pytest.approx(0.1)
    assert report["nonzero_update_count"] == 1


def test_gradient_classification_is_finite_and_nonzero(preflight_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    selected = FakeGradientParameter(requires_grad=True, gradient=FakeTensor([0.25, -0.5]))
    frozen = FakeGradientParameter(requires_grad=False, gradient=None)
    report = preflight_module.classify_gradients([("selected", selected), ("frozen", frozen)])
    preflight_module.validate_gradient_report(report)
    assert report["trainable_tensor_count"] == 1
    assert report["nonzero_gradient_count"] == 1
    assert report["frozen_tensors_with_gradients"] == []


def test_frozen_gradient_is_a_hard_failure(preflight_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    selected = FakeGradientParameter(requires_grad=True, gradient=FakeTensor([0.5]))
    frozen = FakeGradientParameter(requires_grad=False, gradient=FakeTensor([0.1]))
    report = preflight_module.classify_gradients([("selected", selected), ("frozen", frozen)])
    with pytest.raises(RuntimeError, match="Frozen tensors received gradients"):
        preflight_module.validate_gradient_report(report)


def test_checkpoint_manifest_hash_and_cleanup(preflight_module, tmp_path):
    checkpoint = write_text(tmp_path / "_temporary.pt", "selective adapter state\n")
    manifest = preflight_module.checkpoint_file_manifest(checkpoint)
    expected_hash = preflight_module.sha256_file(checkpoint)
    cleaned = preflight_module.cleanup_temporary_checkpoint(checkpoint, manifest)
    assert manifest["sha256"] == expected_hash
    assert manifest["byte_size"] > 0
    assert cleaned["binary_removed_after_success"] is True
    assert not checkpoint.exists()


def test_check_only_uses_no_hunyuan_or_cuda_import_and_writes_no_inputs(
    preflight_module, tmp_path, monkeypatch, capsys
):
    root, config_path, _samples = make_fake_project(tmp_path)
    before = tree_state(root)
    attempted_heavy_imports: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith(("torch.", "hunyuan", "omegaconf", "pytorch_lightning")):
            attempted_heavy_imports.append(name)
            raise AssertionError(f"check-only attempted heavy import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    validated = preflight_module.run_check_only(config_path, root)
    assert len(validated.train_sample_paths) == 80
    assert attempted_heavy_imports == []
    assert tree_state(root) == before
    assert not (root / "outputs").exists()
    assert preflight_module.CHECK_ONLY_TOKEN in capsys.readouterr().out


def test_required_success_token_contract(preflight_module):
    assert preflight_module.REQUIRED_RUNTIME_TOKENS == (
        "PHASE2N_DAY5_MODEL_LOAD_OK",
        "PHASE2N_DAY5_REAL_BATCH_LOSS_OK",
        "PHASE2N_DAY5_PC_S1_SCOPE_OK",
        "PHASE2N_DAY5_PC_S1_GRADIENT_OK",
        "PHASE2N_DAY5_PC_S1_UPDATE_OK",
        "PHASE2N_DAY5_PC_S1_CHECKPOINT_RELOAD_OK",
        "PHASE2N_DAY5_PC_FULL_SCOPE_OK",
        "PHASE2N_DAY5_PC_FULL_GRADIENT_OK",
        "PHASE2N_DAY5_PC_FULL_UPDATE_OK",
        "PHASE2N_DAY5_PC_FULL_AUDIT_OK",
        "PHASE2N_DAY5_A100_PREFLIGHT_OK",
    )
