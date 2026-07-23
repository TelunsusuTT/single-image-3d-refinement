from __future__ import annotations

import builtins
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "phase2n_week2_infer_pilots.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.scope_checkpoint import (  # noqa: E402
    load_scope_checkpoint_into_model,
    sha256_file,
    validate_checkpoint_artifact,
)


def load_script():
    name = "phase2n_week2_infer_pilots_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def infer_module():
    return load_script()


def write_text(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_bytes(path: Path, data: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_json(path: Path, payload: object) -> Path:
    return write_text(path, json.dumps(payload, indent=2) + "\n")


def make_glb(path: Path, values: tuple[float, ...] = (0.0, 1.0, 2.0)) -> Path:
    binary = struct.pack(f"<{len(values)}f", *values)
    while len(binary) % 4:
        binary += b"\x00"
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(values),
                "type": "SCALAR",
            }
        ],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    payload = (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )
    return write_bytes(path, payload)


def numel_map(names: list[str], total: int) -> dict[str, int]:
    assert names and total >= len(names)
    values = {name: 1 for name in names}
    values[names[0]] += total - len(names)
    return values


def make_manifest(
    checkpoint: Path,
    manifest: Path,
    *,
    scope: str,
    step: int,
    names: list[str],
    total_numel: int,
    dtypes: dict[str, str] | None = None,
) -> dict[str, object]:
    per_name = numel_map(names, total_numel)
    payload = {
        "format": "phase2n_week2_model_only_trainable_scope_v1",
        "scope": scope,
        "global_update": step,
        "checkpoint_path": str(checkpoint.resolve()),
        "byte_size": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
        "trainable_parameter_names": names,
        "trainable_parameter_tensor_count": len(names),
        "trainable_parameter_numel": total_numel,
        "dtype_by_name": dtypes or {name: "torch.float32" for name in names},
        "numel_by_name": per_name,
        "scope_report": {
            "scope": scope,
            "total_parameter_tensor_count": len(names) + 1,
            "total_parameter_numel": total_numel + 1,
            "trainable_parameter_tensor_count": len(names),
            "trainable_parameter_numel": total_numel,
            "frozen_parameter_tensor_count": 1,
            "frozen_parameter_numel": 1,
            "trainable_parameter_names": names,
            "frozen_parameter_names": ["frozen"],
        },
        "contains_optimizer_state": False,
        "contains_scheduler_state": False,
        "contains_frozen_parameters": False,
        "contains_full_model": False,
    }
    write_json(manifest, payload)
    return payload


def make_fake_project(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path / "project"
    root.mkdir()
    run = root / "outputs/phase2n/week2_pilot_training/slurm_264123"
    write_json(
        run / "summary.json",
        {
            "status": "OK",
            "scope_order": ["pc_s1", "pc_full"],
            "test_data_used": False,
        },
    )
    write_json(run / "00_RUNTIME_MANIFEST.json", {"status": "RUNNING"})
    write_text(run / "_SUCCESS", "PHASE2N_WEEK2_PILOT_TRAINING_OK\n")
    for scope, token in (
        ("pc_s1", "PHASE2N_WEEK2_PC_S1_TRAINING_OK"),
        ("pc_full", "PHASE2N_WEEK2_PC_FULL_TRAINING_OK"),
    ):
        write_json(run / scope / "summary.json", {"status": "OK", "scope": scope, "test_data_used": False})
        write_text(run / scope / "_SUCCESS", token + "\n")

    variants: dict[str, dict[str, object]] = {}
    artifacts: dict[str, Path] = {}
    expected = {
        "pc_s1_step160": ("pc_s1", 160, 80, 49_574_080),
        "pc_s1_step320": ("pc_s1", 320, 80, 49_574_080),
        "pc_full_step160": ("pc_full", 160, 981, 1_046_761_668),
        "pc_full_step320": ("pc_full", 320, 981, 1_046_761_668),
    }
    for variant_id, (scope, step, count, numel) in expected.items():
        checkpoint = write_bytes(run / scope / "checkpoints" / f"step_{step}_scope_state.pt", variant_id.encode())
        manifest = run / scope / "checkpoints" / f"step_{step}_manifest.json"
        names = [f"parameter_{index:04d}" for index in range(count)]
        make_manifest(checkpoint, manifest, scope=scope, step=step, names=names, total_numel=numel)
        artifacts[variant_id] = checkpoint
        variants[variant_id] = {
            "scope": scope,
            "step": step,
            "checkpoint_path": str(checkpoint.relative_to(root)),
            "checkpoint_manifest_path": str(manifest.relative_to(root)),
            "expected_trainable_tensor_count": count,
            "expected_trainable_numel": numel,
            "expected_checkpoint_sha256": sha256_file(checkpoint),
            "expected_checkpoint_byte_size": checkpoint.stat().st_size,
        }

    frozen_cases = []
    historical_cases = []
    for index in range(8):
        asset_id = f"ASSET_{index:02d}"
        split = "val" if index < 6 else "train_sanity"
        source_split = "val" if index < 6 else "train"
        mesh = write_bytes(root / "data/raw_assets" / f"{asset_id}.glb", b"mesh")
        reference = write_bytes(
            root / "data/hy3dpaint_train_examples/full101" / asset_id / "render_cond/005_light_AL.png",
            b"png",
        )
        frozen_cases.append(
            {
                "asset_id": asset_id,
                "source_split": source_split,
                "eval_split": split,
                "selection_stratum": "fixture",
                "selection_rationale": "fixture",
                "mesh_path": str(mesh.relative_to(root)),
                "reference_image_path": str(reference.relative_to(root)),
                "selected_input_view": "005",
                "reference_lighting": "AL",
            }
        )
        historical_cases.append(
            {
                "item_id": asset_id,
                "source_split": source_split,
                "eval_split": split,
                "selected_input_view": "005",
                "local_mesh_path": str(mesh.resolve()),
                "case_input_mesh": str(mesh.resolve()),
                "selected_input_image": str(reference.resolve()),
                "case_input_image": str(reference.resolve()),
            }
        )

    frozen_path = write_json(
        root / "configs/phase2n_week2_pilot_eval_cases.json",
        {
            "selection_frozen_before_training": True,
            "selected_input_view": "005",
            "reference_lighting": "AL",
            "case_count": 8,
            "split_counts": {"val": 6, "train_sanity": 2, "test": 0},
            "cases": frozen_cases,
        },
    )
    historical_path = write_json(
        root / "outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/eval_cases.json",
        {"cases": historical_cases},
    )
    full_checkpoint = write_bytes(root / "checkpoints/full80.ckpt", b"full80")
    base_root = root / "outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/infer/base"
    full_root = root / "outputs/phase2l/datav2_frame_panels/full80_eval_truepbr500/infer/finetuned"
    for frozen in frozen_cases:
        asset_id = frozen["asset_id"]
        split = frozen["eval_split"]
        mesh = (root / frozen["mesh_path"]).resolve()
        image = (root / frozen["reference_image_path"]).resolve()
        for mode, output_root, filename in (
            ("base", base_root, "base_textured_mesh.glb"),
            ("finetuned", full_root, "finetuned_textured_mesh.glb"),
        ):
            output = make_glb(output_root / split / asset_id / filename)
            write_json(
                output.parent / "run_plan.json",
                {
                    "mode": mode,
                    "max_num_view": 6,
                    "resolution": 512,
                    "use_remesh": False,
                    "dry_run": False,
                    "input_mesh": str(mesh),
                    "input_image": str(image),
                    "planned_output_glb": str(output.resolve()),
                    "checkpoint": "" if mode == "base" else str(full_checkpoint.resolve()),
                },
            )

    config = {
        "phase": "phase2n_week2_pilot_inference",
        "training_run_dir": str(run.relative_to(root)),
        "eval_cases_config": str(frozen_path.relative_to(root)),
        "output_root": "outputs/phase2n/week2_pilot_inference",
        "inference_seed": 0,
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "max_num_view": 6,
        "resolution": 512,
        "minimum_gpu_memory_gib": 70,
        "minimum_free_disk_gib": 20,
        "fixed_mesh": True,
        "use_remesh": False,
        "save_glb": True,
        "fresh_official_true_pbr_base_per_variant": True,
        "variant_order": list(expected),
        "variants": variants,
        "baseline_reuse": {
            "historical_eval_cases": str(historical_path.relative_to(root)),
            "corrected_input_base": {
                "root": str(base_root.relative_to(root)),
                "mode": "base",
                "output_filename": "base_textured_mesh.glb",
            },
            "historical_full80_500": {
                "root": str(full_root.relative_to(root)),
                "mode": "finetuned",
                "output_filename": "finetuned_textured_mesh.glb",
                "checkpoint_path": str(full_checkpoint.relative_to(root)),
            },
        },
    }
    config_path = write_json(root / "configs/phase2n_week2_pilot_inference.json", config)
    artifacts["frozen_config"] = frozen_path
    return root, config_path, artifacts


def test_exact_config_cases_baselines_and_stale_running_manifest_pass(infer_module, tmp_path):
    root, config_path, _ = make_fake_project(tmp_path)
    validated = infer_module.validate_config(config_path, root, check_free_space=False)

    assert tuple(validated.variants) == infer_module.EXPECTED_VARIANT_ORDER
    assert [case["eval_split"] for case in validated.cases].count("val") == 6
    assert [case["eval_split"] for case in validated.cases].count("train_sanity") == 2
    assert validated.training_report["runtime_manifest_status"] == "RUNNING"
    assert validated.training_report["runtime_manifest_status_is_completion_gate"] is False
    assert validated.baseline_reuse_report["complete_compatible_coverage"] is True
    assert validated.baseline_reuse_report["corrected_input_base"]["covered_case_count"] == 8
    assert validated.baseline_reuse_report["historical_full80_500"]["covered_case_count"] == 8


def test_check_only_imports_neither_torch_nor_hunyuan(infer_module, tmp_path, monkeypatch, capsys):
    root, config_path, _ = make_fake_project(tmp_path)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or "hunyuan" in name.lower() or name == "textureGenPipeline":
            raise AssertionError(f"check-only imported forbidden runtime module: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    validated = infer_module.run_check_only(config_path, root)
    output = capsys.readouterr().out
    assert len(validated.cases) == 8
    assert "PHASE2N_WEEK2_SCOPE_CHECKPOINTS_OK" in output
    assert "PHASE2N_WEEK2_FROZEN_CASES_OK" in output
    assert "PHASE2N_WEEK2_BASELINE_REUSE_OK" in output
    assert "PHASE2N_WEEK2_PILOT_INFERENCE_READINESS_OK" in output


def test_test_case_is_rejected(infer_module, tmp_path):
    root, config_path, artifacts = make_fake_project(tmp_path)
    frozen = json.loads(artifacts["frozen_config"].read_text(encoding="utf-8"))
    frozen["cases"][0]["eval_split"] = "test"
    frozen["cases"][0]["source_split"] = "test"
    frozen["split_counts"] = {"val": 5, "train_sanity": 2, "test": 1}
    write_json(artifacts["frozen_config"], frozen)
    with pytest.raises(ValueError, match="Test or unsupported split"):
        infer_module.validate_config(config_path, root, check_free_space=False)


def test_checkpoint_size_hash_scope_step_and_counts_are_strict(tmp_path):
    checkpoint = write_bytes(tmp_path / "state.pt", b"state")
    manifest_path = tmp_path / "manifest.json"
    manifest = make_manifest(
        checkpoint,
        manifest_path,
        scope="pc_s1",
        step=160,
        names=["a", "b"],
        total_numel=4,
    )
    kwargs = {
        "expected_scope": "pc_s1",
        "expected_step": 160,
        "expected_sha256": manifest["sha256"],
        "expected_byte_size": manifest["byte_size"],
        "expected_tensor_count": 2,
        "expected_numel": 4,
    }
    assert validate_checkpoint_artifact(checkpoint, manifest_path, **kwargs)["status"] == "OK"
    for key, value, match in (
        ("expected_sha256", "0" * 64, "sha256"),
        ("expected_byte_size", 99, "byte_size"),
        ("expected_scope", "pc_full", "scope"),
        ("expected_step", 320, "global_update"),
        ("expected_tensor_count", 3, "tensor_count"),
        ("expected_numel", 5, "numel"),
    ):
        changed = dict(kwargs)
        changed[key] = value
        with pytest.raises(ValueError, match=match):
            validate_checkpoint_artifact(checkpoint, manifest_path, **changed)


def test_duplicate_manifest_names_and_unsafe_payload_flags_are_rejected(tmp_path):
    checkpoint = write_bytes(tmp_path / "state.pt", b"state")
    manifest_path = tmp_path / "manifest.json"
    manifest = make_manifest(checkpoint, manifest_path, scope="pc_s1", step=160, names=["a", "b"], total_numel=2)
    manifest["trainable_parameter_names"] = ["a", "a"]
    manifest["scope_report"]["trainable_parameter_names"] = ["a", "a"]
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="duplicate"):
        validate_checkpoint_artifact(
            checkpoint,
            manifest_path,
            expected_scope="pc_s1",
            expected_step=160,
            expected_sha256=manifest["sha256"],
            expected_byte_size=manifest["byte_size"],
            expected_tensor_count=2,
            expected_numel=2,
        )

    manifest = make_manifest(checkpoint, manifest_path, scope="pc_s1", step=160, names=["a", "b"], total_numel=2)
    manifest["contains_optimizer_state"] = True
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="contains_optimizer_state"):
        validate_checkpoint_artifact(
            checkpoint,
            manifest_path,
            expected_scope="pc_s1",
            expected_step=160,
            expected_sha256=manifest["sha256"],
            expected_byte_size=manifest["byte_size"],
            expected_tensor_count=2,
            expected_numel=2,
        )


def tiny_scope_resolver(model, scope):
    names = ["selected"]
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in names
    return {
        "scope": scope,
        "trainable_parameter_names": names,
        "trainable_parameter_tensor_count": 1,
        "trainable_parameter_numel": 2,
    }


class FakeBoolResult:
    def __init__(self, value: bool):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value


class FakeTensor:
    def __init__(
        self,
        values,
        *,
        dtype: str = "torch.float32",
        device: str = "cpu",
        requires_grad: bool = False,
    ):
        self.values = [float(value) for value in values]
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.shape = (len(self.values),)

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def cpu(self):
        return FakeTensor(
            self.values,
            dtype=self.dtype,
            device="cpu",
            requires_grad=self.requires_grad,
        )

    def clone(self):
        return FakeTensor(
            self.values,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def to(self, *, device, dtype):
        return FakeTensor(self.values, dtype=str(dtype), device=str(device))

    def copy_(self, other):
        self.values = list(other.values)
        return self


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


class FakeTorch:
    def __init__(self):
        self.states = {}

    def register(self, path: Path, state):
        self.states[str(path.resolve())] = state

    def load(self, path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        return self.states[str(Path(path).resolve())]

    def no_grad(self):
        return FakeNoGrad()

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @staticmethod
    def isfinite(value):
        return FakeBoolResult(all(math.isfinite(item) for item in value.values))

    @staticmethod
    def equal(left, right):
        return (
            left.values == right.values
            and str(left.dtype) == str(right.dtype)
            and str(left.device) == str(right.device)
        )


class TinyModel:
    def __init__(self):
        self.selected = FakeTensor(
            [0.0, 0.0],
            dtype="torch.float64",
            device="cuda:0",
            requires_grad=False,
        )
        self.frozen = FakeTensor(
            [7.0],
            dtype="torch.float32",
            device="cuda:0",
            requires_grad=True,
        )
        self.training = True

    def named_parameters(self):
        return [("selected", self.selected), ("frozen", self.frozen)]

    def eval(self):
        self.training = False
        return self


def make_tiny_checkpoint(tmp_path: Path, fake_torch: FakeTorch, state: dict[str, object]):
    checkpoint = write_bytes(tmp_path / "tiny.pt", b"tiny-scope-state")
    fake_torch.register(checkpoint, state)
    manifest_path = tmp_path / "tiny_manifest.json"
    manifest = make_manifest(
        checkpoint,
        manifest_path,
        scope="pc_s1",
        step=160,
        names=["selected"],
        total_numel=2,
        dtypes={"selected": "torch.float32"},
    )
    return checkpoint, manifest_path, manifest


def load_tiny(model, fake_torch, checkpoint, manifest_path, manifest):
    return load_scope_checkpoint_into_model(
        model,
        checkpoint,
        manifest_path,
        expected_scope="pc_s1",
        expected_step=160,
        expected_sha256=manifest["sha256"],
        expected_byte_size=manifest["byte_size"],
        expected_tensor_count=1,
        expected_numel=2,
        torch_module=fake_torch,
        scope_resolver=tiny_scope_resolver,
    )


def test_scope_load_is_finite_dtype_safe_exact_and_preserves_frozen_sample(tmp_path):
    fake_torch = FakeTorch()
    model = TinyModel()
    original_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    frozen_before = model.frozen.detach().cpu().clone()
    checkpoint, manifest_path, manifest = make_tiny_checkpoint(
        tmp_path,
        fake_torch,
        {"selected": FakeTensor([1.25, -2.5], dtype="torch.float32", device="cpu")},
    )
    report = load_tiny(model, fake_torch, checkpoint, manifest_path, manifest)

    assert model.selected.dtype == "torch.float64"
    assert model.selected.device == "cuda:0"
    assert model.selected.values == [1.25, -2.5]
    assert FakeTorch.equal(model.frozen.detach().cpu(), frozen_before)
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == original_flags
    assert report["frozen_sample_unchanged"] is True
    assert report["dtype_device_safe_copy"] is True
    assert report["checkpoint_loaded_on_cpu"] is True
    assert report["loaded_target_equality_verified"] is True
    assert model.training is False


def test_pc_full_uses_audited_training_scope_not_inference_grad_defaults(tmp_path):
    fake_torch = FakeTorch()
    model = TinyModel()
    model.selected.requires_grad = True
    model.frozen.requires_grad = True
    original_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    checkpoint = write_bytes(tmp_path / "pc_full.pt", b"tiny-pc-full-state")
    fake_torch.register(
        checkpoint,
        {"selected": FakeTensor([3.0, 4.0], dtype="torch.float32", device="cpu")},
    )
    manifest_path = tmp_path / "pc_full_manifest.json"
    manifest = make_manifest(
        checkpoint,
        manifest_path,
        scope="pc_full",
        step=160,
        names=["selected"],
        total_numel=2,
    )

    def forbidden_resolver(_model, _scope):
        raise AssertionError("PC-Full must use the audited training scope report")

    report = load_scope_checkpoint_into_model(
        model,
        checkpoint,
        manifest_path,
        expected_scope="pc_full",
        expected_step=160,
        expected_sha256=manifest["sha256"],
        expected_byte_size=manifest["byte_size"],
        expected_tensor_count=1,
        expected_numel=2,
        torch_module=fake_torch,
        scope_resolver=forbidden_resolver,
    )

    assert model.selected.values == [3.0, 4.0]
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == original_flags
    assert report["scope_resolution_basis"] == "audited_official_training_scope_report"
    assert report["audited_live_keyspace_verified"] is True


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "exactly match"),
        ({"selected": "not-a-tensor"}, "not a tensor"),
    ],
)
def test_scope_load_rejects_missing_and_non_tensor_entries(tmp_path, state, message):
    fake_torch = FakeTorch()
    checkpoint, manifest_path, manifest = make_tiny_checkpoint(tmp_path, fake_torch, state)
    with pytest.raises((ValueError, TypeError), match=message):
        load_tiny(TinyModel(), fake_torch, checkpoint, manifest_path, manifest)


def test_scope_load_rejects_duplicate_payload_keys(tmp_path):
    class DuplicateState(dict):
        def items(self):
            item = next(iter(super().items()))
            return [item, item]

    fake_torch = FakeTorch()
    checkpoint, manifest_path, manifest = make_tiny_checkpoint(
        tmp_path,
        fake_torch,
        DuplicateState(selected=FakeTensor([0.0, 0.0])),
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        load_tiny(TinyModel(), fake_torch, checkpoint, manifest_path, manifest)


def test_scope_load_rejects_unexpected_and_nonfinite_entries(tmp_path):
    fake_torch = FakeTorch()
    checkpoint, manifest_path, manifest = make_tiny_checkpoint(
        tmp_path / "unexpected",
        fake_torch,
        {
            "selected": FakeTensor([0.0, 0.0]),
            "unexpected": FakeTensor([1.0]),
        },
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_tiny(TinyModel(), fake_torch, checkpoint, manifest_path, manifest)

    fake_torch = FakeTorch()
    checkpoint, manifest_path, manifest = make_tiny_checkpoint(
        tmp_path / "nonfinite",
        fake_torch,
        {"selected": FakeTensor([0.0, float("nan")])},
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_tiny(TinyModel(), fake_torch, checkpoint, manifest_path, manifest)


def test_fresh_pipeline_per_variant_prevents_cumulative_loading(infer_module):
    created = []

    class Target:
        def __init__(self):
            self.value = 0
            self.training = True

        def eval(self):
            self.training = False
            return self

    def factory(**_kwargs):
        target = Target()
        pipeline = SimpleNamespace(
            models={"multiview_model": SimpleNamespace(pipeline=SimpleNamespace(unet=target))}
        )
        created.append(target)
        return pipeline, {"fresh_official_true_pbr_base": True}

    def loader(target, _checkpoint, _manifest, **kwargs):
        target.value += kwargs["expected_step"]
        return {"status": "OK", "loaded": kwargs["expected_step"]}

    values = []
    for variant_id in infer_module.EXPECTED_VARIANT_ORDER:
        scope, step, count, numel = infer_module.EXPECTED_VARIANTS[variant_id]
        variant = {
            "scope": scope,
            "step": step,
            "checkpoint_path": "unused.pt",
            "checkpoint_manifest_path": "unused.json",
            "expected_checkpoint_sha256": "0" * 64,
            "expected_checkpoint_byte_size": 1,
            "expected_trainable_tensor_count": count,
            "expected_trainable_numel": numel,
        }
        pipeline, _, _ = infer_module.initialize_variant_pipeline(
            variant,
            max_num_view=6,
            resolution=512,
            device="cuda",
            pipeline_factory=factory,
            checkpoint_loader=loader,
        )
        values.append(infer_module._target_unet(pipeline).value)
    assert len({id(target) for target in created}) == 4
    assert values == [160, 320, 160, 320]


def materialize_completed_variant(infer_module, validated, variant_id: str) -> Path:
    run_dir = validated.output_root / "fixture_run"
    variant = validated.variants[variant_id]
    variant_dir = run_dir / variant_id
    manifest_paths = []
    for case in validated.cases:
        case_dir = variant_dir / case["asset_id"]
        glb = make_glb(case_dir / "textured_mesh.glb")
        expected = infer_module._case_manifest_expected(case, variant, glb.resolve(), validated)
        report = infer_module.validate_glb(glb)
        expected.update(
            {
                "status": "OK",
                "output_glb_size": report["byte_size"],
                "output_glb_sha256": report["sha256"],
                "output_glb_validation": report,
            }
        )
        manifest = write_json(case_dir / "inference_manifest.json", expected)
        manifest_paths.append(str(manifest.resolve()))
    summary = {
        "status": "OK",
        "variant": variant_id,
        "scope": variant["scope"],
        "checkpoint_step": variant["step"],
        "checkpoint_sha256": variant["expected_checkpoint_sha256"],
        "case_count": 8,
        "split_counts": {"val": 6, "train_sanity": 2, "test": 0},
        "fresh_official_true_pbr_base": True,
        "test_data_used": False,
        "case_manifest_paths": manifest_paths,
    }
    write_json(variant_dir / "summary.json", summary)
    write_text(variant_dir / "summary.md", infer_module._variant_summary_markdown(summary))
    write_text(variant_dir / "_SUCCESS", infer_module.VARIANT_SUCCESS_TOKENS[variant_id] + "\n")
    return variant_dir


def test_completed_variant_is_idempotent_and_incomplete_variant_fails_closed(infer_module, tmp_path):
    root, config_path, _ = make_fake_project(tmp_path)
    validated = infer_module.validate_config(config_path, root, check_free_space=False)
    variant_id = "pc_s1_step160"
    variant_dir = materialize_completed_variant(infer_module, validated, variant_id)
    first = infer_module.validate_completed_variant(
        variant_dir, validated.variants[variant_id], validated.cases, validated
    )
    second = infer_module.validate_completed_variant(
        variant_dir, validated.variants[variant_id], validated.cases, validated
    )
    assert first["summary"] == second["summary"]

    (variant_dir / validated.cases[0]["asset_id"] / "inference_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="inference manifest"):
        infer_module.validate_completed_variant(
            variant_dir, validated.variants[variant_id], validated.cases, validated
        )


def test_glb_nonfinite_accessor_is_rejected(infer_module, tmp_path):
    valid = make_glb(tmp_path / "valid.glb")
    assert infer_module.validate_glb(valid)["nonfinite_value_count"] == 0
    invalid = make_glb(tmp_path / "invalid.glb", (0.0, float("inf")))
    with pytest.raises(ValueError, match="NaN/Inf"):
        infer_module.validate_glb(invalid)
