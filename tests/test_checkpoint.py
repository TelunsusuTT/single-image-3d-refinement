from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hy3dft.checkpoint as checkpoint_module  # noqa: E402
from hy3dft.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT,
    load_scope_checkpoint_into_model,
    validate_checkpoint_artifact,
)


def make_artifact(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    checkpoint = tmp_path / "scope.pt"
    checkpoint.write_bytes(b"scope-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "scope": "multi_view_attention",
        "global_update": 160,
        "checkpoint_path": str(checkpoint.resolve()),
        "byte_size": checkpoint.stat().st_size,
        "sha256": digest,
        "trainable_parameter_tensor_count": 1,
        "trainable_parameter_numel": 4,
        "trainable_parameter_names": ["block.attn_multiview.to_q.weight"],
        "numel_by_name": {"block.attn_multiview.to_q.weight": 4},
        "dtype_by_name": {"block.attn_multiview.to_q.weight": "torch.float32"},
        "contains_optimizer_state": False,
        "contains_scheduler_state": False,
        "contains_frozen_parameters": False,
        "contains_full_model": False,
        "scope_report": {
            "scope": "multi_view_attention",
            "total_parameter_tensor_count": 2,
            "total_parameter_numel": 6,
            "trainable_parameter_tensor_count": 1,
            "trainable_parameter_numel": 4,
            "frozen_parameter_tensor_count": 1,
            "frozen_parameter_numel": 2,
            "trainable_parameter_names": ["block.attn_multiview.to_q.weight"],
            "frozen_parameter_names": ["block.attn_refview.to_q.weight"],
        },
    }
    manifest_path = tmp_path / "scope.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, manifest_path, manifest


def validate(checkpoint: Path, manifest_path: Path, manifest: dict[str, object]):
    return validate_checkpoint_artifact(
        checkpoint,
        manifest_path,
        expected_scope="multi_view_attention",
        expected_step=160,
        expected_sha256=str(manifest["sha256"]),
        expected_byte_size=int(manifest["byte_size"]),
        expected_tensor_count=1,
        expected_numel=4,
    )


def test_valid_scope_checkpoint_manifest(tmp_path: Path) -> None:
    checkpoint, manifest_path, manifest = make_artifact(tmp_path)

    report = validate(checkpoint, manifest_path, manifest)

    assert report["status"] == "OK"
    assert report["format"] == CHECKPOINT_FORMAT
    assert report["trainable_parameter_names"] == [
        "block.attn_multiview.to_q.weight"
    ]


def test_scope_checkpoint_allows_hash_verified_relocation(tmp_path: Path) -> None:
    checkpoint, manifest_path, manifest = make_artifact(tmp_path)
    recorded_path = str(checkpoint.resolve())
    relocated_dir = tmp_path / "relocated"
    relocated_dir.mkdir()
    relocated = checkpoint.rename(relocated_dir / checkpoint.name)

    report = validate(relocated, manifest_path, manifest)

    assert report["checkpoint_path"] == str(relocated.resolve())
    assert report["manifest_recorded_checkpoint_path"] == recorded_path


def test_scope_checkpoint_rejects_unsafe_payload_flag(tmp_path: Path) -> None:
    checkpoint, manifest_path, manifest = make_artifact(tmp_path)
    manifest["contains_optimizer_state"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="contains_optimizer_state=false"):
        validate(checkpoint, manifest_path, manifest)


def test_scope_checkpoint_rejects_duplicate_names(tmp_path: Path) -> None:
    checkpoint, manifest_path, manifest = make_artifact(tmp_path)
    name = str(manifest["trainable_parameter_names"][0])  # type: ignore[index]
    manifest["trainable_parameter_names"] = [name, name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        validate(checkpoint, manifest_path, manifest)


def test_scope_checkpoint_rejects_hash_mismatch(tmp_path: Path) -> None:
    checkpoint, manifest_path, manifest = make_artifact(tmp_path)
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        validate(checkpoint, manifest_path, manifest)


class FakeTruth:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> "FakeTruth":
        return self

    def item(self) -> bool:
        return self.value


class FakeTensor:
    def __init__(
        self,
        value: float,
        *,
        shape: tuple[int, ...] = (1,),
        finite: bool = True,
    ) -> None:
        self.value = value
        self.shape = shape
        self.finite = finite
        self.dtype = "torch.float32"
        self.device = SimpleNamespace(type="cpu")
        self.requires_grad = False
        self.copy_count = 0

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def clone(self) -> "FakeTensor":
        return FakeTensor(self.value, shape=self.shape, finite=self.finite)

    def to(self, *, device: object, dtype: object) -> "FakeTensor":
        converted = FakeTensor(self.value, shape=self.shape, finite=self.finite)
        converted.device = device
        converted.dtype = dtype
        return converted

    def copy_(self, source: "FakeTensor") -> None:
        self.value = source.value
        self.copy_count += 1


class FakeTorch:
    @staticmethod
    def no_grad():
        return contextlib.nullcontext()

    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, FakeTensor)

    @staticmethod
    def isfinite(value: FakeTensor) -> FakeTruth:
        return FakeTruth(value.finite)

    @staticmethod
    def equal(left: FakeTensor, right: FakeTensor) -> bool:
        return (left.value, left.shape, left.dtype) == (right.value, right.shape, right.dtype)


class FakeModel:
    def __init__(self) -> None:
        self.parameters = {
            "scope.first": FakeTensor(1.0),
            "scope.second": FakeTensor(2.0),
            "frozen.weight": FakeTensor(3.0),
        }
        self.training = True

    def named_parameters(self):
        return list(self.parameters.items())

    def eval(self) -> None:
        self.training = False


def install_loader_stubs(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, FakeTensor],
) -> None:
    names = ["scope.first", "scope.second"]
    static = {
        "checkpoint_path": "/relocated/scope.pt",
        "manifest_recorded_checkpoint_path": "/original/scope.pt",
        "manifest_path": "/relocated/scope.manifest.json",
        "sha256": "a" * 64,
        "byte_size": 10,
        "trainable_parameter_names": names,
        "numel_by_name": {name: 1 for name in names},
        "dtype_by_name": {name: "torch.float32" for name in names},
        "manifest": {"scope_report": {}},
    }
    scope_report = {
        "trainable_parameter_names": names,
        "trainable_parameter_tensor_count": 2,
        "trainable_parameter_numel": 2,
        "scope_resolution_basis": "test",
        "audited_live_keyspace_verified": True,
    }
    monkeypatch.setattr(
        checkpoint_module,
        "validate_checkpoint_artifact",
        lambda *args, **kwargs: static,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "derive_expected_scope",
        lambda *args, **kwargs: scope_report,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_load_cpu_state",
        lambda *args, **kwargs: state,
    )


def load_fake_scope(model: FakeModel) -> dict[str, object]:
    return load_scope_checkpoint_into_model(
        model,
        "scope.pt",
        "scope.manifest.json",
        expected_scope="multi_view_attention",
        expected_step=160,
        expected_sha256="a" * 64,
        expected_byte_size=10,
        expected_tensor_count=2,
        expected_numel=2,
        torch_module=FakeTorch,
        frozen_sample_count=1,
    )


def test_scope_loader_validates_all_tensors_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    state = {
        "scope.first": FakeTensor(10.0),
        "scope.second": FakeTensor(20.0, shape=(2,)),
    }
    install_loader_stubs(monkeypatch, state)

    with pytest.raises(ValueError, match="shape mismatch"):
        load_fake_scope(model)

    assert model.parameters["scope.first"].value == 1.0
    assert model.parameters["scope.second"].value == 2.0
    assert model.parameters["scope.first"].copy_count == 0
    assert model.parameters["scope.second"].copy_count == 0


def test_scope_loader_copies_only_after_complete_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    state = {
        "scope.first": FakeTensor(10.0),
        "scope.second": FakeTensor(20.0),
    }
    install_loader_stubs(monkeypatch, state)

    report = load_fake_scope(model)

    assert model.parameters["scope.first"].value == 10.0
    assert model.parameters["scope.second"].value == 20.0
    assert model.parameters["frozen.weight"].value == 3.0
    assert report["loaded_parameter_tensor_count"] == 2
    assert report["manifest_recorded_checkpoint_path"] == "/original/scope.pt"
    assert report["frozen_sample_unchanged"] is True
    assert model.training is False
