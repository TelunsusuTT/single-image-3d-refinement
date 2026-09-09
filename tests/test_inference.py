from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_paint_inference as backend  # noqa: E402
from hy3dft.inference import (  # noqa: E402
    resolve_official_paths,
    run_fixed_mesh_inference,
)


def write_file(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_case(root: Path) -> Path:
    case_dir = root / "case"
    write_file(case_dir / "input" / "mesh.glb", b"mesh")
    write_file(case_dir / "input" / "image.png", b"image")
    return case_dir


def test_baseline_dry_run_records_fixed_mesh_protocol(tmp_path: Path) -> None:
    case_dir = make_case(tmp_path)
    output_dir = tmp_path / "baseline"

    assert backend.main(
        [
            "--case-dir",
            str(case_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            "baseline",
            "--seed",
            "17",
            "--no-remesh",
            "--dry-run",
        ]
    ) == 0

    plan = json.loads((output_dir / "run_plan.json").read_text(encoding="utf-8"))
    assert plan["mode"] == "baseline"
    assert plan["checkpoint"] == ""
    assert plan["conditioning_view"] == "005"
    assert plan["seed"] == 17
    assert plan["use_remesh"] is False
    assert plan["planned_output_mesh"].endswith("textured_mesh.obj")
    assert plan["planned_output_glb"].endswith("textured_mesh.glb")


def test_dry_run_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    case_dir = make_case(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    existing_glb = write_file(output_dir / "textured_mesh.glb", b"old-glb")

    assert backend.main(
        [
            "--case-dir",
            str(case_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            "baseline",
            "--dry-run",
        ]
    ) == 1
    assert existing_glb.read_bytes() == b"old-glb"
    assert not (output_dir / "run_plan.json").exists()


def test_checkpoint_dry_run_requires_existing_checkpoint(tmp_path: Path) -> None:
    case_dir = make_case(tmp_path)
    checkpoint = write_file(tmp_path / "model.ckpt", b"checkpoint")
    output_dir = tmp_path / "checkpoint"

    assert backend.main(
        [
            "--case-dir",
            str(case_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            "checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
        ]
    ) == 0
    plan = json.loads((output_dir / "run_plan.json").read_text(encoding="utf-8"))
    assert plan["checkpoint"] == str(checkpoint.resolve())


def test_checkpoint_dry_run_fails_closed_without_checkpoint(tmp_path: Path) -> None:
    case_dir = make_case(tmp_path)
    output_dir = tmp_path / "missing"

    assert backend.main(
        [
            "--case-dir",
            str(case_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            "checkpoint",
            "--dry-run",
        ]
    ) == 1
    assert not (output_dir / "run_plan.json").exists()


def test_official_paths_use_default_and_optional_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hunyuan_root = tmp_path / "Hunyuan3D-2.1"
    default_paint = hunyuan_root / "hy3dpaint"
    override_paint = tmp_path / "paint-source"
    default_paint.mkdir(parents=True)
    override_paint.mkdir()

    monkeypatch.setenv("HUNYUAN3D_ROOT", str(hunyuan_root))
    monkeypatch.delenv("HUNYUAN3D_PAINT_SOURCE_ROOT", raising=False)
    assert resolve_official_paths() == (hunyuan_root.resolve(), default_paint.resolve())

    monkeypatch.setenv("HUNYUAN3D_PAINT_SOURCE_ROOT", str(override_paint))
    assert resolve_official_paths() == (hunyuan_root.resolve(), override_paint.resolve())


def test_official_paths_require_explicit_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUNYUAN3D_ROOT", raising=False)
    monkeypatch.delenv("HUNYUAN3D_PAINT_SOURCE_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="HUNYUAN3D_ROOT"):
        resolve_official_paths()


def test_checkpoint_key_and_shape_validation_is_strict() -> None:
    tensor = SimpleNamespace(shape=(2, 3))
    state = {"unet.layer.weight": tensor}
    transformed = backend.transformed_state_dict(state, "unet.")

    backend.verify_state_compatibility(
        transformed,
        {"layer.weight": SimpleNamespace(shape=(2, 3))},
    )
    with pytest.raises(RuntimeError, match="key set"):
        backend.verify_state_compatibility(transformed, {"other.weight": tensor})
    with pytest.raises(RuntimeError, match="shape"):
        backend.verify_state_compatibility(
            transformed,
            {"layer.weight": SimpleNamespace(shape=(3, 2))},
        )


def test_checkpoint_loader_requests_weights_only() -> None:
    calls: list[dict[str, object]] = []

    class TorchStub:
        @staticmethod
        def load(path: Path, **kwargs: object) -> dict[str, object]:
            calls.append({"path": path, **kwargs})
            return {"state_dict": {}}

    checkpoint = Path("checkpoint.ckpt")
    assert backend.torch_load_cpu(TorchStub, checkpoint) == {"state_dict": {}}
    assert calls == [
        {
            "path": checkpoint,
            "map_location": "cpu",
            "weights_only": True,
        }
    ]


def test_fixed_mesh_helper_passes_no_remesh() -> None:
    calls: list[dict[str, object]] = []

    def pipeline(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return "result"

    result = run_fixed_mesh_inference(
        pipeline,
        mesh_path="mesh.glb",
        image_path="image.png",
        output_mesh_path="textured_mesh.obj",
    )

    assert result == "result"
    assert calls[0]["use_remesh"] is False
    assert calls[0]["save_glb"] is True


def test_seed_runtime_sets_python_numpy_and_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    calls: list[tuple[str, int]] = []
    numpy_stub = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: calls.append(("numpy", value)))
    )
    cuda_stub = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda value: calls.append(("torch_cuda", value)),
    )
    torch_stub = SimpleNamespace(
        manual_seed=lambda value: calls.append(("torch_cpu", value)),
        cuda=cuda_stub,
    )
    monkeypatch.setattr(random, "seed", lambda value: calls.append(("python", value)))
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    report = backend.seed_runtime(17)

    assert calls == [
        ("python", 17),
        ("numpy", 17),
        ("torch_cpu", 17),
        ("torch_cuda", 17),
    ]
    assert report == {
        "seed": 17,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": True,
    }
