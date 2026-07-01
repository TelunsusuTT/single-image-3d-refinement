from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_phase2h1_eval_readiness as readiness  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_case(root: Path) -> Path:
    case_dir = root / "outputs" / "phase2g" / "infer_cases" / "B075YLTF7Q"
    write_file(case_dir / "input" / "mesh.glb", b"mesh")
    write_file(case_dir / "input" / "image.png", b"image")
    return case_dir


def make_base_output(root: Path) -> Path:
    base_dir = root / "outputs" / "phase2g" / "infer_runs" / "B075YLTF7Q" / "base_a100_noremesh_smoke"
    for filename in (
        "base_textured_mesh.obj",
        "base_textured_mesh.glb",
        "base_textured_mesh.jpg",
        "base_textured_mesh_metallic.jpg",
        "base_textured_mesh_roughness.jpg",
    ):
        write_file(base_dir / filename, filename.encode("utf-8"))
    return base_dir


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def make_fake_project_scripts(project_root: Path, include_wrapper: bool = True) -> Path:
    scripts_dir = project_root / "scripts"
    write_file(scripts_dir / "load_phase2g4_finetuned_checkpoint_only.py", b"script")
    write_file(scripts_dir / "make_phase2g6_texture_comparison.py", b"script")
    if include_wrapper:
        write_file(scripts_dir / "run_phase2g_paint_infer.py", b"script")
    return project_root


def make_paths(root: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    case_dir = make_case(root)
    base_dir = make_base_output(root)
    checkpoint = write_file(
        root
        / "checkpoints"
        / "pilot_v1_conservative_50_lr1e6"
        / "pilot_v1_conservative_50_lr1e6-stepstep=50.ckpt",
        b"checkpoint",
    )
    hypaint = make_hypaint(root)
    load_dir = root / "outputs" / "phase2h" / "load_only" / "pilot_v1_conservative_50_lr1e6"
    infer_dir = (
        root
        / "outputs"
        / "phase2h"
        / "infer_runs"
        / "B075YLTF7Q"
        / "pilot_v1_conservative_50_lr1e6_noremesh"
    )
    compare_dir = (
        root
        / "outputs"
        / "phase2h"
        / "compare"
        / "B075YLTF7Q"
        / "pilot_v1_conservative_50_lr1e6"
    )
    return case_dir, base_dir, checkpoint, hypaint, load_dir, infer_dir, compare_dir


def run_readiness(
    case_dir: Path,
    base_dir: Path,
    checkpoint: Path,
    hypaint: Path,
    load_dir: Path,
    infer_dir: Path,
    compare_dir: Path,
) -> int:
    return readiness.main(
        [
            "--case-dir",
            str(case_dir),
            "--base-dir",
            str(base_dir),
            "--checkpoint",
            str(checkpoint),
            "--hypaint",
            str(hypaint),
            "--load-output-dir",
            str(load_dir),
            "--infer-output-dir",
            str(infer_dir),
            "--compare-output-dir",
            str(compare_dir),
        ]
    )


def test_readiness_passes_with_valid_fake_setup(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))

        assert run_readiness(*paths) == 0
        assert paths[4].parent.is_dir()
        assert paths[5].parent.is_dir()
        assert paths[6].parent.is_dir()


def test_readiness_fails_if_checkpoint_contains_collapsed_token(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, base_dir, _, hypaint, load_dir, infer_dir, compare_dir = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        checkpoint = write_file(
            root
            / "checkpoints"
            / "pilot_v1_overfit_500"
            / "pilot_v1_conservative_50_lr1e6-stepstep=50.ckpt",
            b"checkpoint",
        )

        assert run_readiness(case_dir, base_dir, checkpoint, hypaint, load_dir, infer_dir, compare_dir) == 1


def test_readiness_fails_if_base_albedo_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, base_dir, checkpoint, hypaint, load_dir, infer_dir, compare_dir = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        (base_dir / "base_textured_mesh.jpg").unlink()

        assert run_readiness(case_dir, base_dir, checkpoint, hypaint, load_dir, infer_dir, compare_dir) == 1


def test_readiness_fails_if_infer_output_dir_already_contains_glb(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, base_dir, checkpoint, hypaint, load_dir, infer_dir, compare_dir = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        write_file(infer_dir / "old.glb", b"old")

        assert run_readiness(case_dir, base_dir, checkpoint, hypaint, load_dir, infer_dir, compare_dir) == 1


def test_readiness_fails_if_run_wrapper_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = make_paths(root)
        monkeypatch.setattr(
            readiness,
            "PROJECT_ROOT",
            make_fake_project_scripts(root / "project", include_wrapper=False),
        )

        assert run_readiness(*paths) == 1
