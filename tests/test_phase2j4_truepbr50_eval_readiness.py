from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_phase2j4_truepbr50_eval_readiness as readiness  # noqa: E402


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


def make_fake_project_scripts(project_root: Path) -> Path:
    scripts_dir = project_root / "scripts"
    for filename in (
        "load_phase2g4_finetuned_checkpoint_only.py",
        "compare_phase2i_base_unet_to_checkpoints.py",
        "run_phase2g_paint_infer.py",
        "make_phase2g6_texture_comparison.py",
    ):
        write_file(scripts_dir / filename, b"script")
    return project_root


def make_paths(root: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    case_dir = make_case(root)
    base_dir = make_base_output(root)
    checkpoint = write_file(
        root
        / "checkpoints"
        / "pilot_v1_truepbr_50_lr1e6"
        / "pilot_v1_truepbr_50_lr1e6-stepstep=50.ckpt",
        b"checkpoint",
    )
    hypaint = make_hypaint(root)
    load_dir = root / "outputs" / "phase2j" / "eval_truepbr50" / "load_only"
    delta_dir = root / "outputs" / "phase2j" / "eval_truepbr50" / "delta_audit"
    infer_dir = root / "outputs" / "phase2j" / "eval_truepbr50" / "infer" / "B075YLTF7Q"
    compare_dir = root / "outputs" / "phase2j" / "eval_truepbr50" / "compare" / "B075YLTF7Q"
    return case_dir, base_dir, checkpoint, hypaint, load_dir, delta_dir, infer_dir, compare_dir


def run_readiness(
    case_dir: Path,
    base_dir: Path,
    checkpoint: Path,
    hypaint: Path,
    load_dir: Path,
    delta_dir: Path,
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
            "--delta-output-dir",
            str(delta_dir),
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
        assert paths[7].parent.is_dir()


def test_readiness_fails_if_checkpoint_contains_overfit_token(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, base_dir, _, hypaint, load_dir, delta_dir, infer_dir, compare_dir = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        checkpoint = write_file(
            root
            / "checkpoints"
            / "pilot_v1_truepbr_50_lr1e6"
            / "pilot_v1_overfit_500"
            / "pilot_v1_truepbr_50_lr1e6-stepstep=50.ckpt",
            b"checkpoint",
        )

        assert run_readiness(
            case_dir, base_dir, checkpoint, hypaint, load_dir, delta_dir, infer_dir, compare_dir
        ) == 1


def test_readiness_fails_if_checkpoint_contains_conservative_token(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, base_dir, _, hypaint, load_dir, delta_dir, infer_dir, compare_dir = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        checkpoint = write_file(
            root
            / "checkpoints"
            / "pilot_v1_truepbr_50_lr1e6"
            / "pilot_v1_conservative_50_lr1e6"
            / "pilot_v1_truepbr_50_lr1e6-stepstep=50.ckpt",
            b"checkpoint",
        )

        assert run_readiness(
            case_dir, base_dir, checkpoint, hypaint, load_dir, delta_dir, infer_dir, compare_dir
        ) == 1


def test_readiness_fails_if_base_roughness_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, base_dir, checkpoint, hypaint, load_dir, delta_dir, infer_dir, compare_dir = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        (base_dir / "base_textured_mesh_roughness.jpg").unlink()

        assert run_readiness(
            case_dir, base_dir, checkpoint, hypaint, load_dir, delta_dir, infer_dir, compare_dir
        ) == 1


def test_readiness_fails_if_infer_output_dir_already_contains_glb(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, base_dir, checkpoint, hypaint, load_dir, delta_dir, infer_dir, compare_dir = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        write_file(infer_dir / "old.glb", b"old")

        assert run_readiness(
            case_dir, base_dir, checkpoint, hypaint, load_dir, delta_dir, infer_dir, compare_dir
        ) == 1
