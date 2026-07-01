from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g5_finetuned_infer_readiness import main as readiness_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_case(root: Path) -> Path:
    case_dir = root / "outputs" / "phase2g" / "infer_cases" / "B075YLTF7Q"
    write_file(case_dir / "input" / "mesh.glb", b"mesh")
    write_file(case_dir / "input" / "image.png", b"image")
    return case_dir


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def keyspace_text() -> str:
    return "\n".join(
        [
            "# Phase 2G.3 Keyspace Compare",
            "",
            "## Recommendation",
            "- recommendation status: `RECOMMENDED`",
            "- recommended target path: `paint_pipeline.models['multiview_model'].pipeline.unet`",
            "- recommended transform: `strip:unet.`",
        ]
    )


def load_only_text(strict_ok: bool = True) -> str:
    strict = "strict_load_state_dict: `OK`" if strict_ok else "strict_load_state_dict: `FAILED`"
    return "\n".join(
        [
            "# Phase 2G.4 Checkpoint Load-Only Report",
            strict,
            "PHASE2G4_CHECKPOINT_LOAD_ONLY_OK",
        ]
    )


def make_paths(root: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    case_dir = make_case(root)
    checkpoint = write_file(root / "checkpoints" / "pilot_v1_overfit_500" / "model.ckpt", b"checkpoint")
    wrapper = write_file(root / "scripts" / "run_phase2g_paint_infer.py", b"wrapper")
    hypaint = make_hypaint(root)
    output_dir = root / "outputs" / "phase2g" / "infer_runs" / "B075YLTF7Q" / "finetuned_a100_noremesh_smoke"
    keyspace = root / "outputs" / "phase2g" / "key_inspection" / "pilot_v1_overfit_500" / "keyspace_compare_v2.md"
    load_only = root / "outputs" / "phase2g" / "load_only" / "pilot_v1_overfit_500" / "load_only_report.md"
    return case_dir, checkpoint, wrapper, hypaint, output_dir, keyspace, load_only


def run_readiness(case_dir: Path, checkpoint: Path, wrapper: Path, hypaint: Path, output_dir: Path) -> int:
    return readiness_main(
        [
            "--case-dir",
            str(case_dir),
            "--checkpoint",
            str(checkpoint),
            "--wrapper",
            str(wrapper),
            "--hypaint",
            str(hypaint),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_readiness_passes_with_good_reports() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, wrapper, hypaint, output_dir, keyspace, load_only = make_paths(root)
        write_text(keyspace, keyspace_text())
        write_text(load_only, load_only_text())

        assert run_readiness(case_dir, checkpoint, wrapper, hypaint, output_dir) == 0
        assert output_dir.parent.is_dir()


def test_readiness_fails_if_load_only_report_lacks_strict_ok() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, wrapper, hypaint, output_dir, keyspace, load_only = make_paths(root)
        write_text(keyspace, keyspace_text())
        write_text(load_only, load_only_text(strict_ok=False))

        assert run_readiness(case_dir, checkpoint, wrapper, hypaint, output_dir) == 1


def test_readiness_fails_if_keyspace_recommendation_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, wrapper, hypaint, output_dir, keyspace, load_only = make_paths(root)
        write_text(keyspace, "# no recommendation here\n")
        write_text(load_only, load_only_text())

        assert run_readiness(case_dir, checkpoint, wrapper, hypaint, output_dir) == 1


def test_readiness_fails_if_output_dir_already_contains_glb() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, wrapper, hypaint, output_dir, keyspace, load_only = make_paths(root)
        write_text(keyspace, keyspace_text())
        write_text(load_only, load_only_text())
        write_file(output_dir / "old.glb", b"old")

        assert run_readiness(case_dir, checkpoint, wrapper, hypaint, output_dir) == 1


def test_readiness_fails_if_checkpoint_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, wrapper, hypaint, output_dir, keyspace, load_only = make_paths(root)
        write_text(keyspace, keyspace_text())
        write_text(load_only, load_only_text())

        assert run_readiness(case_dir, checkpoint.parent / "missing.ckpt", wrapper, hypaint, output_dir) == 1
