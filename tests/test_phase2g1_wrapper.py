from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g1_wrapper_readiness import main as readiness_main  # noqa: E402
from inspect_phase2g1_checkpoint_metadata import main as metadata_main  # noqa: E402
from run_phase2g_paint_infer import main as infer_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_case(root: Path) -> Path:
    case_dir = root / "case"
    write_file(case_dir / "input" / "mesh.glb", b"mesh")
    write_file(case_dir / "input" / "image.png", b"image")
    return case_dir


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "demo.py", b"demo")
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "hunyuanpaintpbr" / "pipeline.py", b"paint pipeline")
    return hypaint


def test_checkpoint_metadata_script_writes_json_and_markdown() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint = write_file(root / "model.ckpt", b"checkpoint")
        out_json = root / "metadata.json"
        out_md = root / "metadata.md"

        assert (
            metadata_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )
            == 0
        )
        report = json.loads(out_json.read_text(encoding="utf-8"))
        assert report["exists"] is True
        assert report["size_bytes"] == len(b"checkpoint")
        assert out_md.read_text(encoding="utf-8").startswith("# Phase 2G.1")


def test_wrapper_dry_run_passes_for_base_without_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        out_dir = root / "out_base"

        assert (
            infer_main(
                [
                    "--case-dir",
                    str(case_dir),
                    "--output-dir",
                    str(out_dir),
                    "--mode",
                    "base",
                    "--dry-run",
                ]
            )
            == 0
        )
        plan = json.loads((out_dir / "run_plan.json").read_text(encoding="utf-8"))
        assert plan["mode"] == "base"
        assert plan["checkpoint"] == ""


def test_wrapper_dry_run_passes_for_finetuned_with_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        checkpoint = write_file(root / "model.ckpt", b"checkpoint")
        out_dir = root / "out_ft"

        assert (
            infer_main(
                [
                    "--case-dir",
                    str(case_dir),
                    "--output-dir",
                    str(out_dir),
                    "--mode",
                    "finetuned",
                    "--checkpoint",
                    str(checkpoint),
                    "--dry-run",
                ]
            )
            == 0
        )
        plan = json.loads((out_dir / "run_plan.json").read_text(encoding="utf-8"))
        assert plan["mode"] == "finetuned"
        assert plan["checkpoint"] == str(checkpoint.resolve())


def test_wrapper_dry_run_fails_for_finetuned_without_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        out_dir = root / "out_ft"

        assert (
            infer_main(
                [
                    "--case-dir",
                    str(case_dir),
                    "--output-dir",
                    str(out_dir),
                    "--mode",
                    "finetuned",
                    "--dry-run",
                ]
            )
            == 1
        )
        assert not (out_dir / "run_plan.json").exists()


def test_readiness_fails_if_wrapper_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        checkpoint = write_file(root / "model.ckpt", b"checkpoint")
        hypaint = make_hypaint(root)
        missing_wrapper = root / "missing_wrapper.py"

        assert (
            readiness_main(
                [
                    "--case-dir",
                    str(case_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--wrapper",
                    str(missing_wrapper),
                    "--hypaint",
                    str(hypaint),
                ]
            )
            == 1
        )
