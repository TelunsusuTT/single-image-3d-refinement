from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g2_base_infer_readiness import main as readiness_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_case(root: Path) -> Path:
    case_dir = root / "case"
    write_file(case_dir / "input" / "mesh.glb", b"mesh")
    write_file(case_dir / "input" / "image.png", b"image")
    return case_dir


def make_hypaint(root: Path, include_realesrgan: bool = True) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "demo.py", b"demo")
    write_file(hypaint / "textureGenPipeline.py", b"texture")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    if include_realesrgan:
        write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def run_readiness(case_dir: Path, wrapper: Path, hypaint: Path, output_dir: Path) -> int:
    return readiness_main(
        [
            "--case-dir",
            str(case_dir),
            "--wrapper",
            str(wrapper),
            "--hypaint",
            str(hypaint),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_readiness_passes_when_all_fake_files_exist() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        hypaint = make_hypaint(root)
        wrapper = write_file(root / "run_wrapper.py", b"print('wrapper')")
        output_dir = root / "outputs" / "base"

        assert run_readiness(case_dir, wrapper, hypaint, output_dir) == 0
        assert output_dir.parent.is_dir()


def test_readiness_fails_when_realesrgan_checkpoint_is_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        hypaint = make_hypaint(root, include_realesrgan=False)
        wrapper = write_file(root / "run_wrapper.py", b"print('wrapper')")
        output_dir = root / "outputs" / "base"

        assert run_readiness(case_dir, wrapper, hypaint, output_dir) == 1


def test_readiness_fails_when_output_dir_already_contains_glb() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        hypaint = make_hypaint(root)
        wrapper = write_file(root / "run_wrapper.py", b"print('wrapper')")
        output_dir = root / "outputs" / "base"
        write_file(output_dir / "old.glb", b"old")

        assert run_readiness(case_dir, wrapper, hypaint, output_dir) == 1


def test_readiness_fails_when_wrapper_is_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir = make_case(root)
        hypaint = make_hypaint(root)
        wrapper = root / "missing_wrapper.py"
        output_dir = root / "outputs" / "base"

        assert run_readiness(case_dir, wrapper, hypaint, output_dir) == 1
