from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g6_compare_readiness import main as readiness_main  # noqa: E402
from make_phase2g6_texture_comparison import normalize_extrema  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_layout(root: Path) -> tuple[Path, Path, Path, Path]:
    case_dir = root / "case"
    base_dir = root / "base"
    finetuned_dir = root / "finetuned"
    output_dir = root / "compare" / "B075YLTF7Q"
    write_file(case_dir / "input" / "image.png", b"image")
    for filename in (
        "base_textured_mesh.obj",
        "base_textured_mesh.glb",
        "base_textured_mesh.jpg",
        "base_textured_mesh_metallic.jpg",
        "base_textured_mesh_roughness.jpg",
    ):
        write_file(base_dir / filename, b"base")
    for filename in (
        "finetuned_textured_mesh.obj",
        "finetuned_textured_mesh.glb",
        "finetuned_textured_mesh.jpg",
        "finetuned_textured_mesh_metallic.jpg",
        "finetuned_textured_mesh_roughness.jpg",
    ):
        write_file(finetuned_dir / filename, b"finetuned")
    return case_dir, base_dir, finetuned_dir, output_dir


def run_readiness(case_dir: Path, base_dir: Path, finetuned_dir: Path, output_dir: Path) -> int:
    return readiness_main(
        [
            "--case-dir",
            str(case_dir),
            "--base-dir",
            str(base_dir),
            "--finetuned-dir",
            str(finetuned_dir),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_readiness_passes_with_all_required_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = make_layout(Path(tmpdir))
        assert run_readiness(*paths) == 0


def test_readiness_fails_if_base_glb_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        case_dir, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        (base_dir / "base_textured_mesh.glb").unlink()
        assert run_readiness(case_dir, base_dir, finetuned_dir, output_dir) == 1


def test_readiness_fails_if_finetuned_albedo_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        case_dir, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        (finetuned_dir / "finetuned_textured_mesh.jpg").unlink()
        assert run_readiness(case_dir, base_dir, finetuned_dir, output_dir) == 1


def test_readiness_fails_if_finetuned_metallic_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        case_dir, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        (finetuned_dir / "finetuned_textured_mesh_metallic.jpg").unlink()
        assert run_readiness(case_dir, base_dir, finetuned_dir, output_dir) == 1


def test_readiness_fails_if_reference_image_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        case_dir, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        (case_dir / "input" / "image.png").unlink()
        assert run_readiness(case_dir, base_dir, finetuned_dir, output_dir) == 1

def test_normalize_extrema_handles_grayscale_tuple() -> None:
    assert normalize_extrema((3, 250)) == [(3, 250)]


def test_normalize_extrema_handles_multichannel_tuple() -> None:
    assert normalize_extrema(((1, 2), (3, 4), (5, 6))) == [(1, 2), (3, 4), (5, 6)]
