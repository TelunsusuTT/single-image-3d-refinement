from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g7_target_diagnostic_readiness import main as readiness_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "sample_name", "sample_dir"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_layout(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    dataset_root = root / "pilot_v1"
    sample_dir = dataset_root / "SAMPLE_A"
    render_tex = sample_dir / "render_tex"
    write_file(render_tex / "000_mr.png", b"mr")
    write_file(render_tex / "000_albedo.png", b"albedo")
    manifest = write_manifest(
        root / "manifest.csv",
        [{"source_id": "SAMPLE_A", "sample_name": "SAMPLE_A", "sample_dir": str(sample_dir)}],
    )
    base_dir = root / "base"
    finetuned_dir = root / "finetuned"
    for filename in (
        "base_textured_mesh.jpg",
        "base_textured_mesh_metallic.jpg",
        "base_textured_mesh_roughness.jpg",
    ):
        write_file(base_dir / filename, b"base")
    for filename in (
        "finetuned_textured_mesh.jpg",
        "finetuned_textured_mesh_metallic.jpg",
        "finetuned_textured_mesh_roughness.jpg",
    ):
        write_file(finetuned_dir / filename, b"finetuned")
    output_dir = root / "outputs" / "phase2g7"
    return dataset_root, manifest, base_dir, finetuned_dir, output_dir


def run_readiness(
    dataset_root: Path,
    manifest: Path,
    base_dir: Path,
    finetuned_dir: Path,
    output_dir: Path,
) -> int:
    return readiness_main(
        [
            "--dataset-root",
            str(dataset_root),
            "--manifest-csv",
            str(manifest),
            "--base-dir",
            str(base_dir),
            "--finetuned-dir",
            str(finetuned_dir),
            "--output-dir",
            str(output_dir),
        ]
    )


def test_readiness_passes_with_fake_dataset_and_maps() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        assert run_readiness(*make_layout(Path(tmpdir))) == 0


def test_readiness_fails_if_manifest_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root, manifest, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        assert run_readiness(dataset_root, manifest.parent / "missing.csv", base_dir, finetuned_dir, output_dir) == 1


def test_readiness_fails_if_sample_has_no_mr() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root, manifest, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        for path in dataset_root.glob("*/render_tex/*_mr.png"):
            path.unlink()
        assert run_readiness(dataset_root, manifest, base_dir, finetuned_dir, output_dir) == 1


def test_readiness_fails_if_sample_has_no_albedo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root, manifest, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        for path in dataset_root.glob("*/render_tex/*_albedo.png"):
            path.unlink()
        assert run_readiness(dataset_root, manifest, base_dir, finetuned_dir, output_dir) == 1


def test_readiness_fails_if_finetuned_metallic_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root, manifest, base_dir, finetuned_dir, output_dir = make_layout(Path(tmpdir))
        (finetuned_dir / "finetuned_textured_mesh_metallic.jpg").unlink()
        assert run_readiness(dataset_root, manifest, base_dir, finetuned_dir, output_dir) == 1
