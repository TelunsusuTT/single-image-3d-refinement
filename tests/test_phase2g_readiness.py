from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g_readiness import main as readiness_main  # noqa: E402
from prepare_phase2g_infer_case import main as prepare_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_hypaint(root: Path, include_demo: bool = True) -> Path:
    hypaint = root / "hy3dpaint"
    hypaint.mkdir()
    if include_demo:
        write_file(hypaint / "demo.py", b"print('demo')\n")
    write_file(hypaint / "train.py", b"print('train')\n")
    return hypaint


def prepare_fake_case(root: Path) -> tuple[Path, Path, Path, Path]:
    mesh = write_file(root / "source" / "mesh.glb", b"glb")
    image = write_file(root / "source" / "image.png", b"png")
    checkpoint = write_file(root / "checkpoint.ckpt", b"ckpt")
    case_dir = root / "case"
    assert (
        prepare_main(
            [
                "--asset-id",
                "asset_1",
                "--mesh",
                str(mesh),
                "--reference-image",
                str(image),
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(case_dir),
            ]
        )
        == 0
    )
    return case_dir, checkpoint, mesh, image


def test_prepare_case_writes_manifest_and_links_or_copies_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, mesh, image = prepare_fake_case(root)
        input_mesh = case_dir / "input" / "mesh.glb"
        input_image = case_dir / "input" / "image.png"
        manifest_path = case_dir / "case_manifest.json"

        assert input_mesh.exists()
        assert input_image.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["asset_id"] == "asset_1"
        assert manifest["source_mesh_path"] == str(mesh.resolve())
        assert manifest["source_reference_path"] == str(image.resolve())
        assert manifest["checkpoint_path"] == str(checkpoint.resolve())
        assert manifest["input_mesh_path"] == str(input_mesh)
        assert manifest["input_image_path"] == str(input_image)


def test_phase2g_readiness_passes_for_valid_fake_setup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, _, _ = prepare_fake_case(root)
        hypaint = make_hypaint(root)

        assert (
            readiness_main(
                [
                    "--case-dir",
                    str(case_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--hypaint",
                    str(hypaint),
                ]
            )
            == 0
        )


def test_phase2g_readiness_fails_for_missing_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, _, _ = prepare_fake_case(root)
        checkpoint.unlink()
        hypaint = make_hypaint(root)

        assert (
            readiness_main(
                [
                    "--case-dir",
                    str(case_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--hypaint",
                    str(hypaint),
                ]
            )
            == 1
        )


def test_phase2g_readiness_fails_for_missing_demo_py() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        case_dir, checkpoint, _, _ = prepare_fake_case(root)
        hypaint = make_hypaint(root, include_demo=False)

        assert (
            readiness_main(
                [
                    "--case-dir",
                    str(case_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--hypaint",
                    str(hypaint),
                ]
            )
            == 1
        )
