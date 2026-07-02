from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2j0_readiness import main as readiness_main  # noqa: E402
from locate_phase2j_official_pbr_weights import main as locate_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_hypaint(root: Path, include_train: bool = True) -> Path:
    hypaint = root / "hy3dpaint"
    if include_train:
        write_file(hypaint / "train.py", b"train")
    write_file(hypaint / "hunyuanpaintpbr" / "pipeline.py", b"pipeline")
    write_file(hypaint / "hunyuanpaintpbr" / "unet" / "model.py", b"model")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"cfg")
    return hypaint


def run_readiness(hypaint: Path, search_root: Path, output_dir: Path) -> int:
    return readiness_main(
        [
            "--hypaint",
            str(hypaint),
            "--search-root",
            str(search_root),
            "--output-dir",
            str(output_dir),
        ]
    )


def run_locator(search_root: Path, out_json: Path, out_md: Path) -> int:
    return locate_main(
        [
            "--search-root",
            str(search_root),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )


def test_readiness_passes_with_fake_hunyuan_files_and_search_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        search_root = root / "model_cache"
        search_root.mkdir()
        output_dir = root / "outputs" / "phase2j"

        assert run_readiness(hypaint, search_root, output_dir) == 0
        assert output_dir.parent.is_dir()


def test_readiness_fails_if_train_py_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root, include_train=False)
        search_root = root / "model_cache"
        search_root.mkdir()
        output_dir = root / "outputs" / "phase2j"

        assert run_readiness(hypaint, search_root, output_dir) == 1


def test_locator_detects_fake_pbr_directory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pbr_dir = root / "models" / "hunyuan3d-paintpbr-v2-1"
        write_file(pbr_dir / "model_index.json", b"{}")
        (pbr_dir / "unet").mkdir(parents=True)
        out_json = root / "locator.json"
        out_md = root / "locator.md"

        assert run_locator(root, out_json, out_md) == 0
        report = json.loads(out_json.read_text(encoding="utf-8"))
        paths = [item["path"] for item in report["candidates"]]
        assert str(pbr_dir.resolve()) in paths


def test_locator_confidence_strong_for_model_index_unet_scheduler() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pbr_dir = root / "models" / "hunyuan3d-paintpbr-v2-1"
        write_file(pbr_dir / "model_index.json", b"{}")
        (pbr_dir / "unet").mkdir(parents=True)
        (pbr_dir / "scheduler").mkdir()
        out_json = root / "locator.json"
        out_md = root / "locator.md"

        assert run_locator(root, out_json, out_md) == 0
        report = json.loads(out_json.read_text(encoding="utf-8"))
        match = next(
            item for item in report["candidates"]
            if item["path"] == str(pbr_dir.resolve())
        )
        assert match["confidence"] == "strong"
        assert report["strong_candidate_count"] >= 1
