from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase1e_outputs import main  # noqa: E402


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def make_complete_sample(root: Path, num_view: int = 2) -> Path:
    sample_dir = root / "sample"
    touch(sample_dir / "render_tex" / "transforms.json")
    for index in range(num_view):
        prefix = f"{index:03d}"
        for suffix in (".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"):
            touch(sample_dir / "render_tex" / f"{prefix}{suffix}")
        for suffix in ("_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"):
            touch(sample_dir / "render_cond" / f"{prefix}{suffix}")
    return sample_dir


def test_complete_fake_sample_passes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = make_complete_sample(Path(tmpdir))
        assert main(["--sample-dir", str(sample_dir), "--num-view", "2"]) == 0


def test_missing_render_tex_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = Path(tmpdir) / "sample"
        (sample_dir / "render_cond").mkdir(parents=True)
        assert main(["--sample-dir", str(sample_dir), "--num-view", "2"]) == 1


def test_missing_mr_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = make_complete_sample(Path(tmpdir))
        (sample_dir / "render_tex" / "001_mr.png").unlink()
        assert main(["--sample-dir", str(sample_dir), "--num-view", "2"]) == 1


def test_missing_light_condition_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = make_complete_sample(Path(tmpdir))
        (sample_dir / "render_cond" / "000_light_PL.png").unlink()
        assert main(["--sample-dir", str(sample_dir), "--num-view", "2"]) == 1
