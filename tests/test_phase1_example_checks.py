from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_hy3dpaint_example import check_sample  # noqa: E402


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def make_sample(
    root: Path,
    *,
    num_view: int = 2,
    include_render_cond: bool = True,
    albedo_count: int | None = None,
) -> Path:
    sample_dir = root / "sample_000"
    render_tex = sample_dir / "render_tex"

    if include_render_cond:
        for index in range(num_view):
            touch(sample_dir / "render_cond" / f"view_{index:03d}.png")

    if albedo_count is None:
        albedo_count = num_view

    for index in range(albedo_count):
        touch(render_tex / f"view_{index:03d}_albedo.png")
    for index in range(num_view):
        touch(render_tex / f"view_{index:03d}_mr.png")
        touch(render_tex / f"view_{index:03d}_normal.png")
        touch(render_tex / f"view_{index:03d}_pos.png")

    touch(sample_dir / "transforms.json")
    return sample_dir


class Phase1ExampleChecksTest(unittest.TestCase):
    def test_check_sample_passes_with_enough_fake_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = make_sample(Path(tmpdir), num_view=2)

            result = check_sample(sample_dir, num_view=2, strict=True)

            self.assertTrue(result["ok"])
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["counts"]["condition"], 2)
            self.assertEqual(result["counts"]["albedo"], 2)
            self.assertEqual(result["counts"]["metallic_roughness"], 2)
            self.assertEqual(result["counts"]["normal"], 2)
            self.assertEqual(result["counts"]["position"], 2)
            self.assertTrue(result["transforms_json_exists"])

    def test_check_sample_fails_when_render_cond_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = make_sample(
                Path(tmpdir), num_view=2, include_render_cond=False
            )

            result = check_sample(sample_dir, num_view=2)

            self.assertFalse(result["ok"])
            self.assertIn("render_cond/ missing", result["errors"])
            self.assertIn("condition image count is 0", result["errors"])

    def test_check_sample_fails_when_albedo_count_is_too_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = make_sample(Path(tmpdir), num_view=2, albedo_count=1)

            result = check_sample(sample_dir, num_view=2)

            self.assertFalse(result["ok"])
            self.assertIn("albedo image count 1 < num-view 2", result["errors"])


if __name__ == "__main__":
    unittest.main()
