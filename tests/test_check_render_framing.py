from __future__ import annotations

import struct
import sys
import tempfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_render_framing import check_sample, main  # noqa: E402


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    import binascii

    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_rect_png(
    path: Path,
    width: int,
    height: int,
    rect: tuple[int, int, int, int],
    foreground: tuple[int, ...],
    background: tuple[int, ...],
    color_type: int = 2,
) -> None:
    min_x, min_y, max_x, max_y = rect
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            if min_x <= x <= max_x and min_y <= y <= max_y:
                row.extend(foreground)
            else:
                row.extend(background)
        rows.append(b"\x00" + bytes(row))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    data = PNG_SIGNATURE
    data += png_chunk(b"IHDR", ihdr)
    data += png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
    data += png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_rgb_png(path: Path, width: int, height: int, rect: tuple[int, int, int, int]) -> None:
    write_rect_png(path, width, height, rect, (0, 0, 0), (255, 255, 255))


def write_mask_png(path: Path, width: int, height: int, rect: tuple[int, int, int, int]) -> None:
    write_rect_png(path, width, height, rect, (255, 255, 255), (0, 0, 0))


def write_gray_mask_png(
    path: Path,
    width: int,
    height: int,
    rect: tuple[int, int, int, int],
    background_value: int = 58,
) -> None:
    background = (background_value, background_value, background_value)
    write_rect_png(path, width, height, rect, (255, 255, 255), background)


def write_rgba_mask_png(path: Path, width: int, height: int, rect: tuple[int, int, int, int]) -> None:
    write_rect_png(path, width, height, rect, (255, 255, 255, 255), (0, 0, 0, 255), color_type=6)


def write_opaque_alpha_only_mask_png(path: Path, width: int, height: int) -> None:
    write_rect_png(path, width, height, (0, 0, width - 1, height - 1), (0, 0, 0, 255), (0, 0, 0, 255), color_type=6)


def make_sample(root: Path, rect: tuple[int, int, int, int], num_view: int = 1) -> Path:
    sample_dir = root / "sample"
    for index in range(num_view):
        write_rgb_png(sample_dir / "render_tex" / f"{index:03d}.png", 64, 64, rect)
    return sample_dir


def make_mask(root: Path, rect: tuple[int, int, int, int], num_view: int = 1) -> Path:
    qa_dir = root / "qa" / "sample"
    for index in range(num_view):
        write_mask_png(qa_dir / f"{index:03d}_mask.png", 64, 64, rect)
    return qa_dir


def test_dark_gray_mask_background_passes_with_default_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_sample(root, (0, 0, 63, 63))
        qa_dir = root / "qa" / "sample"
        write_gray_mask_png(qa_dir / "000_mask.png", 64, 64, (20, 20, 43, 43))

        report = check_sample(sample_dir, 1, 4, qa_dir)
        view = report["views"][0]
        assert report["ok"] is True
        assert view["detection_source"] == "mask"
        assert view["mask_threshold"] == 128
        assert view["min_luminance"] == 58.0
        assert view["max_luminance"] == 255.0
        assert view["corner_luminance"]["top_left"] == 58.0
        assert main(
            [
                "--sample-dir",
                str(sample_dir),
                "--qa-dir",
                str(qa_dir),
                "--num-view",
                "1",
                "--border-margin-px",
                "4",
            ]
        ) == 0


def test_dark_gray_mask_border_object_fails_with_default_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_sample(root, (20, 20, 43, 43))
        qa_dir = root / "qa" / "sample"
        write_gray_mask_png(qa_dir / "000_mask.png", 64, 64, (0, 20, 20, 43))
        assert main(
            [
                "--sample-dir",
                str(sample_dir),
                "--qa-dir",
                str(qa_dir),
                "--num-view",
                "1",
                "--border-margin-px",
                "4",
            ]
        ) == 1


def test_mask_based_centered_object_passes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_sample(root, (0, 0, 63, 63))
        qa_dir = make_mask(root, (20, 20, 43, 43))
        assert main(
            [
                "--sample-dir",
                str(sample_dir),
                "--qa-dir",
                str(qa_dir),
                "--num-view",
                "1",
                "--border-margin-px",
                "4",
            ]
        ) == 0


def test_mask_based_border_object_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_sample(root, (20, 20, 43, 43))
        qa_dir = make_mask(root, (0, 20, 20, 43))
        assert main(
            [
                "--sample-dir",
                str(sample_dir),
                "--qa-dir",
                str(qa_dir),
                "--num-view",
                "1",
                "--border-margin-px",
                "4",
            ]
        ) == 1



def test_rgba_mask_with_opaque_black_background_passes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_sample(root, (0, 0, 63, 63))
        qa_dir = root / "qa" / "sample"
        write_rgba_mask_png(qa_dir / "000_mask.png", 64, 64, (20, 20, 43, 43))
        assert main(
            [
                "--sample-dir",
                str(sample_dir),
                "--qa-dir",
                str(qa_dir),
                "--num-view",
                "1",
                "--border-margin-px",
                "4",
            ]
        ) == 0


def test_rgba_mask_with_opaque_black_background_border_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_sample(root, (20, 20, 43, 43))
        qa_dir = root / "qa" / "sample"
        write_rgba_mask_png(qa_dir / "000_mask.png", 64, 64, (0, 20, 20, 43))
        assert main(
            [
                "--sample-dir",
                str(sample_dir),
                "--qa-dir",
                str(qa_dir),
                "--num-view",
                "1",
                "--border-margin-px",
                "4",
            ]
        ) == 1


def test_mask_alpha_alone_does_not_make_full_image_foreground() -> None:
    from check_render_framing import check_sample

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_dir = make_sample(root, (20, 20, 43, 43))
        qa_dir = root / "qa" / "sample"
        write_opaque_alpha_only_mask_png(qa_dir / "000_mask.png", 64, 64)
        report = check_sample(sample_dir, 1, 4, qa_dir)
        view = report["views"][0]
        assert view["detection_source"] == "mask"
        assert view["foreground_area_ratio"] == 0.0
        assert view["foreground_bbox"] is None
        assert report["ok"] is True


def test_rgb_fallback_centered_object_passes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = make_sample(Path(tmpdir), (20, 20, 43, 43))
        assert main(["--sample-dir", str(sample_dir), "--num-view", "1", "--border-margin-px", "4"]) == 0


def test_object_touching_border_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = make_sample(Path(tmpdir), (0, 20, 20, 43))
        assert main(["--sample-dir", str(sample_dir), "--num-view", "1", "--border-margin-px", "4"]) == 1


def test_tiny_object_warns_but_does_not_fail() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = make_sample(Path(tmpdir), (31, 31, 32, 32))
        assert main(["--sample-dir", str(sample_dir), "--num-view", "1", "--border-margin-px", "4"]) == 0


def test_off_center_object_warns_but_does_not_fail() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = make_sample(Path(tmpdir), (44, 26, 54, 37))
        assert main(["--sample-dir", str(sample_dir), "--num-view", "1", "--border-margin-px", "4"]) == 0
