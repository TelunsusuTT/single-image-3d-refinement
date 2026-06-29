#!/usr/bin/env python3
"""Check Phase 1E render framing using sidecar masks when available."""

from __future__ import annotations

import argparse
import binascii
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Callable


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FOREGROUND_DELTA_FROM_WHITE = 18
DEFAULT_MASK_THRESHOLD = 128
MIN_AREA_RATIO = 0.08
MAX_AREA_RATIO = 0.85
MAX_CENTER_OFFSET_RATIO = 0.20
MASK_MODE = "luminance_ignore_alpha"

Pixel = tuple[int, int, int, int]


def read_png(path: Path) -> tuple[int, int, list[Pixel]]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG file: {path}")

    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    idat_parts: list[bytes] = []
    palette: list[tuple[int, int, int]] = []
    transparency = b""

    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError(f"truncated PNG chunk header: {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"PLTE":
            palette = [
                tuple(chunk_data[index : index + 3])
                for index in range(0, len(chunk_data), 3)
            ]
        elif chunk_type == b"tRNS":
            transparency = chunk_data
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"PNG missing IHDR: {path}")
    if bit_depth != 8:
        raise ValueError(f"unsupported PNG bit depth {bit_depth}: {path}")
    if interlace != 0:
        raise ValueError(f"interlaced PNG is not supported: {path}")

    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"unsupported PNG color type {color_type}: {path}")

    bytes_per_pixel = channels_by_type[color_type]
    row_bytes = width * bytes_per_pixel
    raw = zlib.decompress(b"".join(idat_parts))
    rows = unfilter_rows(raw, width, height, row_bytes, bytes_per_pixel)
    pixels = rows_to_rgba(rows, width, color_type, bytes_per_pixel, palette, transparency)
    return width, height, pixels


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_dist = abs(estimate - left)
    above_dist = abs(estimate - above)
    upper_left_dist = abs(estimate - upper_left)
    if left_dist <= above_dist and left_dist <= upper_left_dist:
        return left
    if above_dist <= upper_left_dist:
        return above
    return upper_left


def unfilter_rows(
    raw: bytes, width: int, height: int, row_bytes: int, bytes_per_pixel: int
) -> list[bytes]:
    rows: list[bytes] = []
    previous = bytearray(row_bytes)
    offset = 0
    expected = height * (row_bytes + 1)
    if len(raw) < expected:
        raise ValueError("truncated PNG image data")

    for _ in range(height):
        filter_type = raw[offset]
        scanline = raw[offset + 1 : offset + 1 + row_bytes]
        offset += row_bytes + 1
        recon = bytearray(row_bytes)
        for index, value in enumerate(scanline):
            left = recon[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            above = previous[index]
            upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 0:
                recon[index] = value
            elif filter_type == 1:
                recon[index] = (value + left) & 0xFF
            elif filter_type == 2:
                recon[index] = (value + above) & 0xFF
            elif filter_type == 3:
                recon[index] = (value + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                recon[index] = (value + paeth_predictor(left, above, upper_left)) & 0xFF
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
        rows.append(bytes(recon))
        previous = recon
    return rows


def rows_to_rgba(
    rows: list[bytes],
    width: int,
    color_type: int,
    bytes_per_pixel: int,
    palette: list[tuple[int, int, int]],
    transparency: bytes,
) -> list[Pixel]:
    pixels: list[Pixel] = []
    for row in rows:
        for x in range(width):
            start = x * bytes_per_pixel
            if color_type == 0:
                gray = row[start]
                pixels.append((gray, gray, gray, 255))
            elif color_type == 2:
                pixels.append((row[start], row[start + 1], row[start + 2], 255))
            elif color_type == 3:
                palette_index = row[start]
                red, green, blue = palette[palette_index]
                alpha = transparency[palette_index] if palette_index < len(transparency) else 255
                pixels.append((red, green, blue, alpha))
            elif color_type == 4:
                gray = row[start]
                pixels.append((gray, gray, gray, row[start + 1]))
            elif color_type == 6:
                pixels.append((row[start], row[start + 1], row[start + 2], row[start + 3]))
    return pixels


def luminance(pixel: Pixel) -> float:
    red, green, blue, _alpha = pixel
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def mask_luminance_stats(pixels: list[Pixel], width: int, height: int) -> dict[str, Any]:
    if not pixels:
        return {
            "min_luminance": None,
            "max_luminance": None,
            "corner_luminance": {},
        }

    values = [luminance(pixel) for pixel in pixels]

    def rounded(value: float) -> float:
        return round(float(value), 4)

    return {
        "min_luminance": rounded(min(values)),
        "max_luminance": rounded(max(values)),
        "corner_luminance": {
            "top_left": rounded(luminance(pixels[0])),
            "top_right": rounded(luminance(pixels[width - 1])),
            "bottom_left": rounded(luminance(pixels[(height - 1) * width])),
            "bottom_right": rounded(luminance(pixels[(height * width) - 1])),
        },
    }


def empty_luminance_stats() -> dict[str, Any]:
    return {
        "min_luminance": None,
        "max_luminance": None,
        "corner_luminance": {},
    }


def is_rgb_foreground(pixel: Pixel) -> bool:
    red, green, blue, alpha = pixel
    if alpha <= 16:
        return False
    return (
        255 - red > FOREGROUND_DELTA_FROM_WHITE
        or 255 - green > FOREGROUND_DELTA_FROM_WHITE
        or 255 - blue > FOREGROUND_DELTA_FROM_WHITE
    )


def make_mask_foreground_predicate(mask_threshold: int) -> Callable[[Pixel], bool]:
    def is_mask_foreground(pixel: Pixel) -> bool:
        return luminance(pixel) > mask_threshold

    return is_mask_foreground


def analyze_pixels(
    path: Path,
    pixels: list[Pixel],
    width: int,
    height: int,
    border_margin_px: int,
    detection_source: str,
    mask_path: Path | None,
    foreground_predicate: Callable[[Pixel], bool],
    mask_threshold: int | None,
    mask_mode: str,
) -> dict[str, Any]:
    luminance_summary = (
        mask_luminance_stats(pixels, width, height)
        if detection_source == "mask"
        else empty_luminance_stats()
    )
    foreground: list[tuple[int, int]] = []
    for index, pixel in enumerate(pixels):
        if foreground_predicate(pixel):
            foreground.append((index % width, index // width))

    warnings: list[str] = []
    failures: list[str] = []
    if not foreground:
        warnings.append("no foreground detected")
        return {
            "path": str(path),
            "mask_path": str(mask_path) if mask_path is not None else "",
            "detection_source": detection_source,
            "mask_mode": mask_mode,
            "mask_threshold": mask_threshold,
            **luminance_summary,
            "width": width,
            "height": height,
            "foreground_bbox": None,
            "foreground_area_ratio": 0.0,
            "center_offset_ratio": 0.0,
            "warnings": warnings,
            "failures": failures,
            "ok": True,
        }

    min_x = min(x for x, _ in foreground)
    max_x = max(x for x, _ in foreground)
    min_y = min(y for _, y in foreground)
    max_y = max(y for _, y in foreground)
    area_ratio = len(foreground) / float(width * height)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_offset = math.sqrt((center_x - width / 2.0) ** 2 + (center_y - height / 2.0) ** 2)
    center_offset_ratio = center_offset / float(max(width, height))

    if (
        min_x <= border_margin_px
        or min_y <= border_margin_px
        or max_x >= width - 1 - border_margin_px
        or max_y >= height - 1 - border_margin_px
    ):
        failures.append(f"foreground touches image border within {border_margin_px}px")
    if area_ratio < MIN_AREA_RATIO:
        warnings.append(f"foreground area ratio {area_ratio:.4f} < {MIN_AREA_RATIO}")
    if area_ratio > MAX_AREA_RATIO:
        warnings.append(f"foreground area ratio {area_ratio:.4f} > {MAX_AREA_RATIO}")
    if center_offset_ratio > MAX_CENTER_OFFSET_RATIO:
        warnings.append(
            f"foreground center offset ratio {center_offset_ratio:.4f} > {MAX_CENTER_OFFSET_RATIO}"
        )

    return {
        "path": str(path),
        "mask_path": str(mask_path) if mask_path is not None else "",
        "detection_source": detection_source,
        "mask_mode": mask_mode,
        "mask_threshold": mask_threshold,
        **luminance_summary,
        "width": width,
        "height": height,
        "foreground_bbox": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
        },
        "foreground_area_ratio": area_ratio,
        "center_offset_ratio": center_offset_ratio,
        "warnings": warnings,
        "failures": failures,
        "ok": not failures,
    }


def analyze_render_image(path: Path, border_margin_px: int) -> dict[str, Any]:
    width, height, pixels = read_png(path)
    return analyze_pixels(
        path,
        pixels,
        width,
        height,
        border_margin_px,
        "rgb_background",
        None,
        is_rgb_foreground,
        None,
        "rgb_background_delta_from_white",
    )


def analyze_mask(
    path: Path, mask_path: Path, border_margin_px: int, mask_threshold: int
) -> dict[str, Any]:
    width, height, pixels = read_png(mask_path)
    return analyze_pixels(
        path,
        pixels,
        width,
        height,
        border_margin_px,
        "mask",
        mask_path,
        make_mask_foreground_predicate(mask_threshold),
        mask_threshold,
        MASK_MODE,
    )


def failure_result(
    path: Path,
    mask_path: Path | None,
    detection_source: str,
    failure: str,
    mask_threshold: int | None,
    mask_mode: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "mask_path": str(mask_path) if mask_path is not None else "",
        "detection_source": detection_source,
        "mask_mode": mask_mode,
        "mask_threshold": mask_threshold,
        **empty_luminance_stats(),
        "width": 0,
        "height": 0,
        "foreground_bbox": None,
        "foreground_area_ratio": 0.0,
        "center_offset_ratio": 0.0,
        "warnings": [],
        "failures": [failure],
        "ok": False,
    }


def check_sample(
    sample_dir: Path,
    num_view: int,
    border_margin_px: int,
    qa_dir: Path | None = None,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index in range(num_view):
        path = sample_dir / "render_tex" / f"{index:03d}.png"
        mask_path = qa_dir / f"{index:03d}_mask.png" if qa_dir is not None else None
        mask_exists = mask_path is not None and mask_path.is_file()
        if not path.is_file():
            detection_source = "mask" if mask_exists else "rgb_background"
            threshold = mask_threshold if mask_exists else None
            mode = MASK_MODE if mask_exists else "rgb_background_delta_from_white"
            results.append(
                failure_result(path, mask_path, detection_source, "render image missing", threshold, mode)
            )
            continue

        try:
            if mask_exists:
                results.append(analyze_mask(path, mask_path, border_margin_px, mask_threshold))
            else:
                results.append(analyze_render_image(path, border_margin_px))
        except (OSError, ValueError, zlib.error, struct.error, binascii.Error) as exc:
            detection_source = "mask" if mask_exists else "rgb_background"
            threshold = mask_threshold if mask_exists else None
            mode = MASK_MODE if mask_exists else "rgb_background_delta_from_white"
            results.append(
                failure_result(path, mask_path, detection_source, f"could not analyze PNG: {exc}", threshold, mode)
            )

    return {
        "sample_dir": str(sample_dir),
        "qa_dir": str(qa_dir) if qa_dir is not None else "",
        "num_view": num_view,
        "border_margin_px": border_margin_px,
        "mask_threshold": mask_threshold,
        "views": results,
        "ok": all(result["ok"] for result in results),
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"sample_dir: {report['sample_dir']}")
    print(f"qa_dir: {report['qa_dir'] or 'none'}")
    print(f"num_view: {report['num_view']}")
    print(f"border_margin_px: {report['border_margin_px']}")
    print(f"mask_threshold: {report['mask_threshold']}")
    for index, view in enumerate(report["views"]):
        status = "OK" if view["ok"] else "FAIL"
        print(f"[{index:03d}] {view['path']}")
        print(f"  detection_source: {view['detection_source']}")
        print(f"  mask_path: {view['mask_path'] or 'none'}")
        print(f"  mask_mode: {view['mask_mode']}")
        print(f"  mask_threshold: {view['mask_threshold']}")
        if view["detection_source"] == "mask":
            print(f"  min_luminance: {view['min_luminance']}")
            print(f"  max_luminance: {view['max_luminance']}")
            print(f"  corner_luminance: {view['corner_luminance']}")
        print(f"  size: {view['width']}x{view['height']}")
        print(f"  foreground_bbox: {view['foreground_bbox']}")
        print(f"  area_ratio: {view['foreground_area_ratio']:.4f}")
        print(f"  center_offset_ratio: {view['center_offset_ratio']:.4f}")
        print(f"  status: {status}")
        for warning in view["warnings"]:
            print(f"  warning: {warning}")
        for failure in view["failures"]:
            print(f"  failure: {failure}")
    print(f"overall: {'PASS' if report['ok'] else 'FAIL'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check render framing for Phase 1E outputs.")
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--num-view", type=int, default=6)
    parser.add_argument("--border-margin-px", type=int, default=8)
    parser.add_argument("--mask-threshold", type=int, default=DEFAULT_MASK_THRESHOLD)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_view < 1:
        print("ERROR: --num-view must be >= 1", file=sys.stderr)
        return 2
    if args.border_margin_px < 0:
        print("ERROR: --border-margin-px must be >= 0", file=sys.stderr)
        return 2
    if args.mask_threshold < 0 or args.mask_threshold > 255:
        print("ERROR: --mask-threshold must be between 0 and 255", file=sys.stderr)
        return 2

    report = check_sample(
        args.sample_dir,
        args.num_view,
        args.border_margin_px,
        args.qa_dir,
        args.mask_threshold,
    )
    print_report(report)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote framing report: {args.out_json}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
