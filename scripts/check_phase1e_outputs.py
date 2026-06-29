#!/usr/bin/env python3
"""Check Phase 1E Hunyuan3D-Paint-style render outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


RENDER_TEX_SUFFIXES = [".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"]
RENDER_COND_SUFFIXES = ["_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"]


def expected_files(sample_dir: Path, num_view: int) -> list[Path]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    files = [render_tex / "transforms.json"]
    for index in range(num_view):
        prefix = f"{index:03d}"
        files.extend(render_tex / f"{prefix}{suffix}" for suffix in RENDER_TEX_SUFFIXES)
        files.extend(render_cond / f"{prefix}{suffix}" for suffix in RENDER_COND_SUFFIXES)
    return files


def check_sample(sample_dir: Path, num_view: int) -> dict[str, object]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    missing: list[str] = []

    if not render_tex.is_dir():
        missing.append(str(render_tex))
    if not render_cond.is_dir():
        missing.append(str(render_cond))

    for path in expected_files(sample_dir, num_view):
        if not path.is_file():
            missing.append(str(path))

    return {
        "sample_dir": str(sample_dir),
        "num_view": num_view,
        "render_tex_exists": render_tex.is_dir(),
        "render_cond_exists": render_cond.is_dir(),
        "expected_file_count": 1 + num_view * (len(RENDER_TEX_SUFFIXES) + len(RENDER_COND_SUFFIXES)),
        "missing": missing,
        "ok": not missing,
    }


def print_summary(result: dict[str, object]) -> None:
    missing = result["missing"]
    assert isinstance(missing, list)
    print(f"sample_dir: {result['sample_dir']}")
    print(f"num_view: {result['num_view']}")
    print(f"render_tex: {'OK' if result['render_tex_exists'] else 'MISSING'}")
    print(f"render_cond: {'OK' if result['render_cond_exists'] else 'MISSING'}")
    print(f"expected files: {result['expected_file_count']}")
    print(f"missing files: {len(missing)}")
    if missing:
        print("missing:")
        for path in missing:
            print(f"  - {path}")
        print("status: FAIL")
    else:
        print("status: PASS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase 1E render output structure and filenames."
    )
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--num-view", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_view < 1:
        print("ERROR: --num-view must be >= 1", file=sys.stderr)
        return 2
    result = check_sample(args.sample_dir, args.num_view)
    print_summary(result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
