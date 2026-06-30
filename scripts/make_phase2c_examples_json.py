#!/usr/bin/env python3
"""Write Phase 2C examples JSON from a render manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_sample_dirs(render_manifest: Path, absolute: bool) -> list[str]:
    with render_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_dirs: list[str] = []
    for row in rows:
        sample_dir = Path(row["sample_dir"])
        sample_dirs.append(str(sample_dir.resolve()) if absolute else str(sample_dir))
    return sample_dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Phase 2C examples JSON.")
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--absolute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample_dirs = read_sample_dirs(args.render_manifest, args.absolute)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(sample_dirs, handle, indent=2)
        handle.write("\n")
    print(f"wrote examples JSON: {args.out_json}")
    print(f"sample dirs: {len(sample_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
