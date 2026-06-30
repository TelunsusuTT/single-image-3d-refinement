#!/usr/bin/env python3
"""Download only small ABO metadata files needed for Phase 2A selection."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


ABO_BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"


def metadata_paths(max_listing_index: int) -> list[str]:
    return [
        "3dmodels/metadata/3dmodels.csv.gz",
        "images/metadata/images.csv.gz",
        *[
            f"listings/metadata/listings_{index}.json.gz"
            for index in range(max_listing_index + 1)
        ],
    ]


def download_file(url: str, out_path: Path, skip_existing: bool = False) -> str:
    if skip_existing and out_path.is_file():
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, out_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return "downloaded"


def file_size_text(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download small ABO metadata files only; no images, GLBs, or archives."
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-listing-index", type=int, default=9)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_listing_index < 0:
        print("ERROR: --max-listing-index must be >= 0", file=sys.stderr)
        return 2

    failures = 0
    for rel_path in metadata_paths(args.max_listing_index):
        url = f"{ABO_BASE_URL}/{rel_path}"
        out_path = args.out_dir / Path(rel_path).name
        try:
            status = download_file(url, out_path, skip_existing=args.skip_existing)
            print(f"{status}: {out_path} ({file_size_text(out_path)})")
        except OSError as exc:
            failures += 1
            print(f"ERROR: failed to download {url}: {exc}", file=sys.stderr)

    if failures:
        print(f"metadata download failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
