#!/usr/bin/env python3
"""Create an absolute-path Phase 1F one-sample training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def write_manifest(sample_dir: Path, out_json: Path) -> Path:
    resolved_sample = sample_dir.expanduser().resolve()
    if not resolved_sample.is_dir():
        raise ValueError(f"sample directory missing: {resolved_sample}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump([str(resolved_sample)], handle, indent=2)
        handle.write("\n")
    return resolved_sample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a Phase 1F examples JSON with one absolute sample path."
    )
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        resolved_sample = write_manifest(args.sample_dir, args.out_json)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"wrote examples JSON: {args.out_json}")
    print(f"sample_dir: {resolved_sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
