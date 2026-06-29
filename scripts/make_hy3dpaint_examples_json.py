#!/usr/bin/env python3
"""Write an examples.json containing exactly one Hunyuan3D-Paint sample path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def project_relative_or_absolute(path: Path, project_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a one-sample Hunyuan3D-Paint examples.json."
    )
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample_path = project_relative_or_absolute(args.sample_dir, Path.cwd())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump([sample_path], handle, indent=2)
        handle.write("\n")
    print(f"wrote examples json: {args.out_json}")
    print(f"sample directory: {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
