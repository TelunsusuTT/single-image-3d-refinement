#!/usr/bin/env python3
"""Download only GLBs listed in a Phase 2B manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.request
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"manifest CSV missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest CSV has no header: {path}")
        return list(reader)


def size_text(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def download_url_to_path(url: str, local_path: Path, timeout: int, skip_existing: bool) -> str:
    if skip_existing and local_path.is_file() and local_path.stat().st_size > 0:
        return "skipped"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "hy3dpaint-phase2b/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, local_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return "downloaded"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Phase 2B selected GLBs.")
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout <= 0:
        print("ERROR: --timeout must be > 0", file=sys.stderr)
        return 2

    try:
        rows = load_manifest(args.manifest_csv)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures = 0
    for row in rows:
        candidate_id = row.get("candidate_id", "") or row.get("source_id", "")
        url = row.get("download_url", "").strip()
        local_path_value = row.get("local_glb_path", "").strip()
        if not url or not local_path_value:
            failures += 1
            print(f"FAIL {candidate_id}: missing download_url or local_glb_path", file=sys.stderr)
            continue
        local_path = Path(local_path_value)
        try:
            status = download_url_to_path(url, local_path, args.timeout, args.skip_existing)
            print(f"{status}: {candidate_id} {local_path} ({size_text(local_path)})")
        except OSError as exc:
            failures += 1
            print(f"FAIL {candidate_id}: {url} -> {local_path}: {exc}", file=sys.stderr)

    print(f"download failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
