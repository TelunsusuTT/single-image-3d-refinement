#!/usr/bin/env python3
"""Check local Phase 2B GLB files listed in a manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"manifest CSV missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest CSV has no header: {path}")
        return list(reader)


def check_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in rows:
        path_value = row.get("local_glb_path", "").strip()
        path = Path(path_value) if path_value else None
        exists = path.is_file() if path is not None else False
        size_bytes = path.stat().st_size if exists and path is not None else 0
        errors: list[str] = []
        if path is None:
            errors.append("local_glb_path missing")
        elif not exists:
            errors.append("file missing")
        elif size_bytes <= 0:
            errors.append("file is zero-size")

        results.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "source_id": row.get("source_id", ""),
                "local_glb_path": path_value,
                "exists": exists,
                "size_bytes": size_bytes,
                "ok": not errors,
                "errors": errors,
            }
        )
    return results


def print_results(results: list[dict[str, object]]) -> None:
    for index, result in enumerate(results, start=1):
        status = "OK" if result["ok"] else "FAIL"
        print(f"[{index}/{len(results)}] {result['candidate_id'] or result['source_id']}")
        print(f"  path: {result['local_glb_path']}")
        print(f"  exists: {result['exists']}")
        print(f"  size_bytes: {result['size_bytes']}")
        print(f"  status: {status}")
        for error in result["errors"]:
            print(f"  error: {error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2B local GLB files.")
    parser.add_argument("--manifest-csv", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_manifest(args.manifest_csv)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results = check_rows(rows)
    print_results(results)
    failed = [result for result in results if not result["ok"]]
    print(f"Checked {len(results)} asset(s): {len(results) - len(failed)} OK, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
