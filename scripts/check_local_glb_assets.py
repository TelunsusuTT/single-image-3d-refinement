#!/usr/bin/env python3
"""Check local GLB/GLTF asset files listed in a manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


VALID_SUFFIXES = {".glb", ".gltf"}
BYTES_PER_MB = 1024 * 1024


def is_todo_or_empty(value: str) -> bool:
    stripped = value.strip()
    return not stripped or stripped.upper().startswith("TODO")


def load_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    if not manifest_csv.exists():
        raise ValueError(f"manifest CSV missing: {manifest_csv}")
    if not manifest_csv.is_file():
        raise ValueError(f"manifest CSV is not a file: {manifest_csv}")

    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest CSV has no header: {manifest_csv}")
        return [dict(row) for row in reader]


def local_path_for_row(row: dict[str, str]) -> str:
    resolved = row.get("resolved_local_path", "").strip()
    if not is_todo_or_empty(resolved):
        return resolved
    return row.get("local_path", "").strip()


def check_asset(
    row: dict[str, str], min_size_mb: float, max_size_mb: float
) -> dict[str, Any]:
    selected_id = row.get("selected_id", "") or row.get("source_id", "") or "(unknown)"
    local_path_value = local_path_for_row(row)
    errors: list[str] = []
    size_mb = 0.0

    if is_todo_or_empty(local_path_value):
        errors.append("local path missing or TODO")
        return {
            "selected_id": selected_id,
            "local_path": local_path_value,
            "exists": False,
            "suffix_ok": False,
            "size_mb": size_mb,
            "ok": False,
            "errors": errors,
        }

    local_path = Path(local_path_value)
    exists = local_path.is_file()
    suffix_ok = local_path.suffix.lower() in VALID_SUFFIXES

    if not exists:
        errors.append("file missing")
    if not suffix_ok:
        errors.append("suffix is not .glb or .gltf")

    if exists:
        size_mb = local_path.stat().st_size / BYTES_PER_MB
        if size_mb < min_size_mb:
            errors.append(f"size {size_mb:.6f} MB < min-size-mb {min_size_mb}")
        if size_mb > max_size_mb:
            errors.append(f"size {size_mb:.6f} MB > max-size-mb {max_size_mb}")

    return {
        "selected_id": selected_id,
        "local_path": local_path_value,
        "exists": exists,
        "suffix_ok": suffix_ok,
        "size_mb": size_mb,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(result: dict[str, Any], index: int, total: int) -> None:
    status = "OK" if result["ok"] else "FAIL"
    print(f"[{index}/{total}] {result['selected_id']}")
    print(f"  path: {result['local_path'] or 'missing'}")
    print(f"  exists: {'OK' if result['exists'] else 'MISSING'}")
    print(f"  suffix: {'OK' if result['suffix_ok'] else 'FAIL'}")
    print(f"  size_mb: {result['size_mb']:.6f}")
    print(f"  status: {status}")
    if result["errors"]:
        print("  errors:")
        for error in result["errors"]:
            print(f"    - {error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check local GLB/GLTF asset files without importing them."
    )
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--min-size-mb", type=float, default=0.1)
    parser.add_argument("--max-size-mb", type=float, default=500.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_size_mb < 0:
        print("ERROR: --min-size-mb must be >= 0", file=sys.stderr)
        return 2
    if args.max_size_mb < args.min_size_mb:
        print("ERROR: --max-size-mb must be >= --min-size-mb", file=sys.stderr)
        return 2

    try:
        rows = load_manifest(args.manifest_csv)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results = [
        check_asset(row, args.min_size_mb, args.max_size_mb) for row in rows
    ]
    for index, result in enumerate(results, start=1):
        print_summary(result, index, len(results))

    failed = [result for result in results if not result["ok"]]
    print(f"Checked {len(results)} asset(s): {len(results) - len(failed)} OK, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
