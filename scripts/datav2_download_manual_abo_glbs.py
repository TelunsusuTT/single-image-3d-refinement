#!/usr/bin/env python3
"""Download manually resolved ABO GLBs from a manifest, optionally as a dry run."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = "outputs/phase2l/manual_abo_download"


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def destination_for_row(row: dict[str, str], out_root: Path) -> Path:
    relative_path = (row.get("relative_path") or "").strip()
    return out_root / relative_path


def download_url_to_path(url: str, destination: Path, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def process_rows(
    rows: list[dict[str, str]],
    out_root: Path,
    dry_run: bool,
    overwrite: bool,
    timeout: float,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    results: list[dict[str, str]] = []
    counts = {
        "total_rows": len(rows),
        "found_rows": 0,
        "missing_in_metadata_rows": 0,
        "planned_download_count": 0,
        "downloaded_count": 0,
        "skipped_existing_count": 0,
        "failed_count": 0,
    }
    for row in rows:
        item_id = row.get("item_id", "")
        status = row.get("status", "")
        if status != "found":
            counts["missing_in_metadata_rows"] += 1
            results.append({"item_id": item_id, "action": "skip_missing_in_metadata", "path": "", "error": ""})
            continue
        counts["found_rows"] += 1
        destination = destination_for_row(row, out_root)
        if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
            counts["skipped_existing_count"] += 1
            results.append({"item_id": item_id, "action": "skip_existing", "path": str(destination), "error": ""})
            continue
        counts["planned_download_count"] += 1
        if dry_run:
            results.append({"item_id": item_id, "action": "planned_download", "path": str(destination), "error": ""})
            continue
        try:
            download_url_to_path(row.get("url", ""), destination, timeout)
        except Exception as exc:  # pragma: no cover - network path is intentionally not used in tests.
            counts["failed_count"] += 1
            results.append({"item_id": item_id, "action": "failed", "path": str(destination), "error": str(exc)})
        else:
            counts["downloaded_count"] += 1
            results.append({"item_id": item_id, "action": "downloaded", "path": str(destination), "error": ""})
    return results, counts


def write_summary(summary_json: Path, summary_md: Path, payload: dict[str, Any]) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Phase 2L.2A Manual ABO GLB Download",
        "",
        f"dry run: `{payload['dry_run']}`",
        f"manifest rows: `{payload['counts']['total_rows']}`",
        f"found rows: `{payload['counts']['found_rows']}`",
        f"missing in metadata rows: `{payload['counts']['missing_in_metadata_rows']}`",
        f"planned downloads: `{payload['counts']['planned_download_count']}`",
        f"downloaded: `{payload['counts']['downloaded_count']}`",
        f"skipped existing: `{payload['counts']['skipped_existing_count']}`",
        f"failed: `{payload['counts']['failed_count']}`",
        "",
        "| Item ID | Action | Path | Error |",
        "|---|---|---|---|",
    ]
    for row in payload["results"]:
        lines.append(
            "| `{item}` | `{action}` | `{path}` | `{error}` |".format(
                item=row["item_id"],
                action=row["action"],
                path=row["path"],
                error=row["error"],
            )
        )
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download manual ABO GLBs from a resolved manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    manifest_path = resolve_project_path(args.manifest, project_root)
    out_root = resolve_project_path(args.out_root, project_root)
    summary_root = project_root / DEFAULT_SUMMARY_ROOT
    summary_md = resolve_project_path(args.summary_md, project_root) if args.summary_md else summary_root / "manual_abo_download_summary.md"
    summary_json = (
        resolve_project_path(args.summary_json, project_root)
        if args.summary_json
        else summary_root / "manual_abo_download_summary.json"
    )

    rows = read_csv(manifest_path)
    results, counts = process_rows(rows, out_root, args.dry_run, args.overwrite, args.timeout)
    payload = {
        "manifest": str(manifest_path),
        "out_root": str(out_root),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "counts": counts,
        "results": results,
    }
    write_summary(summary_json, summary_md, payload)

    for row in results:
        print(f"{row['item_id']}: {row['action']} {row['path']}")
        if row["error"]:
            print(f"  error: {row['error']}")
    print("Phase 2L.2A manual ABO GLB download")
    print(f"  dry run: {args.dry_run}")
    print(f"  planned downloads: {counts['planned_download_count']}")
    print(f"  downloaded: {counts['downloaded_count']}")
    print(f"  failed: {counts['failed_count']}")
    print("PHASE2L2A_MANUAL_ABO_DOWNLOAD_OK")
    return 1 if counts["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
