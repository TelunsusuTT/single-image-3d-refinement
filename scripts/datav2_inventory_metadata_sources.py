#!/usr/bin/env python3
"""Inventory local metadata sources for Data v2 candidate mining."""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LIMIT = 25


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def resolve_output_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    return resolve_project_path(path_text, project_root=project_root)


def has_glob_magic(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def configured_metadata_patterns(config: dict[str, Any]) -> list[str]:
    patterns = config.get("metadata_paths", [])
    if not isinstance(patterns, list):
        raise ValueError("config metadata_paths must be a list")
    return [str(pattern) for pattern in patterns]


def resolve_metadata_files(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> list[Path]:
    paths: dict[str, Path] = {}
    parquet_dirs: set[Path] = set()

    for pattern in configured_metadata_patterns(config):
        resolved_pattern = resolve_project_path(pattern, project_root=project_root)
        if has_glob_magic(str(resolved_pattern)):
            for match in glob.glob(str(resolved_pattern)):
                path = Path(match)
                if path.is_file():
                    paths[str(path.resolve())] = path
            parent_text = str(resolved_pattern)
            first_magic = min(
                [idx for idx in (parent_text.find("*"), parent_text.find("?"), parent_text.find("[")) if idx >= 0],
                default=len(parent_text),
            )
            parquet_dirs.add(Path(parent_text[:first_magic]).parent)
        else:
            if resolved_pattern.is_file():
                paths[str(resolved_pattern.resolve())] = resolved_pattern
            parquet_dirs.add(resolved_pattern.parent)

    for directory in sorted(parquet_dirs):
        if not directory.is_dir():
            continue
        for parquet_path in list(directory.glob("*.parquet")) + list(directory.glob("*.parquet.gz")):
            paths[str(parquet_path.resolve())] = parquet_path

    return sorted(paths.values(), key=lambda path: str(path))


def detect_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv.gz") or name.endswith(".csv"):
        return "csv"
    if name.endswith(".jsonl.gz") or name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".json.gz") or name.endswith(".json"):
        return "json"
    if name.endswith(".parquet.gz") or name.endswith(".parquet"):
        return "parquet"
    return "unsupported"


@contextmanager
def open_text_file(path: Path) -> Iterator[Any]:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            yield handle


def object_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys())
    return []


def rows_from_json_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "items", "assets", "objects", "data", "rows"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    return []


def analyze_csv(path: Path) -> dict[str, Any]:
    sampled_rows = 0
    row_count = 0
    columns: list[str] = []
    with open_text_file(path) as handle:
        reader = csv.DictReader(handle)
        columns = [str(name) for name in (reader.fieldnames or [])]
        for row in reader:
            row_count += 1
            if sampled_rows < SAMPLE_LIMIT:
                sampled_rows += 1
    return {
        "parseable": True,
        "detected_columns_or_keys": columns,
        "sampled_row_count": sampled_rows,
        "row_count": row_count,
        "row_count_is_exact": True,
        "error": "",
    }


def analyze_jsonl(path: Path) -> dict[str, Any]:
    sampled_rows = 0
    row_count = 0
    keys: set[str] = set()
    with open_text_file(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            row_count += 1
            if sampled_rows < SAMPLE_LIMIT:
                sampled_rows += 1
                keys.update(object_keys(value))
    return {
        "parseable": True,
        "detected_columns_or_keys": sorted(keys),
        "sampled_row_count": sampled_rows,
        "row_count": row_count,
        "row_count_is_exact": True,
        "error": "",
    }


def analyze_json(path: Path) -> dict[str, Any]:
    with open_text_file(path) as handle:
        value = json.load(handle)
    rows = rows_from_json_value(value)
    keys: set[str] = set()
    for row in rows[:SAMPLE_LIMIT]:
        keys.update(object_keys(row))
    return {
        "parseable": True,
        "detected_columns_or_keys": sorted(keys),
        "sampled_row_count": min(len(rows), SAMPLE_LIMIT),
        "row_count": len(rows),
        "row_count_is_exact": True,
        "error": "",
    }


def analyze_metadata_file(path: Path) -> dict[str, Any]:
    file_format = detect_format(path)
    report: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "format": file_format,
        "parseable": False,
        "detected_columns_or_keys": [],
        "sampled_row_count": 0,
        "row_count": None,
        "row_count_is_exact": False,
        "error": "",
    }
    if file_format == "parquet":
        report["error"] = "parquet metadata is unsupported by the stdlib inventory"
        return report
    if file_format == "unsupported":
        report["error"] = "unsupported metadata format"
        return report
    try:
        if file_format == "csv":
            report.update(analyze_csv(path))
        elif file_format == "jsonl":
            report.update(analyze_jsonl(path))
        elif file_format == "json":
            report.update(analyze_json(path))
    except Exception as exc:  # noqa: BLE001 - report parse failures without crashing inventory.
        report["parseable"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def build_report(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    metadata_files = resolve_metadata_files(config)
    sources = [analyze_metadata_file(path) for path in metadata_files]
    return {
        "config": str(config_path),
        "target_subclass": config.get("target_subclass", ""),
        "metadata_patterns": configured_metadata_patterns(config),
        "metadata_file_count": len(sources),
        "parseable_file_count": sum(1 for source in sources if source["parseable"]),
        "out_dir": str(out_dir),
        "sources": sources,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2L.1 Metadata Sources Inventory",
        "",
        f"target subclass: `{report.get('target_subclass', '')}`",
        f"metadata files found: `{report['metadata_file_count']}`",
        f"parseable files: `{report['parseable_file_count']}`",
        "",
        "## Sources",
        "",
        "| Path | Format | Size bytes | Parseable | Rows | Sampled | Columns/keys | Error |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for source in report["sources"]:
        keys = ", ".join(source.get("detected_columns_or_keys", [])[:20])
        if len(source.get("detected_columns_or_keys", [])) > 20:
            keys += ", ..."
        error = str(source.get("error", "")).replace("|", "\\|")
        lines.append(
            "| `{path}` | `{fmt}` | {size} | `{parseable}` | {rows} | {sampled} | {keys} | {error} |".format(
                path=source["path"],
                fmt=source["format"],
                size=source["size_bytes"],
                parseable=source["parseable"],
                rows=source["row_count"] if source["row_count"] is not None else "",
                sampled=source["sampled_row_count"],
                keys=keys.replace("|", "\\|"),
                error=error,
            )
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory Data v2 metadata sources.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.config, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "metadata_sources_report.json"
    out_md = args.out_dir / "metadata_sources_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_md)
    print("Phase 2L.1 metadata inventory")
    print(f"  metadata files found: {report['metadata_file_count']}")
    print(f"  parseable files: {report['parseable_file_count']}")
    print(f"  out_json: {out_json}")
    print(f"  out_md: {out_md}")
    print("PHASE2L1_METADATA_INVENTORY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
