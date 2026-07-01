#!/usr/bin/env python3
"""Inspect checkpoint filesystem metadata without opening model contents."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def size_human(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def inspect_checkpoint(checkpoint: Path) -> dict[str, Any]:
    resolved = checkpoint.expanduser().resolve()
    exists = resolved.is_file()
    stat = resolved.stat() if exists else None
    size_bytes = stat.st_size if stat is not None else 0
    mtime = (
        datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        if stat is not None
        else ""
    )
    return {
        "checkpoint": str(resolved),
        "exists": exists,
        "size_bytes": size_bytes,
        "size_human": size_human(size_bytes),
        "mtime": mtime,
        "ok": exists and size_bytes > 0,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2G.1 Checkpoint Metadata",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- exists: `{report['exists']}`",
        f"- size_bytes: `{report['size_bytes']}`",
        f"- size_human: `{report['size_human']}`",
        f"- mtime_utc: `{report['mtime']}`",
        f"- status: `{'OK' if report['ok'] else 'FAIL'}`",
        "",
        "This report uses filesystem metadata only. It does not import torch and",
        "does not open or load checkpoint tensor contents.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect checkpoint metadata only.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = inspect_checkpoint(args.checkpoint)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"checkpoint: {report['checkpoint']}")
    print(f"exists: {report['exists']}")
    print(f"size: {report['size_human']}")
    if report["ok"]:
        print("PHASE2G1_CHECKPOINT_METADATA_OK")
        return 0
    print("PHASE2G1_CHECKPOINT_METADATA_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
