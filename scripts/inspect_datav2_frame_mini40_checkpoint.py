#!/usr/bin/env python3
"""Stat-only inspection for Data v2 mini40 true-PBR checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config.get("readiness_report_dir", "outputs/phase2l/datav2_frame_panels/mini40_train_readiness"))


def has_expected_step_marker(filename: str, expected_step: str) -> bool:
    pattern = rf"step[=_-]?{re.escape(expected_step)}(?:\D|$)"
    return re.search(pattern, filename) is not None


def inspect_checkpoint_dir(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_dir = resolve_project_path(config["output_checkpoint_dir"])
    expected_step = str(config.get("steps", 500))
    files = []
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.glob("*.ckpt")):
            stat = path.stat()
            files.append(
                {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "has_expected_step_marker": has_expected_step_marker(path.name, expected_step),
                }
            )
    expected = [item for item in files if item["has_expected_step_marker"] and item["size_bytes"] > 0]
    return {
        "experiment_name": config.get("experiment_name", ""),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_dir_exists": checkpoint_dir.is_dir(),
        "expected_step": expected_step,
        "checkpoint_count": len(files),
        "expected_checkpoint_count": len(expected),
        "checkpoints": files,
        "ok": len(expected) >= 1,
    }


def write_reports(config: dict[str, Any], report: dict[str, Any]) -> None:
    root = report_dir(config)
    out_json = root / "checkpoint_inspection.json"
    out_md = root / "checkpoint_inspection.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.4A Mini40 Checkpoint Inspection",
        "",
        f"experiment: `{report['experiment_name']}`",
        f"checkpoint dir: `{report['checkpoint_dir']}`",
        f"checkpoint count: `{report['checkpoint_count']}`",
        f"expected step marker: `{report['expected_step']}`",
        f"expected checkpoint count: `{report['expected_checkpoint_count']}`",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        "",
        "| File | Size Bytes | Modified | Expected Step |",
        "|---|---:|---|---|",
    ]
    for item in report["checkpoints"]:
        lines.append(
            f"| `{item['path']}` | {item['size_bytes']} | `{item['modified_time']}` | `{item['has_expected_step_marker']}` |"
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect mini40 true-PBR checkpoint files without loading them.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    report = inspect_checkpoint_dir(config)
    write_reports(config, report)
    print("Phase 2L.4A mini40 checkpoint inspection")
    print(f"  checkpoint_dir: {report['checkpoint_dir']}")
    print(f"  checkpoint_count: {report['checkpoint_count']}")
    print(f"  expected_checkpoint_count: {report['expected_checkpoint_count']}")
    if report["ok"]:
        print("PHASE2L4A_MINI40_CHECKPOINT_INSPECTION_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
