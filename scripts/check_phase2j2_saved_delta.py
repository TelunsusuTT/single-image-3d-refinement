#!/usr/bin/env python3
"""Check Phase 2J.2 saved checkpoint delta report threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2J.2 saved checkpoint delta.")
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--max-mean-abs-delta", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report_json.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoints = report.get("checkpoints", [])
    if len(checkpoints) != 1:
        print(f"ERROR: expected exactly one checkpoint report, found {len(checkpoints)}")
        return 1
    item = checkpoints[0]
    status = item.get("status")
    mean_delta = item.get("mean_of_mean_abs_delta")
    if status != "OK":
        print(f"ERROR: checkpoint comparison status is not OK: {status}")
        return 1
    if mean_delta is None:
        print("ERROR: mean_of_mean_abs_delta missing")
        return 1
    if float(mean_delta) > args.max_mean_abs_delta:
        print(
            "ERROR: mean_of_mean_abs_delta exceeds threshold: "
            f"{mean_delta} > {args.max_mean_abs_delta}"
        )
        return 1
    print("Phase 2J.2 saved checkpoint delta")
    print(f"  report_json: {report_path}")
    print(f"  checkpoint_name: {item.get('name')}")
    print(f"  mean_of_mean_abs_delta: {mean_delta}")
    print(f"  max_mean_abs_delta: {args.max_mean_abs_delta}")
    print("PHASE2J2_SAVED_DELTA_TINY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
