#!/usr/bin/env python3
"""Validate, preview, or launch one canonical Paint-stage inference workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _workflow import (
    WorkflowError,
    build_command,
    execute_runner,
    load_json_object,
    print_runner_summary,
    validate_method_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-aligned fixed-mesh Hunyuan3D-Paint inference."
    )
    parser.add_argument("--config", required=True, type=Path, help="Canonical method JSON")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate and print the command")
    mode.add_argument("--run", action="store_true", help="Execute the configured runtime backend")
    parser.add_argument("--case-dir", type=Path, help="Case containing input/mesh.glb and input/image.png")
    parser.add_argument("--output-dir", type=Path, help="Destination for generated artifacts")
    parser.add_argument("--checkpoint", type=Path, help="Adaptation checkpoint when required")
    parser.add_argument("--run-id", help="Optional batch run identifier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path, config = load_json_object(args.config)
        method_id, display_name, runner = validate_method_config(config, stage="inference")
        values = {
            "config": str(config_path),
            "case_dir": str(args.case_dir.resolve()) if args.case_dir else None,
            "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "run_id": args.run_id,
        }
        command, working_directory = build_command(runner, values, preview=args.check_only)
        print(f"method: {display_name} ({method_id})")
        print_runner_summary(
            label="inference",
            runner=runner,
            command=command,
            working_directory=working_directory,
        )
        if args.run:
            execute_runner(runner=runner, command=command, working_directory=working_directory)
    except (WorkflowError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
