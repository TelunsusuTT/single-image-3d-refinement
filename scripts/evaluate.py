#!/usr/bin/env python3
"""Validate, preview, or launch the fixed-view rerendering evaluation stages."""

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
    validate_evaluation_config,
)


STAGE_ORDER = ("render", "compare", "aggregate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper's fixed-view rerendering evaluation workflow."
    )
    parser.add_argument("--config", required=True, type=Path, help="Canonical evaluation JSON")
    parser.add_argument("--stage", choices=(*STAGE_ORDER, "all"), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate and print commands")
    mode.add_argument("--run", action="store_true", help="Execute the selected stages")
    parser.add_argument("--cases-config", type=Path, help="Evaluation case/variant manifest")
    parser.add_argument("--output-root", type=Path, help="Destination for rendered evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    try:
        config_path, config = load_json_object(args.config)
        runners = validate_evaluation_config(config, selected_stages=selected_stages)
        values = {
            "config": str(config_path),
            "cases_config": str(args.cases_config.resolve()) if args.cases_config else None,
            "output_root": str(args.output_root.resolve()) if args.output_root else None,
        }
        print("evaluation: Fixed-View Rerendering (fixed_view_rerendering)")
        for stage, runner in runners:
            command, working_directory = build_command(runner, values, preview=args.check_only)
            print_runner_summary(
                label=stage,
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
