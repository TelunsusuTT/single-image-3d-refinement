#!/usr/bin/env python3
"""Summarize Hunyuan3D-Paint training example packaging."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from check_hy3dpaint_example import check_examples


COUNT_KEYS = ("condition", "albedo", "metallic_roughness", "normal", "position")


def aggregate_counts(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    aggregate: dict[str, dict[str, int]] = {}
    for key in COUNT_KEYS:
        values = [int(result["counts"][key]) for result in results]
        aggregate[key] = {"min": min(values), "max": max(values)}
    return aggregate


def build_summary(
    examples_json: Path, num_view: int, results: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "examples_json": str(examples_json),
        "num_view": num_view,
        "num_samples": len(results),
        "samples": results,
        "aggregate": {
            "counts": aggregate_counts(results),
            "num_ok": sum(1 for result in results if result["ok"]),
            "num_failed": sum(1 for result in results if not result["ok"]),
        },
    }


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_dir",
        "sample_exists",
        "render_cond_exists",
        "render_tex_exists",
        "transforms_json_exists",
        "transforms_json_path",
        "condition_count",
        "albedo_count",
        "metallic_roughness_count",
        "normal_count",
        "position_count",
        "ok",
        "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            counts = result["counts"]
            writer.writerow(
                {
                    "sample_dir": result["sample_dir"],
                    "sample_exists": result["sample_exists"],
                    "render_cond_exists": result["render_cond_exists"],
                    "render_tex_exists": result["render_tex_exists"],
                    "transforms_json_exists": result["transforms_json_exists"],
                    "transforms_json_path": result["transforms_json_path"],
                    "condition_count": counts["condition"],
                    "albedo_count": counts["albedo"],
                    "metallic_roughness_count": counts["metallic_roughness"],
                    "normal_count": counts["normal"],
                    "position_count": counts["position"],
                    "ok": result["ok"],
                    "errors": "; ".join(result["errors"]),
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write JSON and CSV summaries for Hunyuan3D-Paint examples."
    )
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--num-view", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_view < 1:
        print("ERROR: --num-view must be >= 1", file=sys.stderr)
        return 2

    try:
        results = check_examples(args.examples_json, args.num_view, strict=False)
        summary = build_summary(args.examples_json, args.num_view, results)
        write_json(args.out_json, summary)
        write_csv(args.out_csv, results)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failed = summary["aggregate"]["num_failed"]
    print(f"Wrote JSON summary: {args.out_json}")
    print(f"Wrote CSV summary: {args.out_csv}")
    print(f"Summarized {summary['num_samples']} sample(s): {failed} with issues recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
