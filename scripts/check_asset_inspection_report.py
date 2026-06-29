#!/usr/bin/env python3
"""Check a Blender asset inspection JSON report without importing Blender."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_report(report_json: Path) -> dict[str, Any]:
    if not report_json.exists():
        raise ValueError(f"report JSON missing: {report_json}")
    if not report_json.is_file():
        raise ValueError(f"report JSON is not a file: {report_json}")
    try:
        with report_json.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"report JSON is not valid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("report JSON must contain an object")
    return report


def total_uv_layers(report: dict[str, Any]) -> int:
    per_object = report.get("per_object", [])
    if not isinstance(per_object, list):
        return 0
    total = 0
    for item in per_object:
        if isinstance(item, dict):
            total += int(item.get("uv_layer_count", 0) or 0)
    return total


def check_report(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    if report.get("import_status") != "OK":
        failures.append("import_status is not OK")
    if int(report.get("mesh_object_count", 0) or 0) == 0:
        failures.append("mesh_object_count is 0")
    if int(report.get("total_polygons", 0) or 0) == 0:
        failures.append("total_polygons is 0")
    if total_uv_layers(report) == 0:
        failures.append("no UV layers found")
    if int(report.get("material_count", 0) or 0) == 0:
        failures.append("material_count is 0")
    if int(report.get("texture_image_count", 0) or 0) == 0:
        warnings.append("texture_image_count is 0")

    return failures, warnings


def print_summary(report: dict[str, Any], failures: list[str], warnings: list[str]) -> None:
    print(f"input_glb: {report.get('input_glb', '')}")
    print(f"import_status: {report.get('import_status', '')}")
    print(f"object_count: {report.get('object_count', 0)}")
    print(f"mesh_object_count: {report.get('mesh_object_count', 0)}")
    print(f"material_count: {report.get('material_count', 0)}")
    print(f"texture_image_count: {report.get('texture_image_count', 0)}")
    print(f"total_vertices: {report.get('total_vertices', 0)}")
    print(f"total_polygons: {report.get('total_polygons', 0)}")
    print(f"total_uv_layers: {total_uv_layers(report)}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if failures:
        print("status: FAIL")
        print("failures:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("status: PASS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a Blender asset inspection JSON report."
    )
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = load_report(args.report_json)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures, warnings = check_report(report)
    print_summary(report, failures, warnings)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
