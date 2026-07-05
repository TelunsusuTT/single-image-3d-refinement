#!/usr/bin/env python3
"""Static readiness checks for mini40 rendered-view evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER_BIN = "/vol/bitbucket/ct1022/tools/bin/blender"
FORBIDDEN_OUTPUT_MARKERS = ["pilot_v1_overfit_500", "pilot_v1_conservative_50_lr1e6", "sd2-community"]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {path}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {path}")
    return {"path": str(path), "exists": exists, "size_bytes": size}


def blender_bin(config: dict[str, Any]) -> Path:
    value = os.environ.get("BLENDER_BIN") or config.get("blender_bin") or DEFAULT_BLENDER_BIN
    return resolve_project_path(str(value))


def readiness_report(config_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    config = load_json(config_path)
    root = resolve_project_path(config["output_root"])
    render_root = root / "render_eval"
    cases_path = render_root / "render_eval_cases.json"
    cases_info = check_file(cases_path, "render_eval_cases.json", errors)
    render_config = load_json(cases_path) if cases_path.is_file() else {"cases": {}}
    blender_info = check_file(blender_bin(config), "Blender binary", errors)
    output_dirs = {
        "renders": str(render_root / "renders"),
        "metrics": str(render_root / "metrics"),
        "boards": str(render_root / "boards"),
        "reports": str(render_root / "reports"),
    }
    for label, path_text in output_dirs.items():
        for marker in FORBIDDEN_OUTPUT_MARKERS:
            if marker in path_text:
                errors.append(f"{label} output path contains forbidden marker {marker}: {path_text}")

    case_reports = []
    for item_id, case in render_config.get("cases", {}).items():
        base_info = check_file(resolve_project_path(case.get("base_glb", "")), f"{item_id} base GLB", errors)
        fine_info = check_file(resolve_project_path(case.get("finetuned_glb", "")), f"{item_id} finetuned GLB", errors)
        refs = {
            view_id: check_file(resolve_project_path(path), f"{item_id} reference {view_id}", errors)
            for view_id, path in case.get("reference_images", {}).items()
        }
        case_reports.append(
            {
                "item_id": item_id,
                "eval_split": case.get("eval_split", ""),
                "selected_input_view": case.get("selected_input_view", ""),
                "base_glb": base_info,
                "finetuned_glb": fine_info,
                "reference_images": refs,
            }
        )
    render_root.mkdir(parents=True, exist_ok=True)
    return {
        "config_path": str(config_path.resolve()),
        "render_eval_root": str(render_root),
        "render_eval_cases": cases_info,
        "blender_bin": blender_info,
        "case_count": len(case_reports),
        "output_dirs": output_dirs,
        "cases": case_reports,
        "errors": errors,
        "ok": not errors,
    }


def write_report(report: dict[str, Any]) -> None:
    root = Path(report["render_eval_root"])
    out_json = root / "render_eval_readiness_summary.json"
    out_md = root / "render_eval_readiness_summary.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.5B Render-Eval Readiness",
        "",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        f"case_count: `{report['case_count']}`",
        f"blender_bin: `{report['blender_bin']['path']}`",
        "",
        "| Item ID | Eval Split | Selected Input View |",
        "|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(f"| `{case['item_id']}` | `{case['eval_split']}` | `{case['selected_input_view']}` |")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in report["errors"])
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check mini40 render-eval readiness.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = readiness_report(args.config)
    write_report(report)
    print("Phase 2L.5B mini40 render-eval readiness")
    print(f"  case_count: {report['case_count']}")
    print(f"  errors: {len(report['errors'])}")
    if report["ok"]:
        print("PHASE2L5B_MINI40_RENDER_EVAL_READINESS_OK")
        return 0
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
