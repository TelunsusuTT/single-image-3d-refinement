#!/usr/bin/env python3
"""Static readiness checks for mini40 true-PBR evaluation inference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_VIEW_IDS = {"000", "001", "002", "003", "004", "005"}
PREFERRED_INPUT_VIEWS = {"004", "005"}
FORBIDDEN_MARKERS = ["pilot_v1_overfit_500", "pilot_v1_conservative_50_lr1e6", "sd2-community"]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def has_step_marker(filename: str, expected_step: str) -> bool:
    return re.search(rf"step[=_-]?{re.escape(expected_step)}(?:\D|$)", filename) is not None


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {path}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {path}")
    return {"path": str(path), "exists": exists, "size_bytes": size}


def scan_forbidden(config: dict[str, Any], eval_cases: dict[str, Any], errors: list[str]) -> list[str]:
    text = json.dumps({"config": config, "eval_cases": eval_cases}, sort_keys=True)
    hits = [marker for marker in FORBIDDEN_MARKERS if marker in text]
    for marker in hits:
        errors.append(f"forbidden old baseline/checkpoint marker referenced: {marker}")
    return hits


def readiness_report(config_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    config = load_json(config_path)
    root = resolve_project_path(config["output_root"])
    eval_cases_path = root / "eval_cases.json"
    eval_cases = load_json(eval_cases_path) if eval_cases_path.is_file() else {"cases": []}

    checkpoint = resolve_project_path(config["checkpoint_path"])
    checkpoint_info = check_file(checkpoint, "checkpoint", errors)
    expected_step = str(config.get("checkpoint_expected_step", "500"))
    if checkpoint_info["exists"] and not has_step_marker(checkpoint.name, expected_step):
        errors.append(f"checkpoint filename does not contain expected step marker {expected_step}: {checkpoint.name}")

    override = check_file(resolve_project_path(config["input_view_override_csv"]), "input view override CSV", errors)
    eval_cases_info = check_file(eval_cases_path, "eval_cases.json", errors)
    wrapper = check_file(PROJECT_ROOT / "scripts" / "run_phase2g_paint_infer.py", "Hunyuan inference wrapper", errors)
    texture_compare = check_file(PROJECT_ROOT / "scripts" / "make_phase2g6_texture_comparison.py", "texture comparison script", errors)

    case_reports = []
    for case in eval_cases.get("cases", []):
        item_id = case.get("item_id", "")
        selected_view = str(case.get("selected_input_view", ""))
        if selected_view not in VALID_VIEW_IDS:
            errors.append(f"{item_id}: selected_input_view is invalid: {selected_view}")
        elif selected_view not in PREFERRED_INPUT_VIEWS:
            warnings.append(f"{item_id}: selected_input_view is not 004/005: {selected_view}")
        mesh_info = check_file(resolve_project_path(case.get("local_mesh_path", "")), f"{item_id} mesh", errors)
        selected_info = check_file(resolve_project_path(case.get("selected_input_image", "")), f"{item_id} selected input image", errors)
        case_mesh_info = check_file(resolve_project_path(case.get("case_input_mesh", "")), f"{item_id} case input mesh", errors)
        case_image_info = check_file(resolve_project_path(case.get("case_input_image", "")), f"{item_id} case input image", errors)
        reference_infos = {
            view_id: check_file(resolve_project_path(path), f"{item_id} reference view {view_id}", errors)
            for view_id, path in case.get("reference_images", {}).items()
        }
        case_reports.append(
            {
                "item_id": item_id,
                "eval_split": case.get("eval_split", ""),
                "selected_input_view": selected_view,
                "mesh": mesh_info,
                "selected_input_image": selected_info,
                "case_input_mesh": case_mesh_info,
                "case_input_image": case_image_info,
                "reference_images": reference_infos,
            }
        )

    base_dir = root / "infer" / "base"
    fine_dir = root / "infer" / "finetuned"
    unsafe_output_hits = [marker for marker in FORBIDDEN_MARKERS if marker in str(base_dir) or marker in str(fine_dir)]
    for marker in unsafe_output_hits:
        errors.append(f"inference output dir references forbidden marker: {marker}")
    root.mkdir(parents=True, exist_ok=True)
    forbidden_hits = scan_forbidden(config, eval_cases, errors)
    report = {
        "config_path": str(config_path.resolve()),
        "output_root": str(root),
        "checkpoint": checkpoint_info,
        "expected_step": expected_step,
        "override_csv": override,
        "eval_cases_json": eval_cases_info,
        "wrapper": wrapper,
        "texture_compare_script": texture_compare,
        "base_output_dir": str(base_dir),
        "finetuned_output_dir": str(fine_dir),
        "case_count": len(eval_cases.get("cases", [])),
        "cases": case_reports,
        "warnings": warnings,
        "forbidden_hits": forbidden_hits,
        "errors": errors,
        "ok": not errors,
    }
    return report


def write_report(report: dict[str, Any]) -> None:
    root = Path(report["output_root"])
    out_json = root / "readiness_summary.json"
    out_md = root / "readiness_summary.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.5A Mini40 Eval Readiness",
        "",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        f"case_count: `{report['case_count']}`",
        f"checkpoint: `{report['checkpoint']['path']}`",
        f"override CSV: `{report['override_csv']['path']}`",
        "",
        "| Item ID | Eval Split | Selected Input View |",
        "|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(f"| `{case['item_id']}` | `{case['eval_split']}` | `{case['selected_input_view']}` |")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in report["warnings"])
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in report["errors"])
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check mini40 eval readiness.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = readiness_report(args.config)
    write_report(report)
    print("Phase 2L.5A mini40 eval readiness")
    print(f"  output_root: {report['output_root']}")
    print(f"  case_count: {report['case_count']}")
    print(f"  warnings: {len(report['warnings'])}")
    print(f"  errors: {len(report['errors'])}")
    if report["ok"]:
        print("PHASE2L5A_MINI40_EVAL_READINESS_OK")
        return 0
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
