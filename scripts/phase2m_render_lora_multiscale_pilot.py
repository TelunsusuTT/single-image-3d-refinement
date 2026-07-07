#!/usr/bin/env python3
"""Render Phase 2M.3B LoRA multi-scale pilot GLBs from fixed views in Blender."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot"
DEFAULT_PILOT_SUMMARY = DEFAULT_PILOT_ROOT / "pilot_summary.json"
DEFAULT_PER_VARIANT_OUTPUTS = DEFAULT_PILOT_ROOT / "per_variant_outputs.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_rendered"
DEFAULT_VIEW_IDS = ["000", "001", "002", "003", "004", "005"]
DEFAULT_VARIANTS = ["base", "lora_scale050", "lora_scale075", "lora_scale100"]
DEFAULT_BACKGROUND_COLOR = [0.28, 0.28, 0.28]
DEFAULT_RENDER_RESOLUTION = 512
DEFAULT_FRONT_VIEWS = ["004", "005"]


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def all_outputs_exist(output_dir: Path, view_ids: list[str]) -> bool:
    return all((output_dir / f"{view_id}.png").is_file() and (output_dir / f"{view_id}.png").stat().st_size > 0 for view_id in view_ids)


def reference_images_from_input(input_image_path: str | Path, view_ids: list[str]) -> dict[str, str]:
    input_path = resolve_project_path(input_image_path)
    render_cond = input_path.parent
    return {view_id: str((render_cond / f"{view_id}_light_AL.png").resolve()) for view_id in view_ids}


def variant_index(per_variant_outputs: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in per_variant_outputs.get("variants", []):
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", ""))
        variant = str(row.get("variant", ""))
        if case_id and variant:
            index[(case_id, variant)] = row
    return index


def build_render_config(
    pilot_summary: dict[str, Any],
    per_variant_outputs: dict[str, Any],
    output_root: Path,
    view_ids: list[str] | None = None,
    render_resolution: int = DEFAULT_RENDER_RESOLUTION,
    background_color: list[float] | None = None,
    variants: list[str] | None = None,
) -> dict[str, Any]:
    view_ids = list(view_ids or DEFAULT_VIEW_IDS)
    variants = list(variants or DEFAULT_VARIANTS)
    background_color = list(background_color or DEFAULT_BACKGROUND_COLOR)
    if pilot_summary.get("status") != "OK" or pilot_summary.get("success") is not True:
        raise ValueError("pilot_summary must have status OK and success true")
    cases = pilot_summary.get("cases", [])
    if len(cases) != 3:
        raise ValueError(f"expected exactly 3 M3B pilot cases, got {len(cases)}")
    indexed = variant_index(per_variant_outputs)
    rendered_cases: dict[str, Any] = {}
    for case in cases:
        case_id = str(case["case_id"])
        eval_split = str(case.get("eval_split", ""))
        refs = reference_images_from_input(case["input_image_path"], view_ids)
        variant_glbs: dict[str, str] = {}
        for variant in variants:
            row = indexed.get((case_id, variant))
            if row is None:
                raise ValueError(f"missing variant output for {case_id}/{variant}")
            glb = row.get("output_glb_path") or row.get("output_mesh_path")
            if not glb:
                raise ValueError(f"missing GLB path for {case_id}/{variant}")
            variant_glbs[variant] = str(resolve_project_path(glb))
        rendered_cases[case_id] = {
            "item_id": case_id,
            "eval_split": eval_split,
            "source_split": eval_split,
            "selected_input_view": str(case.get("selected_input_view", "005")),
            "primary_front_views": list(DEFAULT_FRONT_VIEWS),
            "reference_images": refs,
            "variant_glbs": variant_glbs,
            "base_glb": variant_glbs.get("base", ""),
            "lora_scale050_glb": variant_glbs.get("lora_scale050", ""),
            "lora_scale075_glb": variant_glbs.get("lora_scale075", ""),
            "lora_scale100_glb": variant_glbs.get("lora_scale100", ""),
        }
    return {
        "phase": "2M.3C",
        "experiment_name": "phase2m_lora_multiscale_pilot_rendered_eval",
        "pilot_summary_path": str(DEFAULT_PILOT_SUMMARY),
        "per_variant_outputs_path": str(DEFAULT_PER_VARIANT_OUTPUTS),
        "output_root": str(output_root.resolve()),
        "view_ids": view_ids,
        "render_resolution": int(render_resolution),
        "background_color": background_color,
        "primary_front_views": list(DEFAULT_FRONT_VIEWS),
        "variants": variants,
        "case_count": len(rendered_cases),
        "variant_count": len(variants),
        "cases": rendered_cases,
    }


def write_outputs(output_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = output_root / "render_results.csv"
    json_path = output_root / "render_summary.json"
    md_path = output_root / "render_summary.md"
    fields = ["case_id", "eval_split", "variant", "input_glb", "output_dir", "status"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    write_json(json_path, summary)
    lines = [
        "# Phase 2M.3C LoRA Pilot Render Summary",
        "",
        f"status: `{'OK' if summary['ok'] else 'FAIL'}`",
        f"case_count: `{summary['case_count']}`",
        f"variant_count: `{summary['variant_count']}`",
        f"rendered_variant_count: `{summary['rendered_variant_count']}`",
        f"skipped_variant_count: `{summary['skipped_variant_count']}`",
        "",
        "| Case | Split | Variant | Status | Output Dir |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['case_id']}` | `{row['eval_split']}` | `{row['variant']}` | `{row['status']}` | `{row['output_dir']}` |")
    if summary.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in summary["errors"])
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 2M.3B LoRA pilot GLBs from fixed Blender views.")
    parser.add_argument("--pilot-summary", type=Path, default=DEFAULT_PILOT_SUMMARY)
    parser.add_argument("--per-variant-outputs", type=Path, default=DEFAULT_PER_VARIANT_OUTPUTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--view-ids", nargs="+", default=list(DEFAULT_VIEW_IDS))
    parser.add_argument("--resolution", type=int, default=DEFAULT_RENDER_RESOLUTION)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)
    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pilot_summary = load_json(resolve_project_path(args.pilot_summary))
    per_variant_outputs = load_json(resolve_project_path(args.per_variant_outputs))
    render_config = build_render_config(
        pilot_summary,
        per_variant_outputs,
        output_root,
        view_ids=list(args.view_ids),
        render_resolution=args.resolution,
    )
    render_config["pilot_summary_path"] = str(resolve_project_path(args.pilot_summary))
    render_config["per_variant_outputs_path"] = str(resolve_project_path(args.per_variant_outputs))
    write_json(output_root / "render_eval_cases.json", render_config)

    scripts_root = PROJECT_ROOT / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    import bpy  # type: ignore
    from render_phase2k3_glb_views_blender import render_variant, write_render_config

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    errors: list[str] = []
    selected_cases = list(render_config["cases"].items())
    if args.limit is not None:
        selected_cases = selected_cases[: max(0, args.limit)]
    rendered_count = 0
    skipped_count = 0
    for case_id, case in selected_cases:
        eval_split = case["eval_split"]
        reports.setdefault(case_id, {})
        for variant in render_config["variants"]:
            input_glb = resolve_project_path(case["variant_glbs"][variant])
            output_dir = output_root / "renders" / eval_split / case_id / variant
            if args.only_missing and all_outputs_exist(output_dir, render_config["view_ids"]):
                status = "skipped_existing"
                skipped_count += 1
            else:
                try:
                    report = render_variant(
                        bpy,
                        input_glb,
                        output_dir,
                        render_config["view_ids"],
                        int(render_config["render_resolution"]),
                        list(render_config["background_color"]),
                    )
                    reports[case_id][variant] = report
                    status = "rendered"
                    rendered_count += 1
                except Exception as exc:  # pragma: no cover - Blender runtime path.
                    status = "failed"
                    errors.append(f"{case_id}/{variant}: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "case_id": case_id,
                    "eval_split": eval_split,
                    "variant": variant,
                    "input_glb": str(input_glb),
                    "output_dir": str(output_dir),
                    "status": status,
                }
            )
            print(f"{status}: {eval_split}/{case_id}/{variant}")
    write_render_config(output_root, render_config, reports)
    summary = {
        "phase": "2M.3C",
        "ok": not errors,
        "case_count": len(selected_cases),
        "variant_count": len(render_config["variants"]),
        "rendered_variant_count": rendered_count,
        "skipped_variant_count": skipped_count,
        "output_root": str(output_root),
        "rows": rows,
        "errors": errors,
    }
    write_outputs(output_root, rows, summary)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PHASE2M_M3C_RENDER_LORA_PILOT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
