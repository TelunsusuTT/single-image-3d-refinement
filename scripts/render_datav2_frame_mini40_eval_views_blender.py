#!/usr/bin/env python3
"""Render mini40 base/fine GLBs from fixed cameras inside Blender."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def all_outputs_exist(output_dir: Path, view_ids: list[str]) -> bool:
    return all((output_dir / f"{view_id}.png").is_file() for view_id in view_ids)


def write_outputs(render_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = render_root / "render_results.csv"
    json_path = render_root / "render_summary.json"
    md_path = render_root / "render_summary.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "eval_split", "variant", "input_glb", "output_dir", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.5B Mini40 Render Summary",
        "",
        f"case_count: `{summary['case_count']}`",
        f"rendered_variant_count: `{summary['rendered_variant_count']}`",
        f"skipped_variant_count: `{summary['skipped_variant_count']}`",
        "",
        "| Item ID | Eval Split | Variant | Status | Output Dir |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['item_id']}` | `{row['eval_split']}` | `{row['variant']}` | `{row['status']}` | `{row['output_dir']}` |"
        )
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render mini40 eval GLB views in Blender.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)
    config = load_json(args.config)
    root = resolve_project_path(config["output_root"])
    render_root = root / "render_eval"
    render_config = load_json(render_root / "render_eval_cases.json")
    view_ids = list(render_config.get("view_ids", ["000", "001", "002", "003", "004", "005"]))
    resolution = int(render_config.get("render_resolution", 512))
    background_color = list(render_config.get("background_color", [0.28, 0.28, 0.28]))

    import bpy  # type: ignore
    from render_phase2k3_glb_views_blender import render_variant, write_render_config

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    selected_cases = list(render_config.get("cases", {}).items())
    if args.limit is not None:
        selected_cases = selected_cases[: max(0, args.limit)]
    rendered_count = 0
    skipped_count = 0
    for item_id, case in selected_cases:
        eval_split = case.get("eval_split", "")
        reports.setdefault(item_id, {})
        for variant, key in (("base", "base_glb"), ("finetuned", "finetuned_glb")):
            output_dir = render_root / "renders" / eval_split / item_id / variant
            input_glb = resolve_project_path(case[key])
            if args.only_missing and all_outputs_exist(output_dir, view_ids):
                status = "skipped_existing"
                skipped_count += 1
            else:
                report = render_variant(bpy, input_glb, output_dir, view_ids, resolution, background_color)
                reports[item_id][variant] = report
                status = "rendered"
                rendered_count += 1
            rows.append(
                {
                    "item_id": item_id,
                    "eval_split": eval_split,
                    "variant": variant,
                    "input_glb": str(input_glb),
                    "output_dir": str(output_dir),
                    "status": status,
                }
            )
            print(f"{status}: {eval_split}/{item_id}/{variant}")
    write_render_config(render_root, render_config, reports)
    summary = {
        "case_count": len(selected_cases),
        "rendered_variant_count": rendered_count,
        "skipped_variant_count": skipped_count,
        "rows": rows,
    }
    write_outputs(render_root, rows, summary)
    print("PHASE2L5B_MINI40_RENDERED_VIEWS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
