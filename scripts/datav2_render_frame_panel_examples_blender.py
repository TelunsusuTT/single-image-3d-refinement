#!/usr/bin/env python3
"""Render mini40 frame-panel GLBs into Hunyuan3D-Paint-style examples.

Run manually with Blender:
  blender -b --python scripts/datav2_render_frame_panel_examples_blender.py -- \
    --config configs/datav2_frame_panels_mini40_render.json --limit 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blender_render_hy3dpaint_example import main as render_one_main  # noqa: E402


RESULT_COLUMNS = [
    "split",
    "item_id",
    "sample_name",
    "sample_dir",
    "qa_dir",
    "selected_input_view",
    "status",
    "error",
]
TEX_SUFFIXES = [".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"]
COND_SUFFIXES = ["_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"]


def argv_after_double_dash(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["report_root"])


def dataset_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_dataset_root"])


def render_plan_csv(config: dict[str, Any]) -> Path:
    return report_root(config) / "render_plan.csv"


def render_results_csv(config: dict[str, Any]) -> Path:
    return report_root(config) / "render_results.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def expected_files(sample_dir: Path, view_ids: list[str]) -> list[Path]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    files = [render_tex / "transforms.json"]
    for view_id in view_ids:
        files.extend(render_tex / f"{view_id}{suffix}" for suffix in TEX_SUFFIXES)
        files.extend(render_cond / f"{view_id}{suffix}" for suffix in COND_SUFFIXES)
    return files


def complete_example(sample_dir: Path, view_ids: list[str]) -> bool:
    return (sample_dir / "render_tex").is_dir() and (sample_dir / "render_cond").is_dir() and all(
        path.is_file() for path in expected_files(sample_dir, view_ids)
    )


def selected_plan_rows(rows: list[dict[str, str]], split: str, start_index: int, limit: int | None) -> list[dict[str, str]]:
    filtered = rows if split == "all" else [row for row in rows if row.get("split") == split]
    start = max(start_index, 0)
    if limit is None:
        return filtered[start:]
    return filtered[start : start + max(limit, 0)]


def result_for_row(row: dict[str, str], status: str, error: str = "") -> dict[str, str]:
    return {
        "split": row.get("split", ""),
        "item_id": row.get("item_id", ""),
        "sample_name": row.get("sample_name", row.get("item_id", "")),
        "sample_dir": row.get("output_example_dir", ""),
        "qa_dir": row.get("qa_dir", ""),
        "selected_input_view": row.get("selected_input_view", ""),
        "status": status,
        "error": error,
    }


def merge_results(existing: list[dict[str, str]], updates: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(row.get("split", ""), row.get("item_id", "")): row for row in existing}
    for row in updates:
        by_key[(row.get("split", ""), row.get("item_id", ""))] = row
    return list(by_key.values())


def write_summary(config: dict[str, Any], rows: list[dict[str, str]]) -> None:
    root = report_root(config)
    out_json = root / "render_summary.json"
    out_md = root / "render_summary.md"
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    payload = {"dataset_name": config["dataset_name"], "result_count": len(rows), "status_counts": counts, "rows": rows}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Data v2 Frame Panels Mini40 Render Summary",
        "",
        f"dataset: `{config['dataset_name']}`",
        f"results: `{len(rows)}`",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | {count} |")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Data v2 frame-panel mini40 Hunyuan examples in Blender.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv_after_double_dash(sys.argv) if argv is None else argv)
    config = load_json(args.config)
    plan_rows = read_csv(render_plan_csv(config))
    rows = selected_plan_rows(plan_rows, args.split, args.start_index, args.limit)
    view_ids = list(config.get("view_ids", ["000", "001", "002", "003", "004", "005"]))
    num_view = len(view_ids)
    resolution = int(config.get("render_resolution", 512))
    root = dataset_root(config)
    qa_root = report_root(config) / "qa"
    updates: list[dict[str, str]] = []

    for row in rows:
        sample_dir = resolve_project_path(row["output_example_dir"])
        if args.only_missing and complete_example(sample_dir, view_ids):
            updates.append(result_for_row(row, "skipped_complete"))
            continue
        try:
            exit_code = render_one_main(
                [
                    "--input-glb",
                    row["local_glb_path"],
                    "--sample-name",
                    row["sample_name"],
                    "--out-root",
                    str(root),
                    "--qa-root",
                    str(qa_root),
                    "--num-view",
                    str(num_view),
                    "--resolution",
                    str(resolution),
                ]
            )
        except Exception as exc:
            updates.append(result_for_row(row, "failed", repr(exc)))
            continue
        if exit_code == 0 and complete_example(sample_dir, view_ids):
            updates.append(result_for_row(row, "rendered"))
        else:
            updates.append(result_for_row(row, "failed", f"renderer_exit={exit_code} complete={complete_example(sample_dir, view_ids)}"))

    existing = read_csv(render_results_csv(config)) if render_results_csv(config).is_file() else []
    results = merge_results(existing, updates)
    write_csv(render_results_csv(config), results)
    write_summary(config, results)
    failed = [row for row in updates if row["status"] == "failed"]
    print("Phase 2L.3A frame-panel render examples")
    print(f"  selected rows: {len(rows)}")
    print(f"  updated results: {len(updates)}")
    print(f"  failed: {len(failed)}")
    print(f"  render_results: {render_results_csv(config)}")
    if failed:
        return 1
    print("PHASE2L3A_FRAME_PANEL_RENDER_EXAMPLES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
