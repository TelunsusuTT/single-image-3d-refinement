#!/usr/bin/env python3
"""Build a deterministic mini40 render plan for Data v2 frame panels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_COLUMNS = [
    "split",
    "item_id",
    "local_glb_path",
    "selected_input_view",
    "sample_name",
    "output_example_dir",
    "qa_dir",
]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def report_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["report_root"])


def dataset_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_dataset_root"])


def render_plan_csv(config: dict[str, Any]) -> Path:
    return report_root(config) / "render_plan.csv"


def split_rows(split_payload: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[str, dict[str, str]]] = []
    for split in ("train", "val", "test"):
        for row in split_payload.get("splits", {}).get(split, []):
            rows.append((split, {key: str(value) for key, value in row.items()}))
    return rows


def index_by_item(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("item_id", ""): row for row in rows if row.get("item_id")}


def build_rows(config: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    split_payload = load_json(resolve_project_path(config["split_file"]))
    curated = index_by_item(read_csv(resolve_project_path(config["curated_manifest_csv"])))
    root = dataset_root(config)
    qa_root = report_root(config) / "qa"
    selected_field = config.get("selected_input_view_field", "selected_input_view")
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for split, split_row in split_rows(split_payload):
        item_id = split_row.get("item_id", "")
        curated_row = curated.get(item_id, {})
        local_glb = resolve_project_path(split_row.get("local_glb_path") or curated_row.get("local_glb_path", ""))
        if not local_glb.is_file() or local_glb.stat().st_size <= 0:
            missing.append(f"{item_id}: {local_glb}")
        selected_input_view = split_row.get(str(selected_field)) or curated_row.get(str(selected_field), "")
        rows.append(
            {
                "split": split,
                "item_id": item_id,
                "local_glb_path": str(local_glb),
                "selected_input_view": selected_input_view,
                "sample_name": item_id,
                "output_example_dir": str(root / item_id),
                "qa_dir": str(qa_root / item_id),
            }
        )
    return rows, missing


def split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


def write_reports(config: dict[str, Any], rows: list[dict[str, str]], missing: list[str]) -> None:
    root = report_root(config)
    summary_json = root / "render_plan_summary.json"
    summary_md = root / "render_plan_summary.md"
    counts = split_counts(rows)
    payload = {
        "dataset_name": config["dataset_name"],
        "render_plan_csv": str(render_plan_csv(config)),
        "asset_count": len(rows),
        "split_counts": counts,
        "missing_glb_count": len(missing),
        "missing_glbs": missing,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Data v2 Frame Panels Mini40 Render Plan",
        "",
        f"dataset: `{config['dataset_name']}`",
        f"assets: `{len(rows)}`",
        f"train: `{counts.get('train', 0)}`",
        f"val: `{counts.get('val', 0)}`",
        f"test: `{counts.get('test', 0)}`",
        f"missing GLBs: `{len(missing)}`",
    ]
    if missing:
        lines.extend(["", "## Missing GLBs", ""])
        lines.extend(f"- `{entry}`" for entry in missing)
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Data v2 frame-panel mini40 render plan.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    rows, missing = build_rows(config)
    out_csv = render_plan_csv(config)
    write_csv(out_csv, rows)
    write_reports(config, rows, missing)
    print("Phase 2L.3A frame-panel render plan")
    print(f"  assets: {len(rows)}")
    print(f"  missing GLBs: {len(missing)}")
    print(f"  render_plan: {out_csv}")
    if missing:
        print(f"ERROR: missing GLBs; first missing: {missing[0]}", file=sys.stderr)
        return 1
    print("PHASE2L3A_FRAME_PANEL_RENDER_PLAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
