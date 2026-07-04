#!/usr/bin/env python3
"""Create mini40 evaluation cases with explicit selected input views."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CONFIG = "mini40"


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def by_item(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("item_id", ""): row for row in rows if row.get("item_id")}


def select_assets(config: dict[str, Any], split_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    split_config = str(config.get("split_config", DEFAULT_SPLIT_CONFIG))
    rows = [row for row in split_rows if row.get("split_config", split_config) == split_config]
    eval_splits = set(config.get("eval_splits", ["val", "test"]))
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") in eval_splits:
            selected.append({**row, "source_split": row.get("split", ""), "eval_split": row.get("split", "")})
    train_count = int(config.get("optional_train_sanity_count", 0))
    for row in [row for row in rows if row.get("split") == "train"][:train_count]:
        selected.append({**row, "source_split": "train", "eval_split": "train_sanity"})
    return selected


def nonempty(*values: str | None) -> str:
    for value in values:
        if value:
            text = str(value).strip()
            if text:
                return text
    return ""


def selected_view_for(
    item_id: str,
    config: dict[str, Any],
    curated: dict[str, dict[str, str]],
    overrides: dict[str, dict[str, str]],
) -> tuple[str, str]:
    override_view = overrides.get(item_id, {}).get("selected_input_view", "")
    curated_view = curated.get(item_id, {}).get("selected_input_view", "")
    default_view = str(config.get("default_selected_input_view", "005"))
    if override_view.strip():
        return override_view.strip(), "override_csv"
    if curated_view.strip():
        return curated_view.strip(), "curated_manifest"
    return default_view, "config_default"


def primary_eval_views_for(item_id: str, config: dict[str, Any], curated: dict[str, dict[str, str]], overrides: dict[str, dict[str, str]]) -> list[str]:
    raw = nonempty(
        overrides.get(item_id, {}).get("primary_eval_views"),
        curated.get(item_id, {}).get("primary_eval_views"),
        ";".join(config.get("primary_front_views", ["004", "005"])),
    )
    return [part.strip() for part in raw.replace(",", ";").split(";") if part.strip()]


def mesh_path_for(row: dict[str, str], curated_row: dict[str, str]) -> str:
    return nonempty(row.get("local_glb_path"), curated_row.get("local_glb_path"))


def link_input(source: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    if dest.exists():
        return "existing_file"
    if source.is_file():
        os.symlink(source, dest)
        return "symlink"
    return "missing_source"


def build_cases(config: dict[str, Any]) -> dict[str, Any]:
    split_rows = read_csv(resolve_project_path(config["split_membership_csv"]))
    curated = by_item(read_csv(resolve_project_path(config["curated_manifest_csv"])))
    overrides = by_item(read_csv(resolve_project_path(config["input_view_override_csv"])))
    assets = select_assets(config, split_rows)
    sample_root = resolve_project_path(config["train_examples_root"])
    root = resolve_project_path(config["output_root"])
    view_ids = list(config.get("eval_view_ids", ["000", "001", "002", "003", "004", "005"]))
    cases = []
    for row in assets:
        item_id = row["item_id"]
        curated_row = curated.get(item_id, {})
        selected_view, selected_source = selected_view_for(item_id, config, curated, overrides)
        render_cond = sample_root / item_id / "render_cond"
        mesh_path = resolve_project_path(mesh_path_for(row, curated_row))
        selected_input_image = render_cond / f"{selected_view}_light_AL.png"
        eval_split = row.get("eval_split", row.get("split", ""))
        case_dir = root / "cases" / eval_split / item_id
        mesh_link = link_input(mesh_path, case_dir / "input" / "mesh.glb")
        image_link = link_input(selected_input_image, case_dir / "input" / "image.png")
        references = {view_id: str(render_cond / f"{view_id}_light_AL.png") for view_id in view_ids}
        cases.append(
            {
                "item_id": item_id,
                "source_split": row.get("source_split", row.get("split", "")),
                "eval_split": eval_split,
                "local_mesh_path": str(mesh_path),
                "case_dir": str(case_dir),
                "case_input_mesh": str(case_dir / "input" / "mesh.glb"),
                "case_input_image": str(case_dir / "input" / "image.png"),
                "case_mesh_link_mode": mesh_link,
                "case_image_link_mode": image_link,
                "selected_input_view": selected_view,
                "selected_input_view_source": selected_source,
                "selected_input_image": str(selected_input_image),
                "primary_eval_views": primary_eval_views_for(item_id, config, curated, overrides),
                "reference_images": references,
            }
        )
    return {
        "experiment_name": config.get("experiment_name", ""),
        "output_root": str(root),
        "case_count": len(cases),
        "cases": cases,
    }


def write_outputs(config: dict[str, Any], data: dict[str, Any]) -> None:
    root = resolve_project_path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "eval_cases.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    summary = {
        "experiment_name": data["experiment_name"],
        "case_count": data["case_count"],
        "cases": [
            {
                "item_id": case["item_id"],
                "source_split": case["source_split"],
                "eval_split": case["eval_split"],
                "selected_input_view": case["selected_input_view"],
                "selected_input_view_source": case["selected_input_view_source"],
                "selected_input_image": case["selected_input_image"],
            }
            for case in data["cases"]
        ],
    }
    (root / "eval_cases_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 2L.5A Mini40 Eval Cases",
        "",
        f"case_count: `{data['case_count']}`",
        "",
        "| Item ID | Source Split | Eval Split | Selected Input View | Source | Selected Input Image |",
        "|---|---|---|---|---|---|",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| `{case['item_id']}` | `{case['source_split']}` | `{case['eval_split']}` | "
            f"`{case['selected_input_view']}` | `{case['selected_input_view_source']}` | `{case['selected_input_image']}` |"
        )
    (root / "eval_cases_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mini40 eval cases.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    data = build_cases(config)
    write_outputs(config, data)
    print("Phase 2L.5A mini40 eval cases")
    print(f"  case_count: {data['case_count']}")
    print(f"  eval_cases: {resolve_project_path(config['output_root']) / 'eval_cases.json'}")
    if data["case_count"] > 0:
        print("PHASE2L5A_MINI40_EVAL_CASES_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
