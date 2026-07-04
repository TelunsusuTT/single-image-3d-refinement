#!/usr/bin/env python3
"""Build a curated Data v2 frame-panel manifest from manual ABO review rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_COLUMNS = [
    "dataset_name",
    "item_id",
    "source",
    "relative_path",
    "url",
    "local_glb_path",
    "exists_local",
    "status",
    "human_decision",
    "reject_reason",
    "subclass",
    "selected_input_view",
    "alternative_input_view",
    "primary_eval_views",
    "front_quality_score",
    "texture_quality_score",
    "leakage_risk",
    "notes",
    "contact_sheet_page",
    "contact_sheet_path",
    "face_count",
    "mesh_count",
    "material_count",
    "texture_image_count",
    "bbox_extents",
    "group_key",
]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURATED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_reject_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    reject_ids: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            reject_ids.add(line)
    return reject_ids


def split_eval_views(values: Any) -> str:
    if isinstance(values, list):
        return ";".join(str(value) for value in values)
    return str(values)


def parse_bbox_extents(row: dict[str, str]) -> tuple[float, float, float]:
    text = row.get("bbox_extents", "")
    values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(values) >= 3:
        return values[0], values[1], values[2]
    fallback: list[float] = []
    for key in ("bbox_extent_x", "bbox_extent_y", "bbox_extent_z"):
        try:
            fallback.append(float(row.get(key, "")))
        except ValueError:
            fallback.append(0.0)
    return tuple(fallback[:3])  # type: ignore[return-value]


def int_value(value: str) -> int:
    try:
        return int(float(value))
    except ValueError:
        return 0


def group_key_for_row(row: dict[str, str]) -> str:
    extents = parse_bbox_extents(row)
    rounded = tuple(round(value, 1) for value in extents)
    sorted_extents = sorted(abs(value) for value in extents)
    mid = sorted_extents[1] if len(sorted_extents) > 1 else 0.0
    large = sorted_extents[2] if len(sorted_extents) > 2 else 0.0
    aspect = round(large / mid, 1) if mid > 0 else 0.0
    face_count = int_value(row.get("face_count", ""))
    face_bin = int(math.floor(face_count / 5000.0)) if face_count > 0 else 0
    return "bbox={:.1f},{:.1f},{:.1f}|face_bin={}|aspect={:.1f}".format(
        rounded[0],
        rounded[1],
        rounded[2],
        face_bin,
        aspect,
    )


def technical_pass(row: dict[str, str]) -> bool:
    return (
        row.get("status", "").strip().lower() == "found"
        and truthy(row.get("exists_local", ""))
        and row.get("import_ok", "").strip().lower() == "yes"
        and row.get("render_ok", "").strip().lower() == "yes"
    )


def explicitly_rejected(row: dict[str, str]) -> bool:
    return row.get("human_decision", "").strip().lower() in {"reject", "rejected", "no"}


def curated_row(config: dict[str, Any], row: dict[str, str]) -> dict[str, str]:
    default_subclass = str(config.get("default_subclass", "framed_wall_art"))
    selected = str(config.get("default_selected_input_view", "005"))
    alternative = str(config.get("default_alternative_input_view", "004"))
    primary_eval = split_eval_views(config.get("default_primary_eval_views", ["004", "005"]))
    output = {column: "" for column in CURATED_COLUMNS}
    output.update(
        {
            "dataset_name": str(config["dataset_name"]),
            "item_id": row.get("item_id", ""),
            "source": "ABO",
            "relative_path": row.get("relative_path", ""),
            "url": row.get("url", ""),
            "local_glb_path": row.get("local_glb_path", ""),
            "exists_local": row.get("exists_local", ""),
            "status": row.get("status", ""),
            "human_decision": "accept",
            "reject_reason": "",
            "subclass": row.get("subclass") or default_subclass,
            "selected_input_view": row.get("selected_input_view") or selected,
            "alternative_input_view": row.get("alternative_input_view") or alternative,
            "primary_eval_views": row.get("primary_eval_views") or primary_eval,
            "front_quality_score": row.get("front_quality_score", ""),
            "texture_quality_score": row.get("texture_quality_score", ""),
            "leakage_risk": row.get("leakage_risk", ""),
            "notes": row.get("notes", ""),
            "contact_sheet_page": row.get("contact_sheet_page", ""),
            "contact_sheet_path": row.get("contact_sheet_path", ""),
            "face_count": row.get("face_count", ""),
            "mesh_count": row.get("mesh_count", ""),
            "material_count": row.get("material_count", ""),
            "texture_image_count": row.get("texture_image_count", ""),
            "bbox_extents": row.get("bbox_extents", ""),
        }
    )
    output["group_key"] = group_key_for_row(output)
    return output


def output_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_dir"])


def report_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["report_dir"])


def manifest_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    base = output_dir(config) / f"{config['dataset_name']}_curated_manifest"
    return base.with_suffix(".csv"), base.with_suffix(".json")


def summary_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    root = report_dir(config)
    return root / "curated_manifest_summary.md", root / "curated_manifest_summary.json"


def build_curated_rows(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = read_csv(resolve_project_path(config["input_review_csv"]))
    reject_ids = read_reject_ids(resolve_project_path(config.get("optional_reject_ids", "")))
    curated: list[dict[str, str]] = []
    counts = {
        "input_rows": len(rows),
        "technical_fail_rows": 0,
        "optional_reject_rows": 0,
        "human_reject_rows": 0,
        "curated_rows": 0,
    }
    for row in rows:
        item_id = row.get("item_id", "")
        if item_id in reject_ids:
            counts["optional_reject_rows"] += 1
            continue
        if explicitly_rejected(row):
            counts["human_reject_rows"] += 1
            continue
        if not technical_pass(row):
            counts["technical_fail_rows"] += 1
            continue
        curated.append(curated_row(config, row))
    counts["curated_rows"] = len(curated)
    return curated, counts


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary(config: dict[str, Any], rows: list[dict[str, str]], counts: dict[str, int]) -> None:
    summary_md, summary_json = summary_paths(config)
    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[row["group_key"]] = group_counts.get(row["group_key"], 0) + 1
    payload = {
        "dataset_name": config["dataset_name"],
        "counts": counts,
        "unique_group_count": len(group_counts),
        "largest_group_size": max(group_counts.values(), default=0),
        "group_counts": group_counts,
    }
    write_json(summary_json, payload)
    lines = [
        "# Data v2 Frame Panels Curated Manifest",
        "",
        f"dataset: `{config['dataset_name']}`",
        f"input rows: `{counts['input_rows']}`",
        f"curated rows: `{counts['curated_rows']}`",
        f"technical fail rows: `{counts['technical_fail_rows']}`",
        f"optional reject rows: `{counts['optional_reject_rows']}`",
        f"human reject rows: `{counts['human_reject_rows']}`",
        f"unique group keys: `{len(group_counts)}`",
        f"largest group size: `{payload['largest_group_size']}`",
    ]
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Data v2 framed-panel curated manifest.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    rows, counts = build_curated_rows(config)
    out_csv, out_json = manifest_paths(config)
    write_csv(out_csv, rows)
    write_json(out_json, {"dataset_name": config["dataset_name"], "rows": rows, "counts": counts})
    write_summary(config, rows, counts)
    print("Phase 2L.2C frame-panel curated manifest")
    print(f"  curated rows: {len(rows)}")
    print(f"  out_csv: {out_csv}")
    print("PHASE2L2C_FRAME_PANEL_CURATED_MANIFEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
