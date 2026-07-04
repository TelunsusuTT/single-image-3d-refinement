#!/usr/bin/env python3
"""Create fixed-seed group-aware train/val/test splits for Data v2 frame panels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "val", "test"]
MEMBERSHIP_COLUMNS = ["split_config", "split", "item_id", "group_key", "local_glb_path", "selected_input_view"]


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_dir"])


def report_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["report_dir"])


def curated_manifest_csv(config: dict[str, Any]) -> Path:
    return output_dir(config) / f"{config['dataset_name']}_curated_manifest.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEMBERSHIP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def targets_from_config(split_config: dict[str, Any], include_total: bool) -> dict[str, int]:
    targets = {name: int(split_config[name]) for name in SPLITS}
    if include_total:
        declared_total = int(split_config.get("total", sum(targets.values())))
        if declared_total != sum(targets.values()):
            raise ValueError(f"split total {declared_total} does not match train/val/test sum {sum(targets.values())}")
    return targets


def grouped_rows(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group_key = row.get("group_key", "") or row.get("item_id", "")
        groups.setdefault(group_key, []).append(row)
    return list(groups.items())


def choose_split(remaining: dict[str, int], group_size: int) -> str | None:
    candidates = [name for name in SPLITS if remaining[name] >= group_size]
    if not candidates:
        return None
    return sorted(candidates, key=lambda name: (-remaining[name], SPLITS.index(name)))[0]


def allocate_rows(
    rows: list[dict[str, str]],
    targets: dict[str, int],
    seed: int,
    split_name: str,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    rng = random.Random(seed)
    groups = grouped_rows(rows)
    rng.shuffle(groups)
    assignments = {name: [] for name in SPLITS}
    remaining = dict(targets)
    split_group_keys: list[str] = []

    for group_key, group_rows in groups:
        if sum(remaining.values()) == 0:
            break
        group_rows = list(group_rows)
        rng.shuffle(group_rows)
        target_split = choose_split(remaining, len(group_rows))
        if target_split is not None:
            assignments[target_split].extend(group_rows)
            remaining[target_split] -= len(group_rows)
            continue
        split_group_keys.append(group_key)
        for row in group_rows:
            target_split = choose_split(remaining, 1)
            if target_split is None:
                break
            assignments[target_split].append(row)
            remaining[target_split] -= 1

    if any(count != 0 for count in remaining.values()):
        raise ValueError(f"{split_name} could not satisfy split sizes; remaining={remaining}")
    return assignments, split_group_keys


def validate_no_overlap(assignments: dict[str, list[dict[str, str]]], split_name: str) -> None:
    seen: dict[str, str] = {}
    for split, rows in assignments.items():
        for row in rows:
            item_id = row["item_id"]
            if item_id in seen:
                raise ValueError(f"{split_name}: item_id {item_id} appears in both {seen[item_id]} and {split}")
            seen[item_id] = split


def split_payload(split_name: str, assignments: dict[str, list[dict[str, str]]], split_group_keys: list[str], seed: int) -> dict[str, Any]:
    return {
        "split_name": split_name,
        "random_seed": seed,
        "counts": {split: len(rows) for split, rows in assignments.items()},
        "split_group_keys": split_group_keys,
        "splits": {
            split: [
                {
                    "item_id": row["item_id"],
                    "group_key": row["group_key"],
                    "local_glb_path": row["local_glb_path"],
                    "selected_input_view": row["selected_input_view"],
                    "contact_sheet_path": row.get("contact_sheet_path", ""),
                }
                for row in rows
            ]
            for split, rows in assignments.items()
        },
    }


def membership_rows(split_name: str, assignments: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        for row in assignments[split]:
            rows.append(
                {
                    "split_config": split_name,
                    "split": split,
                    "item_id": row["item_id"],
                    "group_key": row["group_key"],
                    "local_glb_path": row["local_glb_path"],
                    "selected_input_view": row["selected_input_view"],
                }
            )
    return rows


def make_splits(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = read_csv(curated_manifest_csv(config))
    seed = int(config.get("random_seed", 0))
    mini_targets = targets_from_config(config["mini_split"], include_total=True)
    mini_total = sum(mini_targets.values())
    if len(rows) < mini_total:
        raise ValueError(f"mini40 requires {mini_total} accepted assets, found {len(rows)}")

    full_targets = targets_from_config(config["full_split"], include_total=False)
    full_total = sum(full_targets.values())
    allow_smaller_full = bool(config.get("allow_smaller_full_split", False))
    if not allow_smaller_full and len(rows) != full_total:
        raise ValueError(f"full101 requires exactly {full_total} accepted assets, found {len(rows)}")
    if len(rows) < full_total:
        raise ValueError(f"full101 requires {full_total} accepted assets, found {len(rows)}")

    mini_assignments, mini_split_groups = allocate_rows(rows, mini_targets, seed, "mini40")
    full_assignments, full_split_groups = allocate_rows(rows, full_targets, seed + 1, "full101")
    validate_no_overlap(mini_assignments, "mini40")
    validate_no_overlap(full_assignments, "full101")

    mini_payload = split_payload("mini40", mini_assignments, mini_split_groups, seed)
    full_payload = split_payload("full101", full_assignments, full_split_groups, seed + 1)
    membership = membership_rows("mini40", mini_assignments) + membership_rows("full101", full_assignments)
    summary = {
        "dataset_name": config["dataset_name"],
        "curated_asset_count": len(rows),
        "mini40": {
            "counts": mini_payload["counts"],
            "split_group_keys": mini_split_groups,
        },
        "full101": {
            "counts": full_payload["counts"],
            "split_group_keys": full_split_groups,
        },
    }
    return {"mini40": mini_payload, "full101": full_payload, "summary": summary}, membership


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = output_dir(config)
    dataset = config["dataset_name"]
    return {
        "mini": root / f"{dataset}_mini40_split.json",
        "full": root / f"{dataset}_full101_split.json",
        "membership": root / f"{dataset}_split_membership.csv",
        "summary_md": report_dir(config) / "split_summary.md",
        "summary_json": report_dir(config) / "split_summary.json",
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Data v2 Frame Panels Split Summary",
        "",
        f"dataset: `{summary['dataset_name']}`",
        f"curated assets: `{summary['curated_asset_count']}`",
        "",
        "| Split Config | Train | Val | Test | Split Group Keys |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("mini40", "full101"):
        counts = summary[name]["counts"]
        lines.append(
            f"| `{name}` | {counts['train']} | {counts['val']} | {counts['test']} | {len(summary[name]['split_group_keys'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Data v2 frame-panel train/val/test splits.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    payloads, membership = make_splits(config)
    paths = output_paths(config)
    write_json(paths["mini"], payloads["mini40"])
    write_json(paths["full"], payloads["full101"])
    write_csv(paths["membership"], membership)
    write_json(paths["summary_json"], payloads["summary"])
    write_summary_md(paths["summary_md"], payloads["summary"])
    print("Phase 2L.2C frame-panel splits")
    print(f"  mini40: {payloads['mini40']['counts']}")
    print(f"  full101: {payloads['full101']['counts']}")
    print(f"  membership: {paths['membership']}")
    print("PHASE2L2C_FRAME_PANEL_SPLITS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
