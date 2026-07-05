#!/usr/bin/env python3
"""Build absolute examples JSON files for rendered Data v2 frame-panel examples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUCCESS_STATUSES = {"rendered", "skipped_complete", "complete"}


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


def render_results_csv(config: dict[str, Any]) -> Path:
    return report_root(config) / "render_results.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def examples_by_split(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    outputs = {"train": [], "val": [], "test": []}
    for row in rows:
        if row.get("status") not in SUCCESS_STATUSES:
            continue
        split = row.get("split", "")
        if split not in outputs:
            continue
        outputs[split].append(str(resolve_project_path(row["sample_dir"]).resolve()))
    for split in outputs:
        outputs[split] = sorted(set(outputs[split]))
    return outputs


def write_json(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def write_examples(config: dict[str, Any]) -> dict[str, Path]:
    rows = read_csv(render_results_csv(config))
    by_split = examples_by_split(rows)
    root = dataset_root(config)
    paths = {
        "train": root / "examples_train_abs.json",
        "val": root / "examples_val_abs.json",
        "test": root / "examples_test_abs.json",
        "all": root / "examples_all_abs.json",
    }
    write_json(paths["train"], by_split["train"])
    write_json(paths["val"], by_split["val"])
    write_json(paths["test"], by_split["test"])
    write_json(paths["all"], sorted(set(by_split["train"] + by_split["val"] + by_split["test"])))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build examples JSON files for Data v2 frame-panel rendered examples.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    paths = write_examples(config)
    print("Phase 2L.3A frame-panel examples JSON")
    for split, path in paths.items():
        print(f"  {split}: {path}")
    print("PHASE2L3A_FRAME_PANEL_EXAMPLES_JSON_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
