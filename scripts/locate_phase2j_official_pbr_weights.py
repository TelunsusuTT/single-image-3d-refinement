#!/usr/bin/env python3
"""Locate likely local Hunyuan3D-Paint PBR pipeline directories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PBR_TOKEN = "hunyuan3d-paintpbr-v2-1"
SKIP_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "checkpoints",
    "logs",
    "outputs",
}
SKIP_REL_PREFIXES = (
    ("data", "raw_assets"),
    ("data", "hy3dpaint_train_examples"),
)


def should_skip(path: Path, root: Path) -> bool:
    if path.name in SKIP_DIR_NAMES:
        return True
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(rel_parts[: len(prefix)] == prefix for prefix in SKIP_REL_PREFIXES)


def child_sets(path: Path) -> tuple[set[str], set[str]]:
    dir_names: set[str] = set()
    file_names: set[str] = set()
    try:
        for child in path.iterdir():
            if child.is_dir():
                dir_names.add(child.name)
            elif child.is_file():
                file_names.add(child.name)
    except OSError:
        pass
    return dir_names, file_names


def candidate_signals(path: Path) -> dict[str, Any]:
    dir_names, file_names = child_sets(path)
    lower_path = str(path).lower()
    lower_dirs = {name.lower() for name in dir_names}
    lower_files = {name.lower() for name in file_names}
    signals = {
        "path_contains_pbr_token": PBR_TOKEN in lower_path,
        "has_model_index_json": "model_index.json" in lower_files,
        "has_unet_dir": "unet" in lower_dirs,
        "has_scheduler_dir": "scheduler" in lower_dirs,
        "has_tokenizer_dir": "tokenizer" in lower_dirs,
        "has_text_encoder_dir": "text_encoder" in lower_dirs,
        "child_name_matches": sorted(
            name
            for name in dir_names | file_names
            if any(
                token in name.lower()
                for token in (
                    PBR_TOKEN,
                    "model_index.json",
                    "unet",
                    "scheduler",
                    "tokenizer",
                    "text_encoder",
                )
            )
        ),
    }
    return signals


def confidence(signals: dict[str, Any]) -> str:
    if (
        signals["has_model_index_json"]
        and signals["has_unet_dir"]
        and signals["has_scheduler_dir"]
    ):
        return "strong"
    if (
        signals["has_model_index_json"]
        and (signals["has_unet_dir"] or signals["path_contains_pbr_token"])
    ):
        return "medium"
    if signals["path_contains_pbr_token"] or any(
        signals[key]
        for key in (
            "has_model_index_json",
            "has_unet_dir",
            "has_scheduler_dir",
            "has_tokenizer_dir",
            "has_text_encoder_dir",
        )
    ):
        return "weak"
    return "none"


def score(conf: str) -> int:
    return {"strong": 3, "medium": 2, "weak": 1, "none": 0}[conf]


def scan_root(root: Path) -> list[dict[str, Any]]:
    resolved_root = root.expanduser().resolve()
    candidates: list[dict[str, Any]] = []
    if not resolved_root.exists():
        return candidates

    for current_text, dirs, _files in os.walk(resolved_root, followlinks=False):
        current = Path(current_text)
        dirs[:] = sorted(
            name
            for name in dirs
            if not should_skip(current / name, resolved_root)
        )
        signals = candidate_signals(current)
        conf = confidence(signals)
        if conf == "none":
            continue
        candidates.append(
            {
                "path": str(current),
                "confidence": conf,
                "signals": signals,
            }
        )
    return candidates


def locate(search_roots: list[Path]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for root in search_roots:
        candidates.extend(scan_root(root))
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique[candidate["path"]] = candidate
    ordered = sorted(
        unique.values(),
        key=lambda item: (-score(item["confidence"]), item["path"]),
    )
    return {
        "search_roots": [str(path.expanduser().resolve()) for path in search_roots],
        "candidate_count": len(ordered),
        "strong_candidate_count": sum(1 for item in ordered if item["confidence"] == "strong"),
        "candidates": ordered,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2J.0 Official PBR Weight Locator",
        "",
        "## Search Roots",
    ]
    for root in report["search_roots"]:
        lines.append(f"- `{root}`")
    lines.extend(
        [
            "",
            f"candidate count: `{report['candidate_count']}`",
            f"strong candidate count: `{report['strong_candidate_count']}`",
            "",
            "## Candidates",
        ]
    )
    if not report["candidates"]:
        lines.append("- No candidates found.")
    for candidate in report["candidates"]:
        signals = candidate["signals"]
        lines.extend(
            [
                f"### `{candidate['path']}`",
                f"- confidence: `{candidate['confidence']}`",
                f"- model_index.json: `{signals['has_model_index_json']}`",
                f"- unet/: `{signals['has_unet_dir']}`",
                f"- scheduler/: `{signals['has_scheduler_dir']}`",
                f"- tokenizer/: `{signals['has_tokenizer_dir']}`",
                f"- text_encoder/: `{signals['has_text_encoder_dir']}`",
                f"- path contains `{PBR_TOKEN}`: `{signals['path_contains_pbr_token']}`",
                "",
            ]
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate official PBR weights.")
    parser.add_argument("--search-root", nargs="+", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = locate(args.search_root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print("Phase 2J.0 official PBR weight locator")
    print(f"  candidate_count: {report['candidate_count']}")
    print(f"  strong_candidate_count: {report['strong_candidate_count']}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2J_PBR_WEIGHT_LOCATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
