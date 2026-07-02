#!/usr/bin/env python3
"""Inspect official Hunyuan training initialization text interfaces."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SEARCH_TERMS = [
    "pretrained_model_name_or_path",
    "from_pretrained",
    "resume_from",
    "init_control_from",
    "load_state_dict",
    "model_unet_prefix",
    "stable_diffusion_config",
    "HunyuanPaint",
    "DiffusionPipeline",
]
RELATIVE_FILES = [
    "train.py",
    "hunyuanpaintpbr/unet/model.py",
    "hunyuanpaintpbr/pipeline.py",
    "cfgs/hunyuan-paint-pbr.yaml",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def term_hits(text: str, term: str, context: int = 2) -> list[dict[str, Any]]:
    lines = text.splitlines()
    hits: list[dict[str, Any]] = []
    pattern = re.compile(re.escape(term))
    for index, line in enumerate(lines, start=1):
        if not pattern.search(line):
            continue
        start = max(1, index - context)
        end = min(len(lines), index + context)
        hits.append(
            {
                "line": index,
                "text": line.strip(),
                "context": [
                    {"line": line_no, "text": lines[line_no - 1].rstrip()}
                    for line_no in range(start, end + 1)
                ],
            }
        )
    return hits


def inspect_file(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    item: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "term_hits": {term: [] for term in SEARCH_TERMS},
    }
    if not exists:
        return item
    text = read_text(path)
    item["term_hits"] = {
        term: term_hits(text, term)
        for term in SEARCH_TERMS
    }
    return item


def has_hit(report: dict[str, Any], term: str) -> bool:
    return any(file_item["term_hits"].get(term) for file_item in report["files"])


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    has_from_pretrained = has_hit(report, "from_pretrained")
    has_pretrained_path = has_hit(report, "pretrained_model_name_or_path")
    has_resume = has_hit(report, "resume_from")
    has_load_state = has_hit(report, "load_state_dict")
    return {
        "training_model_initialization": (
            "official text references pretrained_model_name_or_path/from_pretrained"
            if has_from_pretrained or has_pretrained_path
            else "pretrained initialization path not obvious from inspected text"
        ),
        "pretrained_model_name_or_path_likely_accepts_local_pipeline_dir": bool(
            has_from_pretrained and has_pretrained_path
        ),
        "resume_from_loading": (
            "resume_from and load_state_dict references found"
            if has_resume and has_load_state
            else "resume_from load path requires manual review"
        ),
        "expected_resume_checkpoint_key_prefixes": [
            "likely training checkpoint state_dict keys such as unet.*",
            "confirm exact prefixes from train.py/HunyuanPaint load_state_dict paths",
        ],
        "recommended_first_safe_strategy": (
            "Strategy A: try local hunyuan3d-paintpbr-v2-1 as pretrained_model_name_or_path"
            if has_from_pretrained and has_pretrained_path
            else "Strategy B: prepare a training-compatible resume checkpoint from official inference UNet"
        ),
    }


def inspect(hypaint: Path) -> dict[str, Any]:
    resolved = hypaint.expanduser().resolve()
    files = [inspect_file(resolved / rel) for rel in RELATIVE_FILES]
    report: dict[str, Any] = {
        "hypaint": str(resolved),
        "files": files,
    }
    report["summary"] = summarize(report)
    return report


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2J.0 Training Initialization Interface Inspection",
        "",
        f"HYPAINT: `{report['hypaint']}`",
        "",
        "## Summary",
    ]
    summary = report["summary"]
    for key, value in summary.items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - `{item}`" for item in value)
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Term Hits")
    for file_item in report["files"]:
        lines.extend(
            [
                f"### `{file_item['path']}`",
                f"- exists: `{file_item['exists']}`",
                f"- size bytes: `{file_item['size_bytes']}`",
            ]
        )
        for term, hits in file_item["term_hits"].items():
            if not hits:
                continue
            lines.append(f"- `{term}` hits: `{len(hits)}`")
            for hit in hits[:10]:
                lines.append(f"  - line {hit['line']}: `{hit['text']}`")
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect official Hunyuan training initialization interfaces."
    )
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = inspect(args.hypaint)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print("Phase 2J.0 training init interface inspection")
    print(f"  hypaint: {report['hypaint']}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2J_TRAINING_INIT_INTERFACE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
