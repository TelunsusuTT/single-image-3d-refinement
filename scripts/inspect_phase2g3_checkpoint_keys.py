#!/usr/bin/env python3
"""Inspect Phase 2F checkpoint key structure for Phase 2G.3.

This script is intended for A100/Slurm execution. It imports torch only inside
main and loads the checkpoint on CPU.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CANDIDATE_PREFIXES = ("unet.unet.", "unet.unet.unet.", "unet.controlnet.")


def prefix_counts(keys: list[str], max_depth: int = 4) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for depth in range(1, max_depth + 1):
        counter: Counter[str] = Counter()
        for key in keys:
            parts = key.split(".")
            prefix = ".".join(parts[:depth])
            counter[prefix] += 1
        counts[str(depth)] = dict(counter.most_common(50))
    return counts


def tensor_summary(key: str, value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return {
        "key": key,
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "type": type(value).__name__,
    }


def summarize_checkpoint(obj: Any, checkpoint: Path, max_sample_keys: int) -> dict[str, Any]:
    top_level_keys: list[str] = []
    state_dict_exists = False
    state: Any = {}

    if isinstance(obj, dict):
        top_level_keys = [str(key) for key in obj.keys()]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict_exists = True
            state = obj["state_dict"]
        else:
            state = obj

    state_keys = sorted(str(key) for key in state.keys()) if isinstance(state, dict) else []
    sample_keys = state_keys[:max_sample_keys]
    sample_summaries: list[dict[str, Any]] = []
    if isinstance(state, dict):
        for key in sample_keys:
            sample_summaries.append(tensor_summary(key, state[key]))

    candidate_counts = {
        prefix: sum(1 for key in state_keys if key.startswith(prefix))
        for prefix in CANDIDATE_PREFIXES
    }

    return {
        "checkpoint": str(checkpoint),
        "top_level_type": type(obj).__name__,
        "top_level_keys": top_level_keys,
        "state_dict_exists": state_dict_exists,
        "state_dict_key_count": len(state_keys),
        "state_dict_keys_sample": sample_keys,
        "state_dict_all_keys": state_keys,
        "prefix_counts": prefix_counts(state_keys),
        "sample_tensor_summaries": sample_summaries,
        "candidate_prefix_counts": candidate_counts,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2G.3 Checkpoint Key Summary",
        "",
        f"checkpoint: `{report['checkpoint']}`",
        f"top-level type: `{report['top_level_type']}`",
        f"state_dict exists: `{report['state_dict_exists']}`",
        f"state_dict key count: `{report['state_dict_key_count']}`",
        "",
        "## Top-Level Keys",
    ]
    if report["top_level_keys"]:
        for key in report["top_level_keys"]:
            lines.append(f"- `{key}`")
    else:
        lines.append("- None found.")

    lines.extend(["", "## Candidate Prefix Counts"])
    for prefix, count in report["candidate_prefix_counts"].items():
        lines.append(f"- `{prefix}`: {count}")

    lines.extend(["", "## Prefix Counts"])
    for depth, counts in report["prefix_counts"].items():
        lines.append(f"### First {depth} Component(s)")
        for prefix, count in list(counts.items())[:25]:
            lines.append(f"- `{prefix}`: {count}")

    lines.extend(["", "## Sample Keys"])
    for key in report["state_dict_keys_sample"]:
        lines.append(f"- `{key}`")

    lines.extend(["", "## Sample Tensor Shapes"])
    for item in report["sample_tensor_summaries"]:
        lines.append(
            f"- `{item['key']}`: shape=`{item['shape']}` dtype=`{item['dtype']}` type=`{item['type']}`"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Phase 2G.3 checkpoint keys.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--max-sample-keys", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        print(f"ERROR: checkpoint missing or zero-size: {checkpoint}")
        return 1

    import torch  # type: ignore

    obj = torch.load(checkpoint, map_location="cpu")
    report = summarize_checkpoint(obj, checkpoint, args.max_sample_keys)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)

    print("Phase 2G.3 checkpoint key inspection")
    print(f"  checkpoint: {checkpoint}")
    print(f"  state_dict_key_count: {report['state_dict_key_count']}")
    print(f"  sampled_keys: {len(report['state_dict_keys_sample'])}")
    for prefix, count in report["candidate_prefix_counts"].items():
        print(f"  {prefix}: {count}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2G3_CHECKPOINT_KEYS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
