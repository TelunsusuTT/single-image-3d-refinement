#!/usr/bin/env python3
"""Inspect inference UNet state_dict keys for Phase 2G.3.

This script is intended for A100/Slurm execution. It imports torch and official
Hunyuan modules only inside main and does not call the paint pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def prefix_counts(keys: list[str], max_depth: int = 4) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for depth in range(1, max_depth + 1):
        counter: Counter[str] = Counter()
        for key in keys:
            parts = key.split(".")
            counter[".".join(parts[:depth])] += 1
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


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def resolve_paths(hypaint_arg: Path) -> tuple[Path, Path]:
    hypaint = Path(os.environ.get("HYPAINT") or hypaint_arg).expanduser().resolve()
    hy21 = Path(os.environ.get("HY21") or hypaint.parent).expanduser().resolve()
    return hy21, hypaint


def get_multiview_model(paint_pipeline: Any) -> Any:
    models = getattr(paint_pipeline, "models", None)
    if isinstance(models, dict):
        return models.get("multiview_model")
    return getattr(models, "multiview_model", None)


def add_candidate(candidates: list[tuple[str, Any]], seen: set[int], path: str, obj: Any) -> None:
    if obj is None:
        return
    ident = id(obj)
    if ident in seen:
        return
    seen.add(ident)
    candidates.append((path, obj))


def find_unet_candidates(paint_pipeline: Any) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()
    multiview = get_multiview_model(paint_pipeline)
    pipeline = getattr(multiview, "pipeline", None)
    base = getattr(pipeline, "unet", None)
    base_path = "paint_pipeline.models['multiview_model'].pipeline.unet"
    add_candidate(candidates, seen, base_path, base)

    current = base
    current_path = base_path
    for _ in range(3):
        current = getattr(current, "unet", None)
        current_path = f"{current_path}.unet"
        add_candidate(candidates, seen, current_path, current)
        if current is None:
            break
    return candidates


def summarize_state_object(path: str, obj: Any, max_sample_keys: int) -> dict[str, Any]:
    has_state_dict = hasattr(obj, "state_dict")
    state = obj.state_dict() if has_state_dict else {}
    keys = sorted(str(key) for key in state.keys()) if isinstance(state, dict) else []
    sample_keys = keys[:max_sample_keys]
    sample_summaries = [
        tensor_summary(key, state[key])
        for key in sample_keys
        if isinstance(state, dict)
    ]
    return {
        "attribute_path": path,
        "class_name": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "has_state_dict": has_state_dict,
        "key_count": len(keys),
        "keys_sample": sample_keys,
        "all_keys": keys,
        "prefix_counts": prefix_counts(keys),
        "sample_tensor_summaries": sample_summaries,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2G.3 Inference UNet Key Summary",
        "",
        f"HY21: `{report['hy21']}`",
        f"HYPAINT: `{report['hypaint']}`",
        f"multiview config: `{report['multiview_cfg_path']}`",
        f"RealESRGAN checkpoint: `{report['realesrgan_ckpt_path']}`",
        "",
        "## Candidate UNets",
    ]
    for candidate in report["candidates"]:
        lines.extend(
            [
                f"### `{candidate['attribute_path']}`",
                f"- class: `{candidate['class_name']}`",
                f"- has state_dict: `{candidate['has_state_dict']}`",
                f"- key count: `{candidate['key_count']}`",
                "",
                "Sample keys:",
            ]
        )
        for key in candidate["keys_sample"]:
            lines.append(f"- `{key}`")
        lines.append("")
        lines.append("Sample tensor shapes:")
        for item in candidate["sample_tensor_summaries"]:
            lines.append(
                f"- `{item['key']}`: shape=`{item['shape']}` dtype=`{item['dtype']}` type=`{item['type']}`"
            )
        lines.append("")
    if not report["candidates"]:
        lines.append("- No candidate UNet objects found.")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Phase 2G.3 inference UNet keys.")
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--max-sample-keys", type=int, default=200)
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hy21, hypaint = resolve_paths(args.hypaint)
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    import torch  # type: ignore  # noqa: F401
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.multiview_cfg_path = str(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.realesrgan_ckpt_path = str(hypaint / "ckpt" / "RealESRGAN_x4plus.pth")

    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    candidates = [
        summarize_state_object(path, obj, args.max_sample_keys)
        for path, obj in find_unet_candidates(paint_pipeline)
    ]

    report = {
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        "max_num_view": args.max_num_view,
        "resolution": args.resolution,
        "multiview_cfg_path": conf.multiview_cfg_path,
        "realesrgan_ckpt_path": conf.realesrgan_ckpt_path,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)

    print("Phase 2G.3 inference UNet key inspection")
    print(f"  HY21: {hy21}")
    print(f"  HYPAINT: {hypaint}")
    print(f"  candidate_count: {len(candidates)}")
    for candidate in candidates:
        print(
            f"  {candidate['attribute_path']}: "
            f"class={candidate['class_name']} keys={candidate['key_count']}"
        )
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2G3_INFER_UNET_KEYS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
