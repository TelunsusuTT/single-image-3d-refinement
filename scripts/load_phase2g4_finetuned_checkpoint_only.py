#!/usr/bin/env python3
"""Load the Phase 2F checkpoint into the inference UNet without inference.

This script is intended for A100/Slurm execution. It imports torch and official
Hunyuan modules only inside main and never calls paint_pipeline(...).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any


CHECKPOINT_PREFIX = "unet."
TARGET_PATH = "paint_pipeline.models['multiview_model'].pipeline.unet"


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def resolve_paths(hypaint_arg: Path) -> tuple[Path, Path]:
    hypaint = Path(os.environ.get("HYPAINT") or hypaint_arg).expanduser().resolve()
    hy21 = Path(os.environ.get("HY21") or hypaint.parent).expanduser().resolve()
    return hy21, hypaint


def torch_load_cpu(torch_module: Any, checkpoint: Path) -> Any:
    try:
        return torch_module.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint, map_location="cpu")


def extract_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    return {}


def transformed_state_dict(state: dict[str, Any], prefix: str) -> tuple[dict[str, Any], list[str]]:
    transformed: dict[str, Any] = {}
    duplicates: list[str] = []
    for key, value in state.items():
        text_key = str(key)
        if not text_key.startswith(prefix):
            continue
        new_key = text_key[len(prefix):]
        if new_key in transformed:
            duplicates.append(new_key)
        transformed[new_key] = value
    return transformed, duplicates


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def tensor_mean_abs(value: Any) -> float | None:
    try:
        return float(value.detach().float().abs().mean().item())
    except Exception:
        return None


def compare_state_dicts(transformed: dict[str, Any], target_state: dict[str, Any]) -> dict[str, Any]:
    transformed_keys = set(transformed.keys())
    target_keys = set(target_state.keys())
    missing_target_keys = sorted(target_keys - transformed_keys)
    extra_transformed_keys = sorted(transformed_keys - target_keys)
    shape_mismatches = []
    for key in sorted(transformed_keys & target_keys):
        source_shape = shape_of(transformed[key])
        target_shape = shape_of(target_state[key])
        if source_shape != target_shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": source_shape,
                    "target_shape": target_shape,
                    "checkpoint_dtype": dtype_of(transformed[key]),
                    "target_dtype": dtype_of(target_state[key]),
                }
            )
    return {
        "transformed_key_count": len(transformed_keys),
        "target_key_count": len(target_keys),
        "key_set_exact_match": transformed_keys == target_keys,
        "missing_target_key_count": len(missing_target_keys),
        "extra_transformed_key_count": len(extra_transformed_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "missing_target_keys_sample": missing_target_keys[:50],
        "extra_transformed_keys_sample": extra_transformed_keys[:50],
        "shape_mismatches_sample": shape_mismatches[:50],
    }


def checksum_summary(state: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    summaries = []
    for key in sorted(state.keys())[:limit]:
        value = state[key]
        summaries.append(
            {
                "key": key,
                "shape": shape_of(value),
                "dtype": dtype_of(value),
                "mean_abs": tensor_mean_abs(value),
            }
        )
    return summaries


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2G.4 Checkpoint Load-Only Report",
        "",
        f"checkpoint: `{report['checkpoint']}`",
        f"target path: `{report['target_path']}`",
        f"checkpoint prefix: `{report['checkpoint_prefix']}`",
        f"status: `{report['status']}`",
        "",
        "## Compatibility",
        f"- transformed keys: `{report['compatibility']['transformed_key_count']}`",
        f"- target keys: `{report['compatibility']['target_key_count']}`",
        f"- key set exact match: `{report['compatibility']['key_set_exact_match']}`",
        f"- missing target keys: `{report['compatibility']['missing_target_key_count']}`",
        f"- extra transformed keys: `{report['compatibility']['extra_transformed_key_count']}`",
        f"- shape mismatches: `{report['compatibility']['shape_mismatch_count']}`",
        "",
        "## Strict Load",
        f"- attempted: `{report['strict_load_attempted']}`",
        f"- success: `{report['strict_load_success']}`",
        f"- strict_load_state_dict: `{'OK' if report['strict_load_success'] else 'FAILED'}`",
    ]
    if report.get("error"):
        lines.extend(["", "## Error", report["error"]])
    if report.get("load_result"):
        lines.extend(
            [
                "",
                "## Load Result",
                f"- missing keys: `{report['load_result']['missing_keys']}`",
                f"- unexpected keys: `{report['load_result']['unexpected_keys']}`",
            ]
        )
    lines.extend(["", "## Checksum Sample"])
    for item in report.get("checksum_sample", []):
        lines.append(
            f"- `{item['key']}`: shape=`{item['shape']}` dtype=`{item['dtype']}` mean_abs=`{item['mean_abs']}`"
        )
    if report.get("strict_load_success"):
        lines.extend(["", "PHASE2G4_CHECKPOINT_LOAD_ONLY_OK"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_md)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2G.4 checkpoint load-only smoke.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    hy21, hypaint = resolve_paths(args.hypaint)
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    import torch  # type: ignore
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        "target_path": TARGET_PATH,
        "checkpoint_prefix": CHECKPOINT_PREFIX,
        "max_num_view": args.max_num_view,
        "resolution": args.resolution,
        "status": "FAILED",
        "strict_load_attempted": False,
        "strict_load_success": False,
        "load_result": None,
        "checksum_sample": [],
        "error": "",
    }

    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.multiview_cfg_path = str(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.realesrgan_ckpt_path = str(hypaint / "ckpt" / "RealESRGAN_x4plus.pth")
    report["multiview_cfg_path"] = conf.multiview_cfg_path
    report["realesrgan_ckpt_path"] = conf.realesrgan_ckpt_path

    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    target = paint_pipeline.models["multiview_model"].pipeline.unet
    target_state = target.state_dict()

    checkpoint_obj = torch_load_cpu(torch, checkpoint)
    state = extract_state_dict(checkpoint_obj)
    transformed, duplicate_keys = transformed_state_dict(state, CHECKPOINT_PREFIX)
    compatibility = compare_state_dicts(transformed, target_state)
    compatibility["duplicate_transformed_key_count"] = len(duplicate_keys)
    compatibility["duplicate_transformed_keys_sample"] = duplicate_keys[:50]
    report["raw_state_dict_key_count"] = len(state)
    report["compatibility"] = compatibility

    can_load = (
        compatibility["transformed_key_count"] == compatibility["target_key_count"]
        and compatibility["key_set_exact_match"]
        and compatibility["shape_mismatch_count"] == 0
        and compatibility["duplicate_transformed_key_count"] == 0
    )
    if not can_load:
        report["error"] = "Checkpoint tensors are not exactly compatible with the target UNet."
        write_report(report, args.out_json, args.out_md)
        print("ERROR: checkpoint tensors are not compatible with target UNet")
        return 1

    report["strict_load_attempted"] = True
    try:
        load_result = target.load_state_dict(transformed, strict=True)
    except Exception as exc:
        report["error"] = f"strict load_state_dict failed: {exc}"
        write_report(report, args.out_json, args.out_md)
        print(f"ERROR: strict load_state_dict failed: {exc}")
        return 1
    report["strict_load_success"] = True
    report["load_result"] = {
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }
    report["checksum_sample"] = checksum_summary(transformed, limit=20)
    report["status"] = "OK"
    write_report(report, args.out_json, args.out_md)

    print("Phase 2G.4 checkpoint load-only smoke")
    print(f"  checkpoint: {checkpoint}")
    print(f"  target_path: {TARGET_PATH}")
    print(f"  transformed_key_count: {compatibility['transformed_key_count']}")
    print(f"  target_key_count: {compatibility['target_key_count']}")
    print("  strict_load_state_dict: OK")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2G4_CHECKPOINT_LOAD_ONLY_OK")

    del transformed
    del state
    del checkpoint_obj
    del target_state
    del target
    del paint_pipeline
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
