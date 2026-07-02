#!/usr/bin/env python3
"""Compare official base inference UNet weights to training checkpoints.

This script is intended for A100/Slurm execution. It imports torch and official
Hunyuan modules only inside main and never calls paint_pipeline(...).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


CHECKPOINT_PREFIX = "unet."
EPS = 1e-12


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def resolve_paths(hypaint_arg: Path) -> tuple[Path, Path]:
    hypaint = Path(os.environ.get("HYPAINT") or hypaint_arg).expanduser().resolve()
    hy21 = Path(os.environ.get("HY21") or hypaint.parent).expanduser().resolve()
    return hy21, hypaint


def parse_named_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"checkpoint must be NAME=PATH, got: {value}"
        )
    name, path_text = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"checkpoint name is empty: {value}")
    return name, Path(path_text).expanduser()


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


def transform_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    transformed: dict[str, Any] = {}
    for key, value in state.items():
        text_key = str(key)
        if text_key.startswith(CHECKPOINT_PREFIX):
            transformed[text_key[len(CHECKPOINT_PREFIX):]] = value
    return transformed


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def compare_key_sets(transformed: dict[str, Any], base_state: dict[str, Any]) -> dict[str, Any]:
    transformed_keys = set(transformed.keys())
    base_keys = set(base_state.keys())
    shape_mismatches = []
    for key in sorted(transformed_keys & base_keys):
        checkpoint_shape = shape_of(transformed[key])
        base_shape = shape_of(base_state[key])
        if checkpoint_shape != base_shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": checkpoint_shape,
                    "base_shape": base_shape,
                }
            )
    return {
        "transformed_key_count": len(transformed_keys),
        "base_key_count": len(base_keys),
        "key_set_exact_match": transformed_keys == base_keys,
        "missing_base_keys": sorted(base_keys - transformed_keys)[:100],
        "extra_checkpoint_keys": sorted(transformed_keys - base_keys)[:100],
        "missing_base_key_count": len(base_keys - transformed_keys),
        "extra_checkpoint_key_count": len(transformed_keys - base_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches_sample": shape_mismatches[:100],
    }


def tensor_delta_metrics(torch_module: Any, checkpoint_tensor: Any, base_tensor: Any) -> dict[str, Any]:
    checkpoint_cpu = checkpoint_tensor.detach().to(device="cpu", dtype=torch_module.float32)
    base_cpu = base_tensor.detach().to(device="cpu", dtype=torch_module.float32)
    delta = (checkpoint_cpu - base_cpu).abs()
    mean_abs_delta = float(delta.mean().item())
    max_abs_delta = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs_base = float(base_cpu.abs().mean().item())
    return {
        "numel": int(delta.numel()),
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "mean_abs_base": mean_abs_base,
        "relative_mean_delta": mean_abs_delta / (mean_abs_base + EPS),
    }


def interpret_checkpoint(name: str, aggregate: dict[str, Any]) -> str:
    if aggregate.get("status") != "OK":
        return "comparison failed before numeric interpretation"
    mean_delta = aggregate["mean_of_mean_abs_delta"]
    mean_relative = aggregate["mean_relative_delta"]
    pct_lt_1e4 = aggregate["percent_keys_mean_abs_delta_lt_1e-4"]
    if mean_delta > 1e-3 or mean_relative > 0.05 or pct_lt_1e4 < 50.0:
        return (
            f"{name}: large deltas from official base UNet; likely not initialized "
            "from official inference base or changed very substantially"
        )
    if mean_delta < 1e-5 or pct_lt_1e4 >= 90.0:
        return (
            f"{name}: numerically close to official base UNet; collapse may involve "
            "high sensitivity, non-UNet mismatch, or data/export issues"
        )
    return f"{name}: intermediate deltas; review top changed keys and training setup"


def compare_checkpoint(
    torch_module: Any,
    name: str,
    checkpoint: Path,
    base_state: dict[str, Any],
    max_sample_keys: int,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_obj = torch_load_cpu(torch_module, checkpoint)
    raw_state = extract_state_dict(checkpoint_obj)
    transformed = transform_checkpoint_state(raw_state)
    compatibility = compare_key_sets(transformed, base_state)

    report: dict[str, Any] = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_prefix": CHECKPOINT_PREFIX,
        "raw_state_dict_key_count": len(raw_state),
        "compatibility": compatibility,
        "status": "FAILED",
        "interpretation": "",
    }
    if (
        not compatibility["key_set_exact_match"]
        or compatibility["shape_mismatch_count"] != 0
    ):
        report["interpretation"] = interpret_checkpoint(name, report)
        return report

    per_key = []
    total_numel = 0
    for key in sorted(base_state.keys()):
        metrics = tensor_delta_metrics(torch_module, transformed[key], base_state[key])
        total_numel += metrics["numel"]
        per_key.append({"key": key, **metrics})

    mean_abs_values = [item["mean_abs_delta"] for item in per_key]
    relative_values = [item["relative_mean_delta"] for item in per_key]
    top = sorted(per_key, key=lambda item: item["mean_abs_delta"], reverse=True)[:30]
    sample = per_key[: min(max_sample_keys, 30, len(per_key))]
    aggregate = {
        "status": "OK",
        "key_count": len(per_key),
        "total_numel": total_numel,
        "mean_of_mean_abs_delta": statistics.fmean(mean_abs_values) if mean_abs_values else 0.0,
        "median_of_mean_abs_delta": statistics.median(mean_abs_values) if mean_abs_values else 0.0,
        "max_of_mean_abs_delta": max(mean_abs_values) if mean_abs_values else 0.0,
        "mean_relative_delta": statistics.fmean(relative_values) if relative_values else 0.0,
        "percent_keys_mean_abs_delta_lt_1e-6": 100.0
        * sum(1 for value in mean_abs_values if value < 1e-6)
        / max(1, len(mean_abs_values)),
        "percent_keys_mean_abs_delta_lt_1e-5": 100.0
        * sum(1 for value in mean_abs_values if value < 1e-5)
        / max(1, len(mean_abs_values)),
        "percent_keys_mean_abs_delta_lt_1e-4": 100.0
        * sum(1 for value in mean_abs_values if value < 1e-4)
        / max(1, len(mean_abs_values)),
        "top_30_keys_by_mean_abs_delta": top,
        "sample_30_keys": sample,
    }
    aggregate["interpretation"] = interpret_checkpoint(name, aggregate)
    report.update(aggregate)
    return report


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2I Base UNet vs Training Checkpoints",
        "",
        f"HY21: `{report['hy21']}`",
        f"HYPAINT: `{report['hypaint']}`",
        f"base key count: `{report['base_key_count']}`",
        f"checkpoint count: `{report['checkpoint_count']}`",
        "",
        "## Overall Interpretation",
        report["overall_interpretation"],
        "",
    ]
    for item in report["checkpoints"]:
        lines.extend(
            [
                f"## `{item['name']}`",
                "",
                f"- checkpoint: `{item['checkpoint']}`",
                f"- status: `{item['status']}`",
                f"- interpretation: `{item['interpretation']}`",
                f"- raw state_dict keys: `{item['raw_state_dict_key_count']}`",
                f"- transformed keys: `{item['compatibility']['transformed_key_count']}`",
                f"- key set exact match: `{item['compatibility']['key_set_exact_match']}`",
                f"- shape mismatches: `{item['compatibility']['shape_mismatch_count']}`",
            ]
        )
        if item["status"] != "OK":
            lines.extend(
                [
                    f"- missing base keys: `{item['compatibility']['missing_base_key_count']}`",
                    f"- extra checkpoint keys: `{item['compatibility']['extra_checkpoint_key_count']}`",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"- key count: `{item['key_count']}`",
                f"- total numel: `{item['total_numel']}`",
                f"- mean of mean abs delta: `{item['mean_of_mean_abs_delta']:.8e}`",
                f"- median of mean abs delta: `{item['median_of_mean_abs_delta']:.8e}`",
                f"- max of mean abs delta: `{item['max_of_mean_abs_delta']:.8e}`",
                f"- mean relative delta: `{item['mean_relative_delta']:.8e}`",
                "- percent keys mean abs delta < 1e-6: "
                f"`{item['percent_keys_mean_abs_delta_lt_1e-6']:.2f}%`",
                "- percent keys mean abs delta < 1e-5: "
                f"`{item['percent_keys_mean_abs_delta_lt_1e-5']:.2f}%`",
                "- percent keys mean abs delta < 1e-4: "
                f"`{item['percent_keys_mean_abs_delta_lt_1e-4']:.2f}%`",
                "",
                "### Top Changed Keys",
            ]
        )
        for key_item in item["top_30_keys_by_mean_abs_delta"]:
            lines.append(
                f"- `{key_item['key']}`: mean_abs_delta=`{key_item['mean_abs_delta']:.8e}` "
                f"max_abs_delta=`{key_item['max_abs_delta']:.8e}` "
                f"relative=`{key_item['relative_mean_delta']:.8e}`"
            )
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official base inference UNet to training checkpoints."
    )
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--checkpoint", nargs="+", required=True, type=parse_named_checkpoint)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--max-sample-keys", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hy21, hypaint = resolve_paths(args.hypaint)
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    import torch  # type: ignore
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.multiview_cfg_path = str(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.realesrgan_ckpt_path = str(hypaint / "ckpt" / "RealESRGAN_x4plus.pth")

    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    base_state = paint_pipeline.models["multiview_model"].pipeline.unet.state_dict()

    checkpoint_reports = [
        compare_checkpoint(torch, name, path, base_state, args.max_sample_keys)
        for name, path in args.checkpoint
    ]
    conservative = next(
        (item for item in checkpoint_reports if item["name"] == "conservative50"),
        None,
    )
    if conservative is not None and conservative.get("status") == "OK":
        overall = conservative["interpretation"]
    else:
        overall = "conservative50 comparison unavailable or failed; inspect compatibility errors"

    report = {
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        "multiview_cfg_path": conf.multiview_cfg_path,
        "realesrgan_ckpt_path": conf.realesrgan_ckpt_path,
        "target_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
        "base_key_count": len(base_state),
        "checkpoint_count": len(checkpoint_reports),
        "overall_interpretation": overall,
        "checkpoints": checkpoint_reports,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)

    print("Phase 2I base UNet vs checkpoint comparison")
    print(f"  HY21: {hy21}")
    print(f"  HYPAINT: {hypaint}")
    print(f"  base_key_count: {len(base_state)}")
    for item in checkpoint_reports:
        print(f"  {item['name']}: status={item['status']} checkpoint={item['checkpoint']}")
        if item["status"] == "OK":
            print(f"    mean_of_mean_abs_delta={item['mean_of_mean_abs_delta']:.8e}")
            print(f"    mean_relative_delta={item['mean_relative_delta']:.8e}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2I_BASE_VS_CHECKPOINT_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
