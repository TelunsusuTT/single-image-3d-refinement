#!/usr/bin/env python3
"""Compare true-PBR training initialization to official inference base UNet.

This script is intended for A100/Slurm execution. It imports torch, OmegaConf,
and Hunyuan modules only inside main. It does not run train.py, Trainer.fit, or
paint_pipeline(...).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


EPS = 1e-12
TRANSFORMS = ("as_is", "strip unet.", "add unet.")


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def resolve_paths(hypaint_arg: Path) -> tuple[Path, Path]:
    hypaint = Path(os.environ.get("HYPAINT") or hypaint_arg).expanduser().resolve()
    hy21 = Path(os.environ.get("HY21") or hypaint.parent).expanduser().resolve()
    return hy21, hypaint


@contextmanager
def pushd(path: Path) -> Any:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def get_obj_from_str(string: str) -> Any:
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config: Any) -> Any:
    if "target" not in config:
        raise KeyError("config object is missing target")
    params = config.get("params", {})
    return get_obj_from_str(config["target"])(**params)


def get_path(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def add_candidate(candidates: list[tuple[str, Any]], seen: set[int], path: str, obj: Any) -> None:
    if obj is None or not hasattr(obj, "state_dict"):
        return
    ident = id(obj)
    if ident in seen:
        return
    seen.add(ident)
    candidates.append((path, obj))


def discover_candidates(model: Any, max_depth: int = 4) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()
    add_candidate(candidates, seen, "model", model)
    for path in (
        "unet",
        "unet.unet",
        "unet.unet.unet",
        "unet.controlnet",
        "pipeline",
        "pipeline.unet",
        "pipeline.unet.unet",
        "pipeline.unet.unet.unet",
        "pipeline.unet.controlnet",
    ):
        add_candidate(candidates, seen, f"model.{path}", get_path(model, path))

    def walk(prefix: str, module: Any, depth: int) -> None:
        if depth >= max_depth or not hasattr(module, "named_children"):
            return
        try:
            children = list(module.named_children())
        except Exception:
            return
        for name, child in children:
            path = f"{prefix}.{name}"
            add_candidate(candidates, seen, path, child)
            walk(path, child, depth + 1)

    walk("model", model, 0)
    return candidates


def transform_state_keys(state: dict[str, Any], transform: str) -> dict[str, Any]:
    transformed: dict[str, Any] = {}
    for key, value in state.items():
        text_key = str(key)
        if transform == "as_is":
            new_key = text_key
        elif transform == "strip unet.":
            if not text_key.startswith("unet."):
                continue
            new_key = text_key[len("unet."):]
        elif transform == "add unet.":
            new_key = f"unet.{text_key}"
        else:
            raise ValueError(transform)
        transformed[new_key] = value
    return transformed


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def transform_summary(transformed: dict[str, Any], infer_state: dict[str, Any]) -> dict[str, Any]:
    transformed_keys = set(transformed.keys())
    infer_keys = set(infer_state.keys())
    common = sorted(transformed_keys & infer_keys)
    shape_mismatches = []
    same_shape_keys = []
    for key in common:
        candidate_shape = shape_of(transformed[key])
        infer_shape = shape_of(infer_state[key])
        if candidate_shape == infer_shape:
            same_shape_keys.append(key)
        else:
            shape_mismatches.append(
                {
                    "key": key,
                    "candidate_shape": candidate_shape,
                    "infer_shape": infer_shape,
                }
            )
    return {
        "candidate_key_count": len(transformed_keys),
        "infer_key_count": len(infer_keys),
        "overlap_count": len(common),
        "same_shape_overlap_count": len(same_shape_keys),
        "key_set_exact_match": transformed_keys == infer_keys,
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches_sample": shape_mismatches[:50],
        "missing_infer_keys_sample": sorted(infer_keys - transformed_keys)[:50],
        "extra_candidate_keys_sample": sorted(transformed_keys - infer_keys)[:50],
        "same_shape_keys": same_shape_keys,
    }


def tensor_delta_metrics(torch_module: Any, candidate_tensor: Any, infer_tensor: Any) -> dict[str, Any]:
    candidate_cpu = candidate_tensor.detach().to(device="cpu", dtype=torch_module.float32)
    infer_cpu = infer_tensor.detach().to(device="cpu", dtype=torch_module.float32)
    delta = (candidate_cpu - infer_cpu).abs()
    mean_abs_delta = float(delta.mean().item())
    max_abs_delta = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs_infer = float(infer_cpu.abs().mean().item())
    return {
        "numel": int(delta.numel()),
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "mean_abs_infer": mean_abs_infer,
        "relative_mean_delta": mean_abs_delta / (mean_abs_infer + EPS),
    }


def delta_summary(
    torch_module: Any,
    transformed: dict[str, Any],
    infer_state: dict[str, Any],
    same_shape_keys: list[str],
) -> dict[str, Any]:
    per_key = []
    total_numel = 0
    for key in same_shape_keys:
        metrics = tensor_delta_metrics(torch_module, transformed[key], infer_state[key])
        total_numel += metrics["numel"]
        per_key.append({"key": key, **metrics})
    mean_values = [item["mean_abs_delta"] for item in per_key]
    rel_values = [item["relative_mean_delta"] for item in per_key]
    top = sorted(per_key, key=lambda item: item["mean_abs_delta"], reverse=True)[:20]
    return {
        "delta_key_count": len(per_key),
        "total_numel": total_numel,
        "mean_of_mean_abs_delta": statistics.fmean(mean_values) if mean_values else None,
        "median_of_mean_abs_delta": statistics.median(mean_values) if mean_values else None,
        "max_of_mean_abs_delta": max(mean_values) if mean_values else None,
        "mean_relative_delta": statistics.fmean(rel_values) if rel_values else None,
        "percent_keys_mean_abs_delta_lt_1e-7": (
            100.0 * sum(1 for value in mean_values if value < 1e-7) / max(1, len(mean_values))
        ),
        "percent_keys_mean_abs_delta_lt_1e-6": (
            100.0 * sum(1 for value in mean_values if value < 1e-6) / max(1, len(mean_values))
        ),
        "percent_keys_mean_abs_delta_lt_1e-5": (
            100.0 * sum(1 for value in mean_values if value < 1e-5) / max(1, len(mean_values))
        ),
        "top_20_delta_keys": top,
    }


def best_transform_report(torch_module: Any, state: dict[str, Any], infer_state: dict[str, Any]) -> dict[str, Any]:
    transform_reports = []
    for transform in TRANSFORMS:
        transformed = transform_state_keys(state, transform)
        summary = transform_summary(transformed, infer_state)
        deltas = delta_summary(
            torch_module,
            transformed,
            infer_state,
            summary["same_shape_keys"],
        )
        summary.pop("same_shape_keys")
        transform_reports.append(
            {
                "transform": transform,
                **summary,
                **deltas,
            }
        )
    return max(
        transform_reports,
        key=lambda item: (
            item["key_set_exact_match"],
            -item["shape_mismatch_count"],
            item["overlap_count"],
            item["same_shape_overlap_count"],
            -(item["mean_of_mean_abs_delta"] if item["mean_of_mean_abs_delta"] is not None else 1e9),
        ),
    )


def candidate_report(
    torch_module: Any,
    path: str,
    obj: Any,
    infer_state: dict[str, Any],
) -> dict[str, Any]:
    state = obj.state_dict()
    best = best_transform_report(torch_module, state, infer_state)
    return {
        "candidate_path": path,
        "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "candidate_key_count_raw": len(state),
        "infer_key_count": len(infer_state),
        "best_transform": best["transform"],
        "overlap_count": best["overlap_count"],
        "key_set_exact_match": best["key_set_exact_match"],
        "shape_mismatch_count": best["shape_mismatch_count"],
        "candidate_key_count": best["candidate_key_count"],
        "same_shape_overlap_count": best["same_shape_overlap_count"],
        "mean_of_mean_abs_delta": best["mean_of_mean_abs_delta"],
        "median_of_mean_abs_delta": best["median_of_mean_abs_delta"],
        "max_of_mean_abs_delta": best["max_of_mean_abs_delta"],
        "mean_relative_delta": best["mean_relative_delta"],
        "percent_keys_mean_abs_delta_lt_1e-7": best["percent_keys_mean_abs_delta_lt_1e-7"],
        "percent_keys_mean_abs_delta_lt_1e-6": best["percent_keys_mean_abs_delta_lt_1e-6"],
        "percent_keys_mean_abs_delta_lt_1e-5": best["percent_keys_mean_abs_delta_lt_1e-5"],
        "top_20_delta_keys": best["top_20_delta_keys"],
        "missing_infer_keys_sample": best["missing_infer_keys_sample"],
        "extra_candidate_keys_sample": best["extra_candidate_keys_sample"],
        "shape_mismatches_sample": best["shape_mismatches_sample"],
    }


def recommendation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [
        item
        for item in candidates
        if item["key_set_exact_match"]
        and item["shape_mismatch_count"] == 0
        and item["mean_of_mean_abs_delta"] is not None
    ]
    tiny = [
        item
        for item in exact
        if item["mean_of_mean_abs_delta"] < 1e-6
    ]
    if tiny:
        best = min(
            tiny,
            key=lambda item: (
                item["mean_of_mean_abs_delta"],
                item["mean_relative_delta"] or 0.0,
                -item["overlap_count"],
            ),
        )
        return {
            "recommendation_status": "RECOMMENDED",
            "recommended_candidate_path": best["candidate_path"],
            "recommended_candidate_class": best["class"],
            "recommended_transform": best["best_transform"],
            "mean_of_mean_abs_delta": best["mean_of_mean_abs_delta"],
            "mean_relative_delta": best["mean_relative_delta"],
            "reason": "candidate exactly matches inference base key/shape set with tiny weight deltas",
        }
    if exact:
        best = min(exact, key=lambda item: item["mean_of_mean_abs_delta"])
        return {
            "recommendation_status": "UNKNOWN",
            "recommended_candidate_path": best["candidate_path"],
            "recommended_candidate_class": best["class"],
            "recommended_transform": best["best_transform"],
            "mean_of_mean_abs_delta": best["mean_of_mean_abs_delta"],
            "mean_relative_delta": best["mean_relative_delta"],
            "reason": "exact key/shape match exists, but deltas are not tiny",
        }
    return {
        "recommendation_status": "FAILED",
        "recommended_candidate_path": "",
        "recommended_candidate_class": "",
        "recommended_transform": "",
        "mean_of_mean_abs_delta": None,
        "mean_relative_delta": None,
        "reason": "no candidate had exact key and shape match to inference base",
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    rec = report["recommendation"]
    lines = [
        "# Phase 2J.1 True PBR Initialization Equivalence Probe",
        "",
        f"HY21: `{report['hy21']}`",
        f"HYPAINT: `{report['hypaint']}`",
        f"config: `{report['config']}`",
        f"inference base keys: `{report['infer_key_count']}`",
        f"training candidate count: `{report['candidate_count']}`",
        "",
        "## Recommendation",
        f"- status: `{rec['recommendation_status']}`",
        f"- candidate path: `{rec['recommended_candidate_path']}`",
        f"- candidate class: `{rec['recommended_candidate_class']}`",
        f"- transform: `{rec['recommended_transform']}`",
        f"- mean of mean abs delta: `{rec['mean_of_mean_abs_delta']}`",
        f"- mean relative delta: `{rec['mean_relative_delta']}`",
        f"- reason: `{rec['reason']}`",
        "",
        "## Candidates",
    ]
    for item in report["candidates"]:
        lines.extend(
            [
                f"### `{item['candidate_path']}`",
                f"- class: `{item['class']}`",
                f"- raw candidate keys: `{item['candidate_key_count_raw']}`",
                f"- best transform: `{item['best_transform']}`",
                f"- transformed candidate keys: `{item['candidate_key_count']}`",
                f"- inference keys: `{item['infer_key_count']}`",
                f"- overlap: `{item['overlap_count']}`",
                f"- exact key match: `{item['key_set_exact_match']}`",
                f"- shape mismatches: `{item['shape_mismatch_count']}`",
                f"- mean of mean abs delta: `{item['mean_of_mean_abs_delta']}`",
                f"- median of mean abs delta: `{item['median_of_mean_abs_delta']}`",
                f"- max of mean abs delta: `{item['max_of_mean_abs_delta']}`",
                f"- mean relative delta: `{item['mean_relative_delta']}`",
                "- percent keys mean abs delta < 1e-7: "
                f"`{item['percent_keys_mean_abs_delta_lt_1e-7']:.2f}%`",
                "- percent keys mean abs delta < 1e-6: "
                f"`{item['percent_keys_mean_abs_delta_lt_1e-6']:.2f}%`",
                "- percent keys mean abs delta < 1e-5: "
                f"`{item['percent_keys_mean_abs_delta_lt_1e-5']:.2f}%`",
                "",
                "Top delta keys:",
            ]
        )
        for key_item in item["top_20_delta_keys"][:10]:
            lines.append(
                f"- `{key_item['key']}`: mean_abs_delta=`{key_item['mean_abs_delta']:.8e}` "
                f"relative=`{key_item['relative_mean_delta']:.8e}`"
            )
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare true-PBR training initialization to inference base UNet."
    )
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
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
    from omegaconf import OmegaConf  # type: ignore
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.multiview_cfg_path = str(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.realesrgan_ckpt_path = str(hypaint / "ckpt" / "RealESRGAN_x4plus.pth")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    infer_state = paint_pipeline.models["multiview_model"].pipeline.unet.state_dict()

    config = OmegaConf.load(args.config)
    with pushd(hypaint):
        training_model = instantiate_from_config(config.model)

    candidates = [
        candidate_report(torch, path, obj, infer_state)
        for path, obj in discover_candidates(training_model)
    ]
    ordered = sorted(
        candidates,
        key=lambda item: (
            not item["key_set_exact_match"],
            item["shape_mismatch_count"],
            -(item["overlap_count"]),
            item["mean_of_mean_abs_delta"]
            if item["mean_of_mean_abs_delta"] is not None
            else 1e9,
            item["candidate_path"],
        ),
    )
    if args.max_sample_keys > 0:
        ordered = ordered[: args.max_sample_keys]
    report = {
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        "config": str(args.config.expanduser().resolve()),
        "target_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
        "infer_key_count": len(infer_state),
        "candidate_count": len(ordered),
        "recommendation": recommendation(ordered),
        "candidates": ordered,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)

    print("Phase 2J.1 true-PBR init equivalence probe")
    print(f"  HY21: {hy21}")
    print(f"  HYPAINT: {hypaint}")
    print(f"  config: {args.config.expanduser().resolve()}")
    print(f"  infer_key_count: {len(infer_state)}")
    print(f"  candidate_count: {len(ordered)}")
    rec = report["recommendation"]
    print(f"  recommendation_status: {rec['recommendation_status']}")
    print(f"  recommended_candidate_path: {rec['recommended_candidate_path']}")
    print(f"  mean_of_mean_abs_delta: {rec['mean_of_mean_abs_delta']}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2J1_TRUEPBR_INIT_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
