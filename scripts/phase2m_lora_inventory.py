#!/usr/bin/env python3
"""Inventory safe LoRA target modules for Phase 2M.

This script is intentionally runtime-only: it imports official Hunyuan modules
inside main() after environment paths are resolved. Codex/tests can compile it
without loading Hunyuan, torch checkpoints, or model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_HY21 = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1")
DEFAULT_OUT_JSON = PROJECT_ROOT / "outputs" / "phase2m" / "lora_module_inventory.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "outputs" / "phase2m" / "lora_module_inventory.md"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.lora.peft_lora import backend_availability  # noqa: E402
from hy3dft.lora.targeting import inventory_linear_modules, select_lora_targets  # noqa: E402


def resolve_official_paths() -> tuple[Path, Path]:
    hy21 = Path(os.environ.get("HY21") or DEFAULT_HY21).expanduser().resolve()
    hypaint = Path(os.environ.get("HYPAINT") or hy21 / "hy3dpaint").expanduser().resolve()
    if not hy21.is_dir():
        raise FileNotFoundError(f"HY21 path missing: {hy21}")
    if not hypaint.is_dir():
        raise FileNotFoundError(f"HYPAINT path missing: {hypaint}")
    return hy21, hypaint


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if text not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([text, *parts])


def set_absolute_official_config_paths(conf: Any, hypaint: Path) -> dict[str, str]:
    paths = {
        "multiview_cfg_path": hypaint / "cfgs" / "hunyuan-paint-pbr.yaml",
        "realesrgan_ckpt_path": hypaint / "ckpt" / "RealESRGAN_x4plus.pth",
    }
    for attr, path in paths.items():
        if hasattr(conf, attr):
            setattr(conf, attr, str(path))
    return {key: str(value) for key, value in paths.items()}


def load_inference_unet(max_num_view: int, resolution: int, device: str) -> tuple[Any, dict[str, str]]:
    hy21, hypaint = resolve_official_paths()
    prepend_pythonpath(SRC_ROOT)
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    # Heavy official imports happen only in the A100 runtime path.
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.device = device
    config_paths = set_absolute_official_config_paths(conf, hypaint)
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    unet = paint_pipeline.models["multiview_model"].pipeline.unet
    metadata = {
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        "unet_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
        **config_paths,
    }
    return unet, metadata


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    selected = report["selected_target_names"]
    group_counts = report["group_counts"]
    lines = [
        "# Phase 2M LoRA Module Inventory",
        "",
        f"- preset: `{report['preset']}`",
        f"- backend policy: `{report['backend_policy']}`",
        f"- inference target: `{report['official_paths']['unet_path']}`",
        f"- linear module count: {report['linear_module_count']}",
        f"- selected target count: {report['selected_target_count']}",
        "",
        "## Backend Availability",
        "",
    ]
    for name, available in sorted(report["backend_availability"].items()):
        lines.append(f"- {name}: {available}")
    lines.extend(["", "## Linear Module Groups", ""])
    for group, count in sorted(group_counts.items()):
        lines.append(f"- {group}: {count}")
    lines.extend(["", "## Selected Targets", ""])
    for target in selected:
        lines.append(f"- `{target}`")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2M LoRA target inventory.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--preset", default="ref_dino", choices=("ref_dino",))
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    unet, official_paths = load_inference_unet(args.max_num_view, args.resolution, args.device)
    inventory = inventory_linear_modules(unet)
    selected_targets = select_lora_targets(unet, preset=args.preset)
    group_counts = Counter(item.group for item in inventory)
    report = {
        "phase": "2M",
        "preset": args.preset,
        "backend_policy": "local_linear_fallback_by_default; PEFT/Diffusers detected but not auto-used",
        "backend_availability": backend_availability(),
        "official_paths": official_paths,
        "linear_module_count": len(inventory),
        "selected_target_count": len(selected_targets),
        "group_counts": dict(sorted(group_counts.items())),
        "selected_target_names": selected_targets,
        "linear_modules": [asdict(item) for item in inventory],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"wrote inventory json: {args.out_json}")
    print(f"wrote inventory md: {args.out_md}")
    print(f"selected_target_count={len(selected_targets)}")
    print("PHASE2M_M0_INVENTORY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
