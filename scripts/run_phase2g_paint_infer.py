#!/usr/bin/env python3
"""Project-local Hunyuan3D-Paint inference wrapper skeleton for Phase 2G."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_HY21 = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1")


def nonzero_file(path: Path, label: str, errors: list[str]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        errors.append(f"{label} missing: {resolved}")
    elif resolved.stat().st_size <= 0:
        errors.append(f"{label} is zero-size: {resolved}")
    return resolved


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    case_dir = args.case_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    errors: list[str] = []
    if not case_dir.is_dir():
        errors.append(f"case-dir missing: {case_dir}")
    mesh = nonzero_file(case_dir / "input" / "mesh.glb", "input mesh", errors)
    image = nonzero_file(case_dir / "input" / "image.png", "input image", errors)

    checkpoint = None
    if args.mode == "finetuned":
        if args.checkpoint is None:
            errors.append("--checkpoint is required for --mode finetuned")
        else:
            checkpoint = nonzero_file(args.checkpoint, "checkpoint", errors)
    elif args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()

    plan = {
        "case_dir": str(case_dir),
        "input_mesh": str(mesh),
        "input_image": str(image),
        "output_dir": str(output_dir),
        "planned_output_mesh": str(output_dir / f"{args.mode}_textured_mesh.obj"),
        "planned_output_glb": str(output_dir / f"{args.mode}_textured_mesh.glb"),
        "mode": args.mode,
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "max_num_view": args.max_num_view,
        "resolution": args.resolution,
        "device": args.device,
        "dry_run": args.dry_run,
    }
    return plan, errors


def write_plan(plan: dict[str, Any]) -> None:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )


def run_dry_run(plan: dict[str, Any]) -> int:
    write_plan(plan)
    print("Phase 2G inference dry run")
    for key in (
        "mode",
        "input_mesh",
        "input_image",
        "output_dir",
        "planned_output_mesh",
        "planned_output_glb",
        "checkpoint",
        "max_num_view",
        "resolution",
        "device",
    ):
        print(f"  {key}: {plan[key]}")
    print("PHASE2G_INFER_DRY_RUN_OK")
    return 0


def resolve_official_paths() -> tuple[Path, Path]:
    hy21_env = os.environ.get("HY21", "")
    hy21 = Path(hy21_env).expanduser() if hy21_env else DEFAULT_HY21
    hy21 = hy21.resolve()

    hypaint_env = os.environ.get("HYPAINT", "")
    hypaint = Path(hypaint_env).expanduser() if hypaint_env else hy21 / "hy3dpaint"
    hypaint = hypaint.resolve()

    if not hy21.is_dir():
        raise RuntimeError(f"HY21 path missing: {hy21}")
    if not hypaint.is_dir():
        raise RuntimeError(f"HYPAINT path missing: {hypaint}")
    return hy21, hypaint


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def set_absolute_official_config_paths(conf: Any, hypaint: Path) -> dict[str, str]:
    cfg_path = hypaint / "cfgs" / "hunyuan-paint-pbr.yaml"
    realesrgan_path = hypaint / "ckpt" / "RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = str(cfg_path)
    conf.realesrgan_ckpt_path = str(realesrgan_path)
    return {
        "multiview_cfg_path": str(cfg_path),
        "realesrgan_ckpt_path": str(realesrgan_path),
    }


def run_real_inference(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    if args.mode == "finetuned":
        raise NotImplementedError(
            "Fine-tuned inference is intentionally not implemented yet. "
            "The official demo.py has no checkpoint argument, and the exact "
            "Lightning checkpoint key mapping to multiview_model.pipeline.unet "
            "must be confirmed on A100 before loading weights. Refusing to "
            "silently fall back to base mode."
        )

    hy21, hypaint = resolve_official_paths()
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plan["resolved_hy21"] = str(hy21)
    plan["resolved_hypaint"] = str(hypaint)
    write_plan(plan)

    # Imports are intentionally inside the non-dry-run branch so tests and
    # Codex preparation never import or execute Hunyuan.
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.device = args.device
    plan.update(set_absolute_official_config_paths(conf, hypaint))
    write_plan(plan)

    print("Phase 2G real inference")
    print(f"  mode: {args.mode}")
    print(f"  HY21: {hy21}")
    print(f"  HYPAINT: {hypaint}")
    print(f"  multiview_cfg_path: {conf.multiview_cfg_path}")
    print(f"  realesrgan_ckpt_path: {conf.realesrgan_ckpt_path}")
    print(f"  mesh: {plan['input_mesh']}")
    print(f"  image: {plan['input_image']}")
    print(f"  output_mesh: {plan['planned_output_mesh']}")

    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    result = paint_pipeline(
        mesh_path=plan["input_mesh"],
        image_path=plan["input_image"],
        output_mesh_path=plan["planned_output_mesh"],
    )
    plan["actual_output_mesh"] = str(result)
    write_plan(plan)
    print(f"Output mesh path: {result}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2G Hunyuan3D-Paint wrapper.")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("base", "finetuned"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan, errors = validate_inputs(args)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if args.dry_run:
        return run_dry_run(plan)
    return run_real_inference(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
