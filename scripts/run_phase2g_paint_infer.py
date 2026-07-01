#!/usr/bin/env python3
"""Project-local Hunyuan3D-Paint inference wrapper for Phase 2G."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_HY21 = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1")
FINETUNED_CHECKPOINT_PREFIX = "unet."
FINETUNED_TARGET_PATH = "paint_pipeline.models['multiview_model'].pipeline.unet"


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
        "use_remesh": not args.no_remesh,
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
        "use_remesh",
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


def transformed_state_dict(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    transformed: dict[str, Any] = {}
    for key, value in state.items():
        text_key = str(key)
        if text_key.startswith(prefix):
            transformed[text_key[len(prefix):]] = value
    return transformed


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def verify_state_compatibility(transformed: dict[str, Any], target_state: dict[str, Any]) -> None:
    transformed_keys = set(transformed.keys())
    target_keys = set(target_state.keys())
    if transformed_keys != target_keys:
        missing = sorted(target_keys - transformed_keys)[:20]
        extra = sorted(transformed_keys - target_keys)[:20]
        raise RuntimeError(
            "Fine-tuned checkpoint key set does not match inference target. "
            f"missing_sample={missing} extra_sample={extra}"
        )
    mismatches = []
    for key in sorted(target_keys):
        checkpoint_shape = shape_of(transformed[key])
        target_shape = shape_of(target_state[key])
        if checkpoint_shape != target_shape:
            mismatches.append((key, checkpoint_shape, target_shape))
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise RuntimeError(f"Fine-tuned checkpoint tensor shape mismatches: {mismatches}")


def load_finetuned_checkpoint_into_pipeline(checkpoint: Path, paint_pipeline: Any) -> dict[str, Any]:
    import torch  # type: ignore

    target = paint_pipeline.models["multiview_model"].pipeline.unet
    target_state = target.state_dict()
    checkpoint_obj = torch_load_cpu(torch, checkpoint)
    state = extract_state_dict(checkpoint_obj)
    transformed = transformed_state_dict(state, FINETUNED_CHECKPOINT_PREFIX)
    verify_state_compatibility(transformed, target_state)
    load_result = target.load_state_dict(transformed, strict=True)
    return {
        "checkpoint": str(checkpoint),
        "target_path": FINETUNED_TARGET_PATH,
        "checkpoint_prefix": FINETUNED_CHECKPOINT_PREFIX,
        "raw_state_dict_key_count": len(state),
        "transformed_key_count": len(transformed),
        "target_key_count": len(target_state),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }


def run_real_inference(args: argparse.Namespace, plan: dict[str, Any]) -> int:
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
    print(f"  checkpoint: {plan['checkpoint']}")
    print(f"  output_mesh: {plan['planned_output_mesh']}")
    print(f"  planned_output_glb: {plan['planned_output_glb']}")
    print(f"  use_remesh: {plan['use_remesh']}")

    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    if args.mode == "finetuned":
        if args.checkpoint is None:
            raise RuntimeError("--checkpoint is required for --mode finetuned")
        load_summary = load_finetuned_checkpoint_into_pipeline(args.checkpoint.expanduser().resolve(), paint_pipeline)
        plan["finetuned_checkpoint_load"] = load_summary
        write_plan(plan)
        print(f"  finetuned_target_path: {load_summary['target_path']}")
        print(f"  transformed_key_count: {load_summary['transformed_key_count']}")
        print(f"  target_key_count: {load_summary['target_key_count']}")
        print("PHASE2G5_FINETUNED_CHECKPOINT_LOADED_OK")

    result = paint_pipeline(
        mesh_path=plan["input_mesh"],
        image_path=plan["input_image"],
        output_mesh_path=plan["planned_output_mesh"],
        use_remesh=plan["use_remesh"],
        save_glb=True,
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
    parser.add_argument(
        "--no-remesh",
        action="store_true",
        help="Disable official remeshing and keep input geometry fixed.",
    )
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
