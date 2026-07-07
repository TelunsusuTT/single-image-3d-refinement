#!/usr/bin/env python3
"""Phase 2M.3A one-case LoRA inference smoke.

This runtime script runs one corrected-input full80 eval case through the
official base Hunyuan3D-Paint pipeline, then injects and loads the Phase 2M
adapter-only LoRA state and runs the same case at a configurable adapter scale.
It never merges LoRA into the base model and never saves a full Hunyuan model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_HY21 = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1")
DEFAULT_EVAL_CASES = PROJECT_ROOT / "outputs" / "phase2l" / "datav2_frame_panels" / "full80_eval_truepbr500" / "eval_cases.json"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300"
DEFAULT_ADAPTER_PATH = DEFAULT_ADAPTER_DIR / "adapter_final.pt"
DEFAULT_ADAPTER_CONFIG = DEFAULT_ADAPTER_DIR / "adapter_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_infer_smoke_scale075"
OFFICIAL_WORK_ROOT = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_project_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if is_relative_to(resolved, OFFICIAL_WORK_ROOT):
        raise RuntimeError(f"Output dir must not be under Hunyuan3D2.1_Work: {resolved}")
    allowed_root = (PROJECT_ROOT / "outputs" / "phase2m").resolve()
    if not is_relative_to(resolved, allowed_root):
        raise RuntimeError(f"Output dir must be under outputs/phase2m: {resolved}")
    return resolved


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


def format_scale_label(scale: float) -> str:
    return f"scale{int(round(scale * 100)):03d}"


def nonzero_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} missing: {resolved}")
    if resolved.stat().st_size <= 0:
        raise RuntimeError(f"{label} is zero-size: {resolved}")
    return resolved


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(f"No eval cases found in {path}")
    return [case for case in cases if isinstance(case, dict)]


def select_eval_case(cases: list[dict[str, Any]], case_id: str | None = None, case_index: int = 0) -> dict[str, Any]:
    if case_id:
        for case in cases:
            if str(case.get("item_id", "")) == case_id:
                return case
        raise KeyError(f"case_id not found: {case_id}")
    candidates = [case for case in cases if str(case.get("selected_input_view", "")) == "005"]
    if not candidates:
        candidates = cases
    if case_index < 0 or case_index >= len(candidates):
        raise IndexError(f"case_index {case_index} out of range for {len(candidates)} candidate case(s)")
    return candidates[case_index]


def validate_case(case: dict[str, Any]) -> dict[str, str]:
    item_id = str(case.get("item_id", ""))
    eval_split = str(case.get("eval_split", ""))
    selected_input_view = str(case.get("selected_input_view", ""))
    if not item_id:
        raise RuntimeError("Selected case is missing item_id")
    if not eval_split:
        raise RuntimeError(f"{item_id}: selected case is missing eval_split")
    if selected_input_view != "005":
        raise RuntimeError(f"{item_id}: expected selected_input_view=005, got {selected_input_view!r}")
    case_dir = Path(str(case.get("case_dir", ""))).expanduser().resolve()
    mesh = nonzero_file(Path(str(case.get("case_input_mesh") or case_dir / "input" / "mesh.glb")), "case input mesh")
    image = nonzero_file(Path(str(case.get("case_input_image") or case_dir / "input" / "image.png")), "case input image")
    return {
        "item_id": item_id,
        "eval_split": eval_split,
        "selected_input_view": selected_input_view,
        "case_dir": str(case_dir),
        "mesh_path": str(mesh),
        "input_image_path": str(image),
    }


def output_paths(output_root: Path, eval_split: str, item_id: str, scale: float) -> dict[str, Path]:
    scale_label = format_scale_label(scale)
    base_dir = output_root / "base" / eval_split / item_id
    lora_dir = output_root / f"lora_{scale_label}" / eval_split / item_id
    return {
        "base_dir": base_dir,
        "base_obj": base_dir / "base_textured_mesh.obj",
        "base_glb": base_dir / "base_textured_mesh.glb",
        "lora_dir": lora_dir,
        "lora_obj": lora_dir / f"lora_{scale_label}_textured_mesh.obj",
        "lora_glb": lora_dir / f"lora_{scale_label}_textured_mesh.glb",
    }


def validate_adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("backend") != "local_linear_fallback":
        raise RuntimeError(f"Expected local_linear_fallback backend, got {config.get('backend')!r}")
    target_names = config.get("target_names")
    if not isinstance(target_names, list) or len(target_names) != 128:
        raise RuntimeError("adapter_config target_names must contain exactly 128 entries")
    if int(config.get("target_count", len(target_names))) != 128:
        raise RuntimeError(f"adapter_config target_count expected 128, got {config.get('target_count')!r}")
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    if safety.get("merge_into_base") is not False:
        raise RuntimeError("adapter_config safety.merge_into_base must be false")
    if safety.get("save_pretrained_full_model") is not False:
        raise RuntimeError("adapter_config safety.save_pretrained_full_model must be false")
    return {
        "backend": "local_linear_fallback",
        "target_names": [str(name) for name in target_names],
        "target_count": 128,
        "rank": int(config.get("rank", 4)),
        "alpha": float(config.get("alpha", 4.0)),
        "dropout": float(config.get("dropout", 0.0)),
    }


def set_lora_runtime_scale(model: Any, lora_scale: float) -> int:
    """Set LoRA scale as an absolute multiplier, not an accumulated product."""
    updated = 0
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "scale"):
            if not hasattr(module, "_phase2m_base_lora_scale"):
                module._phase2m_base_lora_scale = float(module.scale)
            module.scale = float(module._phase2m_base_lora_scale) * float(lora_scale)
            updated += 1
    if updated == 0:
        raise RuntimeError("No LoRA modules were found for runtime scaling")
    return updated


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def make_initial_summary(args: argparse.Namespace, case_info: dict[str, str], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "phase": "2M.3A",
        "status": "PLANNED",
        "adapter_path": str(args.adapter_path),
        "adapter_config": str(args.adapter_config),
        "lora_scale": float(args.lora_scale),
        "case_id": case_info["item_id"],
        "eval_split": case_info["eval_split"],
        "selected_input_view": case_info["selected_input_view"],
        "input_image_path": case_info["input_image_path"],
        "mesh_path": case_info["mesh_path"],
        "base_output_glb_path": str(paths["base_glb"]),
        "output_glb_path": str(paths["lora_glb"]),
        "lora_output_glb_path": str(paths["lora_glb"]),
        "no_merge_into_base": True,
        "save_full_hunyuan_model": False,
        "success": False,
    }


def run_dry_run(summary: dict[str, Any], output_root: Path) -> int:
    summary = dict(summary)
    summary["status"] = "DRY_RUN_OK"
    summary["dry_run"] = True
    write_summary(output_root / "smoke_summary.json", summary)
    print("Phase 2M.3A LoRA inference smoke dry run")
    print(f"  case_id: {summary['case_id']}")
    print(f"  lora_scale: {summary['lora_scale']}")
    print(f"  output_dir: {output_root}")
    print("PHASE2M_M3A_LORA_INFER_SMOKE_DRY_RUN_OK")
    return 0


def run_real(args: argparse.Namespace, summary: dict[str, Any], paths: dict[str, Path]) -> int:
    hy21, hypaint = resolve_official_paths()
    prepend_pythonpath(SRC_ROOT)
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    # Heavy imports are intentionally runtime-only. Codex/local checks should not
    # import Hunyuan, torch, or load adapter tensors.
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore
    from hy3dft.lora.io import load_adapter_state_dict as read_adapter_state_dict
    from hy3dft.lora.local_linear import load_adapter_state_dict as apply_adapter_state_dict
    from hy3dft.lora.peft_lora import assert_only_lora_trainable, inject_lora

    adapter_spec = validate_adapter_config(load_json(args.adapter_config))
    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.device = args.device
    official_config_paths = set_absolute_official_config_paths(conf, hypaint)

    print("Phase 2M.3A LoRA inference smoke")
    print(f"  HY21: {hy21}")
    print(f"  HYPAINT: {hypaint}")
    print(f"  case_id: {summary['case_id']}")
    print(f"  input_image: {summary['input_image_path']}")
    print(f"  mesh: {summary['mesh_path']}")
    print(f"  adapter: {args.adapter_path}")
    print(f"  lora_scale: {args.lora_scale}")

    paths["base_dir"].mkdir(parents=True, exist_ok=True)
    paths["lora_dir"].mkdir(parents=True, exist_ok=True)
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    base_result = paint_pipeline(
        mesh_path=summary["mesh_path"],
        image_path=summary["input_image_path"],
        output_mesh_path=str(paths["base_obj"]),
        use_remesh=False,
        save_glb=True,
    )
    print(f"  base_output_mesh: {base_result}")

    unet = paint_pipeline.models["multiview_model"].pipeline.unet
    injection = inject_lora(
        unet,
        adapter_spec["target_names"],
        rank=adapter_spec["rank"],
        alpha=adapter_spec["alpha"],
        dropout=adapter_spec["dropout"],
        backend=adapter_spec["backend"],
    )
    assert_only_lora_trainable(unet)
    adapter_state = read_adapter_state_dict(args.adapter_path)
    load_result = apply_adapter_state_dict(unet, adapter_state, strict=True)
    scaled_module_count = set_lora_runtime_scale(unet, args.lora_scale)

    lora_result = paint_pipeline(
        mesh_path=summary["mesh_path"],
        image_path=summary["input_image_path"],
        output_mesh_path=str(paths["lora_obj"]),
        use_remesh=False,
        save_glb=True,
    )
    print(f"  lora_output_mesh: {lora_result}")

    base_glb_exists = paths["base_glb"].is_file() and paths["base_glb"].stat().st_size > 0
    lora_glb_exists = paths["lora_glb"].is_file() and paths["lora_glb"].stat().st_size > 0
    if not base_glb_exists and not paths["base_obj"].is_file():
        raise RuntimeError(f"Base output missing: {paths['base_glb']} or {paths['base_obj']}")
    if not lora_glb_exists and not paths["lora_obj"].is_file():
        raise RuntimeError(f"LoRA output missing: {paths['lora_glb']} or {paths['lora_obj']}")

    summary.update(
        {
            "status": "OK",
            "success": True,
            "dry_run": False,
            "resolved_hy21": str(hy21),
            "resolved_hypaint": str(hypaint),
            "official_config_paths": official_config_paths,
            "base_output_mesh_path": str(base_result),
            "lora_output_mesh_path": str(lora_result),
            "output_glb_exists": lora_glb_exists,
            "base_output_glb_exists": base_glb_exists,
            "lora_injection_backend": injection.backend,
            "loaded_adapter_missing_keys": list(load_result.get("missing_keys", [])),
            "loaded_adapter_unexpected_keys": list(load_result.get("unexpected_keys", [])),
            "scaled_lora_module_count": scaled_module_count,
        }
    )
    write_summary(args.output_dir / "smoke_summary.json", summary)
    print("PHASE2M_M3A_LORA_INFER_SMOKE_OK")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one corrected-input Phase 2M LoRA inference smoke.")
    parser.add_argument("--eval-cases-json", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--lora-scale", type=float, default=0.75)
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.eval_cases_json = args.eval_cases_json.expanduser().resolve()
    args.adapter_path = nonzero_file(args.adapter_path, "adapter")
    args.adapter_config = nonzero_file(args.adapter_config, "adapter config")
    args.output_dir = ensure_project_output_dir(args.output_dir)
    if args.lora_scale <= 0:
        raise ValueError("--lora-scale must be positive")

    adapter_spec = validate_adapter_config(load_json(args.adapter_config))
    cases = load_eval_cases(args.eval_cases_json)
    case = select_eval_case(cases, case_id=args.case_id, case_index=args.case_index)
    case_info = validate_case(case)
    paths = output_paths(args.output_dir, case_info["eval_split"], case_info["item_id"], args.lora_scale)
    summary = make_initial_summary(args, case_info, paths)
    summary["adapter_backend"] = adapter_spec["backend"]
    summary["target_count"] = adapter_spec["target_count"]

    try:
        if args.dry_run:
            return run_dry_run(summary, args.output_dir)
        return run_real(args, summary, paths)
    except Exception as exc:
        summary.update(
            {
                "status": "FAIL",
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_summary(args.output_dir / "smoke_summary.json", summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
