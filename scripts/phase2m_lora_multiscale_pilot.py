#!/usr/bin/env python3
"""Phase 2M.3B three-case LoRA multi-scale inference pilot."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

try:
    from scripts import phase2m_lora_infer_smoke as smoke
except ImportError:  # pragma: no cover - supports direct script execution.
    import phase2m_lora_infer_smoke as smoke  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CASES = PROJECT_ROOT / "outputs" / "phase2l" / "datav2_frame_panels" / "full80_eval_truepbr500" / "eval_cases.json"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300"
DEFAULT_ADAPTER_PATH = DEFAULT_ADAPTER_DIR / "adapter_final.pt"
DEFAULT_ADAPTER_CONFIG = DEFAULT_ADAPTER_DIR / "adapter_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot"
DEFAULT_SCALES = (0.5, 0.75, 1.0)


def normalize_scales(scales: list[float] | tuple[float, ...]) -> list[float]:
    normalized = [float(scale) for scale in scales]
    if any(scale <= 0 for scale in normalized):
        raise ValueError("LoRA scales must be positive")
    return normalized


def output_prefix_for_scale(scale: float) -> str:
    return f"lora_{smoke.format_scale_label(scale)}_textured_mesh"


def lora_output_paths(output_root: Path, eval_split: str, item_id: str, scale: float) -> dict[str, Path]:
    scale_label = smoke.format_scale_label(scale)
    out_dir = output_root / f"lora_{scale_label}" / eval_split / item_id
    prefix = output_prefix_for_scale(scale)
    return {
        "dir": out_dir,
        "obj": out_dir / f"{prefix}.obj",
        "glb": out_dir / f"{prefix}.glb",
    }


def base_output_paths(output_root: Path, eval_split: str, item_id: str) -> dict[str, Path]:
    out_dir = output_root / "base" / eval_split / item_id
    return {
        "dir": out_dir,
        "obj": out_dir / "base_textured_mesh.obj",
        "glb": out_dir / "base_textured_mesh.glb",
    }


def mesh_output_exists(paths: dict[str, Path]) -> bool:
    return any(paths[key].is_file() and paths[key].stat().st_size > 0 for key in ("glb", "obj"))


def choose_first_by_split(cases: list[dict[str, Any]], split_names: set[str]) -> dict[str, Any] | None:
    for case in cases:
        split = str(case.get("eval_split", ""))
        if split in split_names and str(case.get("selected_input_view", "")) == "005":
            return case
    for case in cases:
        split = str(case.get("eval_split", ""))
        if split in split_names:
            return case
    return None


def choose_train_sanity_case(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred = []
    fallback = []
    for case in cases:
        split = str(case.get("eval_split", "")).lower()
        if split in {"val", "test"}:
            continue
        if "train" in split or "sanity" in split:
            if str(case.get("selected_input_view", "")) == "005":
                preferred.append(case)
            else:
                fallback.append(case)
    return preferred[0] if preferred else (fallback[0] if fallback else None)


def select_pilot_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        choose_first_by_split(cases, {"val"}),
        choose_first_by_split(cases, {"test"}),
        choose_train_sanity_case(cases),
    ]
    missing = [label for label, case in zip(("val", "test", "train_sanity"), selected) if case is None]
    if missing:
        raise RuntimeError(f"Could not select required pilot case(s): {missing}")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in selected:
        assert case is not None
        info = smoke.validate_case(case)
        if info["item_id"] in seen:
            raise RuntimeError(f"Pilot case selection produced duplicate item_id: {info['item_id']}")
        seen.add(info["item_id"])
        enriched = dict(case)
        enriched["_phase2m_case_info"] = info
        validated.append(enriched)
    if len(validated) != 3:
        raise RuntimeError(f"Expected exactly 3 pilot cases, got {len(validated)}")
    return validated


def make_conf(max_num_view: int, resolution: int, device: str) -> tuple[Any, dict[str, str]]:
    hy21, hypaint = smoke.resolve_official_paths()
    smoke.prepend_pythonpath(smoke.SRC_ROOT)
    smoke.prepend_pythonpath(hy21)
    smoke.prepend_pythonpath(hypaint)
    from textureGenPipeline import Hunyuan3DPaintConfig  # type: ignore

    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.device = device
    metadata = {
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        **smoke.set_absolute_official_config_paths(conf, hypaint),
    }
    return conf, metadata


def new_pipeline(max_num_view: int, resolution: int, device: str) -> tuple[Any, dict[str, str]]:
    conf, metadata = make_conf(max_num_view, resolution, device)
    from textureGenPipeline import Hunyuan3DPaintPipeline  # type: ignore

    return Hunyuan3DPaintPipeline(conf), metadata


def run_base_variant(case_info: dict[str, str], paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if mesh_output_exists(paths) and not args.overwrite:
        return {
            "variant": "base",
            "status": "REUSED",
            "reused": True,
            "eval_split": case_info["eval_split"],
            "case_id": case_info["item_id"],
            "output_obj_path": str(paths["obj"]),
            "output_glb_path": str(paths["glb"]),
        }
    paths["dir"].mkdir(parents=True, exist_ok=True)
    pipeline, metadata = new_pipeline(args.max_num_view, args.resolution, args.device)
    result = pipeline(
        mesh_path=case_info["mesh_path"],
        image_path=case_info["input_image_path"],
        output_mesh_path=str(paths["obj"]),
        use_remesh=False,
        save_glb=True,
    )
    if not mesh_output_exists(paths):
        raise RuntimeError(f"Base output missing for {case_info['item_id']}: {paths['glb']} or {paths['obj']}")
    return {
        "variant": "base",
        "status": "OK",
        "reused": False,
        "eval_split": case_info["eval_split"],
        "case_id": case_info["item_id"],
        "output_mesh_path": str(result),
        "output_obj_path": str(paths["obj"]),
        "output_glb_path": str(paths["glb"]),
        "runtime_metadata": metadata,
    }


def run_lora_variant(case_info: dict[str, str], paths: dict[str, Path], scale: float, adapter_spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if mesh_output_exists(paths) and not args.overwrite:
        return {
            "variant": f"lora_{smoke.format_scale_label(scale)}",
            "status": "REUSED",
            "reused": True,
            "eval_split": case_info["eval_split"],
            "case_id": case_info["item_id"],
            "lora_scale": scale,
            "output_obj_path": str(paths["obj"]),
            "output_glb_path": str(paths["glb"]),
        }
    paths["dir"].mkdir(parents=True, exist_ok=True)
    pipeline, metadata = new_pipeline(args.max_num_view, args.resolution, args.device)

    from hy3dft.lora.io import load_adapter_state_dict as read_adapter_state_dict
    from hy3dft.lora.local_linear import load_adapter_state_dict as apply_adapter_state_dict
    from hy3dft.lora.peft_lora import assert_only_lora_trainable, inject_lora

    unet = pipeline.models["multiview_model"].pipeline.unet
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
    scaled_module_count = smoke.set_lora_runtime_scale(unet, scale)

    result = pipeline(
        mesh_path=case_info["mesh_path"],
        image_path=case_info["input_image_path"],
        output_mesh_path=str(paths["obj"]),
        use_remesh=False,
        save_glb=True,
    )
    if not mesh_output_exists(paths):
        raise RuntimeError(f"LoRA output missing for {case_info['item_id']} scale={scale}: {paths['glb']} or {paths['obj']}")
    return {
        "variant": f"lora_{smoke.format_scale_label(scale)}",
        "status": "OK",
        "reused": False,
        "eval_split": case_info["eval_split"],
        "case_id": case_info["item_id"],
        "lora_scale": scale,
        "output_mesh_path": str(result),
        "output_obj_path": str(paths["obj"]),
        "output_glb_path": str(paths["glb"]),
        "lora_injection_backend": injection.backend,
        "missing_adapter_keys": list(load_result.get("missing_keys", [])),
        "unexpected_adapter_keys": list(load_result.get("unexpected_keys", [])),
        "scaled_module_count": scaled_module_count,
        "runtime_metadata": metadata,
    }


def case_summary(case: dict[str, Any]) -> dict[str, Any]:
    info = case["_phase2m_case_info"]
    return {
        "case_id": info["item_id"],
        "eval_split": info["eval_split"],
        "selected_input_view": info["selected_input_view"],
        "input_image_path": info["input_image_path"],
        "mesh_path": info["mesh_path"],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def summarize_lora_adapter_reports(variants: list[dict[str, Any]]) -> dict[str, Any]:
    missing_keys: set[str] = set()
    unexpected_keys: set[str] = set()
    reports: list[dict[str, Any]] = []
    scaled_counts: dict[str, list[int]] = {}

    for variant in variants:
        if not str(variant.get("variant", "")).startswith("lora_"):
            continue
        scale = variant.get("lora_scale")
        scale_key = str(scale) if scale is not None else "unknown"
        report = {
            "variant": variant.get("variant"),
            "case_id": variant.get("case_id"),
            "eval_split": variant.get("eval_split"),
            "lora_scale": scale,
            "reused": bool(variant.get("reused", False)),
            "missing_adapter_keys": variant.get("missing_adapter_keys"),
            "unexpected_adapter_keys": variant.get("unexpected_adapter_keys"),
            "scaled_module_count": variant.get("scaled_module_count"),
        }
        reports.append(report)

        for key in variant.get("missing_adapter_keys") or []:
            missing_keys.add(str(key))
        for key in variant.get("unexpected_adapter_keys") or []:
            unexpected_keys.add(str(key))
        if variant.get("scaled_module_count") is not None:
            scaled_counts.setdefault(scale_key, []).append(int(variant["scaled_module_count"]))

    return {
        "missing_adapter_keys": sorted(missing_keys),
        "unexpected_adapter_keys": sorted(unexpected_keys),
        "adapter_key_report_by_variant": reports,
        "scaled_module_count_per_scale": {
            scale: sorted(set(counts)) for scale, counts in sorted(scaled_counts.items())
        },
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = smoke.load_json(args.adapter_config)
    adapter_spec = smoke.validate_adapter_config(adapter_config)
    cases = select_pilot_cases(smoke.load_eval_cases(args.eval_cases_json))
    scales = normalize_scales(args.scales)

    variants: list[dict[str, Any]] = []
    for case in cases:
        info = case["_phase2m_case_info"]
        print(f"BASE {info['eval_split']} {info['item_id']}")
        variants.append(run_base_variant(info, base_output_paths(args.output_dir, info["eval_split"], info["item_id"]), args))
        for scale in scales:
            print(f"LORA scale={scale} {info['eval_split']} {info['item_id']}")
            variants.append(run_lora_variant(info, lora_output_paths(args.output_dir, info["eval_split"], info["item_id"], scale), scale, adapter_spec, args))

    case_records = [case_summary(case) for case in cases]
    lora_adapter_report = summarize_lora_adapter_reports(variants)
    summary = {
        "phase": "2M.3B",
        "status": "OK",
        "adapter_path": str(args.adapter_path),
        "adapter_config": str(args.adapter_config),
        "backend": adapter_spec["backend"],
        "target_count": adapter_spec["target_count"],
        "scales": scales,
        "case_count": len(case_records),
        "cases": case_records,
        "output_dir": str(args.output_dir),
        "variant_count": len(variants),
        "per_variant_outputs_path": str(args.output_dir / "per_variant_outputs.json"),
        "cases_used_path": str(args.output_dir / "cases_used.json"),
        "no_merge_into_base": True,
        "save_full_hunyuan_model": False,
        "fresh_model_per_lora_scale": True,
        "success": True,
        **lora_adapter_report,
    }
    write_json(args.output_dir / "cases_used.json", {"cases": case_records})
    write_json(args.output_dir / "per_variant_outputs.json", {"variants": variants})
    write_json(args.output_dir / "pilot_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 3-case Phase 2M LoRA multi-scale pilot.")
    parser.add_argument("--eval-cases-json", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if OBJ/GLB already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.eval_cases_json = args.eval_cases_json.expanduser().resolve()
    args.adapter_path = smoke.nonzero_file(args.adapter_path, "adapter")
    args.adapter_config = smoke.nonzero_file(args.adapter_config, "adapter config")
    args.output_dir = smoke.ensure_project_output_dir(args.output_dir)
    summary: dict[str, Any] = {
        "phase": "2M.3B",
        "status": "PLANNED",
        "adapter_path": str(args.adapter_path),
        "adapter_config": str(args.adapter_config),
        "output_dir": str(args.output_dir),
        "scales": normalize_scales(args.scales),
        "no_merge_into_base": True,
        "save_full_hunyuan_model": False,
        "success": False,
    }
    try:
        summary = run_pilot(args)
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
        write_json(args.output_dir / "pilot_summary.json", summary)
        raise
    print(f"pilot_summary={args.output_dir / 'pilot_summary.json'}")
    print("PHASE2M_M3B_LORA_MULTISCALE_PILOT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
