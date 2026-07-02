#!/usr/bin/env python3
"""Check Phase 2K.1 true-PBR 200-step train-eval readiness."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SAMPLE_COUNT = 7
PBR_TOKEN = "hunyuan3d-paintpbr-v2-1"
CHECKPOINT_TOKEN = "pilot_v1_truepbr_200_lr1e6"
FORBIDDEN_CONFIG_TOKENS = [
    "sd2-community",
    "stabilityai/stable-diffusion-2-1",
    "pilot_v1_overfit_500",
    "pilot_v1_conservative_50_lr1e6",
]
OUTPUT_SUFFIXES = {".glb", ".obj"}
BASE_OUTPUT_FILES = [
    "base_textured_mesh.obj",
    "base_textured_mesh.glb",
    "base_textured_mesh.jpg",
    "base_textured_mesh_metallic.jpg",
    "base_textured_mesh_roughness.jpg",
]
PROJECT_SCRIPTS = [
    "load_phase2g4_finetuned_checkpoint_only.py",
    "compare_phase2i_base_unet_to_checkpoints.py",
    "run_phase2g_paint_infer.py",
    "make_phase2g6_texture_comparison.py",
]


def contains_scalar(config_text: str, key: str, value: str) -> bool:
    pattern = rf"(?m)^\s*{re.escape(key)}\s*:\s*{re.escape(value)}\s*(?:#.*)?$"
    return re.search(pattern, config_text) is not None


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    size = resolved.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {resolved}")
    return {"path": str(resolved), "exists": exists, "size_bytes": size}


def check_parent(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    parent = resolved.parent
    ok = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        ok = parent.is_dir()
    except OSError as exc:
        errors.append(f"{label} parent cannot be created: {parent}: {exc}")
    return {"path": str(resolved), "parent": str(parent), "parent_exists_or_created": ok}


def existing_mesh_outputs(output_dir: Path) -> list[Path]:
    resolved = output_dir.expanduser().resolve()
    if not resolved.exists():
        return []
    return sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in OUTPUT_SUFFIXES
    )


def check_examples(examples_json: Path, errors: list[str]) -> dict[str, Any]:
    result = check_file(examples_json, "examples JSON", errors)
    sample_results: list[dict[str, Any]] = []
    if not result["exists"]:
        return {**result, "sample_count": 0, "samples": sample_results}

    try:
        data = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"examples JSON is not valid JSON: {exc}")
        data = []

    if not isinstance(data, list):
        errors.append("examples JSON must contain a list")
        data = []

    for index, value in enumerate(data, start=1):
        if not isinstance(value, str):
            errors.append(f"examples JSON entry {index} is not a string")
            continue
        sample_dir = Path(value).expanduser()
        if not sample_dir.is_absolute():
            errors.append(f"sample path is not absolute: {sample_dir}")
        render_tex = sample_dir / "render_tex"
        render_cond = sample_dir / "render_cond"
        sample_errors: list[str] = []
        if not sample_dir.is_dir():
            sample_errors.append("sample directory missing")
        if not render_tex.is_dir():
            sample_errors.append("render_tex missing")
        if not render_cond.is_dir():
            sample_errors.append("render_cond missing")
        errors.extend(f"{sample_dir}: {error}" for error in sample_errors)
        sample_results.append(
            {
                "sample_dir": str(sample_dir),
                "sample_dir_exists": sample_dir.is_dir(),
                "render_tex_exists": render_tex.is_dir(),
                "render_cond_exists": render_cond.is_dir(),
                "ok": not sample_errors,
            }
        )

    sample_count = len(sample_results)
    if sample_count != EXPECTED_SAMPLE_COUNT:
        errors.append(f"sample count {sample_count} != expected {EXPECTED_SAMPLE_COUNT}")
    return {**result, "sample_count": sample_count, "samples": sample_results}


def check_config(config: Path, errors: list[str]) -> dict[str, Any]:
    result = check_file(config, "config", errors)
    checks = {
        "contains_pbr_token": False,
        "has_base_learning_rate_1e6": False,
        "has_max_steps_200": False,
        "has_every_n_train_steps_200": False,
        "has_save_top_k_minus_1": False,
        "has_save_last_false": False,
        "has_save_weights_only_true": False,
        "contains_checkpoint_token": False,
        "forbidden_hits": [],
    }
    if result["exists"]:
        text = Path(result["path"]).read_text(encoding="utf-8")
        checks["contains_pbr_token"] = PBR_TOKEN in text
        checks["has_base_learning_rate_1e6"] = contains_scalar(
            text, "base_learning_rate", "1e-6"
        )
        checks["has_max_steps_200"] = contains_scalar(text, "max_steps", "200")
        checks["has_every_n_train_steps_200"] = contains_scalar(
            text, "every_n_train_steps", "200"
        )
        checks["has_save_top_k_minus_1"] = contains_scalar(text, "save_top_k", "-1")
        checks["has_save_last_false"] = contains_scalar(text, "save_last", "false")
        checks["has_save_weights_only_true"] = contains_scalar(
            text, "save_weights_only", "true"
        )
        checks["contains_checkpoint_token"] = CHECKPOINT_TOKEN in text
        checks["forbidden_hits"] = [
            token for token in FORBIDDEN_CONFIG_TOKENS if token in text
        ]
        if not checks["contains_pbr_token"]:
            errors.append(f"config does not contain {PBR_TOKEN}")
        if not checks["has_base_learning_rate_1e6"]:
            errors.append("config does not contain base_learning_rate: 1e-6")
        if not checks["has_max_steps_200"]:
            errors.append("config does not contain max_steps: 200")
        if not checks["has_every_n_train_steps_200"]:
            errors.append("config does not contain every_n_train_steps: 200")
        if not checks["has_save_top_k_minus_1"]:
            errors.append("config does not contain save_top_k: -1")
        if not checks["has_save_last_false"]:
            errors.append("config does not contain save_last: false")
        if not checks["has_save_weights_only_true"]:
            errors.append("config does not contain save_weights_only: true")
        if not checks["contains_checkpoint_token"]:
            errors.append(f"config does not contain checkpoint token: {CHECKPOINT_TOKEN}")
        for token in checks["forbidden_hits"]:
            errors.append(f"config contains forbidden token: {token}")
    return {**result, **checks}


def check_checkpoint_root(checkpoint_root: Path, errors: list[str]) -> dict[str, Any]:
    resolved = checkpoint_root.expanduser().resolve()
    created = False
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        created = resolved.is_dir()
    except OSError as exc:
        errors.append(f"checkpoint-root cannot be created: {resolved}: {exc}")
    ckpts = sorted(str(path) for path in resolved.rglob("*.ckpt")) if resolved.is_dir() else []
    if ckpts:
        errors.append(f"checkpoint-root already contains {len(ckpts)} .ckpt file(s)")
    return {
        "path": str(resolved),
        "exists_or_created": created,
        "existing_ckpts": ckpts,
    }


def check_infer_output_dir(eval_output_root: Path, errors: list[str]) -> dict[str, Any]:
    infer_dir = eval_output_root.expanduser().resolve() / "infer" / "B075YLTF7Q"
    result = check_parent(infer_dir, "infer-output-dir", errors)
    outputs = existing_mesh_outputs(infer_dir)
    if outputs:
        errors.append(
            "infer-output-dir already contains mesh outputs: "
            + ", ".join(str(path) for path in outputs)
        )
    result["existing_mesh_outputs"] = [str(path) for path in outputs]
    return result


def check_readiness(
    examples_json: Path,
    config: Path,
    hypaint: Path,
    checkpoint_root: Path,
    case_dir: Path,
    base_dir: Path,
    eval_output_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    scripts_dir = PROJECT_ROOT / "scripts"
    resolved_hypaint = hypaint.expanduser().resolve()
    resolved_case = case_dir.expanduser().resolve()
    resolved_base = base_dir.expanduser().resolve()
    resolved_eval_root = eval_output_root.expanduser().resolve()

    if not resolved_hypaint.is_dir():
        errors.append(f"hy3dpaint path missing: {resolved_hypaint}")
    if not resolved_case.is_dir():
        errors.append(f"case-dir missing: {resolved_case}")
    if not resolved_base.is_dir():
        errors.append(f"base-dir missing: {resolved_base}")

    checks: dict[str, Any] = {
        "train_py": check_file(resolved_hypaint / "train.py", "hypaint/train.py", errors),
        "input_mesh": check_file(resolved_case / "input" / "mesh.glb", "input mesh", errors),
        "input_image": check_file(resolved_case / "input" / "image.png", "input image", errors),
    }
    for filename in BASE_OUTPUT_FILES:
        checks[f"base_{filename}"] = check_file(
            resolved_base / filename,
            f"base output {filename}",
            errors,
        )
    for filename in PROJECT_SCRIPTS:
        checks[f"script_{filename}"] = check_file(
            scripts_dir / filename,
            f"project script {filename}",
            errors,
        )

    outputs = {
        "eval_output_root": check_parent(
            resolved_eval_root, "eval-output-root", errors
        ),
        "infer_output_dir": check_infer_output_dir(resolved_eval_root, errors),
    }

    report = {
        "examples_json": check_examples(examples_json, errors),
        "config": check_config(config, errors),
        "hypaint": {"path": str(resolved_hypaint), "exists": resolved_hypaint.is_dir()},
        "checkpoint_root": check_checkpoint_root(checkpoint_root, errors),
        "case_dir": str(resolved_case),
        "base_dir": str(resolved_base),
        "eval_output_root": str(resolved_eval_root),
        "checks": checks,
        "outputs": outputs,
    }
    report["ok"] = not errors
    report["errors"] = errors
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2K.1 true-PBR 200-step train-eval readiness")
    examples = report["examples_json"]
    print(f"  examples_json: exists={examples['exists']} path={examples['path']}")
    print(f"  sample_count: {examples['sample_count']}")
    config = report["config"]
    print(f"  config: exists={config['exists']} path={config['path']}")
    print(f"  config_contains_pbr_token: {config['contains_pbr_token']}")
    print(f"  config_has_base_learning_rate_1e6: {config['has_base_learning_rate_1e6']}")
    print(f"  config_has_max_steps_200: {config['has_max_steps_200']}")
    print(f"  config_has_every_n_train_steps_200: {config['has_every_n_train_steps_200']}")
    print(f"  config_has_save_top_k_minus_1: {config['has_save_top_k_minus_1']}")
    print(f"  config_has_save_last_false: {config['has_save_last_false']}")
    print(f"  config_has_save_weights_only_true: {config['has_save_weights_only_true']}")
    print(f"  config_contains_checkpoint_token: {config['contains_checkpoint_token']}")
    print(f"  config_forbidden_hits: {config['forbidden_hits']}")
    print(f"  hypaint: exists={report['hypaint']['exists']} path={report['hypaint']['path']}")
    checkpoint_root = report["checkpoint_root"]
    print(f"  checkpoint_root: {checkpoint_root['path']}")
    print(f"  checkpoint_root_exists_or_created: {checkpoint_root['exists_or_created']}")
    print(f"  existing_ckpts: {len(checkpoint_root['existing_ckpts'])}")
    print(f"  case_dir: {report['case_dir']}")
    print(f"  base_dir: {report['base_dir']}")
    print(f"  eval_output_root: {report['eval_output_root']}")
    for name, item in report["checks"].items():
        print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
    for name, item in report["outputs"].items():
        print(f"  {name}: {item['path']}")
        print(f"      parent_exists_or_created: {item['parent_exists_or_created']}")
        if "existing_mesh_outputs" in item:
            print(f"      existing_mesh_outputs: {len(item['existing_mesh_outputs'])}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2K1_TRUEPBR200_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase 2K.1 true-PBR 200-step train-eval readiness."
    )
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--eval-output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(
        args.examples_json,
        args.config,
        args.hypaint,
        args.checkpoint_root,
        args.case_dir,
        args.base_dir,
        args.eval_output_root,
    )
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
