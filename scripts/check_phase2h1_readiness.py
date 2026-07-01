#!/usr/bin/env python3
"""Check Phase 2H.1 conservative 50-step recovery readiness."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_DIRPATH = (
    "/vol/bitbucket/ct1022/hy3dpaint_finetune/"
    "checkpoints/pilot_v1_conservative_50_lr1e6"
)
FORBIDDEN_COLLAPSED_TOKEN = "pilot_v1_overfit_500"


def load_examples(examples_json: Path, errors: list[str]) -> list[Path]:
    if not examples_json.is_file():
        errors.append(f"examples JSON missing: {examples_json}")
        return []

    try:
        data = json.loads(examples_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"examples JSON is not valid JSON: {exc}")
        return []

    if not isinstance(data, list):
        errors.append("examples JSON must contain a list")
        return []

    sample_dirs: list[Path] = []
    for index, value in enumerate(data, start=1):
        if not isinstance(value, str):
            errors.append(f"examples JSON entry {index} is not a string")
            continue
        sample_dir = Path(value).expanduser()
        if not sample_dir.is_absolute():
            errors.append(f"sample path is not absolute: {sample_dir}")
        sample_dirs.append(sample_dir)
    return sample_dirs


def check_sample(sample_dir: Path) -> dict[str, Any]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    errors: list[str] = []
    if not sample_dir.is_dir():
        errors.append("sample directory missing")
    if not render_tex.is_dir():
        errors.append("render_tex missing")
    if not render_cond.is_dir():
        errors.append("render_cond missing")
    return {
        "sample_dir": str(sample_dir),
        "sample_dir_exists": sample_dir.is_dir(),
        "render_tex_exists": render_tex.is_dir(),
        "render_cond_exists": render_cond.is_dir(),
        "ok": not errors,
        "errors": errors,
    }


def contains_scalar(config_text: str, key: str, value: str) -> bool:
    pattern = rf"(?m)^\s*{re.escape(key)}\s*:\s*{re.escape(value)}\s*(?:#.*)?$"
    return re.search(pattern, config_text) is not None


def parse_config_number(value: str) -> float | None:
    cleaned = value.strip().strip("'\"")
    try:
        return float(cleaned)
    except ValueError:
        return None


def contains_learning_rate_1e6(config_text: str) -> bool:
    pattern = r"(?m)^\s*base_learning_rate\s*:\s*([^\s#]+)"
    for match in re.finditer(pattern, config_text):
        parsed = parse_config_number(match.group(1))
        if parsed is not None and math.isclose(parsed, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            return True
    return False


def check_checkpoint_root(checkpoint_root: Path, errors: list[str]) -> tuple[bool, list[str]]:
    try:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"checkpoint root cannot be created: {checkpoint_root}: {exc}")
        return False, []

    if not checkpoint_root.is_dir():
        errors.append(f"checkpoint root is not a directory: {checkpoint_root}")
        return False, []

    existing_ckpts = sorted(str(path) for path in checkpoint_root.rglob("*.ckpt"))
    if existing_ckpts:
        errors.append(
            f"checkpoint root already contains {len(existing_ckpts)} .ckpt file(s)"
        )
    return not existing_ckpts, existing_ckpts


def check_readiness(
    examples_json: Path,
    config: Path,
    expected_count: int,
    checkpoint_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    resolved_examples_json = examples_json.expanduser().resolve()
    resolved_config = config.expanduser().resolve()
    resolved_checkpoint_root = checkpoint_root.expanduser().resolve()

    sample_dirs = load_examples(resolved_examples_json, errors)
    sample_results = [check_sample(sample_dir) for sample_dir in sample_dirs]

    if len(sample_dirs) != expected_count:
        errors.append(f"sample count {len(sample_dirs)} != expected-count {expected_count}")
    for result in sample_results:
        errors.extend(
            f"{result['sample_dir']}: {error}"
            for error in result["errors"]
        )

    config_contains_examples_json = False
    config_has_max_steps_50 = False
    config_has_base_learning_rate_1e6 = False
    config_has_every_n_train_steps_50 = False
    config_has_save_top_k_minus_one = False
    config_has_save_weights_only_true = False
    config_has_checkpoint_dirpath = False
    config_contains_forbidden_checkpoint = False

    if not resolved_config.is_file():
        errors.append(f"config missing: {resolved_config}")
    else:
        config_text = resolved_config.read_text(encoding="utf-8")
        config_contains_examples_json = str(resolved_examples_json) in config_text
        config_has_max_steps_50 = contains_scalar(config_text, "max_steps", "50")
        config_has_base_learning_rate_1e6 = contains_learning_rate_1e6(config_text)
        config_has_every_n_train_steps_50 = contains_scalar(
            config_text,
            "every_n_train_steps",
            "50",
        )
        config_has_save_top_k_minus_one = contains_scalar(config_text, "save_top_k", "-1")
        config_has_save_weights_only_true = contains_scalar(
            config_text,
            "save_weights_only",
            "true",
        )
        config_has_checkpoint_dirpath = contains_scalar(
            config_text,
            "dirpath",
            EXPECTED_CHECKPOINT_DIRPATH,
        )
        config_contains_forbidden_checkpoint = FORBIDDEN_COLLAPSED_TOKEN in config_text

        if not config_contains_examples_json:
            errors.append(
                "config does not contain examples JSON path: "
                f"{resolved_examples_json}"
            )
        if not config_has_max_steps_50:
            errors.append("config does not contain max_steps: 50")
        if not config_has_base_learning_rate_1e6:
            errors.append("config does not contain base_learning_rate: 1e-6")
        if not config_has_every_n_train_steps_50:
            errors.append("config does not contain every_n_train_steps: 50")
        if not config_has_save_top_k_minus_one:
            errors.append("config does not contain save_top_k: -1")
        if not config_has_save_weights_only_true:
            errors.append("config does not contain save_weights_only: true")
        if not config_has_checkpoint_dirpath:
            errors.append(
                "config checkpoint dirpath does not point to "
                "checkpoints/pilot_v1_conservative_50_lr1e6"
            )
        if config_contains_forbidden_checkpoint:
            errors.append(
                "config contains forbidden collapsed-checkpoint token: "
                f"{FORBIDDEN_COLLAPSED_TOKEN}"
            )

    checkpoint_root_empty, existing_ckpts = check_checkpoint_root(
        resolved_checkpoint_root,
        errors,
    )

    return {
        "examples_json": str(resolved_examples_json),
        "config": str(resolved_config),
        "expected_count": expected_count,
        "sample_count": len(sample_dirs),
        "checkpoint_root": str(resolved_checkpoint_root),
        "samples": sample_results,
        "config_exists": resolved_config.is_file(),
        "config_contains_examples_json": config_contains_examples_json,
        "config_has_max_steps_50": config_has_max_steps_50,
        "config_has_base_learning_rate_1e6": config_has_base_learning_rate_1e6,
        "config_has_every_n_train_steps_50": config_has_every_n_train_steps_50,
        "config_has_save_top_k_minus_one": config_has_save_top_k_minus_one,
        "config_has_save_weights_only_true": config_has_save_weights_only_true,
        "config_has_checkpoint_dirpath": config_has_checkpoint_dirpath,
        "config_contains_forbidden_checkpoint": config_contains_forbidden_checkpoint,
        "checkpoint_root_empty": checkpoint_root_empty,
        "existing_ckpts": existing_ckpts,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2H.1 conservative recovery readiness")
    print(f"  examples_json: {report['examples_json']}")
    print(f"  config: {report['config']}")
    print(f"  expected_count: {report['expected_count']}")
    print(f"  sample_count: {report['sample_count']}")
    print(f"  checkpoint_root: {report['checkpoint_root']}")
    print(f"  config_exists: {report['config_exists']}")
    print(f"  config_contains_examples_json: {report['config_contains_examples_json']}")
    print(f"  config_has_max_steps_50: {report['config_has_max_steps_50']}")
    print(
        "  config_has_base_learning_rate_1e6: "
        f"{report['config_has_base_learning_rate_1e6']}"
    )
    print(
        "  config_has_every_n_train_steps_50: "
        f"{report['config_has_every_n_train_steps_50']}"
    )
    print(f"  config_has_save_top_k_minus_one: {report['config_has_save_top_k_minus_one']}")
    print(f"  config_has_save_weights_only_true: {report['config_has_save_weights_only_true']}")
    print(f"  config_has_checkpoint_dirpath: {report['config_has_checkpoint_dirpath']}")
    print(
        "  config_contains_forbidden_checkpoint: "
        f"{report['config_contains_forbidden_checkpoint']}"
    )
    print(f"  checkpoint_root_empty: {report['checkpoint_root_empty']}")
    for index, sample in enumerate(report["samples"], start=1):
        print(f"  [{index}] {sample['sample_dir']}")
        print(f"      sample_dir_exists: {sample['sample_dir_exists']}")
        print(f"      render_tex_exists: {sample['render_tex_exists']}")
        print(f"      render_cond_exists: {sample['render_cond_exists']}")
    if report["existing_ckpts"]:
        print("  existing_ckpts:")
        for ckpt in report["existing_ckpts"]:
            print(f"    - {ckpt}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print(f"dataset length = {report['sample_count']}")
        print("PHASE2H1_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase 2H.1 conservative recovery readiness."
    )
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_count < 1:
        print("ERROR: --expected-count must be >= 1", file=sys.stderr)
        return 2
    report = check_readiness(
        args.examples_json,
        args.config,
        args.expected_count,
        args.checkpoint_root,
    )
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
