#!/usr/bin/env python3
"""Inspect Phase 2I training config initialization clues without PyYAML."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SD21_TOKEN = "sd2-community/stable-diffusion-2-1"
HYUNYUAN_TOKEN = "tencent/Hunyuan3D-2.1"
PBR_TOKEN = "hunyuan3d-paintpbr-v2-1"
COLLAPSED_TOKEN = "pilot_v1_overfit_500"


def clean_scalar(value: str) -> str:
    value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def extract_first_scalar(text: str, key: str) -> str | None:
    pattern = rf"(?m)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$"
    match = re.search(pattern, text)
    if not match:
        return None
    return clean_scalar(match.group(1))


def is_null_or_missing(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"", "null", "none", "~"}


def inspect_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    pretrained = extract_first_scalar(text, "pretrained_model_name_or_path")
    resume_from = extract_first_scalar(text, "resume_from")
    init_control_from = extract_first_scalar(text, "init_control_from")
    uses_sd21 = SD21_TOKEN in text
    contains_hunyuan_repo = HYUNYUAN_TOKEN in text
    contains_pbr_model = PBR_TOKEN in text
    not_confirmed = is_null_or_missing(resume_from) and pretrained == SD21_TOKEN

    return {
        "config": str(resolved),
        "base_learning_rate": extract_first_scalar(text, "base_learning_rate"),
        "max_steps": extract_first_scalar(text, "max_steps"),
        "stable_diffusion_config.pretrained_model_name_or_path": pretrained,
        "custom_pipeline": extract_first_scalar(text, "custom_pipeline"),
        "resume_from": resume_from,
        "resume_from_is_null_or_missing": is_null_or_missing(resume_from),
        "init_control_from": init_control_from,
        "init_control_from_is_null_or_missing": is_null_or_missing(init_control_from),
        "contains_tencent_hunyuan3d_2_1": contains_hunyuan_repo,
        "contains_hunyuan3d_paintpbr_v2_1": contains_pbr_model,
        "contains_sd2_community_stable_diffusion_2_1": uses_sd21,
        "contains_pilot_v1_overfit_500": COLLAPSED_TOKEN in text,
        "not_confirmed_hunyuan_pbr_finetune": not_confirmed,
        "interpretation": (
            "not confirmed as Hunyuan3D-Paint PBR fine-tune"
            if not_confirmed
            else "initialization requires review"
        ),
    }


def summarize(configs: list[Path]) -> dict[str, Any]:
    items = [inspect_config(path) for path in configs]
    return {
        "config_count": len(items),
        "configs": items,
        "any_not_confirmed_hunyuan_pbr_finetune": any(
            item["not_confirmed_hunyuan_pbr_finetune"] for item in items
        ),
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2I Training Initialization Inspection",
        "",
        f"config count: `{report['config_count']}`",
        f"any not confirmed Hunyuan PBR fine-tune: `{report['any_not_confirmed_hunyuan_pbr_finetune']}`",
        "",
    ]
    for item in report["configs"]:
        lines.extend(
            [
                f"## `{item['config']}`",
                "",
                f"- base learning rate: `{item['base_learning_rate']}`",
                f"- max steps: `{item['max_steps']}`",
                "- pretrained model: "
                f"`{item['stable_diffusion_config.pretrained_model_name_or_path']}`",
                f"- custom pipeline: `{item['custom_pipeline']}`",
                f"- resume_from: `{item['resume_from']}`",
                f"- init_control_from: `{item['init_control_from']}`",
                f"- contains `{HYUNYUAN_TOKEN}`: `{item['contains_tencent_hunyuan3d_2_1']}`",
                f"- contains `{PBR_TOKEN}`: `{item['contains_hunyuan3d_paintpbr_v2_1']}`",
                f"- contains `{SD21_TOKEN}`: `{item['contains_sd2_community_stable_diffusion_2_1']}`",
                f"- contains `{COLLAPSED_TOKEN}`: `{item['contains_pilot_v1_overfit_500']}`",
                f"- not confirmed as Hunyuan3D-Paint PBR fine-tune: `{item['not_confirmed_hunyuan_pbr_finetune']}`",
                f"- interpretation: `{item['interpretation']}`",
                "",
            ]
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect training config initialization clues for Phase 2I."
    )
    parser.add_argument("--configs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = summarize(args.configs)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print("Phase 2I training initialization inspection")
    print(f"  config_count: {report['config_count']}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2I_CONFIG_INIT_INSPECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
