#!/usr/bin/env python3
"""Compare Phase 2G.3 checkpoint and inference UNet keyspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAPPINGS = [
    ("as_is", ""),
    ("strip:unet.unet.", "unet.unet."),
    ("strip:unet.unet.unet.", "unet.unet.unet."),
    ("strip:model.unet.", "model.unet."),
    ("strip:model.diffusion_model.", "model.diffusion_model."),
]
RECOMMEND_THRESHOLD = 0.8
SAMPLE_LIMIT = 20


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_keys(report: dict[str, Any]) -> list[str]:
    for name in ("state_dict_all_keys", "state_dict_keys", "all_keys", "keys"):
        values = report.get(name)
        if isinstance(values, list):
            return [str(value) for value in values]
    values = report.get("state_dict_keys_sample", [])
    return [str(value) for value in values] if isinstance(values, list) else []


def infer_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = report.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def candidate_keys(candidate: dict[str, Any]) -> list[str]:
    for name in ("all_keys", "keys", "state_dict_all_keys"):
        values = candidate.get(name)
        if isinstance(values, list):
            return [str(value) for value in values]
    values = candidate.get("keys_sample", [])
    return [str(value) for value in values] if isinstance(values, list) else []


def choose_infer_candidate(report: dict[str, Any]) -> dict[str, Any]:
    candidates = infer_candidates(report)
    if not candidates:
        return {
            "attribute_path": "",
            "class_name": "",
            "key_count": 0,
            "all_keys": [],
        }
    return max(candidates, key=lambda item: len(candidate_keys(item)))


def mapped_checkpoint_keys(keys: list[str], prefix: str) -> list[str]:
    if not prefix:
        return keys
    return [key[len(prefix):] for key in keys if key.startswith(prefix)]


def sample_sorted(values: set[str], limit: int = SAMPLE_LIMIT) -> list[str]:
    return sorted(values)[:limit]


def compare_mapping(checkpoint: list[str], infer: list[str], name: str, prefix: str) -> dict[str, Any]:
    mapped = mapped_checkpoint_keys(checkpoint, prefix)
    checkpoint_set = set(mapped)
    infer_set = set(infer)
    matched = checkpoint_set & infer_set
    missing_infer = infer_set - checkpoint_set
    extra_checkpoint = checkpoint_set - infer_set
    infer_count = len(infer_set)
    checkpoint_count = len(checkpoint_set)
    return {
        "name": name,
        "prefix": prefix,
        "mapped_checkpoint_key_count": checkpoint_count,
        "infer_key_count": infer_count,
        "overlap_count": len(matched),
        "overlap_ratio_vs_infer": (len(matched) / infer_count) if infer_count else 0.0,
        "overlap_ratio_vs_checkpoint": (len(matched) / checkpoint_count) if checkpoint_count else 0.0,
        "sample_matched_keys": sample_sorted(matched),
        "sample_missing_inference_keys": sample_sorted(missing_infer),
        "sample_extra_checkpoint_keys": sample_sorted(extra_checkpoint),
    }


def compare_reports(checkpoint_report: dict[str, Any], infer_report: dict[str, Any]) -> dict[str, Any]:
    ckpt_keys = checkpoint_keys(checkpoint_report)
    infer_candidate = choose_infer_candidate(infer_report)
    infer_keys = candidate_keys(infer_candidate)
    mappings = [
        compare_mapping(ckpt_keys, infer_keys, name, prefix)
        for name, prefix in MAPPINGS
    ]
    best = max(
        mappings,
        key=lambda item: (
            item["overlap_ratio_vs_infer"],
            item["overlap_ratio_vs_checkpoint"],
            item["overlap_count"],
        ),
    )
    if best["overlap_count"] > 0 and best["overlap_ratio_vs_infer"] >= RECOMMEND_THRESHOLD:
        recommendation = {
            "status": "RECOMMENDED",
            "mapping": best["name"],
            "prefix_to_strip": best["prefix"],
            "reason": (
                f"{best['overlap_count']} keys overlap; "
                f"{best['overlap_ratio_vs_infer']:.3f} of inference keys covered"
            ),
        }
    else:
        recommendation = {
            "status": "UNKNOWN",
            "mapping": "UNKNOWN",
            "prefix_to_strip": "",
            "reason": "No candidate mapping reached the overlap threshold.",
        }

    return {
        "checkpoint_key_count": len(ckpt_keys),
        "infer_candidate": {
            "attribute_path": infer_candidate.get("attribute_path", ""),
            "class_name": infer_candidate.get("class_name", ""),
            "key_count": len(infer_keys),
        },
        "recommendation": recommendation,
        "mappings": mappings,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    rec = report["recommendation"]
    lines = [
        "# Phase 2G.3 Keyspace Compare",
        "",
        f"checkpoint key count: `{report['checkpoint_key_count']}`",
        f"inference candidate: `{report['infer_candidate']['attribute_path']}`",
        f"inference key count: `{report['infer_candidate']['key_count']}`",
        "",
        "## Recommendation",
        f"- status: `{rec['status']}`",
        f"- mapping: `{rec['mapping']}`",
        f"- prefix to strip: `{rec['prefix_to_strip']}`",
        f"- reason: {rec['reason']}",
        "",
        "## Candidate Mappings",
    ]
    for item in report["mappings"]:
        lines.extend(
            [
                f"### `{item['name']}`",
                f"- mapped checkpoint keys: `{item['mapped_checkpoint_key_count']}`",
                f"- inference keys: `{item['infer_key_count']}`",
                f"- overlap count: `{item['overlap_count']}`",
                f"- overlap vs inference: `{item['overlap_ratio_vs_infer']:.3f}`",
                f"- overlap vs checkpoint: `{item['overlap_ratio_vs_checkpoint']:.3f}`",
                "",
                "Matched samples:",
            ]
        )
        for key in item["sample_matched_keys"]:
            lines.append(f"- `{key}`")
        lines.append("")
        lines.append("Missing inference key samples:")
        for key in item["sample_missing_inference_keys"]:
            lines.append(f"- `{key}`")
        lines.append("")
        lines.append("Extra checkpoint key samples:")
        for key in item["sample_extra_checkpoint_keys"]:
            lines.append(f"- `{key}`")
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 2G.3 keyspaces.")
    parser.add_argument("--checkpoint-json", required=True, type=Path)
    parser.add_argument("--infer-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = compare_reports(load_json(args.checkpoint_json), load_json(args.infer_json))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)

    print("Phase 2G.3 keyspace compare")
    print(f"  checkpoint_key_count: {report['checkpoint_key_count']}")
    print(f"  infer_key_count: {report['infer_candidate']['key_count']}")
    print(f"  recommendation: {report['recommendation']['mapping']}")
    print(f"  status: {report['recommendation']['status']}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2G3_KEYSPACE_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
