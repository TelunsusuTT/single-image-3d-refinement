#!/usr/bin/env python3
"""Compare Phase 2G.3 checkpoint and inference UNet keyspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAPPINGS = [
    ("as_is", ""),
    ("strip:unet.", "unet."),
    ("strip:unet.unet.", "unet.unet."),
    ("strip:unet.unet.unet.", "unet.unet.unet."),
    ("strip:model.", "model."),
    ("strip:model.unet.", "model.unet."),
    ("strip:model.diffusion_model.", "model.diffusion_model."),
]
RECOMMEND_THRESHOLD = 0.95
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


def mapped_checkpoint_keys(keys: list[str], prefix: str) -> list[str]:
    if not prefix:
        return keys
    return [key[len(prefix):] for key in keys if key.startswith(prefix)]


def sample_sorted(values: set[str], limit: int = SAMPLE_LIMIT) -> list[str]:
    return sorted(values)[:limit]


def compare_mapping(
    checkpoint: list[str],
    candidate: dict[str, Any],
    transform: str,
    prefix: str,
) -> dict[str, Any]:
    infer = candidate_keys(candidate)
    mapped = mapped_checkpoint_keys(checkpoint, prefix)
    checkpoint_set = set(mapped)
    infer_set = set(infer)
    matched = checkpoint_set & infer_set
    missing_infer = infer_set - checkpoint_set
    extra_checkpoint = checkpoint_set - infer_set
    infer_count = len(infer_set)
    checkpoint_count = len(checkpoint_set)
    return {
        "candidate_path": str(candidate.get("attribute_path", "")),
        "candidate_class": str(candidate.get("class_name", "")),
        "transform": transform,
        "strip_prefix": prefix,
        "infer_key_count": infer_count,
        "transformed_checkpoint_key_count": checkpoint_count,
        "overlap_count": len(matched),
        "overlap_ratio_vs_infer": (len(matched) / infer_count) if infer_count else 0.0,
        "overlap_ratio_vs_checkpoint_subset": (len(matched) / checkpoint_count) if checkpoint_count else 0.0,
        "sample_matched_keys": sample_sorted(matched),
        "sample_missing_infer_keys": sample_sorted(missing_infer),
        "sample_extra_checkpoint_keys": sample_sorted(extra_checkpoint),
    }


def is_recommended(item: dict[str, Any]) -> bool:
    return (
        item["overlap_count"] > 0
        and item["overlap_ratio_vs_infer"] >= RECOMMEND_THRESHOLD
        and item["overlap_ratio_vs_checkpoint_subset"] >= RECOMMEND_THRESHOLD
    )


def best_recommendation(comparisons: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [item for item in comparisons if is_recommended(item)]
    if not passing:
        return None
    return min(
        passing,
        key=lambda item: (
            -item["overlap_count"],
            -item["overlap_ratio_vs_infer"],
            -item["overlap_ratio_vs_checkpoint_subset"],
            len(item["strip_prefix"]),
        ),
    )


def compare_reports(checkpoint_report: dict[str, Any], infer_report: dict[str, Any]) -> dict[str, Any]:
    ckpt_keys = checkpoint_keys(checkpoint_report)
    candidates = infer_candidates(infer_report)
    comparisons = [
        compare_mapping(ckpt_keys, candidate, transform, prefix)
        for candidate in candidates
        for transform, prefix in MAPPINGS
    ]
    best = best_recommendation(comparisons)

    if best is None:
        recommendation_status = "UNKNOWN"
        recommended_target_path = ""
        recommended_target_class = ""
        recommended_transform = "UNKNOWN"
        recommended_prefix = ""
        reason = "No candidate mapping reached both 0.95 overlap thresholds."
    else:
        recommendation_status = "RECOMMENDED"
        recommended_target_path = best["candidate_path"]
        recommended_target_class = best["candidate_class"]
        recommended_transform = best["transform"]
        recommended_prefix = best["strip_prefix"]
        reason = (
            f"{best['overlap_count']} keys overlap; "
            f"{best['overlap_ratio_vs_infer']:.3f} of inference keys covered and "
            f"{best['overlap_ratio_vs_checkpoint_subset']:.3f} of transformed checkpoint keys covered"
        )

    return {
        "checkpoint_key_count": len(ckpt_keys),
        "inference_candidate_count": len(candidates),
        "recommendation_status": recommendation_status,
        "recommended_target_path": recommended_target_path,
        "recommended_target_class": recommended_target_class,
        "recommended_transform": recommended_transform,
        "recommended_prefix_to_strip": recommended_prefix,
        "recommendation_reason": reason,
        "recommendation": {
            "status": recommendation_status,
            "mapping": recommended_transform,
            "prefix_to_strip": recommended_prefix,
            "target_path": recommended_target_path,
            "target_class": recommended_target_class,
            "reason": reason,
        },
        "comparisons": comparisons,
        "mappings": comparisons,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Phase 2G.3 Keyspace Compare",
        "",
        f"checkpoint key count: `{report['checkpoint_key_count']}`",
        f"inference candidate count: `{report['inference_candidate_count']}`",
        "",
        "## Recommendation",
        f"- recommendation status: `{report['recommendation_status']}`",
        f"- recommended target path: `{report['recommended_target_path']}`",
        f"- recommended target class: `{report['recommended_target_class']}`",
        f"- recommended transform: `{report['recommended_transform']}`",
        f"- prefix to strip: `{report['recommended_prefix_to_strip']}`",
        f"- reason: {report['recommendation_reason']}",
        "",
        "## Candidate And Transform Comparisons",
    ]
    for item in report["comparisons"]:
        lines.extend(
            [
                f"### `{item['candidate_path']}` with `{item['transform']}`",
                f"- candidate class: `{item['candidate_class']}`",
                f"- transformed checkpoint keys: `{item['transformed_checkpoint_key_count']}`",
                f"- inference keys: `{item['infer_key_count']}`",
                f"- overlap count: `{item['overlap_count']}`",
                f"- overlap vs inference: `{item['overlap_ratio_vs_infer']:.3f}`",
                f"- overlap vs checkpoint subset: `{item['overlap_ratio_vs_checkpoint_subset']:.3f}`",
                "",
                "Matched samples:",
            ]
        )
        for key in item["sample_matched_keys"]:
            lines.append(f"- `{key}`")
        lines.append("")
        lines.append("Missing inference key samples:")
        for key in item["sample_missing_infer_keys"]:
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
    print(f"  inference_candidate_count: {report['inference_candidate_count']}")
    print(f"  recommended_target_path: {report['recommended_target_path']}")
    print(f"  recommended_transform: {report['recommended_transform']}")
    print(f"  recommendation_status: {report['recommendation_status']}")
    print(f"  out_json: {args.out_json}")
    print(f"  out_md: {args.out_md}")
    print("PHASE2G3_KEYSPACE_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
