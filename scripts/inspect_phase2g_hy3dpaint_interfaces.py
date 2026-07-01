#!/usr/bin/env python3
"""Read-only inspection of official Hunyuan3D-Paint inference/loading interfaces."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".json"}
MAX_TEXT_BYTES = 2_000_000
SEARCH_TERMS = [
    "checkpoint",
    "ckpt",
    "load_state_dict",
    "load_from_checkpoint",
    "torch.load",
    "init_from_ckpt",
    "resume",
    "HunyuanPaint",
    "argparse",
    "add_argument",
    "DiffusionPipeline.from_pretrained",
    "model loading",
]


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue
        except OSError:
            continue
        files.append(path)
    return sorted(files)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_matches(root: Path, files: list[Path]) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, int]]:
    matches: dict[str, dict[str, list[dict[str, Any]]]] = {}
    counts = {term: 0 for term in SEARCH_TERMS}
    lower_terms = [(term, term.lower()) for term in SEARCH_TERMS]
    for path in files:
        rel = str(path.relative_to(root))
        text = read_text(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            for term, lower_term in lower_terms:
                if lower_term in lower:
                    counts[term] += 1
                    matches.setdefault(rel, {}).setdefault(term, []).append(
                        {
                            "line": line_no,
                            "text": line.strip()[:240],
                        }
                    )
    return matches, counts


def find_add_argument_lines(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in files:
        text = read_text(path)
        rel = str(path.relative_to(root))
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "add_argument" in line:
                results.append({"file": rel, "line": line_no, "text": line.strip()})
    return results


def has_demo_cli(demo_text: str) -> bool:
    return any(token in demo_text for token in ("argparse", "add_argument", "sys.argv"))


def checkpoint_arg_hits(argument_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for line in argument_lines:
        text = line["text"].lower()
        if any(token in text for token in ("checkpoint", "ckpt", "resume", "weight")):
            hits.append(line)
    return hits


def model_loading_hits(matches: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    loading_terms = {
        "torch.load",
        "load_state_dict",
        "load_from_checkpoint",
        "DiffusionPipeline.from_pretrained",
        "resume",
        "ckpt",
        "checkpoint",
    }
    hits: list[dict[str, Any]] = []
    for rel, by_term in matches.items():
        for term, term_hits in by_term.items():
            if term not in loading_terms:
                continue
            for hit in term_hits[:20]:
                hits.append(
                    {
                        "file": rel,
                        "line": hit["line"],
                        "term": term,
                        "text": hit["text"],
                    }
                )
    return hits


def inspect_interfaces(hypaint: Path) -> dict[str, Any]:
    root = hypaint.expanduser().resolve()
    if not root.is_dir():
        return {
            "hypaint": str(root),
            "ok": False,
            "errors": [f"hy3dpaint path missing: {root}"],
        }

    files = iter_text_files(root)
    matches, counts = find_matches(root, files)
    argument_lines = find_add_argument_lines(root, files)
    checkpoint_args = checkpoint_arg_hits(argument_lines)

    demo_path = root / "demo.py"
    demo_exists = demo_path.is_file()
    demo_text = read_text(demo_path) if demo_exists else ""
    demo_checkpoint_args = [hit for hit in checkpoint_args if hit["file"] == "demo.py"]
    demo_has_cli_args = has_demo_cli(demo_text)
    demo_uses_hardcoded_case = "./assets/case_1/mesh.glb" in demo_text and "./assets/case_1/image.png" in demo_text

    train_path = root / "train.py"
    available_entrypoints = []
    if demo_exists:
        available_entrypoints.append(
            {
                "path": "demo.py",
                "kind": "inference",
                "notes": "Quick inference entrypoint; currently hardcodes assets/case_1 inputs.",
            }
        )
    if train_path.is_file():
        available_entrypoints.append(
            {
                "path": "train.py",
                "kind": "training/checkpoint-loading reference",
                "notes": "Contains resume/checkpoint loading logic, but is not an inference entrypoint.",
            }
        )

    return {
        "hypaint": str(root),
        "ok": True,
        "files_scanned": [str(path.relative_to(root)) for path in files],
        "available_inference_entrypoints": available_entrypoints,
        "demo_py": {
            "exists": demo_exists,
            "has_cli_args": demo_has_cli_args,
            "uses_hardcoded_case_1": demo_uses_hardcoded_case,
            "checkpoint_args": demo_checkpoint_args,
        },
        "argparse_argument_lines": argument_lines,
        "checkpoint_argument_locations": checkpoint_args,
        "checkpoint_path_configurable_in_demo": bool(demo_checkpoint_args),
        "custom_checkpoint_argument_found": bool(checkpoint_args),
        "likely_model_loading_locations": model_loading_hits(matches),
        "pattern_counts": counts,
        "matches_by_file": matches,
        "assessment": {
            "custom_checkpoint_argument_status": (
                "demo.py exposes no custom checkpoint argument"
                if not demo_checkpoint_args
                else "demo.py appears to expose a checkpoint-like argument"
            ),
            "wrapper_or_config_patch_may_be_needed": not bool(demo_checkpoint_args),
        },
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    lines = ["# Phase 2G Hunyuan3D-Paint Interface Inspection", ""]
    lines.append(f"hy3dpaint path: `{report['hypaint']}`")
    lines.append("")
    if not report.get("ok"):
        lines.append("## Errors")
        for error in report.get("errors", []):
            lines.append(f"- {error}")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Entrypoints")
    for entry in report["available_inference_entrypoints"]:
        lines.append(f"- `{entry['path']}` ({entry['kind']}): {entry['notes']}")
    if not report["available_inference_entrypoints"]:
        lines.append("- No obvious entrypoints found.")

    demo = report["demo_py"]
    lines.extend(
        [
            "",
            "## demo.py",
            f"- exists: `{demo['exists']}`",
            f"- has argparse/sys.argv CLI args: `{demo['has_cli_args']}`",
            f"- uses hardcoded `./assets/case_1` mesh/image: `{demo['uses_hardcoded_case_1']}`",
            f"- checkpoint path configurable in demo.py: `{report['checkpoint_path_configurable_in_demo']}`",
        ]
    )
    if not report["checkpoint_path_configurable_in_demo"]:
        lines.append("- No custom checkpoint argument was found in `demo.py`.")

    lines.extend(["", "## Checkpoint-Like Arguments"])
    if report["checkpoint_argument_locations"]:
        for hit in report["checkpoint_argument_locations"]:
            lines.append(f"- `{hit['file']}:{hit['line']}`: `{hit['text']}`")
    else:
        lines.append("- No checkpoint-like argparse arguments found.")

    lines.extend(["", "## Likely Model Loading Locations"])
    for hit in report["likely_model_loading_locations"][:40]:
        lines.append(f"- `{hit['file']}:{hit['line']}` [{hit['term']}]: `{hit['text']}`")
    if len(report["likely_model_loading_locations"]) > 40:
        lines.append(f"- ... {len(report['likely_model_loading_locations']) - 40} more hits in JSON.")

    lines.extend(
        [
            "",
            "## Assessment",
            f"- custom checkpoint argument status: {report['assessment']['custom_checkpoint_argument_status']}",
            f"- wrapper or config patch may be needed: `{report['assessment']['wrapper_or_config_patch_may_be_needed']}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Hunyuan3D-Paint interfaces without importing modules.")
    parser.add_argument("--hypaint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = inspect_interfaces(args.hypaint)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"wrote JSON: {args.out_json}")
    print(f"wrote Markdown: {args.out_md}")
    if report.get("ok") and not report.get("checkpoint_path_configurable_in_demo"):
        print("NO_DEMO_CHECKPOINT_ARG_FOUND")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
