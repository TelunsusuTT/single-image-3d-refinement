#!/usr/bin/env python3
"""Static readiness checks for Data v2 full80 true-PBR training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEX_SUFFIXES = [".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"]
COND_SUFFIXES = ["_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"]
VIEW_IDS = ["000", "001", "002", "003", "004", "005"]
PBR_TOKEN = "hunyuan3d-paintpbr-v2-1"


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def contains_scalar(config_text: str, key: str, value: str) -> bool:
    pattern = rf"(?m)^\s*{re.escape(key)}\s*:\s*{re.escape(value)}\s*(?:#.*)?$"
    return re.search(pattern, config_text) is not None


def expected_files(sample_dir: Path) -> list[Path]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    files = [render_tex / "transforms.json"]
    for view_id in VIEW_IDS:
        files.extend(render_tex / f"{view_id}{suffix}" for suffix in TEX_SUFFIXES)
        files.extend(render_cond / f"{view_id}{suffix}" for suffix in COND_SUFFIXES)
    return files


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    size = resolved.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {resolved}")
    return {"path": str(resolved), "exists": exists, "size_bytes": size}


def check_dir(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_dir()
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    return {"path": str(resolved), "exists": exists}


def check_examples_json(path: Path, expected_count: int, label: str, errors: list[str]) -> dict[str, Any]:
    result = check_file(path, f"{label} examples JSON", errors)
    samples: list[dict[str, Any]] = []
    if not result["exists"]:
        return {**result, "expected_count": expected_count, "actual_count": 0, "samples": samples}
    try:
        data = load_json(Path(result["path"]))
    except json.JSONDecodeError as exc:
        errors.append(f"{label} examples JSON invalid: {exc}")
        data = []
    if not isinstance(data, list):
        errors.append(f"{label} examples JSON must be a list")
        data = []
    if len(data) != expected_count:
        errors.append(f"{label} examples count {len(data)} != expected {expected_count}")
    for value in data:
        sample_errors: list[str] = []
        if not isinstance(value, str):
            errors.append(f"{label} examples entry is not a string: {value!r}")
            continue
        sample_dir = Path(value).expanduser()
        if not sample_dir.is_absolute():
            sample_errors.append("sample path is not absolute")
        render_tex = sample_dir / "render_tex"
        render_cond = sample_dir / "render_cond"
        if not sample_dir.is_dir():
            sample_errors.append("sample_dir missing")
        if not render_tex.is_dir():
            sample_errors.append("render_tex missing")
        if not render_cond.is_dir():
            sample_errors.append("render_cond missing")
        missing_files = [str(file_path) for file_path in expected_files(sample_dir) if not file_path.is_file()]
        sample_errors.extend(f"missing file: {file_path}" for file_path in missing_files)
        errors.extend(f"{label} {sample_dir}: {error}" for error in sample_errors)
        samples.append(
            {
                "sample_dir": str(sample_dir),
                "render_tex_exists": render_tex.is_dir(),
                "render_cond_exists": render_cond.is_dir(),
                "missing_count": len(missing_files),
                "ok": not sample_errors,
            }
        )
    return {**result, "expected_count": expected_count, "actual_count": len(data), "samples": samples}


def check_training_yaml(path: Path, config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    result = check_file(path, "training YAML", errors)
    checks: dict[str, Any] = {
        "contains_pbr_token": False,
        "contains_pbr_source": False,
        "has_learning_rate": False,
        "has_max_steps": False,
        "has_every_n_train_steps": False,
        "has_save_top_k_minus_1": False,
        "has_save_last_false": False,
        "has_save_weights_only_true": False,
        "contains_train_examples": False,
        "contains_val_examples": False,
        "contains_checkpoint_dir": False,
        "forbidden_hits": [],
    }
    if not result["exists"]:
        return {**result, **checks}
    text = Path(result["path"]).read_text(encoding="utf-8")
    steps = str(config["steps"])
    learning_rate = str(config["learning_rate"])
    pbr_source = str(resolve_project_path(config["official_pbr_source_dir"]))
    checkpoint_dir = str(resolve_project_path(config["output_checkpoint_dir"]))
    checks.update(
        {
            "contains_pbr_token": PBR_TOKEN in text,
            "contains_pbr_source": pbr_source in text,
            "has_learning_rate": contains_scalar(text, "base_learning_rate", learning_rate),
            "has_max_steps": contains_scalar(text, "max_steps", steps),
            "has_every_n_train_steps": contains_scalar(text, "every_n_train_steps", steps),
            "has_save_top_k_minus_1": contains_scalar(text, "save_top_k", "-1"),
            "has_save_last_false": contains_scalar(text, "save_last", "false"),
            "has_save_weights_only_true": contains_scalar(text, "save_weights_only", "true"),
            "contains_train_examples": str(config["train_examples_json"]) in text,
            "contains_val_examples": str(config["val_examples_json"]) in text,
            "contains_checkpoint_dir": checkpoint_dir in text,
            "forbidden_hits": [token for token in config.get("forbidden_checkpoint_markers", []) if token in text],
        }
    )
    for key, message in (
        ("contains_pbr_token", f"training YAML does not contain {PBR_TOKEN}"),
        ("contains_pbr_source", "training YAML does not contain official PBR source path"),
        ("has_learning_rate", f"training YAML does not contain base_learning_rate: {learning_rate}"),
        ("has_max_steps", f"training YAML does not contain max_steps: {steps}"),
        ("has_every_n_train_steps", f"training YAML does not contain every_n_train_steps: {steps}"),
        ("has_save_top_k_minus_1", "training YAML does not contain save_top_k: -1"),
        ("has_save_last_false", "training YAML does not contain save_last: false"),
        ("has_save_weights_only_true", "training YAML does not contain save_weights_only: true"),
        ("contains_train_examples", "training YAML does not contain train examples JSON path"),
        ("contains_val_examples", "training YAML does not contain val examples JSON path"),
        ("contains_checkpoint_dir", "training YAML does not contain checkpoint output dir"),
    ):
        if not checks[key]:
            errors.append(message)
    for token in checks["forbidden_hits"]:
        errors.append(f"training YAML contains forbidden marker: {token}")
    return {**result, **checks}


def check_forbidden_markers(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    fields = [
        "experiment_name",
        "dataset_name",
        "training_yaml",
        "train_examples_json",
        "val_examples_json",
        "test_examples_json",
        "output_checkpoint_dir",
        "official_pbr_source_dir",
    ]
    hits: list[dict[str, str]] = []
    for field in fields:
        value = str(config.get(field, ""))
        for token in config.get("forbidden_checkpoint_markers", []):
            if token in value:
                hits.append({"field": field, "token": token, "value": value})
                errors.append(f"config field {field} contains forbidden marker {token}: {value}")
    return {"hits": hits}


def check_checkpoint_dir(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    path = resolve_project_path(config["output_checkpoint_dir"])
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    existing_ckpts = sorted(str(path) for path in resolved.glob("*.ckpt"))
    expected_leaf = "datav2_frame_full80_truepbr_500_lr1e6"
    if resolved.name != expected_leaf:
        errors.append(f"checkpoint dir leaf {resolved.name!r} != expected {expected_leaf!r}")
    if config.get("allow_existing_checkpoints", False) is False and existing_ckpts:
        errors.append(f"checkpoint dir already contains .ckpt files: {resolved}")
    return {"path": str(resolved), "exists_or_created": resolved.is_dir(), "existing_ckpts": existing_ckpts}


def check_split_file(config: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    split_path = resolve_project_path(config.get("split_file", ""))
    result = check_file(split_path, "full101 split file", errors)
    if not result["exists"]:
        return result
    try:
        data = load_json(Path(result["path"]))
    except json.JSONDecodeError as exc:
        errors.append(f"full101 split file invalid: {exc}")
        return {**result, "valid_json": False}
    split_counts = {key: len(value) for key, value in data.items() if isinstance(value, list)} if isinstance(data, dict) else {}
    return {**result, "valid_json": isinstance(data, dict), "split_counts": split_counts}


def readiness_report(config_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    config_result = check_file(config_path, "training config JSON", errors)
    config = load_json(Path(config_result["path"])) if config_result["exists"] else {}
    if not isinstance(config, dict):
        errors.append("training config JSON must contain an object")
        config = {}
    report: dict[str, Any] = {"config_json": config_result, "config": config}
    if not config:
        report["ok"] = False
        report["errors"] = errors
        return report

    report["forbidden_marker_scan"] = check_forbidden_markers(config, errors)
    report["training_yaml"] = check_training_yaml(resolve_project_path(config["training_yaml"]), config, errors)
    report["examples"] = {
        "train": check_examples_json(Path(config["train_examples_json"]), int(config["train_count_expected"]), "train", errors),
        "val": check_examples_json(Path(config["val_examples_json"]), int(config["val_count_expected"]), "val", errors),
        "test": check_examples_json(Path(config["test_examples_json"]), int(config["test_count_expected"]), "test", errors),
    }
    report["split_file"] = check_split_file(config, errors)
    report["official_pbr_source_dir"] = check_dir(resolve_project_path(config["official_pbr_source_dir"]), "official PBR source dir", errors)
    if PBR_TOKEN not in str(resolve_project_path(config["official_pbr_source_dir"])):
        errors.append(f"official PBR source path does not contain {PBR_TOKEN}")
    report["checkpoint_dir"] = check_checkpoint_dir(config, errors)
    report["ok"] = not errors
    report["errors"] = errors
    return report


def report_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config.get("readiness_report_dir", "outputs/phase2l/datav2_frame_panels/full80_train_readiness"))


def write_reports(report: dict[str, Any]) -> None:
    root = report_dir(report.get("config", {}))
    out_json = root / "readiness_summary.json"
    out_md = root / "readiness_summary.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    config = report.get("config", {})
    examples = report.get("examples", {})
    lines = [
        "# Phase 2L.6B Full80 Training Readiness",
        "",
        f"experiment: `{config.get('experiment_name', '')}`",
        f"status: `{'OK' if report['ok'] else 'FAIL'}`",
        "",
        "| Split | Expected | Actual | JSON |",
        "|---|---:|---:|---|",
    ]
    for split in ("train", "val", "test"):
        item = examples.get(split, {})
        lines.append(
            f"| `{split}` | {item.get('expected_count', 0)} | {item.get('actual_count', 0)} | `{item.get('path', '')}` |"
        )
    lines.extend(
        [
            "",
            f"checkpoint dir: `{report.get('checkpoint_dir', {}).get('path', '')}`",
            f"official PBR source exists: `{report.get('official_pbr_source_dir', {}).get('exists', False)}`",
        ]
    )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in report["errors"])
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    config = report.get("config", {})
    print("Phase 2L.6B full80 training readiness")
    print(f"  experiment: {config.get('experiment_name', '')}")
    for split, item in report.get("examples", {}).items():
        print(f"  {split}: expected={item['expected_count']} actual={item['actual_count']} path={item['path']}")
    print(f"  training_yaml: {report.get('training_yaml', {}).get('path', '')}")
    print(f"  checkpoint_dir: {report.get('checkpoint_dir', {}).get('path', '')}")
    print(f"  official_pbr_source_exists: {report.get('official_pbr_source_dir', {}).get('exists', False)}")
    print(f"  errors: {len(report['errors'])}")
    for error in report["errors"]:
        print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2L6B_FULL80_TRAINING_READINESS_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Data v2 frame full80 training readiness.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = readiness_report(args.config)
    write_reports(report)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
