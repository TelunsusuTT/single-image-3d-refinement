#!/usr/bin/env python3
"""Static readiness checks for Phase 2M.3B LoRA multi-scale pilot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts import check_phase2m_lora_infer_smoke_ready as m3a_ready
except ImportError:  # pragma: no cover
    import check_phase2m_lora_infer_smoke_ready as m3a_ready  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300"
DEFAULT_ADAPTER_PATH = DEFAULT_ADAPTER_DIR / "adapter_final.pt"
DEFAULT_ADAPTER_CONFIG = DEFAULT_ADAPTER_DIR / "adapter_config.json"
DEFAULT_EVAL_CASES = PROJECT_ROOT / "outputs" / "phase2l" / "datav2_frame_panels" / "full80_eval_truepbr500" / "eval_cases.json"
DEFAULT_M3A_SUMMARY = PROJECT_ROOT / "outputs" / "phase2m" / "lora_infer_smoke_scale075" / "smoke_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot"
DEFAULT_SCRIPT = PROJECT_ROOT / "scripts" / "phase2m_lora_multiscale_pilot.py"
DEFAULT_SBATCH = PROJECT_ROOT / "env" / "run_phase2m_lora_multiscale_pilot_a100.sbatch"
DEFAULT_SCALES = (0.5, 0.75, 1.0)
REQUIRED_PARTITION = "a100"
BANNED_SBATCH_TOKENS = ("gpgpuC", "--constraint=a100")

Report = m3a_ready.Report


def load_json(path: Path) -> dict[str, Any]:
    return m3a_ready.load_json(path)


def path_text(path: Path, project_root: Path) -> str:
    return m3a_ready.path_text(path, project_root)


def check_scales(scales: list[float], report: Report) -> None:
    expected = list(DEFAULT_SCALES)
    normalized = [float(scale) for scale in scales]
    if normalized == expected:
        report.ok("scale list exactly [0.5, 0.75, 1.0]")
    else:
        report.fail(f"scale list must be exactly {expected}, got {normalized}")


def validate_eval_case_splits(path: Path, report: Report, project_root: Path) -> None:
    if not path.is_file():
        report.fail(f"eval_cases.json missing: {path_text(path, project_root)}")
        return
    try:
        data = load_json(path)
    except Exception as exc:
        report.fail(f"eval_cases.json is not readable: {exc}")
        return
    cases = data.get("cases")
    if not isinstance(cases, list):
        report.fail("eval_cases.json must contain a cases list")
        return
    split_counts: dict[str, int] = {}
    selected_005_count = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        split = str(case.get("eval_split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        if str(case.get("selected_input_view", "")) == "005":
            selected_005_count += 1
    if split_counts.get("val", 0) > 0:
        report.ok(f"eval_cases contains val case(s): {split_counts['val']}")
    else:
        report.fail("eval_cases must contain at least one val case")
    if split_counts.get("test", 0) > 0:
        report.ok(f"eval_cases contains test case(s): {split_counts['test']}")
    else:
        report.fail("eval_cases must contain at least one test case")
    train_like = sum(count for split, count in split_counts.items() if split not in {"val", "test"} and ("train" in split.lower() or "sanity" in split.lower()))
    if train_like > 0:
        report.ok(f"eval_cases contains train-like sanity case(s): {train_like}")
    else:
        report.warn("eval_cases has no train-like sanity case; pilot script will fail if none is available")
    if selected_005_count >= 3:
        report.ok(f"eval_cases has selected_input_view=005 case(s): {selected_005_count}")
    else:
        report.fail(f"eval_cases should have at least 3 selected_input_view=005 cases, got {selected_005_count}")


def validate_m3a_summary(path: Path, report: Report, project_root: Path) -> None:
    if not path.is_file():
        report.fail(f"M3A smoke summary missing: {path_text(path, project_root)}")
        return
    try:
        summary = load_json(path)
    except Exception as exc:
        report.fail(f"M3A smoke summary is not readable: {exc}")
        return
    if summary.get("status") == "OK" and summary.get("success") is True:
        report.ok("M3A smoke summary status == OK")
    else:
        report.fail(f"M3A smoke summary expected status OK/success true, got status={summary.get('status')!r} success={summary.get('success')!r}")


def validate_sbatch(path: Path, report: Report, project_root: Path, dry_run: bool) -> None:
    if not path.is_file():
        if dry_run:
            report.warn(f"sbatch missing in dry-run: {path_text(path, project_root)}")
            return
        report.fail(f"sbatch missing: {path_text(path, project_root)}")
        return
    text = path.read_text(encoding="utf-8")
    for token in BANNED_SBATCH_TOKENS:
        if token in text:
            report.fail(f"sbatch contains banned token {token}: {path_text(path, project_root)}")
    partitions = m3a_ready.extract_sbatch_partitions(text)
    if partitions == [REQUIRED_PARTITION]:
        report.ok("sbatch uses #SBATCH -p a100")
    else:
        report.fail(f"sbatch partition must be exactly {REQUIRED_PARTITION!r}, got {partitions!r}")
    gres = m3a_ready.extract_sbatch_gres(text)
    if "gpu:1" in gres:
        report.ok("sbatch uses #SBATCH --gres=gpu:1")
    else:
        report.fail(f"sbatch must include --gres=gpu:1, got {gres!r}")
    required = [
        "check_phase2m_lora_multiscale_pilot_ready.py",
        "phase2m_lora_multiscale_pilot.py",
        "--scales 0.5 0.75 1.0",
    ]
    for token in required:
        if token in text:
            report.ok(f"sbatch contains {token}")
        else:
            report.fail(f"sbatch must contain {token}")


def check_readiness(args: argparse.Namespace) -> Report:
    report = Report()
    project_root = args.project_root.resolve()
    min_adapter_bytes = int(float(args.min_adapter_mb) * 1024 * 1024)
    adapter_path = args.adapter_path.resolve()
    adapter_config = args.adapter_config.resolve()
    eval_cases = args.eval_cases_json.resolve()
    output_dir = args.output_dir.resolve()
    script = args.script.resolve()
    sbatch = args.sbatch.resolve()
    m3a_summary = args.m3a_summary.resolve()

    m3a_ready.check_nonzero(adapter_path, "adapter_final.pt", report, project_root, min_bytes=min_adapter_bytes)
    m3a_ready.validate_adapter_config(adapter_config, report, project_root)
    validate_eval_case_splits(eval_cases, report, project_root)
    m3a_ready.check_nonzero(script, "LoRA multiscale pilot script", report, project_root)
    m3a_ready.validate_output_dir(output_dir, report, project_root)
    check_scales(args.scales, report)
    validate_m3a_summary(m3a_summary, report, project_root)
    validate_sbatch(sbatch, report, project_root, args.dry_run)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2M.3B LoRA multi-scale pilot readiness without running inference.")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--eval-cases-json", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--m3a-summary", type=Path, default=DEFAULT_M3A_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--sbatch", type=Path, default=DEFAULT_SBATCH)
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--min-adapter-mb", type=float, default=1.0)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args)
    print("Phase 2M.3B LoRA multi-scale pilot readiness")
    for line in report.lines:
        print(line)
    if report.errors:
        print(f"PHASE2M_M3B_LORA_MULTISCALE_READY_FAIL errors={len(report.errors)} warnings={len(report.warnings)}")
        return 1
    print("PHASE2M_M3B_LORA_MULTISCALE_READY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
