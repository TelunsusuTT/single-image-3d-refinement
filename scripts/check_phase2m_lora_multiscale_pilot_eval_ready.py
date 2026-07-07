#!/usr/bin/env python3
"""Static readiness checks for Phase 2M.3C rendered-view pilot evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts import check_phase2m_lora_multiscale_pilot_ready as m3b_ready
except ImportError:  # pragma: no cover
    import check_phase2m_lora_multiscale_pilot_ready as m3b_ready  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot"
DEFAULT_PILOT_SUMMARY = DEFAULT_PILOT_ROOT / "pilot_summary.json"
DEFAULT_PER_VARIANT_OUTPUTS = DEFAULT_PILOT_ROOT / "per_variant_outputs.json"
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_rendered"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_eval"
DEFAULT_RENDER_SCRIPT = PROJECT_ROOT / "scripts" / "phase2m_render_lora_multiscale_pilot.py"
DEFAULT_AGGREGATE_SCRIPT = PROJECT_ROOT / "scripts" / "aggregate_phase2m_lora_multiscale_pilot_eval.py"
DEFAULT_BOARD_SCRIPT = PROJECT_ROOT / "scripts" / "make_phase2m_lora_multiscale_pilot_boards.py"
DEFAULT_BLENDER_WRAPPER = PROJECT_ROOT / "scripts" / "render_phase2k3_glb_views_blender.py"
DEFAULT_SBATCH = PROJECT_ROOT / "env" / "run_phase2m_lora_multiscale_pilot_eval_a100.sbatch"
REQUIRED_VARIANTS = ["base", "lora_scale050", "lora_scale075", "lora_scale100"]
BANNED_SBATCH_TOKENS = ("gpgpuC", "--constraint=a100")

Report = m3b_ready.Report


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_text(path: Path, project_root: Path) -> str:
    return m3b_ready.path_text(path, project_root)


def validate_project_output_dir(path: Path, label: str, report: Report, project_root: Path) -> None:
    try:
        resolved = path.resolve()
        allowed = (project_root / "outputs" / "phase2m").resolve()
        if resolved == allowed or allowed in resolved.parents:
            report.ok(f"{label} is under outputs/phase2m: {path_text(resolved, project_root)}")
        else:
            report.fail(f"{label} must be under outputs/phase2m, got {resolved}")
    except Exception as exc:
        report.fail(f"{label} path is invalid: {exc}")


def validate_pilot_summary(path: Path, report: Report, project_root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        report.fail(f"pilot_summary.json missing: {path_text(path, project_root)}")
        return None
    try:
        summary = load_json(path)
    except Exception as exc:
        report.fail(f"pilot_summary.json is not readable: {exc}")
        return None
    if summary.get("status") == "OK" and summary.get("success") is True:
        report.ok("pilot_summary.json status OK")
    else:
        report.fail(f"pilot_summary.json expected status OK/success true, got status={summary.get('status')!r} success={summary.get('success')!r}")
    cases = summary.get("cases")
    if isinstance(cases, list) and len(cases) == 3:
        report.ok("pilot_summary.json contains exactly 3 cases")
    else:
        report.fail(f"pilot_summary.json must contain exactly 3 cases, got {len(cases) if isinstance(cases, list) else 'not-list'}")
    return summary


def validate_variant_outputs(path: Path, summary: dict[str, Any] | None, report: Report, project_root: Path) -> None:
    if not path.is_file():
        report.fail(f"per_variant_outputs.json missing: {path_text(path, project_root)}")
        return
    try:
        data = load_json(path)
    except Exception as exc:
        report.fail(f"per_variant_outputs.json is not readable: {exc}")
        return
    variants = data.get("variants")
    if not isinstance(variants, list):
        report.fail("per_variant_outputs.json must contain a variants list")
        return
    expected_cases = [str(case.get("case_id")) for case in (summary or {}).get("cases", []) if isinstance(case, dict)]
    by_case: dict[str, set[str]] = {case_id: set() for case_id in expected_cases}
    for row in variants:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", ""))
        variant = str(row.get("variant", ""))
        if case_id in by_case:
            by_case[case_id].add(variant)
        glb = row.get("output_glb_path")
        if case_id in by_case and variant in REQUIRED_VARIANTS:
            if not glb:
                report.fail(f"missing output_glb_path for {case_id}/{variant}")
            else:
                glb_path = Path(glb)
                if glb_path.is_file() and glb_path.stat().st_size > 0:
                    report.ok(f"GLB exists for {case_id}/{variant}: {path_text(glb_path, project_root)}")
                else:
                    report.fail(f"GLB missing or empty for {case_id}/{variant}: {glb_path}")
    for case_id, found in by_case.items():
        if set(REQUIRED_VARIANTS).issubset(found):
            report.ok(f"variants include base/scale050/scale075/scale100 for {case_id}")
        else:
            report.fail(f"variants for {case_id} must include {REQUIRED_VARIANTS}, got {sorted(found)}")
    if len(variants) >= 12:
        report.ok(f"per_variant_outputs contains at least 12 variant rows: {len(variants)}")
    else:
        report.fail(f"per_variant_outputs should contain 12 rows, got {len(variants)}")


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
    partitions = m3b_ready.m3a_ready.extract_sbatch_partitions(text)
    if partitions == ["a100"]:
        report.ok("sbatch uses #SBATCH -p a100")
    else:
        report.fail(f"sbatch partition must be exactly 'a100', got {partitions!r}")
    gres = m3b_ready.m3a_ready.extract_sbatch_gres(text)
    if "gpu:1" in gres:
        report.ok("sbatch uses #SBATCH --gres=gpu:1")
    else:
        report.fail(f"sbatch must include --gres=gpu:1, got {gres!r}")
    for token in (
        "check_phase2m_lora_multiscale_pilot_eval_ready.py",
        "phase2m_render_lora_multiscale_pilot.py",
        "aggregate_phase2m_lora_multiscale_pilot_eval.py",
        "make_phase2m_lora_multiscale_pilot_boards.py",
    ):
        if token in text:
            report.ok(f"sbatch contains {token}")
        else:
            report.fail(f"sbatch must contain {token}")


def check_readiness(args: argparse.Namespace) -> Report:
    report = Report()
    project_root = args.project_root.resolve()
    summary = validate_pilot_summary(args.pilot_summary.resolve(), report, project_root)
    validate_variant_outputs(args.per_variant_outputs.resolve(), summary, report, project_root)
    for path, label in (
        (args.blender_wrapper.resolve(), "Blender rendered-view wrapper"),
        (args.render_script.resolve(), "M3C render script"),
        (args.aggregate_script.resolve(), "M3C aggregate script"),
        (args.board_script.resolve(), "M3C board script"),
    ):
        m3b_ready.m3a_ready.check_nonzero(path, label, report, project_root)
    validate_project_output_dir(args.render_root.resolve(), "render output root", report, project_root)
    validate_project_output_dir(args.eval_root.resolve(), "eval output root", report, project_root)
    validate_sbatch(args.sbatch.resolve(), report, project_root, args.dry_run)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2M.3C rendered-view pilot readiness without running Blender.")
    parser.add_argument("--pilot-summary", type=Path, default=DEFAULT_PILOT_SUMMARY)
    parser.add_argument("--per-variant-outputs", type=Path, default=DEFAULT_PER_VARIANT_OUTPUTS)
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--render-script", type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--aggregate-script", type=Path, default=DEFAULT_AGGREGATE_SCRIPT)
    parser.add_argument("--board-script", type=Path, default=DEFAULT_BOARD_SCRIPT)
    parser.add_argument("--blender-wrapper", type=Path, default=DEFAULT_BLENDER_WRAPPER)
    parser.add_argument("--sbatch", type=Path, default=DEFAULT_SBATCH)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args)
    print("Phase 2M.3C LoRA multi-scale rendered-view pilot readiness")
    for line in report.lines:
        print(line)
    if report.errors:
        print(f"PHASE2M_M3C_LORA_PILOT_EVAL_READY_FAIL errors={len(report.errors)} warnings={len(report.warnings)}")
        return 1
    print("PHASE2M_M3C_LORA_PILOT_EVAL_READY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
