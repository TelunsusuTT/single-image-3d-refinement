#!/usr/bin/env python3
"""Static readiness checks for Phase 2M.3A LoRA inference smoke."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300"
DEFAULT_ADAPTER_PATH = DEFAULT_ADAPTER_DIR / "adapter_final.pt"
DEFAULT_ADAPTER_CONFIG = DEFAULT_ADAPTER_DIR / "adapter_config.json"
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "datav2_frame_full80_eval.json"
DEFAULT_EVAL_CASES = PROJECT_ROOT / "outputs" / "phase2l" / "datav2_frame_panels" / "full80_eval_truepbr500" / "eval_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_infer_smoke_scale075"
DEFAULT_SCRIPT = PROJECT_ROOT / "scripts" / "phase2m_lora_infer_smoke.py"
DEFAULT_SBATCH = PROJECT_ROOT / "env" / "run_phase2m_lora_infer_smoke_scale075_a100.sbatch"
OFFICIAL_WORK_ROOT = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work")
REQUIRED_PARTITION = "a100"
BANNED_SBATCH_TOKENS = ("gpgpuC", "--constraint=a100")


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        self.lines.append(f"OK: {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.lines.append(f"WARN: {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.lines.append(f"FAIL: {message}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def path_text(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def check_nonzero(path: Path, label: str, report: Report, project_root: Path, min_bytes: int = 1) -> None:
    if not path.is_file():
        report.fail(f"{label} missing: {path_text(path, project_root)}")
        return
    size = path.stat().st_size
    if size < min_bytes:
        report.fail(f"{label} too small: {path_text(path, project_root)} bytes={size} min={min_bytes}")
    else:
        report.ok(f"{label} exists: {path_text(path, project_root)} bytes={size}")


def validate_adapter_config(path: Path, report: Report, project_root: Path) -> None:
    if not path.is_file():
        report.fail(f"adapter_config missing: {path_text(path, project_root)}")
        return
    try:
        config = load_json(path)
    except Exception as exc:
        report.fail(f"adapter_config is not readable JSON: {exc}")
        return
    if config.get("backend") == "local_linear_fallback":
        report.ok("adapter_config backend == local_linear_fallback")
    else:
        report.fail(f"adapter_config backend expected local_linear_fallback, got {config.get('backend')!r}")
    targets = config.get("target_names")
    if isinstance(targets, list) and len(targets) == 128 and config.get("target_count") == 128:
        report.ok("adapter_config target_count == 128")
    else:
        report.fail("adapter_config must contain target_count == 128 and 128 target_names")
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    if safety.get("merge_into_base") is False:
        report.ok("adapter_config safety.merge_into_base == false")
    else:
        report.fail("adapter_config safety.merge_into_base must be false")
    if safety.get("save_pretrained_full_model") is False:
        report.ok("adapter_config safety.save_pretrained_full_model == false")
    else:
        report.fail("adapter_config safety.save_pretrained_full_model must be false")


def selected_case(cases_json: Path) -> dict[str, Any] | None:
    data = load_json(cases_json)
    cases = data.get("cases")
    if not isinstance(cases, list):
        return None
    for case in cases:
        if isinstance(case, dict) and str(case.get("selected_input_view", "")) == "005":
            return case
    return cases[0] if cases and isinstance(cases[0], dict) else None


def validate_eval_inputs(cases_json: Path, report: Report, project_root: Path) -> None:
    if not cases_json.is_file():
        report.fail(f"full80 eval cases missing: {path_text(cases_json, project_root)}")
        return
    try:
        case = selected_case(cases_json)
    except Exception as exc:
        report.fail(f"full80 eval cases are not readable: {exc}")
        return
    if case is None:
        report.fail("full80 eval cases contain no usable case")
        return
    item_id = str(case.get("item_id", ""))
    selected_view = str(case.get("selected_input_view", ""))
    if selected_view == "005":
        report.ok(f"selected smoke case uses selected_input_view=005: {item_id}")
    else:
        report.fail(f"selected smoke case must use selected_input_view=005, got {selected_view!r}")
    case_dir = Path(str(case.get("case_dir", ""))).expanduser()
    mesh = Path(str(case.get("case_input_mesh") or case_dir / "input" / "mesh.glb")).expanduser()
    image = Path(str(case.get("case_input_image") or case_dir / "input" / "image.png")).expanduser()
    check_nonzero(mesh.resolve(), "case input mesh", report, project_root)
    check_nonzero(image.resolve(), "case input image", report, project_root)


def extract_sbatch_partitions(text: str) -> list[str]:
    partitions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        match = re.search(r"(?:^|\s)-p\s+([^\s#]+)", stripped)
        if match:
            partitions.extend(part.strip() for part in match.group(1).split(",") if part.strip())
        match = re.search(r"(?:^|\s)--partition(?:=|\s+)([^\s#]+)", stripped)
        if match:
            partitions.extend(part.strip() for part in match.group(1).split(",") if part.strip())
    return partitions


def extract_sbatch_gres(text: str) -> list[str]:
    gres: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        match = re.search(r"(?:^|\s)--gres(?:=|\s+)([^\s#]+)", stripped)
        if match:
            gres.append(match.group(1).strip())
    return gres


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
    partitions = extract_sbatch_partitions(text)
    if partitions == [REQUIRED_PARTITION]:
        report.ok("sbatch uses #SBATCH -p a100")
    else:
        report.fail(f"sbatch partition must be exactly {REQUIRED_PARTITION!r}, got {partitions!r}")
    gres = extract_sbatch_gres(text)
    if "gpu:1" in gres:
        report.ok("sbatch uses #SBATCH --gres=gpu:1")
    else:
        report.fail(f"sbatch must include --gres=gpu:1, got {gres!r}")
    required = [
        "phase2m_lora_infer_smoke.py",
        "--lora-scale 0.75",
        "--adapter-path",
        "check_phase2m_lora_infer_smoke_ready.py",
    ]
    for token in required:
        if token in text:
            report.ok(f"sbatch contains {token}")
        else:
            report.fail(f"sbatch must contain {token}")


def validate_output_dir(path: Path, report: Report, project_root: Path) -> None:
    resolved = path.resolve()
    if is_relative_to(resolved, OFFICIAL_WORK_ROOT):
        report.fail(f"output dir must not be under Hunyuan3D2.1_Work: {resolved}")
        return
    allowed = project_root.resolve() / "outputs" / "phase2m"
    if is_relative_to(resolved, allowed):
        report.ok(f"output dir is under outputs/phase2m: {path_text(resolved, project_root)}")
    else:
        report.fail(f"output dir must be under outputs/phase2m: {resolved}")
    if path.parent.exists():
        report.ok(f"output dir parent exists: {path_text(path.parent, project_root)}")
    else:
        report.fail(f"output dir parent missing: {path_text(path.parent, project_root)}")


def check_readiness(args: argparse.Namespace) -> Report:
    report = Report()
    project_root = args.project_root.resolve()
    min_adapter_bytes = int(float(args.min_adapter_mb) * 1024 * 1024)
    adapter_path = args.adapter_path.resolve()
    adapter_config = args.adapter_config.resolve()
    eval_config = args.eval_config.resolve()
    eval_cases = args.eval_cases_json.resolve()
    output_dir = args.output_dir.resolve()
    script = args.script.resolve()
    sbatch = args.sbatch.resolve()

    check_nonzero(adapter_path, "adapter_final.pt", report, project_root, min_bytes=min_adapter_bytes)
    validate_adapter_config(adapter_config, report, project_root)
    check_nonzero(eval_config, "full80 eval config", report, project_root)
    validate_eval_inputs(eval_cases, report, project_root)
    check_nonzero(script, "LoRA inference smoke script", report, project_root)
    validate_output_dir(output_dir, report, project_root)
    validate_sbatch(sbatch, report, project_root, args.dry_run)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2M.3A LoRA inference smoke readiness without running inference.")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--eval-cases-json", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--sbatch", type=Path, default=DEFAULT_SBATCH)
    parser.add_argument("--min-adapter-mb", type=float, default=1.0)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args)
    print("Phase 2M.3A LoRA inference smoke readiness")
    for line in report.lines:
        print(line)
    if report.errors:
        print(f"PHASE2M_M3A_LORA_INFER_SMOKE_READY_FAIL errors={len(report.errors)} warnings={len(report.warnings)}")
        return 1
    print("PHASE2M_M3A_LORA_INFER_SMOKE_READY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
