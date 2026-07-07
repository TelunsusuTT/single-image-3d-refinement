#!/usr/bin/env python3
"""Phase 2M M2 adapter-only training readiness checks.

This checker is intentionally stdlib-only. It validates the M0/M1 LoRA smoke
artifacts and static M2 planning paths without importing Hunyuan, torch, or
loading checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = PROJECT_ROOT / "outputs" / "phase2m" / "lora_module_inventory.json"
DEFAULT_SMOKE_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "zero_lora_smoke"
DEFAULT_FULL80_CONFIG = PROJECT_ROOT / "configs" / "ft_datav2_frame_full80_truepbr_500_lr1e6.yaml"
DEFAULT_M2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300"
DEFAULT_M2_SMOKE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300_smoke"
DEFAULT_M2_SINGLE_RANK_SMOKE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300_smoke_single_rank"
DEFAULT_M2_LOGDIR = PROJECT_ROOT / "logs" / "train" / "phase2m" / "phase2m_refdino_r4_lr5e5_300"
DEFAULT_M2_CHECKPOINT_DIR = DEFAULT_M2_OUTPUT_DIR
DEFAULT_M2_SBATCH = PROJECT_ROOT / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_a100.sbatch"
DEFAULT_M2_SMOKE_SBATCH = PROJECT_ROOT / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch"
DEFAULT_M2_SINGLE_RANK_SMOKE_SBATCH = PROJECT_ROOT / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_single_rank_a100.sbatch"
OFFICIAL_WORK_ROOT = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work")
CANONICAL_M1_SUMMARY_NAME = "zero_lora_smoke_summary.json"
LEGACY_M1_SUMMARY_NAME = "smoke_summary.json"
REQUIRED_PARTITION = "a100"
BANNED_SBATCH_TOKENS = ("gpgpuC", "--constraint=a100")
CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth", ".bin", ".safetensors"}


class CheckReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.lines: list[str] = []

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



def find_m1_summary(smoke_dir: Path) -> Path | None:
    canonical = smoke_dir / CANONICAL_M1_SUMMARY_NAME
    legacy = smoke_dir / LEGACY_M1_SUMMARY_NAME
    if canonical.is_file():
        return canonical
    if legacy.is_file():
        return legacy
    return None



def path_text(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_project_local_path(path: Path, label: str, report: CheckReport, project_root: Path, allowed_roots: tuple[Path, ...]) -> None:
    resolved = path.resolve()
    if is_relative_to(resolved, OFFICIAL_WORK_ROOT):
        report.fail(f"{label} must not be under Hunyuan3D2.1_Work: {path_text(resolved, project_root)}")
        return
    if any(is_relative_to(resolved, root) for root in allowed_roots):
        report.ok(f"{label} is project-local: {path_text(resolved, project_root)}")
    else:
        allowed = [path_text(root, project_root) for root in allowed_roots]
        report.fail(f"{label} must be under one of {allowed}: {path_text(resolved, project_root)}")



def checkpoint_like_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix in CHECKPOINT_SUFFIXES else []
    matches: list[Path] = []
    for child in path.rglob("*"):
        if child.is_file() and child.suffix in CHECKPOINT_SUFFIXES:
            matches.append(child)
    return sorted(matches)


def extract_json_paths_from_config(config_path: Path) -> list[Path]:
    text = config_path.read_text(encoding="utf-8")
    paths: list[Path] = []
    for match in re.finditer(r"json_path:\s*([^\s#]+)", text):
        raw = match.group(1).strip().strip("'\"")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        paths.append(path.resolve())
    return paths



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


def validate_sbatch(path: Path, report: CheckReport, dry_run: bool, expected_trainer_mode: str | None = None) -> None:
    if not path.exists():
        if dry_run:
            report.warn(f"M2 sbatch not present yet, partition/gres validation skipped in dry-run: {path_text(path)}")
            return
        report.fail(f"M2 sbatch missing: {path_text(path)}")
        return
    text = path.read_text(encoding="utf-8")
    for token in BANNED_SBATCH_TOKENS:
        if token in text:
            report.fail(f"M2 sbatch contains banned token {token}: {path_text(path)}")
    partitions = extract_sbatch_partitions(text)
    if partitions == [REQUIRED_PARTITION]:
        report.ok(f"{path.name} uses #SBATCH -p a100")
    else:
        report.fail(f"{path.name} partition must be exactly {REQUIRED_PARTITION!r}, got {partitions!r}")
    gres = extract_sbatch_gres(text)
    if "gpu:1" in gres:
        report.ok(f"{path.name} uses #SBATCH --gres=gpu:1")
    else:
        report.fail(f"{path.name} must include --gres=gpu:1, got {gres!r}")
    if expected_trainer_mode is not None:
        expected_flag = f"--trainer-mode {expected_trainer_mode}"
        if expected_flag in text:
            report.ok(f"{path.name} uses {expected_flag}")
        else:
            report.fail(f"{path.name} must include {expected_flag}")



def check_readiness(args: argparse.Namespace) -> CheckReport:
    report = CheckReport()
    inventory_path = args.inventory_json.resolve()
    smoke_dir = args.smoke_dir.resolve()
    adapter_config_path = smoke_dir / "adapter_config.json"
    adapter_state_path = smoke_dir / "ref_dino_adapter_state.pt"
    full80_config_path = args.full80_config.resolve()
    m2_output_dir = args.m2_output_dir.resolve()
    m2_checkpoint_dir = args.m2_checkpoint_dir.resolve()
    m2_sbatch = args.sbatch.resolve()
    smoke_sbatch = args.smoke_sbatch.resolve()
    single_rank_smoke_sbatch = args.single_rank_smoke_sbatch.resolve()
    smoke_output_dir = args.smoke_output_dir.resolve()
    single_rank_smoke_output_dir = args.single_rank_smoke_output_dir.resolve()
    logdir = args.logdir.resolve()
    project_root = args.project_root.resolve()

    if inventory_path.is_file():
        report.ok(f"M0 inventory exists: {path_text(inventory_path)}")
        try:
            inventory = load_json(inventory_path)
        except Exception as exc:
            report.fail(f"M0 inventory is not readable JSON: {exc}")
            inventory = {}
        if inventory.get("selected_target_count") == 128:
            report.ok("M0 selected_target_count == 128")
        else:
            report.fail(f"M0 selected_target_count expected 128, got {inventory.get('selected_target_count')!r}")
    else:
        report.fail(f"M0 inventory missing: {path_text(inventory_path)}")

    summary_path = find_m1_summary(smoke_dir)
    if summary_path is None:
        report.fail(
            f"M1 summary missing; expected {CANONICAL_M1_SUMMARY_NAME} or {LEGACY_M1_SUMMARY_NAME} under {path_text(smoke_dir)}"
        )
        summary = {}
    else:
        preferred = CANONICAL_M1_SUMMARY_NAME if summary_path.name == CANONICAL_M1_SUMMARY_NAME else LEGACY_M1_SUMMARY_NAME
        report.ok(f"M1 summary exists ({preferred}): {path_text(summary_path)}")
        try:
            summary = load_json(summary_path)
        except Exception as exc:
            report.fail(f"M1 summary is not readable JSON: {exc}")
            summary = {}
    if summary.get("status") == "OK":
        report.ok("M1 status == OK")
    else:
        report.fail(f"M1 status expected OK, got {summary.get('status')!r}")
    if summary.get("selected_target_count") == 128:
        report.ok("M1 selected_target_count == 128")
    else:
        report.fail(f"M1 selected_target_count expected 128, got {summary.get('selected_target_count')!r}")
    if summary.get("backend") == "local_linear_fallback":
        report.ok("M1 backend == local_linear_fallback")
    else:
        report.fail(f"M1 backend expected local_linear_fallback, got {summary.get('backend')!r}")
    tree_diff = summary.get("official_tree_diff") if isinstance(summary.get("official_tree_diff"), dict) else {}
    if tree_diff.get("unchanged") is True:
        report.ok("official_tree_diff.unchanged == true")
    else:
        report.fail(f"official_tree_diff.unchanged expected true, got {tree_diff.get('unchanged')!r}")

    if adapter_config_path.is_file():
        report.ok(f"adapter_config exists: {path_text(adapter_config_path)}")
        try:
            adapter_config = load_json(adapter_config_path)
        except Exception as exc:
            report.fail(f"adapter_config is not readable JSON: {exc}")
            adapter_config = {}
    else:
        report.fail(f"adapter_config missing: {path_text(adapter_config_path)}")
        adapter_config = {}
    safety = adapter_config.get("safety") if isinstance(adapter_config.get("safety"), dict) else {}
    expected_safety = {
        "adapter_only": True,
        "merge_into_base": False,
        "save_pretrained_full_model": False,
    }
    for key, expected in expected_safety.items():
        actual = safety.get(key)
        if actual is expected:
            report.ok(f"adapter_config safety.{key} == {str(expected).lower()}")
        else:
            report.fail(f"adapter_config safety.{key} expected {expected!r}, got {actual!r}")

    if adapter_state_path.is_file() and adapter_state_path.stat().st_size > 0:
        report.ok(f"adapter state exists and is nonzero: {path_text(adapter_state_path)}")
    else:
        report.fail(f"adapter state missing or zero-size: {path_text(adapter_state_path)}")

    if full80_config_path.is_file():
        report.ok(f"full80 train config exists: {path_text(full80_config_path)}")
        json_paths = extract_json_paths_from_config(full80_config_path)
        if json_paths:
            for json_path in json_paths:
                if json_path.is_file():
                    report.ok(f"full80 data json exists: {path_text(json_path)}")
                else:
                    report.fail(f"full80 data json missing: {path_text(json_path)}")
        else:
            report.fail(f"full80 train config contains no json_path entries: {path_text(full80_config_path)}")
    else:
        report.fail(f"full80 train config missing: {path_text(full80_config_path)}")

    allowed_output_roots = (project_root / "outputs" / "phase2m",)
    allowed_log_roots = (project_root / "logs" / "train" / "phase2m", project_root / "outputs" / "phase2m")
    if m2_output_dir == smoke_dir:
        report.fail("M2 output directory must not reuse zero_lora_smoke output directory")
    validate_project_local_path(m2_output_dir, "M2 output directory", report, project_root, allowed_output_roots)
    validate_project_local_path(smoke_output_dir, "M2 official_ddp smoke output directory", report, project_root, allowed_output_roots)
    validate_project_local_path(single_rank_smoke_output_dir, "M2 single_rank_data smoke output directory", report, project_root, allowed_output_roots)
    validate_project_local_path(logdir, "M2 model.logdir", report, project_root, allowed_log_roots)
    distinct_outputs = {m2_output_dir, smoke_output_dir, single_rank_smoke_output_dir}
    if len(distinct_outputs) != 3:
        report.fail("M2 full, official_ddp smoke, and single_rank_data smoke output directories must be distinct")
    else:
        report.ok("M2 full, official_ddp smoke, and single_rank_data smoke output directories are distinct")

    output_checkpoint_files = checkpoint_like_files(m2_output_dir)
    if output_checkpoint_files:
        report.fail(f"M2 output directory already contains checkpoint-like files: {[path_text(p) for p in output_checkpoint_files[:10]]}")
    else:
        report.ok("M2 output directory has no checkpoint-like files")

    checkpoint_files = checkpoint_like_files(m2_checkpoint_dir)
    if checkpoint_files:
        report.fail(f"M2 checkpoint directory already contains checkpoint-like files: {[path_text(p) for p in checkpoint_files[:10]]}")
    else:
        report.ok("M2 checkpoint directory has no checkpoint-like files")

    validate_sbatch(m2_sbatch, report, args.dry_run, expected_trainer_mode="official_ddp")
    validate_sbatch(smoke_sbatch, report, args.dry_run, expected_trainer_mode="official_ddp")
    validate_sbatch(single_rank_smoke_sbatch, report, args.dry_run, expected_trainer_mode="single_rank_data")
    return report



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2M M2 LoRA training readiness without running training.")
    parser.add_argument("--inventory-json", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--full80-config", type=Path, default=DEFAULT_FULL80_CONFIG)
    parser.add_argument("--m2-output-dir", type=Path, default=DEFAULT_M2_OUTPUT_DIR)
    parser.add_argument("--m2-checkpoint-dir", type=Path, default=DEFAULT_M2_CHECKPOINT_DIR)
    parser.add_argument("--smoke-output-dir", type=Path, default=DEFAULT_M2_SMOKE_OUTPUT_DIR)
    parser.add_argument("--single-rank-smoke-output-dir", type=Path, default=DEFAULT_M2_SINGLE_RANK_SMOKE_OUTPUT_DIR)
    parser.add_argument("--logdir", type=Path, default=DEFAULT_M2_LOGDIR)
    parser.add_argument("--sbatch", type=Path, default=DEFAULT_M2_SBATCH)
    parser.add_argument("--smoke-sbatch", type=Path, default=DEFAULT_M2_SMOKE_SBATCH)
    parser.add_argument("--single-rank-smoke-sbatch", type=Path, default=DEFAULT_M2_SINGLE_RANK_SMOKE_SBATCH)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Allow not-yet-created M2 sbatch while checking existing artifacts.")
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args)
    print("Phase 2M M2 LoRA training readiness")
    for line in report.lines:
        print(line)
    if report.errors:
        print(f"PHASE2M_M2_READINESS_FAIL errors={len(report.errors)} warnings={len(report.warnings)}")
        return 1
    print("PHASE2M_M2_READINESS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
