from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import check_phase2m_lora_training_ready as ready  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def sbatch_text_for_mode(mode: str, header: str = "#SBATCH -p a100\n#SBATCH --gres=gpu:1\n") -> str:
    return f"{header}\npython train.py --trainer-mode {mode}\n"


def make_args(
    root: Path,
    sbatch_text: str | None = None,
    smoke_sbatch_text: str | None = None,
    single_rank_smoke_sbatch_text: str | None = None,
) -> argparse.Namespace:
    smoke = root / "outputs" / "phase2m" / "zero_lora_smoke"
    write_json(root / "outputs" / "phase2m" / "lora_module_inventory.json", {"selected_target_count": 128})
    write_json(
        smoke / "zero_lora_smoke_summary.json",
        {
            "status": "OK",
            "selected_target_count": 128,
            "backend": "local_linear_fallback",
            "official_tree_diff": {"unchanged": True},
        },
    )
    write_json(
        smoke / "adapter_config.json",
        {"safety": {"adapter_only": True, "merge_into_base": False, "save_pretrained_full_model": False}},
    )
    (smoke / "ref_dino_adapter_state.pt").write_bytes(b"adapter")
    data_json = root / "data" / "hy3dpaint_train_examples" / "datav2_frame_panels_full101" / "examples_train_abs.json"
    write_json(data_json, [])
    config = root / "configs" / "ft_datav2_frame_full80_truepbr_500_lr1e6.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"model: fake\ndata:\n  params:\n    train:\n    - params:\n        json_path: {data_json}\n", encoding="utf-8")
    sbatch = root / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_a100.sbatch"
    smoke_sbatch = root / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch"
    single_rank_smoke_sbatch = root / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_single_rank_a100.sbatch"
    sbatch.parent.mkdir(parents=True, exist_ok=True)
    sbatch.write_text(sbatch_text or sbatch_text_for_mode("official_ddp"), encoding="utf-8")
    smoke_sbatch.write_text(smoke_sbatch_text or sbatch_text_for_mode("official_ddp"), encoding="utf-8")
    single_rank_smoke_sbatch.write_text(single_rank_smoke_sbatch_text or sbatch_text_for_mode("single_rank_data"), encoding="utf-8")
    return argparse.Namespace(
        inventory_json=root / "outputs" / "phase2m" / "lora_module_inventory.json",
        smoke_dir=smoke,
        full80_config=config,
        m2_output_dir=root / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300",
        m2_checkpoint_dir=root / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300",
        smoke_output_dir=root / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300_smoke",
        single_rank_smoke_output_dir=root / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300_smoke_single_rank",
        logdir=root / "logs" / "train" / "phase2m" / "phase2m_refdino_r4_lr5e5_300",
        sbatch=sbatch,
        smoke_sbatch=smoke_sbatch,
        single_rank_smoke_sbatch=single_rank_smoke_sbatch,
        project_root=root,
        dry_run=False,
    )


def test_phase2m_readiness_passes_with_canonical_summary(tmp_path: Path) -> None:
    args = make_args(tmp_path)

    report = ready.check_readiness(args)

    assert report.errors == []


def test_phase2m_readiness_accepts_legacy_summary_name(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    canonical = args.smoke_dir / "zero_lora_smoke_summary.json"
    legacy = args.smoke_dir / "smoke_summary.json"
    legacy.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    canonical.unlink()

    report = ready.check_readiness(args)

    assert report.errors == []


def test_phase2m_readiness_rejects_old_partition_names(tmp_path: Path) -> None:
    bad = sbatch_text_for_mode("official_ddp", header="#SBATCH --partition=gpgpuC\n#SBATCH --constraint=a100\n")
    args = make_args(tmp_path, sbatch_text=bad, smoke_sbatch_text=bad, single_rank_smoke_sbatch_text=bad)

    report = ready.check_readiness(args)

    assert any("gpgpuC" in error for error in report.errors)
    assert any("--constraint=a100" in error for error in report.errors)
    assert any("partition must be exactly" in error for error in report.errors)


def test_phase2m_readiness_rejects_missing_gpu_gres(tmp_path: Path) -> None:
    missing_gres = sbatch_text_for_mode("official_ddp", header="#SBATCH -p a100\n")
    args = make_args(tmp_path, sbatch_text=missing_gres, smoke_sbatch_text=missing_gres, single_rank_smoke_sbatch_text=missing_gres)

    report = ready.check_readiness(args)

    assert any("--gres=gpu:1" in error for error in report.errors)


def test_phase2m_readiness_dry_run_allows_missing_m2_sbatch(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    args.sbatch.unlink()
    args.smoke_sbatch.unlink()
    args.single_rank_smoke_sbatch.unlink()
    args.dry_run = True

    report = ready.check_readiness(args)

    assert report.errors == []
    assert any("skipped in dry-run" in warning for warning in report.warnings)


def test_phase2m_readiness_rejects_missing_trainer_mode(tmp_path: Path) -> None:
    args = make_args(tmp_path, smoke_sbatch_text="#SBATCH -p a100\n#SBATCH --gres=gpu:1\n")

    report = ready.check_readiness(args)

    assert any("--trainer-mode official_ddp" in error for error in report.errors)


def test_phase2m_readiness_rejects_single_rank_mode_mismatch(tmp_path: Path) -> None:
    args = make_args(tmp_path, single_rank_smoke_sbatch_text=sbatch_text_for_mode("official_ddp"))

    report = ready.check_readiness(args)

    assert any("--trainer-mode single_rank_data" in error for error in report.errors)
