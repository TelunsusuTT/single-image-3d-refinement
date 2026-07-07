from __future__ import annotations

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import phase2m_train_lora_refdino as train_lora  # noqa: E402


class DummyModel:
    pass


def test_official_ddp_mode_uses_official_strategy() -> None:
    source = inspect.getsource(train_lora.main)

    assert "from pytorch_lightning.strategies import DDPStrategy" in source
    assert "DDPStrategy(find_unused_parameters=False)" in source
    assert "inference_mode" in source


def test_single_rank_data_mode_avoids_implicit_distributed_world_size() -> None:
    source = inspect.getsource(train_lora.SingleRankDataModuleFromConfig)

    assert "get_world_size" not in source
    assert "get_rank" not in source
    assert "DistributedSampler" not in source
    assert "sampler=None" in source


def test_configure_project_local_training_paths_sets_safe_logdir(tmp_path: Path) -> None:
    model = DummyModel()
    log_root = tmp_path / "logs" / "train" / "phase2m"
    old_project_root = train_lora.PROJECT_ROOT
    try:
        train_lora.PROJECT_ROOT = tmp_path
        paths = train_lora.configure_project_local_training_paths(model, log_root, "run_a")
    finally:
        train_lora.PROJECT_ROOT = old_project_root

    assert model.logdir == str(log_root / "run_a")
    assert model.ckptdir == str(log_root / "run_a" / "checkpoints")
    assert model.cfgdir == str(log_root / "run_a" / "configs")
    assert model.codedir == str(log_root / "run_a" / "code")
    assert Path(paths["images_val"]).is_dir()
    assert not str(Path(model.logdir).resolve()).startswith("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work")


def test_project_local_training_paths_reject_official_tree() -> None:
    model = DummyModel()
    official_logdir = train_lora.OFFICIAL_WORK_ROOT / "logs" / "bad"

    try:
        train_lora.configure_project_local_training_paths(model, official_logdir, "run_a")
    except RuntimeError as exc:
        assert "Hunyuan3D2.1_Work" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("official-tree logdir was accepted")


def test_smoke_overrides_use_distinct_output_dir_and_one_step() -> None:
    args = train_lora.apply_smoke_overrides(train_lora.parse_args(["--smoke"]))

    assert args.max_train_steps == 1
    assert args.save_every == 1
    assert args.out_dir == train_lora.DEFAULT_SMOKE_OUT_DIR
    assert args.out_dir != train_lora.DEFAULT_OUT_DIR
    assert args.run_name.endswith("_smoke")


def test_single_rank_smoke_uses_distinct_output_dir() -> None:
    args = train_lora.apply_smoke_overrides(train_lora.parse_args(["--smoke", "--trainer-mode", "single_rank_data"]))

    assert args.max_train_steps == 1
    assert args.save_every == 1
    assert args.out_dir == train_lora.DEFAULT_SINGLE_RANK_SMOKE_OUT_DIR
    assert args.out_dir != train_lora.DEFAULT_SMOKE_OUT_DIR
    assert args.out_dir != train_lora.DEFAULT_OUT_DIR
    assert args.run_name.endswith("_smoke_single_rank")


def test_adapter_checkpoint_names_for_full_and_smoke() -> None:
    full = [path.name for path in train_lora.expected_adapter_paths(Path("out"), 300, 100, save_final=True)]
    smoke = [path.name for path in train_lora.expected_adapter_paths(Path("out_smoke"), 1, 1, save_final=False)]

    assert full == [
        "adapter_step_000100.pt",
        "adapter_step_000200.pt",
        "adapter_step_000300.pt",
        "adapter_final.pt",
        "adapter_config.json",
        "training_summary.json",
    ]
    assert smoke == ["adapter_step_000001.pt", "adapter_config.json", "training_summary.json"]


def test_smoke_sbatch_uses_official_ddp_mode() -> None:
    text = (PROJECT_ROOT / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_a100.sbatch").read_text(encoding="utf-8")

    assert "--trainer-mode official_ddp" in text


def test_fallback_smoke_sbatch_uses_single_rank_data_mode() -> None:
    text = (PROJECT_ROOT / "env" / "run_phase2m_train_lora_refdino_r4_lr5e5_300_smoke_single_rank_a100.sbatch").read_text(encoding="utf-8")

    assert "--trainer-mode single_rank_data" in text
    assert "lora_train_refdino_r4_lr5e5_300_smoke_single_rank" in text
    assert "/vol/bitbucket/ct1022/Hunyuan3D2.1_Work" not in text
