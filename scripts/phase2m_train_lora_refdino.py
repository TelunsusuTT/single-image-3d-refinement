#!/usr/bin/env python3
"""Phase 2M ref_dino LoRA adapter-only training entrypoint.

This script is designed to run only inside an A100 Slurm job. It imports Hunyuan,
torch, and Lightning inside runtime functions, uses the existing full80 training
config for model/data construction, injects project-local LoRA adapters into the
exact M0 ref_dino target list, trains only LoRA A/B parameters, and saves
adapter-only state files. It never calls save_pretrained and never saves a full
Hunyuan model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_HY21 = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1")
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "ft_datav2_frame_full80_truepbr_500_lr1e6.yaml"
DEFAULT_INVENTORY = PROJECT_ROOT / "outputs" / "phase2m" / "lora_module_inventory.json"
DEFAULT_M1_SMOKE_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "zero_lora_smoke"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300"
DEFAULT_SMOKE_OUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300_smoke"
DEFAULT_SINGLE_RANK_SMOKE_OUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300_smoke_single_rank"
DEFAULT_TRAIN_LOG_DIR = PROJECT_ROOT / "logs" / "train" / "phase2m"
OFFICIAL_WORK_ROOT = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work")
EXPECTED_TARGET_COUNT = 128
PRESET_NAME = "phase2m_refdino_r4_lr5e5_300"
TRAINER_MODES = ("official_ddp", "single_rank_data")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.lora.targeting import validate_target_names  # noqa: E402


def resolve_official_paths() -> tuple[Path, Path]:
    hy21 = Path(os.environ.get("HY21") or DEFAULT_HY21).expanduser().resolve()
    hypaint = Path(os.environ.get("HYPAINT") or hy21 / "hy3dpaint").expanduser().resolve()
    if not hy21.is_dir():
        raise FileNotFoundError(f"HY21 path missing: {hy21}")
    if not hypaint.is_dir():
        raise FileNotFoundError(f"HYPAINT path missing: {hypaint}")
    return hy21, hypaint


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if text not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([text, *parts])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_inventory_targets(path: Path) -> list[str]:
    data = load_json(path)
    count = data.get("selected_target_count")
    targets = data.get("selected_target_names")
    if count != EXPECTED_TARGET_COUNT:
        raise RuntimeError(f"M0 inventory selected_target_count expected {EXPECTED_TARGET_COUNT}, got {count!r}")
    if not isinstance(targets, list) or len(targets) != EXPECTED_TARGET_COUNT:
        raise RuntimeError(f"M0 inventory target list expected {EXPECTED_TARGET_COUNT} names")
    target_names = [str(target) for target in targets]
    validate_target_names(target_names)
    return target_names


def load_m1_summary(smoke_dir: Path) -> dict[str, Any]:
    summary_path = smoke_dir / "zero_lora_smoke_summary.json"
    if not summary_path.is_file():
        summary_path = smoke_dir / "smoke_summary.json"
    data = load_json(summary_path)
    if data.get("status") != "OK":
        raise RuntimeError(f"M1 summary status is not OK: {summary_path}")
    if data.get("backend") != "local_linear_fallback":
        raise RuntimeError(f"M2 requires local_linear_fallback backend, got {data.get('backend')!r}")
    tree_diff = data.get("official_tree_diff") if isinstance(data.get("official_tree_diff"), dict) else {}
    if tree_diff.get("unchanged") is not True:
        raise RuntimeError("M1 official_tree_diff.unchanged is not true")
    return data


def save_steps(max_steps: int, save_every: int) -> list[int]:
    steps = [step for step in range(save_every, max_steps + 1, save_every)]
    if max_steps not in steps:
        steps.append(max_steps)
    return sorted(set(steps))


def expected_adapter_paths(out_dir: Path, max_steps: int, save_every: int, save_final: bool = True) -> list[Path]:
    paths = [out_dir / f"adapter_step_{step:06d}.pt" for step in save_steps(max_steps, save_every)]
    if save_final:
        paths.append(out_dir / "adapter_final.pt")
    paths.extend([out_dir / "adapter_config.json", out_dir / "training_summary.json"])
    return paths


def ensure_output_safe(out_dir: Path, max_steps: int, save_every: int, save_final: bool = True) -> None:
    if not str(out_dir.resolve()).startswith(str((PROJECT_ROOT / "outputs" / "phase2m").resolve())):
        raise RuntimeError(f"M2 output directory must be under outputs/phase2m: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in expected_adapter_paths(out_dir, max_steps, save_every, save_final=save_final) if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing M2 adapter outputs: {existing}")
    forbidden_suffixes = {".ckpt", ".bin", ".safetensors"}
    existing_forbidden = [path for path in out_dir.rglob("*") if path.is_file() and path.suffix in forbidden_suffixes]
    if existing_forbidden:
        raise RuntimeError(f"M2 output directory contains checkpoint-like files: {existing_forbidden[:10]}")


def import_from_string(target: str) -> Any:
    module_name, attr_name = target.rsplit(".", 1)
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def to_plain(value: Any, omega_conf: Any) -> Any:
    try:
        return omega_conf.to_container(value, resolve=True)
    except Exception:
        return value


def instantiate_from_config(config: Any, omega_conf: Any) -> Any:
    target = config.get("target")
    if not target:
        raise KeyError(f"Config block missing target: {config}")
    params = config.get("params", {})
    params = to_plain(params, omega_conf)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError(f"Config params for {target} must resolve to dict, got {type(params).__name__}")
    cls = import_from_string(str(target))
    return cls(**params)


class SingleRankDataModuleFromConfig:
    """Project-local fallback DataModule that avoids implicit distributed sampling.

    The official DataModule constructs a distributed sampler without
    explicit rank/world-size arguments. That is correct when Lightning has
    initialized DDP, but it fails before step 1 when running a single-rank
    trainer without a process group. This fallback is opt-in and reuses the same
    dataset configs while creating ordinary single-rank DataLoaders.
    """

    def __init__(self, data_config: Any, omega_conf: Any) -> None:
        params = data_config.get("params", {})
        params = to_plain(params, omega_conf)
        if not isinstance(params, dict):
            raise TypeError(f"Data config params must resolve to dict, got {type(params).__name__}")
        self.batch_size = int(params.get("batch_size", 1))
        self.num_workers = int(params.get("num_workers", 0))
        self.dataset_configs: dict[str, list[Any]] = {}
        for split in ("train", "validation", "test"):
            split_configs = params.get(split)
            if split_configs is None:
                continue
            if not isinstance(split_configs, list):
                split_configs = [split_configs]
            self.dataset_configs[split] = split_configs
        self.datasets: dict[str, list[Any]] = {}

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            raise NotImplementedError(f"Single-rank fallback only supports fit stage, got {stage!r}")
        from omegaconf import OmegaConf  # type: ignore

        self.datasets = {}
        for split, configs in self.dataset_configs.items():
            self.datasets[split] = [instantiate_from_config(config, OmegaConf) for config in configs]

    def _concat(self, split: str) -> Any:
        if split not in self.datasets:
            raise KeyError(f"Dataset split is not configured: {split}")
        from torch.utils.data import ConcatDataset  # type: ignore

        return ConcatDataset(self.datasets[split])

    def _loader_kwargs(self) -> dict[str, Any]:
        kwargs = {"num_workers": self.num_workers, "pin_memory": True}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_dataloader(self) -> Any:
        from torch.utils.data import DataLoader  # type: ignore

        return DataLoader(
            self._concat("train"),
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> Any:
        from torch.utils.data import DataLoader  # type: ignore

        return DataLoader(
            self._concat("validation"),
            batch_size=4,
            shuffle=False,
            sampler=None,
            **self._loader_kwargs(),
        )


def make_single_rank_data_module(data_config: Any, omega_conf: Any) -> Any:
    import pytorch_lightning as pl  # type: ignore

    class SingleRankLightningDataModule(SingleRankDataModuleFromConfig, pl.LightningDataModule):  # type: ignore[misc]
        def __init__(self, config: Any, omega_conf_obj: Any) -> None:
            pl.LightningDataModule.__init__(self)
            SingleRankDataModuleFromConfig.__init__(self, config, omega_conf_obj)

    return SingleRankLightningDataModule(data_config, omega_conf)


def setup_model_and_data(train_config: Path, trainer_mode: str = "official_ddp") -> tuple[Any, Any, dict[str, Any]]:
    hy21, hypaint = resolve_official_paths()
    prepend_pythonpath(SRC_ROOT)
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    from omegaconf import OmegaConf  # type: ignore

    config = OmegaConf.load(str(train_config))
    model = instantiate_from_config(config.model, OmegaConf)
    if trainer_mode == "official_ddp":
        data_module = instantiate_from_config(config.data, OmegaConf)
        data_target = str(config.data.get("target"))
    elif trainer_mode == "single_rank_data":
        data_module = make_single_rank_data_module(config.data, OmegaConf)
        data_target = "project_local.SingleRankDataModuleFromConfig"
    else:
        raise ValueError(f"Unsupported trainer mode: {trainer_mode}")
    metadata = {
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        "train_config": str(train_config),
        "model_target": str(config.model.get("target")),
        "data_target": data_target,
        "trainer_mode": trainer_mode,
    }
    return model, data_module, metadata


def find_lora_root(model: Any, target_names: list[str]) -> tuple[str, Any]:
    root_names = {name for name, _module in model.named_modules()}
    if all(target in root_names for target in target_names):
        return "", model
    for prefix, module in model.named_modules():
        names = {name for name, _child in module.named_modules()}
        if all(target in names for target in target_names):
            return prefix, module
    sample = target_names[:5]
    raise RuntimeError(f"Could not find a model submodule containing M0 target names; sample={sample}")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_project_local_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if is_relative_to(resolved, OFFICIAL_WORK_ROOT):
        raise RuntimeError(f"{label} must not be under Hunyuan3D2.1_Work: {resolved}")
    if not (is_relative_to(resolved, PROJECT_ROOT / "logs") or is_relative_to(resolved, PROJECT_ROOT / "outputs")):
        raise RuntimeError(f"{label} must be under project logs/ or outputs/: {resolved}")
    return resolved


def configure_project_local_training_paths(model: Any, trainer_log_dir: Path, run_name: str) -> dict[str, str]:
    logdir = ensure_project_local_path(trainer_log_dir / run_name, "model.logdir")
    ckptdir = ensure_project_local_path(logdir / "checkpoints", "model.ckptdir")
    cfgdir = ensure_project_local_path(logdir / "configs", "model.cfgdir")
    codedir = ensure_project_local_path(logdir / "code", "model.codedir")
    images_val = ensure_project_local_path(logdir / "images_val", "images_val")
    for path in (logdir, ckptdir, cfgdir, codedir, images_val):
        path.mkdir(parents=True, exist_ok=True)
    model.logdir = str(logdir)
    model.ckptdir = str(ckptdir)
    model.cfgdir = str(cfgdir)
    model.codedir = str(codedir)
    return {
        "logdir": str(logdir),
        "ckptdir": str(ckptdir),
        "cfgdir": str(cfgdir),
        "codedir": str(codedir),
        "images_val": str(images_val),
    }


def trainable_lora_parameters(model: Any) -> list[tuple[str, Any]]:
    items = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    offenders = [name for name, _parameter in items if "lora_" not in name]
    if offenders:
        raise RuntimeError(f"Non-LoRA trainable parameters found: {offenders[:20]}")
    if not items:
        raise RuntimeError("No trainable LoRA parameters found")
    return items


def save_adapter_checkpoint(lora_root: Any, path: Path, config: dict[str, Any] | None = None) -> int:
    from hy3dft.lora.io import save_adapter_state_dict
    from hy3dft.lora.local_linear import adapter_state_dict

    state = adapter_state_dict(lora_root)
    save_adapter_state_dict(state, path, config)
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def tensor_loss_value(output: Any) -> float | None:
    loss = None
    if hasattr(output, "detach"):
        loss = output
    elif isinstance(output, dict):
        for key in ("loss", "train/loss", "total_loss"):
            value = output.get(key)
            if hasattr(value, "detach"):
                loss = value
                break
    if loss is None:
        return None
    try:
        return float(loss.detach().cpu().item())
    except Exception:
        return None


def make_adapter_checkpoint_callback(callback_base: Any, lora_root: Any, adapter_config: dict[str, Any], out_dir: Path, max_steps: int, save_every: int, save_final: bool = True) -> Any:
    class AdapterCheckpointCallback(callback_base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.start_time = time.time()
            self.losses: list[float] = []
            self.saved_paths: list[str] = []
            self.saved_steps: set[int] = set()
            self.final_bytes = 0

        def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
            loss_value = tensor_loss_value(outputs)
            if loss_value is not None:
                self.losses.append(loss_value)
            step = int(getattr(trainer, "global_step", 0))
            if step > 0 and (step % save_every == 0 or step == max_steps) and step not in self.saved_steps:
                path = out_dir / f"adapter_step_{step:06d}.pt"
                bytes_saved = save_adapter_checkpoint(lora_root, path, adapter_config)
                self.saved_steps.add(step)
                self.saved_paths.append(str(path))
                print(f"saved_adapter_checkpoint={path} bytes_estimate={bytes_saved}")

        def on_train_end(self, trainer: Any, pl_module: Any) -> None:
            from hy3dft.lora.peft_lora import parameter_summary

            final_path = out_dir / "adapter_final.pt"
            if save_final:
                self.final_bytes = save_adapter_checkpoint(lora_root, final_path, adapter_config)
                self.saved_paths.append(str(final_path))
            summary = parameter_summary(pl_module)
            optimizer_param_count = adapter_config.get("optimizer_parameter_count", 0)
            losses = self.losses
            training_summary = {
                "phase": "2M",
                "preset_name": PRESET_NAME,
                "status": "OK",
                "backend": adapter_config["backend"],
                "rank": adapter_config["rank"],
                "alpha": adapter_config["alpha"],
                "dropout": adapter_config["dropout"],
                "learning_rate": adapter_config["learning_rate"],
                "max_train_steps": max_steps,
                "save_every": save_every,
                "trainer_global_step": int(getattr(trainer, "global_step", 0)),
                "output_dir": str(out_dir),
                "adapter_config_path": str(out_dir / "adapter_config.json"),
                "adapter_final_path": str(final_path) if save_final else "",
                "saved_adapter_paths": self.saved_paths,
                "final_adapter_bytes_estimate": self.final_bytes,
                "target_count": adapter_config["target_count"],
                "lora_root_path": adapter_config["lora_root_path"],
                "trainable_parameter_count": summary["trainable_parameter_count"],
                "total_parameter_count": summary["total_parameter_count"],
                "optimizer_parameter_count": optimizer_param_count,
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_min": min(losses) if losses else None,
                "loss_max": max(losses) if losses else None,
                "elapsed_seconds": time.time() - self.start_time,
                "safety": adapter_config["safety"],
            }
            summary_path = out_dir / "training_summary.json"
            if summary_path.exists():
                raise FileExistsError(f"Refusing to overwrite training summary: {summary_path}")
            summary_path.write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")
            print(f"adapter_final={final_path}")
            print(f"training_summary={summary_path}")
            print("PHASE2M_M2_LORA_ADAPTER_TRAINING_OK")

    return AdapterCheckpointCallback()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2M ref_dino LoRA adapters only.")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--inventory-json", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--m1-smoke-dir", type=Path, default=DEFAULT_M1_SMOKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trainer-log-dir", type=Path, default=DEFAULT_TRAIN_LOG_DIR)
    parser.add_argument("--run-name", default=PRESET_NAME)
    parser.add_argument("--smoke", action="store_true", help="Run one-step smoke into the smoke output directory.")
    parser.add_argument("--trainer-mode", choices=TRAINER_MODES, default="official_ddp")
    parser.add_argument("--backend", default="local_linear_fallback", choices=("local_linear_fallback",))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-steps", "--max-train-steps", dest="max_train_steps", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    return parser.parse_args(argv)


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.max_train_steps = 1
        args.save_every = 1
        if args.trainer_mode == "single_rank_data":
            args.out_dir = DEFAULT_SINGLE_RANK_SMOKE_OUT_DIR
        else:
            args.out_dir = DEFAULT_SMOKE_OUT_DIR
        if args.run_name == PRESET_NAME:
            suffix = "smoke_single_rank" if args.trainer_mode == "single_rank_data" else "smoke"
            args.run_name = f"{PRESET_NAME}_{suffix}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = apply_smoke_overrides(parse_args(argv))
    if args.backend != "local_linear_fallback":
        raise RuntimeError("Phase 2M M2 must use local_linear_fallback; PEFT is unavailable")
    if args.max_train_steps <= 0 or args.save_every <= 0:
        raise ValueError("max-train-steps and save-every must be positive")

    args.train_config = args.train_config.expanduser().resolve()
    args.inventory_json = args.inventory_json.expanduser().resolve()
    args.m1_smoke_dir = args.m1_smoke_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.trainer_log_dir = args.trainer_log_dir.expanduser().resolve()

    target_names = load_inventory_targets(args.inventory_json)
    m1_summary = load_m1_summary(args.m1_smoke_dir)
    save_final = not args.smoke
    ensure_output_safe(args.out_dir, args.max_train_steps, args.save_every, save_final=save_final)
    args.trainer_log_dir.mkdir(parents=True, exist_ok=True)

    print("PHASE2M_M2_LORA_TRAINING_SCRIPT_READY")
    print(f"preset={PRESET_NAME}")
    print(f"backend={args.backend}")
    print(f"trainer_mode={args.trainer_mode}")
    print(f"target_count={len(target_names)}")
    print(f"out_dir={args.out_dir}")

    import torch  # type: ignore
    from hy3dft.lora.peft_lora import assert_only_lora_trainable, freeze_all_parameters, inject_lora, parameter_summary

    from pytorch_lightning import Trainer  # type: ignore
    from pytorch_lightning.callbacks import Callback  # type: ignore
    from pytorch_lightning.strategies import DDPStrategy  # type: ignore

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for Phase 2M M2 training")

    model, data_module, runtime_metadata = setup_model_and_data(args.train_config, trainer_mode=args.trainer_mode)
    model.to(args.device)
    model.train()

    freeze_all_parameters(model)
    lora_root_path, lora_root = find_lora_root(model, target_names)
    injection = inject_lora(
        lora_root,
        target_names,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        backend=args.backend,
    )
    assert_only_lora_trainable(model)
    trainable_items = trainable_lora_parameters(model)
    optimizer = torch.optim.AdamW([parameter for _name, parameter in trainable_items], lr=args.learning_rate)
    optimizer_parameter_count = sum(parameter.numel() for _name, parameter in trainable_items)
    summary_before = parameter_summary(model)

    def configure_lora_optimizers(self: Any) -> Any:
        return optimizer

    model.configure_optimizers = types.MethodType(configure_lora_optimizers, model)

    adapter_config = {
        "phase": "2M",
        "preset_name": PRESET_NAME,
        "backend": injection.backend,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
        "save_every": args.save_every,
        "target_names": target_names,
        "target_count": len(target_names),
        "lora_root_path": lora_root_path,
        "runtime_metadata": runtime_metadata,
        "run_name": args.run_name,
        "smoke": args.smoke,
        "trainer_mode": args.trainer_mode,
        "m1_summary_path": str(args.m1_smoke_dir / "zero_lora_smoke_summary.json"),
        "m1_backend": m1_summary.get("backend"),
        "optimizer_parameter_count": optimizer_parameter_count,
        "safety": {
            "adapter_only": True,
            "merge_into_base": False,
            "save_pretrained_full_model": False,
            "full_hunyuan_model_saved": False,
        },
    }

    lifecycle_paths = configure_project_local_training_paths(model, args.trainer_log_dir, args.run_name)
    adapter_config["lifecycle_paths"] = lifecycle_paths
    callback = make_adapter_checkpoint_callback(Callback, lora_root, adapter_config, args.out_dir, args.max_train_steps, args.save_every, save_final=save_final)
    accelerator = "gpu" if args.device == "cuda" else "cpu"
    trainer_kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "devices": 1,
        "num_nodes": 1,
        "max_steps": args.max_train_steps,
        "max_epochs": -1,
        "default_root_dir": str(args.trainer_log_dir),
        "callbacks": [callback],
        "enable_checkpointing": False,
        "logger": False,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "benchmark": True,
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 10,
        "precision": int(args.precision) if str(args.precision).isdigit() else args.precision,
        "inference_mode": False,
    }
    if args.trainer_mode == "official_ddp":
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
    trainer = Trainer(**trainer_kwargs)

    trainer.logdir = lifecycle_paths["logdir"]
    print(f"model_logdir={lifecycle_paths['logdir']}")
    print(f"trainer_logdir={trainer.logdir}")
    print(f"images_val_dir={lifecycle_paths['images_val']}")
    print(f"lora_root_path={lora_root_path or '<model>'}")
    print(f"trainable_parameter_count={summary_before['trainable_parameter_count']}")
    print(f"total_parameter_count={summary_before['total_parameter_count']}")
    print(f"optimizer_parameter_count={optimizer_parameter_count}")
    trainer.fit(model, datamodule=data_module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
