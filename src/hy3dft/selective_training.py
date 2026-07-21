"""CPU-testable Phase 2N data and selective-training support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytorch_lightning as pl

from .protocol_corrected_dataset import ProtocolCorrectedTextureDataset


HISTORICAL_MODEL_KEYS = (
    "images_cond",
    "images_albedo",
    "images_mr",
    "images_normal",
    "images_position",
    "name",
)
ALLOWED_TRAINABLE_SCOPES = frozenset({"pc_full", "pc_s1"})
_S1_PROJECTION_SUFFIXES = {
    ("to_q",),
    ("to_k",),
    ("to_v",),
    ("to_out", "0"),
}


def _load_absolute_sample_paths(json_path: str | Path) -> tuple[Path, ...]:
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Examples JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Examples JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Examples JSON must contain a list: {path}")
    if not payload:
        raise ValueError(f"Examples JSON must contain at least one sample path: {path}")

    sample_paths: list[Path] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Examples JSON item {index} must be a non-empty path string")
        sample_path = Path(value).expanduser()
        if not sample_path.is_absolute():
            raise ValueError(f"Examples JSON item {index} is not absolute: {value}")
        sample_paths.append(sample_path)
    return tuple(sample_paths)


class ProtocolCorrectedDatasetFromJson:
    """Config-friendly JSON wrapper around the Day 3 selection-only reader."""

    def __init__(
        self,
        json_path: str | Path,
        image_size: int = 512,
        base_seed: int = 42,
        rank: int = 0,
        augmentation_mode: str = "none",
    ) -> None:
        if augmentation_mode != "none":
            raise ValueError('augmentation_mode must be exactly "none"')
        self.json_path = Path(json_path).expanduser().resolve()
        self.sample_dirs = _load_absolute_sample_paths(self.json_path)
        self.dataset = ProtocolCorrectedTextureDataset(
            self.sample_dirs,
            image_size=image_size,
            base_seed=base_seed,
            rank=rank,
            worker_id=0,
            epoch=0,
            augmentation_mode=augmentation_mode,
        )

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        # DataLoader workers own private dataset copies, so this cannot mutate a
        # sibling worker or the parent-process instance.
        self.dataset.worker_id = worker_id
        return self.dataset[index]


def protocol_collate_fn(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Collate model inputs normally while retaining metadata as Python dicts."""

    from torch.utils.data._utils.collate import default_collate

    if not batch:
        raise ValueError("Cannot collate an empty protocol batch")

    model_samples: list[dict[str, object]] = []
    metadata_records: list[dict[str, object]] = []
    for index, sample in enumerate(batch):
        missing = [key for key in HISTORICAL_MODEL_KEYS if key not in sample]
        if missing:
            raise KeyError(f"Batch item {index} is missing historical model keys: {missing}")
        metadata = sample.get("protocol_metadata")
        if not isinstance(metadata, dict):
            raise TypeError(f"Batch item {index} protocol_metadata must be a dict")
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Batch item {index} protocol_metadata is not JSON serializable") from exc
        model_samples.append({key: sample[key] for key in HISTORICAL_MODEL_KEYS})
        metadata_records.append(dict(metadata))

    collated = dict(default_collate(model_samples))
    collated["protocol_metadata"] = metadata_records
    return collated


class ProtocolCorrectedDataModule(pl.LightningDataModule):
    """Small Lightning-compatible bridge for protocol-corrected JSON splits."""

    def __init__(
        self,
        train_json: str | Path,
        val_json: str | Path | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        image_size: int = 512,
        base_seed: int = 42,
        rank: int = 0,
        augmentation_mode: str = "none",
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if augmentation_mode != "none":
            raise ValueError('augmentation_mode must be exactly "none"')
        self.train_json = Path(train_json).expanduser().resolve()
        self.val_json = None if val_json is None else Path(val_json).expanduser().resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.base_seed = base_seed
        self.rank = rank
        self.augmentation_mode = augmentation_mode
        self.shuffle_train = bool(shuffle_train)
        self.epoch = 0
        self.train_dataset: ProtocolCorrectedDatasetFromJson | None = None
        self.val_dataset: ProtocolCorrectedDatasetFromJson | None = None

    def prepare_data(self) -> None:
        """Lightning hook: all data already exists locally, so no action is needed."""

    def _make_dataset(self, json_path: Path) -> ProtocolCorrectedDatasetFromJson:
        dataset = ProtocolCorrectedDatasetFromJson(
            json_path=json_path,
            image_size=self.image_size,
            base_seed=self.base_seed,
            rank=self.rank,
            augmentation_mode=self.augmentation_mode,
        )
        dataset.set_epoch(self.epoch)
        return dataset

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            raise NotImplementedError(f"ProtocolCorrectedDataModule only supports fit stage, got {stage!r}")
        self.train_dataset = self._make_dataset(self.train_json)
        self.val_dataset = None if self.val_json is None else self._make_dataset(self.val_json)

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        if self.train_dataset is None:
            raise RuntimeError('Call setup("fit") before requesting train_dataloader()')
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            collate_fn=protocol_collate_fn,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=protocol_collate_fn,
        )

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch
        for dataset in (self.train_dataset, self.val_dataset):
            if dataset is not None:
                set_epoch_recursive(dataset, epoch)


@dataclass(frozen=True)
class TrainableScopeReport:
    scope: str
    total_parameter_tensor_count: int
    total_parameter_numel: int
    trainable_parameter_tensor_count: int
    trainable_parameter_numel: int
    frozen_parameter_tensor_count: int
    frozen_parameter_numel: int
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "total_parameter_tensor_count": self.total_parameter_tensor_count,
            "total_parameter_numel": self.total_parameter_numel,
            "trainable_parameter_tensor_count": self.trainable_parameter_tensor_count,
            "trainable_parameter_numel": self.trainable_parameter_numel,
            "frozen_parameter_tensor_count": self.frozen_parameter_tensor_count,
            "frozen_parameter_numel": self.frozen_parameter_numel,
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "frozen_parameter_names": list(self.frozen_parameter_names),
        }


def _named_parameters_with_duplicates(model: Any) -> list[tuple[str, Any]]:
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - supported by the project torch.
        raise RuntimeError("PyTorch named_parameters(remove_duplicate=False) is required for identity checks") from exc


def _named_modules_with_duplicates(model: Any) -> list[tuple[str, Any]]:
    try:
        return list(model.named_modules(remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - supported by the project torch.
        raise RuntimeError("PyTorch named_modules(remove_duplicate=False) is required for identity checks") from exc


def _scope_report(model: Any, scope: str) -> TrainableScopeReport:
    named_parameters = list(model.named_parameters())
    trainable = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in named_parameters if not parameter.requires_grad]
    return TrainableScopeReport(
        scope=scope,
        total_parameter_tensor_count=len(named_parameters),
        total_parameter_numel=sum(int(parameter.numel()) for _, parameter in named_parameters),
        trainable_parameter_tensor_count=len(trainable),
        trainable_parameter_numel=sum(int(parameter.numel()) for _, parameter in trainable),
        frozen_parameter_tensor_count=len(frozen),
        frozen_parameter_numel=sum(int(parameter.numel()) for _, parameter in frozen),
        trainable_parameter_names=tuple(name for name, _ in trainable),
        frozen_parameter_names=tuple(name for name, _ in frozen),
    )


def _is_s1_projection_module(module_name: str) -> bool:
    segments = tuple(module_name.split("."))
    for index, segment in enumerate(segments):
        if segment == "attn_multiview" and segments[index + 1 :] in _S1_PROJECTION_SUFFIXES:
            return True
    return False


def _forbidden_s1_segment(parameter_name: str) -> str | None:
    segments = tuple(parameter_name.split("."))
    exact_forbidden = {"attn_refview", "attn_dino", "attn1", "attn2", "conv_in", "conv_out", "unet_dual"}
    for segment in segments:
        if segment in exact_forbidden:
            return segment
        if segment == "ff" or segment.startswith("ff_"):
            return segment
        if segment.startswith("resnet"):
            return segment
        if segment == "norm" or segment.startswith("norm_"):
            return segment
    return None


def _select_pc_s1_parameters(model: Any) -> list[tuple[str, Any]]:
    import torch.nn as nn

    all_named = _named_parameters_with_duplicates(model)
    names_by_identity: dict[int, list[str]] = {}
    parameter_by_name: dict[str, Any] = {}
    for name, parameter in all_named:
        names_by_identity.setdefault(id(parameter), []).append(name)
        parameter_by_name[name] = parameter

    selected: list[tuple[str, Any]] = []
    for module_name, module in _named_modules_with_duplicates(model):
        if not _is_s1_projection_module(module_name):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"PC-S1 projection owner must be nn.Linear: {module_name} -> {type(module).__name__}")
        for leaf_name in ("weight", "bias"):
            parameter = getattr(module, leaf_name, None)
            if parameter is None:
                continue
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(f"PC-S1 {module_name}.{leaf_name} is not an nn.Parameter")
            full_name = f"{module_name}.{leaf_name}"
            if parameter_by_name.get(full_name) is not parameter:
                raise RuntimeError(f"PC-S1 selected parameter is not registered at its exact path: {full_name}")
            forbidden = _forbidden_s1_segment(full_name)
            if forbidden is not None:
                raise ValueError(f"PC-S1 selected forbidden path segment {forbidden!r}: {full_name}")
            aliases = names_by_identity[id(parameter)]
            if len(aliases) != 1:
                raise ValueError(f"Duplicate PC-S1 parameter identity for {full_name}: aliases={aliases}")
            selected.append((full_name, parameter))

    if not selected:
        raise ValueError("PC-S1 matched zero attn_multiview q/k/v/output projection parameters")
    selected_ids = [id(parameter) for _, parameter in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("PC-S1 selected duplicate parameter identities")
    return selected


def apply_trainable_scope(model: Any, scope: str) -> TrainableScopeReport:
    """Apply PC-Full preservation or fail-closed PC-S1 projection selection."""

    if scope not in ALLOWED_TRAINABLE_SCOPES:
        raise ValueError(f"Unsupported trainable scope {scope!r}; allowed={sorted(ALLOWED_TRAINABLE_SCOPES)}")
    if scope == "pc_full":
        report = _scope_report(model, scope)
        if report.trainable_parameter_tensor_count == 0:
            raise ValueError("PC-Full found zero parameters already marked trainable")
        return report

    selected = _select_pc_s1_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, parameter in selected:
        parameter.requires_grad = True
    report = _scope_report(model, scope)
    selected_names = tuple(name for name, _ in selected)
    if set(report.trainable_parameter_names) != set(selected_names):
        raise RuntimeError("PC-S1 report does not exactly match selected parameter names")
    return report


def build_trainable_adamw(model: Any, learning_rate: float, **optional_adamw_kwargs: object):
    """Build AdamW from the exact unique set of currently trainable parameters."""

    import torch

    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate < 0:
        raise ValueError("learning_rate must be a non-negative number")
    forbidden_kwargs = {"params", "lr"}.intersection(optional_adamw_kwargs)
    if forbidden_kwargs:
        raise ValueError(f"Optimizer parameters are controlled by this helper; remove kwargs {sorted(forbidden_kwargs)}")

    names_by_identity: dict[int, list[str]] = {}
    for name, parameter in _named_parameters_with_duplicates(model):
        if parameter.requires_grad:
            names_by_identity.setdefault(id(parameter), []).append(name)
    if not names_by_identity:
        raise ValueError("Cannot build AdamW with zero trainable parameters")
    duplicate_names = [names for names in names_by_identity.values() if len(names) != 1]
    if duplicate_names:
        raise ValueError(f"Trainable parameter identities are duplicated: {duplicate_names}")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_ids = {id(parameter) for parameter in trainable_parameters}
    if len(trainable_parameters) != len(trainable_ids):
        raise ValueError("Trainable parameter iteration produced duplicate identities")
    if trainable_ids != set(names_by_identity):
        raise RuntimeError("Named trainable parameter identities do not match model.parameters()")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(learning_rate), **optional_adamw_kwargs)
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    if len(optimizer_parameters) != len(optimizer_ids):
        raise RuntimeError("AdamW contains duplicate parameter identities")
    if optimizer_ids != trainable_ids:
        raise RuntimeError("AdamW membership does not exactly match the trainable parameter identity set")
    return optimizer


def warmup_constant_multiplier(step: int, warmup_steps: int = 50) -> float:
    """Linearly warm from zero, then remain exactly at multiplier one."""

    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool) or warmup_steps <= 0:
        raise ValueError("warmup_steps must be a positive integer")
    return min(float(step) / float(warmup_steps), 1.0)


def build_warmup_constant_scheduler(optimizer: Any, warmup_steps: int = 50):
    """Build a LambdaLR intended to be stepped once per optimizer update."""

    import torch

    warmup_constant_multiplier(0, warmup_steps=warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_constant_multiplier(step, warmup_steps=warmup_steps),
    )


def set_epoch_recursive(dataset_or_container: object, epoch: int) -> int:
    """Propagate an epoch once through common dataset wrappers."""

    from torch.utils.data import ConcatDataset, Subset

    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    visited: set[int] = set()
    update_count = 0

    def visit(node: object) -> None:
        nonlocal update_count
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, Subset):
            visit(node.dataset)
            return
        if isinstance(node, ConcatDataset):
            for child in node.datasets:
                visit(child)
            return
        if isinstance(node, Mapping):
            for child in node.values():
                visit(child)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
            return
        setter = getattr(node, "set_epoch", None)
        if callable(setter):
            setter(epoch)
            update_count += 1
            return
        child = getattr(node, "dataset", None)
        if child is not None:
            visit(child)
        children = getattr(node, "datasets", None)
        if children is not None:
            visit(children)

    visit(dataset_or_container)
    return update_count


__all__ = [
    "ALLOWED_TRAINABLE_SCOPES",
    "HISTORICAL_MODEL_KEYS",
    "ProtocolCorrectedDataModule",
    "ProtocolCorrectedDatasetFromJson",
    "TrainableScopeReport",
    "apply_trainable_scope",
    "build_trainable_adamw",
    "build_warmup_constant_scheduler",
    "protocol_collate_fn",
    "set_epoch_recursive",
    "warmup_constant_multiplier",
]
