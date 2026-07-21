from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


_FAKE_TORCH_MODULES: dict[str, object] = {}
_existing_torch = sys.modules.get("torch")
if _existing_torch is not None and not hasattr(_existing_torch, "__path__"):
    _FAKE_TORCH_MODULES = {
        name: module for name, module in sys.modules.items() if name == "torch" or name.startswith("torch.")
    }
    for name in _FAKE_TORCH_MODULES:
        sys.modules.pop(name, None)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import ConcatDataset, Dataset, Subset  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.protocol_corrected_dataset import (  # noqa: E402
    LIGHT_CONDITIONS,
    TARGET_VIEW_IDS,
    ProtocolCorrectedTextureDataset,
)
from hy3dft.selective_training import (  # noqa: E402
    HISTORICAL_MODEL_KEYS,
    ProtocolCorrectedDataModule,
    ProtocolCorrectedDatasetFromJson,
    apply_trainable_scope,
    build_trainable_adamw,
    build_warmup_constant_scheduler,
    protocol_collate_fn,
    set_epoch_recursive,
    warmup_constant_multiplier,
)


_REAL_TORCH_MODULES = {
    name: module for name, module in sys.modules.items() if name == "torch" or name.startswith("torch.")
}
if _FAKE_TORCH_MODULES:
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            sys.modules.pop(name, None)
    sys.modules.update(_FAKE_TORCH_MODULES)


TARGET_SUFFIXES = ("", "_albedo", "_mr", "_normal", "_pos")


@pytest.fixture(scope="module", autouse=True)
def real_torch_runtime():
    """Keep an older fake-torch unit test isolated from these real-torch tests."""

    if not _FAKE_TORCH_MODULES:
        yield
        return
    saved = {
        name: module for name, module in sys.modules.items() if name == "torch" or name.startswith("torch.")
    }
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            sys.modules.pop(name, None)
    sys.modules.update(_REAL_TORCH_MODULES)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def write_image(path: Path, color: tuple[int, int, int], size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


def make_sample(root: Path, asset_id: str, size: int = 8) -> Path:
    sample_dir = root / asset_id
    for view_index, view_id in enumerate(TARGET_VIEW_IDS):
        for light_index, light in enumerate(LIGHT_CONDITIONS):
            write_image(
                sample_dir / "render_cond" / f"{view_id}_light_{light}.png",
                (view_index * 20, light_index * 40, 10),
                size=size,
            )
        for kind_index, suffix in enumerate(TARGET_SUFFIXES):
            write_image(
                sample_dir / "render_tex" / f"{view_id}{suffix}.png",
                (view_index * 20, kind_index * 30, 20),
                size=size,
            )
    return sample_dir.resolve()


def write_examples_json(path: Path, sample_dirs: list[Path]) -> Path:
    path.write_text(json.dumps([str(sample_dir) for sample_dir in sample_dirs]), encoding="utf-8")
    return path


def directory_snapshot(root: Path) -> dict[str, tuple[bool, bytes | None]]:
    return {
        str(path.relative_to(root)): (path.is_dir(), None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    }


def test_json_wrapper_loads_absolute_list_and_delegates_to_day3(tmp_path: Path) -> None:
    samples = [make_sample(tmp_path, "ASSET_A"), make_sample(tmp_path, "ASSET_B")]
    examples_json = write_examples_json(tmp_path / "examples.json", samples)

    dataset = ProtocolCorrectedDatasetFromJson(examples_json, image_size=8)

    assert dataset.sample_dirs == tuple(samples)
    assert isinstance(dataset.dataset, ProtocolCorrectedTextureDataset)
    assert len(dataset) == 2


def test_json_wrapper_rejects_non_list_and_relative_paths(tmp_path: Path) -> None:
    not_a_list = tmp_path / "not_a_list.json"
    not_a_list.write_text(json.dumps({"sample": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a list"):
        ProtocolCorrectedDatasetFromJson(not_a_list)

    relative = tmp_path / "relative.json"
    relative.write_text(json.dumps(["relative/sample"]), encoding="utf-8")
    with pytest.raises(ValueError, match="not absolute"):
        ProtocolCorrectedDatasetFromJson(relative)


def test_wrapper_returns_model_keys_shapes_and_protocol_metadata(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path, "ASSET_A")
    dataset = ProtocolCorrectedDatasetFromJson(
        write_examples_json(tmp_path / "examples.json", [sample_dir]),
        image_size=8,
    )

    sample = dataset[0]

    assert set(sample) == set(HISTORICAL_MODEL_KEYS) | {"protocol_metadata"}
    assert tuple(sample["images_cond"].shape) == (2, 3, 8, 8)
    for key in ("images_albedo", "images_mr", "images_normal", "images_position"):
        assert tuple(sample[key].shape) == (6, 3, 8, 8)
    assert sample["name"] == str(sample_dir)


def test_protocol_collate_keeps_metadata_as_dict_list_and_large_seed_as_int() -> None:
    tensor = torch.zeros(2, 3, 4, 4)
    large_seed = 2**63 + 12345
    sample = {
        "images_cond": tensor,
        "images_albedo": torch.zeros(6, 3, 4, 4),
        "images_mr": torch.zeros(6, 3, 4, 4),
        "images_normal": torch.zeros(6, 3, 4, 4),
        "images_position": torch.zeros(6, 3, 4, 4),
        "name": "/tmp/asset",
        "protocol_metadata": {"asset_id": "asset", "derived_seed": large_seed},
    }

    batch = protocol_collate_fn([sample, sample])

    assert tuple(batch["images_cond"].shape) == (2, 2, 3, 4, 4)
    assert batch["name"] == ["/tmp/asset", "/tmp/asset"]
    assert isinstance(batch["protocol_metadata"], list)
    assert all(isinstance(record, dict) for record in batch["protocol_metadata"])
    assert batch["protocol_metadata"][0]["derived_seed"] == large_seed
    assert isinstance(batch["protocol_metadata"][0]["derived_seed"], int)


def test_wrapper_resolves_real_worker_id_without_global_random(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_dir = make_sample(tmp_path, "ASSET_A")
    dataset = ProtocolCorrectedDatasetFromJson(
        write_examples_json(tmp_path / "examples.json", [sample_dir]), image_size=8
    )
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=3))

    sample = dataset[0]

    assert sample["protocol_metadata"]["worker_id"] == 3
    assert dataset.dataset.worker_id == 3


def test_wrapper_epoch_reproducibility_and_changed_sequence(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path, "ASSET_A")
    dataset = ProtocolCorrectedDatasetFromJson(
        write_examples_json(tmp_path / "examples.json", [sample_dir]), image_size=8, base_seed=42
    )

    dataset.set_epoch(4)
    first = dataset[0]["protocol_metadata"]
    dataset.set_epoch(4)
    repeated = dataset[0]["protocol_metadata"]
    sequence = []
    for epoch in range(16):
        dataset.set_epoch(epoch)
        metadata = dataset[0]["protocol_metadata"]
        sequence.append((metadata["selected_reference_view"], tuple(metadata["reference_lighting_pair"])))

    assert first == repeated
    assert len(set(sequence)) > 1


@pytest.mark.parametrize("mode", ["", "rotate", "affine", "flip", "perspective"])
def test_wrapper_and_datamodule_reject_unsupported_augmentation(tmp_path: Path, mode: str) -> None:
    sample_dir = make_sample(tmp_path, "ASSET_A")
    examples_json = write_examples_json(tmp_path / "examples.json", [sample_dir])
    with pytest.raises(ValueError, match='exactly "none"'):
        ProtocolCorrectedDatasetFromJson(examples_json, augmentation_mode=mode)
    with pytest.raises(ValueError, match='exactly "none"'):
        ProtocolCorrectedDataModule(examples_json, augmentation_mode=mode)


def test_data_loading_writes_nothing_into_input_sample(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path, "ASSET_A")
    before = directory_snapshot(sample_dir)
    dataset = ProtocolCorrectedDatasetFromJson(
        write_examples_json(tmp_path / "examples.json", [sample_dir]), image_size=8
    )

    dataset[0]

    assert directory_snapshot(sample_dir) == before


def test_datamodule_setup_loaders_collation_and_epoch_propagation(tmp_path: Path) -> None:
    train_samples = [make_sample(tmp_path, "TRAIN_A"), make_sample(tmp_path, "TRAIN_B")]
    val_sample = make_sample(tmp_path, "VAL_A")
    train_json = write_examples_json(tmp_path / "train.json", train_samples)
    val_json = write_examples_json(tmp_path / "val.json", [val_sample])
    data = ProtocolCorrectedDataModule(
        train_json,
        val_json,
        batch_size=2,
        num_workers=0,
        image_size=8,
        shuffle_train=False,
    )

    data.prepare_data()
    data.setup("fit")
    data.set_epoch(6)
    train_batch = next(iter(data.train_dataloader()))
    val_batch = next(iter(data.val_dataloader()))

    assert len(train_batch["protocol_metadata"]) == 2
    assert all(record["epoch"] == 6 for record in train_batch["protocol_metadata"])
    assert len(val_batch["protocol_metadata"]) == 1
    assert val_batch["protocol_metadata"][0]["epoch"] == 6


def test_datamodule_without_validation_returns_none(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path, "TRAIN_A")
    data = ProtocolCorrectedDataModule(
        write_examples_json(tmp_path / "train.json", [sample_dir]), image_size=8
    )
    data.setup("fit")

    assert data.val_dataloader() is None


class EpochDataset(Dataset):
    def __init__(self) -> None:
        self.epoch = 0
        self.update_count = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> int:
        return index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.update_count += 1


def test_set_epoch_recursive_handles_subset_concat_duplicates_and_cycles() -> None:
    dataset = EpochDataset()
    wrapped = ConcatDataset([Subset(dataset, [0]), dataset])

    assert set_epoch_recursive(wrapped, 9) == 1
    assert dataset.epoch == 9
    assert dataset.update_count == 1

    cycle = SimpleNamespace()
    cycle.dataset = cycle
    assert set_epoch_recursive(cycle, 2) == 0
    with pytest.raises(ValueError, match="non-negative"):
        set_epoch_recursive(wrapped, -1)


class AttentionProjections(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.Sequential(nn.Linear(width, width))


class FakeHunyuanBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_multiview = AttentionProjections()
        self.attn_refview = AttentionProjections()
        self.attn_dino = AttentionProjections()
        self.attn1 = AttentionProjections()
        self.attn2 = AttentionProjections()
        self.ff = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
        self.conv_in = nn.Conv2d(4, 4, 1)
        self.conv_out = nn.Conv2d(4, 4, 1)
        self.resnet_block = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.unet_dual = nn.Linear(4, 4)


class FakeHunyuanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = nn.Module()
        self.unet.block = FakeHunyuanBlock()


def expected_s1_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, _ in model.named_parameters()
        if ".attn_multiview." in name
        and any(
            projection in name
            for projection in (".to_q.", ".to_k.", ".to_v.", ".to_out.0.")
        )
    }


def test_pc_full_preserves_existing_requires_grad_state_exactly() -> None:
    model = FakeHunyuanModel()
    for index, parameter in enumerate(model.parameters()):
        parameter.requires_grad = index % 3 == 0
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}

    report = apply_trainable_scope(model, "pc_full")

    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == before
    assert set(report.trainable_parameter_names) == {name for name, enabled in before.items() if enabled}
    assert report.total_parameter_tensor_count == len(before)
    assert json.loads(json.dumps(report.to_dict()))["scope"] == "pc_full"
    with pytest.raises(FrozenInstanceError):
        report.scope = "pc_s1"  # type: ignore[misc]


def test_pc_full_rejects_zero_trainable_state_without_unfreezing() -> None:
    model = FakeHunyuanModel()
    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(ValueError, match="zero"):
        apply_trainable_scope(model, "pc_full")
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_pc_s1_trains_only_exact_multiview_linear_projections() -> None:
    model = FakeHunyuanModel()
    expected = expected_s1_names(model)

    report = apply_trainable_scope(model, "pc_s1")

    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert actual == expected
    assert set(report.trainable_parameter_names) == expected
    assert len(expected) == 8
    forbidden = ("attn_refview", "attn_dino", ".attn1.", ".attn2.", ".ff.", "conv_", "resnet", ".norm", "unet_dual")
    assert all(not any(token in name for token in forbidden) for name in actual)


def test_pc_s1_rejects_empty_match_and_non_linear_owner() -> None:
    with pytest.raises(ValueError, match="zero"):
        apply_trainable_scope(nn.Sequential(nn.Linear(4, 4)), "pc_s1")

    model = FakeHunyuanModel()
    model.unet.block.attn_multiview.to_q = nn.Conv2d(4, 4, 1)
    with pytest.raises(TypeError, match="nn.Linear"):
        apply_trainable_scope(model, "pc_s1")


def test_pc_s1_rejects_duplicate_parameter_identity() -> None:
    model = FakeHunyuanModel()
    model.unet.block.attn_multiview.to_k.weight = model.unet.block.attn_multiview.to_q.weight

    with pytest.raises(ValueError, match="Duplicate PC-S1 parameter identity"):
        apply_trainable_scope(model, "pc_s1")


def test_pc_s1_rejects_forbidden_selected_prefix_and_unimplemented_scopes() -> None:
    model = nn.Module()
    model.attn_refview = nn.Module()
    model.attn_refview.attn_multiview = AttentionProjections()
    with pytest.raises(ValueError, match="forbidden path segment"):
        apply_trainable_scope(model, "pc_s1")

    with pytest.raises(ValueError, match="Unsupported"):
        apply_trainable_scope(FakeHunyuanModel(), "pc_s2")
    with pytest.raises(ValueError, match="Unsupported"):
        apply_trainable_scope(FakeHunyuanModel(), "pc_s3")


def test_optimizer_membership_exactly_matches_unique_trainable_parameters() -> None:
    model = FakeHunyuanModel()
    apply_trainable_scope(model, "pc_s1")

    optimizer = build_trainable_adamw(model, 1e-6, weight_decay=0.01)

    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in optimizer_parameters} == trainable_ids
    assert len(optimizer_parameters) == len(trainable_ids)
    assert optimizer.state == {}
    assert all(parameter.requires_grad for parameter in optimizer_parameters)


def test_optimizer_rejects_zero_trainable_and_duplicate_aliases() -> None:
    model = FakeHunyuanModel()
    for parameter in model.parameters():
        parameter.requires_grad = False
    with pytest.raises(ValueError, match="zero"):
        build_trainable_adamw(model, 1e-6)

    aliased = nn.Module()
    aliased.left = nn.Linear(4, 4)
    aliased.right = aliased.left
    with pytest.raises(ValueError, match="duplicated"):
        build_trainable_adamw(aliased, 1e-6)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.0), (25, 0.5), (50, 1.0), (160, 1.0), (320, 1.0), (1000, 1.0)],
)
def test_warmup_constant_multiplier_table(step: int, expected: float) -> None:
    assert warmup_constant_multiplier(step, warmup_steps=50) == expected


def test_warmup_constant_multiplier_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        warmup_constant_multiplier(-1)
    with pytest.raises(ValueError, match="positive"):
        warmup_constant_multiplier(0, warmup_steps=0)


def test_actual_scheduler_peaks_at_step_50_and_stays_through_320() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2.0)
    scheduler = build_warmup_constant_scheduler(optimizer, warmup_steps=50)
    observed = {scheduler.last_epoch: optimizer.param_groups[0]["lr"]}

    for step in range(1, 321):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        if step in {25, 50, 160, 320}:
            observed[step] = optimizer.param_groups[0]["lr"]

    assert scheduler.last_epoch == 320
    assert observed == {0: 0.0, 25: 1.0, 50: 2.0, 160: 2.0, 320: 2.0}
    assert max(observed.values()) <= 2.0
    with pytest.raises(ValueError, match="positive"):
        build_warmup_constant_scheduler(optimizer, warmup_steps=0)


def test_synthetic_dataloader_to_scope_optimizer_scheduler_assembly(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path, "TRAIN_A")
    data = ProtocolCorrectedDataModule(
        write_examples_json(tmp_path / "train.json", [sample_dir]),
        batch_size=1,
        num_workers=0,
        image_size=8,
        shuffle_train=False,
    )
    data.setup("fit")

    batch = next(iter(data.train_dataloader()))
    model = FakeHunyuanModel()
    report = apply_trainable_scope(model, "pc_s1")
    optimizer = build_trainable_adamw(model, 1e-6)
    scheduler = build_warmup_constant_scheduler(optimizer, warmup_steps=50)

    assert tuple(batch["images_cond"].shape) == (1, 2, 3, 8, 8)
    assert isinstance(batch["protocol_metadata"], list)
    assert report.trainable_parameter_tensor_count == 8
    assert len(optimizer.param_groups) == 1
    assert scheduler.last_epoch == 0
