from __future__ import annotations

import json
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.dataset import (  # noqa: E402
    LIGHT_CONDITIONS,
    REFERENCE_VIEW_WEIGHTS,
    TARGET_VIEW_IDS,
    InventoryValidationError,
    ProtocolCorrectedTextureDataset,
    inspect_sample_inventory,
    make_protocol_decision,
)


TARGET_SUFFIXES = ("", "_albedo", "_mr", "_normal", "_pos")


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def permute(self, *dimensions: int) -> "FakeTensor":
        return FakeTensor(self.array.transpose(dimensions))

    def contiguous(self) -> "FakeTensor":
        return self

    def float(self) -> "FakeTensor":
        return FakeTensor(self.array.astype(np.float32, copy=False))


@pytest.fixture
def torch_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_module = types.ModuleType("torch")
    torch_module.from_numpy = lambda array: FakeTensor(array)  # type: ignore[attr-defined]
    torch_module.stack = lambda tensors, dim=0: FakeTensor(  # type: ignore[attr-defined]
        np.stack([tensor.array for tensor in tensors], axis=dim)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)


def write_image(path: Path, color: tuple[int, int, int], size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


def make_sample(root: Path, asset_id: str = "ASSET_A", size: int = 8) -> Path:
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
    return sample_dir


def directory_snapshot(root: Path) -> dict[str, tuple[bool, bytes | None]]:
    snapshot: dict[str, tuple[bool, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        snapshot[str(path.relative_to(root))] = (path.is_dir(), None if path.is_dir() else path.read_bytes())
    return snapshot


def test_complete_inventory_validation(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path)

    inventory = inspect_sample_inventory(sample_dir)

    assert inventory.asset_id == "ASSET_A"
    assert inventory.sample_dir == sample_dir.resolve()
    assert len(inventory.reference_images) == 18
    assert len(inventory.target_views) == 6
    assert [target.view_id for target in inventory.target_views] == list(TARGET_VIEW_IDS)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path)
    (sample_dir / "render_tex" / "005_pos.png").unlink()

    with pytest.raises(InventoryValidationError, match="Missing target mapping 005/position"):
        inspect_sample_inventory(sample_dir)


def test_malformed_mapping_is_rejected(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path)
    write_image(sample_dir / "render_cond" / "000_light_SIDE.png", (0, 0, 0))

    with pytest.raises(InventoryValidationError, match="Unexpected condition lighting SIDE"):
        inspect_sample_inventory(sample_dir)


def test_duplicate_mapping_is_rejected(tmp_path: Path) -> None:
    sample_dir = make_sample(tmp_path)
    write_image(sample_dir / "render_tex" / "000_metallic_roughness.jpg", (0, 0, 0))

    with pytest.raises(InventoryValidationError, match="Duplicate target mapping 000/mr"):
        inspect_sample_inventory(sample_dir)


def test_reference_view_weights_are_exact_and_sum_to_one() -> None:
    assert dict(REFERENCE_VIEW_WEIGHTS) == {
        "005": 0.50,
        "004": 0.30,
        "000": 0.05,
        "001": 0.05,
        "002": 0.05,
        "003": 0.05,
    }
    assert sum(REFERENCE_VIEW_WEIGHTS.values()) == pytest.approx(1.0)


def test_protocol_decision_is_reproducible(tmp_path: Path) -> None:
    inventory = inspect_sample_inventory(make_sample(tmp_path))

    first = make_protocol_decision(inventory, base_seed=42, rank=1, worker_id=2, epoch=3)
    second = make_protocol_decision(inventory, base_seed=42, rank=1, worker_id=2, epoch=3)

    assert first == second
    assert first.derived_seed == second.derived_seed
    with pytest.raises(FrozenInstanceError):
        first.epoch = 4  # type: ignore[misc]


def test_changed_epoch_changes_deterministic_sampling_sequence(tmp_path: Path) -> None:
    inventory = inspect_sample_inventory(make_sample(tmp_path))
    decisions = [make_protocol_decision(inventory, epoch=epoch) for epoch in range(16)]

    assert len({decision.derived_seed for decision in decisions}) == 16
    assert len({(decision.selected_reference_view, decision.reference_lighting_pair) for decision in decisions}) > 1


def test_references_share_view_and_use_distinct_lights(tmp_path: Path) -> None:
    inventory = inspect_sample_inventory(make_sample(tmp_path))
    decision = make_protocol_decision(inventory)

    assert decision.reference_lighting_pair[0] != decision.reference_lighting_pair[1]
    for path, lighting in zip(decision.reference_image_paths, decision.reference_lighting_pair, strict=True):
        assert path.name == f"{decision.selected_reference_view}_light_{lighting}.png"


def test_target_paths_are_canonical_and_aligned(tmp_path: Path) -> None:
    inventory = inspect_sample_inventory(make_sample(tmp_path))

    assert [target.view_id for target in inventory.target_views] == list(TARGET_VIEW_IDS)
    for target in inventory.target_views:
        assert target.rgb_path.name == f"{target.view_id}.png"
        assert target.albedo_path.name == f"{target.view_id}_albedo.png"
        assert target.mr_path.name == f"{target.view_id}_mr.png"
        assert target.normal_path.name == f"{target.view_id}_normal.png"
        assert target.position_path.name == f"{target.view_id}_pos.png"


@pytest.mark.parametrize("unsupported", ["", "identity", "rotate", "affine", "flip", "perspective", None])
def test_every_unsupported_augmentation_mode_is_rejected(tmp_path: Path, unsupported: object) -> None:
    sample_dir = make_sample(tmp_path)

    with pytest.raises(ValueError, match='exactly "none"'):
        ProtocolCorrectedTextureDataset([sample_dir], augmentation_mode=unsupported)  # type: ignore[arg-type]


def test_augmentation_mode_none_is_the_only_supported_mode(tmp_path: Path, torch_stub: None) -> None:
    dataset = ProtocolCorrectedTextureDataset([make_sample(tmp_path)], augmentation_mode="none", image_size=8)

    assert dataset.augmentation_mode == "none"
    assert dataset[0]["protocol_metadata"]["spatial_augmentation"] == "none"  # type: ignore[index]


def test_protocol_metadata_is_json_serializable(tmp_path: Path) -> None:
    inventory = inspect_sample_inventory(make_sample(tmp_path))
    decision = make_protocol_decision(inventory, base_seed=7, rank=1, worker_id=2, epoch=3)

    payload = json.loads(json.dumps(decision.to_dict()))

    assert payload["asset_id"] == "ASSET_A"
    assert payload["selected_reference_view"] in REFERENCE_VIEW_WEIGHTS
    assert payload["reference_lighting_pair"][0] != payload["reference_lighting_pair"][1]
    assert payload["target_view_order"] == list(TARGET_VIEW_IDS)
    assert payload["base_seed"] == 7
    assert payload["rank"] == 1
    assert payload["worker_id"] == 2
    assert payload["epoch"] == 3
    assert payload["spatial_augmentation"] == "none"


def test_dataset_sample_matches_upstream_keys_and_shapes(tmp_path: Path, torch_stub: None) -> None:
    sample_dir = make_sample(tmp_path)
    dataset = ProtocolCorrectedTextureDataset([sample_dir], image_size=8)

    sample = dataset[0]

    assert set(sample) == {
        "images_cond",
        "images_albedo",
        "images_mr",
        "images_normal",
        "images_position",
        "name",
        "protocol_metadata",
    }
    assert tuple(sample["images_cond"].shape) == (2, 3, 8, 8)  # type: ignore[union-attr]
    for key in ("images_albedo", "images_mr", "images_normal", "images_position"):
        assert tuple(sample[key].shape) == (6, 3, 8, 8)  # type: ignore[union-attr]
    assert sample["name"] == str(sample_dir.resolve())
    assert json.loads(json.dumps(sample["protocol_metadata"]))["target_view_order"] == list(TARGET_VIEW_IDS)


def test_set_epoch_updates_reproducible_metadata(tmp_path: Path, torch_stub: None) -> None:
    dataset = ProtocolCorrectedTextureDataset([make_sample(tmp_path)], image_size=8)
    epoch_zero = dataset[0]["protocol_metadata"]
    dataset.set_epoch(5)
    epoch_five = dataset[0]["protocol_metadata"]
    dataset.set_epoch(5)
    repeated = dataset[0]["protocol_metadata"]

    assert epoch_zero["derived_seed"] != epoch_five["derived_seed"]  # type: ignore[index]
    assert epoch_five == repeated


def test_inventory_decision_and_loading_do_not_write_to_sample(tmp_path: Path, torch_stub: None) -> None:
    sample_dir = make_sample(tmp_path)
    before = directory_snapshot(sample_dir)

    inventory = inspect_sample_inventory(sample_dir)
    make_protocol_decision(inventory)
    ProtocolCorrectedTextureDataset([sample_dir], image_size=8)[0]

    assert directory_snapshot(sample_dir) == before
