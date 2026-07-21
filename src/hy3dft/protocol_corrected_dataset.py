"""Selection-only Phase 2N dataset core for Hunyuan3D-Paint examples."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Sequence


TARGET_VIEW_IDS = ("000", "001", "002", "003", "004", "005")
LIGHT_CONDITIONS = ("AL", "ENVMAP", "PL")
REFERENCE_VIEW_WEIGHTS = MappingProxyType(
    {
        "005": 0.50,
        "004": 0.30,
        "000": 0.05,
        "001": 0.05,
        "002": 0.05,
        "003": 0.05,
    }
)
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

_CONDITION_PATTERN = re.compile(r"^(?P<view>\d{3})_light_(?P<light>[A-Z]+)$")
_TARGET_PATTERN = re.compile(
    r"^(?P<view>\d{3})(?:_(?P<kind>albedo|mr|metallic_roughness|normal|pos|position))?$"
)
_TARGET_KIND_ALIASES = {
    None: "rgb",
    "albedo": "albedo",
    "mr": "mr",
    "metallic_roughness": "mr",
    "normal": "normal",
    "pos": "position",
    "position": "position",
}


class InventoryValidationError(ValueError):
    """Raised when a sample does not have one complete, unambiguous inventory."""


@dataclass(frozen=True)
class ReferenceImage:
    view_id: str
    lighting: str
    path: Path


@dataclass(frozen=True)
class TargetViewInventory:
    view_id: str
    rgb_path: Path
    albedo_path: Path
    mr_path: Path
    normal_path: Path
    position_path: Path

    def paths_by_kind(self) -> dict[str, Path]:
        return {
            "rgb": self.rgb_path,
            "albedo": self.albedo_path,
            "mr": self.mr_path,
            "normal": self.normal_path,
            "position": self.position_path,
        }


@dataclass(frozen=True)
class SampleInventory:
    sample_dir: Path
    asset_id: str
    reference_images: tuple[ReferenceImage, ...]
    target_views: tuple[TargetViewInventory, ...]

    def reference_path(self, view_id: str, lighting: str) -> Path:
        for image in self.reference_images:
            if image.view_id == view_id and image.lighting == lighting:
                return image.path
        raise KeyError(f"Missing reference image for view={view_id} lighting={lighting}")

    def target_for_view(self, view_id: str) -> TargetViewInventory:
        for target in self.target_views:
            if target.view_id == view_id:
                return target
        raise KeyError(f"Missing target inventory for view={view_id}")


@dataclass(frozen=True)
class ProtocolDecision:
    asset_id: str
    selected_reference_view: str
    reference_lighting_pair: tuple[str, str]
    reference_image_paths: tuple[Path, Path]
    target_view_order: tuple[str, ...]
    base_seed: int
    derived_seed: int
    rank: int
    worker_id: int
    epoch: int
    spatial_augmentation: str = "none"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable protocol audit record."""

        return {
            "asset_id": self.asset_id,
            "selected_reference_view": self.selected_reference_view,
            "reference_lighting_pair": list(self.reference_lighting_pair),
            "reference_image_paths": [str(path) for path in self.reference_image_paths],
            "target_view_order": list(self.target_view_order),
            "base_seed": self.base_seed,
            "derived_seed": self.derived_seed,
            "rank": self.rank,
            "worker_id": self.worker_id,
            "epoch": self.epoch,
            "spatial_augmentation": self.spatial_augmentation,
        }


def _supported_images(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )


def _format_inventory_errors(sample_dir: Path, errors: list[str]) -> InventoryValidationError:
    details = "\n".join(f"- {error}" for error in errors)
    return InventoryValidationError(f"Invalid Hunyuan3D-Paint sample inventory at {sample_dir}:\n{details}")


def inspect_sample_inventory(sample_dir: str | Path) -> SampleInventory:
    """Validate and return the complete image inventory for one sample."""

    sample_path = Path(sample_dir).expanduser().resolve()
    if not sample_path.is_dir():
        raise InventoryValidationError(f"Sample directory does not exist: {sample_path}")

    render_cond = sample_path / "render_cond"
    render_tex = sample_path / "render_tex"
    errors: list[str] = []
    if not render_cond.is_dir():
        errors.append(f"Missing directory: {render_cond}")
    if not render_tex.is_dir():
        errors.append(f"Missing directory: {render_tex}")
    if errors:
        raise _format_inventory_errors(sample_path, errors)

    references: dict[tuple[str, str], Path] = {}
    for path in _supported_images(render_cond):
        match = _CONDITION_PATTERN.fullmatch(path.stem)
        if match is None:
            errors.append(f"Malformed condition image name: {path.name}")
            continue
        view_id = match.group("view")
        lighting = match.group("light")
        if view_id not in TARGET_VIEW_IDS:
            errors.append(f"Unexpected condition view {view_id}: {path.name}")
            continue
        if lighting not in LIGHT_CONDITIONS:
            errors.append(f"Unexpected condition lighting {lighting}: {path.name}")
            continue
        key = (view_id, lighting)
        if key in references:
            errors.append(f"Duplicate condition mapping {view_id}/{lighting}: {references[key].name}, {path.name}")
            continue
        references[key] = path

    targets: dict[tuple[str, str], Path] = {}
    for path in _supported_images(render_tex):
        match = _TARGET_PATTERN.fullmatch(path.stem)
        if match is None:
            errors.append(f"Malformed target image name: {path.name}")
            continue
        view_id = match.group("view")
        if view_id not in TARGET_VIEW_IDS:
            errors.append(f"Unexpected target view {view_id}: {path.name}")
            continue
        kind = _TARGET_KIND_ALIASES[match.group("kind")]
        key = (view_id, kind)
        if key in targets:
            errors.append(f"Duplicate target mapping {view_id}/{kind}: {targets[key].name}, {path.name}")
            continue
        targets[key] = path

    for view_id in TARGET_VIEW_IDS:
        for lighting in LIGHT_CONDITIONS:
            if (view_id, lighting) not in references:
                errors.append(f"Missing condition mapping {view_id}/{lighting}")
        for kind in ("rgb", "albedo", "mr", "normal", "position"):
            if (view_id, kind) not in targets:
                errors.append(f"Missing target mapping {view_id}/{kind}")

    if errors:
        raise _format_inventory_errors(sample_path, errors)

    reference_records = tuple(
        ReferenceImage(view_id=view_id, lighting=lighting, path=references[(view_id, lighting)])
        for view_id in TARGET_VIEW_IDS
        for lighting in LIGHT_CONDITIONS
    )
    target_records = tuple(
        TargetViewInventory(
            view_id=view_id,
            rgb_path=targets[(view_id, "rgb")],
            albedo_path=targets[(view_id, "albedo")],
            mr_path=targets[(view_id, "mr")],
            normal_path=targets[(view_id, "normal")],
            position_path=targets[(view_id, "position")],
        )
        for view_id in TARGET_VIEW_IDS
    )
    return SampleInventory(
        sample_dir=sample_path,
        asset_id=sample_path.name,
        reference_images=reference_records,
        target_views=target_records,
    )


def _derive_seed(*, base_seed: int, rank: int, worker_id: int, epoch: int, asset_id: str) -> int:
    payload = json.dumps(
        {
            "asset_id": asset_id,
            "base_seed": base_seed,
            "epoch": epoch,
            "rank": rank,
            "worker_id": worker_id,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def _weighted_reference_view(rng: random.Random) -> str:
    threshold = rng.random() * sum(REFERENCE_VIEW_WEIGHTS.values())
    cumulative = 0.0
    for view_id, weight in REFERENCE_VIEW_WEIGHTS.items():
        cumulative += weight
        if threshold < cumulative:
            return view_id
    return next(reversed(REFERENCE_VIEW_WEIGHTS))


def make_protocol_decision(
    inventory: SampleInventory,
    base_seed: int = 42,
    rank: int = 0,
    worker_id: int = 0,
    epoch: int = 0,
) -> ProtocolDecision:
    """Choose one deterministic reference view and ordered lighting pair."""

    for label, value in (("rank", rank), ("worker_id", worker_id), ("epoch", epoch)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer")

    derived_seed = _derive_seed(
        base_seed=base_seed,
        rank=rank,
        worker_id=worker_id,
        epoch=epoch,
        asset_id=inventory.asset_id,
    )
    rng = random.Random(derived_seed)
    selected_view = _weighted_reference_view(rng)
    primary = LIGHT_CONDITIONS[rng.randrange(len(LIGHT_CONDITIONS))]
    secondary_pool = tuple(light for light in LIGHT_CONDITIONS if light != primary)
    secondary = secondary_pool[rng.randrange(len(secondary_pool))]
    lighting_pair = (primary, secondary)

    return ProtocolDecision(
        asset_id=inventory.asset_id,
        selected_reference_view=selected_view,
        reference_lighting_pair=lighting_pair,
        reference_image_paths=tuple(inventory.reference_path(selected_view, light) for light in lighting_pair),
        target_view_order=TARGET_VIEW_IDS,
        base_seed=base_seed,
        derived_seed=derived_seed,
        rank=rank,
        worker_id=worker_id,
        epoch=epoch,
    )


def _load_rgb_tensor(path: Path, image_size: int):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as source:
        source.load()
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
            background.alpha_composite(rgba)
            image = background.convert("RGB")
        else:
            image = source.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size))
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float()


class ProtocolCorrectedTextureDataset:
    """Load validated examples with deterministic Phase 2N reference selection."""

    def __init__(
        self,
        sample_dirs: Sequence[str | Path],
        *,
        base_seed: int = 42,
        rank: int = 0,
        worker_id: int = 0,
        epoch: int = 0,
        augmentation_mode: str = "none",
        image_size: int = 512,
    ) -> None:
        if isinstance(sample_dirs, (str, Path)):
            raise TypeError("sample_dirs must be a sequence of sample-directory paths")
        if augmentation_mode != "none":
            raise ValueError('augmentation_mode must be exactly "none"')
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")

        self.base_seed = base_seed
        self.rank = rank
        self.worker_id = worker_id
        self.epoch = epoch
        self.augmentation_mode = augmentation_mode
        self.image_size = image_size
        self.inventories = tuple(inspect_sample_inventory(path) for path in sample_dirs)
        if self.inventories:
            make_protocol_decision(
                self.inventories[0],
                base_seed=self.base_seed,
                rank=self.rank,
                worker_id=self.worker_id,
                epoch=self.epoch,
            )

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.inventories)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        inventory = self.inventories[index]
        decision = make_protocol_decision(
            inventory,
            base_seed=self.base_seed,
            rank=self.rank,
            worker_id=self.worker_id,
            epoch=self.epoch,
        )
        images_cond = [_load_rgb_tensor(path, self.image_size) for path in decision.reference_image_paths]
        targets = [inventory.target_for_view(view_id) for view_id in decision.target_view_order]

        return {
            "images_cond": torch.stack(images_cond, dim=0).float(),
            "images_albedo": torch.stack(
                [_load_rgb_tensor(target.albedo_path, self.image_size) for target in targets], dim=0
            ).float(),
            "images_mr": torch.stack(
                [_load_rgb_tensor(target.mr_path, self.image_size) for target in targets], dim=0
            ).float(),
            "images_normal": torch.stack(
                [_load_rgb_tensor(target.normal_path, self.image_size) for target in targets], dim=0
            ).float(),
            "images_position": torch.stack(
                [_load_rgb_tensor(target.position_path, self.image_size) for target in targets], dim=0
            ).float(),
            "name": str(inventory.sample_dir),
            "protocol_metadata": decision.to_dict(),
        }


__all__ = [
    "InventoryValidationError",
    "LIGHT_CONDITIONS",
    "ProtocolCorrectedTextureDataset",
    "ProtocolDecision",
    "REFERENCE_VIEW_WEIGHTS",
    "ReferenceImage",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SampleInventory",
    "TARGET_VIEW_IDS",
    "TargetViewInventory",
    "inspect_sample_inventory",
    "make_protocol_decision",
]
