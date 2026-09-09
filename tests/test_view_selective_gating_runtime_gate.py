from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from types import ModuleType

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.view_selective_gating import ViewSelectiveConditioningGate  # noqa: E402
from hy3dft.view_selective_gating.audit import parameter_sentinel  # noqa: E402
from hy3dft.view_selective_gating.runtime_gate import (  # noqa: E402
    AUDITED_BLOCK_PREFIXES,
    RuntimeGateError,
    build_processor_provenance_contracts,
    discover_attention_targets as _discover_attention_targets,
    validate_live_inference_contract,
)
from hy3dft.view_selective_gating.tensor_layout import TensorLayoutError  # noqa: E402


AZIMUTHS = [0.0, 90.0, 180.0, 270.0, 0.0, 180.0]
ELEVATIONS = [0.0, 0.0, 0.0, 0.0, 90.0, -90.0]
WEIGHTS = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
KEEP_MASK = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
EXPECTED_CAMERA_METADATA = [
    {
        "index": index,
        "elevation_degrees": elevation,
        "azimuth_degrees": azimuth,
    }
    for index, (elevation, azimuth) in enumerate(zip(ELEVATIONS, AZIMUTHS))
]


def _size(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _strides(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(_size(shape[index + 1 :]) for index in range(len(shape)))


def _coordinates(shape: Sequence[int]) -> Iterable[tuple[int, ...]]:
    return product(*(range(dimension) for dimension in shape))


def _offset(shape: Sequence[int], coordinates: Sequence[int]) -> int:
    return sum(index * stride for index, stride in zip(coordinates, _strides(shape)))


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def item(self) -> float:
        return self.value


class FakeTensor:
    """Small row-major tensor implementing only the gate's audited API."""

    def __init__(self, shape: Sequence[int], values: Sequence[float]) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.values = tuple(float(value) for value in values)
        if _size(self.shape) != len(self.values):
            raise ValueError("fake tensor shape/value mismatch")

    @classmethod
    def sequential(cls, shape: Sequence[int], start: float = 1.0) -> "FakeTensor":
        return cls(shape, [start + index for index in range(_size(shape))])

    def new_tensor(self, values: Sequence[float]) -> "FakeTensor":
        return FakeTensor((len(values),), values)

    def reshape(self, *shape: object) -> "FakeTensor":
        requested = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        dimensions = [int(value) for value in requested]  # type: ignore[arg-type]
        if dimensions.count(-1) > 1:
            raise ValueError("only one inferred dimension is supported")
        if -1 in dimensions:
            known = _size([value for value in dimensions if value != -1])
            dimensions[dimensions.index(-1)] = len(self.values) // known
        return FakeTensor(dimensions, self.values)

    def __getitem__(self, item: object) -> "FakeTensor":
        if not isinstance(item, tuple):
            selected = self.values[item]  # type: ignore[index]
            if isinstance(selected, tuple):
                return FakeTensor((len(selected),), selected)
            return FakeTensor((1,), [selected])

        selectors = item + (slice(None),) * (len(self.shape) - len(item))
        ranges: list[range] = []
        output_shape: list[int] = []
        for dimension, selector in zip(self.shape, selectors):
            if isinstance(selector, int):
                ranges.append(range(selector, selector + 1))
            elif isinstance(selector, slice):
                indices = range(*selector.indices(dimension))
                ranges.append(indices)
                output_shape.append(len(indices))
            else:
                raise TypeError(f"unsupported fake tensor index: {selector!r}")

        values = [
            self.values[_offset(self.shape, coordinates)]
            for coordinates in product(*ranges)
        ]
        return FakeTensor(output_shape or (1,), values)

    def __eq__(self, other: object) -> "FakeTensor":  # type: ignore[override]
        if not isinstance(other, (int, float)):
            return NotImplemented
        return FakeTensor(self.shape, [float(value == other) for value in self.values])

    def _broadcast_value(self, mask: "FakeTensor", coordinates: Sequence[int]) -> float:
        aligned = (1,) * (len(self.shape) - len(mask.shape)) + mask.shape
        if any(
            mask_dimension not in (1, output_dimension)
            for output_dimension, mask_dimension in zip(self.shape, aligned)
        ):
            raise ValueError(f"cannot broadcast {mask.shape} over {self.shape}")
        mask_coordinates = tuple(
            0 if mask_dimension == 1 else coordinate
            for coordinate, mask_dimension in zip(coordinates, aligned)
        )
        return mask.values[_offset(aligned, mask_coordinates)]

    def masked_fill(self, mask: "FakeTensor", value: float) -> "FakeTensor":
        return FakeTensor(
            self.shape,
            [
                float(value) if self._broadcast_value(mask, coordinates) else source
                for coordinates, source in zip(_coordinates(self.shape), self.values)
            ],
        )

    def detach(self) -> "FakeTensor":
        return self

    def numel(self) -> int:
        return len(self.values)

    def float(self) -> "FakeTensor":
        return self

    def norm(self) -> FakeScalar:
        return FakeScalar(math.sqrt(sum(value * value for value in self.values)))


class RefAttnProcessor2_0:
    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class AttnProcessor2_0:
    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class PoseRoPEAttnProcessor2_0:
    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        return None


RefAttnProcessor2_0.__module__ = "hy3dpaint.hunyuanpaintpbr.unet.attn_processor"
AttnProcessor2_0.__module__ = "diffusers.models.attention_processor"
PoseRoPEAttnProcessor2_0.__module__ = "hy3dpaint.hunyuanpaintpbr.unet.attn_processor"


class FakeHookHandle:
    def __init__(self, module: "Attention", hook_id: int) -> None:
        self.module = module
        self.hook_id = hook_id

    def remove(self) -> None:
        self.module._forward_hooks.pop(self.hook_id, None)


class Attention:
    def __init__(self, label: str, kind: str) -> None:
        self.label = label
        self._forward_hooks: dict[int, Callable[..., Any]] = {}
        self._next_hook_id = 0
        processor_class = {
            "attn_refview": RefAttnProcessor2_0,
            "attn_dino": AttnProcessor2_0,
            "attn_multiview": PoseRoPEAttnProcessor2_0,
        }[kind]
        self.processor = processor_class()

    @property
    def hooks(self) -> dict[int, Callable[..., Any]]:
        return self._forward_hooks

    def register_forward_hook(self, hook: Callable[..., Any]) -> FakeHookHandle:
        hook_id = self._next_hook_id
        self._next_hook_id += 1
        self._forward_hooks[hook_id] = hook
        return FakeHookHandle(self, hook_id)

    def forward(self, output: Any) -> Any:
        result = output
        for hook in list(self._forward_hooks.values()):
            replacement = hook(self, (), result)
            if replacement is not None:
                result = replacement
        return result


Attention.__module__ = "diffusers.models.attention_processor"

CANONICAL_PROCESSOR_SOURCE = Path(__file__).resolve()
CANONICAL_CUSTOM_PROCESSOR_MODULE = ModuleType(
    "hy3dpaint.hunyuanpaintpbr.unet.attn_processor"
)
CANONICAL_CUSTOM_PROCESSOR_MODULE.__file__ = str(CANONICAL_PROCESSOR_SOURCE)
CANONICAL_CUSTOM_PROCESSOR_MODULE.RefAttnProcessor2_0 = RefAttnProcessor2_0
CANONICAL_CUSTOM_PROCESSOR_MODULE.PoseRoPEAttnProcessor2_0 = (
    PoseRoPEAttnProcessor2_0
)
sys.modules[CANONICAL_CUSTOM_PROCESSOR_MODULE.__name__] = (
    CANONICAL_CUSTOM_PROCESSOR_MODULE
)

CANONICAL_DINO_PROCESSOR_MODULE = ModuleType("diffusers.models.attention_processor")
CANONICAL_DINO_PROCESSOR_MODULE.__file__ = str(CANONICAL_PROCESSOR_SOURCE)
CANONICAL_DINO_PROCESSOR_MODULE.AttnProcessor2_0 = AttnProcessor2_0
CANONICAL_DINO_PROCESSOR_MODULE.Attention = Attention
sys.modules[CANONICAL_DINO_PROCESSOR_MODULE.__name__] = (
    CANONICAL_DINO_PROCESSOR_MODULE
)


def make_processor_contracts():
    return build_processor_provenance_contracts(
        CANONICAL_PROCESSOR_SOURCE,
        AttnProcessor2_0,
    )


def discover_attention_targets(
    unet: Any,
    *,
    expected_count_per_kind: int,
    processor_provenance_contracts: Any | None = None,
):
    contracts = (
        make_processor_contracts()
        if processor_provenance_contracts is None
        else processor_provenance_contracts
    )
    return _discover_attention_targets(
        unet,
        expected_count_per_kind=expected_count_per_kind,
        processor_provenance_contracts=contracts,
    )



def load_custom_processor_module(
    source_path: Path,
    *,
    module_name: str = "diffusers_modules.local.attn_processor",
    callable_processors: bool = True,
) -> ModuleType:
    module = ModuleType(module_name)
    module.__file__ = str(source_path.resolve())
    call_body = (
        "    def __call__(self, *_args, **_kwargs):\n"
        "        return None\n"
        if callable_processors
        else ""
    )
    source = (
        "class RefAttnProcessor2_0:\n"
        + call_body
        + "\nclass PoseRoPEAttnProcessor2_0:\n"
        + call_body
    )
    exec(compile(source, str(source_path.resolve()), "exec"), module.__dict__)
    return module


def use_custom_processor_module(model: FakeUnet, module: ModuleType) -> None:
    for prefix in AUDITED_BLOCK_PREFIXES:
        model.modules_by_name[f"{prefix}.attn_refview"].processor = (
            module.RefAttnProcessor2_0()
        )
        model.modules_by_name[f"{prefix}.attn_multiview"].processor = (
            module.PoseRoPEAttnProcessor2_0()
        )


class FakeParameterSample:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)

    def float(self) -> "FakeParameterSample":
        return self

    def cpu(self) -> "FakeParameterSample":
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class FakeParameter:
    shape = (3,)
    dtype = "fake.float32"
    requires_grad = False
    _version = 0

    def __init__(self) -> None:
        self.values = [1.0, 2.0, 3.0]

    def numel(self) -> int:
        return len(self.values)

    def data_ptr(self) -> int:
        return 123456

    def detach(self) -> "FakeParameter":
        return self

    def reshape(self, _shape: int) -> "FakeParameter":
        return self

    def __getitem__(self, indices: Sequence[int]) -> FakeParameterSample:
        return FakeParameterSample([self.values[index] for index in indices])


class FakeUnet:
    def __init__(self) -> None:
        self.modules_by_name: dict[str, Attention] = {}
        self.parameter = FakeParameter()
        for prefix in AUDITED_BLOCK_PREFIXES:
            for kind in ("attn_refview", "attn_dino", "attn_multiview"):
                name = f"{prefix}.{kind}"
                self.modules_by_name[name] = Attention(name, kind)

    def named_modules(self):
        yield "", self
        yield from self.modules_by_name.items()

    def named_parameters(self):
        yield "weight", self.parameter

    def attention(self, block: int, kind: str) -> Attention:
        return self.modules_by_name[f"{AUDITED_BLOCK_PREFIXES[block]}.{kind}"]


class FakeViewProcessor:
    def __init__(
        self,
        elevations: Sequence[Any] = ELEVATIONS,
        azimuths: Sequence[Any] = AZIMUTHS,
        weights: Sequence[Any] = WEIGHTS,
    ) -> None:
        self.elevations = list(elevations)
        self.azimuths = list(azimuths)
        self.weights = list(weights)

    def bake_view_selection(self):
        return list(self.elevations), list(self.azimuths), list(self.weights)


def make_gate(
    processor_contracts: Any | None = None,
    *,
    unet: FakeUnet | None = None,
    processor: FakeViewProcessor | None = None,
    policy: str = "view_selective_gating",
    observations: int = 1,
    expected_calls: int = 15,
) -> tuple[FakeUnet, FakeViewProcessor, ViewSelectiveConditioningGate]:
    model = unet or FakeUnet()
    view_processor = processor or FakeViewProcessor()
    expected_mask = [1.0] * 6 if policy == "all_ones" else KEEP_MASK
    gate = ViewSelectiveConditioningGate(
        target_unet=model,
        view_processor=view_processor,
        policy=policy,
        expected_camera_metadata=EXPECTED_CAMERA_METADATA,
        expected_reference_keep_mask=expected_mask,
        metadata_units="degrees",
        reference_elevation_degrees=0.0,
        num_views=6,
        num_materials=2,
        expected_cfg_batch=2,
        expected_target_count=16,
        expected_calls_per_target=expected_calls,
        processor_provenance_contracts=(
            make_processor_contracts()
            if processor_contracts is None
            else processor_contracts
        ),
        max_norm_observations_per_target=observations,
    )
    return model, view_processor, gate


def test_discovery_requires_exact_16_prefixes_classes_and_processors() -> None:
    inventory = discover_attention_targets(FakeUnet(), expected_count_per_kind=16)

    assert inventory.block_prefixes == AUDITED_BLOCK_PREFIXES
    assert {kind: len(items) for kind, items in inventory.targets.items()} == {
        "attn_refview": 16,
        "attn_dino": 16,
        "attn_multiview": 16,
    }
    records = inventory.serializable()["targets"]
    assert records["attn_refview"][0]["class_name"] == (
        "diffusers.models.attention_processor.Attention"
    )
    assert records["attn_refview"][0]["processor_class_name"].endswith(
        ".RefAttnProcessor2_0"
    )
    assert records["attn_dino"][0]["processor_class_name"].endswith(
        ".AttnProcessor2_0"
    )
    assert records["attn_multiview"][0]["processor_class_name"].endswith(
        ".PoseRoPEAttnProcessor2_0"
    )
    for role in ("attn_refview", "attn_dino", "attn_multiview"):
        provenance = records[role][0]["processor_provenance"]
        assert provenance["role"] == role
        assert provenance["source_resolution"] == "canonical_path"
        assert provenance["source_matches_canonical_sha256"] is True
        assert provenance["live_path_matches_canonical_path"] is True
        assert provenance["processor_callable"] is True
        assert provenance["module_owns_class"] is True
    assert records["attn_dino"][0]["processor_provenance"][
        "canonical_class_identity_matches"
    ] is True


def test_diffusers_dynamic_namespace_requires_matching_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic_source = tmp_path / "dynamic/attn_processor.py"
    dynamic_source.parent.mkdir(parents=True)
    dynamic_source.write_bytes(CANONICAL_PROCESSOR_SOURCE.read_bytes())
    dynamic_module = load_custom_processor_module(dynamic_source)
    monkeypatch.setitem(sys.modules, dynamic_module.__name__, dynamic_module)
    model = FakeUnet()
    use_custom_processor_module(model, dynamic_module)

    inventory = discover_attention_targets(model, expected_count_per_kind=16)

    assert {role: len(items) for role, items in inventory.targets.items()} == {
        "attn_refview": 16,
        "attn_dino": 16,
        "attn_multiview": 16,
    }
    serialized = inventory.serializable()["targets"]
    for role in ("attn_refview", "attn_multiview"):
        provenance = serialized[role][0]["processor_provenance"]
        assert provenance["source_resolution"] == "approved_dynamic_copy"
        assert provenance["source_path"] == str(dynamic_source.resolve())
        assert provenance["source_matches_canonical_sha256"] is True
        assert provenance["class_module"] == (
            "diffusers_modules.local.attn_processor"
        )


def test_dynamic_processor_modified_source_wrong_role_and_namespace_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified_source = tmp_path / "modified/attn_processor.py"
    modified_source.parent.mkdir(parents=True)
    modified_source.write_bytes(
        CANONICAL_PROCESSOR_SOURCE.read_bytes() + b"\n# modified source\n"
    )
    modified_module = load_custom_processor_module(modified_source)
    monkeypatch.setitem(sys.modules, modified_module.__name__, modified_module)
    modified_model = FakeUnet()
    use_custom_processor_module(modified_model, modified_module)
    with pytest.raises(RuntimeGateError, match="digest mismatch"):
        discover_attention_targets(modified_model, expected_count_per_kind=16)

    exact_copy = tmp_path / "foreign/attn_processor.py"
    exact_copy.parent.mkdir(parents=True)
    exact_copy.write_bytes(CANONICAL_PROCESSOR_SOURCE.read_bytes())
    foreign_module = load_custom_processor_module(
        exact_copy, module_name="foreign.attn_processor"
    )
    monkeypatch.setitem(sys.modules, foreign_module.__name__, foreign_module)
    foreign_model = FakeUnet()
    use_custom_processor_module(foreign_model, foreign_module)
    with pytest.raises(RuntimeGateError, match="module foreign.attn_processor is not approved"):
        discover_attention_targets(foreign_model, expected_count_per_kind=16)

    matching_module = load_custom_processor_module(exact_copy)
    monkeypatch.setitem(sys.modules, matching_module.__name__, matching_module)
    wrong_role_model = FakeUnet()
    use_custom_processor_module(wrong_role_model, matching_module)
    wrong_role_model.attention(0, "attn_refview").processor = (
        matching_module.PoseRoPEAttnProcessor2_0()
    )
    with pytest.raises(RuntimeGateError, match="role/class mismatch"):
        discover_attention_targets(wrong_role_model, expected_count_per_kind=16)


def test_unresolved_dynamic_source_wrong_class_and_contracts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_source = tmp_path / "missing/attn_processor.py"
    missing_source.parent.mkdir(parents=True)
    missing_source.write_bytes(CANONICAL_PROCESSOR_SOURCE.read_bytes())
    missing_module = load_custom_processor_module(missing_source)
    monkeypatch.setitem(sys.modules, missing_module.__name__, missing_module)
    missing_model = FakeUnet()
    use_custom_processor_module(missing_model, missing_module)
    missing_source.unlink()
    with pytest.raises(RuntimeGateError, match="Could not resolve processor source"):
        discover_attention_targets(missing_model, expected_count_per_kind=16)

    wrong_class = FakeUnet()
    wrong_class.attention(0, "attn_dino").processor = RefAttnProcessor2_0()
    with pytest.raises(RuntimeGateError, match="role/class mismatch"):
        discover_attention_targets(wrong_class, expected_count_per_kind=16)

    with pytest.raises(RuntimeGateError, match="contracts are required"):
        _discover_attention_targets(
            FakeUnet(),
            expected_count_per_kind=16,
            processor_provenance_contracts=None,
        )
    malformed = make_processor_contracts()
    malformed["extra"] = malformed["attn_refview"]
    with pytest.raises(RuntimeGateError, match="roles must be exactly"):
        _discover_attention_targets(
            FakeUnet(),
            expected_count_per_kind=16,
            processor_provenance_contracts=malformed,
        )


def test_gate_revalidates_canonical_digest_before_installing_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_copy = tmp_path / "canonical/attn_processor.py"
    canonical_copy.parent.mkdir(parents=True)
    canonical_copy.write_bytes(CANONICAL_PROCESSOR_SOURCE.read_bytes())
    dynamic_module = load_custom_processor_module(canonical_copy)
    monkeypatch.setitem(sys.modules, dynamic_module.__name__, dynamic_module)
    model = FakeUnet()
    use_custom_processor_module(model, dynamic_module)
    contracts = build_processor_provenance_contracts(
        canonical_copy, AttnProcessor2_0
    )
    _model, processor, gate = make_gate(
        unet=model, observations=0, processor_contracts=contracts
    )
    camera_before = processor.bake_view_selection.__func__
    canonical_copy.write_bytes(canonical_copy.read_bytes() + b"\n# drift\n")

    with pytest.raises(RuntimeGateError, match="digest changed"):
        gate.install()

    assert gate.inventory is None
    assert gate._cleanup_done is True
    assert processor.bake_view_selection.__func__ is camera_before
    assert all(not module.hooks for module in model.modules_by_name.values())


def test_discovery_fails_on_count_prefix_class_processor_and_alias_drift() -> None:
    with pytest.raises(RuntimeGateError, match="requires exactly 16"):
        discover_attention_targets(FakeUnet(), expected_count_per_kind=15)

    missing = FakeUnet()
    missing.modules_by_name.pop(f"{AUDITED_BLOCK_PREFIXES[-1]}.attn_dino")
    with pytest.raises(RuntimeGateError, match="attn_dino count"):
        discover_attention_targets(missing, expected_count_per_kind=16)

    misaligned = FakeUnet()
    old_name = f"{AUDITED_BLOCK_PREFIXES[-1]}.attn_dino"
    module = misaligned.modules_by_name.pop(old_name)
    misaligned.modules_by_name["unet.other.transformer.attn_dino"] = module
    with pytest.raises(RuntimeGateError, match="exact 16 audited"):
        discover_attention_targets(misaligned, expected_count_per_kind=16)

    wrong_class = FakeUnet()
    original = wrong_class.attention(0, "attn_refview")
    lookalike_attention = type("Attention", (Attention,), {})
    lookalike_attention.__module__ = "foreign.attention"
    wrong_class.modules_by_name[f"{AUDITED_BLOCK_PREFIXES[0]}.attn_refview"] = (
        lookalike_attention(original.label, "attn_refview")
    )
    with pytest.raises(RuntimeGateError, match="!= audited"):
        discover_attention_targets(wrong_class, expected_count_per_kind=16)

    wrong_processor = FakeUnet()
    wrong_processor.attention(0, "attn_dino").processor = RefAttnProcessor2_0()
    with pytest.raises(RuntimeGateError, match="AttnProcessor2_0"):
        discover_attention_targets(wrong_processor, expected_count_per_kind=16)

    wrong_source_processor = FakeUnet()
    lookalike_processor = type("AttnProcessor2_0", (), {})
    lookalike_processor.__module__ = "foreign.attention_processor"
    wrong_source_processor.attention(0, "attn_dino").processor = (
        lookalike_processor()
    )
    with pytest.raises(RuntimeGateError, match="not callable"):
        discover_attention_targets(wrong_source_processor, expected_count_per_kind=16)

    aliased = FakeUnet()
    aliased.modules_by_name[f"{AUDITED_BLOCK_PREFIXES[1]}.attn_refview"] = (
        aliased.attention(0, "attn_refview")
    )
    with pytest.raises(RuntimeGateError, match="Aliased attn_refview"):
        discover_attention_targets(aliased, expected_count_per_kind=16)


def test_install_preserves_the_original_discovery_failure() -> None:
    unet = FakeUnet()
    unet.modules_by_name.pop(f"{AUDITED_BLOCK_PREFIXES[-1]}.attn_dino")
    _unet, processor, gate = make_gate(unet=unet, observations=0)
    camera_before = processor.bake_view_selection.__func__

    with pytest.raises(RuntimeGateError, match="Exact attn_dino count"):
        gate.install()

    assert gate.inventory is None
    assert gate._cleanup_done is True
    assert processor.bake_view_selection.__func__ is camera_before
    assert all(not module.hooks for module in unet.modules_by_name.values())


def test_runtime_camera_units_and_order_fail_closed() -> None:
    with pytest.raises(RuntimeGateError, match="metadata_units must be 'degrees'"):
        ViewSelectiveConditioningGate(
            target_unet=FakeUnet(),
            view_processor=FakeViewProcessor(),
            policy="view_selective_gating",
            expected_camera_metadata=EXPECTED_CAMERA_METADATA,
            expected_reference_keep_mask=KEEP_MASK,
            metadata_units="radians",
        )

    unet, processor, gate = make_gate(observations=0)
    reordered = [dict(row) for row in EXPECTED_CAMERA_METADATA]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    gate.expected_camera_metadata = tuple(reordered)

    with pytest.raises(RuntimeGateError, match="order/index mismatch"):
        with gate:
            processor.bake_view_selection()

    assert gate.camera_audit is not None
    assert gate.camera_audit["status"] == "FAIL_CLOSED"
    assert gate.restoration_status["fully_restored"] is True
    assert all(not module.hooks for module in unet.modules_by_name.values())


def test_live_inference_contract_uses_completed_call_observations() -> None:
    scheduler_class = type("UniPCMultistepScheduler", (), {})
    scheduler = scheduler_class()
    scheduler.timesteps = list(range(15))
    pipeline = type("FakeDiffusionPipeline", (), {})()
    pipeline.scheduler = scheduler
    pipeline._num_timesteps = 15
    pipeline._guidance_scale = 3.0

    report = validate_live_inference_contract(
        pipeline,
        expected_scheduler_class="UniPCMultistepScheduler",
        expected_denoising_steps=15,
        expected_guidance_scale=3.0,
    )

    assert report["status"] == "OK"
    assert report["evidence_source"] == "live completed official diffusion call"
    assert report["observed_pipeline_num_timesteps"] == 15
    assert report["observed_scheduler_timestep_count"] == 15
    assert report["observed_guidance_scale"] == 3.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scheduler_class", "DDIMScheduler", "Live scheduler"),
        ("pipeline_steps", 14, "denoising-step observations"),
        ("scheduler_steps", 14, "denoising-step observations"),
        ("guidance", 2.5, "Live guidance scale"),
    ],
)
def test_live_inference_contract_fails_closed_on_runtime_drift(
    field: str,
    value: Any,
    message: str,
) -> None:
    scheduler_class_name = (
        str(value) if field == "scheduler_class" else "UniPCMultistepScheduler"
    )
    scheduler_class = type(scheduler_class_name, (), {})
    scheduler = scheduler_class()
    scheduler.timesteps = list(
        range(int(value) if field == "scheduler_steps" else 15)
    )
    pipeline = type("FakeDiffusionPipeline", (), {})()
    pipeline.scheduler = scheduler
    pipeline._num_timesteps = int(value) if field == "pipeline_steps" else 15
    pipeline._guidance_scale = float(value) if field == "guidance" else 3.0

    with pytest.raises(RuntimeGateError, match=message):
        validate_live_inference_contract(
            pipeline,
            expected_scheduler_class="UniPCMultistepScheduler",
            expected_denoising_steps=15,
            expected_guidance_scale=3.0,
        )


def test_hard_gate_exactly_zeros_suppressed_ref_and_dino_values() -> None:
    unet, processor, gate = make_gate()
    ref_shape = (2, 2, 12, 2)
    dino_shape = (24, 2, 2)
    ref_values = [float(index + 1) for index in range(_size(ref_shape))]
    dino_values = [float(index + 101) for index in range(_size(dino_shape))]
    ref_values[_offset(ref_shape, (0, 0, 4, 0))] = math.nan
    dino_values[_offset(dino_shape, (2, 0, 0))] = math.nan
    original_refview = FakeTensor(ref_shape, ref_values)
    original_dino = FakeTensor(dino_shape, dino_values)

    with gate:
        processor.bake_view_selection()
        masked_refview = unet.attention(0, "attn_refview").forward(original_refview)
        masked_dino = unet.attention(0, "attn_dino").forward(original_dino)
        assert gate.reference_keep_mask == KEEP_MASK

        for coordinates in _coordinates(original_refview.shape):
            index = _offset(original_refview.shape, coordinates)
            view = coordinates[2] // 2
            if KEEP_MASK[view]:
                assert masked_refview.values[index] == original_refview.values[index]
            else:
                assert masked_refview.values[index] == 0.0

        for coordinates in _coordinates(original_dino.shape):
            index = _offset(original_dino.shape, coordinates)
            view = coordinates[0] % 6
            if KEEP_MASK[view]:
                assert masked_dino.values[index] == original_dino.values[index]
            else:
                assert masked_dino.values[index] == 0.0

    assert len(gate.observations) == 2
    assert all(
        row["per_view_sampled_residual_norm_after"][2] == 0.0
        for row in gate.observations
    )


def test_noop_is_same_object_identity_for_ref_and_dino() -> None:
    unet, processor, gate = make_gate(policy="all_ones")
    refview = FakeTensor.sequential((2, 2, 12, 2), start=-50.0)
    dino = FakeTensor.sequential((24, 2, 2), start=-50.0)

    with gate:
        processor.bake_view_selection()
        assert unet.attention(0, "attn_refview").forward(refview) is refview
        assert unet.attention(0, "attn_dino").forward(dino) is dino
        assert gate.reference_keep_mask == [1.0] * 6
        assert all(row["same_output_object"] for row in gate.observations)


def test_multiview_is_never_hooked_replaced_or_gated() -> None:
    unet, processor, gate = make_gate(observations=0)
    multiview = unet.attention(0, "attn_multiview")
    original_processor = multiview.processor
    output = FakeTensor.sequential((24, 2, 2))

    with gate:
        processor.bake_view_selection()
        assert multiview.hooks == {}
        assert multiview.processor is original_processor
        assert multiview.forward(output) is output
        active_audit = gate.audit_record()
        assert active_audit["patched_target_counts"]["attn_multiview"] == 0
        assert active_audit["attn_multiview_patched"] is False

    assert multiview.hooks == {}
    assert gate.restoration_status["attn_multiview_untouched"] is True


def test_invalid_live_layout_fails_closed_and_restores_hooks() -> None:
    unet, processor, gate = make_gate(observations=0)

    with pytest.raises(TensorLayoutError, match="not divisible"):
        with gate:
            processor.bake_view_selection()
            unet.attention(0, "attn_refview").forward(
                FakeTensor.sequential((2, 2, 11, 1))
            )

    assert gate.restoration_status["fully_restored"] is True
    assert all(not module.hooks for module in unet.modules_by_name.values())


def test_validate_complete_requires_all_32_targets_at_15_equal_calls() -> None:
    unet, processor, gate = make_gate(observations=0, expected_calls=15)
    refview = FakeTensor.sequential((2, 2, 12, 1))
    dino = FakeTensor.sequential((24, 2, 1))

    with gate:
        processor.bake_view_selection()
        for _step in range(15):
            for block in range(16):
                unet.attention(block, "attn_refview").forward(refview)
                unet.attention(block, "attn_dino").forward(dino)

    completion = gate.validate_complete()
    assert completion["status"] == "OK"
    assert completion["expected_calls_per_target"] == 15
    assert completion["expected_patched_target_count"] == 32
    assert completion["camera_capture_count"] == 1
    assert completion["all_hook_call_counts_equal"] is True
    assert completion["all_layouts_validated"] is True
    assert completion["fully_restored"] is True
    assert completion["errors"] == []
    assert set(gate.call_counts.values()) == {15}

    unet, processor, incomplete = make_gate(observations=0, expected_calls=15)
    with incomplete:
        processor.bake_view_selection()
        unet.attention(0, "attn_refview").forward(refview)
    with pytest.raises(RuntimeGateError, match="completion validation failed"):
        incomplete.validate_complete()
    assert incomplete.completion_validation["status"] == "FAIL_CLOSED"


def test_normal_teardown_restores_exact_preexisting_hooks_camera_and_processors() -> None:
    unet, processor, gate = make_gate(observations=0)
    refview = unet.attention(0, "attn_refview")
    sentinel_calls: list[str] = []

    def preexisting_hook(_module: Any, _inputs: Any, _output: Any) -> None:
        sentinel_calls.append("called")

    preexisting_handle = refview.register_forward_hook(preexisting_hook)
    hooks_before = {
        name: tuple(module.hooks.items()) for name, module in unet.modules_by_name.items()
    }
    processor_ids_before = {
        name: id(module.processor) for name, module in unet.modules_by_name.items()
    }
    camera_before = processor.bake_view_selection.__func__

    with gate:
        processor.bake_view_selection()
        refview.forward(FakeTensor.sequential((2, 2, 12, 1)))

    assert processor.bake_view_selection.__func__ is camera_before
    assert {
        name: tuple(module.hooks.items()) for name, module in unet.modules_by_name.items()
    } == hooks_before
    assert {
        name: id(module.processor) for name, module in unet.modules_by_name.items()
    } == processor_ids_before
    assert sentinel_calls == ["called"]
    assert gate.restoration_status["fully_restored"] is True
    assert gate.restoration_status["hook_registries_restored"] is True
    assert gate.restoration_status["processor_objects_restored"] is True

    status = dict(gate.restoration_status)
    gate.uninstall()
    assert gate.restoration_status == status
    preexisting_handle.remove()


def test_exception_teardown_restores_exact_hook_and_camera_identities() -> None:
    unet, processor, gate = make_gate(observations=0)
    hooks_before = {
        name: tuple(module.hooks.items()) for name, module in unet.modules_by_name.items()
    }
    camera_before = processor.bake_view_selection.__func__

    with pytest.raises(LookupError, match="synthetic inference failure"):
        with gate:
            processor.bake_view_selection()
            raise LookupError("synthetic inference failure")

    assert processor.bake_view_selection.__func__ is camera_before
    assert {
        name: tuple(module.hooks.items()) for name, module in unet.modules_by_name.items()
    } == hooks_before
    assert gate.restoration_status["fully_restored"] is True


def test_parameter_sentinel_is_preserved_across_gate_lifecycle() -> None:
    unet, processor, gate = make_gate(observations=0)
    before = parameter_sentinel(unet)

    with gate:
        processor.bake_view_selection()
        unet.attention(0, "attn_refview").forward(
            FakeTensor.sequential((2, 2, 12, 1))
        )
        unet.attention(0, "attn_dino").forward(
            FakeTensor.sequential((24, 1, 1))
        )

    assert parameter_sentinel(unet) == before
