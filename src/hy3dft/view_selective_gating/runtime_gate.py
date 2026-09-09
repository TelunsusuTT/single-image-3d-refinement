"""Fail-closed runtime hooks for view-selective conditioning gating."""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import re
import sys
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any, Callable, Mapping, Sequence

from .camera_gate import all_one_keep_mask, build_reference_keep_mask
from .tensor_layout import apply_dino_keep_mask, apply_refview_keep_mask


class RuntimeGateError(RuntimeError):
    """Raised when target discovery, live metadata, or restoration is unsafe."""


TARGET_KINDS = ("attn_refview", "attn_dino", "attn_multiview")
PATCHED_TARGET_KINDS = ("attn_refview", "attn_dino")
AUDITED_BLOCK_PREFIXES = (
    "unet.down_blocks.0.attentions.0.transformer_blocks.0",
    "unet.down_blocks.0.attentions.1.transformer_blocks.0",
    "unet.down_blocks.1.attentions.0.transformer_blocks.0",
    "unet.down_blocks.1.attentions.1.transformer_blocks.0",
    "unet.down_blocks.2.attentions.0.transformer_blocks.0",
    "unet.down_blocks.2.attentions.1.transformer_blocks.0",
    "unet.mid_block.attentions.0.transformer_blocks.0",
    "unet.up_blocks.1.attentions.0.transformer_blocks.0",
    "unet.up_blocks.1.attentions.1.transformer_blocks.0",
    "unet.up_blocks.1.attentions.2.transformer_blocks.0",
    "unet.up_blocks.2.attentions.0.transformer_blocks.0",
    "unet.up_blocks.2.attentions.1.transformer_blocks.0",
    "unet.up_blocks.2.attentions.2.transformer_blocks.0",
    "unet.up_blocks.3.attentions.0.transformer_blocks.0",
    "unet.up_blocks.3.attentions.1.transformer_blocks.0",
    "unet.up_blocks.3.attentions.2.transformer_blocks.0",
)
_EXPECTED_ATTENTION_CLASS = "diffusers.models.attention_processor.Attention"
_EXPECTED_PROCESSOR_QUALNAME = {
    "attn_refview": "RefAttnProcessor2_0",
    "attn_dino": "AttnProcessor2_0",
    "attn_multiview": "PoseRoPEAttnProcessor2_0",
}
_CANONICAL_CUSTOM_PROCESSOR_MODULES = (
    "hunyuanpaintpbr.unet.attn_processor",
    "hy3dpaint.hunyuanpaintpbr.unet.attn_processor",
)
_APPROVED_DYNAMIC_PROCESSOR_MODULE = "diffusers_modules.local.attn_processor"
_CANONICAL_DINO_PROCESSOR_MODULE = "diffusers.models.attention_processor"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProcessorProvenanceContract:
    role: str
    expected_class_name: str
    expected_class_qualname: str
    canonical_modules: tuple[str, ...]
    allowed_dynamic_module: str | None
    canonical_source_path: str
    canonical_source_sha256: str
    canonical_class: type[Any] | None = None

    def serializable(self) -> dict[str, Any]:
        canonical_class_name = None
        if self.canonical_class is not None:
            canonical_class_name = (
                f"{self.canonical_class.__module__}."
                f"{self.canonical_class.__qualname__}"
            )
        return {
            "role": self.role,
            "expected_class_name": self.expected_class_name,
            "expected_class_qualname": self.expected_class_qualname,
            "canonical_modules": list(self.canonical_modules),
            "allowed_dynamic_module": self.allowed_dynamic_module,
            "canonical_source_path": self.canonical_source_path,
            "canonical_source_sha256": self.canonical_source_sha256,
            "canonical_class_name": canonical_class_name,
        }


@dataclass(frozen=True)
class ProcessorProvenanceRecord:
    role: str
    class_name: str
    class_module: str
    class_qualname: str
    source_resolution: str
    source_path: str
    source_sha256: str
    canonical_source_path: str
    canonical_source_sha256: str
    source_matches_canonical_sha256: bool
    live_path_matches_canonical_path: bool
    processor_callable: bool
    module_registered: bool
    module_owns_class: bool
    module_source_matches_class_source: bool
    call_source_matches_class_source: bool
    canonical_class_identity_required: bool
    canonical_class_identity_matches: bool


@dataclass(frozen=True)
class TargetRecord:
    name: str
    class_name: str
    object_id: int
    processor_class_name: str
    processor_object_id: int
    processor_provenance: ProcessorProvenanceRecord
    forward_hook_ids_before: tuple[str, ...]


@dataclass(frozen=True)
class TargetInventory:
    targets: Mapping[str, tuple[tuple[str, Any], ...]]
    records: Mapping[str, tuple[TargetRecord, ...]]
    block_prefixes: tuple[str, ...]

    def serializable(self) -> dict[str, Any]:
        return {
            "counts": {kind: len(self.targets[kind]) for kind in TARGET_KINDS},
            "block_prefixes": list(self.block_prefixes),
            "targets": {
                kind: [asdict(record) for record in self.records[kind]]
                for kind in TARGET_KINDS
            },
        }


def _class_name(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def _forward_hook_ids(module: Any) -> tuple[str, ...]:
    registry = getattr(module, "_forward_hooks", None)
    if registry is None:
        registry = getattr(module, "hooks", None)
    if registry is None or not hasattr(registry, "items"):
        raise RuntimeGateError(
            f"Attention module {_class_name(module)} has no inspectable forward-hook registry"
        )
    return tuple(
        sorted(f"{key!r}:{id(value)}" for key, value in registry.items())
    )


def _processor(module: Any, name: str) -> Any:
    processor = getattr(module, "processor", None)
    if processor is None:
        raise RuntimeGateError(f"Attention target {name} has no live processor")
    return processor


def _sha256_path(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.expanduser().resolve()
    cached = cache.get(resolved)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeGateError(
            f"Could not hash processor source {resolved}: {type(exc).__name__}: {exc}"
        ) from exc
    value = digest.hexdigest()
    cache[resolved] = value
    return value


def _resolved_source_path(value: Any, label: str) -> Path:
    try:
        source = inspect.getsourcefile(value)
    except (OSError, TypeError) as exc:
        raise RuntimeGateError(
            f"Could not resolve processor source for {label}: {type(exc).__name__}: {exc}"
        ) from exc
    if not source:
        raise RuntimeGateError(f"Could not resolve processor source for {label}")
    path = Path(source).expanduser().resolve()
    try:
        is_regular_nonempty = path.is_file() and path.stat().st_size > 0
    except OSError as exc:
        raise RuntimeGateError(
            f"Could not inspect processor source for {label}: {path}: {exc}"
        ) from exc
    if not is_regular_nonempty:
        raise RuntimeGateError(
            f"Processor source for {label} is not a nonempty regular file: {path}"
        )
    return path


def _module_owned_class(module: ModuleType, qualname: str) -> Any:
    owner: Any = module
    for component in qualname.split("."):
        if component == "<locals>" or not component:
            return None
        owner = getattr(owner, component, None)
        if owner is None:
            return None
    return owner


def _direct_call_implementation(processor_class: type[Any], label: str) -> Any:
    implementation = vars(processor_class).get("__call__")
    if isinstance(implementation, (classmethod, staticmethod)):
        implementation = implementation.__func__
    if implementation is None or not callable(implementation):
        raise RuntimeGateError(
            f"Processor class for {label} has no directly declared callable __call__"
        )
    return implementation


def _validate_source_class_structure(path: Path, class_name: str, label: str) -> None:
    try:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RuntimeGateError(
            f"Could not parse canonical processor source for {label}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(definitions) != 1:
        raise RuntimeGateError(
            f"Canonical processor source for {label} must define exactly one "
            f"{class_name}; found {len(definitions)}"
        )
    call_definitions = [
        node for node in definitions[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__call__"
    ]
    if len(call_definitions) != 1:
        raise RuntimeGateError(
            f"Canonical processor class {class_name} for {label} must define "
            "exactly one __call__"
        )


def build_processor_provenance_contracts(
    canonical_hunyuan_source: str | Path,
    canonical_dino_class: type[Any],
) -> dict[str, ProcessorProvenanceContract]:
    """Build frozen, role-specific processor contracts from canonical sources."""

    hash_cache: dict[Path, str] = {}
    hunyuan_path = Path(canonical_hunyuan_source).expanduser().resolve()
    if not hunyuan_path.is_file() or hunyuan_path.stat().st_size <= 0:
        raise RuntimeGateError(
            f"Canonical Hunyuan processor source is unavailable: {hunyuan_path}"
        )
    hunyuan_sha256 = _sha256_path(hunyuan_path, hash_cache)
    for role in ("attn_refview", "attn_multiview"):
        _validate_source_class_structure(
            hunyuan_path, _EXPECTED_PROCESSOR_QUALNAME[role], role
        )

    if not isinstance(canonical_dino_class, type):
        raise RuntimeGateError("canonical_dino_class must be a class")
    if (
        canonical_dino_class.__module__ != _CANONICAL_DINO_PROCESSOR_MODULE
        or canonical_dino_class.__name__ != _EXPECTED_PROCESSOR_QUALNAME["attn_dino"]
        or canonical_dino_class.__qualname__ != _EXPECTED_PROCESSOR_QUALNAME["attn_dino"]
    ):
        raise RuntimeGateError(
            "canonical_dino_class is not the exact audited Diffusers AttnProcessor2_0"
        )
    dino_module = sys.modules.get(_CANONICAL_DINO_PROCESSOR_MODULE)
    if not isinstance(dino_module, ModuleType):
        raise RuntimeGateError("Canonical DINO processor module is not registered")
    if _module_owned_class(dino_module, canonical_dino_class.__qualname__) is not canonical_dino_class:
        raise RuntimeGateError("Canonical DINO processor module does not own the class")
    dino_path = _resolved_source_path(canonical_dino_class, "attn_dino canonical class")
    dino_module_path = _resolved_source_path(dino_module, "attn_dino canonical module")
    if dino_module_path != dino_path:
        raise RuntimeGateError("Canonical DINO module and class sources differ")
    _validate_source_class_structure(
        dino_path, _EXPECTED_PROCESSOR_QUALNAME["attn_dino"], "attn_dino"
    )
    dino_sha256 = _sha256_path(dino_path, hash_cache)

    contracts = {
        "attn_refview": ProcessorProvenanceContract(
            role="attn_refview",
            expected_class_name="RefAttnProcessor2_0",
            expected_class_qualname="RefAttnProcessor2_0",
            canonical_modules=_CANONICAL_CUSTOM_PROCESSOR_MODULES,
            allowed_dynamic_module=_APPROVED_DYNAMIC_PROCESSOR_MODULE,
            canonical_source_path=str(hunyuan_path),
            canonical_source_sha256=hunyuan_sha256,
        ),
        "attn_dino": ProcessorProvenanceContract(
            role="attn_dino",
            expected_class_name="AttnProcessor2_0",
            expected_class_qualname="AttnProcessor2_0",
            canonical_modules=(_CANONICAL_DINO_PROCESSOR_MODULE,),
            allowed_dynamic_module=None,
            canonical_source_path=str(dino_path),
            canonical_source_sha256=dino_sha256,
            canonical_class=canonical_dino_class,
        ),
        "attn_multiview": ProcessorProvenanceContract(
            role="attn_multiview",
            expected_class_name="PoseRoPEAttnProcessor2_0",
            expected_class_qualname="PoseRoPEAttnProcessor2_0",
            canonical_modules=_CANONICAL_CUSTOM_PROCESSOR_MODULES,
            allowed_dynamic_module=_APPROVED_DYNAMIC_PROCESSOR_MODULE,
            canonical_source_path=str(hunyuan_path),
            canonical_source_sha256=hunyuan_sha256,
        ),
    }
    return _validated_processor_contracts(contracts, hash_cache=hash_cache)


def _validated_processor_contracts(
    contracts: Mapping[str, ProcessorProvenanceContract] | None,
    *,
    hash_cache: dict[Path, str] | None = None,
) -> dict[str, ProcessorProvenanceContract]:
    if not isinstance(contracts, Mapping):
        raise RuntimeGateError("Processor provenance contracts are required")
    if set(contracts) != set(TARGET_KINDS):
        raise RuntimeGateError(
            "Processor provenance contract roles must be exactly "
            f"{list(TARGET_KINDS)}; got {sorted(str(key) for key in contracts)}"
        )
    hashes = hash_cache if hash_cache is not None else {}
    validated: dict[str, ProcessorProvenanceContract] = {}
    for role in TARGET_KINDS:
        contract = contracts.get(role)
        if not isinstance(contract, ProcessorProvenanceContract):
            raise RuntimeGateError(f"Malformed processor provenance contract for {role}")
        expected_qualname = _EXPECTED_PROCESSOR_QUALNAME[role]
        expected_modules = (
            (_CANONICAL_DINO_PROCESSOR_MODULE,)
            if role == "attn_dino"
            else _CANONICAL_CUSTOM_PROCESSOR_MODULES
        )
        expected_dynamic = (
            None if role == "attn_dino" else _APPROVED_DYNAMIC_PROCESSOR_MODULE
        )
        if (
            contract.role != role
            or contract.expected_class_name != expected_qualname
            or contract.expected_class_qualname != expected_qualname
            or tuple(contract.canonical_modules) != expected_modules
            or contract.allowed_dynamic_module != expected_dynamic
        ):
            raise RuntimeGateError(f"Processor provenance policy drift for {role}")
        if not _SHA256_RE.fullmatch(contract.canonical_source_sha256):
            raise RuntimeGateError(f"Invalid canonical processor SHA-256 for {role}")
        canonical_path = Path(contract.canonical_source_path).expanduser().resolve()
        if str(canonical_path) != contract.canonical_source_path:
            raise RuntimeGateError(f"Canonical processor path is not resolved for {role}")
        if not canonical_path.is_file() or canonical_path.stat().st_size <= 0:
            raise RuntimeGateError(
                f"Canonical processor source is unavailable for {role}: {canonical_path}"
            )
        current_sha256 = _sha256_path(canonical_path, hashes)
        if current_sha256 != contract.canonical_source_sha256:
            raise RuntimeGateError(
                f"Canonical processor source digest changed for {role}: "
                f"{current_sha256} != {contract.canonical_source_sha256}"
            )
        _validate_source_class_structure(canonical_path, expected_qualname, role)
        if role == "attn_dino":
            canonical_class = contract.canonical_class
            if not isinstance(canonical_class, type):
                raise RuntimeGateError("DINO contract lacks the canonical class")
            if (
                canonical_class.__module__ != _CANONICAL_DINO_PROCESSOR_MODULE
                or canonical_class.__name__ != expected_qualname
                or canonical_class.__qualname__ != expected_qualname
                or _resolved_source_path(canonical_class, role) != canonical_path
            ):
                raise RuntimeGateError("DINO canonical class provenance changed")
            canonical_module = sys.modules.get(_CANONICAL_DINO_PROCESSOR_MODULE)
            if (
                not isinstance(canonical_module, ModuleType)
                or _module_owned_class(canonical_module, expected_qualname)
                is not canonical_class
            ):
                raise RuntimeGateError("DINO canonical class ownership changed")
        elif contract.canonical_class is not None:
            raise RuntimeGateError(f"Custom processor contract for {role} must not pin a class")
        validated[role] = contract
    custom_left = validated["attn_refview"]
    custom_right = validated["attn_multiview"]
    if (
        custom_left.canonical_source_path != custom_right.canonical_source_path
        or custom_left.canonical_source_sha256 != custom_right.canonical_source_sha256
    ):
        raise RuntimeGateError("Custom processor roles do not share one canonical source")
    return validated


def _validate_processor_provenance(
    processor: Any,
    *,
    role: str,
    target_name: str,
    contract: ProcessorProvenanceContract,
    hash_cache: dict[Path, str],
) -> ProcessorProvenanceRecord:
    processor_class = type(processor)
    if not callable(processor):
        raise RuntimeGateError(f"Processor for {target_name} is not callable")
    if (
        processor_class.__name__ != contract.expected_class_name
        or processor_class.__qualname__ != contract.expected_class_qualname
    ):
        raise RuntimeGateError(
            f"Target {target_name} processor role/class mismatch: "
            f"{processor_class.__module__}.{processor_class.__qualname__} "
            f"!= {role}:{contract.expected_class_qualname}"
        )
    class_module = processor_class.__module__
    if class_module in contract.canonical_modules:
        source_resolution = "canonical_path"
    elif class_module == contract.allowed_dynamic_module:
        source_resolution = "approved_dynamic_copy"
    else:
        raise RuntimeGateError(
            f"Target {target_name} processor module {class_module} is not approved "
            f"for role {role}"
        )

    module = sys.modules.get(class_module)
    if not isinstance(module, ModuleType):
        raise RuntimeGateError(
            f"Target {target_name} processor module {class_module} is not registered"
        )
    if _module_owned_class(module, processor_class.__qualname__) is not processor_class:
        raise RuntimeGateError(
            f"Target {target_name} processor module does not own the live class"
        )
    class_source = _resolved_source_path(processor_class, target_name)
    module_source = _resolved_source_path(module, f"{target_name} module")
    if module_source != class_source:
        raise RuntimeGateError(
            f"Target {target_name} module/class sources differ: "
            f"{module_source} != {class_source}"
        )
    call_implementation = _direct_call_implementation(processor_class, target_name)
    call_source = _resolved_source_path(call_implementation, f"{target_name}.__call__")
    code = getattr(call_implementation, "__code__", None)
    code_filename = getattr(code, "co_filename", None)
    if not code_filename:
        raise RuntimeGateError(
            f"Target {target_name} __call__ has no inspectable code filename"
        )
    code_source = Path(str(code_filename)).expanduser().resolve()
    if call_source != class_source or code_source != class_source:
        raise RuntimeGateError(
            f"Target {target_name} __call__ implementation is not owned by "
            f"the resolved class source {class_source}"
        )
    try:
        class_source_text = inspect.getsource(processor_class)
    except (OSError, TypeError) as exc:
        raise RuntimeGateError(
            f"Could not inspect processor class structure for {target_name}: {exc}"
        ) from exc
    if not class_source_text.strip():
        raise RuntimeGateError(f"Processor class structure is empty for {target_name}")

    canonical_path = Path(contract.canonical_source_path).resolve()
    canonical_sha256 = _sha256_path(canonical_path, hash_cache)
    if canonical_sha256 != contract.canonical_source_sha256:
        raise RuntimeGateError(f"Canonical processor source changed for {target_name}")
    live_sha256 = _sha256_path(class_source, hash_cache)
    live_path_matches = class_source == canonical_path
    source_matches = live_sha256 == canonical_sha256
    if source_resolution == "canonical_path" and not live_path_matches:
        raise RuntimeGateError(
            f"Canonical processor namespace for {target_name} resolved to "
            f"{class_source}, expected {canonical_path}"
        )
    if not source_matches:
        raise RuntimeGateError(
            f"Processor source digest mismatch for {target_name}: "
            f"{live_sha256} != canonical {canonical_sha256}"
        )

    identity_required = role == "attn_dino"
    identity_matches = (
        not identity_required or processor_class is contract.canonical_class
    )
    if not identity_matches:
        raise RuntimeGateError(
            f"Target {target_name} DINO processor is not the exact canonical class"
        )
    return ProcessorProvenanceRecord(
        role=role,
        class_name=f"{class_module}.{processor_class.__qualname__}",
        class_module=class_module,
        class_qualname=processor_class.__qualname__,
        source_resolution=source_resolution,
        source_path=str(class_source),
        source_sha256=live_sha256,
        canonical_source_path=str(canonical_path),
        canonical_source_sha256=canonical_sha256,
        source_matches_canonical_sha256=source_matches,
        live_path_matches_canonical_path=live_path_matches,
        processor_callable=True,
        module_registered=True,
        module_owns_class=True,
        module_source_matches_class_source=True,
        call_source_matches_class_source=True,
        canonical_class_identity_required=identity_required,
        canonical_class_identity_matches=identity_matches,
    )



def discover_attention_targets(
    unet: Any,
    *,
    expected_count_per_kind: int,
    processor_provenance_contracts: Mapping[str, ProcessorProvenanceContract] | None = None,
) -> TargetInventory:
    """Discover the exact audited 16-by-three leading-UNet attention inventory."""

    audited_count = len(AUDITED_BLOCK_PREFIXES)
    if int(expected_count_per_kind) != audited_count:
        raise RuntimeGateError(
            f"View-selective gating requires exactly {audited_count} targets per kind; "
            f"configured {expected_count_per_kind}"
        )
    hash_cache: dict[Path, str] = {}
    validated_contracts = _validated_processor_contracts(
        processor_provenance_contracts,
        hash_cache=hash_cache,
    )

    found: dict[str, list[tuple[str, Any]]] = {kind: [] for kind in TARGET_KINDS}
    for name, module in unet.named_modules():
        module_name = str(name)
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in found:
            found[leaf].append((module_name, module))

    prefix_sets: dict[str, set[str]] = {}
    records: dict[str, tuple[TargetRecord, ...]] = {}
    frozen_targets: dict[str, tuple[tuple[str, Any], ...]] = {}
    for kind in TARGET_KINDS:
        items = sorted(found[kind], key=lambda item: item[0])
        if len(items) != audited_count:
            raise RuntimeGateError(
                f"Exact {kind} count {len(items)} != audited expected {audited_count}"
            )
        names = [name for name, _module in items]
        if len(names) != len(set(names)):
            raise RuntimeGateError(f"Duplicate {kind} module names discovered")
        object_ids = [id(module) for _name, module in items]
        if len(object_ids) != len(set(object_ids)):
            raise RuntimeGateError(f"Aliased {kind} module objects discovered")
        target_records: list[TargetRecord] = []
        for name, module in items:
            actual_class = _class_name(module)
            if actual_class != _EXPECTED_ATTENTION_CLASS:
                raise RuntimeGateError(
                    f"Target {name} class {actual_class} != audited "
                    f"{_EXPECTED_ATTENTION_CLASS}"
                )
            processor = _processor(module, name)
            processor_class = _class_name(processor)
            processor_provenance = _validate_processor_provenance(
                processor,
                role=kind,
                target_name=name,
                contract=validated_contracts[kind],
                hash_cache=hash_cache,
            )
            target_records.append(
                TargetRecord(
                    name=name,
                    class_name=_class_name(module),
                    object_id=id(module),
                    processor_class_name=processor_class,
                    processor_object_id=id(processor),
                    processor_provenance=processor_provenance,
                    forward_hook_ids_before=_forward_hook_ids(module),
                )
            )
        prefix_sets[kind] = {name.rsplit(".", 1)[0] for name in names}
        frozen_targets[kind] = tuple(items)
        records[kind] = tuple(target_records)

    audited_prefix_set = set(AUDITED_BLOCK_PREFIXES)
    if any(prefix_sets[kind] != audited_prefix_set for kind in TARGET_KINDS):
        raise RuntimeGateError(
            "refview/DINO/multiview targets do not match the exact 16 audited "
            "leading-UNet block prefixes"
        )
    object_groups = {
        kind: {id(module) for _name, module in frozen_targets[kind]}
        for kind in TARGET_KINDS
    }
    for left_index, left in enumerate(TARGET_KINDS):
        for right in TARGET_KINDS[left_index + 1 :]:
            if object_groups[left] & object_groups[right]:
                raise RuntimeGateError(
                    f"Attention module object is shared between {left} and {right}"
                )
    return TargetInventory(
        targets=frozen_targets,
        records=records,
        block_prefixes=AUDITED_BLOCK_PREFIXES,
    )


def _sample_norm(tensor: Any, max_elements: int = 512) -> float:
    flat = tensor.detach().reshape(-1)
    sample = flat[: min(int(flat.numel()), max_elements)]
    return float(sample.float().norm().item())


def _per_view_sample_norms(
    tensor: Any,
    layout: Mapping[str, int],
    max_elements: int = 512,
) -> list[float]:
    cfg_batch = int(layout["cfg_batch"])
    materials = int(layout["materials"])
    views = int(layout["views"])
    tokens = int(layout["tokens_per_view"])
    channels = int(layout["channels"])
    logical = tensor.detach().reshape(cfg_batch * materials, views, tokens * channels)
    return [
        _sample_norm(logical[:, view_index, :], max_elements=max_elements)
        for view_index in range(views)
    ]


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


def _safe_sequence_repr(values: Any) -> Any:
    try:
        return [_safe_repr(value) for value in values]
    except Exception:
        return _safe_repr(values)


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeGateError(f"{label} must be a finite real number, got {_safe_repr(value)}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeGateError(f"{label} must be finite, got {_safe_repr(value)}")
    return result


def validate_live_inference_contract(
    diffusion_pipeline: Any,
    *,
    expected_scheduler_class: str,
    expected_denoising_steps: int,
    expected_guidance_scale: float,
) -> dict[str, Any]:
    """Validate settings observed from the completed official denoising call."""

    scheduler = getattr(diffusion_pipeline, "scheduler", None)
    if scheduler is None:
        raise RuntimeGateError("Live diffusion pipeline has no scheduler")
    scheduler_class = scheduler.__class__.__name__
    if scheduler_class != expected_scheduler_class:
        raise RuntimeGateError(
            f"Live scheduler {scheduler_class!r} != expected {expected_scheduler_class!r}"
        )

    configured_steps = int(expected_denoising_steps)
    observed_steps = getattr(diffusion_pipeline, "_num_timesteps", None)
    if isinstance(observed_steps, bool) or not isinstance(observed_steps, int):
        raise RuntimeGateError(
            "Live diffusion pipeline did not expose an integer _num_timesteps "
            "from the completed denoising call"
        )
    timesteps = getattr(scheduler, "timesteps", None)
    try:
        scheduler_timestep_count = len(timesteps)
    except (TypeError, AttributeError):
        raise RuntimeGateError(
            "Live scheduler did not expose completed-call timesteps"
        ) from None
    if observed_steps != configured_steps or scheduler_timestep_count != configured_steps:
        raise RuntimeGateError(
            "Live denoising-step observations differ from the View-selective gating contract: "
            f"pipeline={observed_steps}, scheduler={scheduler_timestep_count}, "
            f"expected={configured_steps}"
        )

    observed_guidance = _finite_real(
        getattr(diffusion_pipeline, "_guidance_scale", None),
        "live diffusion pipeline _guidance_scale",
    )
    configured_guidance = _finite_real(
        expected_guidance_scale, "expected_guidance_scale"
    )
    if observed_guidance != configured_guidance:
        raise RuntimeGateError(
            f"Live guidance scale {observed_guidance} != expected {configured_guidance}"
        )
    return {
        "status": "OK",
        "evidence_source": "live completed official diffusion call",
        "live_scheduler_class": scheduler_class,
        "observed_pipeline_num_timesteps": observed_steps,
        "observed_scheduler_timestep_count": scheduler_timestep_count,
        "observed_guidance_scale": observed_guidance,
        "configured_denoising_steps": configured_steps,
        "configured_guidance_scale": configured_guidance,
    }


class ViewSelectiveConditioningGate:
    """Install only audited refview/DINO residual hooks; never hook multiview."""

    def __init__(
        self,
        *,
        target_unet: Any,
        view_processor: Any | None,
        policy: str,
        expected_camera_metadata: Sequence[Mapping[str, Any]] | None = None,
        expected_reference_keep_mask: Sequence[float] | None = None,
        metadata_units: str = "degrees",
        reference_elevation_degrees: float = 0.0,
        num_views: int = 6,
        num_materials: int = 2,
        expected_cfg_batch: int | None = 3,
        expected_target_count: int = 16,
        expected_calls_per_target: int = 15,
        processor_provenance_contracts: (
            Mapping[str, ProcessorProvenanceContract] | None
        ) = None,
        reference_azimuth_degrees: float = 0.0,
        suppression_threshold_degrees: float = 120.0,
        max_norm_observations_per_target: int = 1,
    ) -> None:
        if policy not in {"view_selective_gating", "all_ones"}:
            raise ValueError(f"Unsupported Gating policy: {policy!r}")
        if metadata_units != "degrees":
            raise RuntimeGateError(
                f"View-selective gating camera metadata_units must be 'degrees', got {metadata_units!r}"
            )
        if int(expected_calls_per_target) <= 0:
            raise RuntimeGateError("expected_calls_per_target must be positive")
        if int(max_norm_observations_per_target) < 0:
            raise RuntimeGateError("max_norm_observations_per_target must be non-negative")
        self.target_unet = target_unet
        self.view_processor = view_processor
        self.policy = policy
        self.expected_camera_metadata = (
            tuple(expected_camera_metadata) if expected_camera_metadata is not None else None
        )
        self.metadata_units = metadata_units
        self.reference_elevation_degrees = _finite_real(
            reference_elevation_degrees, "reference_elevation_degrees"
        )
        self.num_views = int(num_views)
        self.num_materials = int(num_materials)
        if expected_reference_keep_mask is None:
            self.expected_reference_keep_mask: tuple[float, ...] | None = None
        else:
            expected_mask = tuple(
                _finite_real(value, f"expected_reference_keep_mask[{index}]")
                for index, value in enumerate(expected_reference_keep_mask)
            )
            if len(expected_mask) != self.num_views:
                raise RuntimeGateError(
                    f"expected_reference_keep_mask count {len(expected_mask)} "
                    f"!= expected {self.num_views}"
                )
            if any(value not in {0.0, 1.0} for value in expected_mask):
                raise RuntimeGateError(
                    "expected_reference_keep_mask must contain only binary 0/1 values"
                )
            self.expected_reference_keep_mask = expected_mask
        self.expected_cfg_batch = expected_cfg_batch
        self.expected_target_count = int(expected_target_count)
        self.expected_calls_per_target = int(expected_calls_per_target)
        self.processor_provenance_contracts = _validated_processor_contracts(
            processor_provenance_contracts
        )
        self.reference_azimuth_degrees = _finite_real(
            reference_azimuth_degrees, "reference_azimuth_degrees"
        )
        self.suppression_threshold_degrees = _finite_real(
            suppression_threshold_degrees, "suppression_threshold_degrees"
        )
        self.max_norm_observations_per_target = int(max_norm_observations_per_target)

        self.inventory: TargetInventory | None = None
        self.reference_keep_mask: list[float] | None = None
        self.camera_audit: dict[str, Any] | None = None
        self.observations: list[dict[str, Any]] = []
        self.call_counts: dict[str, int] = {}
        self.validated_layouts: dict[str, dict[str, int]] = {}
        self.completion_validation: dict[str, Any] | None = None
        self._handles: list[Any] = []
        self._camera_original: Any | None = None
        self._camera_had_instance_attribute = False
        self._camera_wrapper_installed = False
        self._camera_capture_count = 0
        self._installed = False
        self._ever_installed = False
        self._cleanup_done = True
        self.restoration_status: dict[str, Any] = {
            "hooks_removed": False,
            "camera_selection_restored": view_processor is None,
            "module_objects_restored": False,
            "hook_registries_restored": False,
            "processor_objects_restored": False,
            "attn_multiview_untouched": False,
            "fully_restored": False,
            "errors": [],
        }

    def _record_camera_failure(
        self,
        exc: BaseException,
        *,
        elevations: Any = None,
        azimuths: Any = None,
        weights: Any = None,
    ) -> None:
        prior = (
            self.camera_audit
            if self.camera_audit and self.camera_audit.get("status") == "OK"
            else None
        )
        self.reference_keep_mask = None
        self.camera_audit = {
            "status": "FAIL_CLOSED",
            "policy": self.policy,
            "metadata_units": self.metadata_units,
            "capture_count": self._camera_capture_count,
            "camera_elevations_repr": _safe_sequence_repr(elevations),
            "camera_azimuths_repr": _safe_sequence_repr(azimuths),
            "view_weights_repr": _safe_sequence_repr(weights),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if prior is not None:
            self.camera_audit["prior_successful_audit"] = prior

    def _validate_expected_camera_metadata(
        self,
        camera_elevations: Sequence[float],
        camera_azimuths: Sequence[float],
    ) -> list[dict[str, float | int]]:
        if self.expected_camera_metadata is None:
            raise RuntimeGateError(
                "expected_camera_metadata is required; refusing unaudited live camera order"
            )
        if len(self.expected_camera_metadata) != self.num_views:
            raise RuntimeGateError(
                f"expected_camera_metadata count {len(self.expected_camera_metadata)} "
                f"!= expected {self.num_views}"
            )
        if len(camera_elevations) != self.num_views or len(camera_azimuths) != self.num_views:
            raise RuntimeGateError(
                "live camera vectors do not match the configured generation-view count"
            )
        matched: list[dict[str, float | int]] = []
        for index, expected in enumerate(self.expected_camera_metadata):
            if not isinstance(expected, Mapping):
                raise RuntimeGateError(
                    f"expected_camera_metadata[{index}] must be a mapping"
                )
            expected_index = expected.get("index")
            if isinstance(expected_index, bool) or expected_index != index:
                raise RuntimeGateError(
                    f"expected_camera_metadata order/index mismatch at {index}: "
                    f"{_safe_repr(expected_index)}"
                )
            expected_elevation = _finite_real(
                expected.get("elevation_degrees"),
                f"expected_camera_metadata[{index}].elevation_degrees",
            )
            expected_azimuth = _finite_real(
                expected.get("azimuth_degrees"),
                f"expected_camera_metadata[{index}].azimuth_degrees",
            )
            live_elevation = _finite_real(
                camera_elevations[index], f"camera_elevations[{index}]"
            )
            live_azimuth = _finite_real(
                camera_azimuths[index], f"camera_azimuths[{index}]"
            )
            if not math.isclose(
                live_elevation, expected_elevation, rel_tol=0.0, abs_tol=1e-6
            ):
                raise RuntimeGateError(
                    f"generation camera elevation/order mismatch at index {index}: "
                    f"live {live_elevation} != expected {expected_elevation} degrees"
                )
            azimuth_delta = (live_azimuth - expected_azimuth + 180.0) % 360.0 - 180.0
            if not math.isclose(azimuth_delta, 0.0, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeGateError(
                    f"generation camera azimuth/order mismatch at index {index}: "
                    f"live {live_azimuth} != expected {expected_azimuth} degrees"
                )
            matched.append(
                {
                    "index": index,
                    "elevation_degrees": live_elevation,
                    "azimuth_degrees": live_azimuth,
                }
            )
        return matched

    def configure_camera_metadata(
        self,
        camera_elevations: Sequence[float],
        camera_azimuths: Sequence[float],
        view_weights: Sequence[float] | None = None,
    ) -> list[float]:
        try:
            matched_metadata = self._validate_expected_camera_metadata(
                camera_elevations, camera_azimuths
            )
            builder_kwargs = {
                "reference_azimuth_degrees": self.reference_azimuth_degrees,
                "reference_elevation_degrees": self.reference_elevation_degrees,
                "expected_num_views": self.num_views,
            }
            if self.policy == "view_selective_gating":
                mask, records = build_reference_keep_mask(
                    camera_azimuths,
                    camera_elevations,
                    suppression_threshold_degrees=self.suppression_threshold_degrees,
                    **builder_kwargs,
                )
            else:
                mask, records = all_one_keep_mask(
                    camera_azimuths, camera_elevations, **builder_kwargs
                )
            if self.expected_reference_keep_mask is None:
                raise RuntimeGateError(
                    "expected_reference_keep_mask is required before applying live residual gating"
                )
            if tuple(mask) != self.expected_reference_keep_mask:
                raise RuntimeGateError(
                    "metadata-derived reference keep mask differs from the audited expectation: "
                    f"derived={mask}, expected={list(self.expected_reference_keep_mask)}"
                )
            weights: list[float] | None = None
            if view_weights is not None:
                if len(view_weights) != self.num_views:
                    raise RuntimeGateError(
                        f"selected view-weight count {len(view_weights)} != expected {self.num_views}"
                    )
                weights = [
                    _finite_real(value, f"view_weights[{index}]")
                    for index, value in enumerate(view_weights)
                ]
            self.reference_keep_mask = mask
            self.camera_audit = {
                "status": "OK",
                "policy": self.policy,
                "metadata_units": self.metadata_units,
                "capture_count": self._camera_capture_count,
                "expected_metadata_matched_in_order": True,
                "matched_expected_camera_metadata": matched_metadata,
                "selected_camera_count": len(records),
                "selected_cameras": records,
                "selected_view_weights": weights,
                "reference_keep_mask": mask,
                "expected_reference_keep_mask": list(self.expected_reference_keep_mask),
                "mask_matches_audit_expectation": True,
                "gated_indices": [index for index, value in enumerate(mask) if value == 0.0],
                "mask_semantics": {
                    "1.0": "keep reference/DINO residual",
                    "0.0": "suppress residual",
                },
            }
            return mask
        except Exception as exc:
            self._record_camera_failure(
                exc,
                elevations=camera_elevations,
                azimuths=camera_azimuths,
                weights=view_weights,
            )
            raise

    def _install_camera_wrapper(self) -> None:
        if self.view_processor is None:
            raise RuntimeGateError(
                "Gating requires the live view_processor to capture generation cameras"
            )
        instance_dict = getattr(self.view_processor, "__dict__", {})
        self._camera_had_instance_attribute = "bake_view_selection" in instance_dict
        self._camera_original = getattr(self.view_processor, "bake_view_selection")
        gate = self

        def wrapped(_instance: Any, *args: Any, **kwargs: Any) -> Any:
            gate._camera_capture_count += 1
            if gate._camera_capture_count != 1:
                error = RuntimeGateError(
                    "Official bake_view_selection must be captured exactly once per inference"
                )
                gate._record_camera_failure(error)
                raise error
            original = gate._camera_original
            if original is None:
                error = RuntimeGateError("Camera wrapper lost its original method")
                gate._record_camera_failure(error)
                raise error
            try:
                result = original(*args, **kwargs)
                if not isinstance(result, (tuple, list)) or len(result) != 3:
                    raise RuntimeGateError(
                        "Official bake_view_selection no longer returns "
                        "(elevations, azimuths, weights)"
                    )
                elevations, azimuths, weights = result
                gate.configure_camera_metadata(elevations, azimuths, weights)
                return result
            except Exception as exc:
                if gate.camera_audit is None or gate.camera_audit.get("status") != "FAIL_CLOSED":
                    gate._record_camera_failure(exc)
                raise

        setattr(
            self.view_processor,
            "bake_view_selection",
            MethodType(wrapped, self.view_processor),
        )
        self._camera_wrapper_installed = True

    def _mask_for_hook(self) -> list[float]:
        if self.reference_keep_mask is None:
            raise RuntimeGateError(
                "Attention residual reached Gating before audited generation-camera metadata was captured"
            )
        return self.reference_keep_mask

    def _record_observation(
        self,
        *,
        branch: str,
        name: str,
        before: Any,
        after: Any,
        layout: Mapping[str, int],
        broadcast_shape: Sequence[int],
    ) -> None:
        key = f"{branch}:{name}"
        normalized_layout = {
            str(layout_key): int(value) for layout_key, value in layout.items()
        }
        previous_layout = self.validated_layouts.get(key)
        if previous_layout is not None and previous_layout != normalized_layout:
            raise RuntimeGateError(
                f"Live tensor layout changed across calls for {key}: "
                f"{previous_layout} != {normalized_layout}"
            )
        self.validated_layouts[key] = normalized_layout
        call_index = self.call_counts.get(key, 0)
        self.call_counts[key] = call_index + 1
        if call_index >= self.max_norm_observations_per_target:
            return
        self.observations.append(
            {
                "branch": branch,
                "module_name": name,
                "call_index": call_index,
                "layout": normalized_layout,
                "broadcast_shape": [int(value) for value in broadcast_shape],
                "same_output_object": before is after,
                "sampled_residual_norm_before": _sample_norm(before),
                "sampled_residual_norm_after": _sample_norm(after),
                "per_view_sampled_residual_norm_before": _per_view_sample_norms(
                    before, normalized_layout
                ),
                "per_view_sampled_residual_norm_after": _per_view_sample_norms(
                    after, normalized_layout
                ),
            }
        )

    def _make_hook(self, branch: str, name: str) -> Callable[..., Any]:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            mask = self._mask_for_hook()
            if branch == "attn_refview":
                application = apply_refview_keep_mask(
                    output,
                    mask,
                    num_views=self.num_views,
                    num_materials=self.num_materials,
                    expected_cfg_batch=self.expected_cfg_batch,
                )
            elif branch == "attn_dino":
                application = apply_dino_keep_mask(
                    output,
                    mask,
                    num_views=self.num_views,
                    num_materials=self.num_materials,
                    expected_cfg_batch=self.expected_cfg_batch,
                )
            else:
                raise RuntimeGateError(f"Internal error: unsupported hook branch {branch}")
            all_one_mask = all(value == 1.0 for value in mask)
            if all_one_mask and application.output is not output:
                raise RuntimeGateError(
                    f"Gating no-op replaced the residual object for {branch}:{name}"
                )
            self._record_observation(
                branch=branch,
                name=name,
                before=output,
                after=application.output,
                layout=application.layout,
                broadcast_shape=application.broadcast_shape,
            )
            return None if all_one_mask else application.output
        return hook

    def install(self) -> "ViewSelectiveConditioningGate":
        if self._installed:
            raise RuntimeGateError("Gating runtime gate is already installed")
        if self._ever_installed:
            raise RuntimeGateError("Gating runtime gate instances cannot be reinstalled")
        self._ever_installed = True
        try:
            self.inventory = discover_attention_targets(
                self.target_unet,
                expected_count_per_kind=self.expected_target_count,
                processor_provenance_contracts=self.processor_provenance_contracts,
            )
            self._cleanup_done = False
            self._install_camera_wrapper()
            for branch in PATCHED_TARGET_KINDS:
                for name, module in self.inventory.targets[branch]:
                    self._handles.append(
                        module.register_forward_hook(self._make_hook(branch, name))
                    )
            self._installed = True
            return self
        except Exception:
            if not self._cleanup_done:
                self.uninstall()
            raise

    def _target_integrity(self, *, include_hook_registries: bool) -> dict[str, Any]:
        if self.inventory is None:
            return {
                "module_objects_match": False,
                "processors_match": False,
                "hook_registries_match": False,
                "multiview_untouched": False,
                "mismatches": ["target inventory unavailable"],
            }
        try:
            live_modules = {
                str(name): module for name, module in self.target_unet.named_modules()
            }
        except Exception as exc:
            return {
                "module_objects_match": False,
                "processors_match": False,
                "hook_registries_match": False,
                "multiview_untouched": False,
                "mismatches": [
                    f"could not enumerate live modules: {type(exc).__name__}: {exc}"
                ],
            }
        hash_cache: dict[Path, str] = {}
        try:
            validated_contracts = _validated_processor_contracts(
                self.processor_provenance_contracts,
                hash_cache=hash_cache,
            )
        except Exception as exc:
            return {
                "module_objects_match": False,
                "processors_match": False,
                "hook_registries_match": False,
                "multiview_untouched": False,
                "mismatches": [
                    f"processor provenance contract revalidation failed: {type(exc).__name__}: {exc}"
                ],
            }
        module_matches = True
        processor_matches = True
        hook_matches = True
        multiview_matches = True
        mismatches: list[str] = []
        for kind in TARGET_KINDS:
            records_by_name = {
                record.name: record for record in self.inventory.records[kind]
            }
            for name, _original_module in self.inventory.targets[kind]:
                record = records_by_name[name]
                module = live_modules.get(name)
                module_ok = module is not None and id(module) == record.object_id
                if not module_ok:
                    module_matches = False
                    if kind == "attn_multiview":
                        multiview_matches = False
                    mismatches.append(f"{name}: module object changed")
                    continue
                try:
                    processor = _processor(module, name)
                    processor_ok = (
                        id(processor) == record.processor_object_id
                        and _class_name(processor) == record.processor_class_name
                    )
                except Exception as exc:
                    processor_ok = False
                    mismatches.append(
                        f"{name}: processor inspection failed: {type(exc).__name__}: {exc}"
                    )
                if not processor_ok:
                    processor_matches = False
                    if kind == "attn_multiview":
                        multiview_matches = False
                    mismatches.append(f"{name}: processor object/type changed")
                if include_hook_registries or kind == "attn_multiview":
                    try:
                        hooks_ok = _forward_hook_ids(module) == record.forward_hook_ids_before
                    except Exception as exc:
                        hooks_ok = False
                        mismatches.append(
                            f"{name}: hook inspection failed: {type(exc).__name__}: {exc}"
                        )
                    if not hooks_ok:
                        hook_matches = False
                        if kind == "attn_multiview":
                            multiview_matches = False
                        mismatches.append(f"{name}: forward-hook registry changed")
        return {
            "module_objects_match": module_matches,
            "processors_match": processor_matches,
            "hook_registries_match": hook_matches,
            "multiview_untouched": multiview_matches,
            "mismatches": mismatches,
        }

    def uninstall(self) -> None:
        if self._cleanup_done:
            return
        removal_errors: list[str] = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as exc:
                removal_errors.append(f"hook removal {type(exc).__name__}: {exc}")
        self._handles.clear()

        camera_restored = (
            self.view_processor is None or not self._camera_wrapper_installed
        )
        if (
            self._camera_wrapper_installed
            and self.view_processor is not None
            and self._camera_original is not None
        ):
            original = self._camera_original
            try:
                if self._camera_had_instance_attribute:
                    setattr(self.view_processor, "bake_view_selection", original)
                else:
                    delattr(self.view_processor, "bake_view_selection")
                restored = getattr(self.view_processor, "bake_view_selection")
                restored_func = getattr(restored, "__func__", restored)
                original_func = getattr(original, "__func__", original)
                camera_restored = restored_func is original_func
            except Exception as exc:
                removal_errors.append(
                    f"camera restoration {type(exc).__name__}: {exc}"
                )
                camera_restored = False
        self._camera_original = None
        self._camera_wrapper_installed = False

        integrity = self._target_integrity(include_hook_registries=True)
        if not integrity["module_objects_match"]:
            removal_errors.append("target module objects were not restored")
        if not integrity["hook_registries_match"]:
            removal_errors.append("forward-hook registries were not restored")
        if not integrity["processors_match"]:
            removal_errors.append("attention processor objects were not restored")
        if not integrity["multiview_untouched"]:
            removal_errors.append("attn_multiview integrity check failed")
        hooks_removed = not any(
            error.startswith("hook removal") for error in removal_errors
        )
        self.restoration_status = {
            "hooks_removed": hooks_removed,
            "camera_selection_restored": camera_restored,
            "module_objects_restored": bool(integrity["module_objects_match"]),
            "hook_registries_restored": bool(integrity["hook_registries_match"]),
            "processor_objects_restored": bool(integrity["processors_match"]),
            "attn_multiview_untouched": bool(integrity["multiview_untouched"]),
            "integrity_mismatches": list(integrity["mismatches"]),
            "errors": removal_errors,
            "fully_restored": (
                hooks_removed
                and camera_restored
                and bool(integrity["module_objects_match"])
                and bool(integrity["hook_registries_match"])
                and bool(integrity["processors_match"])
                and bool(integrity["multiview_untouched"])
                and not removal_errors
            ),
        }
        self._installed = False
        self._cleanup_done = True
        if removal_errors:
            raise RuntimeGateError(
                "Gating restoration failed: " + "; ".join(removal_errors)
            )

    def validate_complete(self) -> dict[str, Any]:
        """Prove one complete configured-step smoke traversal after teardown."""

        errors: list[str] = []
        if self._installed or not self._cleanup_done:
            errors.append("runtime gate is still installed")
        if self.inventory is None:
            errors.append("exact target inventory was not established")
            expected_keys: set[str] = set()
        else:
            expected_keys = {
                f"{branch}:{name}"
                for branch in PATCHED_TARGET_KINDS
                for name, _module in self.inventory.targets[branch]
            }
        actual_keys = set(self.call_counts)
        if actual_keys != expected_keys:
            errors.append(
                f"hook-call key mismatch; missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        counts = [
            self.call_counts[key] for key in sorted(expected_keys & actual_keys)
        ]
        if not counts or any(count <= 0 for count in counts):
            errors.append("every patched target must be called at least once")
        if counts and len(set(counts)) != 1:
            errors.append(
                f"patched target call counts are unequal: {sorted(set(counts))}"
            )
        if counts and any(count != self.expected_calls_per_target for count in counts):
            errors.append(
                f"patched target calls must equal configured "
                f"{self.expected_calls_per_target}; observed={sorted(set(counts))}"
            )
        if set(self.validated_layouts) != expected_keys:
            errors.append(
                "not every patched target completed live tensor-layout validation"
            )
        if self._camera_capture_count != 1:
            errors.append(
                f"generation-camera metadata capture count "
                f"{self._camera_capture_count} != exactly 1"
            )
        if self.camera_audit is None or self.camera_audit.get("status") != "OK":
            errors.append("generation-camera audit did not complete successfully")
        if self.reference_keep_mask is None:
            errors.append("reference keep mask was not established")
        if not self.restoration_status.get("fully_restored"):
            errors.append("runtime hooks/wrapper/processors were not fully restored")
        self.completion_validation = {
            "status": "OK" if not errors else "FAIL_CLOSED",
            "expected_calls_per_target": self.expected_calls_per_target,
            "expected_patched_target_count": len(expected_keys),
            "camera_capture_count": self._camera_capture_count,
            "all_hook_call_counts_equal": bool(counts and len(set(counts)) == 1),
            "all_layouts_validated": set(self.validated_layouts) == expected_keys,
            "fully_restored": bool(self.restoration_status.get("fully_restored")),
            "errors": errors,
        }
        if errors:
            raise RuntimeGateError(
                "Gating completion validation failed: " + "; ".join(errors)
            )
        return dict(self.completion_validation)

    def audit_record(self) -> dict[str, Any]:
        integrity = self._target_integrity(
            include_hook_registries=not self._installed
        )
        return {
            "policy": self.policy,
            "runtime_framework_installed": self._ever_installed,
            "runtime_framework_active": self._installed,
            "reference_keep_mask": self.reference_keep_mask,
            "camera_capture_count": self._camera_capture_count,
            "camera_audit": self.camera_audit,
            "target_inventory": (
                self.inventory.serializable() if self.inventory else None
            ),
            "patched_target_counts": {
                "attn_refview": self.expected_target_count if self.inventory else 0,
                "attn_dino": self.expected_target_count if self.inventory else 0,
                "attn_multiview": 0,
            },
            "attn_multiview_patched": not bool(integrity["multiview_untouched"]),
            "sampled_residual_norms": list(self.observations),
            "hook_call_counts": dict(sorted(self.call_counts.items())),
            "validated_layouts": dict(sorted(self.validated_layouts.items())),
            "completion_validation": self.completion_validation,
            "restoration_status": dict(self.restoration_status),
            "target_unet_sampled_parameter_sentinel_changed_during_variant": None,
            "parameter_integrity_assessed_by_runner": False,
            "processors_replaced_by_gate": not bool(integrity["processors_match"]),
            "live_integrity": integrity,
        }

    def __enter__(self) -> "ViewSelectiveConditioningGate":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.uninstall()
        return False
