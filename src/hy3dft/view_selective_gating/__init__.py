"""View-selective conditioning gating for Hunyuan3D Paint inference."""

from .camera_gate import (
    CameraMetadataError,
    all_one_keep_mask,
    build_reference_keep_mask,
)
from .runtime_gate import (
    RuntimeGateError,
    ViewSelectiveConditioningGate,
    discover_attention_targets,
)

__all__ = [
    "CameraMetadataError",
    "RuntimeGateError",
    "ViewSelectiveConditioningGate",
    "all_one_keep_mask",
    "build_reference_keep_mask",
    "discover_attention_targets",
]
