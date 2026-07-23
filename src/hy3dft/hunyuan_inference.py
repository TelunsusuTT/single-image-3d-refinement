"""Shared project-local helpers for official Hunyuan3D-Paint inference.

This module is intentionally standard-library-only at import time. Official
Hunyuan modules are imported only when a caller explicitly initializes a real
pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_HY21 = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1")


def resolve_official_paths() -> tuple[Path, Path]:
    """Resolve and validate the existing read-only official source paths."""

    hy21_text = os.environ.get("HY21", "")
    hy21 = Path(hy21_text).expanduser() if hy21_text else DEFAULT_HY21
    hy21 = hy21.resolve()

    hypaint_text = os.environ.get("HYPAINT", "")
    hypaint = Path(hypaint_text).expanduser() if hypaint_text else hy21 / "hy3dpaint"
    hypaint = hypaint.resolve()

    if not hy21.is_dir():
        raise RuntimeError(f"HY21 path missing: {hy21}")
    if not hypaint.is_dir():
        raise RuntimeError(f"HYPAINT path missing: {hypaint}")
    return hy21, hypaint


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def set_absolute_official_config_paths(conf: Any, hypaint: Path) -> dict[str, str]:
    """Remove cwd ambiguity from official config and RealESRGAN paths."""

    cfg_path = (hypaint / "cfgs" / "hunyuan-paint-pbr.yaml").resolve()
    realesrgan_path = (hypaint / "ckpt" / "RealESRGAN_x4plus.pth").resolve()
    conf.multiview_cfg_path = str(cfg_path)
    conf.realesrgan_ckpt_path = str(realesrgan_path)
    return {
        "multiview_cfg_path": str(cfg_path),
        "realesrgan_ckpt_path": str(realesrgan_path),
    }


def initialize_base_paint_pipeline(
    *,
    max_num_view: int,
    resolution: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Load one fresh official true-PBR inference pipeline.

    The returned metadata is JSON serializable and records the exact official
    paths used by the historical corrected-input inference wrapper.
    """

    hy21, hypaint = resolve_official_paths()
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.device = device
    config_paths = set_absolute_official_config_paths(conf, hypaint)
    pipeline = Hunyuan3DPaintPipeline(conf)
    return pipeline, {
        "resolved_hy21": str(hy21),
        "resolved_hypaint": str(hypaint),
        **config_paths,
        "max_num_view": int(max_num_view),
        "resolution": int(resolution),
        "device": str(device),
        "official_pipeline_class": "textureGenPipeline.Hunyuan3DPaintPipeline",
        "fresh_official_true_pbr_base": True,
    }


def run_paint_inference(
    paint_pipeline: Any,
    *,
    mesh_path: str | Path,
    image_path: str | Path,
    output_mesh_path: str | Path,
    use_remesh: bool,
    save_glb: bool = True,
) -> Any:
    """Call the existing official pipeline without changing its algorithm."""

    return paint_pipeline(
        mesh_path=str(Path(mesh_path).expanduser().resolve()),
        image_path=str(Path(image_path).expanduser().resolve()),
        output_mesh_path=str(Path(output_mesh_path).expanduser().resolve()),
        use_remesh=bool(use_remesh),
        save_glb=bool(save_glb),
    )


def run_fixed_mesh_inference(
    paint_pipeline: Any,
    *,
    mesh_path: str | Path,
    image_path: str | Path,
    output_mesh_path: str | Path,
) -> Any:
    """Run the established corrected-input, no-remesh inference behavior."""

    return run_paint_inference(
        paint_pipeline,
        mesh_path=mesh_path,
        image_path=image_path,
        output_mesh_path=output_mesh_path,
        use_remesh=False,
        save_glb=True,
    )
