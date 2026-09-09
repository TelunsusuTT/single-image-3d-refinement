"""Shared project-local helpers for official Hunyuan3D-Paint inference.

This module is intentionally standard-library-only at import time. Official
Hunyuan modules are imported only when a caller explicitly initializes a real
pipeline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBJ_TO_GLB_EXPORTER = PROJECT_ROOT / "scripts/export_obj_to_glb_blender.py"


def resolve_official_paths() -> tuple[Path, Path]:
    """Resolve and validate the existing read-only official source paths."""

    root_text = os.environ.get("HUNYUAN3D_ROOT", "").strip()
    if not root_text:
        raise RuntimeError(
            "HUNYUAN3D_ROOT must point to an external Hunyuan3D 2.1 checkout"
        )
    hunyuan_root = Path(root_text).expanduser().resolve()

    paint_text = os.environ.get("HUNYUAN3D_PAINT_SOURCE_ROOT", "").strip()
    paint_root = (
        Path(paint_text).expanduser().resolve()
        if paint_text
        else (hunyuan_root / "hy3dpaint").resolve()
    )

    if not hunyuan_root.is_dir():
        raise RuntimeError(f"Hunyuan3D source path missing: {hunyuan_root}")
    if not paint_root.is_dir():
        raise RuntimeError(f"Hunyuan3D-Paint source path missing: {paint_root}")
    return hunyuan_root, paint_root


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def set_absolute_official_config_paths(conf: Any, paint_root: Path) -> dict[str, str]:
    """Remove cwd ambiguity from official config and RealESRGAN paths."""

    cfg_path = (paint_root / "cfgs" / "hunyuan-paint-pbr.yaml").resolve()
    realesrgan_path = (paint_root / "ckpt" / "RealESRGAN_x4plus.pth").resolve()
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
    paths used by the corrected-conditioning inference protocol.
    """

    hunyuan_root, paint_root = resolve_official_paths()
    prepend_pythonpath(hunyuan_root)
    prepend_pythonpath(paint_root)

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.device = device
    config_paths = set_absolute_official_config_paths(conf, paint_root)
    pipeline = Hunyuan3DPaintPipeline(conf)
    return pipeline, {
        "resolved_hunyuan3d_root": str(hunyuan_root),
        "resolved_hunyuan3d_paint_root": str(paint_root),
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
    """Run the established corrected-conditioning, no-remesh inference behavior."""

    return run_paint_inference(
        paint_pipeline,
        mesh_path=mesh_path,
        image_path=image_path,
        output_mesh_path=output_mesh_path,
        use_remesh=False,
        save_glb=True,
    )


def run_isolated_fixed_mesh_inference(
    paint_pipeline: Any,
    *,
    mesh_path: str | Path,
    image_path: str | Path,
    output_mesh_path: str | Path,
) -> Any:
    """Run fixed-mesh inference and export GLB in an isolated Blender process."""

    output_obj = Path(output_mesh_path).expanduser().resolve()
    output_glb = output_obj.with_suffix(".glb")
    if output_glb.exists():
        raise FileExistsError(f"Refusing to overwrite GLB: {output_glb}")
    result = run_paint_inference(
        paint_pipeline,
        mesh_path=mesh_path,
        image_path=image_path,
        output_mesh_path=output_obj,
        use_remesh=False,
        save_glb=False,
    )
    if not output_obj.is_file() or output_obj.stat().st_size <= 0:
        raise RuntimeError(f"Fixed-mesh inference did not create its OBJ: {output_obj}")
    blender_name = os.environ.get("BLENDER_BIN", "blender").strip()
    blender = shutil.which(blender_name)
    if blender is None:
        raise RuntimeError(
            f"Isolated Blender executable is unavailable: {blender_name}"
        )
    if not OBJ_TO_GLB_EXPORTER.is_file():
        raise RuntimeError(
            f"Isolated OBJ-to-GLB exporter is unavailable: {OBJ_TO_GLB_EXPORTER}"
        )
    subprocess.run(
        [
            str(blender),
            "-b",
            "--python",
            str(OBJ_TO_GLB_EXPORTER),
            "--",
            "--input-obj",
            str(output_obj),
            "--output-glb",
            str(output_glb),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not output_glb.is_file() or output_glb.stat().st_size <= 0:
        raise RuntimeError(
            f"Isolated OBJ-to-GLB export did not create GLB: {output_glb}"
        )
    return result
