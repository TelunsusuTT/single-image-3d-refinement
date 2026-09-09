#!/usr/bin/env python3
"""Generate TRELLIS.2 geometry and prepare it for MV-Adapter."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
sys.path.insert(0, "/root/autodl-tmp/TRELLIS.2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Input image")
    ap.add_argument("output", help="Output geometry GLB after preprocessing")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/weights/TRELLIS.2-4B",
                    help="TRELLIS.2-4B checkpoint directory")
    ap.add_argument("--resolution", default="512",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decimation_target", type=int, default=400000,
                    help="Target face count for to_glb before MV-Adapter")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    print(f"[stage1] Loading pipeline: {args.ckpt}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt)
    pipeline.cuda()

    # Preserve alpha so RGBA inputs bypass rembg.
    image = Image.open(args.image)
    print(f"[stage1] Generating {args.resolution} shape, seed={args.seed}")
    mesh = pipeline.run(
        image, seed=args.seed, pipeline_type=args.resolution
    )[0]

    # Respect the nvdiffrast face limit.
    if len(mesh.faces) > 16_000_000:
        mesh.simplify(16_777_216)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    raw = args.output + ".raw.glb"
    glb.export(raw)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glue_layer import (clean_and_seal, export_geometry_only,
                            ensure_uv, load_mesh, normalize_mesh)
    m = load_mesh(raw)
    m = clean_and_seal(m)
    m = normalize_mesh(m)
    m = ensure_uv(m)
    export_geometry_only(m, args.output)
    print(f"[stage1] Done: {args.output}")
    print(f"         V={len(m.vertices)} F={len(m.faces)} watertight={m.is_watertight}")
    os.remove(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
