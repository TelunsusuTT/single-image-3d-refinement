#!/usr/bin/env python3
"""Run TRELLIS.2 shape generation and MV-Adapter texturing in separate environments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

# Separate environments isolate incompatible CUDA dependencies.
HERE = os.path.dirname(os.path.abspath(__file__))
T1_PY = "/root/miniconda3/envs/trellis2/bin/python"
T2_PY = "/root/miniconda3/envs/mvadapter/bin/python"

ENV_T1 = {
    "OPENCV_IO_ENABLE_OPENEXR": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_XET": "1",
    "CUDA_HOME": "/root/miniconda3/envs/trellis2",
    "ATTN_BACKEND": "xformers",
    "PYTHONPATH": "/root/autodl-tmp/TRELLIS.2",
}
ENV_T2 = {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_XET": "1",
    "CUDA_HOME": "/root/miniconda3/envs/trellis2",
    "PATH": "/root/miniconda3/envs/mvadapter/bin:" + os.environ.get("PATH", ""),
}


def run_stage(cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    full_env = dict(os.environ)
    full_env.update(env)
    r = subprocess.run(cmd, cwd=HERE, env=full_env)
    if r.returncode != 0:
        sys.exit(f"Stage failed: {r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Image-to-3D: TRELLIS.2 shape + MV-Adapter texture")
    ap.add_argument("image", help="Input image")
    ap.add_argument("output", help="Output PBR GLB")
    ap.add_argument("--resolution", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="sdxl", choices=["sdxl", "sd21"])
    ap.add_argument("--remove_bg", action="store_true")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--normalize_views", action="store_true",
                    help="Normalise stage2 view foreground brightness before baking")
    ap.add_argument("--skip_shape", action="store_true", help="Reuse existing geometry for texturing")
    ap.add_argument("--decimate_target", type=int, default=0,
                    help="A2: override the process_raw face target (normally 50000)")
    ap.add_argument("--watertight", action="store_true", help="A3: watertight voxel remeshing")
    ap.add_argument("--wt_grid", type=int, default=256, help="A3 voxel grid resolution")
    ap.add_argument("--wt_decimate", type=int, default=200000, help="A3 target face count")
    ap.add_argument("--seal_grid", type=int, default=128, help="B2 voxel grid resolution")
    ap.add_argument("--smooth_normals", type=int, default=0, help="B1: normal smoothing iterations")
    ap.add_argument("--strong_seal", action="store_true", help="B2: coarser voxel remeshing")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    base = os.path.splitext(os.path.basename(args.output))[0]
    mesh_glb = os.path.join(args.workdir, f"{base}_shape.glb")
    mv_dir = os.path.join(args.workdir, "mvadapter_out")

    if not args.skip_shape:
        run_stage([T1_PY, os.path.join(HERE, "stage1_shape.py"),
                   args.image, mesh_glb,
                   "--resolution", args.resolution, "--seed", str(args.seed)],
                  ENV_T1)
    else:
        assert os.path.exists(mesh_glb), f"{mesh_glb} does not exist; run the shape stage first"

    cmd = [T2_PY, os.path.join(HERE, "stage2_texture.py"),
           mesh_glb, args.image,
           "--save_dir", mv_dir, "--save_name", base,
           "--variant", args.variant, "--seed", str(args.seed)]
    if args.remove_bg:
        cmd.append("--remove_bg")
    if args.normalize_views:
        cmd.append("--normalize_views")
    for flag, val in [("--decimate_target", args.decimate_target),
                      ("--wt_grid", args.wt_grid),
                      ("--wt_decimate", args.wt_decimate),
                      ("--seal_grid", args.seal_grid),
                      ("--smooth_normals", args.smooth_normals)]:
        if val:
            cmd.extend([flag, str(val)])
    if args.watertight:
        cmd.append("--watertight")
    if args.strong_seal:
        cmd.append("--strong_seal")
    run_stage(cmd, ENV_T2)

    shaded = os.path.join(mv_dir, f"{base}_shaded.glb")
    if os.path.exists(shaded):
        shutil.copy(shaded, args.output)
        print(f"\nDone: {args.output}")
    else:
        sys.exit(f"Output not found: {shaded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
