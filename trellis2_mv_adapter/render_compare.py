#!/usr/bin/env python3
"""Render each GLB as a 2x3 six-view image on a light grey background.

Usage: python render_compare.py <out_dir> <glb1> <glb2> ..."""
import sys, os
sys.path.insert(0, "/root/autodl-tmp/MV-Adapter")
import numpy as np
import torch
from PIL import Image
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper, get_orthogonal_camera, load_mesh, render,
)

out_dir, glbs = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)
ctx = NVDiffRastContextWrapper(device="cuda")

angles = [(25, 18), (115, 18), (200, 15), (290, 20), (60, 55), (240, -50)]

def rnd(mesh, cams, bg=0.78):
    with torch.no_grad():
        out = render(ctx, mesh, cams, height=512, width=512,
                     render_attr=True, render_depth=False, render_normal=False,
                     attr_background=bg)
    return [(t * 255).clamp(0, 255).round().cpu().numpy().astype(np.uint8) for t in out.attr]

print(f"[render_compare] {len(glbs)} models")
for glb in glbs:
    stem = os.path.splitext(os.path.basename(glb))[0]
    try:
        mesh = load_mesh(glb, rescale=True, front_x_to_y=False, device="cuda")
        tiles = []
        for az, el in angles:
            c = get_orthogonal_camera(
                elevation_deg=[el], distance=[1.8],
                left=-0.55, right=0.55, bottom=-0.55, top=0.55,
                azimuth_deg=[az], device="cuda")
            tiles.append(rnd(mesh, c)[0])
        top = np.concatenate(tiles[0:3], axis=1)
        bot = np.concatenate(tiles[3:6], axis=1)
        Image.fromarray(np.concatenate([top, bot], axis=0)).save(
            os.path.join(out_dir, f"{stem}_show.png"))
        print(f"[render_compare] OK {stem}", flush=True)
        del mesh
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[render_compare] FAIL {stem}: {e}", flush=True)
print("[render_compare] DONE")
