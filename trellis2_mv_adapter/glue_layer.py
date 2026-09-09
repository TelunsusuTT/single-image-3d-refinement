#!/usr/bin/env python3
"""Prepare meshes and reference images for MV-Adapter."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load and merge triangle meshes from a scene."""
    scene = trimesh.load(path, force="scene")
    geoms = [g for g in scene.geometry.values()
             if isinstance(g, trimesh.Trimesh)]
    if not geoms:
        raise ValueError(f"{path}: no triangle mesh found")
    if len(geoms) == 1:
        return geoms[0]
    combined = trimesh.util.concatenate(geoms)
    combined.remove_unreferenced_vertices()
    return combined


def save_mesh(mesh: trimesh.Trimesh, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mesh.export(path)


def _largest_component(mesh: trimesh.Trimesh, keep_ratio: float = 0.001) -> trimesh.Trimesh:
    """Keep the largest component and components above its keep_ratio threshold.

    Use vertex connectivity to avoid mesh.split() memory spikes."""
    import trimesh.graph as tg
    comps = tg.connected_components(mesh.edges_unique, min_len=1)
    if len(comps) <= 1:
        return mesh
    comps = sorted(comps, key=len, reverse=True)
    keep = list(comps[0])
    for c in comps[1:]:
        if len(c) > len(comps[0]) * keep_ratio:
            keep.extend(c)
    vmask = np.zeros(len(mesh.vertices), dtype=bool)
    vmask[keep] = True
    fmask = vmask[mesh.faces].all(axis=1)
    out = mesh.submesh([np.nonzero(fmask)[0]])[0]
    out.remove_unreferenced_vertices()
    return out


def clean_and_seal(mesh: trimesh.Trimesh,
                   keep_ratio: float = 0.001,
                   max_hole_edges: int = 32) -> trimesh.Trimesh:
    """Clean a mesh copy and attempt hole filling."""
    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(m)
    except Exception:
        pass
    m = _largest_component(m, keep_ratio=keep_ratio)
    try:
        if not m.is_watertight:
            m.fill_holes()
            m.remove_unreferenced_vertices()
    except Exception:
        pass
    m.fix_normals()
    return m


def normalize_mesh(mesh: trimesh.Trimesh,
                   up: str = "+Y",
                   front: str = "+Z",
                   scale_to: float = 1.0,
                   center: str = "bbox") -> trimesh.Trimesh:
    """Centre, scale, and heuristically orient a mesh copy; default to Y-up and +Z front."""
    AXIS = {"+X": 0, "-X": 0, "+Y": 1, "-Y": 1, "+Z": 2, "-Z": 2}
    SIGN = {"+": 1, "-": -1}

    m = mesh.copy()
    verts = m.vertices.astype(np.float64)

    if center == "bbox":
        c = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    else:
        c = verts.mean(axis=0)
    verts = verts - c

    extent = verts.max(axis=0) - verts.min(axis=0)
    scale = scale_to / float(extent.max())
    verts = verts * scale

    # Heuristic: align the longest axis with up.
    longest = int(np.argmax(verts.max(axis=0) - verts.min(axis=0)))
    target = AXIS[up]
    if longest != target:
        order = [i for i in range(3) if i != longest]
        if target == 0:
            perm = [longest, *order]
        elif target == 1:
            perm = [order[0], longest, order[1]]
        else:
            perm = [*order, longest]
        verts = verts[:, perm]
        m.vertices = verts
        # Preserve handedness after the axis permutation.
        if np.linalg.det(np.eye(3)[:, perm]) < 0:
            m.vertices = np.array(m.vertices)
            m.vertices[:, 0] = -m.vertices[:, 0]

    # Infer front from the mean vertex coordinate.
    f_axis = AXIS[front]
    f_sign = SIGN[front[0]]
    verts = m.vertices
    mean_along = verts[:, f_axis].mean()
    if mean_along < 0 and f_sign == 1 or mean_along > 0 and f_sign == -1:
        verts[:, f_axis] = -verts[:, f_axis]
    m.vertices = verts
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m


def has_uv(mesh: trimesh.Trimesh) -> bool:
    try:
        uv = getattr(mesh.visual, "uv", None)
        return uv is not None and len(uv) > 0
    except Exception:
        return False


def ensure_uv(mesh: trimesh.Trimesh,
              unwrap_resolution: int = 4096) -> trimesh.Trimesh:
    """Keep existing UVs; otherwise unwrap with xatlas."""
    if has_uv(mesh):
        return mesh
    import xatlas
    vmapping, indices, uvs = xatlas.parametrize(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.uint32),
    )
    out = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices.astype(np.int64),
        process=False,
    )
    out.remove_unreferenced_vertices()
    from trimesh.visual import TextureVisuals
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=None, metallicFactor=0.0, roughnessFactor=1.0)
    out.visual = TextureVisuals(uv=uvs.astype(np.float64), material=mat)
    return out


def export_geometry_only(mesh: trimesh.Trimesh, path: str) -> None:
    """Export geometry and UVs with a neutral material."""
    keep_uv = has_uv(mesh)
    verts = np.ascontiguousarray(mesh.vertices)
    faces = np.ascontiguousarray(mesh.faces)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.remove_unreferenced_vertices()
    out.fix_normals()
    if keep_uv:
        from trimesh.visual import TextureVisuals
        mat = trimesh.visual.material.PBRMaterial(
            metallicFactor=0.0, roughnessFactor=1.0)
        out.visual = TextureVisuals(uv=mesh.visual.uv, material=mat)
    save_mesh(out, path)


def matte_image(src: str, dst: str, device: str = "cuda") -> str:
    """Extract an RGBA foreground with BiRefNet; copy the input if imports fail."""
    import shutil
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
    except ImportError:
        shutil.copy(src, dst)
        print("[matte] Missing torch/transformers; copied input without background removal")
        return dst

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True)
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    from PIL import Image
    img = Image.open(src).convert("RGB")
    inp = tf(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = model(inp)[-1].sigmoid().squeeze(0)
    alpha = pred[0].cpu().numpy()
    import numpy as np
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    alpha = Image.fromarray(alpha).resize(img.size, Image.LANCZOS)
    out = img.convert("RGBA")
    out.putalpha(alpha)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    out.save(dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Mesh and image preprocessing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("clean", help="Clean mesh and fill holes")
    p.add_argument("src"); p.add_argument("dst")

    p = sub.add_parser("norm", help="Centre, scale, and orient to Y-up")
    p.add_argument("src"); p.add_argument("dst")

    p = sub.add_parser("uv", help="Check UVs and unwrap with xatlas")
    p.add_argument("src"); p.add_argument("dst")

    p = sub.add_parser("matte", help="Remove reference image background")
    p.add_argument("src"); p.add_argument("dst")
    p.add_argument("--device", default="cuda")

    args = ap.parse_args()

    if args.cmd == "matte":
        matte_image(args.src, args.dst, device=args.device)
        print(f"[matte] Output: {args.dst}")
        return 0

    mesh = load_mesh(args.src)
    print(f"[load] {args.src}: V={len(mesh.vertices)} F={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} uv={has_uv(mesh)}")
    if args.cmd == "clean":
        mesh = clean_and_seal(mesh)
    elif args.cmd == "norm":
        mesh = normalize_mesh(mesh)
    elif args.cmd == "uv":
        mesh = ensure_uv(mesh)
    export_geometry_only(mesh, args.dst)
    print(f"[save] {args.dst}: V={len(mesh.vertices)} F={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} uv={has_uv(mesh)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
