#!/usr/bin/env python3
"""Mesh post-processing and runtime patches for MV-Adapter."""
from __future__ import annotations

import os
import time
import numpy as np


def seal_loops(mesh):
    """Seal boundary loops with triangle fans."""
    import trimesh
    bd = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(bd) == 0:
        return mesh
    bd_e = mesh.edges_sorted[bd]
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in bd_e:
        adj[a].append(b)
        adj[b].append(a)
    seen = set()
    loops = []
    for v0 in list(adj):
        if v0 in seen:
            continue
        seen.add(v0)
        loop = [v0]
        cur = v0
        while True:
            nxts = [x for x in adj[cur] if x not in seen]
            if not nxts:
                if len(adj[cur]) == 1 and adj[cur][0] == loop[0]:
                    break
                break
            nxt = nxts[0]
            seen.add(nxt)
            loop.append(nxt)
            cur = nxt
            if nxt == v0:
                break
        if len(loop) >= 3:
            loops.append(loop if loop[0] != loop[-1] else loop[:-1])
    if not loops:
        return mesh
    vs = np.asarray(mesh.vertices)
    extra = np.array([vs[l].mean(0) for l in loops])
    newf = [[l[i], l[(i + 1) % len(l)], len(vs) + k]
            for k, l in enumerate(loops) for i in range(len(l))]
    out = trimesh.Trimesh(vertices=np.vstack([vs, extra]),
                          faces=np.vstack([np.asarray(mesh.faces), np.array(newf)]),
                          process=False)
    out.remove_unreferenced_vertices()
    out.fix_normals()
    return out


def watertight_voxel_mesh(mesh_path, out_path, grid=256, dilate=1, decimate_to=200000):
    """Voxelise, dilate, fill, and extract a surface; seal loops after decimation."""
    import trimesh
    from trimesh.voxel.creation import voxelize
    from skimage.measure import marching_cubes
    from scipy.ndimage import binary_fill_holes, binary_dilation

    t0 = time.time()
    m = trimesh.load(mesh_path, force="mesh", process=False)
    extent = float(np.max(m.bounds[1] - m.bounds[0]))
    pitch = extent / grid
    vm = voxelize(m, pitch=pitch)
    filled = binary_fill_holes(binary_dilation(vm.matrix, iterations=dilate))
    verts, faces, _, _ = marching_cubes(filled, level=0.5)
    world = (vm.transform[:3, :3] @ verts.T).T + vm.transform[:3, 3]
    wt = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    wt.update_faces(wt.unique_faces())
    wt.remove_unreferenced_vertices()
    wt.fix_normals()
    wt = seal_loops(wt)
    print(f"[mesh_methods] Voxel remesh: grid={grid} sealed F={len(wt.faces)} "
          f"watertight={wt.is_watertight} ({(time.time()-t0):.0f}s)")
    if not wt.is_watertight:
        raise RuntimeError(f"Voxel mesh is not watertight: grid={grid} (F={len(wt.faces)})")
    if decimate_to and len(wt.faces) > decimate_to:
        import open3d as o3d
        o = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(wt.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(wt.faces)))
        o = o.simplify_quadric_decimation(decimate_to)
        wt = trimesh.Trimesh(vertices=np.asarray(o.vertices),
                             faces=np.asarray(o.triangles), process=False)
        wt.fix_normals()
        if not wt.is_watertight:
            wt = seal_loops(wt)
        print(f"[mesh_methods] Decimated to {len(wt.faces)} watertight={wt.is_watertight} "
              f"({time.time()-t0:.0f}s)")
    wt.export(out_path)
    print(f"[mesh_methods] Output: {out_path}")
    return out_path


def _smooth_normals_np(n, faces, iters):
    """Average vertex normals over one-ring neighbours."""
    from scipy import sparse
    f = np.asarray(faces)
    row = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]])
    col = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    adj = sparse.csr_matrix((np.ones(len(row)), (row, col)),
                            shape=(len(n),) * 2)
    A = (adj + sparse.eye(len(n))).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    W = sparse.diags(1.0 / deg) @ A
    n = np.asarray(n, dtype=np.float32).copy()
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    for _ in range(iters):
        n = W @ n
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def patch_load_mesh_smooth_normals(iters: int):
    """Smooth normals in memory without moving vertices.

    Call before importing MV-Adapter inference or texture modules."""
    import mvadapter.utils.mesh_utils as mu

    if getattr(mu, "_p3_smooth_patched", False):
        return
    it = int(iters)
    orig = mu.load_mesh

    def patched(mesh_path, **kwargs):
        import torch
        tm = orig(mesh_path, **kwargs)
        n = getattr(tm, "_v_nrm", None)
        if n is not None and getattr(tm, "t_pos_idx", None) is not None:
            n_smooth = _smooth_normals_np(
                n.detach().cpu().numpy(), tm.t_pos_idx.cpu().numpy(), it)
            tm.set_vertex_normal(
                torch.from_numpy(n_smooth.astype(np.float32)).to(tm.v_pos.device))
        return tm

    mu.load_mesh = patched
    mu._p3_smooth_patched = True
    print(f"[mesh_methods] B1: load_mesh normal smoothing iters={it} enabled")


def patch_process_mesh_decimate_target(targetfacenum: int):
    """Override the face budget used by process_raw at runtime."""
    import mvadapter.utils.mesh_utils.mesh_process as mp

    if getattr(mp, "_p3_patched", False):
        return
    target = int(targetfacenum)
    orig = mp.process_mesh

    def patched(vertices, faces, **kwargs):
        kwargs["targetfacenum"] = target
        return orig(vertices, faces, **kwargs)

    mp.process_mesh = patched
    mp._p3_patched = True
    print(f"[mesh_methods] A2: process_raw face target 50000 -> {target}")
