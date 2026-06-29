# Phase 1D Asset Inspection

Phase 1D checks whether the one downloaded GLB can be imported by Blender and
whether it has the basic ingredients needed for later rendering and conversion:
mesh geometry, UVs, materials, and preferably texture images.

This phase does not convert the asset into Hunyuan3D-Paint training format. It
does not run Hunyuan, submit Slurm jobs, or train a model. Blender import is a
prerequisite for Phase 1E rendering and conversion, where the asset may later be
turned into the `render_cond/` and `render_tex/` structure checked in Phase 1A.

## What The Inspection Checks

The Blender inspection script imports the GLB in a clean scene and records:

- import status and fatal import errors, if any
- object and mesh counts
- total vertices, polygons, and an estimated triangle count
- world-space bounding box
- per-object type, vertex count, polygon count, UV layer count, and material
  slot count
- per-material node usage and detected image texture names
- warning flags for missing mesh, UVs, materials, or texture images

It does not render images, modify the GLB, or inspect the asset with external
packages.

## Pass / Fail Criteria

A technical pass requires:

- `mesh_count > 0`
- `total_faces > 0`
- at least one UV layer
- at least one material
- no fatal import errors

Texture images are preferred. A report with `texture_image_count == 0` should be
treated as a warning rather than an automatic failure, because some assets can
still be useful for an import smoke test.

If the first ABO asset is weak or not especially texture-heavy, it can still
pass as a technical smoke asset. In that case, keep the inspection report and
select a stronger texture-heavy asset for the next content-quality pass.
