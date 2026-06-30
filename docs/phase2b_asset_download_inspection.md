# Phase 2B Asset Download And Inspection

Phase 2B prepares the manually selected Phase 2A ABO candidates for technical
inspection. It downloads only selected GLBs when the user runs the download
script, then summarizes Blender inspection reports after the user runs Blender
manually.

Input selected CSV:

```text
data/candidates/phase2a_abo_selected_assets.csv
```

Raw asset output directory:

```text
data/raw_assets/phase2b_abo_selected/
```

Selected candidates are not training data yet. They must pass local file checks
and Blender inspection before Phase 2C renders Hunyuan3D-Paint-style training
examples.

## Technical Criteria

Each asset should satisfy:

- Blender import status is `OK`
- `mesh_object_count > 0`
- `material_count > 0`
- `texture_image_count > 0`
- at least one UV layer exists
- polygon count is reasonable for local rendering and smoke training

## Gate Before Phase 2C

Assets that fail import, lack meshes, lack UVs, lack materials, lack texture
images, or are obviously too heavy should be rejected or held for review before
Phase 2C. Only technically usable assets should move into rendering and official
Hunyuan-style checker runs.
