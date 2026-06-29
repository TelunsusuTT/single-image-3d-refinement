# Phase 1C Asset Acquisition

Phase 1C prepares one real asset for local acquisition and lightweight file
verification. The first target is a single asset. After that path is understood,
the same process can expand to a maximum of three assets.

Phase 1C does not convert assets into Hunyuan3D-Paint training format. It also
does not run Hunyuan, run Blender, submit Slurm jobs, or download large
datasets. Blender import, UV inspection, and material inspection belong to Phase
1D.

## Recommended Source

Use ABO first.

ABO is preferred because it is product-oriented: many assets are real object
models rather than broad scenes or characters. Its 3D assets are distributed as
glTF 2.0 compatible models, including GLB files, and are intended to carry PBR
material and texture information. That makes ABO a good first match for the
texture-heavy packaging, labeled-container, and product-box cases from Phase 1B.

Objaverse or synthetic packaging can be used later as a fallback if ABO cannot
provide a small, reliable, texture-heavy asset with acceptable metadata and
licensing for the experiment.

## Storage Rule

Never download `abo-3dmodels.tar` or other large archives in Phase 1C.

Only small metadata needed for selection and one selected GLB should be
downloaded manually. The project tools in this phase only build a download
manifest and check that the local file exists with a plausible extension and
size. They do not run `aws`, download data, inspect mesh contents, or validate
materials.

The intended flow is:

1. Pick one real ABO asset from metadata.
2. Record it in `data/candidates/phase1c_selected_assets.csv`.
3. Generate a small download manifest.
4. Manually download exactly that GLB.
5. Run the local GLB file checker.

Only after this succeeds should Phase 1D attempt Blender import and deeper
geometry, UV, material, and texture inspection.
