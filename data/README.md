# Data manifests

Only portable identifiers and split metadata are versioned here. Source ABO
meshes, prepared Hunyuan3D-Paint examples, conditioning images, and generated
renders remain outside Git.

- `manifests/frame_panels/split.json` is the authoritative 80/10/11 split.
- `manifests/frame_panels/assets.csv` records the corresponding ABO-relative
  asset path and lightweight geometry metadata.
- `manifests/evaluation_cases.example.json` documents the input contract used
  by the fixed-view evaluation scripts.

Runtime paths should be supplied through local configuration. No manifest in
this directory depends on a username or machine-specific absolute path.

The ready Paint inference backend expects each case to contain
`input/mesh.glb` and a preselected `input/image.png`. Canonical runners pass
`--conditioning-view 005` so that this selection is recorded in the run plan; the
backend does not infer camera identity from image pixels. Dataset preparation must
therefore ensure that `input/image.png` is the declared conditioning view.
