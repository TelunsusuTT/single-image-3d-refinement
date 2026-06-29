# Phase 1A Data Format

Phase 1A documents and validates the lightweight Hunyuan3D-Paint training
example format used by this project. The goal is to make packaging errors easy
to find before any longer-running training or data conversion work begins.

This phase does not train a model, run Hunyuan, run Blender, submit Slurm jobs,
or download datasets.

## Expected Structure

The training split is represented by an `examples.json` file. It should contain
a JSON list of sample directory paths:

```json
[
  "/absolute/path/to/sample_000001",
  "/absolute/path/to/sample_000002"
]
```

Each listed sample directory is expected to contain:

```text
sample_000001/
  render_cond/
  render_tex/
  transforms.json              # optional unless strict validation is used
```

`transforms.json` may also appear under `render_tex/`.

`render_cond/` contains condition, reference, or lighting images used by the
training pipeline as input context.

`render_tex/` contains the texture-space supervision images. Phase 1A checks
for these map families:

- albedo maps matching `*_albedo.*`
- metallic-roughness maps matching `*_mr.*` or `*_metallic_roughness.*`
- normal maps matching `*_normal.*`
- position maps matching `*_pos.*` or `*_position.*`

Supported image extensions are `.png`, `.jpg`, `.jpeg`, and `.webp`.

## Path Convention

Project-local `examples.json` files should use absolute paths. Absolute paths
make the split unambiguous when commands are launched from different working
directories, from Slurm scripts, or from notebooks and ad hoc shells.

The file itself still lives in this project, for example:

```text
data/hy3dpaint_train_examples/official_overfit/examples.json
```

Keeping the manifest project-local lets us document and validate exactly which
sample directories are being used without modifying the upstream Hunyuan3D
source tree.

## Future Data

Future ABO, Objaverse, and synthetic packaging work must convert each asset into
the same structure before it is used for training. The validator and summarizer
added in Phase 1A are intended to catch missing directories, missing map types,
unexpected map counts, and absent `transforms.json` files early.
