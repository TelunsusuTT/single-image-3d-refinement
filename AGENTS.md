# Codex Operating Guide

This repository is the project-local workspace for Hunyuan3D-Paint fine-tuning
experiments:

`/vol/bitbucket/ct1022/hy3dpaint_finetune`

## Boundaries

- Work only inside `/vol/bitbucket/ct1022/hy3dpaint_finetune`.
- Never edit `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work`.
- Never inspect or edit `/vol/bitbucket/ct1022/seamtex`.
- Never edit external repositories, model caches, or shared tool installs.
- Never delete `checkpoints/`, `outputs/`, `data/`, `caches/`, logs, or generated
  artifacts unless the user explicitly requests that deletion.

## Runtime Safety

Do not run any of the following unless the user explicitly asks for it in the
current turn:

- Hunyuan training or inference
- Slurm submission
- Blender
- package installs
- checkpoint loading
- dataset downloads
- large file creation

Runtime sbatch files may be created when requested, but do not submit them.

## Default Safe Checks

The following checks are generally allowed when relevant:

- `python -m compileall scripts tests`
- targeted `python -m pytest -q ...`
- static shell checks such as `bash -n env/name.sbatch`
- read-only `find`, `rg`, `sed`, `git status`, and `git diff`
- project-local dry-run scripts that do not import Hunyuan, Blender, torch, or
  load checkpoints

## Reuse First

Before creating a new script:

1. Read `docs/tooling_inventory.md`.
2. Inspect `scripts/` for an existing checker, manifest maker, renderer,
   inference wrapper, comparison tool, or aggregator.
3. Reuse or extend the existing tool when practical.
4. Create a new phase-specific script only when existing tools cannot cleanly
   cover the new workflow.

## Generated Artifacts

Generated outputs, logs, checkpoints, caches, rendered images, downloaded assets,
and temporary reports should not be committed unless the user explicitly asks.

## Script Standards

Every new non-trivial script should:

- expose CLI help through `argparse` or a clear shell usage block
- use deterministic input and output paths
- avoid external dependencies unless the phase explicitly permits them
- import heavy runtime libraries only inside runtime entry points
- include tests where practical
- print a clear success token for sbatch/runtime workflows
