# Safety Rules

## Upstream Boundary

- `/vol/bitbucket/ct1022/Hunyuan3D2.1_Work` is read-only upstream.
- Do not modify upstream source, configs, examples, logs, checkpoints, or generated files.
- Do not write logs/checkpoints/data into the Hunyuan repo.

## Storage Boundary

- Do not store HF cache, pip cache, torch cache in home.
- Use this project's `caches/` directory for Hugging Face, torch, pip, and XDG cache locations.
- Use this project's `tmp/` directory for temporary files.
- Keep downloaded datasets and raw assets out of Git.

## Compute Boundary

- Do not run heavy training on `gpu12`.
- Use Slurm A100 for training smoke tests.
- Do not run Hunyuan, Blender, or training from login or unsuitable interactive nodes.

## Phase 0 Boundary

- Do not install packages.
- Do not clone datasets.
- Do not add custom training modules.
- Do not run full fine-tuning.
- Do not commit automatically.
