# Training Notes

Phase 0 uses the official Hunyuan3D-Paint training entry point:

```bash
python3 train.py --base "$PROJ/configs/ft_overfit_official.yaml" --name official_overfit --logdir "$PROJ/logs/train"
```

The command is launched from:

```bash
/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/hy3dpaint
```

Outputs must remain under this project folder, especially:

- `logs/train/`
- `checkpoints/`
- `outputs/`
- `tmp/`
- `caches/`

Phase 0 does not run LoRA or full fine-tuning. It only prepares and launches the official smoke test with a timeout.
