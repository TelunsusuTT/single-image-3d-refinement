#!/usr/bin/env python3
"""Zero-update LoRA adapter smoke for Phase 2M.

The smoke loads the official inference UNet at runtime, injects only adapter
weights into the exact ref/dino projection Linear targets, saves adapter-only
state, reloads it, and verifies no upstream files changed. It does not train,
run inference, merge adapters, or save a full model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_HY21 = Path("/vol/bitbucket/ct1022/Hunyuan3D2.1_Work/src/Hunyuan3D-2.1")
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "phase2m" / "zero_lora_smoke"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.lora.io import load_adapter_state_dict as read_adapter_state_dict  # noqa: E402
from hy3dft.lora.io import save_adapter_state_dict  # noqa: E402
from hy3dft.lora.local_linear import adapter_state_dict, load_adapter_state_dict as apply_adapter_state_dict  # noqa: E402
from hy3dft.lora.peft_lora import assert_only_lora_trainable, inject_lora, print_trainable_summary  # noqa: E402
from hy3dft.lora.targeting import select_lora_targets  # noqa: E402


SKIP_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", "outputs", "logs", "checkpoints", "data", "caches"}


def resolve_official_paths() -> tuple[Path, Path, Path]:
    hy21 = Path(os.environ.get("HY21") or DEFAULT_HY21).expanduser().resolve()
    hypaint = Path(os.environ.get("HYPAINT") or hy21 / "hy3dpaint").expanduser().resolve()
    official_root = Path(os.environ.get("HY21_WORK_ROOT") or hy21.parents[1]).expanduser().resolve()
    if not hy21.is_dir():
        raise FileNotFoundError(f"HY21 path missing: {hy21}")
    if not hypaint.is_dir():
        raise FileNotFoundError(f"HYPAINT path missing: {hypaint}")
    if not official_root.is_dir():
        raise FileNotFoundError(f"official root missing: {official_root}")
    return hy21, hypaint, official_root


def prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if text not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([text, *parts])


def set_absolute_official_config_paths(conf: Any, hypaint: Path) -> dict[str, str]:
    paths = {
        "multiview_cfg_path": hypaint / "cfgs" / "hunyuan-paint-pbr.yaml",
        "realesrgan_ckpt_path": hypaint / "ckpt" / "RealESRGAN_x4plus.pth",
    }
    for attr, path in paths.items():
        if hasattr(conf, attr):
            setattr(conf, attr, str(path))
    return {key: str(value) for key, value in paths.items()}


def manifest_tree(root: Path, max_files: int = 200_000) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        for filename in sorted(filenames):
            path = Path(current) / filename
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            if len(entries) > max_files:
                raise RuntimeError(f"Official tree manifest exceeded max_files={max_files}: {root}")
    return {"root": str(root), "file_count": len(entries), "entries": entries}


def compare_manifests(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_map = {item["path"]: (item["size"], item["mtime_ns"]) for item in before["entries"]}
    after_map = {item["path"]: (item["size"], item["mtime_ns"]) for item in after["entries"]}
    added = sorted(set(after_map) - set(before_map))
    removed = sorted(set(before_map) - set(after_map))
    changed = sorted(path for path in set(before_map) & set(after_map) if before_map[path] != after_map[path])
    return {"added": added, "removed": removed, "changed": changed, "unchanged": not (added or removed or changed)}


def load_inference_unet(max_num_view: int, resolution: int, device: str) -> tuple[Any, dict[str, str]]:
    hy21, hypaint, official_root = resolve_official_paths()
    prepend_pythonpath(SRC_ROOT)
    prepend_pythonpath(hy21)
    prepend_pythonpath(hypaint)

    # Heavy official imports happen only in the A100 runtime path.
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore

    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.device = device
    config_paths = set_absolute_official_config_paths(conf, hypaint)
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    unet = paint_pipeline.models["multiview_model"].pipeline.unet
    metadata = {
        "hy21": str(hy21),
        "hypaint": str(hypaint),
        "official_root": str(official_root),
        "unet_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
        **config_paths,
    }
    return unet, metadata


def tensor_bytes(state: dict[str, Any]) -> int:
    total = 0
    for value in state.values():
        numel = getattr(value, "numel", None)
        element_size = getattr(value, "element_size", None)
        if callable(numel) and callable(element_size):
            total += int(numel()) * int(element_size())
    return total


def ensure_summary_alias(canonical_summary_path: Path) -> Path:
    legacy_path = canonical_summary_path.with_name("smoke_summary.json")
    if legacy_path.exists() or legacy_path.is_symlink():
        return legacy_path
    try:
        legacy_path.symlink_to(canonical_summary_path.name)
    except OSError:
        shutil.copy2(canonical_summary_path, legacy_path)
    return legacy_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2M zero-update LoRA adapter smoke.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend", default="auto", choices=("auto", "local_linear_fallback", "peft", "diffusers"))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--preset", default="ref_dino", choices=("ref_dino",))
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-forward-check", action="store_true", help="Reserved; not enabled for Phase 2M M1.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_forward_check:
        raise NotImplementedError("Phase 2M M1 does not run inference/forward checks.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.out_dir / "ref_dino_adapter_state.pt"
    config_path = args.out_dir / "adapter_config.json"
    summary_path = args.out_dir / "zero_lora_smoke_summary.json"
    before_manifest_path = args.out_dir / "official_tree_before.json"
    after_manifest_path = args.out_dir / "official_tree_after.json"

    hy21, hypaint, official_root = resolve_official_paths()
    before_manifest = manifest_tree(official_root)
    before_manifest_path.write_text(json.dumps(before_manifest, indent=2) + "\n", encoding="utf-8")

    unet, official_paths = load_inference_unet(args.max_num_view, args.resolution, args.device)
    full_state_bytes = tensor_bytes(unet.state_dict())
    targets = select_lora_targets(unet, preset=args.preset)
    injection = inject_lora(unet, targets, rank=args.rank, alpha=args.alpha, dropout=args.dropout, backend=args.backend)
    assert_only_lora_trainable(unet)
    trainable_summary = print_trainable_summary(unet)

    adapter_state = adapter_state_dict(unet)
    adapter_bytes = tensor_bytes(adapter_state)
    if full_state_bytes <= 0:
        raise RuntimeError("Full model state byte estimate is zero; refusing unsafe smoke result")
    if adapter_bytes >= full_state_bytes // 4:
        raise RuntimeError(
            f"Adapter state is unexpectedly large: adapter_bytes={adapter_bytes} full_state_bytes={full_state_bytes}"
        )

    adapter_config = {
        "phase": "2M",
        "preset": args.preset,
        "backend": injection.backend,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_names": targets,
        "target_count": len(targets),
        "official_paths": official_paths,
        "safety": {
            "adapter_only": True,
            "merge_into_base": False,
            "save_pretrained_full_model": False,
        },
    }
    save_adapter_state_dict(adapter_state, adapter_path, adapter_config)
    if not config_path.is_file():
        raise RuntimeError(f"adapter config missing after save: {config_path}")
    loaded_state = read_adapter_state_dict(adapter_path)
    load_result = apply_adapter_state_dict(unet, loaded_state, strict=True)

    after_manifest = manifest_tree(official_root)
    after_manifest_path.write_text(json.dumps(after_manifest, indent=2) + "\n", encoding="utf-8")
    manifest_diff = compare_manifests(before_manifest, after_manifest)
    if not manifest_diff["unchanged"]:
        raise RuntimeError(f"Official tree changed during zero LoRA smoke: {manifest_diff}")

    summary = {
        "phase": "2M",
        "status": "OK",
        "adapter_path": str(adapter_path),
        "adapter_config_path": str(config_path),
        "backend": injection.backend,
        "backend_note": injection.backend_note,
        "selected_target_count": len(targets),
        "selected_target_names": targets,
        "trainable_parameter_count": injection.trainable_parameter_count,
        "total_parameter_count": injection.total_parameter_count,
        "trainable_names_sample": trainable_summary["trainable_names"][:50],
        "full_state_bytes_estimate": full_state_bytes,
        "adapter_state_bytes_estimate": adapter_bytes,
        "adapter_to_full_byte_ratio": adapter_bytes / full_state_bytes,
        "load_result": load_result,
        "official_paths": official_paths,
        "official_tree_diff": manifest_diff,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    alias_path = ensure_summary_alias(summary_path)
    print(f"summary_alias_path={alias_path}")
    print(f"adapter_path={adapter_path}")
    print(f"summary_path={summary_path}")
    print(f"selected_target_count={len(targets)}")
    print(f"adapter_state_bytes_estimate={adapter_bytes}")
    print(f"full_state_bytes_estimate={full_state_bytes}")
    print(f"backend={injection.backend}")
    print("PHASE2M_M1_ZERO_LORA_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
