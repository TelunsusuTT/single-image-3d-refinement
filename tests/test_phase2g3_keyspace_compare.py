from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2g3_key_inspect_readiness import main as readiness_main  # noqa: E402
from compare_phase2g3_keyspaces import main as compare_main  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "textureGenPipeline.py", b"pipeline")
    write_file(hypaint / "hunyuanpaintpbr" / "pipeline.py", b"paint pipeline")
    write_file(hypaint / "cfgs" / "hunyuan-paint-pbr.yaml", b"model: {}")
    write_file(hypaint / "ckpt" / "RealESRGAN_x4plus.pth", b"ckpt")
    return hypaint


def run_compare(checkpoint_json: Path, infer_json: Path, out_json: Path, out_md: Path) -> int:
    return compare_main(
        [
            "--checkpoint-json",
            str(checkpoint_json),
            "--infer-json",
            str(infer_json),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )


def run_readiness(checkpoint: Path, hypaint: Path, output_dir: Path) -> int:
    return readiness_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--hypaint",
            str(hypaint),
            "--output-dir",
            str(output_dir),
        ]
    )


def load_compare_report(root: Path, checkpoint_data: dict, infer_data: dict) -> dict:
    checkpoint_json = write_json(root / "checkpoint.json", checkpoint_data)
    infer_json = write_json(root / "infer.json", infer_data)
    out_json = root / "compare.json"
    out_md = root / "compare.md"

    assert run_compare(checkpoint_json, infer_json, out_json, out_md) == 0
    return json.loads(out_json.read_text(encoding="utf-8"))


def test_compare_recommends_unet_unet_strip_for_full_overlap() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report = load_compare_report(
            root,
            {
                "state_dict_all_keys": [
                    "unet.unet.conv.weight",
                    "unet.unet.conv.bias",
                    "other.module.weight",
                ]
            },
            {
                "candidates": [
                    {
                        "attribute_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
                        "class_name": "FakeUNet",
                        "all_keys": ["conv.bias", "conv.weight"],
                    }
                ]
            },
        )
        assert report["recommendation_status"] == "RECOMMENDED"
        assert report["recommended_transform"] == "strip:unet.unet."
        assert report["recommended_prefix_to_strip"] == "unet.unet."


def test_compare_recommends_nested_candidate_with_unet_unet_strip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report = load_compare_report(
            root,
            {
                "state_dict_all_keys": [
                    "unet.unet.down.weight",
                    "unet.unet.up.bias",
                    "trainer.global_step",
                ]
            },
            {
                "candidates": [
                    {
                        "attribute_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
                        "class_name": "OuterUNet",
                        "all_keys": ["outer.a", "outer.b", "outer.c"],
                    },
                    {
                        "attribute_path": "paint_pipeline.models['multiview_model'].pipeline.unet.unet",
                        "class_name": "NestedUNet",
                        "all_keys": ["down.weight", "up.bias"],
                    },
                ]
            },
        )
        assert report["recommendation_status"] == "RECOMMENDED"
        assert report["recommended_target_path"] == "paint_pipeline.models['multiview_model'].pipeline.unet.unet"
        assert report["recommended_target_class"] == "NestedUNet"
        assert report["recommended_transform"] == "strip:unet.unet."


def test_compare_recommends_strip_unet_transform() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report = load_compare_report(
            root,
            {
                "state_dict_all_keys": [
                    "unet.conv.weight",
                    "unet.conv.bias",
                ]
            },
            {
                "candidates": [
                    {
                        "attribute_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
                        "class_name": "OuterUNet",
                        "all_keys": ["conv.bias", "conv.weight"],
                    }
                ]
            },
        )
        assert report["recommendation_status"] == "RECOMMENDED"
        assert report["recommended_transform"] == "strip:unet."
        assert report["recommended_prefix_to_strip"] == "unet."


def test_compare_reports_unknown_when_overlap_is_low() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report = load_compare_report(
            root,
            {"state_dict_all_keys": ["unet.unet.alpha.weight", "unet.unet.beta.bias"]},
            {
                "candidates": [
                    {
                        "attribute_path": "paint_pipeline.models['multiview_model'].pipeline.unet",
                        "class_name": "FakeUNet",
                        "all_keys": ["conv.weight", "conv.bias", "block.weight"],
                    }
                ]
            },
        )
        assert report["recommendation_status"] == "UNKNOWN"
        assert report["recommended_transform"] == "UNKNOWN"


def test_readiness_fails_if_checkpoint_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        output_dir = root / "outputs" / "keys"

        assert run_readiness(root / "missing.ckpt", hypaint, output_dir) == 1


def test_readiness_passes_with_fake_hunyuan_files_and_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        hypaint = make_hypaint(root)
        checkpoint = write_file(root / "model.ckpt", b"checkpoint")
        output_dir = root / "outputs" / "keys"

        assert run_readiness(checkpoint, hypaint, output_dir) == 0
        assert output_dir.parent.is_dir()
