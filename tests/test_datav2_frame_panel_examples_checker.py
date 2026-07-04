from __future__ import annotations

import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_datav2_frame_panel_examples import main as check_main  # noqa: E402


TEX_SUFFIXES = [".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"]
COND_SUFFIXES = ["_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png(path: Path, width: int = 512, height: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(raw))
    data += png_chunk(b"IEND", b"")
    path.write_bytes(data)


def write_sample(sample_dir: Path) -> None:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    render_tex.mkdir(parents=True, exist_ok=True)
    render_cond.mkdir(parents=True, exist_ok=True)
    (render_tex / "transforms.json").write_text("{}\n", encoding="utf-8")
    for index in range(6):
        view = f"{index:03d}"
        for suffix in TEX_SUFFIXES:
            write_png(render_tex / f"{view}{suffix}")
        for suffix in COND_SUFFIXES:
            write_png(render_cond / f"{view}{suffix}")


def write_config(root: Path) -> Path:
    split_file = root / "split.json"
    split_file.write_text(json.dumps({"counts": {"train": 1, "val": 1, "test": 1}}), encoding="utf-8")
    config = {
        "dataset_name": "datav2_frame_panels_mini40",
        "split_file": str(split_file),
        "split_membership_csv": str(root / "membership.csv"),
        "curated_manifest_csv": str(root / "curated.csv"),
        "output_dataset_root": str(root / "dataset"),
        "report_root": str(root / "reports"),
        "view_ids": ["000", "001", "002", "003", "004", "005"],
        "light_conditions": ["AL", "ENVMAP", "PL"],
        "render_resolution": 512,
        "selected_input_view_field": "selected_input_view",
        "background_rgb": [71, 71, 71],
        "blender_bin_default": "/vol/bitbucket/ct1022/tools/bin/blender",
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_examples_checker_accepts_complete_fake_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train = root / "dataset" / "ITEM_A"
        val = root / "dataset" / "ITEM_B"
        test = root / "dataset" / "ITEM_C"
        for sample in (train, val, test):
            write_sample(sample)
        (root / "dataset").mkdir(parents=True, exist_ok=True)
        (root / "dataset" / "examples_train_abs.json").write_text(json.dumps([str(train.resolve())]), encoding="utf-8")
        (root / "dataset" / "examples_val_abs.json").write_text(json.dumps([str(val.resolve())]), encoding="utf-8")
        (root / "dataset" / "examples_test_abs.json").write_text(json.dumps([str(test.resolve())]), encoding="utf-8")
        (root / "dataset" / "examples_all_abs.json").write_text(
            json.dumps(sorted([str(train.resolve()), str(val.resolve()), str(test.resolve())])),
            encoding="utf-8",
        )
        config = write_config(root)

        assert check_main(["--config", str(config)]) == 0
        summary = json.loads((root / "reports" / "check_summary.json").read_text(encoding="utf-8"))

        assert summary["summary"]["failure_count"] == 0
        assert summary["summary"]["sample_count"] == 3


def test_examples_checker_fails_missing_expected_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train = root / "dataset" / "ITEM_A"
        val = root / "dataset" / "ITEM_B"
        test = root / "dataset" / "ITEM_C"
        for sample in (train, val, test):
            write_sample(sample)
        (test / "render_tex" / "005_pos.png").unlink()
        (root / "dataset" / "examples_train_abs.json").write_text(json.dumps([str(train.resolve())]), encoding="utf-8")
        (root / "dataset" / "examples_val_abs.json").write_text(json.dumps([str(val.resolve())]), encoding="utf-8")
        (root / "dataset" / "examples_test_abs.json").write_text(json.dumps([str(test.resolve())]), encoding="utf-8")
        (root / "dataset" / "examples_all_abs.json").write_text(
            json.dumps(sorted([str(train.resolve()), str(val.resolve()), str(test.resolve())])),
            encoding="utf-8",
        )
        config = write_config(root)

        assert check_main(["--config", str(config)]) == 1
        summary = json.loads((root / "reports" / "check_summary.json").read_text(encoding="utf-8"))

        assert summary["summary"]["failure_count"] > 0
        assert any("ITEM_C" in failure for failure in summary["summary"]["failures"])
