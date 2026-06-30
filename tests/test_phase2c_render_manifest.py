from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_phase2c_rendered_dataset import main as check_dataset_main  # noqa: E402
from make_phase2c_examples_json import main as examples_main  # noqa: E402
from make_phase2c_render_commands import main as commands_main  # noqa: E402
from make_phase2c_render_manifest import main as manifest_main  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_phase2b_inputs(root: Path) -> tuple[Path, Path]:
    download_manifest = root / "download_manifest.csv"
    inspection_summary = root / "inspection_summary.csv"
    write_csv(
        download_manifest,
        [
            "candidate_id",
            "source_id",
            "local_glb_path",
            "product_type",
            "name",
            "faces",
            "textures",
            "images",
        ],
        [
            {
                "candidate_id": "candidate_pass",
                "source_id": "PASS",
                "local_glb_path": "data/raw/PASS.glb",
                "product_type": "BOX",
                "name": "Pass Box",
                "faces": "10",
                "textures": "3",
                "images": "3",
            },
            {
                "candidate_id": "candidate_fail",
                "source_id": "FAIL",
                "local_glb_path": "data/raw/FAIL.glb",
                "product_type": "BOX",
                "name": "Fail Box",
                "faces": "20",
                "textures": "3",
                "images": "3",
            },
        ],
    )
    write_csv(
        inspection_summary,
        ["candidate_id", "source_id", "pass"],
        [
            {"candidate_id": "candidate_pass", "source_id": "PASS", "pass": "yes"},
            {"candidate_id": "candidate_fail", "source_id": "FAIL", "pass": "no"},
        ],
    )
    return download_manifest, inspection_summary


def make_render_manifest(root: Path) -> Path:
    download_manifest, inspection_summary = make_phase2b_inputs(root)
    render_manifest = root / "render_manifest.csv"
    assert (
        manifest_main(
            [
                "--download-manifest",
                str(download_manifest),
                "--inspection-summary",
                str(inspection_summary),
                "--out-csv",
                str(render_manifest),
                "--dataset-root",
                "data/hy3dpaint_train_examples/pilot_v1",
                "--qa-root",
                "outputs/qa/framing/pilot_v1",
            ]
        )
        == 0
    )
    return render_manifest


def test_render_manifest_only_includes_pass_assets() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        render_manifest = make_render_manifest(Path(tmpdir))
        rows = read_csv(render_manifest)
        assert len(rows) == 1
        row = rows[0]
        assert row["source_id"] == "PASS"
        assert row["candidate_id"] == "candidate_pass"
        assert row["input_glb"] == "data/raw/PASS.glb"
        assert row["sample_name"] == "PASS"
        assert row["sample_dir"] == "data/hy3dpaint_train_examples/pilot_v1/PASS"
        assert row["qa_dir"] == "outputs/qa/framing/pilot_v1/PASS"


def test_render_commands_include_renderer_and_qa_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        render_manifest = make_render_manifest(root)
        out_sh = root / "render.sh"
        assert (
            commands_main(
                [
                    "--render-manifest",
                    str(render_manifest),
                    "--out-sh",
                    str(out_sh),
                    "--blender-bin",
                    "/vol/bitbucket/ct1022/tools/bin/blender",
                    "--num-view",
                    "6",
                    "--resolution",
                    "512",
                ]
            )
            == 0
        )
        text = out_sh.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "scripts/blender_render_hy3dpaint_example.py" in text
        assert "--qa-root outputs/qa/framing/pilot_v1" in text
        assert "--sample-name PASS" in text


def test_examples_json_relative_and_absolute() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        render_manifest = make_render_manifest(root)
        rel_json = root / "examples.json"
        abs_json = root / "examples_train_abs.json"

        assert examples_main(["--render-manifest", str(render_manifest), "--out-json", str(rel_json)]) == 0
        assert (
            examples_main(
                [
                    "--render-manifest",
                    str(render_manifest),
                    "--out-json",
                    str(abs_json),
                    "--absolute",
                ]
            )
            == 0
        )

        rel_data = json.loads(rel_json.read_text(encoding="utf-8"))
        abs_data = json.loads(abs_json.read_text(encoding="utf-8"))
        assert rel_data == ["data/hy3dpaint_train_examples/pilot_v1/PASS"]
        assert Path(abs_data[0]).is_absolute()
        assert abs_data[0].endswith("data/hy3dpaint_train_examples/pilot_v1/PASS")


def write_expected_render_files(sample_dir: Path, num_view: int = 1) -> None:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    render_tex.mkdir(parents=True, exist_ok=True)
    render_cond.mkdir(parents=True, exist_ok=True)
    (render_tex / "transforms.json").write_text("{}", encoding="utf-8")
    for index in range(num_view):
        prefix = f"{index:03d}"
        for suffix in [".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"]:
            (render_tex / f"{prefix}{suffix}").write_text("x", encoding="utf-8")
        for suffix in ["_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"]:
            (render_cond / f"{prefix}{suffix}").write_text("x", encoding="utf-8")


def test_dataset_checker_detects_missing_and_present_samples() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        good_sample = root / "dataset" / "GOOD"
        missing_sample = root / "dataset" / "MISSING"
        qa_root = root / "qa"
        manifest = root / "render_manifest.csv"
        out_csv = root / "summary.csv"
        out_md = root / "summary.md"
        write_expected_render_files(good_sample, num_view=1)
        (qa_root / "GOOD").mkdir(parents=True)
        (qa_root / "GOOD" / "framing_report.json").write_text("{}", encoding="utf-8")
        write_csv(
            manifest,
            ["source_id", "sample_dir"],
            [
                {"source_id": "GOOD", "sample_dir": str(good_sample)},
                {"source_id": "MISSING", "sample_dir": str(missing_sample)},
            ],
        )

        assert (
            check_dataset_main(
                [
                    "--render-manifest",
                    str(manifest),
                    "--num-view",
                    "1",
                    "--framing-qa-root",
                    str(qa_root),
                    "--out-csv",
                    str(out_csv),
                    "--out-md",
                    str(out_md),
                ]
            )
            == 1
        )

        rows = {row["source_id"]: row for row in read_csv(out_csv)}
        assert rows["GOOD"]["status"] == "pass"
        assert rows["GOOD"]["expected_file_count"] == "9"
        assert rows["GOOD"]["framing_report_exists"] == "yes"
        assert rows["MISSING"]["status"] == "fail"
        assert int(rows["MISSING"]["missing_count"]) > 0
