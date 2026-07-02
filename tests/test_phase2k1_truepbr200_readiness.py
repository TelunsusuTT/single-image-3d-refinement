from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_phase2k1_truepbr200_readiness as readiness  # noqa: E402


def write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_samples(root: Path, count: int = 7) -> list[Path]:
    samples = []
    for index in range(count):
        sample = root / f"sample_{index}"
        (sample / "render_tex").mkdir(parents=True)
        (sample / "render_cond").mkdir(parents=True)
        samples.append(sample)
    return samples


def write_examples(path: Path, samples: list[Path]) -> Path:
    return write_text(path, json.dumps([str(sample.resolve()) for sample in samples]))


def make_hypaint(root: Path) -> Path:
    hypaint = root / "hy3dpaint"
    write_file(hypaint / "train.py", b"train")
    return hypaint


def make_case(root: Path) -> Path:
    case_dir = root / "outputs" / "phase2g" / "infer_cases" / "B075YLTF7Q"
    write_file(case_dir / "input" / "mesh.glb", b"mesh")
    write_file(case_dir / "input" / "image.png", b"image")
    return case_dir


def make_base_output(root: Path) -> Path:
    base_dir = root / "outputs" / "phase2g" / "infer_runs" / "B075YLTF7Q" / "base_a100_noremesh_smoke"
    for filename in (
        "base_textured_mesh.obj",
        "base_textured_mesh.glb",
        "base_textured_mesh.jpg",
        "base_textured_mesh_metallic.jpg",
        "base_textured_mesh_roughness.jpg",
    ):
        write_file(base_dir / filename, filename.encode("utf-8"))
    return base_dir


def make_fake_project_scripts(project_root: Path) -> Path:
    scripts_dir = project_root / "scripts"
    for filename in (
        "load_phase2g4_finetuned_checkpoint_only.py",
        "compare_phase2i_base_unet_to_checkpoints.py",
        "run_phase2g_paint_infer.py",
        "make_phase2g6_texture_comparison.py",
    ):
        write_file(scripts_dir / filename, b"script")
    return project_root


def write_config(
    path: Path,
    max_steps: int = 200,
    extra_text: str = "",
) -> Path:
    return write_text(
        path,
        "\n".join(
            [
                "model:",
                "  base_learning_rate: 1e-6",
                "  params:",
                "    stable_diffusion_config:",
                "      pretrained_model_name_or_path: /tmp/hunyuan3d-paintpbr-v2-1",
                "      custom_pipeline: ./hunyuanpaintpbr",
                "lightning:",
                "  modelcheckpoint:",
                "    params:",
                "      dirpath: /tmp/checkpoints/pilot_v1_truepbr_200_lr1e6",
                "      filename: pilot_v1_truepbr_200_lr1e6-step{step}",
                "      every_n_train_steps: 200",
                "      save_top_k: -1",
                "      save_last: false",
                "      save_weights_only: true",
                "  trainer:",
                f"    max_steps: {max_steps}",
                extra_text,
            ]
        )
        + "\n",
    )


def make_paths(root: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    samples = make_samples(root)
    examples_json = write_examples(root / "examples.json", samples)
    config = write_config(root / "config.yaml")
    hypaint = make_hypaint(root)
    checkpoint_root = root / "checkpoints" / "pilot_v1_truepbr_200_lr1e6"
    case_dir = make_case(root)
    base_dir = make_base_output(root)
    eval_output_root = root / "outputs" / "phase2k" / "eval_truepbr200"
    return examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_output_root


def run_readiness(
    examples_json: Path,
    config: Path,
    hypaint: Path,
    checkpoint_root: Path,
    case_dir: Path,
    base_dir: Path,
    eval_output_root: Path,
) -> int:
    return readiness.main(
        [
            "--examples-json",
            str(examples_json),
            "--config",
            str(config),
            "--hypaint",
            str(hypaint),
            "--checkpoint-root",
            str(checkpoint_root),
            "--case-dir",
            str(case_dir),
            "--base-dir",
            str(base_dir),
            "--eval-output-root",
            str(eval_output_root),
        ]
    )


def test_readiness_passes_with_valid_fake_setup(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))

        assert run_readiness(*paths) == 0


def test_readiness_fails_if_config_contains_sd2_community(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        write_config(config, extra_text="old: sd2-community/stable-diffusion-2-1")

        assert run_readiness(examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root) == 1


def test_readiness_fails_if_max_steps_is_not_200(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        write_config(config, max_steps=50)

        assert run_readiness(examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root) == 1


def test_readiness_fails_if_checkpoint_root_already_contains_ckpt(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        write_file(checkpoint_root / "old.ckpt", b"old")

        assert run_readiness(examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root) == 1


def test_readiness_fails_if_base_metallic_map_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root = make_paths(root)
        monkeypatch.setattr(readiness, "PROJECT_ROOT", make_fake_project_scripts(root / "project"))
        (base_dir / "base_textured_mesh_metallic.jpg").unlink()

        assert run_readiness(examples_json, config, hypaint, checkpoint_root, case_dir, base_dir, eval_root) == 1
