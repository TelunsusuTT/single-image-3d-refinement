from __future__ import annotations

import builtins
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "phase2n_week2_train_pilots.py"
PROTOCOL_PATH = PROJECT_ROOT / "src" / "hy3dft" / "protocol_corrected_dataset.py"


def load_script():
    spec = importlib.util.spec_from_file_location("phase2n_week2_train_pilots_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pilot_module():
    return load_script()


def write_text(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_fake_project(tmp_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    root = tmp_path / "project"
    root.mkdir()
    sample_paths: list[Path] = []
    for index in range(80):
        sample = root / "samples" / f"asset_{index:03d}"
        sample.mkdir(parents=True)
        sample_paths.append(sample.resolve())

    train_json = write_text(
        root / "data/hy3dpaint_train_examples/datav2_frame_panels_full101/examples_train_abs.json",
        json.dumps([str(path) for path in sample_paths]) + "\n",
    )
    test_sample = root / "test_samples" / "held_out"
    test_sample.mkdir(parents=True)
    write_text(train_json.parent / "examples_test_abs.json", json.dumps([str(test_sample.resolve())]) + "\n")

    write_text(
        root / "scripts/phase2n_day5_a100_preflight.py",
        "\n".join(
            [
                "class ValidatedConfiguration: pass",
                "def validate_config(): pass",
                "def initialize_true_pbr_model(): pass",
                "def run_production_loss(): pass",
                "def _reset_runtime_seeds(): pass",
                "def _prepend_runtime_paths(): pass",
                "def move_model_batch_to_device(): pass",
                "def optimizer_membership_report(): pass",
                "",
            ]
        ),
    )
    write_text(root / "configs/phase2n_day5_a100_preflight.json", "{}\n")
    write_text(
        root / "src/hy3dft/selective_training.py",
        "\n".join(
            [
                "class ProtocolCorrectedDatasetFromJson: pass",
                "def protocol_collate_fn(): pass",
                "def apply_trainable_scope(): pass",
                "def build_trainable_adamw(): pass",
                "def build_warmup_constant_scheduler(): pass",
                "",
            ]
        ),
    )
    write_text(
        root / "src/hy3dft/protocol_corrected_dataset.py",
        "\n".join(
            [
                "REFERENCE_VIEW_WEIGHTS = MappingProxyType({",
                "    '005': 0.50, '004': 0.30,",
                "    '000': 0.05, '001': 0.05, '002': 0.05, '003': 0.05,",
                "})",
                "",
            ]
        ),
    )

    cases = []
    for index in range(8):
        asset_id = f"eval_{index:02d}"
        mesh = write_text(root / "fixtures" / asset_id / "mesh.glb", "glb\n")
        reference = write_text(root / "fixtures" / asset_id / "005_light_AL.png", "png\n")
        cases.append(
            {
                "asset_id": asset_id,
                "source_split": "val" if index < 6 else "train",
                "eval_split": "val" if index < 6 else "train_sanity",
                "selection_stratum": "fixture",
                "selection_rationale": "fixture",
                "mesh_path": str(mesh.relative_to(root)),
                "reference_image_path": str(reference.relative_to(root)),
                "selected_input_view": "005",
                "reference_lighting": "AL",
            }
        )
    eval_manifest = {
        "phase": "phase2n_week2_pilot_evaluation",
        "selection_frozen_before_training": True,
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "case_count": 8,
        "split_counts": {"val": 6, "train_sanity": 2, "test": 0},
        "cases": cases,
    }
    write_text(root / "configs/phase2n_week2_pilot_eval_cases.json", json.dumps(eval_manifest) + "\n")

    sbatch = "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH -p a100",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=04:00:00",
            "set -eo pipefail",
            'export PYTHONPATH="$PROJ/src:$PROJ/scripts"',
            "phase2n_week2_train_pilots.py --check-only",
            "phase2n_week2_train_pilots.py --run-all --run-id fixture",
            "===== JOB END: SUCCESS =====",
            "",
        ]
    )
    write_text(root / "env/run_phase2n_week2_train_pilots_a100.sbatch", sbatch)
    write_text(root / "fixtures/audited_upstream_model.py", "# audited fixture\n")

    config = {
        "phase": "phase2n_week2_pilot_training",
        "train_json": "data/hy3dpaint_train_examples/datav2_frame_panels_full101/examples_train_abs.json",
        "output_root": "outputs/phase2n/week2_pilot_training",
        "base_seed": 42,
        "schedule_seed": 42,
        "image_size": 512,
        "batch_size": 1,
        "num_workers": 0,
        "augmentation_mode": "none",
        "conditioning_dropout_policy": {
            "drop_cond_prob": 0.1,
            "require_mva_active": True,
            "maximum_seed_retries": 128,
        },
        "max_steps": 320,
        "warmup_steps": 50,
        "gradient_clip_norm": 1.0,
        "precision": "bf16",
        "checkpoint_steps": [160, 320],
        "log_every_n_steps": 10,
        "scope_order": ["pc_s1", "pc_full"],
        "scopes": {"pc_s1": {"learning_rate": 1e-6}, "pc_full": {"learning_rate": 5e-7}},
        "minimum_gpu_memory_gib": 70,
        "minimum_free_disk_gib": 15,
    }
    config_path = write_text(root / "configs/phase2n_week2_pilot_training.json", json.dumps(config) + "\n")
    return root, config_path, tuple(sample_paths)


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def fake_schedule_contract(pilot_module, tmp_path: Path) -> dict[str, object]:
    source = write_text(tmp_path / "audited_upstream_model.py", "# audited fixture\n")
    return {
        "conditioning_dropout_policy": dict(pilot_module.EXPECTED_CONDITIONING_DROPOUT_POLICY),
        "conditioning_source_audit": {
            "source_path": str(source.resolve()),
            "sha256": pilot_module.sha256_file(source),
            "batch_size": 1,
            "use_dino": True,
            "draw_order": ["fixture_draw_order"],
        },
    }


def load_protocol_module():
    spec = importlib.util.spec_from_file_location(
        "phase2n_protocol_corrected_dataset_under_test",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_validation_accepts_exact_contract(pilot_module, tmp_path, monkeypatch):
    root, config_path, samples = make_fake_project(tmp_path)
    monkeypatch.setattr(
        pilot_module,
        "validate_free_disk",
        lambda path, minimum: {"checked_path": str(path), "free_bytes": 1 << 40, "free_gib": 1024.0},
    )
    validated = pilot_module.validate_training_config(config_path, root)
    assert validated.train_sample_paths == samples
    assert validated.values["scope_order"] == ["pc_s1", "pc_full"]
    assert validated.values["scopes"]["pc_s1"]["learning_rate"] == 1e-6
    assert validated.values["scopes"]["pc_full"]["learning_rate"] == 5e-7
    assert validated.values["conditioning_dropout_policy"] == {
        "drop_cond_prob": 0.1,
        "require_mva_active": True,
        "maximum_seed_retries": 128,
    }
    assert validated.output_root == (root / "outputs/phase2n/week2_pilot_training").resolve()


def test_config_requires_exact_conditioning_dropout_policy(pilot_module, tmp_path):
    root, config_path, _samples = make_fake_project(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["conditioning_dropout_policy"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen schema"):
        pilot_module.validate_training_config(
            config_path,
            root,
            require_repository_contracts=False,
            check_free_space=False,
        )

    config["conditioning_dropout_policy"] = {
        "drop_cond_prob": 0.1,
        "require_mva_active": True,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain exactly"):
        pilot_module.validate_training_config(
            config_path,
            root,
            require_repository_contracts=False,
            check_free_space=False,
        )


def test_config_rejects_test_json_and_test_asset_references(pilot_module, tmp_path):
    root, config_path, samples = make_fake_project(tmp_path)
    test_json = config_path.parent.parent / "data/test_examples.json"
    write_text(test_json, json.dumps([str(path) for path in samples]))
    with pytest.raises(ValueError, match="Test JSON"):
        pilot_module.reject_test_training_references(test_json, samples)

    held_out = root / "test_samples/held_out"
    modified = list(samples)
    modified[0] = held_out.resolve()
    train_json = root / pilot_module.EXPECTED_TRAIN_JSON_RELATIVE
    train_json.write_text(json.dumps([str(path) for path in modified]), encoding="utf-8")
    with pytest.raises(ValueError, match="includes test assets"):
        pilot_module.validate_training_config(
            config_path,
            root,
            require_repository_contracts=False,
            check_free_space=False,
        )


def test_failed_seed_preview_matches_audited_upstream_draws(pilot_module):
    preview = pilot_module.preview_conditioning_dropout(2775693311, 0.1)
    assert preview["normal_drop_draw"] == pytest.approx(0.5166324730649756)
    assert preview["position_drop_draw"] == pytest.approx(0.36297839298666623)
    assert preview["dino_primary_drop_draw"] == pytest.approx(0.8487201457374238)
    assert preview["dino_secondary_drop_draw"] == pytest.approx(0.6841948775649342)
    assert preview["mva_ref_branch_draw"] == pytest.approx(0.09623141240259903)
    assert preview["mva_ref_choice_draw"] is None
    assert preview["expected_mva_scale"] == 0.0
    assert preview["expected_ref_scale"] == 0.0


def test_schedule_is_exact_four_epoch_deterministic_mva_active_permutation(pilot_module, tmp_path):
    paths = tuple((tmp_path / f"asset_{index:03d}").resolve() for index in range(80))
    contract = fake_schedule_contract(pilot_module, tmp_path)
    first = pilot_module.build_training_schedule(
        paths,
        **contract,
        schedule_seed=42,
        max_steps=320,
    )
    second = pilot_module.build_training_schedule(
        paths,
        **contract,
        schedule_seed=42,
        max_steps=320,
    )
    assert first == second
    assert first["record_count"] == 320
    assert first["epoch_count"] == 4
    assert pilot_module.validate_training_schedule(first, paths, **contract) == first["schedule_hash"]
    assert pilot_module.mva_schedule_stats(first) == {
        "mva_active_records": 320,
        "mva_inactive_records": 0,
        "mva_seed_retry_records": first["mva_seed_retry_record_count"],
    }
    assert all(row["expected_mva_scale"] == 1.0 for row in first["records"])

    for epoch in range(4):
        rows = [row for row in first["records"] if row["epoch"] == epoch]
        assert len(rows) == 80
        assert {row["asset_index"] for row in rows} == set(range(80))
        assert [row["within_epoch_position"] for row in rows] == list(range(80))


def test_real_update_one_is_deterministically_remapped_to_mva_active(pilot_module):
    config = json.loads((PROJECT_ROOT / "configs/phase2n_week2_pilot_training.json").read_text(encoding="utf-8"))
    train_paths = tuple(
        Path(value).resolve()
        for value in json.loads(
            (PROJECT_ROOT / config["train_json"]).read_text(encoding="utf-8")
        )
    )
    source_audit = pilot_module.validate_conditioning_source_audit()
    contract = {
        "conditioning_dropout_policy": config["conditioning_dropout_policy"],
        "conditioning_source_audit": source_audit,
    }
    schedule = pilot_module.build_training_schedule(train_paths, **contract)
    assert schedule["conditioning_source_audit"]["source_path"] == str(
        pilot_module.AUDITED_CONDITIONING_SOURCE.resolve()
    )
    assert schedule["conditioning_source_audit"]["sha256"] == pilot_module.AUDITED_CONDITIONING_SOURCE_SHA256
    first = schedule["records"][0]
    assert first["global_update"] == 1
    assert first["asset_id"] == "B073P77B6B"
    assert first["original_candidate_seed"] == 2775693311
    assert first["training_step_seed"] != first["original_candidate_seed"]
    assert first["mva_seed_retry_count"] > 0
    assert first["expected_mva_scale"] == 1.0
    assert all(row["expected_mva_scale"] == 1.0 for row in schedule["records"])

    eighth = schedule["records"][7]
    assert eighth["global_update"] == 8
    assert eighth["original_candidate_seed"] == 383309399
    eighth_original = pilot_module.preview_conditioning_dropout(eighth["original_candidate_seed"], 0.1)
    assert eighth_original["mva_ref_branch_draw"] == pytest.approx(0.9223304803617985)
    assert eighth_original["mva_ref_choice_draw"] == pytest.approx(0.2981178873197329)
    assert eighth_original["expected_mva_scale"] == 0.0
    assert eighth_original["expected_ref_scale"] == 1.0
    assert eighth["expected_mva_scale"] == 1.0

    repeated = pilot_module.resolve_mva_active_seed(
        first["original_candidate_seed"],
        pilot_module._schedule_identity_from_record(first, schedule["schedule_seed"]),
        drop_cond_prob=0.1,
        maximum_seed_retries=128,
    )
    assert repeated["training_step_seed"] == first["training_step_seed"]
    assert repeated["mva_seed_retry_count"] == first["mva_seed_retry_count"]


def test_retry_limit_failure_is_explicit(pilot_module):
    with pytest.raises(RuntimeError, match="within 0 retries"):
        pilot_module.resolve_mva_active_seed(
            2775693311,
            {
                "schedule_seed": 42,
                "global_update": 1,
                "epoch": 0,
                "within_epoch_position": 0,
                "asset_index": 49,
                "asset_id": "B073P77B6B",
                "asset_path": "/sample/B073P77B6B",
                "original_candidate_seed": 2775693311,
            },
            drop_cond_prob=0.1,
            maximum_seed_retries=0,
        )


def test_shared_final_seeds_do_not_change_day3_reference_or_light(pilot_module, tmp_path):
    paths = tuple((tmp_path / f"asset_{index:03d}").resolve() for index in range(80))
    contract = fake_schedule_contract(pilot_module, tmp_path)
    schedule = pilot_module.build_training_schedule(paths, **contract)
    pc_s1_seeds = [row["training_step_seed"] for row in schedule["records"]]
    pc_full_seeds = [row["training_step_seed"] for row in schedule["records"]]
    assert pc_s1_seeds == pc_full_seeds
    assert len(set(pc_s1_seeds)) == 320
    assert all(0 <= seed < 2**32 for seed in pc_s1_seeds)

    protocol = load_protocol_module()
    references = tuple(
        protocol.ReferenceImage(view_id=view, lighting=light, path=tmp_path / f"{view}_{light}.png")
        for view in protocol.TARGET_VIEW_IDS
        for light in protocol.LIGHT_CONDITIONS
    )
    inventory = protocol.SampleInventory(
        sample_dir=tmp_path / "B073P77B6B",
        asset_id="B073P77B6B",
        reference_images=references,
        target_views=(),
    )
    before = protocol.make_protocol_decision(inventory, base_seed=42, epoch=0)
    first = schedule["records"][0]
    pilot_module.resolve_mva_active_seed(
        first["original_candidate_seed"],
        pilot_module._schedule_identity_from_record(first, schedule["schedule_seed"]),
        drop_cond_prob=0.1,
        maximum_seed_retries=128,
    )
    after = protocol.make_protocol_decision(inventory, base_seed=42, epoch=0)
    assert before == after
    assert before.selected_reference_view == after.selected_reference_view
    assert before.reference_lighting_pair == after.reference_lighting_pair


def make_trace_rows(count: int = 320) -> list[dict[str, object]]:
    return [
        {
            "global_update": update,
            "epoch": (update - 1) // 80,
            "asset_index": (update - 1) % 80,
            "asset_id": f"asset_{(update - 1) % 80:03d}",
            "asset_path": f"/sample/asset_{(update - 1) % 80:03d}",
            "original_candidate_seed": update * 13,
            "training_step_seed": update * 17,
            "expected_mva_scale": 1.0,
            "expected_ref_scale": 1.0,
            "mva_seed_retry_count": int(update % 7 == 0),
            "selected_reference_view": "005",
            "reference_lighting_pair": ["AL", "PL"],
            "reference_image_paths": ["/ref/al.png", "/ref/pl.png"],
            "target_view_order": ["000", "001", "002", "003", "004", "005"],
            "protocol_derived_seed": update * 19,
            "spatial_augmentation": "none",
        }
        for update in range(1, count + 1)
    ]


def write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_trace_equivalence_passes_and_fails_on_protocol_drift(pilot_module, tmp_path):
    rows = make_trace_rows()
    left = write_jsonl(tmp_path / "left.jsonl", rows)
    right = write_jsonl(tmp_path / "right.jsonl", rows)
    report = pilot_module.compare_sampling_traces(left, right)
    assert report["status"] == "OK"
    assert report["trace_hashes_equal"] is True

    changed = make_trace_rows()
    changed[17]["selected_reference_view"] = "004"
    drifted = write_jsonl(tmp_path / "drifted.jsonl", changed)
    with pytest.raises(RuntimeError, match="sampling traces differ"):
        pilot_module.compare_sampling_traces(left, drifted)

    changed_seed = make_trace_rows()
    changed_seed[0]["training_step_seed"] = 999999
    seed_drifted = write_jsonl(tmp_path / "seed_drifted.jsonl", changed_seed)
    with pytest.raises(RuntimeError, match="sampling traces differ"):
        pilot_module.compare_sampling_traces(left, seed_drifted)


class FakeTensor:
    dtype = "fake.float32"

    def __init__(self, values):
        self.values = tuple(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeTensor(self.values)

    def numel(self):
        return len(self.values)


class FakeParameter(FakeTensor):
    def __init__(self, values, *, requires_grad):
        super().__init__(values)
        self.requires_grad = requires_grad
        self.grad = None


class FakeModel:
    def __init__(self):
        self.items = [
            ("trainable.weight", FakeParameter([1.0, 2.0], requires_grad=True)),
            ("frozen.weight", FakeParameter([3.0], requires_grad=False)),
        ]

    def named_parameters(self):
        return iter(self.items)


class FakeTorch:
    @staticmethod
    def save(payload, path):
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    @staticmethod
    def load(path, map_location=None, weights_only=True):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def fake_scope_report():
    return {
        "scope": "pc_s1",
        "total_parameter_tensor_count": 2,
        "total_parameter_numel": 3,
        "trainable_parameter_tensor_count": 1,
        "trainable_parameter_numel": 2,
        "frozen_parameter_tensor_count": 1,
        "frozen_parameter_numel": 1,
        "trainable_parameter_names": ["trainable.weight"],
        "frozen_parameter_names": ["frozen.weight"],
    }


def test_checkpoint_is_atomic_trainable_only_and_manifest_is_hashed(pilot_module, tmp_path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "pc_s1/checkpoints"
    manifest = pilot_module.save_scope_checkpoint(
        scope_root=FakeModel(),
        scope="pc_s1",
        scope_report=fake_scope_report(),
        step=160,
        checkpoints_dir=checkpoints,
        run_dir=run_dir,
        schedule_hash="schedule-hash",
        sampling_trace_hash="trace-hash",
        learning_rate=1e-6,
        base_identifier="true-pbr",
        torch=FakeTorch,
    )
    state_path = checkpoints / "step_160_scope_state.pt"
    loaded = FakeTorch.load(state_path)
    assert list(loaded) == ["trainable.weight"]
    assert "frozen.weight" not in loaded
    assert manifest["byte_size"] == state_path.stat().st_size
    assert manifest["sha256"] == pilot_module.sha256_file(state_path)
    assert manifest["contains_optimizer_state"] is False
    assert manifest["contains_full_model"] is False
    assert not list(checkpoints.glob("*.tmp-*"))

    with pytest.raises(ValueError, match="Checkpoint step"):
        pilot_module.save_scope_checkpoint(
            scope_root=FakeModel(),
            scope="pc_s1",
            scope_report=fake_scope_report(),
            step=100,
            checkpoints_dir=checkpoints,
            run_dir=run_dir,
            schedule_hash="schedule-hash",
            sampling_trace_hash="trace-hash",
            learning_rate=1e-6,
            base_identifier="true-pbr",
            torch=FakeTorch,
        )


def make_complete_scope(pilot_module, run_dir: Path, scope: str, schedule_hash: str):
    scope_dir = run_dir / scope
    paths = pilot_module._scope_paths(scope_dir)
    paths["checkpoints"].mkdir(parents=True)
    report = fake_scope_report()
    report["scope"] = scope
    pilot_module.atomic_write_json(paths["scope_report"], report, run_dir)
    rows = make_trace_rows()
    write_jsonl(paths["trace"], rows)
    write_jsonl(paths["metrics"], [{"global_update": index} for index in range(1, 321)])
    for step in (160, 320):
        pilot_module.save_scope_checkpoint(
            scope_root=FakeModel(),
            scope=scope,
            scope_report=report,
            step=step,
            checkpoints_dir=paths["checkpoints"],
            run_dir=run_dir,
            schedule_hash=schedule_hash,
            sampling_trace_hash=pilot_module.sha256_jsonl_prefix(paths["trace"], step),
            learning_rate=1e-6,
            base_identifier="true-pbr",
            torch=FakeTorch,
        )
    summary = {
        "status": "OK",
        "scope": scope,
        "schedule_sha256": schedule_hash,
        "sampling_trace_sha256": pilot_module.sha256_file(paths["trace"]),
    }
    pilot_module.atomic_write_json(paths["summary_json"], summary, run_dir)
    pilot_module.atomic_write_text(paths["summary_md"], "complete\n", run_dir)
    pilot_module.atomic_write_text(paths["success"], pilot_module.SCOPE_SUCCESS_TOKENS[scope] + "\n", run_dir)
    return scope_dir


def test_scope_completion_is_idempotent_and_incomplete_scope_fails(pilot_module, tmp_path):
    run_dir = tmp_path / "run"
    scope_dir = make_complete_scope(pilot_module, run_dir, "pc_s1", "schedule-hash")
    first = pilot_module.validate_scope_completion(
        scope_dir, "pc_s1", "schedule-hash", torch=FakeTorch
    )
    second = pilot_module.validate_scope_completion(
        scope_dir, "pc_s1", "schedule-hash", torch=FakeTorch
    )
    assert first == second
    assert first["summary"]["status"] == "OK"

    incomplete = tmp_path / "other_run/pc_full"
    incomplete.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        pilot_module.validate_scope_completion(
            incomplete, "pc_full", "schedule-hash", torch=FakeTorch
        )


def test_nan_inf_and_scope_drift_are_hard_failures(pilot_module):
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(RuntimeError, match="NaN or Inf"):
            pilot_module.require_finite_scalar(value, "loss")
    model = FakeModel()
    pilot_module.validate_scope_integrity(model, ["trainable.weight"])
    model.items[1][1].requires_grad = True
    with pytest.raises(RuntimeError, match="scope drift"):
        pilot_module.validate_scope_integrity(model, ["trainable.weight"])


def test_predicted_active_all_zero_gradient_remains_fatal(pilot_module):
    class ZeroNorm:
        @staticmethod
        def item():
            return 0.0

    class ZeroGradientTorch:
        class nn:
            class utils:
                @staticmethod
                def clip_grad_norm_(parameters, max_norm, error_if_nonfinite):
                    assert len(parameters) == 1
                    assert max_norm == 1.0
                    assert error_if_nonfinite is True
                    return ZeroNorm()

    model = FakeModel()
    model.items[0][1].grad = object()
    with pytest.raises(RuntimeError, match="all-zero gradient norm"):
        pilot_module.validate_and_clip_gradients(
            model,
            ["trainable.weight"],
            1.0,
            ZeroGradientTorch,
        )


def test_check_only_is_stdlib_no_cuda_no_hunyuan_and_writes_nothing(
    pilot_module, tmp_path, monkeypatch, capsys
):
    root, config_path, _samples = make_fake_project(tmp_path)
    failed_run_record = write_text(
        root / "outputs/phase2n/week2_pilot_training/slurm_264014/training_schedule.json",
        "historical failed-run fixture\n",
    )
    failed_run_bytes = failed_run_record.read_bytes()
    audited_source = root / "fixtures/audited_upstream_model.py"
    monkeypatch.setattr(pilot_module, "AUDITED_CONDITIONING_SOURCE", audited_source)
    monkeypatch.setattr(
        pilot_module,
        "AUDITED_CONDITIONING_SOURCE_SHA256",
        pilot_module.sha256_file(audited_source),
    )
    before = tree_state(root)
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith(
            ("torch.", "pytorch_lightning", "lightning", "hunyuan", "omegaconf")
        ):
            attempted.append(name)
            raise AssertionError(f"check-only attempted heavy import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        pilot_module,
        "validate_free_disk",
        lambda path, minimum: {"checked_path": str(path), "free_bytes": 1 << 40, "free_gib": 1024.0},
    )
    validated = pilot_module.run_check_only(config_path, root)
    output = capsys.readouterr().out
    assert len(validated.train_sample_paths) == 80
    assert attempted == []
    assert tree_state(root) == before
    assert failed_run_record.read_bytes() == failed_run_bytes
    assert "schedule_records=320" in output
    assert "schedule_epochs=4" in output
    assert "mva_active_records=320" in output
    assert "mva_inactive_records=0" in output
    assert "mva_seed_retry_records=" in output
    assert pilot_module.MVA_SCHEDULE_TOKEN in output
    assert pilot_module.READINESS_TOKEN in output


def test_check_only_fails_if_audited_upstream_hash_changes(pilot_module, tmp_path, monkeypatch):
    root, config_path, _samples = make_fake_project(tmp_path)
    audited_source = root / "fixtures/audited_upstream_model.py"
    monkeypatch.setattr(pilot_module, "AUDITED_CONDITIONING_SOURCE", audited_source)
    monkeypatch.setattr(pilot_module, "AUDITED_CONDITIONING_SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(
        pilot_module,
        "validate_free_disk",
        lambda path, minimum: {"checked_path": str(path), "free_bytes": 1 << 40, "free_gib": 1024.0},
    )
    with pytest.raises(RuntimeError, match="SHA-256 changed"):
        pilot_module.run_check_only(config_path, root)


def test_frozen_repository_eval_manifest_is_six_val_two_train_and_zero_test(pilot_module):
    report = pilot_module.validate_eval_manifest(
        PROJECT_ROOT / "configs/phase2n_week2_pilot_eval_cases.json",
        PROJECT_ROOT,
    )
    assert report["split_counts"] == {"val": 6, "train_sanity": 2, "test": 0}
    assert len(report["case_ids"]) == 8
    assert len(set(report["case_ids"])) == 8
