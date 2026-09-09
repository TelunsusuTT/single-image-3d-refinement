from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _workflow import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    WorkflowError,
    build_command,
    execute_runner,
    load_json_object,
    validate_evaluation_config,
    validate_method_config,
    validate_runner,
)


METHOD_CASES = (
    ("configs/baseline/corrected_conditioning_baseline.json", "inference", "ready"),
    ("configs/adaptation/broad_scope_finetuning.json", "training", "planned"),
    ("configs/adaptation/reference_conditioning_lora.json", "training", "planned"),
    ("configs/adaptation/protocol_corrected_broad_finetuning.json", "training", "planned"),
    ("configs/adaptation/protocol_corrected_mva_finetuning.json", "training", "planned"),
    ("configs/gating/view_selective_conditioning_gating.json", "inference", "planned"),
)


@pytest.mark.parametrize(("relative_path", "stage", "status"), METHOD_CASES)
def test_method_configs_use_canonical_names(
    relative_path: str,
    stage: str,
    status: str,
) -> None:
    _path, config = load_json_object(PROJECT_ROOT / relative_path)
    method_id, display_name, runner = validate_method_config(config, stage=stage)

    assert display_name == METHOD_DISPLAY_NAMES[method_id]
    assert runner["status"] == status


def test_fixed_view_evaluation_has_three_config_driven_stages() -> None:
    _path, config = load_json_object(
        PROJECT_ROOT / "configs/evaluation/fixed_view_rerendering.json"
    )
    runners = validate_evaluation_config(
        config,
        selected_stages=("render", "compare", "aggregate"),
    )

    assert [stage for stage, _runner in runners] == ["render", "compare", "aggregate"]
    assert all(runner["status"] == "ready" for _stage, runner in runners)
    assert [pose["view_id"] for pose in config["camera"]["poses"]] == [
        "000",
        "001",
        "002",
        "003",
        "004",
        "005",
    ]
    assert set(config["view_groups"]) == {
        "all",
        "front",
        "conditioning",
        "non_front",
    }
    assert config["metrics"]["enabled"] == ["mae", "rmse", "psnr", "ssim_like"]
    for _stage, runner in runners:
        option_index = runner["arguments"].index("--evaluation-config")
        assert runner["arguments"][option_index + 1] == "{config}"


def test_evaluation_runner_rejects_missing_evaluation_config_argument() -> None:
    _path, config = load_json_object(
        PROJECT_ROOT / "configs/evaluation/fixed_view_rerendering.json"
    )
    config["stages"]["compare"]["runner"]["arguments"] = [
        "--cases-config",
        "{cases_config}",
        "--output-root",
        "{output_root}",
    ]

    with pytest.raises(WorkflowError, match="--evaluation-config"):
        validate_evaluation_config(config, selected_stages=("compare",))


def test_ready_entrypoints_exist() -> None:
    for path in PROJECT_ROOT.glob("configs/**/*.json"):
        config = json.loads(path.read_text(encoding="utf-8"))
        containers = []
        for stage in ("training", "inference"):
            value = config.get(stage)
            if isinstance(value, dict):
                containers.append(value)
        stages = config.get("stages")
        if isinstance(stages, dict):
            containers.extend(value for value in stages.values() if isinstance(value, dict))
        for container in containers:
            runner = container.get("runner")
            if isinstance(runner, dict) and runner.get("status") == "ready":
                assert (PROJECT_ROOT / runner["entrypoint"]).is_file()


def test_preview_preserves_missing_placeholders() -> None:
    _path, config = load_json_object(
        PROJECT_ROOT / "configs/baseline/corrected_conditioning_baseline.json"
    )
    _method_id, _display_name, runner = validate_method_config(config, stage="inference")

    command, working_directory = build_command(runner, {}, preview=True)

    assert "<CASE_DIR>" in command
    assert "<OUTPUT_DIR>" in command
    assert working_directory == PROJECT_ROOT


def test_runtime_command_requires_values() -> None:
    _path, config = load_json_object(
        PROJECT_ROOT / "configs/baseline/corrected_conditioning_baseline.json"
    )
    _method_id, _display_name, runner = validate_method_config(config, stage="inference")

    with pytest.raises(WorkflowError, match="--case-dir"):
        build_command(runner, {}, preview=False)


def test_planned_runner_refuses_execution() -> None:
    runner = {
        "status": "planned",
        "executor": "python",
        "entrypoint": "scripts/not_migrated.py",
        "arguments": [],
        "working_directory": ".",
    }

    with pytest.raises(WorkflowError, match="not marked ready"):
        execute_runner(
            runner=runner,
            command=["python", "scripts/not_migrated.py"],
            working_directory=PROJECT_ROOT,
        )


def test_ready_runner_requires_an_existing_entrypoint() -> None:
    with pytest.raises(WorkflowError, match="entrypoint does not exist"):
        validate_runner(
            {
                "status": "ready",
                "executor": "python",
                "entrypoint": "scripts/missing.py",
                "arguments": [],
                "working_directory": ".",
            },
            "test",
        )


def test_paper_protocol_constants_are_locked_in_configs() -> None:
    broad = json.loads(
        (PROJECT_ROOT / "configs/adaptation/protocol_corrected_broad_finetuning.json").read_text(
            encoding="utf-8"
        )
    )
    mva = json.loads(
        (PROJECT_ROOT / "configs/adaptation/protocol_corrected_mva_finetuning.json").read_text(
            encoding="utf-8"
        )
    )
    gating = json.loads(
        (PROJECT_ROOT / "configs/gating/view_selective_conditioning_gating.json").read_text(
            encoding="utf-8"
        )
    )

    expected_probabilities = {
        "005": 0.5,
        "004": 0.3,
        "000": 0.05,
        "001": 0.05,
        "002": 0.05,
        "003": 0.05,
    }
    expected_shared_protocol = {
        "image_size": 512,
        "base_seed": 42,
        "schedule_seed": 42,
        "num_workers": 0,
        "gradient_clip_norm": 1.0,
        "precision": "bf16",
        "checkpoint_steps": [160, 320],
        "log_every_n_steps": 10,
        "conditioning_dropout_policy": {
            "drop_cond_prob": 0.1,
            "require_multi_view_attention_active": True,
            "maximum_seed_retries": 128,
        },
        "reference_lighting_policy": {
            "available_conditions": ["AL", "ENVMAP", "PL"],
            "images_per_sample": 2,
            "selection": "ordered_uniform_without_replacement",
        },
    }
    for config in (broad, mva):
        for key, value in expected_shared_protocol.items():
            assert config["training"][key] == value
    assert broad["training"]["updates"] == 320
    assert broad["weights"]["trainable_scope"] == "broad_scope"
    assert broad["training"]["conditioning_view_probabilities"] == expected_probabilities
    assert mva["training"]["updates"] == 320
    assert mva["weights"]["trainable_scope"] == "multi_view_attention"
    assert mva["training"]["conditioning_view_probabilities"] == expected_probabilities
    assert gating["gating"]["suppression_threshold_degrees"] == 120
    assert gating["gating"]["keep_mask"] == [1, 1, 0, 1, 1, 1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/baseline/corrected_conditioning_baseline.json",
        "configs/adaptation/broad_scope_finetuning.json",
    ],
)
def test_ready_paint_inference_runners_forward_view_and_seed(relative_path: str) -> None:
    _path, config = load_json_object(PROJECT_ROOT / relative_path)
    _method_id, _display_name, runner = validate_method_config(config, stage="inference")

    view_index = runner["arguments"].index("--conditioning-view")
    seed_index = runner["arguments"].index("--seed")
    assert runner["arguments"][view_index + 1] == "005"
    assert runner["arguments"][seed_index + 1] == "0"


def _set_value(path: tuple[str, ...], value: object):
    def mutate(config: dict) -> None:
        target = config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _set_inference_argument(option: str, value: str):
    def mutate(config: dict) -> None:
        arguments = config["inference"]["runner"]["arguments"]
        arguments[arguments.index(option) + 1] = value

    return mutate


def _remove_inference_argument(option: str):
    def mutate(config: dict) -> None:
        config["inference"]["runner"]["arguments"].remove(option)

    return mutate


@pytest.mark.parametrize(
    ("relative_path", "stage", "mutate", "message"),
    [
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            "inference",
            _set_value(("category",), "adaptation"),
            "category does not match",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            "inference",
            _set_value(
                ("evaluation_config",),
                "configs/baseline/corrected_conditioning_baseline.json",
            ),
            "evaluation_config is invalid",
        ),
        (
            "configs/adaptation/broad_scope_finetuning.json",
            "inference",
            _set_value(("baseline_config",), "/tmp/baseline.json"),
            "project-relative",
        ),
        (
            "configs/adaptation/broad_scope_finetuning.json",
            "inference",
            _set_value(("training", "runner", "status"), "unknown"),
            "training.runner.status",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            "inference",
            _set_value(("weights", "training_updates"), 1),
            "untrained official weights",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            "inference",
            _set_value(("view_roles", "evaluation_views"), ["005"]),
            "canonical ordering",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            "inference",
            _set_value(("inference_protocol", "fixed_mesh"), False),
            "fixed input mesh",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            "inference",
            _set_value(("inference_protocol", "seed"), -1),
            "inference_protocol.seed",
        ),
        (
            "configs/adaptation/broad_scope_finetuning.json",
            "training",
            _set_value(("weights", "initialization"), "other"),
            "invalid initialization",
        ),
        (
            "configs/adaptation/broad_scope_finetuning.json",
            "training",
            _set_value(("weights", "trainable_scope"), "multi_view_attention"),
            "invalid trainable_scope",
        ),
        (
            "configs/gating/view_selective_conditioning_gating.json",
            "inference",
            _set_value(("weights", "checkpoint_loading"), True),
            "frozen official weights",
        ),
        (
            "configs/gating/view_selective_conditioning_gating.json",
            "inference",
            _set_value(("gating", "target_pathways"), ["reference_attention"]),
            "policy and pathways",
        ),
        (
            "configs/gating/view_selective_conditioning_gating.json",
            "inference",
            _set_value(("gating", "preserved_pathway"), "reference_attention"),
            "policy and pathways",
        ),
        (
            "configs/gating/view_selective_conditioning_gating.json",
            "inference",
            _set_value(("gating", "reference_slot"), 99),
            "reference slot or threshold",
        ),
        (
            "configs/gating/view_selective_conditioning_gating.json",
            "inference",
            _set_value(("gating", "keep_mask"), [1, 1, 1, 1, 1, 1]),
            "gating mask",
        ),
    ],
)
def test_method_descriptor_mutations_are_rejected(
    relative_path: str, stage: str, mutate, message: str
) -> None:
    _path, config = load_json_object(PROJECT_ROOT / relative_path)
    mutate(config)

    with pytest.raises(WorkflowError, match=message):
        validate_method_config(config, stage=stage)


@pytest.mark.parametrize(
    ("relative_path", "mutate", "message"),
    [
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            _set_inference_argument("--conditioning-view", "004"),
            "--conditioning-view",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            _set_inference_argument("--seed", "1"),
            "--seed",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            _set_inference_argument("--resolution", "256"),
            "--resolution",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            _set_inference_argument("--max-num-view", "5"),
            "--max-num-view",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            _remove_inference_argument("--no-remesh"),
            "--no-remesh",
        ),
        (
            "configs/baseline/corrected_conditioning_baseline.json",
            _set_inference_argument("--mode", "checkpoint"),
            "--mode",
        ),
        (
            "configs/adaptation/broad_scope_finetuning.json",
            _set_inference_argument("--checkpoint", "wrong.ckpt"),
            "--checkpoint",
        ),
    ],
)
def test_ready_paint_runner_mutations_are_rejected(
    relative_path: str, mutate, message: str
) -> None:
    _path, config = load_json_object(PROJECT_ROOT / relative_path)
    mutate(config)

    with pytest.raises(WorkflowError, match=message):
        validate_method_config(config, stage="inference")
