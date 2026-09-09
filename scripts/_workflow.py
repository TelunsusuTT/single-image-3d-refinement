#!/usr/bin/env python3
"""Shared validation and command construction for the public workflow CLIs."""

from __future__ import annotations

import json
import os
import shlex
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from _evaluation import VIEW_IDS, EvaluationContractError, validate_evaluation_config_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_DISPLAY_NAMES = {
    "corrected_conditioning_baseline": "Corrected-Conditioning Baseline",
    "broad_scope_finetuning": "Broad-Scope Fine-Tuning",
    "reference_conditioning_lora": "Reference-Conditioning LoRA",
    "protocol_corrected_broad_finetuning": "Protocol-Corrected Broad-Scope Fine-Tuning",
    "protocol_corrected_mva_finetuning": "Protocol-Corrected Multi-View-Attention Fine-Tuning",
    "view_selective_conditioning_gating": "View-Selective Conditioning Gating",
}

METHOD_CATEGORIES = {
    "corrected_conditioning_baseline": "baseline",
    "broad_scope_finetuning": "adaptation",
    "reference_conditioning_lora": "adaptation",
    "protocol_corrected_broad_finetuning": "adaptation",
    "protocol_corrected_mva_finetuning": "adaptation",
    "view_selective_conditioning_gating": "inference_control",
}
CATEGORY_STAGES = {
    "baseline": {"inference"},
    "adaptation": {"training", "inference"},
    "inference_control": {"inference"},
}
METHOD_STAGES = ("training", "inference")
GENERATION_SLOTS = ("front", "right", "back", "left", "top", "bottom")
OFFICIAL_INITIALIZATION = "official_hunyuan3d_paint_pbr"
FRESH_INITIALIZATION = "fresh_official_hunyuan3d_paint_pbr"
ADAPTATION_WEIGHTS = {
    "broad_scope_finetuning": (OFFICIAL_INITIALIZATION, "broad_scope", "lightning_full_unet_state_dict"),
    "reference_conditioning_lora": (OFFICIAL_INITIALIZATION, None, "adapter_only"),
    "protocol_corrected_broad_finetuning": (FRESH_INITIALIZATION, "broad_scope", "trainable_scope_checkpoint_v1"),
    "protocol_corrected_mva_finetuning": (FRESH_INITIALIZATION, "multi_view_attention", "trainable_scope_checkpoint_v1"),
}

RUNNER_STATUSES = {"planned", "ready"}
RUNNER_EXECUTORS = {"python", "blender_python"}
TEMPLATE_FIELDS = {
    "case_dir",
    "cases_config",
    "checkpoint",
    "config",
    "output_dir",
    "output_root",
    "run_id",
}


class WorkflowError(RuntimeError):
    """Raised when a public workflow configuration is invalid."""


def load_json_object(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise WorkflowError(f"configuration file does not exist: {source}") from None
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"configuration must contain a JSON object: {source}")
    return source, payload


def _require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{key} must be a non-empty string")
    return value


def _validate_relative_project_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise WorkflowError(f"{label} must be a project-relative path without '..': {value}")
    return path


def _template_fields(value: str) -> set[str]:
    fields: set[str] = set()
    try:
        parsed = string.Formatter().parse(value)
        for _, field_name, _, _ in parsed:
            if field_name:
                fields.add(field_name)
    except ValueError as exc:
        raise WorkflowError(f"invalid command template {value!r}: {exc}") from exc
    unknown = fields - TEMPLATE_FIELDS
    if unknown:
        raise WorkflowError(f"unsupported command placeholders: {sorted(unknown)}")
    return fields


def validate_runner(runner: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(runner, Mapping):
        raise WorkflowError(f"{label}.runner must be an object")

    status = _require_text(runner, "status")
    if status not in RUNNER_STATUSES:
        raise WorkflowError(f"{label}.runner.status must be one of {sorted(RUNNER_STATUSES)}")

    executor = _require_text(runner, "executor")
    if executor not in RUNNER_EXECUTORS:
        raise WorkflowError(f"{label}.runner.executor must be one of {sorted(RUNNER_EXECUTORS)}")

    entrypoint = _require_text(runner, "entrypoint")
    _validate_relative_project_path(entrypoint, f"{label}.runner.entrypoint")

    entrypoint_path = (PROJECT_ROOT / entrypoint).resolve()
    if status == "ready" and not entrypoint_path.is_file():
        raise WorkflowError(f"{label}.runner.entrypoint does not exist: {entrypoint}")

    arguments = runner.get("arguments")
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise WorkflowError(f"{label}.runner.arguments must be a list of strings")
    for argument in arguments:
        _template_fields(argument)

    working_directory = runner.get("working_directory", ".")
    if not isinstance(working_directory, str) or not working_directory:
        raise WorkflowError(f"{label}.runner.working_directory must be a non-empty string")
    _validate_relative_project_path(working_directory, f"{label}.runner.working_directory")

    return {
        "status": status,
        "executor": executor,
        "entrypoint": entrypoint,
        "arguments": arguments,
        "working_directory": working_directory,
    }


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowError(f"{label} must be an object")
    return value


def _project_json(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    relative = _validate_relative_project_path(_require_text(config, key), key)
    source = (PROJECT_ROOT / relative).resolve()
    try:
        source.relative_to(PROJECT_ROOT)
    except ValueError:
        raise WorkflowError(f"{key} resolves outside the project") from None
    if not source.is_file():
        raise WorkflowError(f"{key} does not exist: {relative}")
    return load_json_object(source)[1]


def _evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _project_json(config, "evaluation_config")
    try:
        validate_evaluation_config_payload(evaluation)
    except EvaluationContractError as exc:
        raise WorkflowError(f"evaluation_config is invalid: {exc}") from exc
    return evaluation


def _text_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise WorkflowError(f"{label} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise WorkflowError(f"{label} must not contain duplicates")
    return tuple(value)


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkflowError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_baseline(config: Mapping[str, Any], evaluation: Mapping[str, Any]) -> None:
    if "baseline_config" in config:
        raise WorkflowError("a baseline must not define baseline_config")
    weights = _object(config.get("weights"), "weights")
    training_updates = _positive_int(
        weights.get("training_updates"), "weights.training_updates", allow_zero=True)
    if (
        weights.get("initialization") != OFFICIAL_INITIALIZATION
        or training_updates != 0
    ):
        raise WorkflowError("baseline must use untrained official weights")

    roles = _object(config.get("view_roles"), "view_roles")
    slots = _text_list(roles.get("generation_view_slots"), "generation_view_slots")
    views = _text_list(roles.get("evaluation_views"), "evaluation_views")
    conditioning = roles.get("conditioning_view")
    if slots != GENERATION_SLOTS or views != VIEW_IDS:
        raise WorkflowError("baseline view lists must use the canonical ordering")
    if tuple(evaluation["view_groups"]["conditioning"]) != (conditioning,):
        raise WorkflowError("conditioning_view must match the evaluation protocol")

    protocol = _object(config.get("inference_protocol"), "inference_protocol")
    if protocol.get("fixed_mesh") is not True or protocol.get("use_remesh") is not False:
        raise WorkflowError("baseline inference must keep the fixed input mesh")
    seed = _positive_int(protocol.get("seed"), "inference_protocol.seed", allow_zero=True)
    if seed > 2**32 - 1:
        raise WorkflowError("inference_protocol.seed must be uint32")
    _positive_int(protocol.get("resolution"), "inference_protocol.resolution")
    max_views = _positive_int(protocol.get("max_num_view"), "inference_protocol.max_num_view")
    if max_views != len(slots):
        raise WorkflowError("max_num_view must match generation_view_slots")


def _validate_adaptation(config: Mapping[str, Any], method_id: str) -> None:
    weights = _object(config.get("weights"), "weights")
    expected_initialization, expected_scope, expected_format = ADAPTATION_WEIGHTS[method_id]
    if weights.get("initialization") != expected_initialization:
        raise WorkflowError(f"invalid initialization for {method_id}")
    if weights.get("checkpoint_format") != expected_format:
        raise WorkflowError(f"invalid checkpoint_format for {method_id}")
    if expected_scope is None:
        if "trainable_scope" in weights:
            raise WorkflowError(f"{method_id} must not define trainable_scope")
    elif weights.get("trainable_scope") != expected_scope:
        raise WorkflowError(f"invalid trainable_scope for {method_id}")


def _validate_gating(config: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    weights = _object(config.get("weights"), "weights")
    training_updates = _positive_int(
        weights.get("training_updates"), "weights.training_updates", allow_zero=True)
    if (
        weights.get("initialization") != OFFICIAL_INITIALIZATION
        or training_updates != 0
        or weights.get("checkpoint_loading") is not False
    ):
        raise WorkflowError("gating must keep frozen official weights")

    gating = _object(config.get("gating"), "gating")
    targets = set(_text_list(gating.get("target_pathways"), "gating.target_pathways"))
    preserved = gating.get("preserved_pathway")
    if (
        gating.get("policy") != "view_selective_gating"
        or targets != {"reference_attention", "dino_conditioning"}
        or preserved != "multi_view_attention"
        or preserved in targets
    ):
        raise WorkflowError("gating policy and pathways are inconsistent")
    slots = _text_list(gating.get("generation_view_slots"), "gating.generation_view_slots")
    if slots != tuple(baseline["view_roles"]["generation_view_slots"]):
        raise WorkflowError("gating view slots must match baseline_config")
    reference_slot = _positive_int(
        gating.get("reference_slot"), "gating.reference_slot", allow_zero=True
    )
    threshold = gating.get("suppression_threshold_degrees")
    if (
        reference_slot >= len(slots)
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < threshold <= 180
    ):
        raise WorkflowError("invalid gating reference slot or threshold")
    mask = gating.get("keep_mask")
    if (
        not isinstance(mask, list)
        or len(mask) != len(slots)
        or any(type(value) is not int or value not in {0, 1} for value in mask)
        or 0 not in mask
        or mask[reference_slot] != 1
        or type(gating.get("suppressed_value")) not in {int, float}
        or gating.get("suppressed_value") != 0
    ):
        raise WorkflowError("invalid gating mask or suppressed value")


def _option(runner: Mapping[str, Any], name: str, label: str) -> str:
    arguments = runner["arguments"]
    positions = [index for index, item in enumerate(arguments) if item == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise WorkflowError(f"{label} must pass {name} exactly once with a value")
    return arguments[positions[0] + 1]


def _validate_ready_paint_runner(
    runner: Mapping[str, Any], category: str, baseline: Mapping[str, Any], label: str
) -> None:
    if runner["status"] != "ready" or runner["entrypoint"] != "scripts/run_paint_inference.py":
        return
    protocol = baseline["inference_protocol"]
    expected = {
        "--conditioning-view": baseline["view_roles"]["conditioning_view"],
        "--seed": protocol["seed"],
        "--resolution": protocol["resolution"],
        "--max-num-view": protocol["max_num_view"],
    }
    for option, value in expected.items():
        if _option(runner, option, label) != str(value):
            raise WorkflowError(f"{label} {option} must match baseline_config")
    if runner["arguments"].count("--no-remesh") != int(not protocol["use_remesh"]):
        raise WorkflowError(f"{label} --no-remesh must match baseline_config")

    mode = "checkpoint" if category == "adaptation" else "baseline"
    if _option(runner, "--mode", label) != mode:
        raise WorkflowError(f"{label} --mode must be {mode}")
    has_checkpoint = "--checkpoint" in runner["arguments"]
    if mode == "checkpoint":
        if _option(runner, "--checkpoint", label) != "{checkpoint}":
            raise WorkflowError(f"{label} must pass --checkpoint {{checkpoint}}")
    elif has_checkpoint:
        raise WorkflowError(f"{label} baseline mode must not pass --checkpoint")


def _validate_method_descriptor(
    config: Mapping[str, Any],
) -> tuple[str, str, str, dict[str, dict[str, Any]]]:
    if config.get("schema_version") != 1:
        raise WorkflowError("schema_version must be 1")
    method_id = _require_text(config, "method_id")
    if method_id not in METHOD_DISPLAY_NAMES:
        raise WorkflowError(f"unknown method_id {method_id!r}")
    display_name = _require_text(config, "display_name")
    if display_name != METHOD_DISPLAY_NAMES[method_id]:
        raise WorkflowError(f"display_name does not match {method_id}")
    category = _require_text(config, "category")
    if category != METHOD_CATEGORIES[method_id]:
        raise WorkflowError(f"category does not match {method_id}")

    allowed = CATEGORY_STAGES[category]
    runners: dict[str, dict[str, Any]] = {}
    for candidate in METHOD_STAGES:
        stage_config = config.get(candidate)
        if stage_config is None:
            continue
        if candidate not in allowed:
            raise WorkflowError(f"category {category} does not allow {candidate}")
        runners[candidate] = validate_runner(
            _object(stage_config, candidate).get("runner"), candidate
        )
    if runners.keys() != allowed:
        raise WorkflowError(f"category {category} must define {sorted(allowed)}")

    evaluation = _evaluation(config)
    if category == "baseline":
        baseline = config
        _validate_baseline(config, evaluation)
    else:
        baseline = _project_json(config, "baseline_config")
        if baseline.get("method_id") != "corrected_conditioning_baseline":
            raise WorkflowError("baseline_config must reference corrected_conditioning_baseline")
        _validate_method_descriptor(baseline)
        if category == "adaptation":
            _validate_adaptation(config, method_id)
        else:
            _validate_gating(config, baseline)

    for runner_stage, runner in runners.items():
        _validate_ready_paint_runner(
            runner, category, baseline, f"{runner_stage}.runner"
        )
    return method_id, display_name, category, runners


def validate_method_config(
    config: Mapping[str, Any],
    *,
    stage: str,
) -> tuple[str, str, dict[str, Any]]:
    method_id, display_name, category, runners = _validate_method_descriptor(config)
    if stage not in CATEGORY_STAGES[category]:
        raise WorkflowError(f"{display_name} does not define a {stage} workflow")
    return method_id, display_name, runners[stage]


def validate_evaluation_config(
    config: Mapping[str, Any],
    *,
    selected_stages: Sequence[str],
) -> list[tuple[str, dict[str, Any]]]:
    try:
        validate_evaluation_config_payload(config)
    except EvaluationContractError as exc:
        raise WorkflowError(str(exc)) from exc

    stages = config["stages"]
    validated: list[tuple[str, dict[str, Any]]] = []
    for stage in selected_stages:
        stage_config = stages.get(stage)
        if not isinstance(stage_config, Mapping):
            raise WorkflowError(f"evaluation stage {stage!r} is missing")
        runner = validate_runner(stage_config.get("runner"), f"stages.{stage}")
        arguments = runner["arguments"]
        try:
            option_index = arguments.index("--evaluation-config")
        except ValueError:
            raise WorkflowError(
                f"stages.{stage}.runner must pass --evaluation-config"
            ) from None
        if (
            option_index + 1 >= len(arguments)
            or arguments[option_index + 1] != "{config}"
        ):
            raise WorkflowError(
                f"stages.{stage}.runner must pass --evaluation-config {{config}}"
            )
        validated.append((stage, runner))
    return validated


class _PreviewValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return f"<{key.upper()}>"


def _render_arguments(
    arguments: Sequence[str],
    values: Mapping[str, str | None],
    *,
    preview: bool,
) -> list[str]:
    rendered_values: dict[str, str] = {
        key: str(value) for key, value in values.items() if value is not None
    }
    if preview:
        format_values: Mapping[str, str] = _PreviewValues(rendered_values)
    else:
        missing: set[str] = set()
        for argument in arguments:
            for field in _template_fields(argument):
                if not rendered_values.get(field):
                    missing.add(field)
        if missing:
            options = ", ".join(f"--{name.replace('_', '-')}" for name in sorted(missing))
            raise WorkflowError(f"runtime command requires: {options}")
        format_values = rendered_values
    return [argument.format_map(format_values) for argument in arguments]


def build_command(
    runner: Mapping[str, Any],
    values: Mapping[str, str | None],
    *,
    preview: bool,
) -> tuple[list[str], Path]:
    entrypoint = (PROJECT_ROOT / runner["entrypoint"]).resolve()
    arguments = _render_arguments(runner["arguments"], values, preview=preview)

    if runner["executor"] == "python":
        command = [os.environ.get("PYTHON_BIN", sys.executable), str(entrypoint), *arguments]
    else:
        blender = os.environ.get("BLENDER_BIN", "blender")
        command = [blender, "--background", "--python", str(entrypoint), "--", *arguments]

    working_directory = (PROJECT_ROOT / runner["working_directory"]).resolve()
    return command, working_directory


def print_runner_summary(
    *,
    label: str,
    runner: Mapping[str, Any],
    command: Sequence[str],
    working_directory: Path,
) -> None:
    print(f"{label}: status={runner['status']} executor={runner['executor']}")
    print(f"working_directory: {working_directory}")
    print(f"command: {shlex.join(command)}")


def execute_runner(
    *,
    runner: Mapping[str, Any],
    command: Sequence[str],
    working_directory: Path,
) -> None:
    if runner["status"] != "ready":
        raise WorkflowError(
            "runtime backend is not marked ready; migrate and verify the canonical entrypoint first"
        )
    entrypoint = (PROJECT_ROOT / runner["entrypoint"]).resolve()
    if not entrypoint.is_file():
        raise WorkflowError(f"runtime entrypoint does not exist: {entrypoint}")
    if not working_directory.is_dir():
        raise WorkflowError(f"runtime working directory does not exist: {working_directory}")
    subprocess.run(list(command), cwd=working_directory, check=True)
