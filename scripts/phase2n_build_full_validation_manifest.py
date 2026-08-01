#!/usr/bin/env python3
"""Resolve the Phase 2N ten-asset validation matrix without running models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PHASE = "phase2n_full_validation_manifest"
VIEW_IDS = ("000", "001", "002", "003", "004", "005")
VARIANT_ORDER = (
    "corrected_input_base",
    "historical_full80_500",
    "pc_s1_step160",
    "pc_full_step320",
)
CANDIDATE_VARIANTS = ("pc_s1_step160", "pc_full_step320")
EXPECTED_COUNTS = {
    "validation_assets": 10,
    "pilot_validation_assets": 6,
    "remaining_validation_assets": 4,
    "test_assets": 0,
    "train_sanity_assets": 0,
    "variants": 4,
    "source_glbs": 40,
    "reused_glbs": 32,
    "new_inference_glbs": 8,
    "planned_renders": 240,
}
EXPECTED_CONFIG_KEYS = {
    "phase",
    "full101_split",
    "pilot_cases_config",
    "historical_eval_cases",
    "historical_base_root",
    "historical_full80_root",
    "historical_full80_checkpoint",
    "pilot_inference_run",
    "new_inference_run",
    "training_run_dir",
    "output_manifest",
    "selected_input_view",
    "reference_lighting",
    "view_ids",
    "variant_order",
    "candidate_variants",
    "expected_counts",
}


class ManifestError(RuntimeError):
    """Raised when the full-validation source matrix is not exact."""


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"expected JSON object: {path}")
    return value


def require_nonempty(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ManifestError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ManifestError(f"empty {label}: {path}")
    return path


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).expanduser().resolve() == expected.resolve()


def validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != EXPECTED_CONFIG_KEYS:
        raise ManifestError("manifest config fields differ from the frozen schema")
    if config.get("phase") != EXPECTED_PHASE:
        raise ManifestError(f"phase must be {EXPECTED_PHASE}")
    if config.get("selected_input_view") != "005":
        raise ManifestError("selected_input_view must be 005")
    if config.get("reference_lighting") != "AL":
        raise ManifestError("reference_lighting must be AL")
    if config.get("view_ids") != list(VIEW_IDS):
        raise ManifestError("view_ids must be exactly 000 through 005")
    if config.get("variant_order") != list(VARIANT_ORDER):
        raise ManifestError("variant_order differs from the four frozen variants")
    if list(config.get("candidate_variants", {})) != list(CANDIDATE_VARIANTS):
        raise ManifestError("candidate_variants must be PC-S1-160 then PC-Full-320")
    if config.get("expected_counts") != EXPECTED_COUNTS:
        raise ManifestError("expected_counts differ from the frozen 10 x 4 x 6 plan")


def _historical_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ManifestError("historical eval cases has no cases list")
    indexed = {
        str(row.get("item_id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("item_id"), str)
    }
    if len(indexed) != len(rows):
        raise ManifestError("historical eval cases contain invalid or duplicate item IDs")
    return indexed


def _validate_historical_run_plan(
    run_plan_path: Path,
    *,
    mode: str,
    mesh: Path,
    input_image: Path,
    output_glb: Path,
    checkpoint: Path | None,
) -> None:
    plan = load_json(require_nonempty(run_plan_path, "historical run plan"))
    expected = {
        "mode": mode,
        "max_num_view": 6,
        "resolution": 512,
        "use_remesh": False,
        "dry_run": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ManifestError(f"historical run plan {key} mismatch: {run_plan_path}")
    for key, value in (("input_mesh", mesh), ("input_image", input_image)):
        if not _same_path(plan.get(key), value):
            raise ManifestError(f"historical run plan {key} mismatch: {run_plan_path}")
    if not _same_path(plan.get("planned_output_glb"), output_glb):
        raise ManifestError(f"historical planned GLB mismatch: {run_plan_path}")
    recorded_checkpoint = plan.get("checkpoint")
    if checkpoint is None:
        if recorded_checkpoint not in {None, ""}:
            raise ManifestError(f"base run unexpectedly records a checkpoint: {run_plan_path}")
    elif not _same_path(recorded_checkpoint, checkpoint):
        raise ManifestError(f"historical checkpoint mismatch: {run_plan_path}")


def _validate_candidate_manifest(
    manifest_path: Path,
    *,
    asset_id: str,
    variant_id: str,
    output_glb: Path,
    checkpoint_manifest: Path,
) -> None:
    manifest = load_json(require_nonempty(manifest_path, "candidate inference manifest"))
    expected = {
        "status": "OK",
        "asset_id": asset_id,
        "variant": variant_id,
        "eval_split": "val",
        "source_split": "val",
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "test_data_used": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ManifestError(f"candidate manifest {key} mismatch: {manifest_path}")
    if not _same_path(manifest.get("output_glb_path"), output_glb):
        raise ManifestError(f"candidate output path mismatch: {manifest_path}")
    exact = manifest.get("exact_inference_settings")
    if not isinstance(exact, dict) or exact.get("use_remesh") is not False:
        raise ManifestError(f"candidate manifest does not prove fixed mesh: {manifest_path}")
    if exact.get("selected_input_view") != "005" or exact.get("save_glb") is not True:
        raise ManifestError(f"candidate inference settings mismatch: {manifest_path}")
    require_nonempty(checkpoint_manifest, f"{variant_id} checkpoint manifest")


def _source_record(
    *,
    asset_id: str,
    variant_id: str,
    source_path: Path,
    source_run: Path,
    source_manifest: Path,
    reuse_status: str,
    mesh_path: Path,
    reference_paths: Mapping[str, str],
    checkpoint_manifest: Path | None,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "eval_split": "val",
        "variant_id": variant_id,
        "source_path": str(source_path),
        "source_run": str(source_run),
        "source_manifest": str(source_manifest),
        "selected_input_view": "005",
        "mesh_path": str(mesh_path),
        "reference_paths": dict(reference_paths),
        "reuse_status": reuse_status,
        "checkpoint_manifest": str(checkpoint_manifest) if checkpoint_manifest else None,
    }


def build_manifest(
    config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    *,
    require_new_outputs: bool = False,
    validate_files: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = resolve_project_path(config_path, root)
    config = load_json(path)
    validate_config(config)

    split_path = resolve_project_path(config["full101_split"], root)
    pilot_path = resolve_project_path(config["pilot_cases_config"], root)
    historical_path = resolve_project_path(config["historical_eval_cases"], root)
    base_root = resolve_project_path(config["historical_base_root"], root)
    full80_root = resolve_project_path(config["historical_full80_root"], root)
    full80_checkpoint = resolve_project_path(config["historical_full80_checkpoint"], root)
    pilot_run = resolve_project_path(config["pilot_inference_run"], root)
    new_run = resolve_project_path(config["new_inference_run"], root)
    training_run = resolve_project_path(config["training_run_dir"], root)
    output_manifest = resolve_project_path(config["output_manifest"], root)

    allowed_output = (root / "outputs" / "phase2n").resolve()
    if not is_within(output_manifest, allowed_output):
        raise ManifestError(f"output_manifest is outside outputs/phase2n: {output_manifest}")
    expected_new_run = (
        root
        / "outputs/phase2n/full_validation_inference/phase2n_full_validation_infer_v1"
    ).resolve()
    if new_run != expected_new_run:
        raise ManifestError(f"new inference run must be exactly {expected_new_run}")
    if validate_files:
        require_nonempty(full80_checkpoint, "historical full80 checkpoint")
        require_nonempty(pilot_run / "_SUCCESS", "pilot inference success marker")
        for variant_id in CANDIDATE_VARIANTS:
            require_nonempty(
                pilot_run / variant_id / "_SUCCESS",
                f"pilot {variant_id} success marker",
            )

    split = load_json(split_path)
    splits = split.get("splits")
    if not isinstance(splits, dict):
        raise ManifestError("full101 split has no splits object")
    raw_val = splits.get("val")
    if not isinstance(raw_val, list) or len(raw_val) != 10:
        raise ManifestError("canonical full101 validation split must contain exactly ten assets")
    validation_ids = [str(row.get("item_id")) for row in raw_val if isinstance(row, dict)]
    if len(validation_ids) != 10 or len(set(validation_ids)) != 10:
        raise ManifestError("canonical validation IDs are invalid or duplicated")
    if split.get("counts") != {"train": 80, "val": 10, "test": 11}:
        raise ManifestError("full101 declared split counts are not 80/10/11")

    pilot = load_json(pilot_path)
    pilot_rows = pilot.get("cases")
    if not isinstance(pilot_rows, list):
        raise ManifestError("pilot case config has no cases list")
    if any(
        isinstance(row, dict) and (row.get("eval_split") == "test" or row.get("source_split") == "test")
        for row in pilot_rows
    ):
        raise ManifestError("pilot case config unexpectedly contains test data")
    pilot_validation_ids = [
        str(row.get("asset_id"))
        for row in pilot_rows
        if isinstance(row, dict) and row.get("eval_split") == "val"
    ]
    if len(pilot_validation_ids) != 6 or len(set(pilot_validation_ids)) != 6:
        raise ManifestError("pilot must contain exactly six unique validation assets")
    validation_set = set(validation_ids)
    pilot_set = set(pilot_validation_ids)
    if not pilot_set <= validation_set:
        raise ManifestError("pilot validation assets are not a subset of full101 validation")
    remaining_ids = [asset_id for asset_id in validation_ids if asset_id not in pilot_set]
    remaining_set = set(remaining_ids)
    if len(remaining_ids) != 4 or pilot_set & remaining_set:
        raise ManifestError("pilot/remaining validation partition is not exact 6/4")
    if pilot_set | remaining_set != validation_set:
        raise ManifestError("pilot and remaining validation union is not the canonical split")

    historical = _historical_cases(historical_path)
    split_by_id = {str(row["item_id"]): row for row in raw_val}
    cases: list[dict[str, Any]] = []
    for asset_id in validation_ids:
        split_row = split_by_id[asset_id]
        if str(split_row.get("selected_input_view")) != "005":
            raise ManifestError(f"canonical selected input view is not 005 for {asset_id}")
        historical_row = historical.get(asset_id)
        if historical_row is None:
            raise ManifestError(f"historical eval metadata is missing validation asset {asset_id}")
        if historical_row.get("eval_split") != "val" or historical_row.get("source_split") != "val":
            raise ManifestError(f"historical split mismatch for {asset_id}")
        if historical_row.get("selected_input_view") != "005":
            raise ManifestError(f"historical selected input view is not 005 for {asset_id}")
        mesh = resolve_project_path(split_row.get("local_glb_path", ""), root)
        if not _same_path(historical_row.get("local_mesh_path"), mesh):
            raise ManifestError(f"historical mesh mismatch for {asset_id}")
        raw_references = historical_row.get("reference_images")
        if not isinstance(raw_references, dict) or list(raw_references) != list(VIEW_IDS):
            raise ManifestError(f"references are not exactly 000-005 for {asset_id}")
        references = {
            view_id: str(resolve_project_path(raw_references[view_id], root))
            for view_id in VIEW_IDS
        }
        selected_reference = Path(references["005"])
        if selected_reference.name != "005_light_AL.png":
            raise ManifestError(f"selected reference is not 005_light_AL.png for {asset_id}")
        if validate_files:
            require_nonempty(mesh, f"{asset_id} mesh")
            for view_id, reference in references.items():
                require_nonempty(Path(reference), f"{asset_id} reference {view_id}")
        cases.append(
            {
                "asset_id": asset_id,
                "source_split": "val",
                "eval_split": "val",
                "pilot_validation_reuse": asset_id in pilot_set,
                "selected_input_view": "005",
                "reference_lighting": "AL",
                "mesh_path": str(mesh),
                "input_image_path": references["005"],
                "reference_image_path": references["005"],
                "reference_paths": references,
            }
        )

    candidate_specs = config["candidate_variants"]
    resolved_candidate_specs: dict[str, dict[str, Path]] = {}
    for variant_id in CANDIDATE_VARIANTS:
        raw = candidate_specs.get(variant_id)
        if not isinstance(raw, dict) or set(raw) != {
            "scope",
            "step",
            "checkpoint_manifest_path",
        }:
            raise ManifestError(f"invalid candidate spec for {variant_id}")
        expected_scope = "pc_s1" if variant_id == "pc_s1_step160" else "pc_full"
        expected_step = 160 if variant_id == "pc_s1_step160" else 320
        if raw.get("scope") != expected_scope or raw.get("step") != expected_step:
            raise ManifestError(f"candidate scope/step mismatch for {variant_id}")
        checkpoint_manifest = resolve_project_path(raw["checkpoint_manifest_path"], root)
        expected_manifest = (
            training_run
            / expected_scope
            / "checkpoints"
            / f"step_{expected_step}_manifest.json"
        ).resolve()
        if checkpoint_manifest != expected_manifest:
            raise ManifestError(f"candidate checkpoint manifest mismatch for {variant_id}")
        if validate_files:
            require_nonempty(checkpoint_manifest, f"{variant_id} checkpoint manifest")
        resolved_candidate_specs[variant_id] = {
            "checkpoint_manifest": checkpoint_manifest,
        }

    if require_new_outputs and validate_files:
        root_marker = require_nonempty(
            new_run / "_SUCCESS", "full-validation inference success marker"
        )
        if "PHASE2N_FULL_VALIDATION_INFERENCE_OK" not in root_marker.read_text(encoding="utf-8"):
            raise ManifestError(f"invalid full-validation inference success marker: {root_marker}")
        marker_tokens = {
            "pc_s1_step160": "PHASE2N_FULL_VALIDATION_PC_S1_STEP160_INFERENCE_OK",
            "pc_full_step320": "PHASE2N_FULL_VALIDATION_PC_FULL_STEP320_INFERENCE_OK",
        }
        for variant_id, token in marker_tokens.items():
            marker = require_nonempty(
                new_run / variant_id / "_SUCCESS",
                f"full-validation {variant_id} success marker",
            )
            if token not in marker.read_text(encoding="utf-8"):
                raise ManifestError(f"invalid full-validation success marker: {marker}")

    source_records: list[dict[str, Any]] = []
    for case in cases:
        asset_id = case["asset_id"]
        mesh = Path(case["mesh_path"])
        references = case["reference_paths"]
        for variant_id in VARIANT_ORDER:
            checkpoint_manifest: Path | None = None
            if variant_id == "corrected_input_base":
                source_path = base_root / "val" / asset_id / "base_textured_mesh.glb"
                source_manifest = base_root / "val" / asset_id / "run_plan.json"
                source_run = base_root.parent.parent
                reuse_status = "reused_phase2l"
                if validate_files:
                    require_nonempty(source_path, f"base GLB for {asset_id}")
                    _validate_historical_run_plan(
                        source_manifest,
                        mode="base",
                        mesh=mesh,
                        input_image=Path(references["005"]),
                        output_glb=source_path,
                        checkpoint=None,
                    )
            elif variant_id == "historical_full80_500":
                source_path = full80_root / "val" / asset_id / "finetuned_textured_mesh.glb"
                source_manifest = full80_root / "val" / asset_id / "run_plan.json"
                source_run = full80_root.parent.parent
                reuse_status = "reused_phase2l"
                if validate_files:
                    require_nonempty(source_path, f"full80 GLB for {asset_id}")
                    _validate_historical_run_plan(
                        source_manifest,
                        mode="finetuned",
                        mesh=mesh,
                        input_image=Path(references["005"]),
                        output_glb=source_path,
                        checkpoint=full80_checkpoint,
                    )
            else:
                checkpoint_manifest = resolved_candidate_specs[variant_id]["checkpoint_manifest"]
                if asset_id in pilot_set:
                    source_run = pilot_run
                    reuse_status = "reused_phase2n_pilot"
                else:
                    source_run = new_run
                    reuse_status = "new_inference_required"
                source_path = source_run / variant_id / asset_id / "textured_mesh.glb"
                source_manifest = source_run / variant_id / asset_id / "inference_manifest.json"
                should_validate = validate_files and (
                    asset_id in pilot_set or require_new_outputs
                )
                if should_validate:
                    require_nonempty(source_path, f"{variant_id} GLB for {asset_id}")
                    _validate_candidate_manifest(
                        source_manifest,
                        asset_id=asset_id,
                        variant_id=variant_id,
                        output_glb=source_path,
                        checkpoint_manifest=checkpoint_manifest,
                    )
            source_records.append(
                _source_record(
                    asset_id=asset_id,
                    variant_id=variant_id,
                    source_path=source_path.resolve(),
                    source_run=source_run.resolve(),
                    source_manifest=source_manifest.resolve(),
                    reuse_status=reuse_status,
                    mesh_path=mesh,
                    reference_paths=references,
                    checkpoint_manifest=checkpoint_manifest,
                )
            )

    reuse_counts = {
        "reused_phase2l": sum(row["reuse_status"] == "reused_phase2l" for row in source_records),
        "reused_phase2n_pilot": sum(
            row["reuse_status"] == "reused_phase2n_pilot" for row in source_records
        ),
        "new_inference_required": sum(
            row["reuse_status"] == "new_inference_required" for row in source_records
        ),
    }
    actual_counts = {
        "validation_assets": len(cases),
        "pilot_validation_assets": len(pilot_validation_ids),
        "remaining_validation_assets": len(remaining_ids),
        "test_assets": 0,
        "train_sanity_assets": 0,
        "variants": len(VARIANT_ORDER),
        "source_glbs": len(source_records),
        "reused_glbs": reuse_counts["reused_phase2l"] + reuse_counts["reused_phase2n_pilot"],
        "new_inference_glbs": reuse_counts["new_inference_required"],
        "planned_renders": len(source_records) * len(VIEW_IDS),
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ManifestError(f"resolved counts {actual_counts} differ from {EXPECTED_COUNTS}")

    return {
        "phase": EXPECTED_PHASE,
        "status": "OK",
        "config_path": str(path),
        "canonical_split_path": str(split_path),
        "pilot_cases_path": str(pilot_path),
        "historical_eval_cases_path": str(historical_path),
        "output_manifest_path": str(output_manifest),
        "validation_ids": validation_ids,
        "pilot_validation_ids": pilot_validation_ids,
        "remaining_validation_ids": remaining_ids,
        "selected_input_view": "005",
        "reference_lighting": "AL",
        "view_ids": list(VIEW_IDS),
        "variant_order": list(VARIANT_ORDER),
        "candidate_variants": list(CANDIDATE_VARIANTS),
        "split_counts": {"val": 10, "train_sanity": 0, "test": 0},
        "counts": actual_counts,
        "reuse_counts": reuse_counts,
        "new_outputs_validated": bool(require_new_outputs),
        "test_data_used": False,
        "cases": cases,
        "source_records": source_records,
    }


def write_manifest(manifest: Mapping[str, Any]) -> Path:
    output = Path(str(manifest["output_manifest_path"]))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary manifest already exists: {temporary}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def print_summary(manifest: Mapping[str, Any]) -> None:
    print("validation_ids=" + ",".join(manifest["validation_ids"]))
    print("pilot_validation_ids=" + ",".join(manifest["pilot_validation_ids"]))
    print("remaining_validation_ids=" + ",".join(manifest["remaining_validation_ids"]))
    counts = manifest["counts"]
    print(
        f"reused_glbs={counts['reused_glbs']} "
        f"new_inference_glbs={counts['new_inference_glbs']} "
        f"source_glbs={counts['source_glbs']} planned_renders={counts['planned_renders']}"
    )
    print("PHASE2N_FULL_VALIDATION_SPLIT_OK")
    print("PHASE2N_FULL_VALIDATION_REUSE_OK")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or write the deduplicated Phase 2N full-validation source manifest."
    )
    parser.add_argument("--config", required=True, help="Full-validation manifest config")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="Validate without writing")
    modes.add_argument("--write", action="store_true", help="Write the resolved manifest once")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.config)
    print_summary(manifest)
    if args.write:
        print(f"wrote_manifest={write_manifest(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
