#!/usr/bin/env python3
"""Check Phase 2G.7 training-target diagnostic readiness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


OUTPUT_MAPS = {
    "base_albedo": "base_textured_mesh.jpg",
    "base_metallic": "base_textured_mesh_metallic.jpg",
    "base_roughness": "base_textured_mesh_roughness.jpg",
    "finetuned_albedo": "finetuned_textured_mesh.jpg",
    "finetuned_metallic": "finetuned_textured_mesh_metallic.jpg",
    "finetuned_roughness": "finetuned_textured_mesh_roughness.jpg",
}


def check_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    size = resolved.stat().st_size if exists else 0
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    elif size <= 0:
        errors.append(f"{label} is zero-size: {resolved}")
    return {"path": str(resolved), "exists": exists, "size_bytes": size}


def check_dir(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_dir()
    if not exists:
        errors.append(f"{label} missing: {resolved}")
    return {"path": str(resolved), "exists": exists}


def check_output_dir(output_dir: Path, errors: list[str]) -> dict[str, Any]:
    resolved = output_dir.expanduser().resolve()
    parent = resolved.parent
    ok = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        ok = parent.is_dir()
    except OSError as exc:
        errors.append(f"output-dir parent cannot be created: {parent}: {exc}")
    return {"path": str(resolved), "parent": str(parent), "parent_exists_or_created": ok}


def read_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_sample_dir(row: dict[str, str], dataset_root: Path) -> Path:
    raw = row.get("sample_dir") or row.get("sample_path") or ""
    if raw:
        path = Path(raw)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
    sample_name = row.get("sample_name") or row.get("source_id") or row.get("candidate_id") or ""
    return (dataset_root / sample_name).resolve()


def sample_checks(rows: list[dict[str, str]], dataset_root: Path, errors: list[str]) -> list[dict[str, Any]]:
    checks = []
    for index, row in enumerate(rows):
        sample_dir = resolve_sample_dir(row, dataset_root)
        render_tex = sample_dir / "render_tex"
        mr_files = sorted(render_tex.glob("*_mr.png")) if render_tex.is_dir() else []
        albedo_files = sorted(render_tex.glob("*_albedo.png")) if render_tex.is_dir() else []
        if not sample_dir.is_dir():
            errors.append(f"sample_dir missing for manifest row {index}: {sample_dir}")
        if not render_tex.is_dir():
            errors.append(f"render_tex missing for manifest row {index}: {render_tex}")
        if not mr_files:
            errors.append(f"no *_mr.png files for manifest row {index}: {render_tex}")
        if not albedo_files:
            errors.append(f"no *_albedo.png files for manifest row {index}: {render_tex}")
        checks.append(
            {
                "row_index": index,
                "source_id": row.get("source_id", ""),
                "sample_name": row.get("sample_name", ""),
                "sample_dir": str(sample_dir),
                "render_tex_exists": render_tex.is_dir(),
                "mr_count": len(mr_files),
                "albedo_count": len(albedo_files),
            }
        )
    return checks


def check_readiness(
    dataset_root: Path,
    manifest_csv: Path,
    base_dir: Path,
    finetuned_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    resolved_dataset = dataset_root.expanduser().resolve()
    resolved_manifest = manifest_csv.expanduser().resolve()
    resolved_base = base_dir.expanduser().resolve()
    resolved_finetuned = finetuned_dir.expanduser().resolve()

    checks: dict[str, Any] = {
        "dataset_root": check_dir(resolved_dataset, "dataset-root", errors),
        "manifest_csv": check_file(resolved_manifest, "manifest csv", errors),
        "base_dir": check_dir(resolved_base, "base-dir", errors),
        "finetuned_dir": check_dir(resolved_finetuned, "finetuned-dir", errors),
    }

    rows: list[dict[str, str]] = []
    if resolved_manifest.is_file() and resolved_manifest.stat().st_size > 0:
        try:
            rows = read_manifest(resolved_manifest)
        except csv.Error as exc:
            errors.append(f"manifest csv parse failed: {exc}")
    if not rows:
        errors.append(f"manifest csv has no rows: {resolved_manifest}")
    samples = sample_checks(rows, resolved_dataset, errors)

    for label, filename in OUTPUT_MAPS.items():
        root = resolved_base if label.startswith("base_") else resolved_finetuned
        checks[label] = check_file(root / filename, label, errors)

    output = check_output_dir(output_dir, errors)
    return {
        "dataset_root": str(resolved_dataset),
        "manifest_csv": str(resolved_manifest),
        "base_dir": str(resolved_base),
        "finetuned_dir": str(resolved_finetuned),
        "output_dir": output,
        "checks": checks,
        "samples": samples,
        "ok": not errors,
        "errors": errors,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Phase 2G.7 training-target diagnostic readiness")
    print(f"  dataset_root: {report['dataset_root']}")
    print(f"  manifest_csv: {report['manifest_csv']}")
    print(f"  base_dir: {report['base_dir']}")
    print(f"  finetuned_dir: {report['finetuned_dir']}")
    output = report["output_dir"]
    print(f"  output_dir: {output['path']}")
    print(f"  output_parent_exists_or_created: {output['parent_exists_or_created']}")
    print(f"  sample_count: {len(report['samples'])}")
    for name, item in report["checks"].items():
        if "size_bytes" in item:
            print(f"  {name}: exists={item['exists']} size_bytes={item['size_bytes']} path={item['path']}")
        else:
            print(f"  {name}: exists={item['exists']} path={item['path']}")
    for sample in report["samples"]:
        print(
            "  sample: "
            f"source_id={sample['source_id']} sample_dir={sample['sample_dir']} "
            f"mr_count={sample['mr_count']} albedo_count={sample['albedo_count']}"
        )
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")
    print(f"  status: {'OK' if report['ok'] else 'FAIL'}")
    if report["ok"]:
        print("PHASE2G7_TARGET_DIAGNOSTIC_PREFLIGHT_OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2G.7 target diagnostic readiness.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--finetuned-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_readiness(args.dataset_root, args.manifest_csv, args.base_dir, args.finetuned_dir, args.output_dir)
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
