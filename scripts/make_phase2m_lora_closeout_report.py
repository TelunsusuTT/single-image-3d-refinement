#!/usr/bin/env python3
"""Create the Phase 2M LoRA negative-result closeout report packet."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_SUMMARY = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300" / "training_summary.json"
DEFAULT_ADAPTER_CONFIG = PROJECT_ROOT / "outputs" / "phase2m" / "lora_train_refdino_r4_lr5e5_300" / "adapter_config.json"
DEFAULT_PILOT_SUMMARY = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot" / "pilot_summary.json"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_eval"
DEFAULT_AGGREGATE_SUMMARY = DEFAULT_EVAL_ROOT / "aggregate_summary.json"
DEFAULT_SCALE_COMPARISON = DEFAULT_EVAL_ROOT / "scale_comparison.json"
DEFAULT_METRICS_ROWS = DEFAULT_EVAL_ROOT / "metrics_rows.csv"
DEFAULT_BOARDS_ROOT = DEFAULT_EVAL_ROOT / "boards"
DEFAULT_REPORT_PACKET = DEFAULT_EVAL_ROOT / "report_packet"
DEFAULT_DOC_REPORT = PROJECT_ROOT / "docs" / "phase2m_lora_negative_result_report.md"
LORA_VARIANTS = ["lora_scale050", "lora_scale075", "lora_scale100"]
VARIANT_LABELS = {
    "base": "Base",
    "lora_scale050": "LoRA 0.50",
    "lora_scale075": "LoRA 0.75",
    "lora_scale100": "LoRA 1.00",
}


def resolve_project_path(path_text: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def ensure_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    allowed = root.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"{label} must be under {allowed}, got {resolved}")
    return resolved


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def adapter_size_mb(training: dict[str, Any]) -> float:
    bytes_value = training.get("final_adapter_bytes_estimate")
    if not isinstance(bytes_value, (int, float)):
        paths = training.get("saved_adapter_paths") or []
        if paths:
            candidate = Path(paths[-1])
            if candidate.is_file():
                bytes_value = candidate.stat().st_size
    return float(bytes_value or 0) / 1_000_000.0


def target_module_families(adapter_config: dict[str, Any]) -> list[str]:
    names = adapter_config.get("target_names") or []
    families = []
    if any("attn_refview" in str(name) for name in names):
        families.append("attn_refview")
    if any("attn_dino" in str(name) for name in names):
        families.append("attn_dino")
    return families


def variant_data(aggregate: dict[str, Any], group: str, variant: str) -> dict[str, Any]:
    return aggregate.get("by_view_group", {}).get(group, {}).get(variant, {})


def split_data(aggregate: dict[str, Any], split: str, variant: str) -> dict[str, Any]:
    return aggregate.get("by_split", {}).get(split, {}).get(variant, {})


def variant_table_rows(aggregate: dict[str, Any], group: str, variants: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for variant in variants or ["base", *LORA_VARIANTS]:
        data = variant_data(aggregate, group, variant)
        rows.append(
            {
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "view_count": data.get("view_count"),
                "base_mae": data.get("base_mean_mae"),
                "variant_mae": data.get("variant_mean_mae"),
                "mae_delta": data.get("variant_minus_base_mean_mae"),
                "base_ssim": data.get("base_mean_ssim_like"),
                "variant_ssim": data.get("variant_mean_ssim_like"),
                "ssim_delta": data.get("variant_minus_base_mean_ssim_like"),
                "improved_mae_views": data.get("views_where_variant_improves_mae"),
            }
        )
    return rows


def split_table_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in ["val", "test", "train_sanity"]:
        for variant in LORA_VARIANTS:
            data = split_data(aggregate, split, variant)
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "label": VARIANT_LABELS.get(variant, variant),
                    "view_count": data.get("view_count"),
                    "base_mae": data.get("base_mean_mae"),
                    "variant_mae": data.get("variant_mean_mae"),
                    "mae_delta": data.get("variant_minus_base_mean_mae"),
                    "ssim_delta": data.get("variant_minus_base_mean_ssim_like"),
                    "improved_mae_views": data.get("views_where_variant_improves_mae"),
                }
            )
    return rows


def heldout_and_front_degrade(aggregate: dict[str, Any]) -> bool:
    for variant in LORA_VARIANTS:
        front_delta = variant_data(aggregate, "front_views_004_005", variant).get("variant_minus_base_mean_mae")
        val_delta = split_data(aggregate, "val", variant).get("variant_minus_base_mean_mae")
        test_delta = split_data(aggregate, "test", variant).get("variant_minus_base_mean_mae")
        if not all(isinstance(value, (int, float)) and float(value) > 0 for value in [front_delta, val_delta, test_delta]):
            return False
    return True


def train_sanity_improves(aggregate: dict[str, Any]) -> bool:
    values = [split_data(aggregate, "train_sanity", variant).get("variant_minus_base_mean_mae") for variant in LORA_VARIANTS]
    return all(isinstance(value, (int, float)) and float(value) < 0 for value in values)


def make_decision(aggregate: dict[str, Any]) -> dict[str, Any]:
    heldout_front_bad = heldout_and_front_degrade(aggregate)
    train_only_good = train_sanity_improves(aggregate)
    decision = "stop_full_lora_eval" if heldout_front_bad else "review_before_full_lora_eval"
    return {
        "decision": decision,
        "heldout_val_test_and_front_degrade": heldout_front_bad,
        "train_sanity_improves": train_only_good,
        "reason": (
            "All LoRA scales degrade held-out val/test and front views versus corrected-input base; train_sanity-only gains suggest overfitting or domain-specific drift."
            if heldout_front_bad
            else "Pilot metrics are mixed; inspect boards before deciding whether to expand."
        ),
        "selected_for_full_24_case_expansion": False if heldout_front_bad else None,
    }


def markdown_variant_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| Variant | Views | Base MAE | Variant MAE | MAE Delta | Base SSIM-like | Variant SSIM-like | SSIM Delta | MAE Improved Views |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row.get('view_count', 'n/a')} | {fmt(row.get('base_mae'))} | {fmt(row.get('variant_mae'))} | {fmt(row.get('mae_delta'))} | {fmt(row.get('base_ssim'))} | {fmt(row.get('variant_ssim'))} | {fmt(row.get('ssim_delta'))} | {row.get('improved_mae_views', 'n/a')} |"
        )
    lines.append("")
    return lines


def markdown_split_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Split Table", "", "| Split | Variant | Views | Base MAE | Variant MAE | MAE Delta | SSIM Delta | MAE Improved Views |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['label']} | {row.get('view_count', 'n/a')} | {fmt(row.get('base_mae'))} | {fmt(row.get('variant_mae'))} | {fmt(row.get('mae_delta'))} | {fmt(row.get('ssim_delta'))} | {row.get('improved_mae_views', 'n/a')} |"
        )
    lines.append("")
    return lines


def decision_table(decision: dict[str, Any]) -> list[str]:
    return [
        "## Decision Table",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Held-out val/test and front views degrade for all LoRA scales | `{decision['heldout_val_test_and_front_degrade']}` |",
        f"| Train-sanity improves for all LoRA scales | `{decision['train_sanity_improves']}` |",
        f"| Decision | `{decision['decision']}` |",
        f"| Selected for full 24-case LoRA expansion | `{decision['selected_for_full_24_case_expansion']}` |",
        f"| Reason | {decision['reason']} |",
        "",
    ]


def make_summary_tables_md(aggregate: dict[str, Any], decision: dict[str, Any]) -> str:
    lines = ["# Phase 2M LoRA Closeout Summary Tables", ""]
    lines.extend(markdown_variant_table("Overall All Views", variant_table_rows(aggregate, "all_views")))
    lines.extend(markdown_variant_table("Front Views 004/005", variant_table_rows(aggregate, "front_views_004_005")))
    lines.extend(markdown_variant_table("Non-Front Views 000-003", variant_table_rows(aggregate, "non_front_views_000_003")))
    lines.extend(markdown_split_table(split_table_rows(aggregate)))
    lines.extend(decision_table(decision))
    return "\n".join(lines).rstrip() + "\n"


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_report_artifacts(args: argparse.Namespace, packet_dir: Path) -> list[str]:
    copied: list[str] = []
    for src, name in [
        (args.aggregate_summary, "aggregate_summary.json"),
        (args.scale_comparison, "scale_comparison.json"),
        (args.metrics_rows, "metrics_rows.csv"),
        (args.training_summary, "training_summary.json"),
        (args.pilot_summary, "pilot_summary.json"),
    ]:
        dst = packet_dir / name
        copy_file(src, dst)
        copied.append(str(dst))
    boards_root = args.boards_root
    if not boards_root.is_dir():
        raise FileNotFoundError(f"boards directory missing: {boards_root}")
    board_paths = sorted(boards_root.rglob("*.jpg"))
    if not board_paths:
        raise FileNotFoundError(f"no board JPG files found under {boards_root}")
    for board in board_paths:
        rel = board.relative_to(boards_root)
        dst = packet_dir / "boards" / rel
        copy_file(board, dst)
        copied.append(str(dst))
    return copied


def make_report_md(training: dict[str, Any], adapter_config: dict[str, Any], pilot: dict[str, Any], aggregate: dict[str, Any], decision: dict[str, Any]) -> str:
    families = target_module_families(adapter_config)
    target_modules = " + ".join(families) if families else "attn_refview + attn_dino"
    pilot_cases = ", ".join(f"{case['case_id']} ({case['eval_split']})" for case in pilot.get("cases", []))
    trainable = int(training.get("trainable_parameter_count", 829952))
    total = int(training.get("total_parameter_count", 3099557192))
    size_mb = adapter_size_mb(training)
    steps = int(training.get("max_train_steps", 300))
    lines = [
        "# Phase 2M LoRA Negative Result Report",
        "",
        "## Experiment Goal",
        "",
        "Phase 2M tested whether a small adapter-only LoRA update on the reference-conditioning attention path could improve corrected-input Hunyuan3D-Paint outputs without modifying the official Hunyuan source tree or saving a full model checkpoint.",
        "",
        "## Setup",
        "",
        f"- Target modules: `{target_modules}`",
        f"- Trainable parameters: `{trainable:,}`",
        f"- Total parameters: `{total:,}`",
        f"- Adapter size: `{size_mb:.1f} MB`",
        f"- Training steps: `{steps}`",
        "- Backend: `local_linear_fallback`",
        "- Adapter-only safety: `merge_into_base=false`, `save_pretrained_full_model=false`",
        "- Scales evaluated: `0.5`, `0.75`, `1.0`",
        f"- Pilot cases: {pilot_cases}",
        "",
        "The evaluated cases cover one validation asset, one test asset, and one train-sanity asset, all using selected input view `005`.",
        "",
    ]
    lines.extend(markdown_variant_table("All-View Table", variant_table_rows(aggregate, "all_views")))
    lines.extend(markdown_variant_table("Front 004/005 Table", variant_table_rows(aggregate, "front_views_004_005")))
    lines.extend(markdown_variant_table("Non-Front 000-003 Table", variant_table_rows(aggregate, "non_front_views_000_003")))
    lines.extend(markdown_split_table(split_table_rows(aggregate)))
    lines.extend(
        [
            "## Qualitative Board Observations",
            "",
            "The visual boards show scale-dependent front-to-back leakage. Higher LoRA scales increase visible drift on front/input views and do not produce a reliable held-out improvement over corrected-input base. The only consistent metric improvement appears on the train_sanity case, which is compatible with overfitting or narrow domain-specific drift rather than a robust improvement.",
            "",
            "Board paths are included in the report packet under `boards/`.",
            "",
            "## Final Conclusion",
            "",
            "`ref_dino` LoRA is not selected for full 24-case expansion.",
            "",
            f"Decision string: `{decision['decision']}`",
            "",
            decision["reason"],
            "",
            "## Recommended Future Work",
            "",
            "- LoRA with an explicit non-front/base preservation objective.",
            "- Visibility-aware loss so the model is not rewarded for leaking front texture onto unseen surfaces.",
            "- Avoid a plain reference-conditioning adapter as the only trainable path; pair it with constraints or supervision that preserve corrected-input base behavior.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def read_metrics_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    packet_dir = ensure_under(resolve_project_path(args.report_packet, project_root), project_root / "outputs" / "phase2m", "report packet")
    doc_report = resolve_project_path(args.doc_report, project_root)
    ensure_under(doc_report, project_root / "docs", "doc report")

    training = load_json(resolve_project_path(args.training_summary, project_root))
    adapter_config = load_json(resolve_project_path(args.adapter_config, project_root))
    pilot = load_json(resolve_project_path(args.pilot_summary, project_root))
    aggregate = load_json(resolve_project_path(args.aggregate_summary, project_root))
    scale_comparison = load_json(resolve_project_path(args.scale_comparison, project_root))
    metrics_rows = resolve_project_path(args.metrics_rows, project_root)
    boards_root = resolve_project_path(args.boards_root, project_root)

    args.training_summary = resolve_project_path(args.training_summary, project_root)
    args.pilot_summary = resolve_project_path(args.pilot_summary, project_root)
    args.aggregate_summary = resolve_project_path(args.aggregate_summary, project_root)
    args.scale_comparison = resolve_project_path(args.scale_comparison, project_root)
    args.metrics_rows = metrics_rows
    args.boards_root = boards_root

    decision = make_decision(aggregate)
    packet_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_report_artifacts(args, packet_dir)
    summary_tables = make_summary_tables_md(aggregate, decision)
    (packet_dir / "summary_tables.md").write_text(summary_tables, encoding="utf-8")
    copied.append(str(packet_dir / "summary_tables.md"))
    report_md = make_report_md(training, adapter_config, pilot, aggregate, decision)
    doc_report.parent.mkdir(parents=True, exist_ok=True)
    doc_report.write_text(report_md, encoding="utf-8")

    manifest = {
        "phase": "2M.4",
        "status": "OK",
        "decision": decision["decision"],
        "report_packet": str(packet_dir),
        "doc_report": str(doc_report),
        "copied_files": copied,
        "board_count": len(list((packet_dir / "boards").rglob("*.jpg"))),
        "metrics_row_count": read_metrics_row_count(metrics_rows),
        "aggregate_status": aggregate.get("status"),
        "scale_comparison_best_variant": scale_comparison.get("best_variant_by_all_views_mae_delta"),
    }
    write_json(packet_dir / "report_packet_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 2M LoRA closeout report packet from existing local metrics.")
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--adapter-config", type=Path, default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--pilot-summary", type=Path, default=DEFAULT_PILOT_SUMMARY)
    parser.add_argument("--aggregate-summary", type=Path, default=DEFAULT_AGGREGATE_SUMMARY)
    parser.add_argument("--scale-comparison", type=Path, default=DEFAULT_SCALE_COMPARISON)
    parser.add_argument("--metrics-rows", type=Path, default=DEFAULT_METRICS_ROWS)
    parser.add_argument("--boards-root", type=Path, default=DEFAULT_BOARDS_ROOT)
    parser.add_argument("--report-packet", type=Path, default=DEFAULT_REPORT_PACKET)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_report(args)
    print("Phase 2M LoRA closeout report packet")
    print(f"  report_packet: {manifest['report_packet']}")
    print(f"  doc_report: {manifest['doc_report']}")
    print(f"  board_count: {manifest['board_count']}")
    print(f"  metrics_row_count: {manifest['metrics_row_count']}")
    print(f"  decision: {manifest['decision']}")
    print("PHASE2M_LORA_CLOSEOUT_REPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
