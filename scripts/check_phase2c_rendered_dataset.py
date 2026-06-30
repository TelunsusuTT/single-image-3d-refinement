#!/usr/bin/env python3
"""Summarize Phase 2C rendered dataset structure without running Hunyuan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RENDER_TEX_SUFFIXES = [".png", "_albedo.png", "_mr.png", "_normal.png", "_pos.png"]
RENDER_COND_SUFFIXES = ["_light_AL.png", "_light_ENVMAP.png", "_light_PL.png"]
OUTPUT_FIELDS = [
    "source_id",
    "sample_dir",
    "render_tex_exists",
    "render_cond_exists",
    "expected_file_count",
    "present_expected_file_count",
    "missing_count",
    "missing_files",
    "render_tex_file_count",
    "render_cond_file_count",
    "qa_dir",
    "qa_dir_exists",
    "camera_framing_report_exists",
    "framing_report_exists",
    "status",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_files(sample_dir: Path, num_view: int) -> list[Path]:
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    files = [render_tex / "transforms.json"]
    for index in range(num_view):
        prefix = f"{index:03d}"
        files.extend(render_tex / f"{prefix}{suffix}" for suffix in RENDER_TEX_SUFFIXES)
        files.extend(render_cond / f"{prefix}{suffix}" for suffix in RENDER_COND_SUFFIXES)
    return files


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file())


def check_row(row: dict[str, str], num_view: int, framing_qa_root: Path) -> dict[str, str]:
    source_id = row["source_id"]
    sample_dir = Path(row["sample_dir"])
    render_tex = sample_dir / "render_tex"
    render_cond = sample_dir / "render_cond"
    qa_dir = framing_qa_root / source_id
    files = expected_files(sample_dir, num_view)
    missing = [str(path) for path in files if not path.is_file()]
    if not render_tex.is_dir() and str(render_tex) not in missing:
        missing.insert(0, str(render_tex))
    if not render_cond.is_dir() and str(render_cond) not in missing:
        missing.insert(0, str(render_cond))

    expected_file_count = len(files)
    present_expected_count = expected_file_count - sum(1 for path in files if not path.is_file())
    ok = not missing
    return {
        "source_id": source_id,
        "sample_dir": str(sample_dir),
        "render_tex_exists": "yes" if render_tex.is_dir() else "no",
        "render_cond_exists": "yes" if render_cond.is_dir() else "no",
        "expected_file_count": str(expected_file_count),
        "present_expected_file_count": str(present_expected_count),
        "missing_count": str(len(missing)),
        "missing_files": "; ".join(missing),
        "render_tex_file_count": str(count_files(render_tex)),
        "render_cond_file_count": str(count_files(render_cond)),
        "qa_dir": str(qa_dir),
        "qa_dir_exists": "yes" if qa_dir.is_dir() else "no",
        "camera_framing_report_exists": "yes" if (qa_dir / "camera_framing_report.json").is_file() else "no",
        "framing_report_exists": "yes" if (qa_dir / "framing_report.json").is_file() else "no",
        "status": "pass" if ok else "fail",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2C Rendered Dataset Summary",
        "",
        "| source_id | status | expected | present | missing | render_tex files | render_cond files | framing report |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {source_id} | {status} | {expected} | {present} | {missing} | {tex} | {cond} | {framing} |".format(
                source_id=row["source_id"],
                status=row["status"],
                expected=row["expected_file_count"],
                present=row["present_expected_file_count"],
                missing=row["missing_count"],
                tex=row["render_tex_file_count"],
                cond=row["render_cond_file_count"],
                framing=row["framing_report_exists"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2C rendered dataset files.")
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--num-view", type=int, default=6)
    parser.add_argument("--framing-qa-root", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [check_row(row, args.num_view, args.framing_qa_root) for row in read_rows(args.render_manifest)]
    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows)
    failed = [row for row in rows if row["status"] != "pass"]
    print(f"wrote CSV summary: {args.out_csv}")
    print(f"wrote Markdown summary: {args.out_md}")
    print(f"Phase 2C rendered dataset: {len(rows) - len(failed)} pass, {len(failed)} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
