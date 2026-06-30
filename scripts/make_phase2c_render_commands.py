#!/usr/bin/env python3
"""Generate a shell script of Blender render commands for Phase 2C."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def command_for_row(row: dict[str, str], blender_bin: str, num_view: int, resolution: int) -> str:
    qa_root = str(Path(row["qa_dir"]).parent)
    parts = [
        blender_bin,
        "--background",
        "--python",
        "scripts/blender_render_hy3dpaint_example.py",
        "--",
        "--input-glb",
        row["input_glb"],
        "--sample-name",
        row["sample_name"],
        "--out-root",
        row["dataset_root"],
        "--qa-root",
        qa_root,
        "--num-view",
        str(num_view),
        "--resolution",
        str(resolution),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def write_script(rows: list[dict[str, str]], out_sh: Path, blender_bin: str, num_view: int, resolution: int) -> None:
    out_sh.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    lines.extend(command_for_row(row, blender_bin, num_view, resolution) for row in rows)
    out_sh.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_sh.chmod(0o755)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 2C Blender render commands.")
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--out-sh", required=True, type=Path)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--num-view", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.render_manifest)
    write_script(rows, args.out_sh, args.blender_bin, args.num_view, args.resolution)
    print(f"wrote render command script: {args.out_sh}")
    print(f"render commands: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
