#!/usr/bin/env python3
"""Make visual boards for Phase 2M.3C LoRA multi-scale pilot rendered views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDER_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_rendered"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "phase2m" / "lora_multiscale_pilot_eval"
DEFAULT_COLUMNS = ["reference", "base", "lora_scale050", "lora_scale075", "lora_scale100"]


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


def image_path_for(render_root: Path, case: dict[str, Any], case_id: str, column: str, view_id: str) -> Path:
    if column == "reference":
        return resolve_project_path(case["reference_images"][view_id])
    return render_root / "renders" / case["eval_split"] / case_id / column / f"{view_id}.png"


def board_path_for(eval_root: Path, case: dict[str, Any], case_id: str) -> Path:
    return eval_root / "boards" / case["eval_split"] / f"{case_id}_lora_multiscale_board.jpg"


def make_case_board(render_root: Path, eval_root: Path, render_config: dict[str, Any], case_id: str, case: dict[str, Any]) -> Path:
    from PIL import Image, ImageDraw

    columns = list(DEFAULT_COLUMNS)
    view_ids = list(render_config.get("view_ids", []))
    front_views = set(case.get("primary_front_views", ["004", "005"]))
    selected_view = str(case.get("selected_input_view", ""))
    target_size = (176, 176)
    label_height = 34
    header_height = 34
    board = Image.new(
        "RGB",
        (len(columns) * target_size[0], header_height + len(view_ids) * (target_size[1] + label_height)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(board)
    headers = {
        "reference": "reference",
        "base": "base",
        "lora_scale050": "LoRA 0.50",
        "lora_scale075": "LoRA 0.75",
        "lora_scale100": "LoRA 1.00",
    }
    for col, column in enumerate(columns):
        draw.text((col * target_size[0] + 8, 10), headers.get(column, column), fill=(0, 0, 0))
    for row, view_id in enumerate(view_ids):
        y = header_height + row * (target_size[1] + label_height)
        suffix = []
        if view_id in front_views:
            suffix.append("front")
        if view_id == selected_view:
            suffix.append("input")
        view_label = view_id if not suffix else f"{view_id} ({', '.join(suffix)})"
        draw.text((6, y + 8), view_label, fill=(0, 0, 0))
        for col, column in enumerate(columns):
            path = image_path_for(render_root, case, case_id, column, view_id)
            x = col * target_size[0]
            try:
                image = Image.open(path).convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
            except Exception:
                image = Image.new("RGB", target_size, (230, 230, 230))
                placeholder = ImageDraw.Draw(image)
                placeholder.text((10, 74), "missing", fill=(150, 0, 0))
            board.paste(image, (x, y + label_height))
    path = board_path_for(eval_root, case, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path, quality=92)
    return path


def make_boards(render_root: Path, eval_root: Path, render_config: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_id, case in render_config.get("cases", {}).items():
        path = make_case_board(render_root, eval_root, render_config, case_id, case)
        rows.append({"case_id": case_id, "eval_split": case.get("eval_split", ""), "board_path": str(path)})
    return rows


def write_summary(eval_root: Path, rows: list[dict[str, str]]) -> None:
    write_json(eval_root / "boards" / "board_summary.json", {"board_count": len(rows), "boards": rows})
    lines = [
        "# Phase 2M.3C LoRA Pilot Boards",
        "",
        f"board_count: `{len(rows)}`",
        "",
        "| Case | Split | Board |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['case_id']}` | `{row['eval_split']}` | `{row['board_path']}` |")
    (eval_root / "boards" / "board_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 2M.3C LoRA multi-scale pilot boards.")
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_root = resolve_project_path(args.render_root)
    eval_root = resolve_project_path(args.eval_root)
    render_config = load_json(render_root / "render_eval_cases.json")
    rows = make_boards(render_root, eval_root, render_config)
    write_summary(eval_root, rows)
    print("Phase 2M.3C LoRA multi-scale pilot boards")
    print(f"  board_count: {len(rows)}")
    print(f"  boards_root: {eval_root / 'boards'}")
    print("PHASE2M_M3C_LORA_PILOT_BOARDS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
