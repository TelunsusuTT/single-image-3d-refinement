#!/usr/bin/env python3
"""Generate a standalone local HTML gallery for Phase 2A candidates."""

from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path


THUMB_SUFFIXES = (".jpg", ".png", ".jpeg", ".webp")


def read_candidates(path: Path, top_k: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:top_k]


def find_thumbnail(candidate_id: str, thumb_dir: Path) -> Path | None:
    for suffix in THUMB_SUFFIXES:
        path = thumb_dir / f"{candidate_id}{suffix}"
        if path.is_file():
            return path
    return None


def esc(value: str | None) -> str:
    return html.escape(value or "", quote=True)


def relative_thumb_path(path: Path, out_html: Path) -> str:
    return os.path.relpath(path, start=out_html.parent)


def card_html(row: dict[str, str], thumb_dir: Path, out_html: Path, candidates_csv: Path) -> str:
    candidate_id = row.get("candidate_id", "")
    thumb_path = find_thumbnail(candidate_id, thumb_dir)
    if thumb_path:
        rel_path = relative_thumb_path(thumb_path, out_html)
        image_html = f'<img src="{esc(rel_path)}" alt="{esc(candidate_id)}">'
    else:
        image_html = '<div class="missing-thumb">No thumbnail</div>'

    command = (
        "python scripts/mark_phase2a_candidates.py "
        f"--candidates-csv {candidates_csv} "
        "--selected-csv data/candidates/phase2a_selected_assets.csv "
        f"--candidate-id {candidate_id} "
        '--status selected --reason "texture-heavy product candidate"'
    )
    facts = [
        ("source_id", row.get("source_id", "")),
        ("product", row.get("name", "")),
        ("type", row.get("product_type", "")),
        ("node", row.get("node_text", "")),
        ("brand", row.get("brand", "")),
        ("material", row.get("material", "")),
        ("textures", row.get("textures", "")),
        ("images", row.get("images", "")),
        (
            "resolution",
            f"{row.get('image_width_max', '')} x {row.get('image_height_max', '')}".strip(),
        ),
        ("faces", row.get("faces", "")),
        ("materials", row.get("materials", "")),
        ("score", row.get("final_candidate_score", "")),
    ]
    fact_html = "\n".join(
        f"<dt>{esc(label)}</dt><dd>{esc(value)}</dd>" for label, value in facts if value
    )
    return f"""
      <article class="card">
        {image_html}
        <h2>{esc(candidate_id)}</h2>
        <dl>{fact_html}</dl>
        <p><strong>thumbnail_url</strong><br><code>{esc(row.get('thumbnail_url', ''))}</code></p>
        <p><strong>original_image_url</strong><br><code>{esc(row.get('original_image_url', ''))}</code></p>
        <p><strong>asset_s3_uri</strong><br><code>{esc(row.get('asset_s3_uri', ''))}</code></p>
        <p><strong>select</strong><br><code>{esc(command)}</code></p>
      </article>
    """


def write_gallery(rows: list[dict[str, str]], thumb_dir: Path, out_html: Path, candidates_csv: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    cards = "\n".join(card_html(row, thumb_dir, out_html, candidates_csv) for row in rows)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 2A ABO Candidate Gallery</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f6f6f4;
      color: #1f2328;
    }}
    header {{
      padding: 20px 24px;
      background: #263238;
      color: white;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
      padding: 16px;
    }}
    .card {{
      background: white;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 12px;
      overflow-wrap: anywhere;
    }}
    img, .missing-thumb {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #eee;
      border: 1px solid #d8dee4;
      border-radius: 4px;
    }}
    .missing-thumb {{
      display: grid;
      place-items: center;
      color: #6e7781;
    }}
    h1, h2 {{
      margin: 0;
    }}
    h2 {{
      margin-top: 10px;
      font-size: 16px;
    }}
    dl {{
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 4px 8px;
      font-size: 13px;
    }}
    dt {{
      font-weight: 700;
      color: #57606a;
    }}
    dd {{
      margin: 0;
    }}
    code {{
      font-size: 12px;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Phase 2A ABO Candidate Gallery</h1>
    <p>{len(rows)} candidate cards. Review thumbnails manually, then mark selected rows.</p>
  </header>
  <main>
{cards}
  </main>
</body>
</html>
"""
    out_html.write_text(html_text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML candidate gallery.")
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--thumb-dir", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_candidates(args.candidates_csv, args.top_k)
    write_gallery(rows, args.thumb_dir, args.out_html, args.candidates_csv)
    print(f"wrote gallery: {args.out_html}")
    print(f"candidate cards: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
