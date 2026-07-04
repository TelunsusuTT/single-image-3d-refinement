#!/usr/bin/env python3
"""Write a safe, non-executing ABO download plan for missing probe assets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ABO_HTTPS_PREFIX = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def missing_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if (row.get("needs_download") or "").strip().lower() in {"yes", "true", "1"}]


def write_plan(rows: list[dict[str, str]], out_sh: Path) -> None:
    out_sh.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Phase 2L.2A ABO probe download plan.",
        "# This script is intentionally non-executing: all download commands are comments.",
        "# Existing project downloader for active downloads: scripts/download_phase2b_assets.py",
        "# URL prefix follows scripts/make_phase2b_download_manifest.py.",
        "",
        f"echo 'Missing ABO assets planned: {len(rows)}'",
        "",
    ]
    for row in rows:
        asset_id = row.get("asset_id", "")
        relative = row.get("expected_glb_relative_path", "")
        url = f"{ABO_HTTPS_PREFIX}/{relative}" if relative else ""
        out_path = f"data/raw_assets/abo/{relative}" if relative else f"data/raw_assets/abo/{asset_id}.glb"
        lines.extend(
            [
                f"echo 'PLAN {asset_id}: {url} -> {out_path}'",
                f"# mkdir -p {Path(out_path).parent}",
                f"# python scripts/download_phase2b_assets.py --manifest-csv REPLACE_WITH_PHASE2B_COMPAT_MANIFEST --skip-existing",
                f"# curl -L --fail --output {out_path} {url}",
                "",
            ]
        )
    out_sh.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    out_sh.chmod(0o755)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a safe ABO download plan.")
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--out-sh", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = missing_rows(read_rows(args.probe_manifest))
    write_plan(rows, args.out_sh)
    print("Phase 2L.2A ABO download plan")
    print(f"  missing assets: {len(rows)}")
    print(f"  out_sh: {args.out_sh}")
    print("PHASE2L2A_ABO_DOWNLOAD_PLAN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
