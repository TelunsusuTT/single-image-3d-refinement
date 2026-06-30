#!/usr/bin/env python3
"""Download small candidate thumbnails from a Phase 2A candidate CSV."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def thumbnail_suffix(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    return ".jpg"


def read_candidates(path: Path, top_k: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:top_k]


def download_thumbnail(url: str, out_path: Path, skip_existing: bool, timeout: int) -> str:
    if skip_existing and out_path.is_file():
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "hy3dpaint-phase2a/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, out_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return "downloaded"


def write_failure_report(thumb_dir: Path, failures: list[dict[str, str]]) -> None:
    if not failures:
        return
    out_csv = thumb_dir / "thumbnail_failures.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "thumbnail_url", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"thumbnail failure report: {out_csv}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only small ABO candidate thumbnails.")
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--thumb-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-consecutive-failures", type=int, default=30)
    return parser.parse_args(argv)


def should_abort(consecutive_failures: int, downloaded: int, skipped: int, max_failures: int) -> bool:
    return consecutive_failures >= max_failures and downloaded == 0 and skipped == 0


def print_abort_message(consecutive_failures: int) -> None:
    print(
        "ERROR: aborting thumbnail downloads after "
        f"{consecutive_failures} consecutive failures and zero successes. "
        "The thumbnail URL prefix may be wrong; expected ABO small-image URLs "
        "like https://amazon-berkeley-objects.s3.amazonaws.com/images/small/<image_path>.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout <= 0:
        print("ERROR: --timeout must be > 0", file=sys.stderr)
        return 2
    if args.max_consecutive_failures <= 0:
        print("ERROR: --max-consecutive-failures must be > 0", file=sys.stderr)
        return 2

    rows = read_candidates(args.candidates_csv, args.top_k)
    args.thumb_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    consecutive_failures = 0
    aborted_early = False
    failures: list[dict[str, str]] = []

    for row in rows:
        candidate_id = row.get("candidate_id", "").strip()
        url = row.get("thumbnail_url", "").strip()
        if not candidate_id or not url:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "thumbnail_url": url,
                    "error": "missing candidate_id or thumbnail_url",
                }
            )
            consecutive_failures += 1
            if should_abort(consecutive_failures, downloaded, skipped, args.max_consecutive_failures):
                aborted_early = True
                print_abort_message(consecutive_failures)
                break
            continue

        out_path = args.thumb_dir / f"{candidate_id}{thumbnail_suffix(url)}"
        try:
            status = download_thumbnail(url, out_path, args.skip_existing, args.timeout)
        except OSError as exc:
            failures.append(
                {"candidate_id": candidate_id, "thumbnail_url": url, "error": str(exc)}
            )
            print(f"thumbnail failed: {candidate_id} {url} ({exc})", file=sys.stderr)
            consecutive_failures += 1
            if should_abort(consecutive_failures, downloaded, skipped, args.max_consecutive_failures):
                aborted_early = True
                print_abort_message(consecutive_failures)
                break
            continue

        if status == "skipped":
            skipped += 1
        else:
            downloaded += 1
        consecutive_failures = 0
        print(f"{status}: {out_path}")

    write_failure_report(args.thumb_dir, failures)
    print(f"rows considered: {len(rows)}")
    print(f"downloaded: {downloaded}")
    print(f"skipped: {skipped}")
    print(f"failures: {len(failures)}")
    return 1 if aborted_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
