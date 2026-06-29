#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env/env.sh"

echo "Disk usage by first-level project directory:"
du -h -d 1 "$PROJ" | sort -h

echo
echo "Largest files under project, max depth 5:"
find "$PROJ" -maxdepth 5 -type f -printf '%s\t%p\n' \
  | sort -nr \
  | head -50 \
  | while IFS=$'\t' read -r bytes path; do
      human="$(numfmt --to=iec --suffix=B "$bytes" 2>/dev/null || printf '%sB' "$bytes")"
      printf '%10s  %s\n' "$human" "$path"
    done
