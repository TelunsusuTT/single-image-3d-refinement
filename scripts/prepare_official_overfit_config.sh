#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env/env.sh"

UPSTREAM_CFG="$HYPAINT/cfgs/hunyuan-paint-pbr.yaml"
TARGET_CFG="$PROJ/configs/ft_overfit_official.yaml"

if [[ ! -f "$UPSTREAM_CFG" ]]; then
  echo "ERROR: upstream config not found: $UPSTREAM_CFG" >&2
  exit 1
fi

mkdir -p "$PROJ/configs"

if [[ -e "$TARGET_CFG" ]]; then
  echo "Project config already exists: $TARGET_CFG"
else
  cp "$UPSTREAM_CFG" "$TARGET_CFG"
  echo "Copied official upstream config to: $TARGET_CFG"
fi

echo "WARNING: this config is copied from upstream and should be edited only in this project folder."
echo "Do not modify the upstream config: $UPSTREAM_CFG"
