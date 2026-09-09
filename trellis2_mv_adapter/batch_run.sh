#!/bin/bash
# Skip existing outputs; continue after individual failures.
set -u
SRC=${SRC:-/root/autodl-tmp/batch_in}
OUT=${OUT:-/root/autodl-tmp/batch_out}
WORK=${WORK:-/root/autodl-tmp/batch_work}
LOG=${LOG:-/root/autodl-tmp/batch_run.log}
PIPE=/root/autodl-tmp/pipeline/pipeline.py
mkdir -p "$OUT" "$WORK"

echo "[$(date '+%F %T')] BATCH START ($(ls "$SRC"/*.png | wc -l) images) -> $OUT" >> "$LOG"

for f in "$SRC"/*.png; do
  name=$(basename "$f" .png)
  out="$OUT/${name}_textured.glb"
  if [ -f "$out" ]; then
    echo "[$(date '+%F %T')] SKIP $name (output exists)" >> "$LOG"
    continue
  fi
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  timeout 3600 /root/miniconda3/envs/trellis2/bin/python "$PIPE" "$f" "$out" \
      --remove_bg --normalize_views --workdir "$WORK/$name" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    echo "[$(date '+%F %T')] DONE $name rc=$rc" >> "$LOG"
  else
    echo "[$(date '+%F %T')] FAIL $name rc=$rc" >> "$LOG"
  fi
done

echo "[$(date '+%F %T')] BATCH FINISHED" >> "$LOG"
