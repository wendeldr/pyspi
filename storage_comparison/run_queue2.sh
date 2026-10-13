#!/usr/bin/env bash
# Native-codec screen, lossless headroom, and peak memory; runs after run_queue.sh finishes.
set -uo pipefail
cd "$(dirname "$0")"
until grep -q "== queue done" reports_queue.log 2>/dev/null; do sleep 20; done
PY=".venv/bin/python -W ignore"
export TMPDIR=/tmp/claude-1000/-media-dan-Data-git-pyspi/f80d5487-3eee-4147-9877-5020d4958266/scratchpad
echo "== native codec screen $(date)"
$PY screen.py --run screen_native_w64 --codecs pcodec8 openzl_tsplit_zstd openzl_tsplit_entropy vortex duckdb_native \
  --structures full exact --layouts wm --precisions f64 f32 --workers 12 || echo "!! native screen failed"
echo "== headroom $(date)"
$PY headroom.py --workers 12 || echo "!! headroom failed"
echo "== memory probe $(date)"
$PY memory_probe.py || echo "!! memory probe failed"
echo "== queue2 done $(date)"
