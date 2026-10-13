#!/usr/bin/env bash
# Native screen without DuckDB (not bit-exact: signed zeros), then thread scaling on a quiet machine.
set -uo pipefail
cd "$(dirname "$0")"
until grep -q "== queue2 done" reports_queue2.log 2>/dev/null; do sleep 20; done
PY=".venv/bin/python -W ignore"
export TMPDIR=/tmp/claude-1000/-media-dan-Data-git-pyspi/f80d5487-3eee-4147-9877-5020d4958266/scratchpad
echo "== native codec screen $(date)"
$PY screen.py --run screen_native_w64 --codecs pcodec8 openzl_tsplit_zstd openzl_tsplit_entropy vortex \
  --structures full exact --layouts wm --precisions f64 f32 --workers 12 || echo "!! native screen failed"
echo "== thread scaling $(date)"
$PY thread_scaling.py || echo "!! thread scaling failed"
echo "== queue3 done $(date)"
