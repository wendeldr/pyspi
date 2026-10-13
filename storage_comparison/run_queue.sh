#!/usr/bin/env bash
# Trimmed follow-ups after the main container run; each step runs alone on the machine.
# A failing step is logged and the queue moves on.
set -uo pipefail
cd "$(dirname "$0")"
while pgrep -f "[b]ench_containers.py --run containers_full" > /dev/null; do sleep 15; done
SUBSET="cov_EmpiricalCovariance prec_ShrunkCovariance wpli_multitaper_mean_fs-1_fmin-0_fmax-0-5 phase_multitaper_mean_fs-1_fmin-0_fmax-0-5 te_symbolic_k-1_kt-1_l-1_lt-1 gd_multitaper_delay_fs-1_fmin-0-0342_fmax-0-0732 pli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195 plv_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195 dspli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342 lmfit_BayesianRidge"
PY=".venv/bin/python -W ignore"
SCRATCH=/tmp/claude-1000/-media-dan-Data-git-pyspi/f80d5487-3eee-4147-9877-5020d4958266/scratchpad
echo "== 111 subset $(date)"
$PY bench_containers.py --run containers_111_subset --patients 111 --metrics $SUBSET \
  --containers zarr_current zarr_shard_pcodec8 --variants f64:exact f32:exact || echo "!! 111 subset failed"
echo "== 105 chunk sweep $(date)"
$PY bench_containers.py --run chunk_sweep_105 --patients 105 --metrics $SUBSET --variants f64:exact \
  --containers zarr_shard_pcodec8_c16k zarr_shard_pcodec8_c1024k zarr_shard_pcodec8_c4096k || echo "!! chunk sweep failed"
echo "== 105 kept conversion $(date)"
$PY bench_containers.py --run transfer_105 --patients 105 --no-corrupt --keep \
  --containers zarr_current zarr_shard_pcodec8 --variants f64:exact || echo "!! keep f64 failed"
$PY bench_containers.py --run transfer_105_f32 --patients 105 --no-corrupt --keep \
  --containers zarr_shard_pcodec8 --variants f32:exact || echo "!! keep f32 failed"
mv runs/transfer_105_f32/105/* runs/transfer_105/105/ && rm -rf runs/transfer_105_f32
echo "== transfer test $(date)"
mkdir -p runs/transfer_105/dest
$PY transfer_test.py --formats-root runs/transfer_105/105 --dest runs/transfer_105/dest \
  --out reports/transfer_105/transfer.json || echo "!! transfer test failed"
rm -rf runs/transfer_105
echo "== thread scaling $(date)"
$PY thread_scaling.py || echo "!! thread scaling failed"
echo "== access examples $(date)"
TMPDIR=$SCRATCH $PY examples_access.py 105 || echo "!! examples failed"
echo "== queue done $(date)"
