"""Isolated wall re-time of one wavelet identifier (Workflow 3 fix stage): warm-up at 8x1024, then n timed compute_spi calls
on fresh Data, OMP/OPENBLAS/NUMBA threads 1. Usage: .venv/bin/python audit/profiles/micro/retime_wavelet.py <identifier> <PxT> <n>.
Results are recorded in prof_wavelet__index.json (retime)."""
import os, sys, time, resource
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMBA_NUM_THREADS"]="1"
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B
ident, shape, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
_, m8 = B.data("8x1024"); B.h.compute_spi(B.spi(ident), m8())
_, make = B.data(shape)
for k in range(n):
    s = B.spi(ident); d = make(); B.h.seed_all(42)
    t0=time.perf_counter(); B.h.compute_spi(s, d); print(ident, shape, "wall", round(time.perf_counter()-t0,3), "maxrss_MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024, flush=True)
