"""Extra cProfile runs for the spectral group (Workflow 3): one identifier at one shape.

Usage: .venv/bin/python audit/profiles/micro/prof_spectral.py <unit_id> <identifier> <PxT>
Writes audit/profiles/micro/prof_<unit_id>__<PxT>.pstats after an untimed warm-up call at 8x256.
OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1 (no numba or JIDT code on these paths). Walls are inflated by
profiling; quote shares of the compute_spi frame, not walls. Index: prof_spectral__index.json.
"""
import sys, os, time, cProfile, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B
unit, ident, shape = sys.argv[1], sys.argv[2], sys.argv[3]
out = f"/media/dan/Data/git/pyspi/audit/profiles/micro/prof_{unit}__{shape}.pstats"
_, mk_small = B.data("8x256")
s = B.spi(ident); B.h.seed_all(42); B.h.compute_spi(s, mk_small())
_, make = B.data(shape)
d = make(); s = B.spi(ident); B.h.seed_all(42)
pr = cProfile.Profile(); t0 = time.perf_counter()
pr.enable(); B.h.compute_spi(s, d); pr.disable()
wall = time.perf_counter() - t0
pr.dump_stats(out)
import resource
print(json.dumps(dict(unit=unit, ident=ident, shape=shape, profiled_wall_s=wall, pstats=out,
      ru_maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      omp=os.environ["OMP_NUM_THREADS"], numba=os.environ["NUMBA_NUM_THREADS"])))
