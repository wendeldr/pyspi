"""xc-blas-threads: OpenBLAS thread count (threadpoolctl) on BLAS/LAPACK-heavy calls; JVM interaction.

Part A (this process, JVM never started; asserted): wall and output of
  - gpfit_DotProduct, one bivariate pair (rows 0,1 of the 8x1024 synthetic set; sklearn GPR,
    Cholesky/cho_solve on 1024x1024; the most BLAS-heavy unit in the 8x1024/8x256 profiles),
  - cov_GraphicalLassoCV at 118x1024 (retained EEG selection), fresh Data per call,
  - Data construction at 118x1024 (scipy detrend = LAPACK lstsq, then zscore): the
    preprocessing cost per window (preproc.detrend / preproc.zscore have no profile of their own),
under threadpool_limits(1 | 4 | 8, user_api='blas'); diffs are against the 1-thread output.
SPIs are built directly with harness.build_spi (the catalogue loader starts the JVM).
Part B (isolated subprocesses, a crash kills only the child): the historical "JVM + OpenBLAS
segfault" (pyspi/__init__.py:1). Variants: JIDT JVM started or not, then numpy.linalg.solve on
1500x1500 under 1/2/8 BLAS threads, and a JVM child running only GEMM, 20000 batched 8x8 solves
and eigh under 8 threads. Records each child's return code (139 / -11 = SIGSEGV) and last step.
Settings: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1 at load, NUMBA_NUM_THREADS=1; at most 8 BLAS
threads because other audit groups share the machine.
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import sys
import time
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402
import numpy as np  # noqa: E402
import jpype  # noqa: E402
from threadpoolctl import threadpool_limits, threadpool_info  # noqa: E402

THREADS = (1, 4, 8)


def build(module, cls, **params):
    return B.h.build_spi(dict(module=f".statistics.{module}", cls=cls, params=params))


def blas_threads():
    return [d["num_threads"] for d in threadpool_info() if d["user_api"] == "blas"]


def sweep(label, fn, repeats):
    out, ref = {}, None
    for n in THREADS:
        with threadpool_limits(limits=n, user_api="blas"):
            eff = blas_threads()
            wall, val, walls = B.timeit(fn, repeats=repeats, cutoff_s=20.0)
        val = np.asarray(val, dtype=np.float64)
        if ref is None:
            ref, t1 = val, wall
        r = dict(wall_s=wall, walls_s=walls, effective_blas_threads=eff, speedup_vs_1=t1 / wall)
        r.update(B.diff(ref, val))
        out[f"threads_{n}"] = r
        print(label, n, eff, round(wall, 4), r["max_abs_diff"], flush=True)
    return out


shapes = {}
raw8, make8 = B.data("8x1024")
d8 = make8()
gp = build("misc", "GPModel", kernel="DotProduct")
assert gp.identifier == "gpfit_DotProduct", gp.identifier


def gp_pair():
    B.h.seed_all(B.SPI_SEED)
    return [gp.bivariate(d8, i=0, j=1)]


shapes["gpfit_DotProduct_pair_8x1024"] = sweep("gpfit", gp_pair, repeats=2)

raw118, make118 = B.data("118x1024")
shapes["data_construction_118x1024"] = sweep("data", lambda: make118().to_numpy(squeeze=True), repeats=5)


def glcv():
    s = build("basic", "Covariance", estimator="GraphicalLassoCV")
    assert s.identifier == "cov_GraphicalLassoCV", s.identifier
    d = make118()
    B.h.seed_all(B.SPI_SEED)
    return B.h.compute_spi(s, d)


shapes["cov_GraphicalLassoCV_118x1024"] = sweep("glcv", glcv, repeats=1)
jvm_in_part_a = bool(jpype.isJVMStarted())
assert not jvm_in_part_a

CHILD = r'''
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "1"
use_jvm, n, ops = sys.argv[1] == "jvm", int(sys.argv[2]), sys.argv[3]
import numpy as np
from threadpoolctl import threadpool_limits
if use_jvm:
    import jpype as jp
    jp.startJVM(jp.getDefaultJVMPath(), "-ea",
                "-Djava.class.path=/media/dan/Data/git/pyspi/pyspi/lib/jidt/infodynamics.jar")
    print("STEP jvm", flush=True)
rng = np.random.default_rng(0)
a = rng.standard_normal((1500, 1500))
with threadpool_limits(limits=n, user_api="blas"):
    for k in range(3):
        if ops == "solve":
            np.linalg.solve(a + 1500 * np.eye(1500), rng.standard_normal(1500)); print("STEP solve", k, flush=True)
        else:
            a @ a.T; print("STEP gemm", k, flush=True)
            b = rng.standard_normal((20000, 8, 8)) + 8 * np.eye(8)
            np.linalg.solve(b, rng.standard_normal((20000, 8, 1))); print("STEP batched8x8", k, flush=True)
            np.linalg.eigh(a @ a.T); print("STEP eigh", k, flush=True)
print("CHILD_OK", flush=True)
'''
children = {}
for use_jvm, n, ops in (("jvm", 8, "solve"), ("jvm", 2, "solve"), ("jvm", 1, "solve"),
                        ("nojvm", 8, "solve"), ("jvm", 8, "gemm_batched_eigh")):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    t0 = time.perf_counter()
    cp = subprocess.run([sys.executable, "-c", CHILD, use_jvm, str(n), ops], env=env,
                        capture_output=True, text=True, timeout=120)
    steps = [ln for ln in cp.stdout.splitlines() if ln.startswith(("STEP", "CHILD_OK"))]
    key = f"{use_jvm}_threads-{n}_{ops}"
    children[key] = dict(returncode=cp.returncode, completed=bool(steps and steps[-1] == "CHILD_OK"),
                         last_step=steps[-1] if steps else None, wall_s=time.perf_counter() - t0)
    print(key, children[key], flush=True)

p = B.write("xc-blas-threads", shapes,
            notes="Thread-count sweep, no candidate implementation. Diffs are against the 1-thread output "
                  "of the same call; speedup_vs_1 is the 1-thread wall over the n-thread wall. "
                  "jvm_openblas_children: returncode -11 is SIGSEGV.",
            jvm_openblas_children=children, jvm_started_in_part_a=jvm_in_part_a,
            machine_load="five other audit groups ran concurrently on the 64-core machine",
            settings=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1 at load, raised by threadpool_limits",
                          NUMBA_NUM_THREADS="1", blas_threads_swept=list(THREADS),
                          openblas=[{k: d.get(k) for k in ("internal_api", "version")}
                                    for d in threadpool_info() if d["user_api"] == "blas"],
                          java="OpenJDK 21.0.12.1 (java -version)"))
print(p)
