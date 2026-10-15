"""xc-jni-overhead: per-call cost of one JIDT call sequence through JPype (no candidate code).

Times the sequence pyspi runs per pair for the Gaussian MI/TLMI/TE units
(initialise, two JArray conversions, setObservations, computeAverageLocalOfObservations)
on short (T=16) and window-length (T=1024) arrays, each step on its own, and the full
pyspi ``mi_gaussian`` call at 8x1024 for comparison. The T=16 sequence is the per-pair
JNI/marshalling floor; T=1024 adds array copying and the Java-side estimate.
Settings: OMP_NUM_THREADS=1 (helper), NUMBA_NUM_THREADS=1; the Gaussian JIDT calculators
are single-threaded (JIDT NUM_THREADS is set to 1 anyway; outcome recorded).
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import time
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402
import numpy as np  # noqa: E402
import jpype as jp  # noqa: E402

B.jvm()
spi = B.spi("mi_gaussian")
calc = spi._calc
try:
    calc.setProperty("NUM_THREADS", "1")
    jidt_threads = "NUM_THREADS=1 set"
except Exception as e:  # property unknown to this calculator
    jidt_threads = f"NUM_THREADS not accepted: {e!r}"


def per_call(fn, n, blocks=3):
    """Min over ``blocks`` timed blocks of ``n`` calls (after a warm-up block): seconds per call."""
    for _ in range(min(500, n)):
        fn()
    best = None
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = (time.perf_counter() - t0) / n
        best = dt if best is None else min(best, dt)
    return best


rng = np.random.default_rng(0)
shapes = {}
for T in (16, 1024):
    x, y = rng.standard_normal((2, T))
    jx, jy = jp.JArray(jp.JDouble)(x), jp.JArray(jp.JDouble)(y)

    def seq():
        calc.initialise(1, 1)
        calc.setObservations(jp.JArray(jp.JDouble)(x), jp.JArray(jp.JDouble)(y))
        return calc.computeAverageLocalOfObservations()

    def conv():
        jp.JArray(jp.JDouble)(x)
        jp.JArray(jp.JDouble)(y)

    def init():
        calc.initialise(1, 1)

    def setobs():
        calc.initialise(1, 1)
        calc.setObservations(jx, jy)

    calc.initialise(1, 1)
    calc.setObservations(jx, jy)

    def comp():
        calc.computeAverageLocalOfObservations()

    n = 4000 if T == 16 else 2000
    shapes[f"T{T}"] = dict(
        sequence_s=per_call(seq, n), two_jarray_s=per_call(conv, n), initialise_s=per_call(init, n),
        initialise_plus_setObservations_s=per_call(setobs, n), compute_only_s=per_call(comp, n), n_calls=n)

raw, make = B.data("8x1024")
S, wall = B.baseline("mi_gaussian", make, repeats=5)
d = make()
shapes["8x1024_pyspi_mi_gaussian"] = dict(
    baseline_s=wall, n_pairs=56, per_pair_s=wall / 56,
    bivariate_per_call_s=per_call(lambda: spi.bivariate(d, i=0, j=1), 1000),
    note="baseline_s: whole SPI incl. Data construction (make) inside the timed call; "
         "bivariate_per_call_s: pyspi MutualInfo.bivariate on rows 0,1 (T=1024), min over 3 blocks of 1000")
floor = shapes["T16"]["sequence_s"]
shapes["projection_118"] = dict(
    ordered_pairs=118 * 117, unordered_pairs=118 * 117 // 2,
    floor_per_sequence_s=floor, floor_per_ordered_spi_s=floor * 118 * 117,
    seq_T1024_per_ordered_spi_s=shapes["T1024"]["sequence_s"] * 118 * 117)
p = B.write("xc-jni-overhead", shapes,
            notes="Measurement only, no candidate implementation. T=16 sequence = per-pair JNI/marshalling floor; "
                  "projection_118 multiplies it by 13806 ordered pairs per infotheory SPI.",
            machine_load="five other audit groups ran concurrently; per-call times are min over 3 blocks and vary about 2x between runs",
            settings=dict(NUMBA_NUM_THREADS="1", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                          OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS"),
                          jidt=jidt_threads, jidt_calculator="MutualInfoCalculatorMultiVariateGaussian"))
print(p)
import json; print(json.dumps(shapes, indent=1))
