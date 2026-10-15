"""dcorr_hsic-2: hyppo's own O(T log T) 1-D distance-covariance path for dcorr / dcorr_biased.

Candidate (lives only here): the same hyppo Dcorr(bias=...) object with ``is_fast = True`` set before
``statistic(x, y)``; hyppo then skips the T x T distance matrices and runs _dcorr(x, y, bias, is_fast=True),
i.e. three _fast_1d_dcov calls (Chaudhuri & Hu 2019) with the same U-statistic (bias=False:
n(n-3), n(n-3)(n-2), n(n-3)(n-2)(n-1)) or V-statistic (bias=True: n^2, n^3, n^4) denominators and the
same 0-clamp / sqrt logic. Every ordered pair is computed and the upper triangle mirrored, as in the
baseline, so the timing isolates the fast path. Expected class: within-tolerance (different summation).
Single-threaded (OMP/NUMBA = 1).
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import numpy as np
import _bench as B
from hyppo.independence import Dcorr


def candidate(data, bias):
    z = data.to_numpy()
    P = z.shape[0]
    S = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                d = Dcorr(bias=bias)
                d.is_fast = True
                S[i, j] = d.statistic(z[i], z[j])
    li = np.tril_indices(P, -1)
    S[li] = S.T[li]
    np.fill_diagonal(S, np.nan)
    return S


shapes, golden = {}, {}
for ident, bias in (("dcorr", False), ("dcorr_biased", True)):
    for shape in ("7x100", "8x1024"):
        raw, make = B.data(shape)
        S0, t0 = B.baseline(ident, make, repeats=3)
        candidate(make(), bias)               # warm-up (numba compile / cache load)
        t1, S1, _ = B.timeit(lambda: candidate(make(), bias), repeats=3)
        shapes[f"{ident}|{shape}"] = B.result(S0, t0, S1, t1)
        if shape == "7x100":
            g = B.golden(ident)
            if g is not None:
                golden[ident] = B.diff(g, S1)
B.write("dcorr_hsic-2", shapes, golden_7x100=golden,
        threads=dict(OMP_NUM_THREADS=os.environ["OMP_NUM_THREADS"], NUMBA_NUM_THREADS=os.environ["NUMBA_NUM_THREADS"]),
        notes="Baseline = seeded pyspi call. Candidate = hyppo Dcorr with is_fast=True (hyppo dcorr.py:_fast_1d_dcov), "
              "same bias flag, every ordered pair, mirrored. Agreement with the baseline says only that the candidate "
              "reproduces current output within the quoted differences. 32x1024 not run (baseline 66 s). The baseline "
              "is O(T^2) per pair and the candidate O(T log T), so the ratio grows roughly as T / log T.")
print(open(B.MICRO / "opp_dcorr_hsic-2.json").read())
