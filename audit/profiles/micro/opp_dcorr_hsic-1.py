"""dcorr_hsic-1: per-channel distance-matrix cache for dcorr / dcorr_biased (hyppo Dcorr).

Candidate (lives only here): per channel, the hyppo/sklearn Euclidean distance matrix, its
centred form (hyppo _center_distmat, bias-specific) and its self inner product (hyppo _dcov) are
computed ONCE; per pair only the cross inner product _dcov(A_i, A_j) is left. The same numba
functions are called on the same arrays, so the output is expected to match the baseline bit for
bit. Two candidate variants are timed: every ordered pair (isolates the cache) and upper triangle
only (cache + the cross-cutting xc-undirected-upper avenue). Single-threaded (OMP/NUMBA = 1).
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import numpy as np
import _bench as B
from hyppo.tools.common import compute_dist
from hyppo.independence.dcorr import _center_distmat, _dcov


def candidate(data, bias, upper_only):
    z = data.to_numpy()                       # (P, T, 1), what DistanceCorrelation.bivariate slices
    P = z.shape[0]
    A = [_center_distmat(compute_dist(z[k], metric="euclidean")[0], bias) for k in range(P)]
    V = [_dcov(a, a, bias=bias, only_dcov=False) for a in A]
    S = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j or (upper_only and j < i):
                continue
            covar, varx, vary = _dcov(A[i], A[j], bias=bias, only_dcov=False), V[i], V[j]
            if varx <= 0 or vary <= 0 or (covar <= 0 and bias):   # hyppo _dcorr, dcorr.py:438-447
                stat = 0.0
            else:
                stat = covar / np.sqrt(np.real(varx * vary))
                if bias:
                    stat = np.sqrt(stat)
            S[i, j] = stat
    li = np.tril_indices(P, -1)
    S[li] = S.T[li]                           # Undirected.multivariate mirror (base.py:136-137)
    np.fill_diagonal(S, np.nan)
    return S


shapes, golden = {}, {}
for ident, bias in (("dcorr", False), ("dcorr_biased", True)):
    for shape in ("7x100", "8x1024"):
        raw, make = B.data(shape)
        S0, t0 = B.baseline(ident, make, repeats=3)
        candidate(make(), bias, False)        # warm-up (numba cache load)
        t1, S1, _ = B.timeit(lambda: candidate(make(), bias, False), repeats=3)
        t2, S2, _ = B.timeit(lambda: candidate(make(), bias, True), repeats=3)
        shapes[f"{ident}|{shape}"] = B.result(S0, t0, S1, t1, upper_only=B.result(S0, t0, S2, t2))
        if shape == "7x100":
            g = B.golden(ident)
            if g is not None:
                golden[ident] = B.diff(g, S1)
B.write("dcorr_hsic-1", shapes, golden_7x100=golden,
        threads=dict(OMP_NUM_THREADS=os.environ["OMP_NUM_THREADS"], NUMBA_NUM_THREADS=os.environ["NUMBA_NUM_THREADS"]),
        notes="Baseline = seeded pyspi call (both ordered pairs, mirrored). Candidate = per-channel cache of "
              "hyppo distance, _center_distmat and self _dcov; per pair only _dcov(A_i, A_j). 'upper_only' adds the "
              "cross-cutting upper-triangle avenue. Agreement with the baseline says only that the candidate "
              "reproduces current output. 32x1024 not run (baseline 66 s); per-pair cost of both is O(T^2), so the "
              "ratio is expected to be roughly independent of P and T apart from the (P-1)-fold cache reuse.")
print(open(B.MICRO / "opp_dcorr_hsic-1.json").read())
