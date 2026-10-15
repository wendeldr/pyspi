"""rank_corr-1: Spearman by ranking each channel once + one corrcoef of the P x T rank matrix.

Baseline: pyspi SpearmanR.bivariate -> scipy.stats.spearmanr per ordered pair (both triangles,
then Undirected mirror). Candidate (lives only here): scipy.stats.rankdata(X, axis=1) once,
np.corrcoef(R) once; pyspi's A[a,b] (a<b) is spearmanr(x_a, x_b) = rs[1,0] = C[b,a], mirrored.
Agreement with the baseline means only that the candidate reproduces current output.
Run: .venv/bin/python audit/profiles/micro/opp_rank_corr-1.py  (OMP_NUM_THREADS=1 via _bench)
"""
import sys
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


def candidate(d, squared):
    X = d.to_numpy(squeeze=True)                  # (P, T)
    C = np.corrcoef(stats.rankdata(X, axis=1))    # C[b, a] = cov(r_b, r_a) / sd_b / sd_a
    L = np.tril(C, -1)
    A = L + L.T                                   # A[a,b] = A[b,a] = C[b,a] for a < b
    if squared:
        # pyspi squares each pair's numpy scalar (scalar ** 2 -> C pow); array ** 2 is x*x and
        # differed by 1 ulp on 8 of 13924 entries at 118x1024, so square through the scalar path.
        A = np.frompyfunc(lambda v: np.float64(v) ** 2, 1, 1)(A).astype(np.float64)
    np.fill_diagonal(A, np.nan)
    return A


shapes = {}
for shape in ("7x100", "8x1024", "32x1024", "118x1024"):
    raw, make = B.data(shape)
    for ident, sq in (("spearmanr-sq", True), ("spearmanr", False)):
        reps = 2 if shape == "118x1024" else 3
        S_base, t_base = B.baseline(ident, make, repeats=reps)
        t_cand, S_cand, _ = B.timeit(lambda: candidate(make(), sq), repeats=reps)
        shapes[f"{shape}:{ident}"] = B.result(S_base, t_base, S_cand, t_cand)
        print(shape, ident, shapes[f"{shape}:{ident}"], flush=True)
g = B.golden("spearmanr-sq")
if g is not None:
    raw, make = B.data("7x100")
    shapes["7x100:golden:spearmanr-sq"] = B.diff(g, candidate(make(), True))
print(B.write("rank_corr-1", shapes,
              notes="Timings include Data construction (make()) for baseline and candidate alike. "
                    "Baseline computes one identifier per call (no sharing between spearmanr and spearmanr-sq)."))
