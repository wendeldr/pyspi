"""lmfit-1: Ridge(alpha=1) in-sample MSE for all ordered pairs from one batched closed form.

Baseline: pyspi misc.lmfit.Ridge (sklearn Ridge(random_state=42).fit / predict / mean_squared_error per ordered pair).
Candidate (lives only here): with centred sums Sxx_i, Syy_j, Sxy_ij (one P x P Gram matrix),
w = Sxy / (Sxx + 1), MSE[i, j] = (Syy_j - 2 w Sxy_ij + w^2 Sxx_i) / T  (sklearn's intercept is the centring).
Different float operation order from sklearn's cholesky solve, so within-tolerance is expected.
Run: OMP_NUM_THREADS=1 .venv/bin/python audit/profiles/micro/opp_lmfit-1.py
"""
import sys
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import numpy as np
import _bench as B


def candidate(d, alpha=1.0):
    z = d.to_numpy(squeeze=True)
    T = z.shape[1]
    zc = z - z.mean(axis=1, keepdims=True)
    G = zc @ zc.T                                # G[i, j] = Sxy with x = z_i, y = z_j
    s = np.diag(G).copy()
    w = G / (s[:, None] + alpha)
    A = (s[None, :] - 2.0 * w * G + w * w * s[:, None]) / T
    np.fill_diagonal(A, np.nan)
    return A


def one_shot_baseline(make):
    d = make(); s = B.spi("lmfit_Ridge"); B.h.seed_all(B.SPI_SEED)
    t, S, _ = B.timeit(lambda: B.h.compute_spi(s, d), repeats=1)
    return S, t


shapes = {}
for shape in ("7x100", "8x1024", "32x1024", "118x1024"):
    raw, make = B.data(shape)
    S_base, t_base = B.baseline("lmfit_Ridge", make) if shape != "118x1024" else one_shot_baseline(make)
    t_cand, S_cand, _ = B.timeit(lambda: candidate(make()), repeats=5)
    shapes[shape] = B.result(S_base, t_base, S_cand, t_cand,
                             baseline_repeats=3 if shape != "118x1024" else 1)
    print(shape, {k: shapes[shape][k] for k in ("baseline_s", "candidate_s", "speedup", "max_abs_diff", "max_rel_diff", "nan_mask_equal")})
g = B.golden("lmfit_Ridge")
if g is not None:
    _, make = B.data("7x100")
    shapes["7x100"]["golden_vs_candidate"] = B.diff(g, candidate(make()))
print(B.write("lmfit-1", shapes,
              notes="Ridge alpha=1 batched closed form from one centred Gram matrix; candidate_s includes building a "
                    "fresh Data, so the speedup is a lower bound. 118x1024 baseline is one timed call without the "
                    "helper's warm-up. Agreement means the candidate matches the baseline output, not that either "
                    "value is mathematically right.",
              threads=dict(OMP_NUM_THREADS=B.os.environ.get("OMP_NUM_THREADS"), NUMBA_NUM_THREADS="n/a (no numba)", jidt="n/a")))
