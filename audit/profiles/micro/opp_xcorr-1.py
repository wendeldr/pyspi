"""xcorr-1: batched rfft cross-correlation from P channel FFTs (sig-False variants only).

Baseline: pyspi CrossCorrelation.bivariate -> scipy.signal.correlate(x, y, 'full') per unordered
pair (method='auto' picks fftconvolve at T=1024, see the 8x1024 pstats), cached per pair on Data.
For a < b the first ordered call is (i=b, j=a), so A[a,b] = A[b,a] = stat(r) with
r = correlate(x_b, x_a) / x_b.std() / x_a.std() / (T-1), lags (T-1)-T//4+1 .. (T-1)+T//4 of the full output.
Candidate (lives only here): one rfft per channel (n = next_fast_len(2T-1)), for each a one batched
irfft of F[b] * conj(F[a]) over b > a, keep the same 2*(T//4) lags, same divisions, mean / max.
-sq identifiers square r elementwise (r**2) before mean / max, as CrossCorrelation.bivariate does.
sigonly=True variants (first non-significant crossing) are not covered.
Agreement with the baseline means only that the candidate reproduces current output.
"""
import sys
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
import _bench as B  # noqa: E402
import numpy as np  # noqa: E402
from scipy import fft as sfft  # noqa: E402


def candidate(d, statistic, squared=False):
    X = d.to_numpy(squeeze=True)                  # (P, T)
    P, T = X.shape
    n = sfft.next_fast_len(2 * T - 1, real=True)
    F = sfft.rfft(X, n, axis=1)
    sd = X.std(axis=1)
    lags = np.arange(-(T // 4) + 1, T // 4 + 1)  # full-output index m = lag + T - 1, m in [T - T//4, T + T//4)
    idx = lags % n
    A = np.full((P, P), np.nan)
    for a in range(P - 1):
        b = np.arange(a + 1, P)
        # c[k] = sum_t x_b[t + k] x_a[t]  ==  correlate(x_b, x_a)[k + T - 1]
        c = sfft.irfft(F[b] * np.conj(F[a])[None, :], n, axis=1)[:, idx]
        r = c / sd[b][:, None] / sd[a] / (T - 1)
        if squared:
            r = r**2
        v = r.mean(axis=1) if statistic == "mean" else r.max(axis=1)
        A[a, b] = v
        A[b, a] = v
    return A


IDS = (("xcorr_mean_sig-False", "mean", False), ("xcorr_max_sig-False", "max", False),
       ("xcorr-sq_mean_sig-False", "mean", True), ("xcorr-sq_max_sig-False", "max", True))
shapes = {}
for shape in ("7x100", "8x1024", "32x1024", "118x1024"):
    raw, make = B.data(shape)
    for ident, stat, sq in IDS:
        reps = 2 if shape == "118x1024" else 3
        S_base, t_base = B.baseline(ident, make, repeats=reps)
        t_cand, S_cand, _ = B.timeit(lambda: candidate(make(), stat, sq), repeats=reps)
        shapes[f"{shape}:{ident}"] = B.result(S_base, t_base, S_cand, t_cand)
        print(shape, ident, shapes[f"{shape}:{ident}"], flush=True)
golden_missing = []
for ident, stat, sq in IDS:
    g = B.golden(ident)
    if g is None:
        golden_missing.append(ident)
        continue
    raw, make = B.data("7x100")
    shapes[f"7x100:golden:{ident}"] = B.diff(g, candidate(make(), stat, sq))
print(B.write("xcorr-1", shapes,
              notes="Timings include Data construction (make()) for both. Baseline = one identifier on a fresh Data "
                    "(pays the per-pair correlate). Candidate covers the four sig-False identifiers (signed and -sq); "
                    "sigonly=True is not covered.", golden_missing=golden_missing))
