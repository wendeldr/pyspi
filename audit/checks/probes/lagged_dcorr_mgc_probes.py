"""Probes for the lagged_dcorr_mgc family (PROJECT venv; called by pyspi_outputs.py as fn(x, run)).

Units: distance.dcorrx.maxlag-{1,10} (pyspi CrossDistanceCorrelation -> hyppo 0.5.2 DcorrX) and
distance.mgcx.maxlag-{1,10} (CrossMultiscaleGraphCorrelation -> hyppo MGCX -> scipy 1.17.1
multiscale_graphcorr). pyspi: pyspi/statistics/distance.py:114-151 at HEAD a42ebc1 (bivariate(i, j)
passes x = z[i], y = z[j], each (T, R), to .statistic and keeps only the statistic).

The probes expose what the plain SPI path cannot:

  antisym      A - A.T of the pyspi matrix (harness.build_spi path: harness.compute_spi on the run's
               preprocessing path), so the 'direction' criterion can test the sign of the asymmetric part.
  lagterms     the weighted per-lag terms of hyppo's compute_stat (time_series/_utils.py:85-108),
               recomputed by calling hyppo's own Dcorr / MGC exactly as compute_stat does, stacked
               with the DcorrX / MGCX .statistic output: shape (L + 2, P, P); slice l = weighted
               lag-l term for the ordered pair (i, j) = (x = z[i], y = z[j]); slice L + 1 = statistic.
  mgcmap       scipy's compiled _local_correlations map (scipy/stats/_stats.pyx, not shipped) for the
               lag-0 Euclidean distance matrices of processes (0, 1) as hyppo builds them.
  lag0_mgc     (2, P, P): slice 0 = MGCX(max_lag=0).statistic per ordered pair (the lag-0 term of
               mgcx), slice 1 = the pyspi mgc SPI (MultiscaleGraphCorrelation) matrix.
  calls        multiscale_graphcorr call count per ordered pair, with and without the
               compute_scale_at_lag step, global-RNG state check and timing (efficiency candidate).
  smalln       truncation of processes (0, 1) to n = L+1 .. L+6 samples: rows (n, code, value).
  multireal    multi-realisation Data: the (P, T) dataset is split into R = 2 realisations of T/2
               samples, Data(x3, dim_order='psr', normalise=False), then the SPI matrix.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/media/dan/Data/git/pyspi/tests")

ERR = {"ok": 0, "ZeroDivisionError": 1, "ValueError": 2, "IndexError": 3, "other": 4}


def _data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(x, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _kind_lag(identifier):
    kind, lag = identifier.split("_maxlag-")
    return kind, int(lag)


def _spi(identifier):
    from pyspi.statistics.distance import CrossDistanceCorrelation, CrossMultiscaleGraphCorrelation
    kind, lag = _kind_lag(identifier)
    return (CrossDistanceCorrelation if kind == "dcorrx" else CrossMultiscaleGraphCorrelation)(max_lag=lag)


def antisym(x, run):
    from regression import harness
    A = harness.compute_spi(_spi(run["identifier"]), _data(x, run["path"]))
    return A - A.T


def _hyppo_terms(a, b, kind, L):
    """compute_stat (hyppo/time_series/_utils.py:85-108) unrolled with hyppo's own objects."""
    from hyppo.independence import Dcorr, MGC
    from hyppo.tools import compute_dist
    test = (Dcorr if kind == "dcorrx" else MGC)(compute_distance="euclidean")      # _utils.py:92
    distx, disty = compute_dist(a, b, metric="euclidean")                         # _utils.py:88
    terms = [test.statistic(a, b)]                                                # _utils.py:93
    n = distx.shape[0]
    for j in range(1, L + 1):                                                     # _utils.py:98-102
        terms.append((n - j) * test.statistic(distx[j:n, j:n], disty[0:n - j, 0:n - j]) / n)
    return terms


def lagterms(x, run):
    from hyppo.time_series import DcorrX, MGCX
    kind, L = _kind_lag(run["identifier"])
    z = _data(x, run["path"]).to_numpy()            # (P, T, 1)
    P = z.shape[0]
    out = np.full((L + 2, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            a, b = z[i], z[j]
            out[:L + 1, i, j] = _hyppo_terms(a, b, kind, L)
            out[L + 1, i, j] = (DcorrX if kind == "dcorrx" else MGCX)(max_lag=L).statistic(a, b)[0]
    return out


def mgcmap(x, run):
    from hyppo.tools import compute_dist
    from scipy.stats._stats import _local_correlations
    z = _data(x, run["path"]).to_numpy()
    dx, dy = compute_dist(z[0], z[1], metric="euclidean")
    return np.asarray(_local_correlations(dx, dy, global_corr="mgc"))


def lag0_mgc(x, run):
    from regression import harness
    from hyppo.time_series import MGCX
    from pyspi.statistics.distance import MultiscaleGraphCorrelation
    data = _data(x, run["path"])
    z = data.to_numpy()
    P = z.shape[0]
    out = np.full((2, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                out[0, i, j] = MGCX(max_lag=0).statistic(z[i], z[j])[0]
    out[1] = harness.compute_spi(MultiscaleGraphCorrelation(), data)
    return out


def calls(x, run):
    """[calls per ordered pair with MGCX.statistic, calls with compute_stat alone, max |stat difference|,
    global RNG state unchanged (1/0), mean time MGCX.statistic (s), mean time compute_stat alone (s)]."""
    import hyppo.independence.mgc as hm
    from hyppo.independence import MGC
    from hyppo.time_series import MGCX
    from hyppo.time_series._utils import compute_stat
    kind, L = _kind_lag(run["identifier"])
    z = _data(x, run["path"]).to_numpy()
    orig = hm.multiscale_graphcorr
    count = [0]

    def counted(*a, **k):
        count[0] += 1
        return orig(*a, **k)

    hm.multiscale_graphcorr = counted
    try:
        n_full, n_stat, dmax, rng_ok, t_full, t_stat, npair = 0, 0, 0.0, 1, 0.0, 0.0, 0
        for i in range(z.shape[0]):
            for j in range(z.shape[0]):
                if i == j:
                    continue
                npair += 1
                st0 = np.random.get_state()
                count[0] = 0
                t0 = time.perf_counter()
                s_full = MGCX(max_lag=L).statistic(z[i], z[j])[0]
                t_full += time.perf_counter() - t0
                n_full += count[0]
                st1 = np.random.get_state()
                rng_ok &= int(st0[0] == st1[0] and all(np.array_equal(u, v) for u, v in zip(st0[1:3], st1[1:3])))
                count[0] = 0
                t0 = time.perf_counter()
                s_stat = compute_stat(z[i], z[j], MGC, "euclidean", L)[0]
                t_stat += time.perf_counter() - t0
                n_stat += count[0]
                dmax = max(dmax, abs(s_full - s_stat))
    finally:
        hm.multiscale_graphcorr = orig
    return np.array([n_full / npair, n_stat / npair, dmax, rng_ok, t_full / npair, t_stat / npair])


def smalln(x, run):
    """Rows (n, code, value): processes (0, 1) truncated to the first n samples, raw input."""
    from hyppo.time_series import DcorrX, MGCX
    kind, L = _kind_lag(run["identifier"])
    cls = DcorrX if kind == "dcorrx" else MGCX
    rows = []
    for n in range(L + 1, L + 7):
        a, b = x[0, :n, None].copy(), x[1, :n, None].copy()
        try:
            v = float(cls(max_lag=L).statistic(a, b)[0])
            code = ERR["ok"]
        except Exception as exc:     # the failure type is the measurement
            v = np.nan
            code = ERR.get(type(exc).__name__, ERR["other"])
        rows.append((n, code, v))
    return np.array(rows, dtype=float)


def multireal(x, run):
    """pyspi matrix on Data with R = 2 realisations: x3[p, t, r] = x[p, r * T/2 + t]."""
    from pyspi.data import Data
    from regression import harness
    P, T = x.shape
    h = T // 2
    x3 = np.stack([x[:, :h], x[:, h:2 * h]], axis=-1)
    data = Data(np.array(x3, dtype=np.float64), dim_order="psr", normalise=False, detrend=False)
    assert data.to_numpy().shape == (P, h, 2)
    return harness.compute_spi(_spi(run["identifier"]), data)
