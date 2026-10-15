"""Probes for the barycenters family (distance.bary.*). Runs in the PROJECT venv via pyspi_outputs.run_probe.

Every function takes ``(x, run)``: ``x`` is the raw dataset (P, T) after process selection, ``run`` the manifest entry.
``run["identifier"]`` names the Barycenter configuration (mode, squared, statistic) and ``run["path"]`` the
preprocessing (kernel = raw, pipeline = z-score ddof=1, pipeline-detrend = detrend then z-score), mirroring
``pyspi_outputs.make_pyspi_data``. pyspi_outputs has already called ``harness.seed_all(run["seed"])``.

Pair order: pyspi's ``Directed.multivariate`` (base.py:85-94) loops j outer, i inner, so the barycenter of the
unordered pair {a < b} is computed from ``z[[b, a]]`` and cached under both keys (distance.py:247-263).
Functions returning per-pair rows list the pairs as (a, b) with a < b in lexicographic order.
"""
import numpy as np


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


def _spi(identifier):
    from pyspi.statistics.distance import Barycenter
    head, rest = identifier.split("_", 1)
    mode, stat = rest.rsplit("_", 1)
    return Barycenter(mode=mode, squared=(head == "bary-sq"), statistic=stat)


def _pairs(P):
    return [(a, b) for a in range(P) for b in range(a + 1, P)]


def barycenters(x, run):
    """Raw barycenters pyspi cached during ``multivariate``: array (n_pairs, T), rows in _pairs order."""
    data = _data(x, run["path"])
    spi = _spi(run["identifier"])
    spi.multivariate(data)
    cache = data.barycenter[spi._mode]
    return np.stack([np.asarray(cache[(a, b)], dtype=float).ravel() for a, b in _pairs(x.shape[0])])


def cold_matrix(x, run):
    """M[i, j] = Barycenter.bivariate on a FRESH Data object for the ordered pair (i, j), i.e. the statistic of the
    barycenter of z[[i, j]] with no cache. The global numpy RNG is re-seeded with run["seed"] before every call, so for
    sgddtw each ordered pair sees the same RNG draws and M - M.T isolates input-order dependence."""
    P = x.shape[0]
    spi = _spi(run["identifier"])
    M = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            np.random.seed(int(run["seed"]))
            M[i, j] = spi.bivariate(_data(x, run["path"]), i=i, j=j)
    return M


def sgd_twice(x, run):
    """Determinism under the harness seed: seed_all(seed), multivariate on a fresh Data, twice. Returns (2, P, P)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    from regression import harness
    out = []
    for _ in range(2):
        harness.seed_all(int(run["seed"]))
        A = _spi(run["identifier"]).multivariate(_data(x, run["path"]))
        A = np.array(A, dtype=float); np.fill_diagonal(A, np.nan)
        out.append(A)
    return np.stack(out)


def sgd_after_consumer(x, run):
    """RNG-stream dependence: after the harness seed, another stochastic SPI consumes the global RNG first (pyspi
    Barycenter(mode='sgddtw') on iid_gaussian T=64, pipeline path, as an earlier SPI in a Calculator run would), then
    the target identifier is computed on the run's dataset. Returns the target (P, P) matrix."""
    import make_data   # audit/checks is on sys.path inside pyspi_outputs
    xc = np.asarray(make_data.load("iid_gaussian", 64)["x"], dtype=float)
    _spi("bary_sgddtw_mean").multivariate(_data(xc, "pipeline"))
    A = np.array(_spi(run["identifier"]).multivariate(_data(x, run["path"])), dtype=float)
    np.fill_diagonal(A, np.nan)
    return A


def sgd_cache(x, run):
    """Cache/order effect: on ONE shared Data object compute bary_sgddtw_mean first, then run["identifier"]
    (a different sgddtw statistic). Returns concat(A_second.ravel() with NaN diagonal, [rng_advanced]) where
    rng_advanced = 1.0 if the global RNG state changed while the second SPI ran."""
    data = _data(x, run["path"])
    _spi("bary_sgddtw_mean").multivariate(data)
    s0 = np.random.get_state()
    A = np.array(_spi(run["identifier"]).multivariate(data), dtype=float)
    s1 = np.random.get_state()
    np.fill_diagonal(A, np.nan)
    advanced = float(not (np.array_equal(s0[1], s1[1]) and s0[2] == s1[2]))
    return np.concatenate([A.ravel(), [advanced]])


def sgd_seeds(x, run):
    """Seed spread: for seeds 0..49, np.random.seed(s) then multivariate on a fresh Data (P must be 2). Returns the
    cached barycenter of the single pair (computed from z[[1, 0]]) per seed, shape (50, T)."""
    assert x.shape[0] == 2
    rows = []
    for s in range(50):
        np.random.seed(s)
        data = _data(x, run["path"])
        spi = _spi(run["identifier"])
        spi.multivariate(data)
        rows.append(np.asarray(data.barycenter["sgddtw"][(0, 1)], dtype=float).ravel())
    return np.stack(rows)


def softdtw_obj_at_init(x, run):
    """tslearn's soft-DTW barycenter objective and gradient (tslearn/barycenters/softdtw.py:19-35, gamma=1, unit
    weights) evaluated at the Euclidean-mean initialisation of each pair, as softdtw_barycenter builds it from
    z[[b, a]]. Returns (n_pairs, 1 + T): [objective, gradient...]."""
    from tslearn.barycenters.softdtw import _softdtw_func
    from tslearn.barycenters import euclidean_barycenter
    from tslearn.utils import to_time_series_dataset
    z = _data(x, run["path"]).to_numpy(squeeze=True)
    rows = []
    for a, b in _pairs(x.shape[0]):
        X = to_time_series_dataset(z[[b, a]])
        w = np.ones(2)
        b0 = euclidean_barycenter(X, w)
        obj, G = _softdtw_func(b0.ravel(), [X[0], X[1]], w, b0, 1.0)
        rows.append(np.concatenate([[obj], G]))
    return np.stack(rows)
