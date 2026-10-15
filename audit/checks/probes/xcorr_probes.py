"""Probes for the xcorr family (PROJECT venv; called by pyspi_outputs.py as fn(x, run)).

They expose pyspi's own intermediates of CrossCorrelation (pyspi/statistics/basic.py:75-133):
the cached lag vector ``data.xcorr[(i, j)]`` (basic.py:103-113), its peak lag, null exceedance
fractions of pyspi's r against the documented and the implemented sigonly thresholds, and the
error status of the sigonly branch (basic.py:116-120) without letting an exception abort the run
(pyspi_outputs records a raised exception as a failed run, so expected IndexErrors are caught here
and returned as status codes).  ``endlag_pair`` evaluates a constructed pair whose CCF peak lies on
the window end lag T//4 in both channel orders (end-lag order dependence of max_sig-False).

Path handling mirrors audit/checks/pyspi_outputs.py:make_pyspi_data: ``kernel`` =
Data(normalise=False, detrend=False), ``pipeline`` = Data() default (z-score ddof=1),
``pipeline-detrend`` = Data(detrend=True).  The identifier in the run (when given) is parsed into
CrossCorrelation(squared, statistic, sigonly).
"""
import numpy as np

STATUS = {"ok": 0, "IndexError": 1, "ValueError": 2, "other": 3, "nonfinite": 4}


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
    from pyspi.statistics.basic import CrossCorrelation
    if identifier is None:
        return CrossCorrelation(statistic="max", sigonly=False)
    squared = identifier.startswith("xcorr-sq")
    statistic = "max" if "_max_" in identifier else "mean"
    sigonly = identifier.endswith("sig-True")
    spi = CrossCorrelation(squared=squared, statistic=statistic, sigonly=sigonly)
    assert spi.identifier == identifier, (spi.identifier, identifier)
    return spi


def _cache_after_multivariate(x, path):
    """Run multivariate of a sigonly=False SPI (never raises) and return (data, T, L)."""
    data = _data(x, path)
    _spi(None).multivariate(data)
    T = data.n_observations
    return data, T, T // 4


def cached_r(x, run):
    """pyspi's cached r for every unordered pair a<b, in pair order (0,1),(0,2),...; shape
    (n_pairs, 2*(T//4)).  The cache key (a, b) holds the array multivariate computed first."""
    data, T, L = _cache_after_multivariate(x, run.get("path", "pipeline"))
    P = data.n_processes
    rows = [np.asarray(data.xcorr[(a, b)], dtype=np.float64) for a in range(P) for b in range(a + 1, P)]
    return np.vstack(rows)


def peak_lag_cached(x, run):
    """(P, P) lag of max r in the cache used by multivariate; lag of position p is p - (T//4 - 1)."""
    data, T, L = _cache_after_multivariate(x, run.get("path", "pipeline"))
    P = data.n_processes
    M = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                M[i, j] = int(np.argmax(data.xcorr[(i, j)])) - (L - 1)
    return M


def peak_lag_fresh(x, run):
    """(P, P) lag of max r when bivariate(i, j) is evaluated FIRST on a fresh Data, i.e.
    r = correlate(x_i, x_j) for the ordered pair (no cache sharing with (j, i))."""
    spi = _spi(None)
    P = x.shape[0]
    M = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            data = _data(x, run.get("path", "pipeline"))
            spi.bivariate(data, i=i, j=j)
            L = data.n_observations // 4
            M[i, j] = int(np.argmax(data.xcorr[(i, j)])) - (L - 1)
    return M


def null_exceed(x, run):
    """Exceedance counts of pyspi's cached r over every pair and every window lag:
    [n_lags, n(|r| > 1.96/sqrt(T)) (documented), n(|r| > 1.96/sqrt(T//4)) (implemented, basic.py:118-119)]."""
    data, T, L = _cache_after_multivariate(x, run.get("path", "pipeline"))
    P = data.n_processes
    R = np.concatenate([np.abs(data.xcorr[(a, b)]) for a in range(P) for b in range(a + 1, P)])
    return np.array([R.size, np.sum(R > 1.96 / np.sqrt(T)), np.sum(R > 1.96 / np.sqrt(L))], dtype=np.float64)


def _code(exc):
    if isinstance(exc, IndexError):
        return STATUS["IndexError"]
    if isinstance(exc, ValueError):
        return STATUS["ValueError"]
    return STATUS["other"]


def sig_status(x, run):
    """(P, P) status codes for the run's identifier on the run's path.
    Off-diagonal [a, b] (a != b): bivariate evaluated in the multivariate orientation
    (i = max(a, b), j = min(a, b), i.e. r = correlate(x_hi, x_lo)) on a fresh Data:
    0 ok (finite), 1 IndexError, 2 ValueError, 3 other exception, 4 non-finite value.
    Diagonal: status of spi.multivariate on the whole dataset (what Calculator.compute calls,
    pyspi/calculator.py:329; any exception there makes the whole matrix NaN, :336-337)."""
    spi = _spi(run["identifier"])
    path = run.get("path", "pipeline")
    P = x.shape[0]
    M = np.zeros((P, P))
    for a in range(P):
        for b in range(a + 1, P):
            data = _data(x, path)
            try:
                v = spi.bivariate(data, i=b, j=a)
                c = STATUS["ok"] if np.isfinite(v) else STATUS["nonfinite"]
            except Exception as exc:  # noqa: BLE001 - the error class is the measurement
                c = _code(exc)
            M[a, b] = M[b, a] = c
    try:
        A = spi.multivariate(_data(x, path))
        d = STATUS["ok"] if np.all(np.isfinite(A[~np.eye(P, dtype=bool)])) else STATUS["nonfinite"]
    except Exception as exc:  # noqa: BLE001
        d = _code(exc)
    np.fill_diagonal(M, d)
    return M


def crafted_r(T, case):
    """Deterministic crafted lag vectors on pyspi's window (lags -(T//4-1)..T//4); shared formula with
    refs/xcorr.py.  case 'A': r = 0.9 for |lag| <= 2, else 0 (significant only near lag 0);
    case 'B': r = 0.9 for lags -(T//4-2)..0, 0 at lag -(T//4-1) and at lags >= +1 (fzr = 0, fzf = 0)."""
    L = T // 4
    lags = np.arange(-(L - 1), L + 1)
    if case == "A":
        return np.where(np.abs(lags) <= 2, 0.9, 0.0)
    return np.where((lags >= -(L - 2)) & (lags <= 0), 0.9, 0.0)


def crafted_window(x, run):
    """Inject crafted cached r arrays into data.xcorr and evaluate the run's SPI on the pair (1, 0)
    (cache read at basic.py:98-99, window rule basic.py:116-120).
    Returns [value_A, status_A, value_B, status_B] (value NaN when an exception was raised)."""
    spi = _spi(run["identifier"])
    out = []
    for case in ("A", "B"):
        data = _data(x, run.get("path", "pipeline"))
        r = crafted_r(data.n_observations, case)
        data.xcorr = {(1, 0): r, (0, 1): r}
        try:
            v = float(spi.bivariate(data, i=1, j=0))
            c = STATUS["ok"] if np.isfinite(v) else STATUS["nonfinite"]
        except Exception as exc:  # noqa: BLE001
            v, c = float("nan"), _code(exc)
        out += [v, c]
    return np.array(out)


def endlag_pair(x, run):
    """Constructed pair whose CCF peak lies exactly on the window end lag (review follow-up for the
    max_sig-False candidate 'asymmetric window -(T//4-1)..T//4 makes the value channel-order
    dependent through the end lag', basic.py:107).  x0 = x[0]; y = np.roll(x0, L) with L = T//4
    (y(t) = x0(t - L) for t >= L).  The run's SPI is evaluated with multivariate on [x0, y]
    (forward: pyspi's cached r is correlate(x_hi = y, x_lo = x0), peak at lag +L, inside the window)
    and on [y, x0] (reversed: correlate(x0, y), peak at lag -L, outside the window).
    Returns [A_fwd[0, 1], A_rev[0, 1]]."""
    x0 = np.asarray(x[0], dtype=np.float64)
    L = x0.size // 4
    y = np.roll(x0, L)
    spi = _spi(run["identifier"])
    out = []
    for arr in (np.vstack([x0, y]), np.vstack([y, x0])):
        data = _data(arr, run.get("path", "pipeline"))
        A = np.asarray(spi.multivariate(data), dtype=np.float64)
        out.append(float(A[0, 1]))
    return np.array(out)
