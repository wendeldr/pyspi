"""Probes for the hhg_mgc family (distance.hhg, distance.mgc). PROJECT venv.

Each function is called by ``pyspi_outputs.py`` as ``fn(x, run)``: ``x`` is the raw dataset
array (P, T) (after the optional process selection) and ``run`` the manifest entry. The SPI
is chosen from ``run["unit_id"]`` (distance.hhg -> HellerHellerGorfine, distance.mgc ->
MultiscaleGraphCorrelation). ``run["path"]`` selects the preprocessing exactly as
``pyspi_outputs.make_pyspi_data`` does. The reference script ``refs/hhg_mgc.py`` applies the
same input derivations (``derive``) before computing its own values.

Functions
---------
copies        pyspi matrix on the rows (x0, x0, 2 x0 + 1, x1) (bit-identical copy, affine copy,
              independent row); kernel path = raw rows.
hyppo_exact   attribution aid (same lineage): the delegate (hyppo HHG / scipy
              multiscale_graphcorr as hyppo calls it) fed EXACT distance matrices |x_a - x_b|
              instead of sklearn's dot-product distances, for every ordered pair; (P, P).
optscale      hyppo's MGC path (hyppo.tools.compute_dist + scipy.stats.multiscale_graphcorr,
              reps=0, exactly as hyppo/independence/mgc.py:151-166) with its mgc_dict exported:
              array (3, P, P) = [opt_scale k, opt_scale l, global-scale map value].
ordered       Directed.multivariate (every ordered pair, no Undirected mirror) for the SPI;
              (P, P), NaN diagonal.
small_n       MGC on the first 4 and first 5 samples of rows 0 and 1: [raised ValueError at
              n=4 (1/0), value at n=5].
"""
import io
import contextlib
import warnings

import numpy as np


def _cls(run):
    from pyspi.statistics.distance import HellerHellerGorfine, MultiscaleGraphCorrelation
    return {"distance.hhg": HellerHellerGorfine, "distance.mgc": MultiscaleGraphCorrelation}[run["unit_id"]]


def _data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if path == "kernel":
            return Data(x, dim_order="ps", normalise=False, detrend=False)
        if path == "pipeline":
            return Data(x, dim_order="ps")
        if path == "pipeline-detrend":
            return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def derive(kind, x):
    """Derived rows; mirrored in refs/hhg_mgc.py derive()."""
    x = np.asarray(x, dtype=np.float64)
    if kind == "copies":
        return np.vstack([x[0], x[0].copy(), 2.0 * x[0] + 1.0, x[1]])
    raise ValueError(kind)


def _rows(x, run):
    d = _data(x, run.get("path", "kernel"))
    z = d.to_numpy(squeeze=True)
    return z


def copies(x, run):
    spi = _cls(run)()
    d = _data(derive("copies", x), run.get("path", "kernel"))
    S = np.array(spi.multivariate(d), dtype=np.float64)
    np.fill_diagonal(S, np.nan)
    return S


def hyppo_exact(x, run):
    z = _rows(x, run)
    P = z.shape[0]
    S = np.full((P, P), np.nan)
    if run["unit_id"] == "distance.hhg":
        from hyppo.independence import HHG
        f = lambda dx, dy: HHG(compute_distance=None).statistic(dx, dy)
    else:
        from scipy.stats import multiscale_graphcorr

        def f(dx, dy):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                return multiscale_graphcorr(dx, dy, compute_distance=None, reps=0).stat
    D = [np.abs(z[i][:, None] - z[i][None, :]) for i in range(P)]
    for i in range(P):
        for j in range(P):
            if i != j:
                S[i, j] = f(D[i], D[j])
    return S


def optscale(x, run):
    from hyppo.tools import compute_dist
    from scipy.stats import multiscale_graphcorr
    z = _rows(x, run)
    P = z.shape[0]
    out = np.full((3, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            dx, dy = compute_dist(z[i][:, None], z[j][:, None])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                r = multiscale_graphcorr(dx, dy, compute_distance=None, reps=0)
            k, l = r.mgc_dict["opt_scale"]
            out[0, i, j], out[1, i, j] = float(k), float(l)
            out[2, i, j] = r.mgc_dict["mgc_map"][-1, -1]
    return out


def ordered(x, run):
    from pyspi.base import Directed
    spi = _cls(run)()
    d = _data(x, run.get("path", "kernel"))
    S = np.array(Directed.multivariate(spi, d), dtype=np.float64)
    np.fill_diagonal(S, np.nan)
    return S


def small_n(x, run):
    from pyspi.statistics.distance import MultiscaleGraphCorrelation
    spi = MultiscaleGraphCorrelation()
    raised = 0.0
    try:
        spi.bivariate(_data(x[:2, :4], "kernel"), i=0, j=1)
    except ValueError:
        raised = 1.0
    v5 = float(spi.bivariate(_data(x[:2, :5], "kernel"), i=0, j=1))
    return np.array([raised, v5])
