"""lmfit probe: pyspi LinearModel on a synthetic pair with a prescribed sample correlation r.

u = z(x0); v = z(x1 orthogonalised against u); y_r = r u + sqrt(1 - r^2) v, so corr(u, y_r) = r
to rounding.  For each r on the grid the pair is passed through pyspi's default Data()
(z-score, ddof=1) and LinearModel(model).multivariate; returns (n_r, 3) =
[sample r of the z-scored pair, A[0, 1] (predict y_r from u), A[1, 0]].
The grid and construction are duplicated in refs/lmfit.py (r_grid, grid_pair).
"""
import contextlib
import io

import numpy as np
from scipy.stats import zscore


def r_grid(kind):
    if kind == "enet":
        return np.unique(np.concatenate([np.linspace(0, 1, 201), np.linspace(0.49, 0.52, 601)]))
    return np.linspace(0, 1, 2001)


def grid_pair(x, r):
    u = zscore(x[0], ddof=1)
    v = x[1] - x[1].mean()
    v = v - (v @ u) / (u @ u) * u
    v = zscore(v, ddof=1)
    return np.vstack([u, r * u + np.sqrt(max(1.0 - r * r, 0.0)) * v])


def _grid(x, model, kind):
    from pyspi.data import Data
    from pyspi.statistics.misc import LinearModel
    spi = LinearModel(model)
    rs = r_grid(kind)
    out = np.full((rs.size, 3), np.nan)
    for k, r in enumerate(rs):
        with contextlib.redirect_stdout(io.StringIO()):
            d = Data(grid_pair(x, r), dim_order="ps")
        A = spi.multivariate(d)
        z = d.to_numpy(squeeze=True)
        out[k] = (np.corrcoef(z)[0, 1], A[0, 1], A[1, 0])
    return out


def enet_grid(x, run):
    return _grid(x, "ElasticNet", "enet")


def br_grid(x, run):
    return _grid(x, "BayesianRidge", "br")
