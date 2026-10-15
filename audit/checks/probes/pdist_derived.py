"""pdist family probes (PROJECT venv): run pyspi PairwiseDistance on inputs
DERIVED deterministically from a shared dataset, so the reference side
(refs/pdist.py, comparison venv) can rebuild the same derived rows with plain
numpy and compare.  Every derivation here is mirrored in refs/pdist.py
``derive()``; the metric is read from ``run["identifier"]`` (pdist_<metric>).

Paths: "pipeline" = Data(x) (Calculator default: z-score ddof=1), "kernel" =
Data(x, normalise=False, detrend=False), as in pyspi_outputs.make_pyspi_data.
"""
import os
import tempfile
import warnings

import numpy as np

PERM_SEED = 20261014     # joint time permutation
MIX_SEED = 7             # 64 x P mixing matrix for the wide symmetry probe
NEAR_EPS = (1e-3, 1e-5, 1e-7, 1e-9)


def derive(kind, x):
    """Derived input rows (P', T) for each probe kind; mirrored in refs/pdist.py."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[1]
    if kind == "permuted":                      # common random permutation of the time index
        perm = np.random.default_rng(PERM_SEED).permutation(T)
        return x[:, perm]
    if kind == "wide64":                        # 64 correlated rows mixed from the dataset rows
        M = np.random.default_rng(MIX_SEED).standard_normal((64, x.shape[0]))
        return M @ x
    if kind == "negated":                       # flip rows 1 and 3: rho -> -rho for the (0,1), (2,3) pairs
        return x * np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    if kind == "near_identical":                # x0 and x0 + eps * x1 for eps in NEAR_EPS
        return np.vstack([x[0]] + [x[0] + e * x[1] for e in NEAR_EPS])
    if kind == "edge":                          # (x, -x) and two rows sharing exact zeros
        a = x[1].copy(); b = x[2].copy()
        a[::4] = 0.0; b[::4] = 0.0
        return np.vstack([x[0], -x[0], a, b])
    if kind == "signed_affine":                 # (x, -x, 2x+1, independent row)
        return np.vstack([x[0], -x[0], 2.0 * x[0] + 1.0, x[1]])
    raise ValueError(kind)


def _spi(run):
    from pyspi.statistics.distance import PairwiseDistance
    ident = run["identifier"]
    assert ident.startswith("pdist_"), ident
    return PairwiseDistance(metric=ident[len("pdist_"):])


def _data(X, path):
    from pyspi.data import Data
    X = np.array(X, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(X, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(X, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(X, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _compute(kind, x, run):
    X = derive(kind, x)
    S = np.array(_spi(run).multivariate(_data(X, run["path"])), dtype=np.float64, copy=True)
    np.fill_diagonal(S, np.nan)       # as Calculator.compute (pyspi/calculator.py:332)
    return S


def permuted(x, run):
    return _compute("permuted", x, run)


def wide64(x, run):
    return _compute("wide64", x, run)


def negated(x, run):
    return _compute("negated", x, run)


def near_identical(x, run):
    return _compute("near_identical", x, run)


def edge(x, run):
    return _compute("edge", x, run)


def signed_affine(x, run):
    """Matrix on (x, -x, 2x+1, x1) through the requested path; warnings are
    counted separately by signed_affine_warnings."""
    return _compute("signed_affine", x, run)


def signed_affine_warnings(x, run):
    """[n_warnings_total, n_runtime_warnings, S01, S10, S02, S12] for the signed_affine
    input; only warnings raised inside multivariate() (np.errstate(all="warn")) count."""
    X = derive("signed_affine", x)
    spi, data = _spi(run), _data(X, run["path"])     # imports and z-scoring outside the capture
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with np.errstate(all="warn"):
            S = np.array(spi.multivariate(data), dtype=np.float64)
    n_rt = sum(issubclass(m.category, RuntimeWarning) for m in w)
    return np.array([len(w), n_rt, S[0, 1], S[1, 0], S[0, 2], S[1, 2]], dtype=np.float64)


_CFG = """.statistics.distance:
  PairwiseDistance:
    labels:
      - unsigned
      - unordered
      - nonlinear
      - undirected
    dependencies:
    configs:
      - metric: "euclidean"
      - metric: "canberra"
      - metric: "braycurtis"
"""


def calc_downstream(x, run):
    """Full Calculator.compute on (x, -x, 2x+1, x1) with a 3-SPI config (euclidean,
    canberra, braycurtis), then the pyspi correlation-of-SPIs step
    Calculator._get_correlation_df(rmmin=True) (pyspi/calculator.py:427-458).
    Returns a (4, 3) array: rows 0-2 = the 3x3 Spearman SPI-SPI matrix in the
    order (euclidean, canberra, braycurtis); row 3 = [braycurtis table value at
    (0,1) after compute, same entry after _rmmin, number of non-finite entries
    in the correlation matrix]."""
    from pyspi.calculator import Calculator
    X = derive("signed_affine", x)
    fd, cfg = tempfile.mkstemp(suffix=".yaml", prefix="pdist_cfg_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(_CFG)
        calc = Calculator(dataset=X, configfile=cfg)
        calc.compute()
        bc_raw = float(np.asarray(calc.table["pdist_braycurtis"])[0, 1])
        cf = calc._get_correlation_df(rmmin=True)
        bc_rm = float(np.asarray(calc.table["pdist_braycurtis"])[0, 1])
    finally:
        os.unlink(cfg)
    order = ["pdist_euclidean", "pdist_canberra", "pdist_braycurtis"]
    C = cf.loc[order, order].to_numpy(dtype=np.float64)
    out = np.full((4, 3), np.nan)
    out[:3] = C
    out[3] = [bc_raw, bc_rm, float((~np.isfinite(C[~np.eye(3, dtype=bool)])).sum())]
    return out
