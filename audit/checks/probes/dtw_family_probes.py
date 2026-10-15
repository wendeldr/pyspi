"""dtw_family probes (PROJECT venv): pyspi / tslearn quantities that the plain SPI path cannot produce.

Every derivation of input rows is mirrored in refs/dtw_family.py ``derive()``. The run tag (5th field
of the run_id) selects the derivation; the identifier (when given) selects the pyspi SPI.

Functions (signature ``fn(x, run)``, x = raw dataset rows (P, T)):

  tslearn_mask   tslearn.metrics._masks.compute_mask(T, T, code) as a 0/1 float array: the mask that the
                 numpy DTW / LCSS paths receive (code 1 = itakura, 2 = sakoe_chiba, chosen from the unit_id;
                 radius / slope None -> tslearn defaults 1 / 2.0, _masks.py:339-348)
  unmirrored     pyspi.base.Directed.multivariate(spi, Data) on the run's path: every ordered pair (i, j)
                 computed by tslearn separately and NOT mirrored (Undirected.multivariate, base.py:132-138,
                 is bypassed) -> (P, P), NaN diagonal; measures the argument symmetry of the delegate
  derived_spi    pyspi SPI (identifier) on derived rows; path kernel -> Data(x', normalise=False),
                 otherwise Data(x') (z-score ddof=1); tags null-neg, near, offset, ident
  raw_call       spi.bivariate(x0, x1) and spi.bivariate(x1, x0) with RAW arrays (the two-array API,
                 pyspi/base.py:27-49) -> length-2 vector
  torch_lcss     tslearn.metrics.lcss(..., be='pytorch') on Data(x)-z-scored rows (mask-honouring branch,
                 tslearn/metrics/dtw_variants.py:2306) -> (P, P) matrix, NaN diagonal, both orders computed
  softdtw_gamma  tslearn.metrics.soft_dtw(z_i, z_j, gamma=g) on Data(x)-z-scored rows, g from the tag
                 '<name>-<g>' -> (P, P) matrix, NaN diagonal, both orders computed
"""
import numpy as np


def derive(kind, x):
    """Derived rows (mirrored in refs/dtw_family.py)."""
    x = np.asarray(x, dtype=np.float64)
    if kind == "null-neg":       # x0, its negation (perfect anti-correlation), two independent rows
        return np.vstack([x[0], -x[0], x[1], x[2]])
    if kind == "near":           # near-identical rows: cost cells close to 0 (dot-product cancellation)
        return np.vstack([x[0], x[0] + 1e-3 * x[1], x[0] + 1e-6 * x[1]])
    if kind == "offset":         # large common offset, used on the kernel path (no z-scoring)
        return np.vstack([x[0] + 1e4, x[1] + 1e4, x[2] + 1e4])
    if kind == "ident":          # x0 twice plus an independent row: f(x, x) on the off-diagonal
        return np.vstack([x[0], x[0], x[1]])
    raise ValueError(kind)


def _tag(run):
    """Run tag = 5th field of the run_id (<unit>|<dataset>|<T>|<path>|<tag>[|<identifier>])."""
    return run["run_id"].split("|")[4]


def _spi(identifier):
    from pyspi.statistics.distance import (DynamicTimeWarping, LongestCommonSubsequence,
                                           SoftDynamicTimeWarping)
    base, _, con = identifier.partition("_constraint-")
    cls = {"dtw": DynamicTimeWarping, "lcss": LongestCommonSubsequence,
           "softdtw": SoftDynamicTimeWarping}[base]
    gc = {"": None, "itakura": "itakura", "sakoe-chiba": "sakoe_chiba"}[con]
    spi = cls(global_constraint=gc)
    assert spi.identifier == identifier, (spi.identifier, identifier)
    return spi


def _data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    return Data(x, dim_order="ps")


def _zrows(x):
    return _data(x, "pipeline").to_numpy(squeeze=True)


def tslearn_mask(x, run):
    from tslearn.metrics._masks import compute_mask
    T = x.shape[1]
    code = 1 if "itakura" in run["unit_id"] else 2
    return np.asarray(compute_mask(T, T, code), dtype=np.float64)


def unmirrored(x, run):
    from pyspi.base import Directed
    spi = _spi(run["identifier"])
    A = Directed.multivariate(spi, _data(x, run.get("path", "pipeline")))
    A = np.array(A, dtype=np.float64, copy=True)
    np.fill_diagonal(A, np.nan)
    return A


def derived_spi(x, run):
    xd = derive(_tag(run), x)
    A = np.asarray(_spi(run["identifier"]).multivariate(_data(xd, run.get("path", "pipeline"))), dtype=np.float64)
    A = A.copy()
    np.fill_diagonal(A, np.nan)
    return A


def raw_call(x, run):
    spi = _spi(run["identifier"])
    x0 = np.array(x[0], dtype=np.float64, copy=True)
    x1 = np.array(x[1], dtype=np.float64, copy=True)
    return np.array([spi.bivariate(x0, x1), spi.bivariate(x1, x0)], dtype=np.float64)


def torch_lcss(x, run):
    import torch
    from tslearn.metrics import lcss
    z = _zrows(x)
    gc = "itakura" if "itakura" in run["unit_id"] else "sakoe_chiba"
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = float(lcss(torch.tensor(z[i][:, None]), torch.tensor(z[j][:, None]),
                                     global_constraint=gc, be="pytorch"))
    return A


def softdtw_gamma(x, run):
    from tslearn.metrics import soft_dtw
    g = float(_tag(run).split("-", 1)[1])
    z = _zrows(x)
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = float(soft_dtw(z[i], z[j], gamma=g))
    return A
