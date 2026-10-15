"""dcorr_hsic family probes (PROJECT venv).

``unmirrored(x, run)`` returns the matrix pyspi's ``bivariate`` produces BEFORE
``Undirected.multivariate`` mirrors the upper triangle into the lower one
(pyspi/base.py:132-138): A[i, j] = SPI.bivariate(data, i=i, j=j), i.e. hyppo's
statistic with x = process i and y = process j, for every ordered pair, NaN on the
diagonal. This exposes the argument-order behaviour of the delegate
(hyppo.independence.Dcorr / Hsic), which the stored matrix hides.

The SPI is built from ``run["identifier"]`` (dcorr, dcorr_biased, hsic, hsic_biased)
with the constructor arguments recorded in audit/units.yaml; the data path follows
pyspi_outputs.make_pyspi_data ("kernel" = raw rows, "pipeline" = z-score ddof=1,
"pipeline-detrend" = scipy detrend then z-score).
"""
import numpy as np

_PARAMS = {"dcorr": ("DistanceCorrelation", False), "dcorr_biased": ("DistanceCorrelation", True),
           "hsic": ("HilbertSchmidtIndependenceCriterion", False),
           "hsic_biased": ("HilbertSchmidtIndependenceCriterion", True)}


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


def _spi(identifier):
    import pyspi.statistics.distance as dist
    cls, biased = _PARAMS[identifier]
    return getattr(dist, cls)(biased=biased)


def unmirrored(x, run):
    spi = _spi(run["identifier"])
    data = _data(x, run.get("path", "kernel"))
    P = data.n_processes
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = float(spi.bivariate(data, i=i, j=j))
    return A


def gamma_used(x, run):
    """Gaussian-kernel gamma that hyppo actually applies to the SECOND argument's kernel.

    For every ordered pair (i, j) on the run's data path, calls hyppo.tools.common.compute_kern(x_i, x_j,
    metric='gaussian') exactly as hyppo.independence.Hsic.statistic does (hsic.py:117) and recovers gamma from the
    returned y-kernel L: gamma = median over off-diagonal entries of -log(L_ab) / (y_a - y_b)^2 (entries with
    (y_a - y_b)^2 > 1e-12 and L_ab > 1e-300). G[i, j] = gamma used for y = process j when x = process i; NaN diagonal.
    If the bandwidth is taken from x only, G[i, j] = 1 / (2 med(|x_i,a - x_i,b|)^2) for every j.
    """
    from hyppo.tools.common import compute_kern
    data = _data(x, run.get("path", "pipeline"))
    X = data.to_numpy(squeeze=True)
    P = X.shape[0]
    G = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            xi, yj = X[i][:, None], X[j][:, None]
            _, L = compute_kern(xi, yj, metric="gaussian")
            d2 = (yj - yj.T) ** 2
            m = (d2 > 1e-12) & (L > 1e-300)
            G[i, j] = float(np.median(-np.log(L[m]) / d2[m]))
    return G
