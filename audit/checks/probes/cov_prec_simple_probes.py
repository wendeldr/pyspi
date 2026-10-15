"""cov_prec_simple probes (PROJECT venv): pyspi Covariance / Precision on inputs and
intermediates that the plain manifest SPI path cannot express.

Every function has the pyspi_outputs probe signature ``fn(x, run)``.

* ``spi``        - the SPI named by ``run["identifier"]`` on a transformed input:
                   a stacked P = 19 array (``|stackA`` / ``|stackB`` in the run_id tag)
                   and/or an average-referenced input (``avgref`` in the tag), which has
                   rank P - 1 by construction (EEG average reference).
* ``shrinkage``  - the fitted ``shrinkage_`` of the sklearn estimator pyspi caches in
                   ``data.covariance[estimator]`` (pyspi/statistics/basic.py:31-44), as a
                   scalar; this is how the OAS 2/p omission is quantified directly.
* ``mutation``   - order-independence and aliasing flags for the in-place
                   ``np.fill_diagonal`` on the cached sklearn attribute
                   (pyspi/statistics/basic.py:46-54); ``mutation_others`` repeats the
                   order-independence test for the EllipticEnvelope, MinCovDet,
                   GraphicalLasso and GraphicalLassoCV estimators (other families,
                   same code path).

The stacked-array composition (STACKS) is duplicated in refs/cov_prec_simple.py and
must stay identical; the reference comparison fails if it does not.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "audit" / "checks"))

from regression import harness  # noqa: E402
import make_data  # noqa: E402

# P = 19 stacks of independent datasets (different make_data seeds; bivariate_gaussian,
# scaled_offset and trending share one draw, so at most one of them per stack).
STACKS = {
    "stackA": ["iid_gaussian", "bivariate_gaussian", "sparse_precision", "gauss_ar1_pair", "var2_chain"],
    "stackB": ["delayed_noise", "narrowband_f0-0.1", "narrowband_f0-0.25", "var2_bivariate",
               "bivariate_gaussian", "var2_chain"],
}


def _tag(run):
    return run["run_id"].split("|")[-1]


def stack(name, T):
    return np.vstack([np.asarray(make_data.load(d, T)["x"], dtype=np.float64) for d in STACKS[name]])


def _input(x, run):
    tag = _tag(run)
    for name in STACKS:
        if name in tag:
            x = stack(name, run["T"])
    if "avgref" in tag:
        x = x - x.mean(axis=0, keepdims=True)
    return np.array(x, dtype=np.float64, copy=True)


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


def _build(identifier):
    """'cov-sq_OAS' -> Covariance(estimator='OAS', squared=True) (config.yaml:1-71)."""
    from pyspi.statistics.basic import Covariance, Precision
    head, est = identifier.split("_", 1)
    squared = head.endswith("-sq")
    kind = head.replace("-sq", "")
    cls = {"cov": Covariance, "prec": Precision}[kind]
    return cls(estimator=est, squared=squared)


def _estimator(run):
    return run["unit_id"].split(".")[-1]


def spi(x, run):
    data = _data(_input(x, run), run["path"])
    harness.seed_all(int(run.get("seed", 42)))
    return harness.compute_spi(_build(run["identifier"]), data)


def shrinkage(x, run):
    est = _estimator(run)
    data = _data(_input(x, run), run["path"])
    _build(f"cov_{est}").multivariate(data)
    return np.array(float(data.covariance[est].shrinkage_))


def _same(a, b):
    return bool(np.array_equal(a, b, equal_nan=True))


def _order_flags(x, path, est):
    idents = [f"cov_{est}", f"cov-sq_{est}", f"prec_{est}", f"prec-sq_{est}"]
    fresh = {}
    for ident in idents:
        harness.seed_all(42)
        fresh[ident] = harness.compute_spi(_build(ident), _data(x, path))
    flags = []
    for order in (idents, idents[::-1]):
        data = _data(x, path)
        ok = True
        for ident in order:
            harness.seed_all(42)
            ok &= _same(harness.compute_spi(_build(ident), data), fresh[ident])
        flags.append(float(ok))
    return flags


def mutation(x, run):
    """[order_indep_fwd, order_indep_rev, cov_returned_is_cache, cache_cov_diag_all_nan,
        prec_returned_is_cache, cache_prec_diag_all_nan, score_is_nan_after_prec,
        mahalanobis_nan_fraction_after_prec, score_is_nan_after_cov_call_only]"""
    est = _estimator(run)
    path = run["path"]
    xin = _input(x, run)
    flags = _order_flags(xin, path, est)
    data = _data(xin, path)
    z = data.to_numpy(squeeze=True).T
    harness.seed_all(42)
    R = _build(f"cov_{est}").multivariate(data)
    m = data.covariance[est]
    score_before = m.score(z)
    flags += [float(R is m.covariance_), float(np.isnan(np.diag(m.covariance_)).all())]
    Q = _build(f"prec_{est}").multivariate(data)
    flags += [float(Q is m.precision_), float(np.isnan(np.diag(m.precision_)).all())]
    flags += [float(np.isnan(m.score(z))), float(np.isnan(m.mahalanobis(z)).mean()),
              float(np.isnan(score_before))]
    return np.array(flags)


def mutation_others(x, run):
    """Order-independence (fwd, rev) for estimators outside this family that share
    Estimators._from_cache: EllipticEnvelope, MinCovDet, GraphicalLasso, GraphicalLassoCV."""
    xin = _input(x, run)
    out = []
    for est in ("EllipticEnvelope", "MinCovDet", "GraphicalLasso", "GraphicalLassoCV"):
        out += _order_flags(xin, run["path"], est)
    return np.array(out)
