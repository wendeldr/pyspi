"""Probes for the cov_prec_glasso family (PROJECT venv; called by pyspi_outputs.run_probe).

Every function has the signature ``fn(x, run)``: ``x`` is the raw (P, T) dataset
and ``run`` the manifest entry.  ``run["path"]`` selects the same preprocessing as
``pyspi_outputs.make_pyspi_data`` (kernel / pipeline / pipeline-detrend) and
``run["identifier"]`` names the SPI (cov|prec[-sq]_<GraphicalLasso|GraphicalLassoCV>).
The probes go through pyspi's own code path (``Estimators._from_cache`` /
``multivariate`` in pyspi/statistics/basic.py:31-54, harness.compute_spi which
mirrors Calculator.compute at pyspi/calculator.py:329-338) and only read what the
fitted sklearn estimator already holds.  Nothing is refitted with other settings.

Return layouts (1-D float arrays unless stated; the reference script
refs/cov_prec_glasso.py parses them):

full_fit -> header[9] + W.ravel() + Theta.ravel() + cv_alphas + cv_mean + cv_splits.ravel()
    header = [P, err_code, n_iter_, alpha, final_dual_gap, n_cv, n_splits, n_warnings_seen,
              n_convergence_warnings_direct]
    err_code: 0 ok, 1 FloatingPointError, 2 other exception.
    W / Theta are copies of covariance_ / precision_ taken BEFORE multivariate()
    writes NaN into their diagonals.  n_warnings_seen counts warnings that reach a
    caller through pyspi.calculation._capture_metric_warnings (the HEAD Calculator
    hook) while pyspi fits the estimator; n_convergence_warnings_direct counts the
    sklearn ConvergenceWarnings emitted when the same estimator class is fitted
    directly (library defaults, same (T, P) array, warnings.simplefilter("always")),
    i.e. the warnings pyspi's basic.py:37-38 suppresses.
calc_style -> [P, err_code] + S.ravel()   (S = Calculator-style output: all-NaN on exception)
perm -> (P, P) matrix: the SPI computed on x[p] mapped back to the original order.
side_effects -> [n_fits, n_allnan_identifiers, diag_mutated, returned_is_cached,
                 value_neutral, n_identifiers]
"""
import warnings

import numpy as np

ESTIMATORS = ("GraphicalLasso", "GraphicalLassoCV")


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


def _parse(identifier):
    """'cov-sq_GraphicalLassoCV' -> ('cov', True, 'GraphicalLassoCV')."""
    head, est = identifier.split("_", 1)
    squared = head.endswith("-sq")
    kind = head.replace("-sq", "")
    assert kind in ("cov", "prec") and est in ESTIMATORS, identifier
    return kind, squared, est


def _spi(kind, squared, est):
    from pyspi.statistics.basic import Covariance, Precision
    cls = Covariance if kind == "cov" else Precision
    return cls(estimator=est, squared=squared)


def _err_code(exc):
    return 1 if isinstance(exc, FloatingPointError) else 2


def full_fit(x, run):
    from pyspi.calculation import _capture_metric_warnings
    _, _, est = _parse(run["identifier"])
    P = x.shape[0]
    data = _data(x, run["path"])
    spi = _spi("prec", False, est)
    err, n_iter, alpha, gap, n_warn = 0, np.nan, np.nan, np.nan, 0
    W = np.full((P, P), np.nan)
    Th = np.full((P, P), np.nan)
    cv_a, cv_m, cv_s = np.zeros(0), np.zeros(0), np.zeros((0, 0))
    with _capture_metric_warnings() as emitted:
        try:
            mycov = spi._from_cache(data)          # pyspi's own fit (warnings silenced inside)
        except Exception as exc:                   # noqa: BLE001 - recorded, not hidden
            err = _err_code(exc)
            mycov = None
    n_warn = len(emitted)
    import sklearn.covariance as skc
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as direct:
        warnings.simplefilter("always")
        try:
            getattr(skc, est)().fit(data.to_numpy(squeeze=True).T)
        except Exception:                          # noqa: BLE001 - the failure itself is recorded via err
            pass
    n_direct = sum(issubclass(w.category, ConvergenceWarning) for w in direct)
    if mycov is not None:
        W = np.array(mycov.covariance_, dtype=np.float64, copy=True)
        Th = np.array(mycov.precision_, dtype=np.float64, copy=True)
        n_iter = float(mycov.n_iter_)
        alpha = float(mycov.alpha_ if est == "GraphicalLassoCV" else mycov.alpha)
        if getattr(mycov, "costs_", None):
            gap = float(mycov.costs_[-1][1])
        if est == "GraphicalLassoCV":
            res = mycov.cv_results_
            cv_a = np.asarray(res["alphas"], dtype=np.float64)
            cv_m = np.asarray(res["mean_test_score"], dtype=np.float64)
            ks = sorted(k for k in res if k.startswith("split") and k.endswith("_test_score"))
            cv_s = np.array([np.asarray(res[k], dtype=np.float64) for k in ks])
    header = np.array([P, err, n_iter, alpha, gap, cv_a.size, cv_s.shape[0], n_warn, n_direct], dtype=np.float64)
    return np.concatenate([header, W.ravel(), Th.ravel(), cv_a, cv_m, cv_s.ravel()])


def calc_style(x, run):
    from regression import harness
    kind, squared, est = _parse(run["identifier"])
    P = x.shape[0]
    data = _data(x, run["path"])
    err = 0
    try:
        S = harness.compute_spi(_spi(kind, squared, est), data)
    except Exception as exc:                       # noqa: BLE001 - Calculator.compute does the same
        err = _err_code(exc)
        S = np.full((P, P), np.nan)
    return np.concatenate([[P, err], np.asarray(S, dtype=np.float64).ravel()])


def perm(x, run):
    from regression import harness
    kind, squared, est = _parse(run["identifier"])
    P = x.shape[0]
    p = np.random.default_rng(12345).permutation(P)
    if np.array_equal(p, np.arange(P)):
        p = p[::-1].copy()
    Sp = harness.compute_spi(_spi(kind, squared, est), _data(x[p], run["path"]))
    B = np.full((P, P), np.nan)
    B[np.ix_(p, p)] = Sp                            # Sp[a, b] belongs to original processes p[a], p[b]
    return B


def side_effects(x, run):
    import sklearn.covariance as skc
    from regression import harness
    _, _, est = _parse(run["identifier"])
    P = x.shape[0]
    cls = getattr(skc, est)
    orig_fit = cls.fit
    count = [0]

    def counting_fit(self, *a, **k):
        count[0] += 1
        return orig_fit(self, *a, **k)

    order1 = [("cov", False), ("cov", True), ("prec", False), ("prec", True)]   # pyspi/config.yaml order
    order2 = [("prec", False), ("prec", True), ("cov", False), ("cov", True)]

    def compute_all(order, data):
        out = {}
        for kind, sq in order:
            try:
                out[(kind, sq)] = harness.compute_spi(_spi(kind, sq, est), data)
            except Exception:                       # noqa: BLE001 - Calculator.compute does the same
                out[(kind, sq)] = np.full((P, P), np.nan)
        return out

    cls.fit = counting_fit
    try:
        d1 = _data(x, run["path"])
        out1 = compute_all(order1, d1)
        n_fits = count[0]
    finally:
        cls.fit = orig_fit
    n_allnan = sum(int(np.isnan(v).all()) for v in out1.values())
    mutated = returned_is_cached = np.nan
    cache = getattr(d1, "covariance", {}) or {}
    if est in cache:
        c = cache[est]
        mutated = float(np.isnan(np.diag(c.precision_)).all() and np.isnan(np.diag(c.covariance_)).all())
        returned_is_cached = float(_spi("prec", False, est).multivariate(d1) is c.precision_)
    d2 = _data(x, run["path"])
    out2 = compute_all(order2, d2)
    neutral = float(all(np.array_equal(out1[k], out2[k], equal_nan=True) for k in out1))
    return np.array([n_fits, n_allnan, mutated, returned_is_cached, neutral, len(order1)], dtype=np.float64)
