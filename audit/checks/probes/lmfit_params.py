"""lmfit probes: export sklearn's fitted attributes for the pyspi LinearModel fits (PROJECT venv).

Each function receives the raw dataset x (P, T), z-scores it exactly as pyspi's default
Data() does (pyspi/data.py:184-195, via Data.to_numpy()), refits the delegate model exactly
as LinearModel.bivariate does (pyspi/statistics/misc.py:117-124: random_state=42 when the
signature has it) and returns a (P, P, 3) array with NaN on the diagonal.
"""
import contextlib
import inspect
import io

import numpy as np


def _z(x):
    from pyspi.data import Data
    with contextlib.redirect_stdout(io.StringIO()):
        d = Data(np.array(x, dtype=np.float64, copy=True), dim_order="ps")
    return d.to_numpy()  # (P, T, 1)


def _fit(model_name, x, attrs):
    from sklearn import linear_model
    z = _z(x)
    P = z.shape[0]
    model = getattr(linear_model, model_name)
    out = np.full((P, P, len(attrs)), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            if "random_state" in inspect.signature(model).parameters:
                m = model(random_state=42).fit(z[i], np.ravel(z[j]))
            else:
                m = model().fit(z[i], np.ravel(z[j]))
            out[i, j] = [float(np.ravel(getattr(m, a))[0]) for a in attrs]
    return out


def lasso_params(x, run):
    """(n_iter_, coef_, intercept_) of Lasso(random_state=42)."""
    return _fit("Lasso", x, ("n_iter_", "coef_", "intercept_"))


def enet_params(x, run):
    """(n_iter_, coef_, intercept_) of ElasticNet(random_state=42)."""
    return _fit("ElasticNet", x, ("n_iter_", "coef_", "intercept_"))


def br_params(x, run):
    """(n_iter_, alpha_, lambda_) of BayesianRidge()."""
    return _fit("BayesianRidge", x, ("n_iter_", "alpha_", "lambda_"))
