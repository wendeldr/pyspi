"""lmfit probes for SGDRegressor (PROJECT venv)."""
import contextlib
import io
import random

import numpy as np


def _data(x):
    from pyspi.data import Data
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(np.array(x, dtype=np.float64, copy=True), dim_order="ps")


def determinism(x, run):
    """LinearModel('SGDRegressor') recomputed after three different global RNG seeds -> (3, P, P)."""
    from pyspi.statistics.misc import LinearModel
    out = []
    for s in (42, 0, 12345):
        np.random.seed(s)
        random.seed(s)
        A = LinearModel("SGDRegressor").multivariate(_data(x))
        np.fill_diagonal(A, np.nan)
        out.append(A)
    return np.stack(out)


def seed_sweep(x, run):
    """sklearn SGDRegressor(random_state=s), s = 0..19, fitted as LinearModel does -> (20, P, P)."""
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error
    z = _data(x).to_numpy()
    P = z.shape[0]
    out = np.full((20, P, P), np.nan)
    for s in range(20):
        for i in range(P):
            for j in range(P):
                if i != j:
                    m = SGDRegressor(random_state=s).fit(z[i], np.ravel(z[j]))
                    out[s, i, j] = mean_squared_error(m.predict(z[i]), np.ravel(z[j]))
    return out
