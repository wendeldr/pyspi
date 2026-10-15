"""gpfit probes (PROJECT venv): expose what sklearn's GaussianProcessRegressor does inside
pyspi's GPModel (pyspi/statistics/misc.py:129-147).

Every probe receives the raw dataset x (P, T) after process selection and the manifest run
dict.  The kernel ('RBF' or 'DotProduct') is taken from the run's unit_id suffix, the
preprocessing from run["path"] exactly as audit/checks/pyspi_outputs.py:make_pyspi_data does
(kernel = Data(normalise=False, detrend=False); pipeline = Data(); pipeline-detrend =
Data(detrend=True)).  Each fit replicates misc.py:143-147 line for line:
``GaussianProcessRegressor(kernel=GPModel(kernel)._kernel).fit(z[i], ravel(z[j]))``,
``predict(z[i])``, ``mean_squared_error``.

Functions
  params(x, run)       -> (P, P, 12): per ordered pair (input i, target j)
       0 mse              in-sample MSE of the default fit (what pyspi stores in A[i, j])
       1-3 theta          kernel_.theta = log[c, w, l | sigma_0] (sklearn order, see below)
       4 lml              log_marginal_likelihood_value_
       5 n_restarts       gp.n_restarts_optimizer (pyspi leaves the default, 0)
       6 mse_rec          MSE of a refit whose optimizer is a recording callable that runs
                          exactly sklearn's call scipy.optimize.minimize(obj, theta0,
                          method='L-BFGS-B', jac=True, bounds=bounds) (_gpr.py:658-667)
       7 nit              L-BFGS-B iterations of that run
       8 nfev             L-BFGS-B function evaluations
       9 stop             0 = 'NORM OF PROJECTED GRADIENT <= PGTOL', 1 = 'RELATIVE REDUCTION
                          OF F <= FACTR*EPSMCH', 2 = anything else (res.message)
      10 lml_rec          -res.fun of the recorded run
      11 n_rng_draws      number of draws taken from gp._rng during fit (0 means the
                          random_state is never used)
     theta order: ConstantKernel() + WhiteKernel() + K() is Sum(Sum(C, W), K), so
     theta = [log c, log w, log l] (RBF) or [log c, log w, log sigma_0] (DotProduct).
  postmean(x, run)     -> (5 + T,) for the ordered pair input 0 -> target 1 of the selected
                          processes: [log c, log w, log l | log sigma_0 (kernel_.theta), lml,
                          in-sample MSE (as misc.py:145-147), posterior mean f_bar (T values,
                          gp.predict(z[0]))]
  determinism(x, run)  -> (5, P, P): [GPModel.multivariate twice; a manual refit loop with
                          random_state=0; with random_state=12345; GPModel.multivariate after
                          np.random.seed(7) and random.seed(7)]
  warnings_count(x, run) -> (P, P, 3): [warnings emitted by the unsuppressed sklearn fit,
                          of which bound warnings ('close to the specified'), warnings that
                          escape GPModel.bivariate under simplefilter('always')]
  crossproc(x, run)    -> (5, P, P): GPModel.multivariate(Data(x)) recomputed in five FRESH
                          project-venv processes from the raw rows, one per import mode
                          (_XP_MODES): 0 'pipeline' = numpy, then pyspi, then sklearn (the
                          order of audit/checks/pyspi_outputs.py); 1 'pipeline' again; 2
                          'scipy_first' = scipy.linalg imported before pyspi; 3 'pyspi_first'
                          = pyspi imported before numpy; 4 'env1' = 'pipeline' with
                          OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1 in the environment.  The
                          off-diagonal holds A; the DIAGONAL is reused for bookkeeping:
                          [k, 0, 0] = thread count of numpy's OpenBLAS, [k, 1, 1] = thread
                          count of SciPy's OpenBLAS during the fit (threadpoolctl), and, if
                          P > 2, [k, 2, 2] = 1 if pyspi was imported before SciPy's OpenBLAS
                          was loaded.  pyspi/__init__.py:4 sets OMP_NUM_THREADS=1, which only
                          reaches BLAS libraries loaded after that import.
  fixedhp(x, run)      -> (P, P): in-sample MSE of sklearn's GaussianProcessRegressor with
                          pyspi's kernel and optimizer=None, i.e. with the documented values
                          (constant 1, noise 1, length scale 1 | sigma_0 = 1) held fixed
  amplitude(x, run)    -> (P, P, 5): sklearn fit of the standard ConstantKernel() * RBF() +
                          WhiteKernel() (R&W eq. 2.31) with n_restarts_optimizer=10,
                          random_state=0: [in-sample MSE, log C, log l, log w (kernel_.theta
                          order), log_marginal_likelihood_value_]
"""
import contextlib
import io
import random
import warnings

import numpy as np


def _kernel_name(run):
    return run["unit_id"].rsplit(".", 1)[-1]


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


def _template(kname):
    from pyspi.statistics.misc import GPModel
    return GPModel(kernel=kname)._kernel


class _Recorder:
    """Callable optimizer identical to sklearn's 'fmin_l_bfgs_b' branch (_gpr.py:659-667)."""

    def __init__(self):
        self.results = []

    def __call__(self, obj_func, initial_theta, bounds):
        import scipy.optimize
        res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
        self.results.append(res)
        return res.x, res.fun


class _CountingRNG:
    """Wraps gp._rng to count draws (sklearn uses it only for restarts, _gpr.py:322-333)."""

    def __init__(self, rng):
        self._rng, self.n = rng, 0

    def uniform(self, *a, **k):
        self.n += 1
        return self._rng.uniform(*a, **k)

    def __getattr__(self, name):
        self.n += 1
        return getattr(self._rng, name)


def _stop_code(msg):
    m = str(msg).upper()
    if "PROJECTED GRADIENT" in m:
        return 0.0
    if "REDUCTION OF F" in m:
        return 1.0
    return 2.0


def params(x, run):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.metrics import mean_squared_error
    kname = _kernel_name(run)
    z = _data(x, run["path"]).to_numpy()
    P = z.shape[0]
    out = np.full((P, P, 12), np.nan)
    tmpl = _template(kname)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            X, y = z[i], np.ravel(z[j])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(kernel=tmpl).fit(X, y)
            mse = mean_squared_error(gp.predict(X), y)
            rec = _Recorder()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp2 = GaussianProcessRegressor(kernel=tmpl, optimizer=rec).fit(X, y)
            mse2 = mean_squared_error(gp2.predict(X), y)
            r = rec.results[0]
            out[i, j] = [mse, *gp.kernel_.theta, gp.log_marginal_likelihood_value_, gp.n_restarts_optimizer,
                         mse2, r.nit, r.nfev, _stop_code(r.message), -float(r.fun), _rng_draws(tmpl, X, y)]
    return out


def _rng_draws(tmpl, X, y):
    """Refit with a GaussianProcessRegressor subclass whose _rng counts draws."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    class _GPR(GaussianProcessRegressor):
        def __setattr__(self, name, value):
            if name == "_rng" and not isinstance(value, _CountingRNG):
                value = _CountingRNG(value)
            object.__setattr__(self, name, value)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = _GPR(kernel=tmpl).fit(X, y)
    return float(g._rng.n)


def postmean(x, run):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.metrics import mean_squared_error
    z = _data(x, run["path"]).to_numpy()
    X, y = z[0], np.ravel(z[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(kernel=_template(_kernel_name(run))).fit(X, y)
    f = gp.predict(X)
    return np.concatenate([gp.kernel_.theta, [gp.log_marginal_likelihood_value_, mean_squared_error(f, y)], f])


def determinism(x, run):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.metrics import mean_squared_error
    from pyspi.statistics.misc import GPModel
    kname = _kernel_name(run)
    d = _data(x, run["path"])
    z = d.to_numpy()
    P = z.shape[0]
    out = np.full((5, P, P), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out[0] = GPModel(kernel=kname).multivariate(d)
        out[1] = GPModel(kernel=kname).multivariate(d)
        tmpl = _template(kname)
        for k, rs in ((2, 0), (3, 12345)):
            for i in range(P):
                for j in range(P):
                    if i != j:
                        gp = GaussianProcessRegressor(kernel=tmpl, random_state=rs).fit(z[i], np.ravel(z[j]))
                        out[k, i, j] = mean_squared_error(gp.predict(z[i]), np.ravel(z[j]))
        np.random.seed(7)
        random.seed(7)
        out[4] = GPModel(kernel=kname).multivariate(d)
    for k in range(5):
        np.fill_diagonal(out[k], np.nan)
    return out


def warnings_count(x, run):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from pyspi.statistics.misc import GPModel
    kname = _kernel_name(run)
    d = _data(x, run["path"])
    z = d.to_numpy()
    P = z.shape[0]
    out = np.full((P, P, 3), np.nan)
    tmpl = _template(kname)
    spi = GPModel(kernel=kname)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            with warnings.catch_warnings(record=True) as w1:
                warnings.simplefilter("always")
                GaussianProcessRegressor(kernel=tmpl).fit(z[i], np.ravel(z[j]))
            with warnings.catch_warnings(record=True) as w2:
                warnings.simplefilter("always")
                spi.bivariate(d, i=i, j=j)
            out[i, j] = [len(w1), sum("close to the specified" in str(m.message) for m in w1), len(w2)]
    return out


_XP_MODES = ("pipeline", "pipeline", "scipy_first", "pyspi_first", "env1")

_XP_SCRIPT = r"""
import sys
mode, kname, xin, out = sys.argv[1:5]
if mode == "pyspi_first":
    import pyspi  # noqa: F401  (sets OMP_NUM_THREADS=1 before numpy's OpenBLAS loads)
import numpy as np
if mode == "scipy_first":
    import scipy.linalg  # noqa: F401  (SciPy's OpenBLAS loads before pyspi's OMP_NUM_THREADS=1)
loaded_before = "scipy.linalg" in sys.modules
import contextlib, io, warnings
import pyspi  # noqa: F401
from pyspi.data import Data
from pyspi.statistics.misc import GPModel
from threadpoolctl import threadpool_info
x = np.load(xin)
with contextlib.redirect_stdout(io.StringIO()):
    d = Data(np.array(x, dtype=np.float64, copy=True), dim_order="ps")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    A = np.array(GPModel(kernel=kname).multivariate(d), dtype=np.float64)
th = {"numpy": np.nan, "scipy": np.nan}
for lib in threadpool_info():
    if lib.get("internal_api") == "openblas":
        key = "numpy" if "numpy.libs" in lib["filepath"] else ("scipy" if "scipy.libs" in lib["filepath"] else None)
        if key:
            th[key] = float(lib["num_threads"])
A[0, 0], A[1, 1] = th["numpy"], th["scipy"]
if A.shape[0] > 2:
    A[2, 2] = float(not loaded_before)
np.save(out, A)
"""


def crossproc(x, run):
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    kname = _kernel_name(run)
    ex = Path(__file__).resolve().parents[1] / "out" / "exchange" / "gpfit"
    ex.mkdir(parents=True, exist_ok=True)
    P = x.shape[0]
    out = np.full((len(_XP_MODES), P, P), np.nan)
    with tempfile.TemporaryDirectory(dir=ex) as td:
        xin = os.path.join(td, "x.npy")
        np.save(xin, np.asarray(x, dtype=np.float64))
        script = os.path.join(td, "xp.py")
        Path(script).write_text(_XP_SCRIPT)
        for k, mode in enumerate(_XP_MODES):
            env = {kk: vv for kk, vv in os.environ.items()
                   if kk not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
            if mode == "env1":
                env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            o = os.path.join(td, f"A{k}.npy")
            subprocess.run([sys.executable, "-I", script, "scipy_first" if mode == "scipy_first" else
                            ("pyspi_first" if mode == "pyspi_first" else "pipeline"), kname, xin, o],
                           env=env, check=True, cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out[k] = np.load(o)
    return out


def fixedhp(x, run):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.metrics import mean_squared_error
    z = _data(x, run["path"]).to_numpy()
    P = z.shape[0]
    out = np.full((P, P), np.nan)
    tmpl = _template(_kernel_name(run))
    for i in range(P):
        for j in range(P):
            if i != j:
                gp = GaussianProcessRegressor(kernel=tmpl, optimizer=None).fit(z[i], np.ravel(z[j]))
                out[i, j] = mean_squared_error(gp.predict(z[i]), np.ravel(z[j]))
    return out


def amplitude(x, run):
    from sklearn.gaussian_process import GaussianProcessRegressor, kernels
    from sklearn.metrics import mean_squared_error
    z = _data(x, run["path"]).to_numpy()
    P = z.shape[0]
    out = np.full((P, P, 5), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            ker = kernels.ConstantKernel() * kernels.RBF() + kernels.WhiteKernel()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=10, random_state=0).fit(z[i], np.ravel(z[j]))
            out[i, j] = [mean_squared_error(gp.predict(z[i]), np.ravel(z[j])), *gp.kernel_.theta,
                         gp.log_marginal_likelihood_value_]
    return out
