import warnings
import numpy as np
import inspect

from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from pyspi._envelope import envelope_workspace
from pyspi._regression import regression_result
from pyspi.lib.ids.dependence import compute_IDS

from pyspi.base import (
    Directed,
    Undirected,
    Signed,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)


class Cointegration(Undirected, Unsigned):

    name = "Cointegration"
    identifier = "coint"
    labels = ["misc", "unsigned", "temporal", "undirected", "nonlinear"]

    def __init__(
        self,
        method="johansen",
        statistic="trace_stat",
        det_order=1,
        k_ar_diff=1,
        autolag="aic",
        maxlag=10,
        trend="c",
    ):
        self._method = method
        self._statistic = statistic
        if method == "johansen":
            self.identifier += (
                f"_{method}_{statistic}_order-{det_order}_ardiff-{k_ar_diff}"
            )
            self._det_order = det_order
            self._k_ar_diff = k_ar_diff
        else:
            self._autolag = autolag
            self._maxlag = maxlag
            self._trend = trend
            self.identifier += (
                f"_{method}_{statistic}_trend-{trend}_autolag-{autolag}_maxlag-{maxlag}"
            )

    @property
    def key(self):
        key = (self._method,)
        if self._method == "johansen":
            return key + (self._det_order, self._k_ar_diff)
        else:
            return key + (self._autolag, self._maxlag, self._trend)

    def _from_cache(self, data, i, j):
        idx = (i, j)
        try:
            ci = data.coint[self.key][idx]
        except (KeyError, AttributeError):
            z = data.to_numpy(squeeze=True)

            if self._method == "aeg":
                stats = stattools.coint(
                    z[i],
                    z[j],
                    autolag=self._autolag,
                    maxlag=self._maxlag,
                    trend=self._trend,
                )

                ci = {"tstat": stats[0]}
            else:
                stats = coint_johansen(
                    z[[i, j]].T, det_order=self._det_order, k_ar_diff=self._k_ar_diff
                )

                ci = {
                    "max_eig_stat": stats.max_eig_stat[0],
                    "trace_stat": stats.trace_stat[0],
                }

            try:
                data.coint[self.key][idx] = ci
            except AttributeError:
                data.coint = {self.key: {idx: ci}}
            except KeyError:
                data.coint[self.key] = {idx: ci}
            data.coint[self.key][(j, i)] = ci

        return ci

    # Return the negative t-statistic (proxy for how co-integrated they are)
    @parse_bivariate
    def bivariate(self, data, i=None, j=None, verbose=False):
        ci = self._from_cache(data, i, j)
        return ci[self._statistic]


class LinearModel(Directed, Unsigned):
    name = "Linear model regression"
    identifier = "lmfit"
    labels = ["misc", "unsigned", "unordered", "normal", "linear", "directed"]

    def __init__(self, model):
        self.identifier += f"_{model}"
        self._model_name = model
        self._model = getattr(linear_model, model)

    @parse_multivariate
    def multivariate(self, data):
        if self._model_name == "Ridge":
            return regression_result(data, self._model_name)
        return super().multivariate(data)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if self._model_name == "Ridge":
            return regression_result(data, self._model_name)[i, j]
        z = data.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_params = inspect.signature(self._model).parameters
            if "random_state" in model_params:
                mdl = self._model(random_state=42).fit(z[i], np.ravel(z[j]))
            else:
                mdl = self._model().fit(z[i], np.ravel(z[j]))
        y_predict = mdl.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))


class GPModel(Directed, Unsigned):
    name = "Gaussian process regression"
    identifier = "gpfit"
    labels = ["misc", "unsigned", "unordered", "normal", "nonlinear", "directed"]

    def __init__(self, kernel="RBF"):
        self.identifier += f"_{kernel}"
        self._kernel = kernels.ConstantKernel() + kernels.WhiteKernel()
        self._kernel += getattr(kernels, kernel)()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=self._kernel).fit(z[i], np.ravel(z[j]))
        y_predict = gp.predict(z[i])
        return mean_squared_error(y_predict, np.ravel(z[j]))


class PowerEnvelopeCorrelation(Undirected, Signed):
    """Amplitude or log-power envelope correlation of the supplied signal.

    No internal band filter is applied. Orthogonalized results average both
    projection directions; absolute variants take abs before that average.
    """

    name = "Envelope correlation"
    identifier = "pec"
    labels = ["misc", "undirected"]

    def __init__(self, orth=False, log=False, absolute=False):
        if absolute and not orth:
            raise ValueError("absolute=True requires pairwise orthogonalization.")
        self._orth = bool(orth)
        self.labels = self.labels + ["unsigned" if absolute else "signed"]
        if self._orth:
            self.identifier += "_orth"
        self._log = log
        if log:
            self.identifier += "_log"
        self._absolute = absolute
        if absolute:
            self.identifier += "_abs"

    def issigned(self):
        return not self._absolute

    @parse_multivariate
    def multivariate(self, data):
        return envelope_workspace(data).result(self._orth, self._log, self._absolute)

    @parse_multivariate
    def diagnostics(self, data):
        return envelope_workspace(data).diagnostics(self._orth, self._log)

class InterDependenceScore(Undirected, Unsigned):
    name = "Interdependence score"
    identifier = "ids"
    labels = ["unsigned", "misc", "undirected", "nonlinear"]

    def __init__(
            self,
            terms=6,
            pnorm='max',
            bandwidth=0.5
    ):
        self._num_terms = terms
        self._p_norm = pnorm
        self._bandwidth_term = bandwidth


    @parse_multivariate
    def multivariate(self, data):
        # reshape for the compute_IDS function which expects shape (obs, proc)
        z = np.squeeze(data.to_numpy(), axis=2).T
        ids = compute_IDS(z, num_terms=self._num_terms, p_norm=self._p_norm, 
                           bandwidth_term=self._bandwidth_term)
        return ids
