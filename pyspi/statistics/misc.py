import warnings
import numpy as np

from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from pyspi._envelope import envelope_workspace
from pyspi._regression import regression_result, regression_diagnostics
from pyspi.lib.ids.dependence import compute_IDS
from pyspi._naming import compose, parameter, word

from pyspi.base import (
    Directed,
    Undirected,
    Signed,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)


# Deterministic terms: Johansen det_order and Engle-Granger trend codes.
_DETERMINISTIC = {-1: "nodeterministic", 0: "constant", 1: "lineartrend",
                  "n": "nodeterministic", "nc": "nodeterministic", "c": "constant",
                  "ct": "lineartrend", "ctt": "quadratictrend"}


def _deterministic(value, name):
    # Codes without a descriptive name keep the backend's own spelling.
    if value in _DETERMINISTIC:
        return _DETERMINISTIC[value]
    return parameter(name, value) if isinstance(value, (int, np.integer)) else name + word(value)


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
        # trace_stat -> trace, max_eig_stat -> maxeig
        statistic_name = word(statistic.removesuffix("_stat"))
        if method == "johansen":
            deterministic = _deterministic(det_order, "detorder")
            fields = [word(method), statistic_name, parameter("lagdiffs", k_ar_diff)]
            fields.insert(1 if deterministic.isalpha() else 3, deterministic)
            self.identifier = compose(self.identifier, *fields)
            self._det_order = det_order
            self._k_ar_diff = k_ar_diff
        else:
            self._autolag = autolag
            self._maxlag = maxlag
            self._trend = trend
            lag_choice = "fixedlag" if autolag is None else "autolag" + word(autolag)
            # maxlag=None lets statsmodels choose its default maximum lag.
            lag_limit = ["defaultmaxlag"] if maxlag is None else []
            self.identifier = compose(self.identifier, word(method), _deterministic(trend, "trend"), lag_choice,
                                      *lag_limit, statistic_name,
                                      *([] if maxlag is None else [parameter("maxlag", maxlag)]))

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

    def __init__(self, model, *, alpha=None, l1_ratio=None, penalty_scale=None):
        if model not in ("Ridge", "Lasso", "ElasticNet", "BayesianRidge"):
            raise ValueError("model must be Ridge, Lasso, ElasticNet or BayesianRidge. "
                             "SGDRegressor was replaced by Ridge(alpha=0.0001, penalty_scale='mean').")
        if model == "BayesianRidge" and any(value is not None for value in (alpha, l1_ratio, penalty_scale)):
            raise ValueError("BayesianRidge estimates its own penalty using the default hyperpriors.")
        if l1_ratio is not None and model != "ElasticNet":
            raise ValueError("l1_ratio applies only to ElasticNet.")
        alpha = 1. if alpha is None else float(alpha)
        l1_ratio = .5 if l1_ratio is None else float(l1_ratio)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1.")
        default_scale = "sum" if model in ("Ridge", "BayesianRidge") else "mean"
        penalty_scale = default_scale if penalty_scale is None else penalty_scale
        if penalty_scale not in ("sum", "mean") or (model != "Ridge" and penalty_scale != default_scale):
            raise ValueError("penalty_scale must be sum or mean for Ridge; Lasso/ElasticNet use mean.")
        self._model_name = model
        self._parameters = dict(alpha=alpha, l1_ratio=l1_ratio, penalty_scale=penalty_scale)
        # Only Ridge can choose how the penalty scales with the sample count;
        # BayesianRidge estimates its own penalty.
        self.identifier = compose(
            self.identifier, word(model), *(["penalty" + penalty_scale] if model == "Ridge" else []),
            *([] if model == "BayesianRidge" else [parameter("alpha", alpha, 1)]),
            *([parameter("lassoratio", l1_ratio, 1)] if model == "ElasticNet" else []))

    @parse_multivariate
    def multivariate(self, data):
        return regression_result(data, self._model_name, **self._parameters)

    @parse_multivariate
    def diagnostics(self, data):
        return regression_diagnostics(data, self._model_name, **self._parameters)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return regression_result(data, self._model_name, **self._parameters)[i, j]


class GPModel(Directed, Unsigned):
    name = "Gaussian process regression"
    identifier = "gpfit"
    labels = ["misc", "unsigned", "unordered", "normal", "nonlinear", "directed"]

    def __init__(self, kernel="RBF"):
        self.identifier = compose(self.identifier, word(kernel))
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
        self._log = log
        self._absolute = absolute
        self.identifier = compose(self.identifier, *[flag for flag, on in
                                                     (("orth", self._orth), ("log", log), ("abs", absolute)) if on])

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
        # A named norm ('max') is a variant word; a numeric p is a parameter.
        fields = [parameter("terms", terms), parameter("bandwidth", bandwidth, 1)]
        if isinstance(pnorm, str):
            fields.insert(0, "pnorm" + word(pnorm))
        else:
            fields.append(parameter("pnorm", pnorm, 1))
        self.identifier = compose(self.identifier, *fields)


    @parse_multivariate
    def multivariate(self, data):
        # reshape for the compute_IDS function which expects shape (obs, proc)
        z = np.squeeze(data.to_numpy(), axis=2).T
        ids = compute_IDS(z, num_terms=self._num_terms, p_norm=self._p_norm, 
                           bandwidth_term=self._bandwidth_term)
        return ids
