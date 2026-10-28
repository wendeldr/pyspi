import warnings
import numpy as np

from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from pyspi._envelope import envelope_workspace
from pyspi._bands import band_identifier, band_inputs, frequency_band, phase_amplitude_result, PAC_REASONS
from pyspi._regression import regression_result, regression_diagnostics
from pyspi._moments import moment_workspace
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
        self.identifier += f"_{model}"
        self._model_name = model
        self._parameters = dict(alpha=alpha, l1_ratio=l1_ratio, penalty_scale=penalty_scale)
        if alpha != 1.:
            self.identifier += f"_alpha-{alpha}".replace(".", "-")
        if model == "ElasticNet" and l1_ratio != .5:
            self.identifier += f"_l1-{l1_ratio}".replace(".", "-")
        if penalty_scale != default_scale:
            self.identifier += f"_scale-{penalty_scale}"

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

    No internal band filter is applied. An optional Hz band requires supplied
    continuously filtered windows. Orthogonalized results average both
    projection directions; absolute variants take abs before that average.
    """

    name = "Envelope correlation"
    identifier = "pec"
    labels = ["misc", "undirected"]

    def __init__(self, orth=False, log=False, absolute=False, band=None):
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
        self.band = None if band is None else frequency_band(band)
        self.continuous_bands = () if self.band is None else (self.band,)
        if self.band is not None:
            self.identifier += "_band-" + band_identifier(self.band)

    def issigned(self):
        return not self._absolute

    @parse_multivariate
    def multivariate(self, data):
        work = envelope_workspace(data) if self.band is None else band_inputs(data).envelope(self.band, preprocess=True)
        return work.result(self._orth, self._log, self._absolute)

    @parse_multivariate
    def diagnostics(self, data):
        if self.band is None:
            return envelope_workspace(data).diagnostics(self._orth, self._log)
        return band_inputs(data).envelope_diagnostics(self.band, self._orth, self._log)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self.multivariate(data)[i, j]


class PhaseAmplitudeCoupling(Directed, Unsigned):
    """Exploratory source-phase / target-amplitude association, not causality.

    The magnitude of sum(exp(i*phase_source) * centered_amplitude_target),
    divided by sqrt(samples) * norm(centered_amplitude_target). Both analytic
    signals are obtained within the window from continuously filtered inputs;
    no additional window detrending or z-scoring is applied to those inputs.
    """

    name = "Cross-channel phase-amplitude coupling"
    identifier = "pac"
    labels = ["misc", "directed", "unsigned", "nonlinear", "frequency-dependent"]

    def __init__(self, phase_band=(16, 30), amplitude_band=(70, 150)):
        self.phase_band = frequency_band(phase_band)
        self.amplitude_band = frequency_band(amplitude_band)
        if self.phase_band[1] > self.amplitude_band[0]:
            raise ValueError("phase and amplitude bands must be ordered and nonoverlapping")
        self.continuous_bands = (self.phase_band, self.amplitude_band)
        self.identifier += f"_phase-{band_identifier(self.phase_band)}_amp-{band_identifier(self.amplitude_band)}"

    @parse_multivariate
    def multivariate(self, data):
        return phase_amplitude_result(data, self.phase_band, self.amplitude_band)[0].copy()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self.multivariate(data)[i, j]

    @parse_multivariate
    def diagnostics(self, data):
        return {"reason": PAC_REASONS[phase_amplitude_result(data, self.phase_band, self.amplitude_band)[1]].copy()}

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
        if isinstance(terms, (bool, np.bool_)) or not isinstance(terms, (int, np.integer)) or terms < 1:
            raise ValueError("terms must be a positive integer")
        if isinstance(pnorm, (bool, np.bool_)) or pnorm not in ("max", 1, 2):
            raise ValueError("pnorm must be max, 1 or 2")
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be finite and positive")
        self._num_terms = terms
        self._p_norm = pnorm
        self._bandwidth_term = bandwidth
        if (terms, pnorm, bandwidth) != (6, "max", .5):
            self.identifier += f"_terms-{terms}_pnorm-{pnorm}_bandwidth-{bandwidth:g}".replace(".", "-")


    @parse_multivariate
    def multivariate(self, data):
        work = moment_workspace(data)
        def compute():
            z = work.source
            valid = np.isfinite(z).all(axis=1) & np.any(z != z[:, :1], axis=1)
            result = np.full((len(z), len(z)), np.nan)
            if valid.any():
                result[np.ix_(valid, valid)] = compute_IDS(
                    z[valid].T, num_terms=self._num_terms, p_norm=self._p_norm,
                    bandwidth_term=self._bandwidth_term)
            np.fill_diagonal(result, np.nan)
            return result
        return work.arrays.get(("ids", self._num_terms, self._p_norm, self._bandwidth_term), compute).copy()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self.multivariate(data)[i, j]
