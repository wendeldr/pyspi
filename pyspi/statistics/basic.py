import warnings

import sklearn.covariance as cov
import numpy as np

from pyspi._cache import cached_workspace
from pyspi._lightweight import lightweight_result
from pyspi._naming import compose, duration, parameter, word
from pyspi._rank import KendallWorkspace, SpearmanWorkspace
from pyspi._robust import FastMinCovDet
from pyspi._xcorr import CrossCorrelationWorkspace, SUMMARY_KEYS
from pyspi.base import Directed, Undirected, Signed, parse_bivariate, parse_multivariate


#: FAST-MCD draws its candidate subsets from a random stream. sklearn defaults
#: to the unseeded global numpy RNG, so a fixed seed keeps the robust fits
#: reproducible run to run. It equals the regression harness seed, so the
#: stored goldens are unchanged.
ROBUST_RANDOM_STATE = 42


class _BasicWorkspace:
    def __init__(self, source):
        self.values = source[:, :, 0]
        self.estimators = {}


def _basic_workspace(data):
    if data.n_replications != 1:
        raise ValueError("Covariance and precision require a single realisation.")
    return cached_workspace(data, "basic", _BasicWorkspace)


def _fit_name(estimator):
    # EllipticEnvelope.fit is MinCovDet.fit followed by an outlier offset that
    # covariance_/precision_ never use, so both serve from one FAST-MCD fit.
    return "MinCovDet" if estimator == "EllipticEnvelope" else estimator


def _fit_estimator(name, values):
    if name == "MinCovDet":
        return FastMinCovDet(random_state=ROBUST_RANDOM_STATE).fit(values)
    return getattr(cov, name)().fit(values)


class Estimators(Undirected, Signed):
    """Base class for (functional) connectivity-based statistics

    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    name = "Covariance"
    labels = ["basic", "unordered", "linear", "undirected"]

    def __init__(self, kind, estimator="EmpiricalCovariance", squared=False):
        if squared:
            self.labels = Estimators.labels + ["unsigned"]
            self.issigned = lambda: False
        else:
            self.labels = Estimators.labels + ["signed"]
        # EmpiricalCovariance -> empirical, ShrunkCovariance -> shrunk.
        name = word(estimator)
        name = name.removesuffix("covariance") or name
        self.identifier = compose(self.identifier, *(["sq"] if squared else []), name)
        self._squared = squared
        self._estimator = estimator
        self._kind = kind

    def input_problem(self, n_observations, n_processes):
        if self._estimator == "GraphicalLassoCV":
            return self._minimum_observations_problem(n_observations, 5, "GraphicalLassoCV uses 5-fold cross-validation")
        # Regularization can make GraphicalLasso estimable even when processes
        # outnumber observations. Convergence is data-dependent, not a shape veto.
        return None

    def _from_cache(self, data):
        work = _basic_workspace(data)
        name = _fit_name(self._estimator)
        if name not in work.estimators:
            work.estimators[name] = _fit_estimator(name, work.values.T)
        return work.estimators[name]

    @parse_multivariate
    def multivariate(self, data):
        mycov = self._from_cache(data)
        matrix = getattr(mycov, self._kind + "_").copy()
        np.fill_diagonal(matrix, np.nan)
        if self._squared:
            return np.square(matrix)
        else:
            return matrix


class Covariance(Estimators):

    name = "Covariance"
    identifier = "cov"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="covariance", squared=squared, estimator=estimator)


class Precision(Estimators):

    name = "Precision"
    identifier = "prec"

    def __init__(self, estimator="EmpiricalCovariance", squared=False):
        super().__init__(kind="precision", squared=squared, estimator=estimator)


class PartialCorrelation(Estimators):
    """LedoitWolf-regularized partial correlation conditional on all input channels."""

    name = "LedoitWolf partial correlation"
    identifier = "partialcorr"

    def __init__(self):
        super().__init__(kind="precision", estimator="LedoitWolf")

    def input_problem(self, n_observations, n_processes):
        return self._minimum_observations_problem(n_observations, 2, "sample variation requires two observations")

    def _result(self, data):
        work = _basic_workspace(data)
        values = work.values
        p = len(values)
        reason = np.full((p, p), "estimated", dtype="U24")
        # Dropping a nonfinite channel would silently change the conditioning set.
        if not np.isfinite(values).all():
            result = np.full((p, p), np.nan)
            reason[:] = "nonfinite_conditioning"
        else:
            precision = self._from_cache(data).precision_
            diagonal = np.diag(precision)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                scale = np.sqrt(diagonal)
                result = -precision/scale[:, None]/scale[None, :]
            invalid = ~np.isfinite(result) | (abs(result) > 1+8*values.shape[1]*np.finfo(float).eps)
            reason[invalid] = "numerical_failure"
            constant = np.all(values == values[:, :1], axis=1)
            reason[constant[:, None] | constant[None, :]] = "constant_input"
            result = np.clip(result, -1., 1.)
            result[reason != "estimated"] = np.nan
        np.fill_diagonal(result, np.nan)
        np.fill_diagonal(reason, "diagonal")
        return result, reason

    @parse_multivariate
    def multivariate(self, data):
        return self._result(data)[0]

    @parse_multivariate
    def diagnostics(self, data):
        return {"reason": self._result(data)[1]}

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self.multivariate(data)[i, j]


class _MomentReduction:
    @parse_multivariate
    def multivariate(self, data):
        return lightweight_result(data, self._reduction, self._setting)[0]

    @parse_multivariate
    def diagnostics(self, data):
        return {"reason": lightweight_result(data, self._reduction, self._setting)[1]}

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self.multivariate(data)[i, j]

    def input_problem(self, n_observations, n_processes):
        lag = self._setting if self._reduction == "lag" else 0
        return self._minimum_observations_problem(n_observations, lag+2, "two aligned observations are required")


class FixedLagCorrelation(_MomentReduction, Directed, Signed):
    """Pearson correlation of source[:-lag] and target[lag:], each recentered."""

    name = "Fixed-lag correlation"
    labels = ["basic", "signed", "directed", "linear", "temporal", "bivariate"]
    _reduction = "lag"

    def __init__(self, lag_samples=20, fs=2048):
        if isinstance(lag_samples, (bool, np.bool_)) or not isinstance(lag_samples, (int, np.integer)) or lag_samples < 1:
            raise ValueError("lag_samples must be a positive integer")
        if isinstance(fs, (bool, str)) or not np.isscalar(fs) or np.iscomplexobj(fs) or not np.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be finite and positive")
        self._setting = int(lag_samples)
        self.fs = float(fs)
        self.identifier = compose("lagcorr", parameter("samples", lag_samples), duration("lag", lag_samples/self.fs))


class ThresholdEventExcess(_MomentReduction, Undirected, Signed):
    """Covariance of abs(sample-z-score) > threshold indicators; an event proxy."""

    name = "Threshold-event excess"
    labels = ["basic", "signed", "undirected", "nonlinear", "bivariate"]
    _reduction = "event"

    def __init__(self, threshold=3):
        if isinstance(threshold, (bool, str)) or not np.isscalar(threshold) or np.iscomplexobj(threshold) or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        self._setting = float(threshold)
        self.identifier = compose("eventexcess", parameter("threshold", threshold, 1))


class LogScaleRatio(_MomentReduction, Directed, Signed):
    """log(std(target)/std(source)); signed amplitude contrast, not causal flow."""

    name = "Log scale ratio"
    identifier = "logscaleratio"
    labels = ["basic", "signed", "directed", "unordered", "bivariate"]
    _reduction = "log_scale"
    _setting = None


class CrossCorrelationUnavailableWarning(UserWarning):
    """Some pair summaries are undefined; diagnostics explain each reason."""


class CrossCorrelation(Undirected, Signed):
    """Summaries of centred, energy-normalized linear cross-correlation.

    Squaring precedes aggregation. ``peak`` retains the sign of the strongest
    magnitude, with NaN for an equal-magnitude opposite-sign tie. ``min`` is
    the numerical minimum; ``min`` with ``squared=True`` is min(r**2).
    See docs/calculation_definitions.md for formulas and inference assumptions.
    """

    name = "Cross correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, squared=False, statistic="max", sigonly=True, *,
                 fs=1, max_lag_seconds=0.125, alpha=0.05):
        if statistic not in ("mean", "max", "min", "peak"):
            raise ValueError("Cross-correlation statistic must be mean, max, min, or peak.")
        if type(squared) is not bool or type(sigonly) is not bool:
            raise TypeError("squared and sigonly must be booleans.")
        if statistic == "peak" and squared:
            raise ValueError("Squared peak duplicates max(r**2); use statistic='max', squared=True.")
        for name, value in (("fs", fs), ("max_lag_seconds", max_lag_seconds), ("alpha", alpha)):
            if (isinstance(value, (bool, str)) or not np.isscalar(value)
                    or np.iscomplexobj(value) or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite number.")
        if fs <= 0 or max_lag_seconds < 0 or not 0 < alpha < 1:
            raise ValueError("Require fs > 0, max_lag_seconds >= 0, and 0 < alpha < 1.")
        if not sigonly and alpha != 0.05:
            raise ValueError("alpha only applies when sigonly=True.")
        self._squared, self._statistic, self._sigonly = squared, statistic, sigonly
        self.fs, self.max_lag_seconds, self.alpha = float(fs), float(max_lag_seconds), float(alpha)
        self.labels = self.labels + ["unsigned" if squared else "signed"]
        self._index = SUMMARY_KEYS.index((statistic, squared))
        # The lag window is in milliseconds when the sampling rate is known;
        # with the default fs=1 it is a number of samples.
        lag = (parameter("maxlag", self.max_lag_seconds, 1) if self.fs == 1
               else duration("maxlag", self.max_lag_seconds))
        # holm: Bartlett significance with Holm correction at level alpha.
        self.identifier = compose("xcorr", *(["sq"] if squared else []), *(["holm"] if sigonly else []),
                                  statistic, lag, *([parameter("alpha", self.alpha, 1)] if sigonly else []))

    def issigned(self):
        return not self._squared

    def _workspace(self, data):
        if data.n_replications != 1:
            raise ValueError("Cross-correlation requires a single realisation.")
        requested = self.fs * self.max_lag_seconds
        if not np.isfinite(requested):
            raise ValueError("Requested lag is too large.")
        nearest = round(requested)
        if abs(requested-nearest) <= 8*np.finfo(float).eps*max(1, abs(requested)):
            requested = nearest
        lag = int(np.floor(requested))
        if lag >= data.n_observations:
            raise ValueError("Requested maximum lag must be shorter than the data window.")
        work = cached_workspace(data, "xcorr", CrossCorrelationWorkspace)
        return work, lag, self.alpha if self._sigonly else None

    def _diagnostics(self, counts, status, ambiguous, lag):
        status = np.array(status, copy=True, dtype="U24")
        if self._statistic == "peak":
            status[np.asarray(ambiguous)] = "ambiguous_peak"
        return {"status": status, "selected_lag_count": np.array(counts, copy=True),
                "maximum_lag_samples": lag, "maximum_lag_seconds": lag/self.fs,
                "selection": "bartlett-holm" if self._sigonly else "all_lags",
                "alpha": self.alpha if self._sigonly else None}

    def _warn_unavailable(self, status):
        status = np.asarray(status)
        if status.ndim == 2:
            status = status[np.triu_indices(len(status), 1)]
        counts = {str(reason): int(np.count_nonzero(status == reason))
                  for reason in np.unique(status) if reason not in ("ok", "diagonal")}
        if counts:
            warnings.warn("Cross-correlation unavailable pair summaries: "
                          + "; ".join(f"{reason}={count}" for reason, count in counts.items()),
                          CrossCorrelationUnavailableWarning, stacklevel=3)

    @parse_multivariate
    def multivariate(self, data):
        work, lag, alpha = self._workspace(data)
        values, counts, status, ambiguous = work.all_pairs(lag, alpha)
        self.last_diagnostics = self._diagnostics(counts, status, ambiguous, lag)
        self._warn_unavailable(self.last_diagnostics["status"])
        return values[self._index].copy()

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        work, lag, alpha = self._workspace(data)
        if i == j:
            self.last_diagnostics = self._diagnostics(0, "diagonal", False, lag)
            return np.nan
        values, count, status, ambiguous = work.pair(i, j, lag, alpha)
        self.last_diagnostics = self._diagnostics(count, status, ambiguous, lag)
        self._warn_unavailable(self.last_diagnostics["status"])
        return values[self._index]


class _RankCorrelation(Undirected, Signed):

    def __init__(self, squared=False):
        self._squared = squared
        self.labels = self.labels + ["unsigned" if squared else "signed"]
        if squared:
            self.identifier = compose(self.identifier, "sq")

    def issigned(self):
        return not self._squared

    def _workspace(self, data):
        if data.n_replications != 1:
            raise ValueError(f"{self.name} requires a single realisation.")
        return cached_workspace(data, self._workspace_type,
                                lambda source: self._workspace_type(source[:, :, 0]))

    @parse_multivariate
    def multivariate(self, data):
        result = self._workspace(data).all_pairs().copy()
        if self._squared:
            np.square(result, out=result)
        np.fill_diagonal(result, np.nan)
        return result

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        value = self._workspace(data).pair(i, j)
        return value ** 2 if self._squared else value


class SpearmanR(_RankCorrelation):
    """Spearman rank correlation for one realisation of each process."""

    name = "Spearman's correlation coefficient"
    identifier = "spearmanr"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]
    _workspace_type = SpearmanWorkspace


class KendallTau(_RankCorrelation):
    """Kendall's tau-b, with tie correction and one calculation per pair."""

    name = "Kendall's tau"
    identifier = "kendalltau"
    labels = ["basic", "unordered", "rank", "linear", "undirected"]

    _workspace_type = KendallWorkspace
