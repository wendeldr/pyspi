from functools import cached_property

import sklearn.covariance as cov
from scipy import signal
import numpy as np

from pyspi._cache import ArrayCache, cached_workspace
from pyspi._rank import KendallWorkspace, SpearmanWorkspace
from pyspi.base import Undirected, Signed, parse_bivariate, parse_multivariate


class _BasicWorkspace:
    def __init__(self, source):
        self.values = source[:, :, 0]
        self.estimators = {}
        self.correlations = ArrayCache()
        self.correlation_directions = {}

    @cached_property
    def std(self):
        return self.values.std(axis=1)


def _basic_workspace(data):
    if data.n_replications != 1:
        raise ValueError("Basic covariance and cross-correlation require a single realisation.")
    return cached_workspace(data, "basic", _BasicWorkspace)


class Estimators(Undirected, Signed):
    """Base class for (functional) connectivity-based statistics

    Information on covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    name = "Covariance"
    labels = ["basic", "unordered", "linear", "undirected"]

    def __init__(self, kind, estimator="EmpiricalCovariance", squared=False):
        paramstr = f"_{estimator}"
        if squared:
            paramstr = "-sq" + paramstr
            self.labels = Estimators.labels + ["unsigned"]
            self.issigned = lambda: False
        else:
            self.labels = Estimators.labels + ["signed"]
        self.identifier = self.identifier + paramstr
        self._squared = squared
        self._estimator = estimator
        self._kind = kind

    def _from_cache(self, data):
        work = _basic_workspace(data)
        if self._estimator not in work.estimators:
            work.estimators[self._estimator] = getattr(cov, self._estimator)().fit(work.values.T)
        return work.estimators[self._estimator]

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


class CrossCorrelation(Undirected, Signed):

    name = "Cross correlation"
    labels = ["basic", "linear", "undirected", "temporal"]

    def __init__(self, squared=False, statistic="max", sigonly=True):
        self.identifier = "xcorr"
        self._squared = squared
        self._statistic = statistic
        self._sigonly = sigonly

        if self._squared:
            self.issigned = lambda: False
            self.identifier = self.identifier + "-sq"
            self.labels = CrossCorrelation.labels + ["unsigned"]
        else:
            self.labels = CrossCorrelation.labels + ["signed"]
        self.identifier += f"_{statistic}_sig-{sigonly}"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self._pair_statistic(_basic_workspace(data), i, j)

    @parse_multivariate
    def multivariate(self, data):
        work = _basic_workspace(data)
        result = np.full((data.n_processes, data.n_processes), np.nan)
        # Preserve the existing first-visited (higher, lower) orientation.
        # The lag window and direction convention need a separate review.
        for j in range(data.n_processes):
            for i in range(j + 1, data.n_processes):
                result[i, j] = result[j, i] = self._pair_statistic(work, i, j)
        return result

    def _pair_statistic(self, work, i, j):
        T = work.values.shape[1]
        key = tuple(sorted((i, j)))
        first_i, first_j = work.correlation_directions.setdefault(key, (i, j))

        def correlate():
            x, y = work.values[[first_i, first_j]]
            r = signal.correlate(x, y, "full") / work.std[first_i] / work.std[first_j] / (T - 1)
            # Retain only the selected window, not a view of the full buffer.
            return r[T - T // 4 : T + T // 4].copy()

        r_ij = work.correlations.get(key, correlate)

        # Truncate at first statistically significant zero (i.e., |r| <= 1.96/sqrt(T))
        if self._sigonly:
            N = len(r_ij) // 2
            fzf = np.where(np.abs(r_ij[len(r_ij) // 2 :]) <= 1.96 / np.sqrt(N))[0][0]
            fzr = np.where(np.abs(r_ij[: len(r_ij) // 2]) <= 1.96 / np.sqrt(N))[0][-1]
            r_ij = r_ij[N - fzr : N + fzf]

        if self._statistic == "max":
            if self._squared:
                return np.max(r_ij**2)
            else:
                return np.max(r_ij)
        elif self._statistic == "mean":
            if self._squared:
                return np.mean(r_ij**2)
            else:
                return np.mean(r_ij)
        else:
            raise TypeError(f"Unknown statistic: {self._statistic}")


class _RankCorrelation(Undirected, Signed):

    def __init__(self, squared=False):
        self._squared = squared
        self.labels = self.labels + ["unsigned" if squared else "signed"]
        if squared:
            self.identifier = self.identifier + "-sq"

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
