import numpy as np
from sklearn.metrics import pairwise_distances
import tslearn.metrics
from tslearn.barycenters import (
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    softdtw_barycenter,
)
from hyppo.independence import (
    MGC,
    HHG,
    Hsic,
)
from hyppo.time_series import MGCX, DcorrX
from pyspi._barycenter import euclidean_summary
from pyspi._distance_correlation import distance_correlation
from pyspi._histogram import histogram_relative_entropy
from pyspi._dtw import banded_dtw
from pyspi._naming import compose, parameter, word

from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    Signed,
    parse_bivariate,
    parse_multivariate,
)


class PairwiseDistance(Undirected, Unsigned):

    name = "Pairwise distance"
    identifier = "pdist"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, metric="euclidean", **kwargs):
        # Processes are points with one coordinate per observation. These two
        # cannot apply: wminkowski needs weights that are never passed, and
        # precomputed expects a distance matrix instead of time series.
        if metric in ("wminkowski", "precomputed"):
            raise ValueError(f"PairwiseDistance cannot use metric {metric!r} on time series.")
        self._metric = metric
        self.identifier = compose(self.identifier, word(metric))

    def input_problem(self, n_observations, n_processes):
        if self._metric == "haversine" and n_observations != 2:
            return (f"{self.identifier}: haversine distance takes 2-D (latitude, longitude) points, so it needs "
                    f"exactly 2 observations per window; got {n_observations}. Use another metric.")
        if self._metric == "mahalanobis" and n_processes <= n_observations:
            return (f"{self.identifier}: Mahalanobis distance inverts the covariance of the {n_observations} "
                    f"observations across processes, which needs more processes than observations; "
                    f"got {n_processes}. Use another metric.")
        return None

    @parse_multivariate
    def multivariate(self, data):
        return pairwise_distances(data.to_numpy(squeeze=True), metric=self._metric)


""" TODO: include optional kernels in each method
"""


class HilbertSchmidtIndependenceCriterion(Undirected, Unsigned):
    """Hilbert-Schmidt Independence Criterion (HSIC)"""

    name = "Hilbert-Schmidt Independence Criterion"
    identifier = "hsic"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def input_problem(self, n_observations, n_processes):
        return self._minimum_observations_problem(n_observations, 2 if self._biased else 3, "hyppo's HSIC")

    def __init__(self, biased=False):
        self._biased = biased
        if biased:
            self.identifier = compose(self.identifier, "biased")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = Hsic(bias=self._biased).statistic(x, y)
        return stat


class HellerHellerGorfine(Directed, Unsigned):
    """Heller-Heller-Gorfine independence criterion"""

    name = "Heller-Heller-Gorfine Independence Criterion"
    identifier = "hhg"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = HHG().statistic(x, y)
        return stat


class DistanceCorrelation(Undirected, Unsigned):
    """Distance correlation"""

    name = "Distance correlation"
    identifier = "dcorr"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    def __init__(self, biased=False):
        self._biased = biased
        self.labels = [label for label in self.labels if label != "unsigned"] + [
            "unsigned" if biased else "signed"]
        if biased:
            self.identifier = compose(self.identifier, "biased")

    def input_problem(self, n_observations, n_processes):
        # The native U-centered coefficient is undefined below four samples.
        return self._minimum_observations_problem(
            n_observations, 2 if self._biased else 4, "distance correlation centering")

    def issigned(self):
        return not self._biased

    @parse_multivariate
    def multivariate(self, data):
        return distance_correlation(data, self._biased)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return distance_correlation(data, self._biased)[i, j]


class HistogramRelativeEntropy(Undirected, Unsigned):
    """Maximum bidirectional histogram KL, in nats; compares marginal shapes."""

    name = "Histogram relative entropy"
    identifier = "histre"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected", "bivariate"]

    def __init__(self, bins=10, pseudocount=1e-10):
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 2:
            raise ValueError("bins must be an integer of at least two")
        if not np.isfinite(pseudocount) or pseudocount <= 0:
            raise ValueError("pseudocount must be finite and positive")
        self._bins = int(bins)
        self._pseudocount = float(pseudocount)
        self.identifier = compose("histre", parameter("bins", bins),
                                  parameter("pseudocount", pseudocount, 1))

    @parse_multivariate
    def multivariate(self, data):
        return histogram_relative_entropy(data, self._bins, self._pseudocount)[0]

    @parse_multivariate
    def diagnostics(self, data):
        return {"reason": histogram_relative_entropy(data, self._bins, self._pseudocount)[1]}

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return self.multivariate(data)[i, j]


class MultiscaleGraphCorrelation(Undirected, Unsigned):
    """Multiscale graph correlation"""

    name = "Multiscale graph correlation"
    identifier = "mgc"
    labels = ["distance", "unsigned", "unordered", "nonlinear", "undirected"]

    def input_problem(self, n_observations, n_processes):
        return self._minimum_observations_problem(n_observations, 5, "hyppo's MGC needs at least 5 samples")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        stat = MGC().statistic(x, y)
        return stat


class CrossDistanceCorrelation(Directed, Unsigned):
    """Cross-distance correlation"""

    name = "Cross-distance correlation"
    identifier = "dcorrx"
    labels = ["distance", "unsigned", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier = compose(self.identifier, parameter("maxlag", max_lag))

    def input_problem(self, n_observations, n_processes):
        # The largest lag correlates only the last n - max_lag samples.
        return self._minimum_observations_problem(
            n_observations, self._max_lag + 3, f"distance correlation of the {self._max_lag}-sample lag")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # DcorrX(x, y) relates x[t+lag] to y[t], i.e. y -> x; pass the target
        # first so [i, j] reads source i -> target j.
        z = data.to_numpy()
        x = z[j]
        y = z[i]
        stat, _ = DcorrX(max_lag=self._max_lag).statistic(x, y)
        return stat


class CrossMultiscaleGraphCorrelation(Directed, Unsigned):
    """Cross-multiscale graph correlation"""

    name = "Cross-multiscale graph correlation"
    identifier = "mgcx"
    labels = ["unsigned", "distance", "temporal", "directed", "nonlinear"]

    def __init__(self, max_lag=1):
        self._max_lag = max_lag
        self.identifier = compose(self.identifier, parameter("maxlag", max_lag))

    def input_problem(self, n_observations, n_processes):
        # MGC needs 5 samples, and the largest lag keeps only n - max_lag.
        return self._minimum_observations_problem(
            n_observations, self._max_lag + 5, f"MGC of the {self._max_lag}-sample lag needs 5 samples")

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        # As DcorrX: MGCX(x, y) measures y -> x, so the target goes first.
        z = data.to_numpy()
        x = z[j]
        y = z[i]
        stat, _, _ = MGCX(max_lag=self._max_lag).statistic(x, y)
        return stat


class TimeWarping(Undirected, Unsigned):

    labels = ["unsigned", "distance", "temporal", "undirected", "nonlinear"]

    def __init__(self, global_constraint=None):
        if global_constraint is not None:
            self.identifier = compose(self.identifier, word(global_constraint))
        self._global_constraint = global_constraint

    @property
    def simfn(self):
        try:
            return self._simfn
        except AttributeError:
            raise NotImplementedError(
                f"Add the similarity function for {self.identifier}"
            )

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        return self._simfn(z[i], z[j], global_constraint=self._global_constraint)


class DynamicTimeWarping(TimeWarping):

    name = "Dynamic time warping"
    identifier = "dtw"

    def __init__(self, global_constraint=None, sakoe_chiba_radius=1):
        super().__init__(global_constraint=global_constraint)
        if (isinstance(sakoe_chiba_radius, (bool, np.bool_))
                or not isinstance(sakoe_chiba_radius, (int, np.integer)) or sakoe_chiba_radius < 0):
            raise ValueError("sakoe_chiba_radius must be a nonnegative integer")
        if global_constraint != "sakoe_chiba" and sakoe_chiba_radius != 1:
            raise ValueError("sakoe_chiba_radius applies only to the sakoe_chiba constraint")
        self._radius = int(sakoe_chiba_radius)
        if self._radius != 1:
            self.identifier += "_" + parameter("radius", self._radius)
        self._simfn = tslearn.metrics.dtw

    @parse_multivariate
    def multivariate(self, data):
        if self._global_constraint == "sakoe_chiba":
            return banded_dtw(data, self._radius)
        return super().multivariate(data)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if self._global_constraint == "sakoe_chiba":
            return banded_dtw(data, self._radius)[i, j]
        return super().bivariate(data, i=i, j=j)


class LongestCommonSubsequence(TimeWarping):

    name = "Longest common subsequence"
    identifier = "lcss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.lcss


class SoftDynamicTimeWarping(TimeWarping):

    name = "Dynamic time warping"
    identifier = "softdtw"

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.soft_dtw(z[i], z[j])


class Barycenter(Directed, Signed):

    name = "Barycenter"
    identifier = "bary"
    labels = ["distance", "signed", "undirected", "temporal", "nonlinear"]

    def __init__(self, mode="euclidean", squared=False, statistic="mean"):
        if mode == "euclidean":
            pass
        elif mode == "dtw":
            self._fn = dtw_barycenter_averaging
        elif mode == "sgddtw":
            self._fn = dtw_barycenter_averaging_subgradient
        elif mode == "softdtw":
            self._fn = softdtw_barycenter
        else:
            raise NameError(f"Unknown Barycenter mode: {mode}")
        self._mode = mode

        self._squared = squared
        self._statistic = statistic
        self._preproc = lambda x: x
        if squared:
            self._preproc = lambda x: x**2
            self.issigned = lambda: False
            self.labels = [label for label in Barycenter.labels if label != "signed"] + ["unsigned"]

        if statistic == "mean":
            self._statfn = lambda x: np.nanmean(self._preproc(x))
        elif statistic == "max":
            self._statfn = lambda x: np.nanmax(self._preproc(x))
        elif statistic == "max_time":
            self._statfn = lambda x: np.argmax(self._preproc(x))
        else:
            raise NameError(f"Unknown statistic: {statistic}")

        self.identifier = compose(self.identifier, *(["sq"] if squared else []), word(mode), word(statistic))

    @parse_multivariate
    def multivariate(self, data):
        if self._mode == "euclidean":
            return euclidean_summary(data, self._statistic, self._squared)
        return super().multivariate(data)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):

        if self._mode == "euclidean":
            return euclidean_summary(data, self._statistic, self._squared)[i, j]

        try:
            bc = data.barycenter[self._mode][(i, j)]
        except (AttributeError, KeyError):
            z = data.to_numpy(squeeze=True)
            bc = self._fn(z[[i, j]])
            try:
                data.barycenter[self._mode][(i, j)] = bc
            except AttributeError:
                data.barycenter = {self._mode: {(i, j): bc}}
            except KeyError:
                data.barycenter[self._mode] = {(i, j): bc}
            data.barycenter[self._mode][(j, i)] = data.barycenter[self._mode][(i, j)]

        return self._statfn(bc)


class GromovWasserstainTau(Undirected, Unsigned):
    """Gromov-Wasserstain distance (GWTau)"""

    name = "Gromov-Wasserstain Distance"
    identifier = "gwtau"
    labels = ["unsigned", "distance", "unordered", "nonlinear", "undirected"]

    @staticmethod
    def vec_geo_dist(x):
        diffs = np.diff(x, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.cumsum(distances)
    
    @staticmethod
    def wass_sorted(x1, x2):
        x1 = np.sort(x1)[::-1] # sort in descending order
        x2 = np.sort(x2)[::-1] 

        if len(x1) == len(x2):
            res = np.sqrt(np.mean((x1 - x2) ** 2))
        else:
            N, M = len(x1), len(x2)
            i_ratios = np.arange(1, N + 1) / N
            j_ratios = np.arange(1, M + 1) / M
        
        
            min_values = np.minimum.outer(i_ratios, j_ratios)
            max_values = np.maximum.outer(i_ratios - 1/N, j_ratios - 1/M)
        
            lam = np.where(min_values > max_values, min_values - max_values, 0)
        
            diffs_squared = (x1[:, None] - x2) ** 2
            my_sum = np.sum(lam * diffs_squared)
        
            res = np.sqrt(my_sum)

        return res
    
    @staticmethod
    def gwtau(xi, xj):
        timei = np.arange(len(xi))
        timej = np.arange(len(xj))
        traji = np.column_stack([timei, xi])
        trajj = np.column_stack([timej, xj])

        vi = GromovWasserstainTau.vec_geo_dist(traji)
        vj = GromovWasserstainTau.vec_geo_dist(trajj)
        gw = GromovWasserstainTau.wass_sorted(vi, vj)
    
        return gw

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        x, y = data.to_numpy()[[i, j]]
        # insert compute SPI code here (computes on x and y)
        stat = self.gwtau(x, y)
        return stat
