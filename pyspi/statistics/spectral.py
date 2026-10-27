import numpy as np
from copy import deepcopy

from pyspi._spectral import (
    data_spectral_cache,
    frequency_mask,
    spectral_workspace,
    validate_frequencies,
)
from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
import nitime.analysis as nta
import nitime.timeseries as ts
import warnings


class NonparametricSpectral(Unsigned):
    """Base class for the nonparametric spectral methods from the Eden-Kramer repo"""

    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean"):
        if fmax is None:
            fmax = fs / 2
        validate_frequencies(fs, fmin, fmax)
        self._fs = fs
        self._fmin = fmin
        self._fmax = fmax
        if statistic == "mean":
            self._statfn = np.nanmean
        elif statistic == "max":
            self._statfn = np.nanmax
        elif statistic not in ["delay", "slope", "rvalue"]:
            raise NameError(f"Unknown statistic: {statistic}")
        else:
            self._statfn = None
        self._statistic = statistic
        paramstr = (
            f"_multitaper_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}".replace(
                ".", "-"
            )
        )
        self.identifier += paramstr

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f"Include measure for {self.identifier}")

    def _spectral_options(self):
        return {}

    def _summarise_band(self, values, frequencies):
        selected = frequency_mask(frequencies, self._fmin, self._fmax)
        if not selected.any():
            # No estimate is available; zero would imply a measured absence of
            # connectivity. PSI/GD retain their own estimator's requirements.
            return np.full(values.shape[1:], np.nan)
        return self._statfn(values[selected], axis=0)


class NonparametricSpectralMultivariate(NonparametricSpectral):
    def _get_cache(self, data):
        conn = spectral_workspace(data, self._fs)
        return conn.measure(self.measure, **self._spectral_options()), conn.frequencies

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        if isinstance(self, PhaseSlopeIndex):
            adj = adj_freq[0].copy()
        elif isinstance(self, GroupDelay):
            stat_id = ("delay", "slope", "rvalue").index(self._statistic)
            adj = adj_freq[stat_id][0].copy()
        else:
            adj = self._summarise_band(adj_freq[0], freq)
        np.fill_diagonal(adj, np.nan)
        return adj


class NonparametricSpectralBivariate(NonparametricSpectral):
    def _get_cache(self, data):
        conn = spectral_workspace(data, self._fs)
        return conn.ordered_pair_measure(self.measure), conn.frequencies

    def _pair_summary(self, conn, i, j):
        pair = conn.pair(i, j)
        values = pair.measure(self.measure)
        return self._summarise_band(values[0], pair.frequencies)

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if i is None or j is None:
            raise ValueError("Both channel indices are required for a bivariate spectral calculation.")
        if i == j:
            return np.nan
        conn = spectral_workspace(data, self._fs)
        values = self._pair_summary(conn, i, j)
        return values[0, 1]

    @parse_multivariate
    def multivariate(self, data):
        values, frequencies = self._get_cache(data)
        result = self._summarise_band(values[0], frequencies)
        np.fill_diagonal(result, np.nan)
        return result


class CoherenceMagnitude(NonparametricSpectralMultivariate, Undirected):
    name = "Coherence magnitude"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "cohmag"
        super().__init__(**kwargs)
        self._measure = "coherence_magnitude"


class CoherencePhase(NonparametricSpectralMultivariate, Undirected):
    name = "Coherence phase"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "phase"
        super().__init__(**kwargs)
        self._measure = "coherence_phase"


class ImaginaryCoherence(NonparametricSpectralMultivariate, Undirected):
    name = "Imaginary coherence"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "icoh"
        super().__init__(**kwargs)
        self._measure = "imaginary_coherence"


class PhaseLockingValue(NonparametricSpectralMultivariate, Undirected):
    name = "Phase locking value"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "plv"
        super().__init__(**kwargs)
        self._measure = "phase_locking_value"

        myfn = deepcopy(self._statfn)
        self._statfn = lambda x, **kwargs: myfn(np.absolute(x), **kwargs)


class PhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "pli"
        super().__init__(**kwargs)
        self._measure = "phase_lag_index"


class WeightedPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Weighted phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "wpli"
        super().__init__(**kwargs)
        self._measure = "weighted_phase_lag_index"


class DebiasedSquaredPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Debiased squared phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dspli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_phase_lag_index"


class DebiasedSquaredWeightedPhaseLagIndex(
    NonparametricSpectralMultivariate, Undirected
):
    name = "Debiased squared weighted phase-lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dswpli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_weighted_phase_lag_index"


class PairwisePhaseConsistency(NonparametricSpectralMultivariate, Undirected):
    name = "Pairwise phase consistency"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "ppc"
        super().__init__(**kwargs)
        self._measure = "pairwise_phase_consistency"


"""
    These next several seem to segfault for large vector autoregressive processes (something to do with np.linalg solver).
    Switched them to bivariate for now until the issue is resolved
"""


class DirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "dcoh"
        super().__init__(**kwargs)
        self._measure = "directed_coherence"


class PartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "pdcoh"
        super().__init__(**kwargs)
        self._measure = "partial_directed_coherence"


class GeneralizedPartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Generalized partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "gpdcoh"
        super().__init__(**kwargs)
        self._measure = "generalized_partial_directed_coherence"


class DirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    name = "Directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "dtf"
        super().__init__(**kwargs)
        self._measure = "directed_transfer_function"


class DirectDirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    name = "Direct directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "ddtf"
        super().__init__(**kwargs)
        self._measure = "direct_directed_transfer_function"


class PhaseSlopeIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Phase slope index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        self._measure = "phase_slope_index"

    def _spectral_options(self):
        return dict(
            frequencies_of_interest=(self._fmin, self._fmax),
        )


class GroupDelay(NonparametricSpectralMultivariate, Directed):
    name = "Group delay"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "gd"
        super().__init__(**kwargs)
        self._measure = "group_delay"

    def _spectral_options(self):
        return dict(
            frequencies_of_interest=(self._fmin, self._fmax),
        )


class SpectralGrangerCausality(NonparametricSpectralMultivariate, Directed, Unsigned):
    name = "Spectral Granger causality"
    identifier = "sgc"
    labels = ["unsigned", "embedding", "spectral", "directed", "lagged"]

    def __init__(
        self,
        fs=1,
        fmin=0,
        fmax=None,
        method="nonparametric",
        order=None,
        max_order=50,
        statistic="mean",
        ignore_nan=True,
        nan_threshold=0.5,
    ):
        if fmax is None:
            fmax = fs / 2
        self._fs = fs
        self._fmin = fmin
        self._fmax = fmax
        self.ignore_nan = ignore_nan
        self.nan_threshold = nan_threshold

        validate_frequencies(fs, self._fmin, self._fmax)

        if statistic == "mean":
            if self.ignore_nan:
                self._statfn = np.nanmean
            else:
                self._statfn = np.mean
        elif statistic == "max":
            if self.ignore_nan:
                self._statfn = np.nanmax
            else:
                self._statfn = np.max
        else:
            raise NameError(f"Unknown statistic {statistic}")

        self._method = method
        if self._method == "nonparametric":
            self._measure = "pairwise_spectral_granger_prediction"
            paramstr = f"_nonparametric_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}".replace(
                ".", "-"
            )
        else:
            self._order = order
            self._max_order = max_order
            paramstr = f"_parametric_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}_order-{order}".replace(
                ".", "-"
            )

        self.identifier = self.identifier + paramstr

    def _get_cache(self, data):
        if self._method == "nonparametric":
            return super()._get_cache(data)
        if data.n_replications != 1:
            raise ValueError("Spectral SPIs require a single realisation.")
        cache = data_spectral_cache(data)
        key = (self._fs, self._order, self._max_order)
        return cache.parametric.get(key, lambda: self._parametric_spectrum(data))

    def _parametric_spectrum(self, data):
        time_series = ts.TimeSeries(data.to_numpy(realisation=0), sampling_interval=1 / self._fs)
        GA = nta.GrangerAnalyzer(time_series, order=self._order, max_order=self._max_order)
        triu_id = np.triu_indices(data.n_processes)
        F = np.full(GA.causality_xy.shape, np.nan)
        F[triu_id[0], triu_id[1], :] = GA.causality_xy[triu_id[0], triu_id[1], :]
        F[triu_id[1], triu_id[0], :] = GA.causality_yx[triu_id[0], triu_id[1], :]
        F = np.transpose(np.expand_dims(F, axis=3), axes=[3, 2, 1, 0])
        return F, GA.frequencies

    @parse_multivariate
    def multivariate(self, data):
        try:
            cache, freq = self._get_cache(data)
            freq_id = frequency_mask(freq, self._fmin, self._fmax)
            if not freq_id.any():
                return np.full((data.n_processes, data.n_processes), np.nan)
            result = self._summarise_band(cache[0], freq)

            nan_pct = np.isnan(cache[0, freq_id, :, :]).mean(axis=0)
            np.fill_diagonal(nan_pct, 0.0)

            isna = nan_pct > self.nan_threshold
            if isna.any():
                warnings.warn(
                    f"Spectral GC: the following processes have >{self.nan_threshold*100:.1f}% "
                    f"NaN values:\n{np.transpose(np.where(isna))}\nThese indices will be set to NaN. "
                    "Set ignore_nan to False or modify nan_threshold parameter if required."
                )
                result[isna] = np.nan

            return result
        except ValueError as err:
            warnings.warn(str(err))
            return np.full((data.n_processes, data.n_processes), np.nan)
