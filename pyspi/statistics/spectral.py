import numpy as np

from pyspi._spectral import (
    PHASE_MOMENTS,
    band_bins,
    band_problem,
    canonical_directed_measure,
    data_spectral_cache,
    directed_workspace,
    frequency_mask,
    group_delay_step,
    multitaper_grid,
    one_sided_frequencies,
    spectral_workspace,
    validate_frequencies,
)
from pyspi._naming import compose, frequency_band, parameter, word
from pyspi.base import (
    Directed,
    Undirected,
    Signed,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
import nitime.analysis as nta
import nitime.timeseries as ts
import warnings


class NonparametricSpectral:
    """Base class for the nonparametric spectral methods from the Eden-Kramer repo"""

    _absolute_band_values = False
    _name_variants = ()
    #: Fewest in-band frequencies the estimator needs (PSI needs an adjacent pair).
    _minimum_bins = 1

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
        self.identifier = compose(self.identifier, *self._name_variants, word(statistic),
                                  frequency_band(fmin, fmax, fs))

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f"Include measure for {self.identifier}")

    def _spectral_options(self):
        return {}

    def _band_grid(self, n_observations):
        return multitaper_grid(n_observations, self._fs)[0]

    def input_problem(self, n_observations, n_processes=None):
        """Why this SPI has no estimate for windows of n_observations samples, or None.

        Uses the calculation's own transform limits, frequency grid and inclusive
        band selection, so it can be checked before any data are processed.
        """
        return self._transform_problem(n_observations) or self._band_problem(n_observations)

    def _transform_problem(self, n_observations):
        return self._minimum_observations_problem(n_observations, 7, "the multitaper transform with NW=3")

    def _band_problem(self, n_observations):
        frequencies = self._band_grid(n_observations)
        found = band_bins(frequencies, self._fmin, self._fmax)
        return band_problem(self, n_observations, frequencies, found, self._minimum_bins, self._minimum_bins)

    def _summarise_band(self, values, frequencies):
        # The shared one-sided grid is sorted. An inclusive slice avoids
        # copying the full frequency/channel tensor just to select a band.
        first = np.searchsorted(frequencies, self._fmin, side="left")
        last = np.searchsorted(frequencies, self._fmax, side="right")
        if first == last:
            # No estimate is available; zero would imply a measured absence of
            # connectivity. PSI/GD retain their own estimator's requirements.
            return np.full(values.shape[1:], np.nan)
        band = values[first:last]
        if self._absolute_band_values:
            band = np.abs(band)
        return self._statfn(band, axis=0)


class NonparametricSpectralMultivariate(NonparametricSpectral):
    def _register_shared_work(self, data):
        kinds = PHASE_MOMENTS.get(self.measure)
        if kinds:
            data_spectral_cache(data).phase_requests.setdefault(self._fs, set()).update(kinds)

    def _get_cache(self, data):
        conn = spectral_workspace(data, self._fs)
        return conn.measure(self.measure, **self._spectral_options()), conn.frequencies

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        if isinstance(self, (PhaseSlopeIndex, CoherencePhase)):
            adj = adj_freq[0].copy()
        elif isinstance(self, GroupDelay):
            stat_id = ("delay", "slope", "rvalue").index(self._statistic)
            adj = adj_freq[stat_id][0].copy()
        else:
            adj = self._summarise_band(adj_freq[0], freq)
        np.fill_diagonal(adj, np.nan)
        return adj


class NonparametricSpectralBivariate(NonparametricSpectral):
    def _band_grid(self, n_observations):
        # directed_workspace transforms at the native, unpadded length.
        return one_sided_frequencies(n_observations, self._fs)

    def _register_shared_work(self, data):
        cache = data_spectral_cache(data)
        cache.directed_requests.setdefault(self._fs, set()).add(
            (canonical_directed_measure(self.measure), (self._fmin, self._fmax)))

    def _get_cache(self, data):
        conn = directed_workspace(data, self._fs)
        return conn.pairwise_measure(self.measure), conn.frequencies

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        if i is None or j is None:
            raise ValueError("Both channel indices are required for a bivariate spectral calculation.")
        if i == j:
            return np.nan
        conn = directed_workspace(data, self._fs)
        band = (self._fmin, self._fmax)
        key = ("directed_summary", canonical_directed_measure(self.measure), band)
        statistic = ("mean", "max").index(self._statistic)
        if key in conn._arrays.entries:
            return conn.directed_summary(self.measure, band)[statistic][0, i, j]
        values = conn.pair(i, j).directed_summary(self.measure, band)[statistic][0]
        return values[0, 1] if i < j else values[1, 0]

    @parse_multivariate
    def multivariate(self, data):
        conn = directed_workspace(data, self._fs)
        statistic = ("mean", "max").index(self._statistic)
        result = conn.directed_summary(self.measure, (self._fmin, self._fmax))[statistic][0].copy()
        np.fill_diagonal(result, np.nan)
        return result

    @parse_multivariate
    def diagnostics(self, data):
        conn = directed_workspace(data, self._fs)
        return {key: value[0].copy() for key, value in
                conn.directed_diagnostics(self.measure, (self._fmin, self._fmax)).items()}


class CoherenceMagnitude(NonparametricSpectralMultivariate, Undirected, Unsigned):
    name = "Coherence magnitude"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "cohmag"
        super().__init__(**kwargs)
        self._measure = "coherence_magnitude"


class CoherencePhase(NonparametricSpectralMultivariate, Directed, Signed):
    """Equal-weight circular mean across the inclusive frequency band."""

    name = "Coherence phase"
    labels = ["signed", "spectral", "directed"]

    def __init__(self, statistic="mean", **kwargs):
        if statistic != "mean":
            raise ValueError(
                "Coherence phase supports only statistic='mean' (circular mean); "
                "angles have no ordinary maximum."
            )
        self.identifier = "phase"
        super().__init__(statistic=statistic, **kwargs)
        self._measure = "coherence_phase_mean"

    def _spectral_options(self):
        return {"frequencies_of_interest": (self._fmin, self._fmax)}


class ImaginaryCoherence(NonparametricSpectralMultivariate, Undirected, Unsigned):
    name = "Imaginary coherence"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "icoh"
        super().__init__(**kwargs)
        self._measure = "imaginary_coherence"


class PhaseLockingValue(NonparametricSpectralMultivariate, Undirected, Unsigned):
    name = "Phase locking value"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "plv"
        super().__init__(**kwargs)
        self._measure = "phase_locking_value"


class PhaseLagIndex(NonparametricSpectralMultivariate, Undirected, Unsigned):
    name = "Phase lag index"
    labels = ["unsigned", "spectral", "undirected"]
    _absolute_band_values = True

    def __init__(self, **kwargs):
        self.identifier = "pli"
        super().__init__(**kwargs)
        self._measure = "signed_phase_lag_index"


class SignedPhaseLagIndex(NonparametricSpectralMultivariate, Directed, Signed):
    """Phase-lag sign consistency; the sign alone is not causal direction."""

    name = "Signed phase lag index"
    labels = ["signed", "spectral", "directed"]
    _name_variants = ("signed",)

    def __init__(self, **kwargs):
        self.identifier = "pli"
        super().__init__(**kwargs)
        self._measure = "signed_phase_lag_index"


class WeightedPhaseLagIndex(NonparametricSpectralMultivariate, Undirected, Unsigned):
    name = "Weighted phase lag index"
    labels = ["unsigned", "spectral", "undirected"]
    _absolute_band_values = True

    def __init__(self, **kwargs):
        self.identifier = "wpli"
        super().__init__(**kwargs)
        self._measure = "signed_weighted_phase_lag_index"


class SignedWeightedPhaseLagIndex(NonparametricSpectralMultivariate, Directed, Signed):
    """Weighted phase-lag sign consistency, sharing the unsigned estimator's work."""

    name = "Signed weighted phase lag index"
    labels = ["signed", "spectral", "directed"]
    _name_variants = ("signed",)

    def __init__(self, **kwargs):
        self.identifier = "wpli"
        super().__init__(**kwargs)
        self._measure = "signed_weighted_phase_lag_index"


class DebiasedSquaredPhaseLagIndex(NonparametricSpectralMultivariate, Undirected, Signed):
    name = "Debiased squared phase lag index"
    labels = ["signed", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dspli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_phase_lag_index"


class DebiasedSquaredWeightedPhaseLagIndex(
    NonparametricSpectralMultivariate, Undirected, Signed
):
    name = "Debiased squared weighted phase-lag index"
    labels = ["signed", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dswpli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_weighted_phase_lag_index"


class PairwisePhaseConsistency(NonparametricSpectralMultivariate, Undirected, Signed):
    name = "Pairwise phase consistency"
    labels = ["signed", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "ppc"
        super().__init__(**kwargs)
        self._measure = "pairwise_phase_consistency"


"""
    These next several seem to segfault for large vector autoregressive processes (something to do with np.linalg solver).
    Switched them to bivariate for now until the issue is resolved
"""


class DirectedCoherence(NonparametricSpectralBivariate, Directed, Unsigned):
    """Squared directed coherence at each frequency, then band mean/max."""

    name = "Directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "dcoh"
        super().__init__(**kwargs)
        self._measure = "directed_coherence"


class PartialDirectedCoherence(NonparametricSpectralBivariate, Directed, Unsigned):
    name = "Partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "pdcoh"
        super().__init__(**kwargs)
        self._measure = "partial_directed_coherence"


class GeneralizedPartialDirectedCoherence(NonparametricSpectralBivariate, Directed, Unsigned):
    name = "Generalized partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "gpdcoh"
        super().__init__(**kwargs)
        self._measure = "generalized_partial_directed_coherence"


class DirectedTransferFunction(NonparametricSpectralBivariate, Directed, Unsigned):
    name = "Directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "dtf"
        super().__init__(**kwargs)
        self._measure = "directed_transfer_function"


class DirectDirectedTransferFunction(NonparametricSpectralBivariate, Directed, Unsigned):
    name = "Direct directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "ddtf"
        super().__init__(**kwargs)
        self._measure = "direct_directed_transfer_function"


class PhaseSlopeIndex(NonparametricSpectralMultivariate, Directed, Signed):
    name = "Phase slope index"
    labels = ["signed", "spectral", "directed"]

    _minimum_bins = 2

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        self._measure = "phase_slope_index"

    def _spectral_options(self):
        return dict(
            frequencies_of_interest=(self._fmin, self._fmax),
        )


class GroupDelay(NonparametricSpectralMultivariate, Directed, Signed):
    name = "Group delay"
    labels = ["signed", "spectral", "directed", "lagged"]

    def __init__(self, statistic="delay", **kwargs):
        if statistic not in ("delay", "slope", "rvalue"):
            raise ValueError("Group delay statistic must be delay, slope, or rvalue.")
        self.identifier = "gd"
        super().__init__(statistic=statistic, **kwargs)
        self._measure = "group_delay"

    def _band_problem(self, n_observations):
        # The regression needs three in-band points spaced by the resolution.
        frequencies, resolution = multitaper_grid(n_observations, self._fs)
        step = group_delay_step(frequencies, resolution)
        first = np.searchsorted(frequencies, self._fmin, side="left")
        last = np.searchsorted(frequencies, self._fmax, side="right")
        found = frequencies[first:last][::step].size
        return band_problem(self, n_observations, frequencies, found, 3, 2 * step + 1)

    def _spectral_options(self):
        return dict(
            frequencies_of_interest=(self._fmin, self._fmax),
        )

    @parse_multivariate
    def diagnostics(self, data):
        """Return detached pair-wise reasons/counts for this band's delay fit.

        Unsupported bands raise the same descriptive error as multivariate().
        A constant phase has an estimated zero delay and an undefined rvalue.
        """
        from pyspi._group_delay import REASONS
        result = spectral_workspace(data, self._fs).group_delay_details(**self._spectral_options())
        return dict(reason=np.asarray(REASONS)[result[3][0]],
                    retained_points=result[4][0].copy(), invalid_bins=result[5][0].copy())


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
        self._statistic = statistic
        if method not in ("nonparametric", "parametric"):
            raise ValueError("Spectral Granger method must be nonparametric or parametric.")

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
            variants, parameters = [], []
        else:
            self._order = order
            self._max_order = max_order
            # Without a fixed order, the model order is selected up to max_order.
            variants = ["autoorder"] if order is None else []
            parameters = [parameter("maxorder", max_order) if order is None else parameter("order", order)]
        self.identifier = compose(self.identifier, method, *variants, *([] if ignore_nan else ["nanstrict"]),
                                  word(statistic), frequency_band(self._fmin, self._fmax, fs), *parameters)

    def _get_cache(self, data):
        if self._method == "nonparametric":
            conn = directed_workspace(data, self._fs)
            return conn.pairwise_spectral_granger_prediction(), conn.frequencies
        if data.n_replications != 1:
            raise ValueError("Spectral SPIs require a single realisation.")
        cache = data_spectral_cache(data)
        key = (self._fs, self._order, self._max_order)
        return cache.parametric.get(key, lambda: self._parametric_spectrum(data))[:2]

    def _register_shared_work(self, data):
        if self._method == "nonparametric":
            NonparametricSpectralBivariate._register_shared_work(self, data)

    def _transform_problem(self, n_observations):
        if self._method == "nonparametric":
            return super()._transform_problem(n_observations)
        if self._order is None:
            # nitime may select any order below max_order from the data.
            return self._minimum_observations_problem(
                n_observations, self._max_order, f"AR order selection up to order {self._max_order - 1}")
        # The backend requests autocovariances through lag `order`. A larger
        # empirical reliability margin is not a mathematical impossibility.
        return self._minimum_observations_problem(
            n_observations, self._order + 1, f"autocovariances through lag {self._order}")

    def _band_grid(self, n_observations):
        if self._method == "nonparametric":
            return NonparametricSpectralBivariate._band_grid(self, n_observations)
        # nitime's transfer grid has 1024 points whatever the window length.
        return np.fft.rfftfreq(1024, 1/self._fs)

    def _parametric_spectrum(self, data):
        # Retain nitime's pairwise model/order-selection definition and 1024-bin
        # transfer grid. Isolate a failed pair instead of discarding every pair.
        raw = data.to_numpy(realisation=0)
        p = data.n_processes
        frequencies = np.fft.rfftfreq(1024, 1/self._fs)
        result = np.full((1, len(frequencies), p, p), np.nan)
        reason = np.full((p, p), "diagonal", dtype="U32")
        for i in range(p):
            for j in range(i+1, p):
                time_series = ts.TimeSeries(raw[[i, j]], sampling_interval=1/self._fs)
                analyzer = nta.GrangerAnalyzer(time_series, ij=[(0, 1)], order=self._order,
                                               max_order=self._max_order, n_freqs=1024)
                try:
                    forward = analyzer.causality_xy[0, 1]
                    reverse = analyzer.causality_yx[0, 1]
                    result[0, :, i, j], result[0, :, j, i] = forward, reverse
                    reason[i, j] = "estimated" if np.isfinite(forward).all() else "undefined_frequency_values"
                    reason[j, i] = "estimated" if np.isfinite(reverse).all() else "undefined_frequency_values"
                except (ValueError, np.linalg.LinAlgError):
                    reason[i, j] = reason[j, i] = "model_fit_failed"
        return result, frequencies, reason

    @parse_multivariate
    def diagnostics(self, data):
        if self._method == "nonparametric":
            conn = directed_workspace(data, self._fs)
            return {key: value[0].copy() for key, value in
                    conn.directed_diagnostics(self.measure, (self._fmin, self._fmax)).items()}
        self._get_cache(data)
        cache = data_spectral_cache(data)
        key = (self._fs, self._order, self._max_order)
        values, frequencies, reasons = cache.parametric.get(key, lambda: self._parametric_spectrum(data))
        mask = frequency_mask(frequencies, self._fmin, self._fmax)
        fraction = (~np.isfinite(values[0, mask])).mean(axis=0) if mask.any() else np.ones(reasons.shape)
        return dict(reason=reasons.copy(), unavailable_fraction=fraction)

    @parse_multivariate
    def multivariate(self, data):
        if self._method == "nonparametric":
            conn = directed_workspace(data, self._fs)
            summaries = conn.directed_summary(self.measure, (self._fmin, self._fmax))
            result = summaries[("mean", "max").index(self._statistic)][0].copy()
            nan_pct = summaries[2][0].copy()
            if not self.ignore_nan:
                result[nan_pct > 0] = np.nan
        else:
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
