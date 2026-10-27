"""Shared CPU spectral work, independent of SPI band summaries.

PSI implements Nolte et al. (2008), equation 3, on adjacent FFT bins in an
inclusive band. All Fourier estimators share a one-sided grid including DC and
Nyquist. Group delay derives its regression spacing from the transform's
spectral resolution. Two-sided coefficients remain available for Wilson
factorization.
"""

from collections import OrderedDict
from functools import cached_property
from itertools import combinations

import numpy as np
from scipy.stats import linregress
from spectral_connectivity.connectivity import (
    _estimate_predictive_power,
    _estimate_transfer_function,
    _find_significant_frequencies,
    _remove_instantaneous_causality,
)
from spectral_connectivity.statistics import coherence_bias

from pyspi._cache import ArrayCache, cached_workspace
from pyspi._spectral_compat import spectral_connectivity as sc


class SpectralConnectivity(sc.Connectivity):
    """One transform, shared powers/coherency, and bounded taper products.

    Only the trials+tapers expectation used by pyspi is supported here.
    The inherited directed estimators reuse cached factorization properties.
    """

    # This bounds each temporary complex outer-product block, independently of
    # channel count. Real transforms of a block can require extra temporary space.
    block_bytes = 16 * 2**20
    max_pairs = 16

    def __init__(self, fourier_coefficients, frequencies, time, *, frequency_resolution=None):
        super().__init__(fourier_coefficients, frequencies=frequencies, time=time)
        self.frequency_resolution = frequency_resolution
        self._arrays = ArrayCache()
        self._pairs = OrderedDict()

    @cached_property
    def frequencies(self):
        """DC through Nyquist, or the last positive bin for an odd FFT."""
        count = self.fourier_coefficients.shape[-2] // 2 + 1
        return np.abs(self.all_frequencies[:count])

    @cached_property
    def _power(self):
        return super()._power

    @cached_property
    def _minimum_phase_factor(self):
        return super()._minimum_phase_factor

    @cached_property
    def _transfer_function(self):
        return _estimate_transfer_function(self._minimum_phase_factor)[..., :self.frequencies.size, :, :]

    @cached_property
    def _noise_covariance(self):
        return super()._noise_covariance

    @cached_property
    def _MVAR_Fourier_coefficients(self):
        return super()._MVAR_Fourier_coefficients

    @property
    def _positive_coefficients(self):
        return self.fourier_coefficients[..., : self.frequencies.size, :]

    def power(self):
        return self._power[..., :self.frequencies.size, :]

    @staticmethod
    def _mean_cross_spectrum(coefficients):
        # Contract the observation axes directly: no (W,R,K,F,P,P) tensor.
        # Unoptimized einsum avoids BLAS introducing tiny imaginary parts into
        # exactly real cross-spectra. Near the negative real axis those can
        # change the phase branch by 2*pi, not merely by rounding error.
        result = np.einsum(
            "wrkfi,wrkfj->wfij", coefficients, coefficients.conj(),
            dtype=np.complex128, optimize=False,
        )
        result /= coefficients.shape[1] * coefficients.shape[2]
        return result

    def _expectation_cross_spectral_matrix(self, fcn=None, dtype=None):
        if fcn is not None:
            raise NotImplementedError("Use the bounded phase statistics for nonlinear expectations.")
        return self._mean_cross_spectrum(self.fourier_coefficients)

    @cached_property
    def _coherency(self):
        spectrum = self._mean_cross_spectrum(self._positive_coefficients)
        power = self.power()
        norm = np.sqrt(power[..., :, None] * power[..., None, :])
        norm[norm == 0] = np.nan
        spectrum /= norm
        diagonal = np.arange(spectrum.shape[-1])
        spectrum[..., diagonal, diagonal] = np.nan
        return spectrum

    def coherency(self):
        return self._coherency

    def phase_slope_index(self, frequencies_of_interest=None):
        """Raw PSI: sum Im(conj(C(f)) C(f+df)) over adjacent in-band bins.

        Both band endpoints are included; df is the existing FFT grid spacing.
        No frequencies are skipped or interpolated. DC and Nyquist participate
        when requested. Undefined coherency propagates to the affected pairs.
        Positive [i, j] means i leads j under the X_i * conj(X_j) convention.

        Reference: Nolte et al., Phys. Rev. Lett. 100, 234101 (2008), eq. 3.
        https://doi.org/10.1103/PhysRevLett.100.234101
        """
        frequencies = self.frequencies
        if frequencies_of_interest is None:
            first, last = 0, frequencies.size
        else:
            low, high = frequencies_of_interest
            first = np.searchsorted(frequencies, low, side="left")
            last = np.searchsorted(frequencies, high, side="right")
        count = last - first
        if count < 2:
            raise ValueError(
                f"PSI requires at least two FFT bins in the inclusive band; got {count}. "
                "No adjacent-frequency estimate is available."
            )

        coherency = self._coherency
        windows, _, channels, _ = coherency.shape
        result = np.zeros((windows, channels, channels))
        # Bound the conjugate temporary; never construct all frequency pairs.
        row_step = min(channels, max(1, self.block_bytes // (windows * channels * 16)))
        frequency_step = max(1, self.block_bytes // (windows * row_step * channels * 16))
        for row in range(0, channels, row_step):
            end = min(row + row_step, channels)
            for start in range(first, last - 1, frequency_step):
                stop = min(start + frequency_step, last - 1)
                left = coherency[:, start:stop, row:end, :]
                right = coherency[:, start+1:stop+1, row:end, :]
                result[:, row:end, :] += np.einsum(
                    "wfij,wfij->wij", left.conj(), right, optimize=False,
                ).imag
        return result

    def group_delay(self, frequencies_of_interest=None, significance_threshold=0.05):
        """Fit unwrapped coherence phase on spaced, significant in-band bins.

        The spacing is ceil(transform bandwidth / FFT spacing), independent of
        the requested band width. Keep the dependency's coherence significance
        test, largest-cluster selection, and minimum of three retained points.
        Nominal bandwidth spacing approximates independence; it is not a proof
        that neighboring spectral estimates are statistically independent.
        """
        frequencies = self.frequencies
        resolution = self.frequency_resolution
        if resolution is None or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Group delay requires the transform's positive spectral resolution.")
        ratio = resolution / (frequencies[1] - frequencies[0])
        # Unit conversion can put an integer ratio a few ulps above itself.
        nearest = np.rint(ratio)
        if np.isclose(ratio, nearest, rtol=8 * np.finfo(float).eps, atol=0):
            ratio = nearest
        step = max(1, int(np.ceil(ratio)))
        selected = (np.ones(frequencies.size, dtype=bool) if frequencies_of_interest is None
                    else frequency_mask(frequencies, *frequencies_of_interest))
        frequencies = frequencies[selected]
        if frequencies[::step].size < 3:
            raise ValueError(
                "Group delay requires at least three frequency bins spaced by the "
                "transform's spectral resolution in the inclusive band."
            )
        coherency = self.coherency()[:, selected]
        windows, _, channels, _ = coherency.shape
        slope = np.full((windows, channels, channels), np.nan)
        rvalue = np.full_like(slope, np.nan)
        i, j = np.triu_indices(channels, 1)
        if not i.size:
            return slope.copy(), slope, rvalue
        pairs = coherency[..., i, j]
        significant = _find_significant_frequencies(
            pairs, coherence_bias(self.n_observations), step,
            significance_threshold=significance_threshold,
        )
        phase = np.unwrap(np.angle(pairs), axis=1)
        for window in range(windows):
            for pair, (row, col) in enumerate(zip(i, j)):
                keep = significant[window, :, pair]
                if keep.sum() < 3:
                    continue
                fit = linregress(frequencies[keep], phase[window, keep, pair])
                slope[window, row, col] = fit.slope
                slope[window, col, row] = -fit.slope
                # r is the signed correlation of frequency with unwrapped
                # phase. Reversing the pair negates phase, hence negates r.
                rvalue[window, row, col] = fit.rvalue
                rvalue[window, col, row] = -fit.rvalue
        return slope / (2 * np.pi), slope, rvalue

    def imaginary_coherence(self):
        # The public matrix wrapper excludes the diagonal for all these measures.
        return np.abs(self.coherency().imag)

    def _phase_statistic(self, kind):
        return self._phase_statistics(kind)[kind]

    def _phase_statistics(self, *kinds):
        keys = [("phase_statistic", kind) for kind in kinds]
        def compute(missing):
            reduced = self._reduce_phase([key[1] for key in missing])
            return {key: reduced[key[1]] for key in missing}
        return {key[1]: value for key, value in self._arrays.get_many(keys, compute).items()}

    def _reduce_phase(self, kinds):
        """Form each bounded taper cross-product once for the requested moments."""
        coefficients = self._positive_coefficients
        w, r, k, f, p = coefficients.shape
        results = {kind: np.empty((w, f, p, p), dtype=complex if kind == "unit" else float)
                   for kind in kinds}
        row_step = min(p, max(1, self.block_bytes // (w * r * k * p * 16)))
        frequency_step = max(1, self.block_bytes // (w * r * k * row_step * p * 16))
        for start in range(0, f, frequency_step):
            stop = min(start + frequency_step, f)
            right = coefficients[..., start:stop, None, :].conj()
            for row in range(0, p, row_step):
                end = min(row + row_step, p)
                left = coefficients[..., start:stop, row:end, None]
                cross = np.matmul(left, right, dtype=np.complex128)
                for kind in kinds:
                    if kind == "unit":
                        values = cross / np.abs(cross)
                    elif kind == "sign":
                        values = np.sign(cross.imag)
                    elif kind == "imag":
                        values = cross.imag
                    elif kind == "abs_imag":
                        values = np.abs(cross.imag)
                    elif kind == "imag_squared":
                        values = cross.imag ** 2
                    else:
                        raise ValueError(f"Unknown phase statistic: {kind}")
                    results[kind][:, start:stop, row:end, :] = values.mean(axis=(1, 2))
                    del values
        return results

    def _phase_locking_value(self):
        return self._phase_statistic("unit")

    def phase_lag_index(self):
        return np.abs(self.signed_phase_lag_index())

    def signed_phase_lag_index(self):
        return self._phase_statistic("sign")

    def debiased_squared_phase_lag_index(self):
        # Squaring the shared signed moment already removes its sign. Avoid
        # constructing the ordinary PLI's full-frequency magnitude array.
        n = self.n_observations
        return (n * self.signed_phase_lag_index() ** 2 - 1) / (n - 1)

    def weighted_phase_lag_index(self):
        return np.abs(self.measure("signed_weighted_phase_lag_index"))

    def signed_weighted_phase_lag_index(self):
        moments = self._phase_statistics("imag", "abs_imag")
        weights = moments["abs_imag"].copy()
        weights[weights < np.finfo(float).eps] = 1
        return moments["imag"] / weights

    def debiased_squared_weighted_phase_lag_index(self):
        n = self.n_observations
        moments = self._phase_statistics("imag", "imag_squared", "abs_imag")
        imag_sum = moments["imag"] * n
        squared_sum = moments["imag_squared"] * n
        magnitude_sum = moments["abs_imag"] * n
        weights = magnitude_sum ** 2 - squared_sum
        weights[weights == 0] = np.nan
        return (imag_sum ** 2 - squared_sum) / weights

    def measure(self, name, **kwargs):
        if name == "signed_phase_lag_index":
            # Already cached as the sign moment; do not count the same array
            # twice against the cache's byte budget.
            return self.signed_phase_lag_index()
        key = ("measure", name, tuple(sorted(kwargs.items())))
        return self._arrays.get(key, lambda: getattr(self, name)(**kwargs))

    def pair(self, i, j):
        """Reuse an ordered two-channel model, with bounded storage.

        The dependency's factorization is observably order-sensitive. Sharing
        between (i,j) and (j,i) requires a separate numerical/scientific review.
        """
        key = (i, j)
        if key not in self._pairs:
            self._pairs[key] = SpectralConnectivity(
                self.fourier_coefficients[..., list(key)], self.all_frequencies, self.time,
                frequency_resolution=self.frequency_resolution,
            )
            if len(self._pairs) > self.max_pairs:
                self._pairs.popitem(last=False)
        self._pairs.move_to_end(key)
        return self._pairs[key]

    def pairwise_spectral_granger_prediction(self):
        # Same pairwise estimator as the dependency, sharing the pair models used
        # by directed coherence. Avoid constructing a full two-sided P x P CSM.
        w, _, _, _, p = self.fourier_coefficients.shape
        result = np.full((w, self.frequencies.size, p, p), np.nan)
        for i, j in combinations(range(p), 2):
            pair = self.pair(i, j)
            try:
                values = _estimate_predictive_power(
                    pair._power[..., : self.frequencies.size, :],
                    _remove_instantaneous_causality(pair._noise_covariance),
                    pair._transfer_function,
                )
                result[..., i, j] = values[..., 0, 1]
                result[..., j, i] = values[..., 1, 0]
            except np.linalg.LinAlgError:
                pass  # The dependency marks a failed pair NaN as well.
        return result

    def ordered_pair_measure(self, name):
        """Retain a measure's spectrum across bands even when pair models evict."""
        return self._arrays.get(("ordered_pairs", name), lambda: self._ordered_pair_measure(name))

    def _ordered_pair_measure(self, name):
        w, _, _, _, p = self.fourier_coefficients.shape
        result = np.full((w, self.frequencies.size, p, p), np.nan)
        for j in range(p):
            for i in range(p):
                if i != j:
                    result[..., i, j] = self.pair(i, j).measure(name)[..., 0, 1]
        return result


class _DataSpectralCache:
    def __init__(self, source):
        self.source = source
        self.workspaces = OrderedDict()
        self.parametric = ArrayCache()


def data_spectral_cache(data):
    return cached_workspace(data, "spectral", _DataSpectralCache)


def spectral_workspace(data, sampling_frequency):
    if data.n_replications != 1:
        raise ValueError("Spectral SPIs require a single realisation.")
    if data.n_observations <= 6:
        raise ValueError("Multitaper with time_halfbandwidth_product=3 requires more than six observations.")
    cache = data_spectral_cache(data)
    # These are all transform settings exposed by this SPI family. Make the
    # previously implicit dependency defaults explicit, and key them together.
    settings = (sampling_frequency, 3, "constant", True)
    if settings not in cache.workspaces:
        m = sc.Multitaper(
            cache.source[:, :, 0].T,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=3,
            detrend_type="constant",
            is_low_bias=True,
        )
        cache.workspaces[settings] = SpectralConnectivity(
            m.fft(), m.frequencies, m.time, frequency_resolution=m.frequency_resolution,
        )
        if len(cache.workspaces) > 2:
            cache.workspaces.popitem(last=False)
    cache.workspaces.move_to_end(settings)
    return cache.workspaces[settings]


def frequency_mask(frequencies, fmin, fmax):
    """Select both endpoints without interpolating or widening the band."""
    return (frequencies >= fmin) & (frequencies <= fmax)


def validate_frequencies(fs, fmin, fmax):
    """All frequencies share the units of fs (Hz when fs is samples/second)."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be finite and positive.")
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 0 <= fmin <= fmax <= fs / 2):
        raise ValueError("Frequency band must satisfy 0 <= fmin <= fmax <= fs/2.")
