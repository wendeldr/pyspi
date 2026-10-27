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

    def directed_coherence(self):
        """Squared DC: source-variance-weighted inflow, normalized per target.

        DC[i,j] = var[j] * abs(H[i,j])**2 / sum_k(var[k] * abs(H[i,k])**2).
        The matrix convention remains row=target, column=source. This uses
        diagonal innovation variances; correlated innovations prevent reading
        the normalized strengths as exact shares of observed target power.
        """
        variance = np.diagonal(self._noise_covariance, axis1=-2, axis2=-1)
        amplitude = np.abs(self._transfer_function)
        amplitude *= np.sqrt(variance)[..., None, None, :]
        # A common row scale cancels from the ratio. Scale amplitudes before
        # squaring to avoid overflow/underflow after changes of channel units.
        scale = amplitude.max(axis=-1, keepdims=True)
        np.divide(amplitude, scale, out=amplitude, where=scale > 0)
        np.square(amplitude, out=amplitude)
        total = amplitude.sum(axis=-1, keepdims=True)
        return np.divide(amplitude, total, out=np.full_like(amplitude, np.nan),
                         where=total > 0)

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

    def coherence_phase_mean(self, frequencies_of_interest=None):
        """Equal-weight circular mean of defined in-band coherency phases.

        Normalize complex coherency before summing: every nonzero, finite bin
        has equal weight, regardless of its magnitude. Zero coherency has no
        phase. Missing bins are omitted; an empty or numerically zero resultant
        gives NaN. Results are radians in [-pi, pi] (the endpoints coincide).
        """
        first, last = 0, self.frequencies.size
        if frequencies_of_interest is not None:
            low, high = frequencies_of_interest
            first = np.searchsorted(self.frequencies, low, side="left")
            last = np.searchsorted(self.frequencies, high, side="right")
        coherency = self._coherency
        windows, _, channels, _ = coherency.shape
        result = np.full((windows, channels, channels), np.nan)
        row_step = min(channels, max(1, self.block_bytes // (windows * channels * 16)))
        frequency_step = max(1, self.block_bytes // (windows * row_step * channels * 16))
        eps = np.finfo(float).eps
        for row in range(0, channels, row_step):
            end = min(row + row_step, channels)
            total = np.zeros((windows, end-row, channels), complex)
            count = np.zeros(total.shape, dtype=np.int64)
            for start in range(first, last, frequency_step):
                stop = min(start + frequency_step, last)
                values = coherency[:, start:stop, row:end, :]
                magnitude = np.abs(values)
                valid = np.isfinite(magnitude) & (magnitude > 0)
                unit = np.zeros_like(values)
                # Componentwise real division avoids complex division's
                # overflowing reciprocal for finite subnormal magnitudes.
                np.divide(values.real, magnitude, out=unit.real, where=valid)
                np.divide(values.imag, magnitude, out=unit.imag, where=valid)
                total += unit.sum(axis=1)
                count += valid.sum(axis=1)
            # Forward rounding bound for normalizing and summing unit vectors.
            # This detects numerical cancellation, not statistical significance
            # or weak coherence; tiny nonzero coherency still has unit weight.
            error = (count + 2) * eps
            bound = count * error / (1 - error)
            angle = np.angle(total)
            angle[(count == 0) | (np.abs(total) <= bound)] = np.nan
            result[:, row:end] = angle
        return result

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
        if "sign" in kinds and "nonzero" not in kinds:
            # Preserve a compact count alongside PLI so its debiased variant
            # never needs another cross-product pass to identify zero signs.
            kinds = (*kinds, "nonzero")
        keys = [("phase_statistic", kind) for kind in kinds]
        def compute(missing):
            reduced = self._reduce_phase([key[1] for key in missing])
            return {key: reduced[key[1]] for key in missing}
        return {key[1]: value for key, value in self._arrays.get_many(keys, compute).items()}

    def _reduce_phase(self, kinds):
        """Form each bounded taper cross-product once for the requested moments."""
        coefficients = self._positive_coefficients
        w, r, k, f, p = coefficients.shape
        results = {kind: np.empty((w, f, p, p), dtype=(
                       complex if kind == "unit" else np.min_scalar_type(r*k)
                       if kind == "nonzero" else float))
                   for kind in kinds}
        # Every moment has known conjugate/signed/symmetric pair reversal.
        # Tile the upper triangle and fill its reverse instead of evaluating
        # both cross-products. Small square tiles retain bounded vectorized work.
        row_step = min(p, 32, max(1, self.block_bytes // (w * r * k * p * 16)))
        frequency_step = max(1, self.block_bytes // (w * r * k * row_step * p * 16))
        for start in range(0, f, frequency_step):
            stop = min(start + frequency_step, f)
            for row in range(0, p, row_step):
                end = min(row + row_step, p)
                right = coefficients[..., start:stop, None, row:].conj()
                left = coefficients[..., start:stop, row:end, None]
                cross = np.matmul(left, right, dtype=np.complex128)
                if "imag_pairs" in kinds or "abs_imag_pairs" in kinds:
                    # Sum distinct observation products using prefix sums, not
                    # sum(y)**2 - sum(y**2), which can lose the entire denominator
                    # when one observation dominates. Normalize per pair/frequency
                    # before multiplication to avoid squaring tiny/large amplitudes.
                    y = cross.imag.reshape(w, r*k, stop-start, end-row, p-row)
                    magnitude = np.abs(y)
                    scale = magnitude.max(axis=1, keepdims=True)
                    if "imag_pairs" in kinds:
                        normalized = np.divide(y, scale, out=np.zeros_like(y), where=scale != 0)
                        np.abs(normalized, out=magnitude)
                        prefix = np.cumsum(normalized[:, :-1], axis=1)
                        results["imag_pairs"][:, start:stop, row:end, row:] = np.einsum(
                            "wkfij,wkfij->wfij", normalized[:, 1:], prefix, optimize=False)
                        del normalized, prefix
                    else:
                        np.divide(magnitude, scale, out=magnitude, where=scale != 0)
                    if "abs_imag_pairs" in kinds:
                        prefix = np.cumsum(magnitude[:, :-1], axis=1)
                        results["abs_imag_pairs"][:, start:stop, row:end, row:] = np.einsum(
                            "wkfij,wkfij->wfij", magnitude[:, 1:], prefix, optimize=False)
                        del prefix
                    del y, magnitude, scale
                for kind in kinds:
                    if kind in ("imag_pairs", "abs_imag_pairs"):
                        continue
                    if kind == "nonzero":
                        if "sign" not in kinds:
                            results[kind][:, start:stop, row:end, row:] = np.count_nonzero(
                                cross.imag, axis=(1, 2))
                        continue
                    if kind == "unit":
                        values = cross / np.abs(cross)
                    elif kind == "sign":
                        values = np.sign(cross.imag)
                        if "nonzero" in kinds:
                            count = np.einsum("wrkfij,wrkfij->wfij", values, values, optimize=False)
                            # A NaN sign will propagate through the sign average;
                            # its count is immaterial but must fit the integer cache.
                            np.copyto(count, 0, where=np.isnan(count))
                            results["nonzero"][:, start:stop, row:end, row:] = count
                            del count
                    elif kind == "imag":
                        values = cross.imag
                    elif kind == "abs_imag":
                        values = np.abs(cross.imag)
                    else:
                        raise ValueError(f"Unknown phase statistic: {kind}")
                    results[kind][:, start:stop, row:end, row:] = values.mean(axis=(1, 2))
                    del values
                for kind, result in results.items():
                    reverse = result[:, start:stop, row:end, end:].swapaxes(-1, -2)
                    if kind == "unit":
                        reverse = reverse.conj()
                    elif kind in ("sign", "imag"):
                        reverse = -reverse
                    result[:, start:stop, end:, row:end] = reverse
        return results

    def _phase_locking_value(self):
        return self._phase_statistic("unit")

    def phase_lag_index(self):
        return np.abs(self.signed_phase_lag_index())

    def signed_phase_lag_index(self):
        return self._phase_statistic("sign")

    def debiased_squared_phase_lag_index(self):
        # Sum over distinct observation pairs without forming those pairs:
        # ((sum(sign(y)))**2 - sum(sign(y)**2)) / (n*(n-1)).
        # sign(0)**2 is zero, so subtract the actual nonzero count, not n.
        n = self.n_observations
        moments = self._phase_statistics("sign", "nonzero")
        result = np.square(moments["sign"])
        result *= n*n
        result -= moments["nonzero"]
        result /= n*(n-1)
        return result

    def weighted_phase_lag_index(self):
        return np.abs(self.measure("signed_weighted_phase_lag_index"))

    def signed_weighted_phase_lag_index(self):
        moments = self._phase_statistics("imag", "abs_imag")
        weights = moments["abs_imag"]
        return np.divide(moments["imag"], weights, out=np.full_like(weights, np.nan),
                         where=weights > 0)

    def debiased_squared_weighted_phase_lag_index(self):
        moments = self._phase_statistics("imag_pairs", "abs_imag_pairs")
        weights = moments["abs_imag_pairs"]
        return np.divide(moments["imag_pairs"], weights,
                         out=np.full_like(weights, np.nan), where=weights > 0)

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
