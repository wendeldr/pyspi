"""Shared CPU spectral work, independent of SPI band summaries.

PSI implements Nolte et al. (2008), equation 3, on adjacent FFT bins in an
inclusive band. Fourier estimators use one-sided grids including DC and
even-length Nyquist. Directed calculations use the native unpadded grid;
group delay derives its regression spacing from the transform's spectral
resolution. Two-sided coefficients remain available for Wilson factorization.
"""

from collections import OrderedDict
from functools import cached_property
from itertools import combinations

import numpy as np
from scipy.fft import fft, ifft
from pyspi import _directed
from pyspi._cache import ArrayCache, cached_workspace
from pyspi._group_delay import estimate as estimate_group_delay
from pyspi._spectral_compat import spectral_connectivity as sc
from pyspi._wilson import REASONS as FIT_REASONS, adjoint, factorize, inverse


DIRECTED_MEASURES = ("directed_coherence", "directed_transfer_function",
                     "direct_directed_transfer_function", "pairwise_spectral_granger_prediction")


#: The bounded taper moments each phase-synchrony measure derives from, by the
#: pyspi measure name. Keys absent here (coherency-based measures) need none.
PHASE_MOMENTS = {
    "phase_locking_value": ("unit",),
    "pairwise_phase_consistency": ("unit",),
    "signed_phase_lag_index": ("sign", "nonzero"),
    "debiased_squared_phase_lag_index": ("sign", "nonzero"),
    "signed_weighted_phase_lag_index": ("imag", "abs_imag"),
    "debiased_squared_weighted_phase_lag_index": ("imag_pairs", "abs_imag_pairs"),
}
PHASE_MOMENT_ORDER = ("unit", "sign", "nonzero", "imag", "abs_imag", "imag_pairs", "abs_imag_pairs")
# A representative full-resolution measure result retained per moment family.
PHASE_MOMENTS_MEASURE = {kind: measure for measure, kinds in PHASE_MOMENTS.items() for kind in kinds}


def canonical_directed_measure(name):
    # Off-diagonal equalities for pairwise, not conditional multichannel, fits.
    return {"partial_directed_coherence": "directed_transfer_function",
            "generalized_partial_directed_coherence": "directed_coherence"}.get(name, name)


class SpectralConnectivity(sc.Connectivity):
    """One transform, shared powers/coherency, and bounded taper products.

    Only the trials+tapers expectation used by pyspi is supported here.
    Directed formulas reuse cached factorization properties within each pair batch.
    """

    # This bounds each temporary complex outer-product block, independently of
    # channel count. Real transforms of a block can require extra temporary space.
    block_bytes = 16 * 2**20
    max_pairs = 16

    def __init__(self, fourier_coefficients, frequencies, time, *, frequency_resolution=None,
                 cache_bytes=256 * 2**20):
        super().__init__(fourier_coefficients, frequencies=frequencies, time=time)
        self.frequency_resolution = frequency_resolution
        self._arrays = ArrayCache(max_bytes=cache_bytes)
        self._pairs = OrderedDict()
        self.directed_requests = set()
        self.phase_requests = set()
        self._fit_diagnostics = None
        self._native_directed = None

    def phase_cache_required_bytes(self):
        """Bytes for registered moments and derived phase results.

        Excludes FFT arrays, other cached measures and temporary arrays; this
        decides whether phase fusion fits, not total process memory.
        """
        if not self.phase_requests:
            return 0
        w, r, k, f, p = self._positive_coefficients.shape
        element = w * f * p * p
        moments = sum(element * (16 if kind == "unit" else 8 if kind != "nonzero"
                                 else np.dtype(np.min_scalar_type(r*k)).itemsize)
                      for kind in set(self.phase_requests))
        measures = len({PHASE_MOMENTS_MEASURE[kind] for kind in self.phase_requests}) * element * 8
        return moments + measures

    @cached_property
    def frequencies(self):
        """DC through Nyquist, or the last positive bin for an odd FFT."""
        count = self.fourier_coefficients.shape[-2] // 2 + 1
        return np.abs(self.all_frequencies[:count])

    @cached_property
    def _power(self):
        return super()._power

    @cached_property
    def _factorization(self):
        return factorize(self._expectation_cross_spectral_matrix())

    @cached_property
    def _minimum_phase_factor(self):
        return self._factorization.factor

    @cached_property
    def _factor_zero(self):
        return np.fft.ifft(self._minimum_phase_factor, axis=-3)[..., 0, :, :].real

    @cached_property
    def _transfer_function(self):
        return (self._minimum_phase_factor[..., :self.frequencies.size, :, :]
                @ inverse(self._factor_zero)[..., None, :, :])

    @cached_property
    def _noise_covariance(self):
        return self._factor_zero @ adjoint(self._factor_zero)

    @cached_property
    def _MVAR_Fourier_coefficients(self):
        return inverse(self._transfer_function)

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
        return _directed.directed_coherence(self._transfer_function, self._noise_covariance)

    def directed_transfer_function(self):
        return _directed.directed_transfer_function(self._transfer_function)

    def partial_directed_coherence(self):
        return _directed.partial_directed_coherence(self._transfer_function)

    def generalized_partial_directed_coherence(self):
        return _directed.generalized_partial_directed_coherence(self._transfer_function, self._noise_covariance)

    def direct_directed_transfer_function(self):
        return _directed.direct_directed_transfer_function(self._transfer_function, self._noise_covariance)

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
        """Signed delay/slope/r using per-pair zero-coherence significance tests.

        The spacing is ceil(transform bandwidth / FFT spacing), independent of
        the requested band width. Retain the largest significant cluster and
        require at least three spaced points. BH corrects each pair/window's
        frequencies separately; it does not control network-wide FDR.
        Nominal bandwidth spacing approximates independence; it is not a proof
        that neighboring spectral estimates are statistically independent.
        """
        return self.group_delay_details(frequencies_of_interest, significance_threshold)[:3]

    def group_delay_details(self, frequencies_of_interest=None, significance_threshold=0.05):
        band = None if frequencies_of_interest is None else tuple(frequencies_of_interest)
        key = ("group_delay", band, significance_threshold)
        return self._arrays.get(key, lambda: self._group_delay_details(band, significance_threshold))

    def _group_delay_details(self, band, alpha):
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("Group delay significance threshold must be between zero and one.")
        frequencies = self.frequencies
        if frequencies.size < 2:
            raise ValueError("Group delay requires at least two transform frequencies.")
        resolution = self.frequency_resolution
        if resolution is None or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Group delay requires the transform's positive spectral resolution.")
        ratio = resolution / (frequencies[1] - frequencies[0])
        # Unit conversion can put an integer ratio a few ulps above itself.
        nearest = np.rint(ratio)
        if np.isclose(ratio, nearest, rtol=8 * np.finfo(float).eps, atol=0):
            ratio = nearest
        step = max(1, int(np.ceil(ratio)))
        first = 0 if band is None else np.searchsorted(frequencies, band[0], side="left")
        last = frequencies.size if band is None else np.searchsorted(frequencies, band[1], side="right")
        selected = slice(first, last)
        frequencies = frequencies[selected]
        if frequencies[::step].size < 3:
            raise ValueError(
                "Group delay requires at least three frequency bins spaced by the "
                "transform's spectral resolution in the inclusive band."
            )
        nfft = self.fourier_coefficients.shape[-2]
        bins = np.arange(first, last)
        real_bins = (bins == 0) | ((nfft % 2 == 0) & (bins == nfft//2))
        return estimate_group_delay(self.coherency()[:, selected], frequencies,
                                    self.n_observations, real_bins, step, alpha, self.block_bytes)

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
        # Registered moments not yet cached join the same pass: their
        # cross-products are the cost, and the extra reductions are cheap.
        fuse = self.phase_cache_required_bytes() <= self._arrays.max_bytes
        wanted = list(kinds) + [kind for kind in PHASE_MOMENT_ORDER
                                if fuse and kind in self.phase_requests and kind not in kinds
                                and ("phase_statistic", kind) not in self._arrays.entries]
        keys = [("phase_statistic", kind) for kind in wanted]
        def compute(missing):
            reduced = self._reduce_phase([key[1] for key in missing])
            return {key: reduced[key[1]] for key in missing}
        values = self._arrays.get_many(keys, compute)
        return {kind: values[("phase_statistic", kind)] for kind in kinds}

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
        if name == "group_delay":
            # Fits and diagnostic arrays share one bounded cache entry.
            return self.group_delay(**kwargs)
        if name == "signed_phase_lag_index":
            # Already cached as the sign moment; do not count the same array
            # twice against the cache's byte budget.
            return self.signed_phase_lag_index()
        key = ("measure", name, tuple(sorted(kwargs.items())))
        return self._arrays.get(key, lambda: getattr(self, name)(**kwargs))

    def pair(self, i, j):
        """One model per unique pair, in ascending channel-index order."""
        key = tuple(sorted((i, j)))
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
        return self.pairwise_measure("pairwise_spectral_granger_prediction")

    def pairwise_measure(self, name):
        """Retain a measure's spectrum across bands even when pair models evict."""
        name = canonical_directed_measure(name)
        return self._arrays.get(("directed_pairs", name), lambda: self._pairwise_measure(name))

    def _pairwise_measure(self, name):
        w, _, _, _, p = self.fourier_coefficients.shape
        result = np.full((w, self.frequencies.size, p, p), np.nan)
        for pairs, model in self._directed_blocks():
            values = self._directed_values(model, name).reshape(w, len(pairs), -1, 2)
            i, j = pairs.T
            result[..., i, j] = values[..., 0].transpose(0, 2, 1)
            result[..., j, i] = values[..., 1].transpose(0, 2, 1)
        return result

    def _directed_blocks(self):
        """Fit bounded batches of unique pairs, recording pair/window diagnostics."""
        coefficients = self.fourier_coefficients
        w, r, k, n, p = coefficients.shape
        pairs = np.array(list(combinations(range(p), 2)), dtype=int).reshape(-1, 2)
        # Bound each coefficient/CSM temporary. Iteration has several such arrays;
        # block_bytes is not a total-process memory limit.
        step = max(1, min(64, self.block_bytes//(w*n*max(2*r*k, 4)*16)))
        reason = np.full((w, p, p), len(FIT_REASONS), dtype=np.uint8)
        residual = np.full((w, p, p), np.nan)
        iterations = np.zeros((w, p, p), dtype=np.int32)
        for start in range(0, len(pairs), step):
            block = pairs[start:start+step]
            selected = coefficients[..., block].transpose(0, 4, 1, 2, 3, 5)
            selected = selected.reshape(w*len(block), r, k, n, 2)
            model = SpectralConnectivity(selected, self.all_frequencies,
                                         np.arange(w*len(block)),
                                         frequency_resolution=self.frequency_resolution)
            fit = model._factorization
            i, j = block.T
            reason[:, i, j] = reason[:, j, i] = fit.reason.reshape(w, -1)
            residual[:, i, j] = residual[:, j, i] = fit.residual.reshape(w, -1)
            iterations[:, i, j] = iterations[:, j, i] = fit.iterations.reshape(w, -1)
            yield block, model
        self._fit_diagnostics = (reason, residual, iterations)

    @staticmethod
    def _directed_values(model, name):
        if name == "pairwise_spectral_granger_prediction":
            values = _directed.pairwise_granger(model._transfer_function, model._noise_covariance)
        else:
            values = getattr(model, name)()
        # Last axis is direction: ascending pair's first->second, second->first.
        return values[..., [1, 0], [0, 1]]

    def directed_summary(self, name, band):
        """Mean, maximum and unavailable fraction from one shared pair-fit pass.

        Calculator registers all requested bands/measures before evaluation,
        enabling streaming fits even when full spectra exceed the cache budget.
        Standalone calls share all four algebraic families for their requested
        band. Cached results are small band matrices, not all pair models.
        """
        name = canonical_directed_measure(name)
        band = tuple(band)
        key = ("directed_summary", name, band)
        if key in self._arrays.entries:
            return self._arrays.get(key, None)
        requests = self.directed_requests or {(kind, band) for kind in DIRECTED_MEASURES}
        requests = requests | {(name, band)}
        keys = [("directed_summary", kind, bounds) for kind, bounds in sorted(requests)]
        return self._arrays.get_many(keys, self._directed_summaries)[key]

    def _directed_summaries(self, missing):
        w, _, _, _, p = self.fourier_coefficients.shape
        output = {key: (np.full((w, p, p), np.nan), np.full((w, p, p), np.nan),
                        np.ones((w, p, p))) for key in missing}
        selections = {}
        for key in missing:
            low, high = key[2]
            first = np.searchsorted(self.frequencies, low, side="left")
            last = np.searchsorted(self.frequencies, high, side="right")
            if last > first:
                selections[key] = slice(first, last)
        if not selections:
            return output
        names = sorted({key[1] for key in selections})
        for pairs, model in self._directed_blocks():
            i, j = pairs.T
            for name in names:
                values = self._directed_values(model, name).reshape(w, len(pairs), -1, 2)
                for key, selected in selections.items():
                    if key[1] != name:
                        continue
                    band = values[:, :, selected]
                    finite = np.isfinite(band)
                    count = finite.sum(axis=2)
                    total = np.where(finite, band, 0).sum(axis=2)
                    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
                    maximum = np.where(finite, band, -np.inf).max(axis=2)
                    maximum[count == 0] = np.nan
                    fraction = 1-count/band.shape[2]
                    for matrix, reduced in zip(output[key], (mean, maximum, fraction)):
                        matrix[:, i, j], matrix[:, j, i] = reduced[..., 0], reduced[..., 1]
        return output

    def directed_diagnostics(self, name, band):
        _, _, fraction = self.directed_summary(name, band)
        empty_band = not ((self.frequencies >= band[0]) & (self.frequencies <= band[1])).any()
        if empty_band or self._fit_diagnostics is None:
            # An empty requested band needs no fit and has no frequency estimate.
            reason = np.full(fraction.shape, "empty_frequency_band", dtype="U32")
            diagonal = np.arange(reason.shape[-1])
            reason[..., diagonal, diagonal] = "diagonal"
            return dict(reason=reason, reconstruction_error=np.full(fraction.shape, np.nan),
                        iterations=np.zeros(fraction.shape, dtype=int), unavailable_fraction=fraction.copy())
        codes, residual, iterations = self._fit_diagnostics
        reasons = np.append(FIT_REASONS, "diagonal")[codes].astype("U32")
        reasons[(codes == 0) & (fraction > 0)] = "undefined_frequency_values"
        return dict(reason=reasons.copy(), reconstruction_error=residual.copy(),
                    iterations=iterations.copy(), unavailable_fraction=fraction.copy())


class _DataSpectralCache:
    def __init__(self, source, max_bytes=256 * 2**20):
        self.source = source
        self.workspaces = OrderedDict()
        self.max_bytes = max_bytes
        self.parametric = ArrayCache(max_bytes=max_bytes)
        self.directed_requests = {}
        self.phase_requests = {}


def data_spectral_cache(data):
    budget = getattr(data, "_spectral_cache_bytes", 256 * 2**20)
    cache = cached_workspace(data, "spectral", lambda source: _DataSpectralCache(source, budget))
    if cache.max_bytes != budget:
        # Release old large workspaces when the caller lowers the budget.
        cache.workspaces.clear()
        cache.parametric = ArrayCache(max_bytes=budget)
        cache.max_bytes = budget
    return cache


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
            cache_bytes=cache.max_bytes,
        )
        if len(cache.workspaces) > 2:
            cache.workspaces.popitem(last=False)
    cache.workspaces.move_to_end(settings)
    workspace = cache.workspaces[settings]
    workspace.directed_requests = cache.directed_requests.get(sampling_frequency, set())
    workspace.phase_requests = cache.phase_requests.get(sampling_frequency, set())
    return workspace


def directed_workspace(data, sampling_frequency):
    """Evaluate directed fits on the native observation grid, reusing tapers.

    The dependency may pad other Fourier measures to a fast FFT length. Recover
    the already tapered time samples and transform at native length, only when
    needed. This avoids silently making dDTF scale depend on FFT-speed padding
    while preserving other measures' previously reviewed frequency grids.
    """
    parent = spectral_workspace(data, sampling_frequency)
    n = data.n_observations
    if parent.fourier_coefficients.shape[-2] == n:
        return parent
    if parent._native_directed is None:
        samples = ifft(parent.fourier_coefficients, axis=-2)[..., :n, :].real
        coefficients = fft(samples, axis=-2)
        parent._native_directed = SpectralConnectivity(
            coefficients, np.fft.fftfreq(n, 1/sampling_frequency), parent.time,
            frequency_resolution=parent.frequency_resolution, cache_bytes=parent._arrays.max_bytes)
    parent._native_directed.directed_requests = parent.directed_requests
    return parent._native_directed


def frequency_mask(frequencies, fmin, fmax):
    """Select both endpoints without interpolating or widening the band."""
    return (frequencies >= fmin) & (frequencies <= fmax)


def validate_frequencies(fs, fmin, fmax):
    """All frequencies share the units of fs (Hz when fs is samples/second)."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be finite and positive.")
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 0 <= fmin <= fmax <= fs / 2):
        raise ValueError("Frequency band must satisfy 0 <= fmin <= fmax <= fs/2.")
