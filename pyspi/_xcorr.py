"""Batched linear cross-correlations and approximate lag-selection inference.

The null assumes independent, covariance-stationary channels. For M overlapping
products at a lag, Bartlett's null variance numerator is
M + 2 sum_{k=1}^{M-1} (M-k) acf_x(k) acf_y(k). Divide by T**2 for our full-window
energy normalization. This finite-overlap extension follows by counting the
covariances of the overlapping products; it is not an exact finite-sample test.
See Cliff et al. (2021), Appendix D, doi:10.1103/PhysRevResearch.3.013145.
"""

from functools import cached_property

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.special import ndtr

from pyspi._cache import ArrayCache


SUMMARY_KEYS = (("mean", False), ("max", False), ("min", False), ("peak", False),
                ("mean", True), ("max", True), ("min", True))


def bartlett_variance(acf_x, acf_y, overlaps, samples):
    """Plug-in null variance, using all autocovariance terms that overlap."""
    products = acf_x[..., 1:] * acf_y[..., 1:]
    sums = np.concatenate([np.zeros(products.shape[:-1] + (1,)), np.cumsum(products, axis=-1)], axis=-1)
    weighted = np.concatenate([np.zeros(products.shape[:-1] + (1,)),
                               np.cumsum(products * np.arange(1, samples), axis=-1)], axis=-1)
    return (overlaps + 2 * (overlaps * sums[..., overlaps - 1] - weighted[..., overlaps - 1])) / samples**2


def holm_mask(pvalues, alpha):
    """Holm's step-down correction, separately for each row/channel pair."""
    pvalues = np.where(np.isfinite(pvalues), pvalues, 1.)
    order = np.argsort(pvalues, axis=-1, kind="stable")
    sorted_p = np.take_along_axis(pvalues, order, axis=-1)
    accepted = np.logical_and.accumulate(
        sorted_p <= alpha / np.arange(pvalues.shape[-1], 0, -1), axis=-1)
    result = np.zeros_like(accepted)
    np.put_along_axis(result, order, accepted, axis=-1)
    return result


def summarize(curves, selected):
    """Select signed lags first, then square if requested, then aggregate."""
    count = selected.sum(axis=-1)
    signed = np.where(selected, curves, 0.)
    low = np.min(np.where(selected, curves, np.inf), axis=-1)
    high = np.max(np.where(selected, curves, -np.inf), axis=-1)
    squared = signed * signed
    mean = np.divide(signed.sum(axis=-1), count, out=np.full(count.shape, np.nan), where=count > 0)
    mean_sq = np.divide(squared.sum(axis=-1), count, out=np.full(count.shape, np.nan), where=count > 0)
    low_sq = np.min(np.where(selected, curves * curves, np.inf), axis=-1)
    high_sq = np.max(squared, axis=-1)
    for result in (low, high, low_sq, high_sq):
        result[count == 0] = np.nan
    peak = np.where(abs(low) > abs(high), low, high)
    # Choosing the first lag on an opposite-sign tie would depend on direction.
    ambiguous = ((low < 0) & (high > 0)
                 & np.isclose(-low, high, rtol=32*np.finfo(float).eps, atol=0))
    peak[ambiguous] = np.nan
    return np.stack([mean, high, low, peak, mean_sq, high_sq, low_sq]), count, ambiguous


class CrossCorrelationWorkspace:
    block_bytes = 16 * 2**20

    def __init__(self, source):
        if np.iscomplexobj(source):
            raise ValueError("Cross-correlation requires real-valued signals.")
        values = np.asarray(source[:, :, 0], dtype=np.float64)
        self.p, self.n = values.shape
        if self.n < 2:
            raise ValueError("Cross-correlation requires at least two observations.")
        # Scale before centring to avoid overflow/underflow in energy products.
        finite = np.isfinite(values).all(axis=1)
        values = np.where(finite[:, None], values, 0.)
        scale = np.max(abs(values), axis=1, keepdims=True)
        scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)
        centred = scaled - scaled.mean(axis=1, keepdims=True)
        energy = np.sqrt(np.sum(centred * centred, axis=1, keepdims=True))
        self.valid = finite & np.isfinite(energy[:, 0]) & (energy[:, 0] > 0)
        self.unit = np.divide(centred, energy, out=np.zeros_like(centred),
                              where=self.valid[:, None])
        self.pad = next_fast_len(2 * self.n - 1)
        self.arrays = ArrayCache()

    @cached_property
    def fourier(self):
        return rfft(self.unit, n=self.pad, axis=1)

    @cached_property
    def acf(self):
        return irfft(abs(self.fourier)**2, n=self.pad, axis=1)[:, :self.n]

    @cached_property
    def pairs(self):
        return np.triu_indices(self.p, 1)

    def _indices(self, lag):
        return np.r_[np.arange(self.pad - lag, self.pad), np.arange(lag + 1)]

    def _curves(self, i, j, lag):
        full = irfft(self.fourier[i] * self.fourier[j].conj(), n=self.pad, axis=-1)
        result = full[..., self._indices(lag)]
        result[~(self.valid[i] & self.valid[j])] = np.nan
        return result

    def blocks(self, lag):
        i, j = self.pairs
        # Fourier products, inverse output, selected lags, and inference arrays
        # are processed in bounded batches instead of a P x P x T tensor.
        step = max(1, self.block_bytes // (8 * self.pad * 12))
        key = ("curves", lag)
        if len(i) * (2*lag + 1) * 8 <= self.arrays.max_bytes:
            def collect():
                curves = np.empty((len(i), 2*lag + 1))
                for start in range(0, len(i), step):
                    part = slice(start, start + step)
                    curves[part] = self._curves(i[part], j[part], lag)
                return curves
            retained = self.arrays.get(key, collect)
        else:
            retained = None
        for start in range(0, len(i), step):
            part = slice(start, start + step)
            curves = self._curves(i[part], j[part], lag) if retained is None else retained[part]
            yield i[part], j[part], curves

    def _summaries(self, i, j, curves, lag, alpha):
        valid = self.valid[i] & self.valid[j]
        status = np.where(valid, "ok", "undefined_input")
        selected = np.isfinite(curves)
        if alpha is not None:
            overlap = self.n - abs(np.arange(-lag, lag + 1))
            variance = bartlett_variance(self.acf[i], self.acf[j], overlap, self.n)
            testable = (self.n >= 3) & np.isfinite(variance).all(axis=-1) & (variance > 0).all(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                pvalues = 2 * ndtr(-abs(curves) / np.sqrt(variance))
            selected &= holm_mask(pvalues, alpha) & testable[:, None]
            status = status.astype("U24")
            status[valid & ~testable] = "undefined_inference"
            status[valid & testable & ~selected.any(axis=-1)] = "no_significant_lags"
        summary, counts, ambiguous = summarize(curves, selected)
        return summary, counts, status, ambiguous

    def all_pairs(self, lag, alpha):
        def calculate():
            result = np.full((len(SUMMARY_KEYS), self.p, self.p), np.nan)
            counts = np.zeros((self.p, self.p), dtype=int)
            status = np.full((self.p, self.p), "diagonal", dtype="U24")
            ambiguous = np.zeros((self.p, self.p), dtype=bool)
            for i, j, curves in self.blocks(lag):
                values, number, reason, tied = self._summaries(i, j, curves, lag, alpha)
                result[:, i, j] = result[:, j, i] = values
                counts[i, j] = counts[j, i] = number
                status[i, j] = status[j, i] = reason
                ambiguous[i, j] = ambiguous[j, i] = tied
            return result, counts, status, ambiguous
        return self.arrays.get(("summaries", lag, alpha), calculate)

    def pair(self, i, j, lag, alpha):
        key = ("summaries", lag, alpha)
        if key in self.arrays.entries:
            values, counts, status, ambiguous = self.arrays.get(key, lambda: None)
            return values[:, i, j], counts[i, j], status[i, j], ambiguous[i, j]
        i, j = sorted((i, j))
        def calculate():
            curves = self._curves(np.array([i]), np.array([j]), lag)
            return self._summaries(np.array([i]), np.array([j]), curves, lag, alpha)
        values, counts, status, ambiguous = self.arrays.get(("pair", i, j, lag, alpha), calculate)
        return values[:, 0], counts[0], status[0], ambiguous[0]
