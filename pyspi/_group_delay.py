"""Group-delay selection under the independent, equal-weight observation model.

The zero-coherence null is exact for independent isotropic Gaussian Fourier
observations, and approximate for estimated multitaper spectra. BH is applied
across frequencies separately for each pair/window, not across the network.
"""

import numpy as np
from scipy.special import betaincc


REASONS = (
    "diagonal", "estimated", "no_significant_frequencies",
    "insufficient_frequency_span", "invalid_coherency", "numerical_failure",
)


def coherence_pvalues(coherency, observations, real_bins):
    """Return p-values and invalid-bin flags; frequency is the final axis.

    Complex bins: |C|² ~ Beta(1, K-1). DC/even-FFT Nyquist are real:
    |C|² ~ Beta(1/2, (K-1)/2), the squared cosine of two independent real
    Gaussian K-vectors. These observations are not centered across tapers.
    Undefined bins count as unrejected tests in the original band family.
    """
    if observations < 2:
        raise ValueError("Group delay significance requires at least two Fourier observations.")
    magnitude = np.abs(coherency)
    # Dot products and normalization can exceed unity by rounding error.
    tolerance = 16 * observations * np.finfo(float).eps
    invalid = ~np.isfinite(magnitude) | (magnitude > 1 + tolerance)
    squared = np.minimum(magnitude, 1)**2
    with np.errstate(invalid="ignore"):
        p = (1 - squared)**(observations - 1)
        p[..., real_bins] = betaincc(.5, (observations - 1)/2, squared[..., real_bins])
    p[invalid] = 1
    return p, invalid


def frequency_bh(p, alpha):
    """BH step-up for each independent family on the last axis."""
    ordered = np.sort(p, axis=-1)
    line = alpha * np.arange(1, p.shape[-1] + 1) / p.shape[-1]
    threshold = np.max(np.where(ordered <= line, ordered, -1), axis=-1, keepdims=True)
    return p <= threshold


def longest_run(significant):
    """First longest contiguous run, vectorized over pairs/windows."""
    shape = significant.shape[:-1]
    current = np.zeros(shape, dtype=np.intp)
    length = np.zeros_like(current)
    start = np.zeros_like(current)
    for f in range(significant.shape[-1]):
        current = np.where(significant[..., f], current + 1, 0)
        longer = current > length
        start = np.where(longer, f + 1 - current, start)
        np.maximum(length, current, out=length)
    return start, length


def regress_runs(pairs, frequencies, start, length, step):
    """Centered OLS with a free intercept, only within each selected run.

    Unwrap the complete contiguous run before spacing its regression points.
    Masking outside the run first prevents an unrelated NaN from poisoning
    the fit. Scaling frequency to the band width keeps the sums well behaved
    under changes of frequency units. No p-values for the regression are used.
    """
    bins = np.arange(len(frequencies))
    inside = (bins >= start[..., None]) & (bins < (start + length)[..., None])
    phase = np.unwrap(np.where(inside, np.angle(pairs), 0), axis=-1)
    keep = inside & ((bins - start[..., None]) % step == 0)
    count = keep.sum(axis=-1)
    width = frequencies[-1] - frequencies[0]
    x = (frequencies - frequencies[0]) / width
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xmean = np.sum(np.where(keep, x, 0), axis=-1) / count
        ymean = np.sum(np.where(keep, phase, 0), axis=-1) / count
        dx = np.where(keep, x - xmean[..., None], 0)
        dy = np.where(keep, phase - ymean[..., None], 0)
        xx = np.sum(dx*dx, axis=-1)
        yy = np.sum(dy*dy, axis=-1)
        xy = np.sum(dx*dy, axis=-1)
        slope = xy / xx / width
        rvalue = xy / np.sqrt(xx*yy)
    # A constant phase has a defined zero slope, but undefined correlation.
    return slope, np.clip(rvalue, -1, 1), count


def estimate(coherency, frequencies, observations, real_bins, step, alpha, block_bytes):
    """Return delay/slope/r plus symmetric reason, point-count, invalid-count arrays."""
    windows, nfreq, channels, _ = coherency.shape
    shape = (windows, channels, channels)
    slope = np.full(shape, np.nan)
    rvalue = np.full(shape, np.nan)
    reason = np.zeros(shape, dtype=np.uint8)
    points = np.zeros(shape, dtype=np.intp)
    invalid_count = np.zeros(shape, dtype=np.intp)
    rows, cols = np.triu_indices(channels, 1)
    size = max(1, block_bytes // (windows * nfreq * 16))
    for first in range(0, len(rows), size):
        i, j = rows[first:first+size], cols[first:first+size]
        pairs = np.ascontiguousarray(np.moveaxis(coherency[..., i, j], 1, -1))
        p, invalid = coherence_pvalues(pairs, observations, real_bins)
        significant = frequency_bh(p, alpha) & ~invalid
        start, length = longest_run(significant)
        used = (length + step - 1) // step
        codes = np.where(length == 0, 2, 3).astype(np.uint8)
        codes[invalid.all(axis=-1)] = 4
        valid = used >= 3
        fitted = np.full(valid.shape, np.nan)
        r = np.full_like(fitted, np.nan)
        if valid.any():
            fitted[valid], r[valid], _ = regress_runs(
                pairs[valid], frequencies, start[valid], length[valid], step)
            codes[valid] = np.where(np.isfinite(fitted[valid]), 1, 5)
            failed = codes == 5
            fitted[failed], r[failed] = np.nan, np.nan
        slope[..., i, j], slope[..., j, i] = fitted, -fitted
        rvalue[..., i, j], rvalue[..., j, i] = r, -r
        reason[..., i, j] = reason[..., j, i] = codes
        points[..., i, j] = points[..., j, i] = used
        invalid_count[..., i, j] = invalid_count[..., j, i] = invalid.sum(axis=-1)
    return slope/(2*np.pi), slope, rvalue, reason, points, invalid_count
