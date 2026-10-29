"""Shared independent-range histograms and marginal-shape divergences."""

import numpy as np
from numba import njit
from scipy.special import logsumexp

from pyspi._moments import moment_workspace


@njit(cache=True)
def _counts(values, edges):
    """Same half-open bins/final closed bin as histogram, including boundaries."""
    bins = len(edges) - 1
    counts = np.zeros((len(values), bins), dtype=np.float64)
    for i in range(len(values)):
        for value in values[i]:
            index = np.searchsorted(edges, value, side="right") - 1
            if index == bins:  # the final edge belongs to the final bin
                index -= 1
            counts[i, index] += 1
    return counts


@njit(cache=True)
def _jensen_shannon(probability, logs):
    p, bins = probability.shape
    result = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            value = 0.
            for k in range(bins):
                a, b = logs[i, k], logs[j, k]
                mixture = max(a, b) + np.log1p(np.exp(-abs(a-b))) - np.log(2.)
                value += .5 * (probability[i, k]*(a-mixture) + probability[j, k]*(b-mixture))
            result[i, j] = result[j, i] = max(0., value)
    return result


def _histograms(work, bins, pseudocount):
    def compute():
        source = work.source
        finite = np.isfinite(source).all(axis=1)
        varying = np.any(source != source[:, :1], axis=1)
        valid = finite & varying
        reasons = np.full((len(source), len(source)), "estimated", dtype="U20")
        reasons[~varying[:, None] | ~varying[None, :]] = "constant_input"
        reasons[~finite[:, None] | ~finite[None, :]] = "nonfinite_input"
        np.fill_diagonal(reasons, "diagonal")
        # Preserve the existing minmax transformation, including its rounding
        # at bin boundaries and protection against overflowing amplitude ranges.
        counts = _counts(work.values("minmax")[valid], np.linspace(0., 1., bins+1))
        # Log arithmetic supports even subnormal positive pseudocounts.
        with np.errstate(divide="ignore"):
            logs = np.logaddexp(np.log(counts), np.log(pseudocount))
        logs -= logsumexp(logs, axis=1, keepdims=True)
        return valid, np.exp(logs), logs, reasons
    return work.arrays.get(("histograms", bins, pseudocount), compute)


def histogram_divergence(data, bins, pseudocount, kind):
    work = moment_workspace(data)

    def compute():
        valid, probability, logs, reasons = _histograms(work, bins, pseudocount)
        result = np.full((len(valid), len(valid)), np.nan)
        if kind == "relative_entropy":
            kl = np.sum(probability * logs, axis=1)[:, None] - probability @ logs.T
            values = np.maximum(np.maximum(kl, kl.T), 0.)
        elif kind == "jensen_shannon":
            values = _jensen_shannon(probability, logs)
        else:
            raise ValueError(f"Unknown histogram divergence: {kind}")
        result[np.ix_(valid, valid)] = values
        np.fill_diagonal(result, np.nan)
        return result, reasons

    result, reason = work.arrays.get(("histogram_divergence", kind, bins, pseudocount), compute)
    return result.copy(), reason.copy()


def histogram_relative_entropy(data, bins, pseudocount):
    return histogram_divergence(data, bins, pseudocount, "relative_entropy")
