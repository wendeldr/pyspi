"""Pairwise shape dissimilarity between independent-range amplitude histograms."""

import numpy as np
from scipy.special import logsumexp

from pyspi._moments import moment_workspace


def histogram_relative_entropy(data, bins, pseudocount):
    work = moment_workspace(data)

    def compute():
        source = work.source
        finite = np.isfinite(source).all(axis=1)
        varying = np.any(source != source[:, :1], axis=1)
        valid = finite & varying
        p = len(source)
        result = np.full((p, p), np.nan)
        reasons = np.full((p, p), "estimated", dtype="U20")
        reasons[~varying[:, None] | ~varying[None, :]] = "constant_input"
        reasons[~finite[:, None] | ~finite[None, :]] = "nonfinite_input"
        if valid.any():
            # Positive affine scaling maps each independent range to [0, 1].
            # Explicit bins make this valid even for extreme voltage units.
            x = work.values("minmax")[valid]
            counts = np.stack([np.histogram(row, bins=bins, range=(0., 1.))[0] for row in x])
            # Log arithmetic keeps empty-bin probabilities well defined for
            # every finite positive pseudocount, even at subnormal scale.
            with np.errstate(divide="ignore"):
                logs = np.logaddexp(np.log(counts.astype(float)), np.log(pseudocount))
            logs -= logsumexp(logs, axis=1, keepdims=True)
            probability = np.exp(logs)
            kl = np.sum(probability * logs, axis=1)[:, None] - probability @ logs.T
            result[np.ix_(valid, valid)] = np.maximum(np.maximum(kl, kl.T), 0.)
        np.fill_diagonal(result, np.nan)
        np.fill_diagonal(reasons, "diagonal")
        return result, reasons

    result, reason = work.arrays.get(("histogram_relative_entropy", bins, pseudocount), compute)
    return result.copy(), reason.copy()
