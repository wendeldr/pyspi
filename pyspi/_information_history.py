"""Raw aligned histories for the retained non-Gaussian information estimators."""

import numpy as np


def history(work, start, lags):
    lags = tuple(lags)
    p, t = work.source.shape
    if start >= t or start < max((0, *lags)):
        raise ValueError("The requested history leaves no aligned observations.")
    def compute():
        if not lags:
            return np.empty((p, t-start, 0))
        return np.stack([work.source[:, start-lag:t-lag] for lag in lags], axis=-1)
    return work.arrays.get(("information_history", start, lags), compute)
