"""Batched single-predictor regressions; row = predictor, column = target."""

import numpy as np

from pyspi._moments import moment_workspace


def regression_result(data, model):
    work = moment_workspace(data)

    def compute():
        if model == "Ridge":
            _, xx = work.centered()
            coefficient = work.cross() / (xx[:, None] + 1.)
            return work.mse(coefficient)
        if model == "RECI":
            _, xx = work.centered("cube_minmax")
            xy = work.cross("cube_minmax", "minmax")
            coefficient = np.divide(xy, xx[:, None], out=np.zeros_like(xy), where=xx[:, None] > 0)
            return work.mse(coefficient, "cube_minmax", "minmax")
        raise ValueError(f"Unsupported batched regression: {model}")

    # Returning a copy prevents callers from mutating cached estimates.
    return work.arrays.get(("regression", model), compute).copy()
