"""Euclidean barycenter summaries without storing a time series per pair."""

import numpy as np

from pyspi._moments import moment_workspace


def euclidean_summary(data, statistic, squared):
    work = moment_workspace(data)
    source = work.source
    n_channels, n = source.shape

    def means():
        mean = np.sum(source/n, axis=1)
        ordinary = (mean[:, None]+mean[None, :])*.5
        _, ss = work.centered()
        variance = (ss[:, None]+ss[None, :]+2*work.cross())/(4*n)
        # Nearly opposite channels can cancel all their centered energy.
        # Compute those means of squares directly, in bounded blocks.
        near = variance <= 1e-10*(ss[:, None]+ss[None, :])/(4*n)
        np.fill_diagonal(near, False)
        i, j = np.nonzero(near)
        step = max(1, work.block_bytes//(3*n*8))
        square = variance+ordinary**2
        for start in range(0, len(i), step):
            a, b = i[start:start+step], j[start:start+step]
            barycenter = source[a]*.5+source[b]*.5
            square[a, b] = np.mean(barycenter**2, axis=1)
        for matrix in (ordinary, square):
            np.fill_diagonal(matrix, np.nan)
        return ordinary, square

    def extrema():
        result = tuple(np.full((n_channels, n_channels), np.nan) for _ in range(4))
        i, j = np.triu_indices(n_channels, 1)
        step = max(1, min(128, work.block_bytes//(3*n*8)))
        for start in range(0, len(i), step):
            a, b = i[start:start+step], j[start:start+step]
            barycenter = source[a]*.5+source[b]*.5
            squared_values = barycenter**2
            for matrix, values in zip(result, (np.max(barycenter, axis=1), np.max(squared_values, axis=1),
                                              np.argmax(barycenter, axis=1), np.argmax(squared_values, axis=1))):
                matrix[a, b] = matrix[b, a] = values
        return result

    if statistic == "mean":
        result = work.arrays.get(("euclidean_barycenter", "means"), means)[int(squared)]
    else:
        result = work.arrays.get(("euclidean_barycenter", "extrema"), extrema)[
            int(squared) + (2 if statistic == "max_time" else 0)]
    result = result.copy()
    invalid = ~np.isfinite(source).all(axis=1)
    result[invalid[:, None] | invalid[None, :]] = np.nan
    result[~np.isfinite(result)] = np.nan
    return result
