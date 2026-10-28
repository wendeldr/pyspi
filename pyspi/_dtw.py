"""Exact Sakoe–Chiba DTW with rolling bands and one evaluation per pair."""

import numpy as np
from numba import njit

from pyspi._moments import moment_workspace


@njit(cache=True)
def _banded_pair(x, y, radius):
    n = len(x)
    width = 2 * radius + 1
    previous = np.full(width, np.inf)
    current = np.full(width, np.inf)
    # Index of column zero in the fictitious row zero (column-row+radius).
    previous[radius] = 0.
    for i in range(1, n + 1):
        current[:] = np.inf
        low = max(1, i - radius)
        high = min(n, i + radius)
        for j in range(low, high + 1):
            k = j - i + radius
            above = previous[k + 1] if k + 1 < width else np.inf
            left = current[k - 1] if k > 0 else np.inf
            current[k] = (x[i - 1] - y[j - 1])**2 + min(above, left, previous[k])
        previous, current = current, previous
    return np.sqrt(previous[radius])


@njit(cache=True)
def _banded_matrix(x, radius):
    p, n = x.shape
    result = np.full((p, p), np.nan)
    for i in range(p):
        if not np.isfinite(x[i]).all():
            continue
        for j in range(i + 1, p):
            if not np.isfinite(x[j]).all():
                continue
            scale = max(np.max(np.abs(x[i])), np.max(np.abs(x[j])))
            # Ordinary data retain the reference arithmetic. Only extreme units
            # need rescaling to keep the squared sample costs representable.
            if scale > 1e150 or (0 < scale < 1e-150):
                value = _banded_pair(x[i] / scale, x[j] / scale, radius) * scale
            else:
                value = _banded_pair(x[i], x[j], radius)
            result[i, j] = result[j, i] = value
    return result


def banded_dtw(data, radius):
    work = moment_workspace(data)
    # Wider than the recording is equivalent to the full admissible grid, but
    # allocation is still bounded by the actual observation count.
    effective = min(radius, work.source.shape[1] - 1)
    return work.arrays.get(("dtw_sakoe_chiba", effective),
                           lambda: _banded_matrix(np.ascontiguousarray(work.source), effective)).copy()
