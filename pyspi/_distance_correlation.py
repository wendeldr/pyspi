"""Exact distance correlations with shared, bounded distance products.

Only one triangle of each sample-distance matrix is generated. A block contains
all channels but at most ``block_bytes`` of distances; no T-by-T matrix is
retained. Both centering conventions reuse the same raw inner products and row
sums. This is exact Euclidean distance correlation, not a random approximation.
"""

import numpy as np

from pyspi._moments import moment_workspace


def _sample_pairs(n, size):
    """Yield bounded arrays of strictly upper-triangular sample indices."""
    first = np.empty(size, dtype=np.intp)
    second = np.empty(size, dtype=np.intp)
    used = 0
    for i in range(n - 1):
        j = i + 1
        while j < n:
            take = min(size - used, n - j)
            first[used:used + take] = i
            second[used:used + take] = np.arange(j, j + take)
            used += take
            j += take
            if used == size:
                yield first, second
                used = 0
    if used:
        yield first[:used], second[:used]


def _resolve_rows(x, sums, biased, indices, block_bytes):
    """Recompute cancellation-prone rows using bounded centered-distance blocks."""
    p, n = x.shape
    row_means = sums / (n if biased else n - 2)
    grand = sums.sum(axis=1) / (n**2 if biased else (n - 1) * (n - 2))
    result = np.zeros((len(indices), p), dtype=np.longdouble)
    size = max(1, block_bytes // max(8 * p, 1))
    for i, j in _sample_pairs(n, size):
        block = abs(x[:, i] - x[:, j])
        block -= row_means[:, i]
        block -= row_means[:, j]
        block += grand[:, None]
        result += 2 * (block[indices] @ block.T)
    if biased:
        diagonal = -2 * row_means + grand[:, None]
        result += diagonal[indices] @ diagonal.T
    return result


def _products(x, block_bytes):
    p, n = x.shape
    order = np.argsort(x, axis=1)
    sorted_x = np.take_along_axis(x, order, axis=1)
    prefix = np.cumsum(sorted_x, axis=1)
    sorted_sums = ((2 * np.arange(n) - n + 2) * sorted_x
                   + prefix[:, -1, None] - 2 * prefix)
    sums = np.take_along_axis(sorted_sums, np.argsort(order, axis=1), axis=1)
    gram = np.zeros((p, p))
    size = max(1, block_bytes // max(8 * p, 1))
    for i, j in _sample_pairs(n, size):
        block = x[:, i] - x[:, j]
        np.abs(block, out=block)
        block = np.ascontiguousarray(block)
        gram += block @ block.T
    return gram * 2, sums


def distance_correlation(data, biased):
    work = moment_workspace(data)

    def compute():
        # Each channel's positive scaling cancels in the final coefficient.
        # Min-max scaling prevents overflow from arbitrary physical units.
        values = work.values("minmax")
        finite = np.isfinite(values).all(axis=1)
        x = np.where(finite[:, None], values, 0.)
        p, n = x.shape
        output = [np.full((p, p), np.nan) for _ in range(2)]
        if n < 2:
            return tuple(output)
        raw, sums = _products(x, work.block_bytes)
        # The final three-term subtraction is small P-by-P arithmetic. Extended
        # precision reduces cancellation without increasing the sample buffers.
        raw = raw.astype(np.longdouble)
        row_product = (sums @ sums.T).astype(np.longdouble)
        totals = sums.sum(axis=1).astype(np.longdouble)
        total_product = totals[:, None] * totals[None, :]
        for bias in (False, True):
            if not bias and n < 4:
                continue
            row_term = 2 * row_product / (n if bias else n - 2)
            total_term = total_product / (n**2 if bias else (n - 1) * (n - 2))
            centered = raw - row_term + total_term
            variance = np.diag(centered).copy()
            # Rare tied distributions can have exactly zero U-centered energy.
            # Resolve cancellation with centered distances rather than turning
            # a rounding residual into a normalized apparent relationship.
            error = 64 * n * np.finfo(float).eps * (abs(raw) + abs(row_term) + abs(total_term))
            constant = np.all(x == x[:, :1], axis=1)
            uncertain = (variance <= np.diag(error)) & ~constant
            if uncertain.any():
                indices = np.flatnonzero(uncertain)
                resolved = _resolve_rows(x, sums, bias, indices, work.block_bytes)
                centered[indices] = resolved
                centered[:, indices] = resolved.T
                variance[indices] = resolved[np.arange(len(indices)), indices]
            variance[constant] = 0.
            denom = np.sqrt(variance[:, None] * variance[None, :])
            result = np.divide(centered, denom, out=np.zeros_like(centered), where=denom > 0)
            if bias:
                result = np.sqrt(np.maximum(result, 0.))
            result = np.asarray(result, dtype=float)
            result[~finite[:, None] | ~finite[None, :]] = np.nan
            np.fill_diagonal(result, np.nan)
            output[int(bias)] = result
        return tuple(output)

    return work.arrays.get(("distance_correlation",), compute)[int(biased)].copy()
