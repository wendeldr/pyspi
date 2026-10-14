"""Optional analysis helpers, separate from storing and reconstructing results."""

import warnings

import numpy as np


MEAN_BLOCK_COLUMNS = 8192


def _rows_nanmean(payload):
    out = np.empty(payload.shape[1], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, payload.shape[1], MEAN_BLOCK_COLUMNS):
            block = np.ascontiguousarray(payload[:, start:start + MEAN_BLOCK_COLUMNS].T)
            out[start:start + MEAN_BLOCK_COLUMNS] = np.nanmean(block, axis=1, dtype=np.float64)
    return out


def time_mean(results, metric):
    """Float64 NaN-ignoring mean over saved windows of one metric.

    ``results`` is a ``pyspi.storage.Results`` reader. Complete readable input
    is required; unavailable data warns and raises StorageIntegrityError. This
    helper loads the packed payload, reducing in column blocks rather than
    reconstructing every window's full matrix. It is not a streaming reducer.
    Empty/all-NaN series return NaN; reduction RuntimeWarnings are suppressed.
    For float32 inputs this explicitly uses float64 accumulation.
    """
    from .storage._packing import UINT, _negate_bits

    meta, arrays = results._arrays(metric)
    n, mode, dtype = meta["n"], meta["mode"], np.dtype(meta["dtype"])
    if mode == "full":
        return _rows_nanmean(arrays["full"]).reshape(n, n)
    iu, js = np.triu_indices(n), np.triu_indices(n, 1)
    out = np.empty((n, n), dtype=np.float64)
    out[iu] = _rows_nanmean(arrays["upper"])
    if mode == "sym":
        out[js[1], js[0]] = out[js]
    else:
        # Reconstruct each lower series before averaging. Negating the reduced
        # mean instead gives the wrong zero sign after cancellation (review R3).
        columns = np.flatnonzero(iu[0] != iu[1])
        lower = np.empty(len(columns), dtype=np.float64)
        for start in range(0, len(columns), MEAN_BLOCK_COLUMNS):
            strict = np.ascontiguousarray(arrays["upper"][:, columns[start:start + MEAN_BLOCK_COLUMNS]])
            bits = (_negate_bits(strict, dtype) if mode.startswith("anti")
                    else strict.view(UINT[dtype]).copy())
            if mode.endswith("_res"):
                bits ^= arrays["residual"][:, start:start + MEAN_BLOCK_COLUMNS]
            lower[start:start + MEAN_BLOCK_COLUMNS] = _rows_nanmean(bits.view(dtype))
        out[js[1], js[0]] = lower
    return out
