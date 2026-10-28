"""Explicit output conversion; estimator arithmetic stays unchanged."""

import numpy as np

from ._config import WorkflowError


DTYPES = ("float16", "float32", "float64")


def dtype_name(value, where):
    if not isinstance(value, str) or value not in DTYPES:
        raise WorkflowError(f"{where} must be one of {', '.join(DTYPES)}")
    return value


def resolve_dtypes(cfg, identifiers):
    unknown = cfg.output_dtype_overrides.keys() - set(identifiers)
    if unknown:
        raise WorkflowError(f"output_dtype_overrides contains unselected metrics: {sorted(unknown)}")
    return {name: np.dtype(cfg.output_dtype_overrides.get(name, cfg.output_dtype)).str
            for name in identifiers}


def convert_output(values, dtype, identifier):
    """Round once and return a diagnostic for finite values becoming infinity.

    Underflow is permitted: negligible values can round to zero. The precision
    audit quantifies this separately from estimator-invalid or nonfinite values.
    Existing NaNs/infinities are not conversion overflows. Diagnostic positions
    refer to the input array; the writer maps them onto the saved channel axes.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = values.astype(dtype, copy=False)
    overflow = np.isfinite(values) & ~np.isfinite(converted)
    diagnostic = None
    if overflow.any():
        diagnostic = {"kind": "finite_overflow", "metric": identifier,
                      "source_dtype": values.dtype.name, "output_dtype": converted.dtype.name,
                      "count": int(overflow.sum()),
                      "positive_count": int(np.count_nonzero(converted[overflow] > 0)),
                      "negative_count": int(np.count_nonzero(converted[overflow] < 0)),
                      "original_values": values[overflow].astype(np.float64).tolist(),
                      "positions": np.argwhere(overflow).tolist()}
    return converted, diagnostic
