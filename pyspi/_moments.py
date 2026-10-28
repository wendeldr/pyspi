"""Shared centered observations and cross-products for numerical estimators.

Keys include the transformation and observation interval: statistics computed
on different lag alignments must never silently share a full-window covariance.
The source is the existing mutation-checked Data snapshot. Retained derived
arrays share one bounded LRU; callers must not modify returned arrays.
"""

import numpy as np

from pyspi._cache import ArrayCache, cached_workspace


class MomentWorkspace:
    block_bytes = 16 * 2**20

    def __init__(self, source):
        if source.ndim != 3 or source.shape[2] != 1:
            raise ValueError("Shared numerical calculations require a single realisation.")
        if np.iscomplexobj(source):
            raise ValueError("Shared numerical calculations require real observations.")
        self.source = np.asarray(source[:, :, 0], dtype=np.float64)
        if not self.source.shape[1]:
            raise ValueError("At least one observation is required.")
        self.arrays = ArrayCache()

    def values(self, transform="identity"):
        if transform == "identity":
            return self.source
        if transform == "minmax":
            def compute():
                # Scaling first avoids overflow in max-min. It also avoids an
                # absolute epsilon threshold that would flatten small EEG units.
                scale = np.max(abs(self.source), axis=1, keepdims=True)
                scaled = np.divide(self.source, scale, out=np.zeros_like(self.source), where=scale > 0)
                low = scaled.min(axis=1, keepdims=True)
                span = scaled.max(axis=1, keepdims=True) - low
                result = np.divide(scaled-low, span, out=np.zeros_like(scaled), where=span > 0)
                result[~np.isfinite(self.source).all(axis=1)] = np.nan
                return result
        elif transform == "cube_minmax":
            def compute():
                return self.values("minmax") ** 3
        else:
            raise ValueError(f"Unknown observation transform: {transform}")
        return self.arrays.get(("values", transform), compute)

    def centered(self, transform="identity", start=0, stop=None):
        stop = self.source.shape[1] if stop is None else stop
        if not 0 <= start < stop <= self.source.shape[1]:
            raise ValueError("Observation interval must be nonempty and within the data.")

        def compute():
            values = self.values(transform)[:, start:stop]
            # Divide before summing to keep a large finite offset finite.
            mean = np.sum(values / values.shape[1], axis=1, keepdims=True)
            centered = values - mean
            squares = np.einsum("ij,ij->i", centered, centered)
            return centered, squares

        return self.arrays.get(("centered", transform, start, stop), compute)

    def cross(self, predictor="identity", target="identity", start=0, stop=None):
        stop = self.source.shape[1] if stop is None else stop
        return self.arrays.get(
            ("cross", predictor, target, start, stop),
            lambda: self.centered(predictor, start, stop)[0]
            @ self.centered(target, start, stop)[0].T,
        )

    def mse(self, coefficient, predictor="identity", target="identity"):
        """In-sample error, with direct residuals for cancellation-prone pairs.

        The ordinary path requires only P x P arithmetic. Near an exact fit,
        subtracting nearly equal sums loses the small residual; recompute just
        those pairs in bounded blocks instead of clipping a negative MSE to 0.
        """
        x, xx = self.centered(predictor)
        y, yy = self.centered(target)
        xy = self.cross(predictor, target)
        with np.errstate(over="ignore", invalid="ignore"):
            error = yy[None, :] - 2*coefficient*xy + coefficient**2*xx[:, None]
        near = np.isfinite(error) & (error <= 1e-10*yy[None, :])
        np.fill_diagonal(near, False)
        i, j = np.nonzero(near)
        step = max(1, self.block_bytes // (3 * x.shape[1] * 8))
        for offset in range(0, len(i), step):
            a, b = i[offset:offset+step], j[offset:offset+step]
            residual = y[b] - coefficient[a, b, None]*x[a]
            error[a, b] = np.einsum("ij,ij->i", residual, residual)
        error /= x.shape[1]
        error[~np.isfinite(error)] = np.nan
        np.fill_diagonal(error, np.nan)
        return error


def moment_workspace(data):
    return cached_workspace(data, "moments", MomentWorkspace)
