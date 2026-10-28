"""Directed spectral formulas; internal matrices are target-by-source.

Public SPI wrappers transpose these matrices exactly once. All squares and
products in this module precede any frequency-band reduction.
"""

import numpy as np

from pyspi._wilson import adjoint, inverse


def normalized_power(amplitude, axis):
    """Normalize squared amplitudes without squaring their physical scale."""
    amplitude = np.abs(amplitude)
    scale = np.max(amplitude, axis=axis, keepdims=True)
    np.divide(amplitude, scale, out=amplitude, where=scale > 0)
    np.square(amplitude, out=amplitude)
    total = np.sum(amplitude, axis=axis, keepdims=True)
    return np.divide(amplitude, total, out=np.full_like(amplitude, np.nan), where=total > 0)


def directed_coherence(transfer, covariance):
    variance = covariance.diagonal(axis1=-2, axis2=-1)
    return normalized_power(np.abs(transfer)*np.sqrt(variance)[..., None, None, :], -1)


def directed_transfer_function(transfer):
    return normalized_power(transfer, -1)


def partial_directed_coherence(transfer):
    if transfer.shape[-1] == 2:
        # In a pair, adj(H) swaps the diagonal; |det(H)| cancels. Off-diagonal
        # PDC is exactly DTF, and each diagonal is the opposite DTF diagonal.
        result = directed_transfer_function(transfer)
        result[..., 0, 0], result[..., 1, 1] = result[..., 1, 1].copy(), result[..., 0, 0].copy()
        return result
    return normalized_power(inverse(transfer), -2)


def generalized_partial_directed_coherence(transfer, covariance):
    if transfer.shape[-1] == 2:
        result = directed_coherence(transfer, covariance)
        result[..., 0, 0], result[..., 1, 1] = result[..., 1, 1].copy(), result[..., 0, 0].copy()
        return result
    variance = covariance.diagonal(axis1=-2, axis2=-1)
    return normalized_power(inverse(transfer)/np.sqrt(variance)[..., None, :, None], -2)


def squared_partial_coherence(transfer, covariance):
    """|precision_ij|^2/(precision_ii precision_jj), from the fitted model."""
    ar = inverse(transfer)
    precision = adjoint(ar) @ inverse(covariance)[..., None, :, :] @ ar
    # Normalize before the squared ratio to avoid products of precision scales.
    scale = np.max(abs(precision), axis=(-2, -1), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        precision = precision/scale
        diagonal = np.sqrt(precision.diagonal(axis1=-2, axis2=-1).real)
        magnitude = abs(precision)/diagonal[..., :, None]/diagonal[..., None, :]
        return magnitude**2


def direct_directed_transfer_function(transfer, covariance):
    # Frequency is third-last. The supplied grid must be the complete native
    # nonnegative grid, including DC and the even-length Nyquist bin once.
    return normalized_power(transfer, (-3, -1))*squared_partial_coherence(transfer, covariance)


def pairwise_granger(transfer, covariance):
    """Geweke pairwise log(total/intrinsic), retaining a genuine zero.

    Use the sum-of-squares decomposition after removing correlated innovations
    instead of subtracting almost equal powers or replacing zeros with epsilon.
    Zero intrinsic power is reported as unavailable, not assigned a finite floor.
    """
    if transfer.shape[-1] != 2:
        raise ValueError("Pairwise spectral Granger requires a two-channel fitted model.")
    result = np.full(transfer.shape, np.nan, dtype=float)
    for target, source in ((0, 1), (1, 0)):
        var_target = covariance[..., target, target].real
        var_source = covariance[..., source, source].real
        cross = covariance[..., target, source].real
        row = transfer[..., target, :]
        # A common row scale cancels from extrinsic/intrinsic power.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            row = row / np.max(abs(row), axis=-1, keepdims=True)
            correlation = cross/np.sqrt(var_target)/np.sqrt(var_source)
            residual_variance = var_source*(1-correlation**2)
            intrinsic = abs(row[..., target] + row[..., source]*(cross/var_target)[..., None])
            extrinsic = abs(row[..., source])
            log_ratio = (2*(np.log(extrinsic)-np.log(intrinsic))
                         + np.log(residual_variance/var_target)[..., None])
            values = np.logaddexp(0., log_ratio)
            valid = ((intrinsic > 0) & (residual_variance[..., None] > 0)
                     & (var_target[..., None] > 0) & np.isfinite(values))
        result[..., target, source] = np.where(valid, values, np.nan)
    return result
