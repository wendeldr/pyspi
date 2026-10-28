"""Permutation-equivariant Wilson factorization on a finite Fourier grid.

The plus operator splits self-mirrored lags (zero and the even-grid midpoint)
equally and retains the positive lags. For a Hermitian spectrum G it satisfies
G_plus + G_plus.H = G. A symmetric positive square-root start avoids imposing
a channel order through a triangular zero-lag convention. As with any finite
Fourier grid, reconstruction accuracy does not establish statistical accuracy
or resolve impulse-response aliasing from insufficient observed duration.
"""

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, ifft


REASONS = np.array([
    "estimated", "nonfinite_spectrum", "singular_spectrum", "invalid_spectrum",
    "nonconverged", "numerical_failure",
])


def adjoint(a):
    return a.swapaxes(-1, -2).conj()


def inverse(a):
    """Batched inverse, isolating singular matrices; optimized for pair fits."""
    if a.shape[-1] == 2:
        scale = np.max(np.abs(a), axis=(-2, -1), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            b = a / scale
            determinant = b[..., 0, 0]*b[..., 1, 1] - b[..., 0, 1]*b[..., 1, 0]
            result = np.empty_like(b)
            result[..., 0, 0], result[..., 1, 1] = b[..., 1, 1], b[..., 0, 0]
            result[..., 0, 1], result[..., 1, 0] = -b[..., 0, 1], -b[..., 1, 0]
            result /= determinant[..., None, None]
            result /= scale
        return result
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        result = np.full_like(a, np.nan)
        for original, target in zip(a.reshape(-1, *a.shape[-2:]),
                                    result.reshape(-1, *a.shape[-2:])):
            try:
                target[:] = np.linalg.inv(original)
            except np.linalg.LinAlgError:
                pass
        return result


def plus_operator(g):
    """Symmetric discrete causal split; frequency is the third-last axis."""
    n = g.shape[-3]
    lag = ifft(g, axis=-3)
    lag[..., 0, :, :] *= .5
    lag[..., n//2+1:, :, :] = 0
    if n % 2 == 0:
        lag[..., n//2, :, :] *= .5
    return fft(lag, axis=-3)


@dataclass
class Factorization:
    factor: np.ndarray
    reason: np.ndarray
    residual: np.ndarray
    iterations: np.ndarray


def factorize(spectrum, *, tolerance=1e-10, max_iterations=100):
    """Factor independent real-process spectra, with pair/window-local failure.

    Input shape (..., frequencies, channels, channels), in two-sided FFT order.
    Both relative iterate change and reconstruction residual must meet tolerance.
    Residual is measured after diagonal spectral-power equilibration, so arbitrary
    channel units cannot hide a poorly reconstructed low-power channel. Numerical
    rank uses p*eps*largest_eigenvalue on the equilibrated Hermitian spectrum;
    it identifies unresolved rank, not weak coupling. No ridge or fallback fit
    is substituted. The caller bounds batch size and retains diagnostics.
    """
    original = np.asarray(spectrum, dtype=np.complex128)
    if original.ndim < 3 or original.shape[-1] != original.shape[-2]:
        raise ValueError("Expected square spectral matrices on a two-sided frequency grid.")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Positive factorization tolerance and iteration count are required.")
    shape = original.shape[:-3]
    n, p, _ = original.shape[-3:]
    if n < 2:
        raise ValueError("Factorization requires at least two Fourier bins.")
    s = original.reshape(-1, n, p, p)
    count = len(s)
    factors = np.full_like(s, np.nan)
    reason = np.full(count, 4, dtype=np.uint8)
    residual = np.full(count, np.nan)
    iterations = np.zeros(count, dtype=np.int32)
    finite = np.isfinite(s).all(axis=(1, 2, 3))
    reason[~finite] = 1
    # Scale by channel standard deviations; applying the two divisions separately
    # avoids an unnecessary product of very large/small channel scales.
    diagonal = s.diagonal(axis1=-2, axis2=-1).real
    scale = np.sqrt(np.maximum(diagonal, 0).max(axis=1))
    usable = finite & (scale > 0).all(axis=-1)
    reason[finite & ~usable] = 2
    reason[finite & (diagonal.max(axis=1) < 0).any(axis=-1)] = 3
    ids = np.flatnonzero(usable)
    if ids.size:
        d = scale[ids]
        normalized = s[ids] / d[:, None, :, None] / d[:, None, None, :]
        magnitude = np.max(np.abs(normalized), axis=(1, 2, 3))
        bound = 32*p*np.finfo(float).eps*magnitude
        hermitian_error = np.max(abs(normalized-adjoint(normalized)), axis=(1, 2, 3))
        mirror = (-np.arange(n)) % n
        real_process_error = np.max(abs(normalized-normalized[:, mirror].conj()), axis=(1, 2, 3))
        symmetric = (hermitian_error <= bound) & (real_process_error <= bound)
        reason[ids[~symmetric]] = 3
        ids, normalized = ids[symmetric], normalized[symmetric]
        # Remove only verified floating-point asymmetry before Hermitian eigh.
        normalized = (normalized + adjoint(normalized))*.5
        if ids.size:
            eigenvalues = np.linalg.eigvalsh(normalized)
            rank_bound = p*np.finfo(float).eps*np.max(abs(eigenvalues), axis=-1)
            indefinite = (eigenvalues[..., 0] < -rank_bound).any(axis=-1)
            singular = (eigenvalues[..., 0] <= rank_bound).any(axis=-1)
            reason[ids[singular]] = 2
            reason[ids[indefinite]] = 3
            ids, normalized = ids[~singular], normalized[~singular]
    else:
        normalized = np.empty((0, n, p, p), complex)

    if ids.size:
        values, vectors = np.linalg.eigh(normalized.mean(axis=1).real)
        root = (vectors*np.sqrt(values)[:, None, :]) @ vectors.swapaxes(-1, -2)
        g = np.broadcast_to(root[:, None], normalized.shape).astype(complex).copy()
        norm_s = np.max(abs(normalized), axis=(1, 2, 3))
        active = np.arange(len(ids))
        for iteration in range(1, max_iterations+1):
            old = g[active]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                inv = inverse(old)
                predictor = inv @ normalized[active] @ adjoint(inv) + np.eye(p)
                predictor = (predictor + adjoint(predictor))*.5
                new = old @ plus_operator(predictor)
                reconstruction = new @ adjoint(new)
                error = np.max(abs(reconstruction-normalized[active]), axis=(1, 2, 3)) / norm_s[active]
                change = np.max(abs(new-old), axis=(1, 2, 3)) / np.max(abs(new), axis=(1, 2, 3))
            g[active] = new
            residual[ids[active]] = error
            iterations[ids[active]] = iteration
            good = np.isfinite(error) & np.isfinite(change)
            converged = good & (error <= tolerance) & (change <= tolerance)
            reason[ids[active[~good]]] = 5
            reason[ids[active[converged]]] = 0
            active = active[good & ~converged]
            if not active.size:
                break
        accepted = reason[ids] == 0
        factors[ids[accepted]] = g[accepted] * scale[ids[accepted], None, :, None]
    return Factorization(factors.reshape(original.shape), reason.reshape(shape),
                         residual.reshape(shape), iterations.reshape(shape))
