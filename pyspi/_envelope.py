"""Shared broadband amplitude/log-power envelope correlations.

Pairwise orthogonalization removes the instantaneous analytic component. The
two projection directions are averaged, with abs applied to each coefficient
first for absolute variants. A zero-variance envelope has undefined correlation.
"""

from functools import cached_property

import numpy as np
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert

from pyspi._cache import cached_workspace


REASONS = np.array(["estimated", "nonfinite_input", "zero_log_envelope", "constant_envelope",
                    "zero_reference_envelope", "degenerate_orthogonal_signal", "numerical_failure",
                    "diagonal"])


def _unit_centered(values, error):
    """Centered unit vectors and numerical unavailability, along the last axis.

    Error bounds describe propagated arithmetic resolution, not statistical
    uncertainty. Centering can at most double an input's normwise error. The
    additional bound covers mean subtraction/reduction in float64. Scale before
    taking norms so a valid small envelope is not lost by squaring its units.
    """
    finite = np.isfinite(values).all(axis=-1)
    safe = np.where(finite[..., None], values, 0.)
    scale = np.max(abs(safe), axis=-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.)
    safe = safe/scale
    centered = safe-safe.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1)
    gamma = (8*np.ceil(np.log2(max(2, values.shape[-1])))+8)*np.finfo(float).eps
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        bound = 2*np.linalg.norm(error/scale, axis=-1) + gamma*np.linalg.norm(safe, axis=-1)
    resolved = finite & np.isfinite(bound) & (norm > bound)
    unit = np.divide(centered, norm[..., None], out=np.zeros_like(centered), where=resolved[..., None])
    reason = np.where(resolved, 0, 3).astype(np.uint8)
    reason[~finite] = 1
    return unit, reason


class EnvelopeWorkspace:
    block_bytes = 16*2**20

    def __init__(self, source):
        self.source = np.asarray(source, dtype=np.complex128 if np.iscomplexobj(source) else np.float64)
        self._bases = {}
        self._ordinary = {}
        self._orthogonal = None

    @cached_property
    def analytic(self):
        source = self.source
        finite = np.isfinite(source).all(axis=-1)
        safe = np.where(finite[:, None], source, 0.)
        scale = np.max(abs(safe), axis=-1, keepdims=True)
        scaled = np.divide(safe, scale, out=np.zeros_like(safe, dtype=source.dtype), where=scale > 0)
        if np.iscomplexobj(source):
            analytic = scaled.astype(complex)
            operations = 8
        else:
            nfft = next_fast_len(source.shape[-1])
            analytic = hilbert(scaled, N=nfft, axis=-1)[..., :source.shape[-1]]
            # Conservative float64 operation budget for two FFTs and analytic
            # weighting. It is a rounding-resolution guard, not an amplitude
            # threshold; positive channel rescaling is removed before the FFT.
            operations = 16*np.ceil(np.log2(max(2, nfft)))+16
        amplitude = abs(analytic)
        eps = np.finfo(float).eps
        gamma = operations*eps/(1-operations*eps)
        error = gamma*np.max(amplitude, axis=-1, keepdims=True)
        self.input_finite = finite
        return analytic, amplitude, error

    def base(self, log):
        if log not in self._bases:
            _, amplitude, error = self.analytic
            if log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = 2*np.log(amplitude)
                    errors = 2*error/amplitude
            else:
                values = amplitude
                errors = np.broadcast_to(error, amplitude.shape)
            unit, reason = _unit_centered(values, errors)
            if log:
                reason[(amplitude == 0).any(axis=-1)] = 2
            reason[~self.input_finite] = 1
            self._bases[log] = (unit, reason)
        return self._bases[log]

    @staticmethod
    def _bounded_correlation(values, reason, n):
        bound = 8*n*np.finfo(float).eps
        failure = ~np.isfinite(values) | (abs(values) > 1+bound)
        reason[failure & (reason == 0)] = 6
        # Only endpoint rounding is clipped; material out-of-range errors are NaN.
        values = np.clip(values, -1, 1)
        values[reason != 0] = np.nan
        return values, reason

    def ordinary(self, log):
        if log not in self._ordinary:
            unit, channel_reason = self.base(log)
            reason = np.maximum(channel_reason[:, None], channel_reason[None, :])
            values, reason = self._bounded_correlation(unit @ unit.T, reason, self.source.shape[-1])
            np.fill_diagonal(values, np.nan)
            np.fill_diagonal(reason, 7)
            self._ordinary[log] = values, reason
        return self._ordinary[log]

    def orthogonal(self, log):
        if self._orthogonal is None:
            self._orthogonal = self._orthogonal_directions()
        return self._orthogonal[log]

    def _orthogonal_directions(self):
        z, amplitude, error = self.analytic
        p, n = z.shape
        pairs = np.column_stack(np.triu_indices(p, 1))
        result = {log: (np.full((p, p), np.nan), np.full((p, p), 7, dtype=np.uint8))
                  for log in (False, True)}
        # Each real temporary is bounded; several coexist during a block.
        step = max(1, min(128, self.block_bytes//(n*8)))
        for start in range(0, len(pairs), step):
            i, j = pairs[start:start+step].T
            first, second = z[i], z[j]
            # Explicit real products make exact duplicate-channel cancellation
            # exact, unlike complex division followed by complex multiplication.
            product_a = second.imag*first.real
            product_b = second.real*first.imag
            cross = abs(product_a-product_b)
            cross_error = (amplitude[j]*error[i] + amplitude[i]*error[j] + error[i]*error[j]
                           + 3*np.finfo(float).eps*(abs(product_a)+abs(product_b)))
            for seed, other in ((i, j), (j, i)):
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    orth = cross/amplitude[seed]
                    orth_error = (cross_error + orth*error[seed])/amplitude[seed]
                    degenerate = np.linalg.norm(orth, axis=-1) <= np.linalg.norm(orth_error, axis=-1)
                zero_reference = (amplitude[seed] == 0).any(axis=-1)
                nonfinite_input = ~self.input_finite[seed] | ~self.input_finite[other]
                for log in (False, True):
                    unit_seed, seed_reason = self.base(log)
                    if log:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            values, errors = 2*np.log(orth), 2*orth_error/orth
                    else:
                        values, errors = orth, orth_error
                    unit_orth, reason = _unit_centered(values, errors)
                    reason = np.maximum(reason, seed_reason[seed])
                    if log:
                        reason[(orth == 0).any(axis=-1)] = 2
                    reason[degenerate] = 5
                    reason[zero_reference] = 4
                    reason[nonfinite_input] = 1
                    correlation = np.einsum("ij,ij->i", unit_seed[seed], unit_orth)
                    correlation, reason = self._bounded_correlation(correlation, reason, n)
                    result[log][0][seed, other] = correlation
                    result[log][1][seed, other] = reason
        return result

    def result(self, orth, log, absolute):
        if not orth:
            return self.ordinary(log)[0].copy()
        directional, _ = self.orthogonal(log)
        values = abs(directional) if absolute else directional
        return (values+values.T)*.5

    def diagnostics(self, orth, log):
        if not orth:
            return {"reason": REASONS[self.ordinary(log)[1]].copy()}
        _, directional_reason = self.orthogonal(log)
        reason = np.maximum(directional_reason, directional_reason.T)
        return dict(reason=REASONS[reason].copy(), directional_reason=REASONS[directional_reason].copy())


def envelope_workspace(data):
    if data.n_replications != 1:
        raise ValueError("Envelope correlation requires a single realisation.")
    return cached_workspace(data, "envelope", lambda source: EnvelopeWorkspace(source[:, :, 0]))
