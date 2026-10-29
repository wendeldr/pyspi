"""Compiled PEC projection/reduction with the same float64 resolution guards.

No fastmath: cancellation, infinities and NaNs determine availability. Scratch
space is O(samples), independent of the number of channel pairs.
"""

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def _correlation(seed, values, errors, gamma):
    n = len(values)
    scale = 0.
    for t in range(n):
        if not np.isfinite(values[t]):
            return np.nan, 1
        scale = max(scale, abs(values[t]))
    if scale == 0.:
        scale = 1.
    mean = 0.
    safe_energy = 0.
    error_energy = 0.
    for t in range(n):
        value = values[t]/scale
        mean += value
        safe_energy += value*value
        error_energy += (errors[t]/scale)**2
    mean /= n
    norm = 0.
    product = 0.
    for t in range(n):
        centered = values[t]/scale - mean
        norm += centered*centered
        product += seed[t]*centered
    norm = np.sqrt(norm)
    bound = 2*np.sqrt(error_energy) + gamma*np.sqrt(safe_energy)
    if not np.isfinite(bound) or not norm > bound:
        return np.nan, 3
    return product/norm, 0


@njit(cache=True, error_model="numpy")
def orthogonal_directions(z, amplitude, error, input_finite, unit, seed_reason, log):
    p, n = z.shape
    result = np.full((p, p), np.nan)
    reason = np.full((p, p), 7, dtype=np.uint8)
    cross, cross_error = np.empty(n), np.empty(n)
    values, errors = np.empty(n), np.empty(n)
    eps = np.finfo(np.float64).eps
    gamma = (8*np.ceil(np.log2(max(2, n)))+8)*eps
    for i in range(p):
        for j in range(i+1, p):
            if not input_finite[i] or not input_finite[j]:
                reason[i, j] = reason[j, i] = 1
                continue
            for t in range(n):
                a = z[j, t].imag*z[i, t].real
                b = z[j, t].real*z[i, t].imag
                cross[t] = abs(a-b)
                cross_error[t] = (amplitude[j, t]*error[i, 0] + amplitude[i, t]*error[j, 0]
                                  + error[i, 0]*error[j, 0] + 3*eps*(abs(a)+abs(b)))
            for seed, other in ((i, j), (j, i)):
                orth_energy, error_energy = 0., 0.
                zero_reference, zero_orth = False, False
                for t in range(n):
                    orth = cross[t]/amplitude[seed, t]
                    orth_error = (cross_error[t] + orth*error[seed, 0])/amplitude[seed, t]
                    orth_energy += orth*orth
                    error_energy += orth_error*orth_error
                    zero_reference |= amplitude[seed, t] == 0.
                    zero_orth |= orth == 0.
                    if log:
                        values[t] = 2*np.log(orth)
                        errors[t] = 2*orth_error/orth
                    else:
                        values[t], errors[t] = orth, orth_error
                correlation, status = _correlation(unit[seed], values, errors, gamma)
                status = max(status, seed_reason[seed])
                if log and zero_orth:
                    status = 2
                if np.sqrt(orth_energy) <= np.sqrt(error_energy):
                    status = 5
                if zero_reference:
                    status = 4
                if status == 0 and (not np.isfinite(correlation) or abs(correlation) > 1+8*n*eps):
                    status = 6
                reason[seed, other] = status
                if status == 0:
                    result[seed, other] = min(1., max(-1., correlation))
    return result, reason
