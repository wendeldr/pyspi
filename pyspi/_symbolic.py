"""Ordinal-pattern transfer entropy, with shared symbols and target transitions.

Patterns contain oldest-to-newest observations. Stable sorting breaks exact
ties by time order. The plug-in conditional mutual information is in nats:
I(source pattern at t; target pattern at t+h | target pattern at t).
"""

import math

import numpy as np

from pyspi._information_history import history
from pyspi._moments import moment_workspace


REASONS = np.array(["estimated", "constant_pattern", "nonfinite_input",
                    "insufficient_observations", "diagonal", "numerical_failure"])


def _entropy(counts, axis=-1):
    counts = np.asarray(counts)
    total = counts.sum(axis=axis)
    logarithms = np.zeros_like(counts, dtype=float)
    np.log(counts, out=logarithms, where=counts > 0)
    return np.log(total)-(counts*logarithms).sum(axis=axis)/total


def _sparse_entropy(*columns):
    _, counts = np.unique(np.column_stack(columns), axis=0, return_counts=True)
    return float(_entropy(counts)), len(counts)


def _count_logcount(work, n):
    def compute():
        counts = np.arange(n+1, dtype=float)
        logarithms = np.zeros_like(counts)
        np.log(counts, out=logarithms, where=counts > 0)
        return counts*logarithms
    return work.arrays.get(("symbolic_count_logcount", n), compute)


def _symbols(work, length, spacing):
    def compute():
        p, t = work.source.shape
        start = (length-1)*spacing
        if start >= t:
            return np.empty((p, 0), dtype=np.int64), np.zeros(p, dtype=int), np.full(p, np.nan)
        embedded = history(work, start, spacing*np.arange(length))[:, :, ::-1]
        order = np.argsort(embedded, axis=2, kind="stable")
        codes = np.zeros(order.shape[:2], dtype=np.int64)
        # Lehmer codes give a compact alphabet 0..m!-1, independent of which
        # patterns happen to be observed in this window or another channel.
        for i in range(length-1):
            codes += (order[:, :, i+1:] < order[:, :, i:i+1]).sum(axis=2)*math.factorial(length-i-1)
        finite = np.isfinite(embedded).all(axis=2)
        codes[~finite] = -1
        if math.factorial(length) <= np.iinfo(np.int16).max:
            codes = codes.astype(np.int16)
        elif math.factorial(length) <= np.iinfo(np.int32).max:
            codes = codes.astype(np.int32)
        sorted_values = np.take_along_axis(embedded, order, axis=2)
        tied = (np.diff(sorted_values, axis=2) == 0).any(axis=2) & finite
        count = finite.sum(axis=1)
        fraction = np.divide(tied.sum(axis=1), count, out=np.full(p, np.nan), where=count > 0)
        occupied = np.array([len(np.unique(row[row >= 0])) for row in codes])
        return codes, occupied, fraction
    return work.arrays.get(("ordinal_symbols", length, spacing), compute)


def _transitions(work, length, spacing, shift):
    def compute():
        symbols, occupied, ties = _symbols(work, length, spacing)
        p, samples = symbols.shape
        h = np.full(p, np.nan)
        if samples > shift:
            n = samples-shift
            alphabet = math.factorial(length)
            bins = alphabet**2
            valid = np.flatnonzero((symbols >= 0).all(axis=1))
            if bins*8 <= work.block_bytes//4:
                lookup = _count_logcount(work, n)
                step = max(1, min(256, work.block_bytes//(8*(3*n+4*bins))))
                for first in range(0, len(valid), step):
                    rows = valid[first:first+step]
                    codes = symbols[rows, :-shift].astype(np.int32)*alphabet+symbols[rows, shift:]
                    offsets = np.arange(len(rows), dtype=np.int32)[:, None]*bins
                    counts = np.bincount((codes+offsets).ravel(), minlength=len(rows)*bins).reshape(len(rows), alphabet, alphabet)
                    past_counts = counts.sum(axis=2)
                    h[rows] = (lookup[past_counts].sum(axis=1)-lookup[counts].sum(axis=(1,2)))/n
            else:
                for j in valid:
                    past, future = symbols[j, :-shift], symbols[j, shift:]
                    h[j] = _sparse_entropy(past, future)[0]-_sparse_entropy(past)[0]
        return h, occupied, ties
    return work.arrays.get(("ordinal_transitions", length, spacing, shift), compute)


def _result(data, pattern_length=3, pattern_spacing=1, prediction_shift=1):
    work = moment_workspace(data)
    def compute():
        symbols, _, _ = _symbols(work, pattern_length, pattern_spacing)
        uncertainty, occupied, ties = _transitions(work, pattern_length, pattern_spacing, prediction_shift)
        p, samples = symbols.shape
        n = samples-prediction_shift
        value = np.full((p, p), np.nan)
        reason = np.full((p, p), 3, dtype=np.uint8)
        joint_occupied = np.zeros((p, p), dtype=int)
        if n > 0:
            source, target = np.nonzero(~np.eye(p, dtype=bool))
            xp, yp, yf = symbols[:, :n], symbols[:, :n], symbols[:, prediction_shift:]
            finite = (xp >= 0).all(axis=1)
            valid = finite[source] & np.isfinite(uncertainty[target])
            reason[source[~valid], target[~valid]] = 2
            source, target = source[valid], target[valid]
            alphabet = math.factorial(pattern_length)
            bins = alphabet**3
            dense = bins*8 <= work.block_bytes//4
            if dense:
                lookup = _count_logcount(work, n)
                step = max(1, min(256, work.block_bytes//(8*(4*n+4*bins))))
                for first in range(0, len(source), step):
                    a, b = source[first:first+step], target[first:first+step]
                    codes = (xp[a].astype(np.int32)*alphabet+yp[b])*alphabet+yf[b]
                    offsets = np.arange(len(a), dtype=np.int32)[:, None]*bins
                    counts = np.bincount((codes+offsets).ravel(), minlength=len(a)*bins).reshape(len(a), bins)
                    past_counts = counts.reshape(len(a), alphabet, alphabet, alphabet).sum(axis=3).reshape(len(a), -1)
                    value[a, b] = uncertainty[b]+(lookup[counts].sum(axis=1)-lookup[past_counts].sum(axis=1))/n
                    joint_occupied[a, b] = np.count_nonzero(counts, axis=1)
            else:
                # Large alphabets use occupied tuples only: neither m!^3 memory
                # nor packed integer overflow is allowed to change the result.
                for a, b in zip(source, target):
                    full, states = _sparse_entropy(xp[a], yp[b], yf[b])
                    given = _sparse_entropy(xp[a], yp[b])[0]
                    value[a, b] = uncertainty[b]-(full-given)
                    joint_occupied[a, b] = states
            reason[source, target] = 0
            constant = (np.array([len(np.unique(row)) for row in xp]) == 1)
            reason[(constant[:, None] | (occupied == 1)[None, :]) & np.isfinite(value)] = 1
            inconsistent = value < -1e-10
            value[inconsistent], reason[inconsistent] = np.nan, 5
            value[np.isfinite(value) & (value < 0)] = 0.
        np.fill_diagonal(value, np.nan)
        np.fill_diagonal(reason, 4)
        return value, reason, joint_occupied, np.asarray(max(n, 0)), uncertainty, occupied, ties
    return work.arrays.get(("symbolic_transfer", pattern_length, pattern_spacing, prediction_shift), compute)


def symbolic_result(data, **parameters):
    return _result(data, **parameters)[0].copy()


def symbolic_diagnostics(data, **parameters):
    _, reason, joint, n, uncertainty, occupied, ties = _result(data, **parameters)
    return {"reason": REASONS[reason].copy(), "joint_states_observed": joint.copy(),
            "n_observations": int(n), "patterns_observed": occupied.copy(),
            "target_transition_entropy": uncertainty.copy(), "tied_pattern_fraction": ties.copy()}
