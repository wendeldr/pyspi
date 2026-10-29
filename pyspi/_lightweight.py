"""Small shared moment reductions for lag, event and amplitude comparisons."""

import numpy as np

from pyspi._moments import moment_workspace


def _reasons(valid_source, valid_target, finite_source, finite_target):
    reason = np.full((len(valid_source), len(valid_source)), "estimated", dtype="U24")
    reason[~valid_source[:, None] | ~valid_target[None, :]] = "constant_input"
    reason[~finite_source[:, None] | ~finite_target[None, :]] = "nonfinite_input"
    np.fill_diagonal(reason, "diagonal")
    return reason


def lightweight_result(data, kind, setting=None):
    work = moment_workspace(data)

    def compute():
        n = work.source.shape[1]
        if n < 2:
            raise ValueError("At least two observations are required.")
        if kind == "lag":
            lag = setting
            if lag >= n-1:
                raise ValueError("Fixed lag must leave at least two overlapping observations.")
            a, va, _ = work.standardized(stop=n-lag)
            b, vb, _ = work.standardized(start=lag)
            finite_source = np.isfinite(work.source[:, :n-lag]).all(axis=1)
            finite_target = np.isfinite(work.source[:, lag:]).all(axis=1)
            # Row = source now, column = target lag samples later.
            result = np.clip(a @ b.T, -1., 1.)
        else:
            unit, variance, _ = work.standardized()
            va = vb = variance
            finite_source = finite_target = np.isfinite(work.source).all(axis=1)
            if kind == "event":
                # Sample-standardized z; strict exceedance, abs includes both polarities.
                events = (abs(unit)*np.sqrt(n-1) > setting).astype(float)
                rate = events.mean(axis=1)
                result = events @ events.T/n - rate[:, None]*rate[None, :]
            elif kind == "log_scale":
                with np.errstate(invalid="ignore"):
                    result = .5*(variance[None, :]-variance[:, None])
            else:
                raise ValueError(f"Unknown lightweight reduction: {kind}")
        reason = _reasons(np.isfinite(va), np.isfinite(vb), finite_source, finite_target)
        failure = ~np.isfinite(result) & (reason == "estimated")
        reason[failure] = "numerical_failure"
        result[reason != "estimated"] = np.nan
        return result, reason

    result, reason = work.arrays.get(("lightweight", kind, setting), compute)
    return result.copy(), reason.copy()
