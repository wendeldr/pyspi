"""Shared Gaussian information calculations, in nats, using sample covariance.

No variance floor in physical units and no artificial ridge/jitter. Singular
differential entropies are reported as undefined; exact nonconstant linear
dependence has infinite mutual information. Only affected pairs are invalid.
"""

import numpy as np

from pyspi._moments import moment_workspace


H_UNIT = .5 * (1 + np.log(2*np.pi))
REASONS = np.array(["estimated", "constant_variable", "singular_covariance",
                    "nonfinite_input", "insufficient_observations", "diagonal",
                    "numerical_failure", "redundant_conditioning"])


def _pair_information(work, lag=0):
    """MI, source/target marginal entropies and reasons for aligned observations."""
    def compute():
        p, t = work.source.shape
        n = t-lag
        if n < 2:
            return (np.full((p, p), np.nan), np.full(p, np.nan),
                    np.full(p, np.nan), np.full((p, p), 4, dtype=np.uint8))
        x, vx, scaled_x = work.standardized(0, n)
        y, vy, scaled_y = work.standardized(lag, t)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            # Ordinary inputs reuse the regression Gram matrix. Rescaled rows
            # use unit vectors directly so covariance units cannot overflow.
            if lag == 0 and not (scaled_x.any() or scaled_y.any()):
                _, ss = work.centered()
                rho = work.cross()/np.sqrt(ss[:, None])/np.sqrt(ss[None, :])
            else:
                rho = x @ y.T
            remaining = (1-rho)*(1+rho)
            # Near a perfect fit, subtraction loses the residual. Measure it
            # directly; average both directions for an order-invariant result.
            near = np.isfinite(rho) & (remaining < 1e-8)
            if lag == 0:
                near = np.triu(near, 1)
            else:
                np.fill_diagonal(near, False)
            i, j = np.nonzero(near)
            block = max(1, work.block_bytes//(5*n*8))
            for start in range(0, len(i), block):
                a, b = i[start:start+block], j[start:start+block]
                r = np.einsum("ij,ij->i", x[a], y[b])
                e1 = y[b]-r[:, None]*x[a]
                e2 = x[a]-r[:, None]*y[b]
                residual = .5*(np.einsum("ij,ij->i", e1, e1)+np.einsum("ij,ij->i", e2, e2))
                residual[residual <= (8*np.finfo(float).eps)**2] = 0.
                remaining[a, b] = residual
                if lag == 0:
                    remaining[b, a] = residual
            mi = -.5*np.log1p(-rho**2)
            mi[near] = -.5*np.log(remaining[near])
            if lag == 0:
                a, b = np.triu_indices(p, 1)
                mi[b, a] = mi[a, b]
            reason = np.zeros((p, p), dtype=np.uint8)
            singular = remaining == 0
            reason[singular] = 2
            constants = np.isneginf(vx[:, None]) | np.isneginf(vy[None, :])
            mi[constants], reason[constants] = 0., 1
            invalid = np.isnan(vx[:, None]) | np.isnan(vy[None, :])
            reason[np.isnan(mi)] = 6
            mi[invalid], reason[invalid] = np.nan, 3
        return mi, H_UNIT+.5*vx, H_UNIT+.5*vy, reason
    return work.arrays.get(("gaussian_pairs", lag), compute)


def gaussian_marginal_entropy(data):
    work = moment_workspace(data)
    _, variance, _ = work.standardized()
    result = H_UNIT+.5*variance
    result[~np.isfinite(result)] = np.nan
    return result


def _simple_result(data, measure):
    work = moment_workspace(data)

    def compute():
        mi, hx, hy, status = _pair_information(work, int(measure == "tlmi"))
        status = status.copy()
        if measure in ("mi", "tlmi"):
            result = mi.copy()
        elif measure == "je":
            result = hx[:, None]+hy[None, :]-mi
            result[(status == 1) | (status == 2)] = np.nan
        elif measure == "ce":
            # Row conditions the column; a constant source adds no information.
            result = hy[None, :]-mi
            result[np.broadcast_to(~np.isfinite(hy), result.shape) | (status == 2)] = np.nan
        else:
            raise ValueError(f"Unknown simple Gaussian calculation: {measure}")
        np.fill_diagonal(result, np.nan)
        np.fill_diagonal(status, 5)
        return result, status

    return work.arrays.get(("gaussian_result", measure), compute)


def gaussian_result(data, measure, **parameters):
    if measure in ("mi", "je", "ce", "tlmi"):
        return _simple_result(data, measure)[0].copy()
    from pyspi._gaussian_history import history_result
    return history_result(data, measure, **parameters)[0].copy()


def gaussian_diagnostics(data, measure, **parameters):
    if measure in ("mi", "je", "ce", "tlmi"):
        reason = _simple_result(data, measure)[1]
    else:
        from pyspi._gaussian_history import history_result
        reason = history_result(data, measure, **parameters)[1]
    result = {"reason": REASONS[reason].copy()}
    if measure == "te" and parameters.get("auto_embed_method") is not None:
        from pyspi._gaussian_history import embedding_diagnostics
        result.update(embedding_diagnostics(data, parameters["k_search_max"], parameters["tau_search_max"]))
    return result
