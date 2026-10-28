"""Batched single-predictor regressions; row = predictor, column = target."""

import numpy as np

from pyspi._moments import moment_workspace


REASONS = np.array(["estimated", "nonfinite_input", "numerical_failure", "iteration_limit", "diagonal"])


def _bayesian_ridge(work):
    """One-predictor MacKay updates, independently stopped for each pair.

    Keep sklearn BayesianRidge's priors (all 1e-6), initialization, tolerance
    (1e-3), 300-iteration limit and final posterior-mean update. Only the linear
    algebra and dispatch are batched; no evidence optimum is substituted.
    """
    _, xx = work.centered()
    _, yy = work.centered()
    xy = work.cross()
    n = work.source.shape[1]
    active = np.isfinite(xy) & np.isfinite(xx[:, None]) & np.isfinite(yy[None, :])
    np.fill_diagonal(active, False)
    iterations = np.zeros(xy.shape, dtype=np.uint16)
    coefficient = np.zeros_like(xy)
    alpha = np.broadcast_to(1/(yy/n+np.finfo(float).eps), xy.shape).copy()
    lam = np.ones_like(xy)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for step in range(300):
            updated = xy / (xx[:, None] + lam/alpha)
            difference = abs(updated-coefficient)
            coefficient = np.where(active, updated, coefficient)
            sse = work.mse(coefficient)*n
            gamma = alpha*xx[:, None]/(lam + alpha*xx[:, None])
            lam = np.where(active, (gamma+2e-6)/(coefficient**2+2e-6), lam)
            alpha = np.where(active, (n-gamma+2e-6)/(sse+2e-6), alpha)
            iterations[active] = step+1
            if step:
                active &= difference >= 1e-3
            if not active.any():
                break
        coefficient = xy / (xx[:, None] + lam/alpha)
    return work.mse(coefficient), iterations, active


def _regression(data, model, alpha=1., l1_ratio=.5, penalty_scale="sum"):
    work = moment_workspace(data)

    def compute():
        shape = (len(work.source), len(work.source))
        iterations = np.zeros(shape, dtype=np.uint16)
        limited = np.zeros(shape, dtype=bool)
        if model == "Ridge":
            _, xx = work.centered()
            penalty = alpha * (work.source.shape[1] if penalty_scale == "mean" else 1)
            xy = work.cross()
            coefficient = np.divide(xy, xx[:, None]+penalty, out=np.zeros_like(xy), where=xx[:, None]+penalty > 0)
            error = work.mse(coefficient)
        elif model in ("Lasso", "ElasticNet"):
            _, xx = work.centered()
            xy = work.cross()
            ratio = 1. if model == "Lasso" else l1_ratio
            n = work.source.shape[1]
            numerator = np.sign(xy)*np.maximum(abs(xy) - n*alpha*ratio, 0.)
            denominator = xx[:, None] + n*alpha*(1-ratio)
            coefficient = np.divide(numerator, denominator, out=np.zeros_like(xy), where=denominator > 0)
            error = work.mse(coefficient)
        elif model == "BayesianRidge":
            error, iterations, limited = _bayesian_ridge(work)
        elif model == "RECI":
            _, xx = work.centered("cube_minmax")
            xy = work.cross("cube_minmax", "minmax")
            coefficient = np.divide(xy, xx[:, None], out=np.zeros_like(xy), where=xx[:, None] > 0)
            error = work.mse(coefficient, "cube_minmax", "minmax")
        else:
            raise ValueError(f"Unsupported batched regression: {model}")
        reason = np.zeros(shape, dtype=np.uint8)
        reason[limited] = 3
        reason[~np.isfinite(error)] = 2
        invalid = ~np.isfinite(work.source).all(axis=1)
        reason[invalid[:, None] | invalid[None, :]] = 1
        error[reason == 1] = np.nan
        np.fill_diagonal(reason, 4)
        return error, iterations, reason

    return work.arrays.get(("regression", model, alpha, l1_ratio, penalty_scale), compute)


def regression_result(data, model, **kwargs):
    # Returning a copy prevents callers from mutating cached estimates.
    return _regression(data, model, **kwargs)[0].copy()


def regression_diagnostics(data, model, **kwargs):
    _, iterations, reason = _regression(data, model, **kwargs)
    return {"reason": REASONS[reason].copy(), "iterations": iterations.copy()}
