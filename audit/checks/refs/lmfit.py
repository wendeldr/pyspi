"""References for the lmfit family (pyspi/statistics/misc.py:106-126 LinearModel).

Runs in the COMPARISON venv (no pyspi, no sklearn import here: every reference
below is numpy/scipy only, so it is independent of the scikit-learn delegate).

A[i, j] in pyspi = in-sample MSE of the sklearn model fitted with predictor
x_i and target x_j (misc.py:122-126).  Every reference matrix below is built
with the same orientation: M[i, j] = f(x_i -> x_j).

Units and their from-scratch references
  Ridge          _kernels_basic_misc.ridge_mse (w = sxy/(sxx + 1)) and, on the
                 z-scored path, the analytic closed form
                 ((T-1)/T)(1 - r^2 (T^2-1)/T^2).
  Lasso          enet_sklearn_tol(..., l1_ratio=1): exact single-coordinate
                 minimiser S(sxy, a)/sxx with a = alpha*T, plus sklearn's
                 initial duality-gap exit (_cd_fast.pyx:391-400) modelled.
                 On z-scored data this is the constant (T-1)/T
                 (= _kernels_basic_misc.lasso_1d_mse).
  ElasticNet     enet_sklearn_tol(..., l1_ratio=0.5) (exit modelled; tag
                 "exitmodel") and _kernels_basic_misc.elasticnet_1d_mse
                 (pure minimiser, no exit; tag "pure").
  BayesianRidge  _kernels_basic_misc.bayesian_ridge_mse (MacKay/Tipping
                 recursion, own code) and an evidence-maximisation optimality
                 check (direct maximisation of the log evidence with the
                 Gamma hyperpriors; tag "evidence").
  SGDRegressor   _kernels_basic_misc.ols_mse (the documented "ordinary least
                 squares fit"); tolerance atol = 1/(10 T) (tag "ols"),
                 optimality A >= OLS (tag "optimality"), determinism and seed
                 sensitivity (probe tags).

Run:  <comparison venv>/bin/python audit/checks/refs/lmfit.py [run_id-substring]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import optimize  # noqa: E402
from scipy.signal import detrend  # noqa: E402
from scipy.stats import zscore  # noqa: E402

import make_data  # noqa: E402
import _kernels_basic_misc as K  # noqa: E402

FAMILY = "lmfit"
MANIFEST = CHECKS / "manifest" / f"{FAMILY}.json"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
SELF = "audit/checks/refs/lmfit.py"
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"


# ------------------------------------------------------------------ preprocessing
def prep(x, path):
    """Mirror pyspi/data.py:184-195 (detrend, then zscore ddof=1), using scipy primitives."""
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return x
    if path == "pipeline":
        return zscore(x, axis=1, ddof=1)
    if path == "pipeline-detrend":
        return zscore(detrend(x, axis=1), axis=1, ddof=1)
    raise ValueError(path)


def pair_matrix(X, f):
    P = X.shape[0]
    M = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                M[i, j] = f(X[i], X[j])
    return M


def corr_matrix(X):
    return np.corrcoef(X)


# ------------------------------------------------------------------ from-scratch estimators
def enet_sklearn_tol(x, y, alpha=1.0, l1_ratio=0.5, tol=1e-4, return_params=False):
    """Single-predictor elastic net / lasso as sklearn's coordinate descent returns it.

    Objective (sklearn _coordinate_descent.py:884-894, scaled by n as in enet_path
    :781-784): 0.5||y - w x||^2 + a|w| + 0.5 b w^2 on centred x, y with a = alpha*l1_ratio*n,
    b = alpha*(1-l1_ratio)*n.  Exact minimiser w* = sign(s) max(|s| - a, 0)/(sxx + b), s = x.y.
    Solver behaviour modelled (_cd_fast.pyx:391-400, gap formula 138-157): at w = 0 the
    duality gap is 0.5 yy (1 - min(1, a/|s|))^2; if it is <= tol*yy the solver returns
    w = 0 with n_iter 0.  Otherwise one cyclic update lands on w*, the second update leaves it
    unchanged and the gap check (gap(w*) = 0 analytically) stops with n_iter 2.
    Intercept = ybar - w xbar (_base.py:318-334); MSE on the training data."""
    x, y = np.ravel(np.asarray(x, float)), np.ravel(np.asarray(y, float))
    xc, yc = x - x.mean(), y - y.mean()
    n = x.size
    a, b = alpha * l1_ratio * n, alpha * (1.0 - l1_ratio) * n
    s, yy, xx = xc @ yc, yc @ yc, xc @ xc
    scale = 1.0 if abs(s) <= a else a / abs(s)
    gap0 = 0.5 * yy * (1.0 - scale) ** 2
    if gap0 <= tol * yy:
        w, n_iter = 0.0, 0
    else:
        w, n_iter = np.sign(s) * max(abs(s) - a, 0.0) / (xx + b), 2
    mse = float(np.mean((y - (y.mean() + w * (x - x.mean()))) ** 2))
    if return_params:
        return mse, dict(coef=w, n_iter=n_iter, intercept=y.mean() - w * x.mean())
    return mse


def ridge_closed_form(r, T):
    """Ridge alpha=1 on ddof=1 z-scored data: w = r(T-1)/T, MSE = ((T-1)/T)(1 - r^2 (T^2-1)/T^2)."""
    return (T - 1) / T * (1.0 - r ** 2 * (T * T - 1.0) / (T * T))


def br_log_evidence(la, ll, n, xx, xy, yy, a1=1e-6, a2=1e-6, l1=1e-6, l2=1e-6):
    """Log evidence + log Gamma hyperpriors (MacKay 1992; Tipping 2001 eq. 33 form), one centred
    predictor; written from the Gaussian marginal y ~ N(0, alpha^-1 I + lambda^-1 x x^T)."""
    al, lam = np.exp(la), np.exp(ll)
    A = lam + al * xx
    m = al * xy / A
    sse = yy - 2 * m * xy + m * m * xx
    return (l1 * ll - l2 * lam + a1 * la - a2 * al
            + 0.5 * (ll + n * la - al * sse - lam * m * m - np.log(A) - n * np.log(2 * np.pi)))


def br_evidence_mse(x, y):
    """MSE of the posterior mean at the maximiser of the log evidence (direct maximisation:
    coarse grid in (log alpha, log lambda), then Nelder-Mead polish)."""
    x, y = np.ravel(x), np.ravel(y)
    xc, yc = x - x.mean(), y - y.mean()
    n, xx, xy, yy = x.size, xc @ xc, xc @ yc, yc @ yc
    f = lambda p: -br_log_evidence(p[0], p[1], n, xx, xy, yy)  # noqa: E731
    la0 = -np.log(yy / n + K.EPS)
    gla = np.linspace(la0 - 20, la0 + 40, 241)
    gll = np.linspace(-20, 40, 241)
    G = -br_log_evidence(gla[:, None], gll[None, :], n, xx, xy, yy)
    ia, il = np.unravel_index(np.argmin(G), G.shape)
    res = optimize.minimize(f, x0=[gla[ia], gll[il]], method="Nelder-Mead",
                            options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000, maxfev=40000))
    al, lam = np.exp(res.x)
    m = al * xy / (lam + al * xx)
    return float(np.mean((yc - m * xc) ** 2)), dict(log_alpha=res.x[0], log_lambda=res.x[1],
                                                     at_grid_edge=bool(il in (0, 240) or ia in (0, 240)))


UNIT_FN = {
    "misc.lmfit.Ridge": ("ridge_mse", lambda x, y: K.ridge_mse(x, y, 1.0)),
    "misc.lmfit.Lasso": ("enet_sklearn_tol(l1_ratio=1)", lambda x, y: enet_sklearn_tol(x, y, 1.0, 1.0)),
    "misc.lmfit.ElasticNet": ("enet_sklearn_tol(l1_ratio=0.5)", lambda x, y: enet_sklearn_tol(x, y, 1.0, 0.5)),
    "misc.lmfit.BayesianRidge": ("bayesian_ridge_mse", lambda x, y: K.bayesian_ridge_mse(x, y)),
    "misc.lmfit.SGDRegressor": ("ols_mse", K.ols_mse),
}
FN_SRC = {
    "ridge_mse": "refs/_kernels_basic_misc.py:ridge_mse",
    "enet_sklearn_tol(l1_ratio=1)": f"{SELF}:enet_sklearn_tol (alpha=1, l1_ratio=1, tol exit modelled)",
    "enet_sklearn_tol(l1_ratio=0.5)": f"{SELF}:enet_sklearn_tol (alpha=1, l1_ratio=0.5, tol exit modelled)",
    "bayesian_ridge_mse": "refs/_kernels_basic_misc.py:bayesian_ridge_mse",
    "ols_mse": "refs/_kernels_basic_misc.py:ols_mse",
}


# ------------------------------------------------------------------ helpers
def load_pyspi(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], dtype=np.float64)


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def asym(A):
    return float(np.nanmax(np.abs(A - A.T))) if A is not None else float("nan")


def write(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in kw.items():
        if isinstance(v, str):
            clean[k] = np.array(v)
        elif isinstance(v, (bool, np.bool_)):
            clean[k] = np.array(int(v))
        else:
            clean[k] = np.asarray(v)
    np.savez(OUT_REF / f"{rid}.npz", **clean)


def r_grid(kind):
    if kind == "enet":
        return np.unique(np.concatenate([np.linspace(0, 1, 201), np.linspace(0.49, 0.52, 601)]))
    return np.linspace(0, 1, 2001)


def grid_pair(x, r):
    """Pair (u, y_r) with sample correlation exactly r (to rounding): u = z(x0), v = z(x1 orth u)."""
    u = zscore(x[0], ddof=1)
    v = x[1] - x[1].mean()
    v = v - (v @ u) / (u @ u) * u
    v = zscore(v, ddof=1)
    return np.vstack([u, r * u + np.sqrt(max(1.0 - r * r, 0.0)) * v])


# ------------------------------------------------------------------ run handlers
def h_exact(run, X, P):
    """Exact comparison with the unit's from-scratch estimator on the same preprocessed data."""
    uid, T, rid = run["unit_id"], run["T"], run["run_id"]
    name, f = UNIT_FN[uid]
    M = pair_matrix(X, f)
    extra = {}
    if P is not None:
        extra["asym_max"] = asym(P)
        extra["transpose_max_abs"] = float(np.nanmax(np.abs(P - M.T)))  # orientation sensitivity
        extra["min_offdiag"] = float(np.min(offdiag(P)))
        extra["max_offdiag"] = float(np.max(offdiag(P)))
    c = (T - 1) / T
    if uid in ("misc.lmfit.Lasso", "misc.lmfit.ElasticNet") and run["path"] != "kernel":
        R = corr_matrix(X)
        lim = 0.5 * T / ((T - 1) * (1 - np.sqrt(2e-4)))  # tol-exit bound (l1_ratio 0.5)
        thr = 0.5 * T / (T - 1)                            # exact-minimiser threshold
        ar = np.abs(offdiag(R))
        extra["max_abs_r"] = float(ar.max())
        extra["max_abs_r_times_c"] = float(ar.max() * c)
        if P is not None:
            extra["n_offdiag"] = int(ar.size)
            extra["n_const"] = int(np.sum(np.abs(offdiag(P) - c) <= 1e-12))  # entries equal to (T-1)/T to 1e-12
            extra["max_abs_dev_from_c"] = float(np.max(np.abs(offdiag(P) - c)))
        if uid == "misc.lmfit.ElasticNet":
            pure = pair_matrix(X, lambda a, b: K.elasticnet_1d_mse(a, b, 1.0, 0.5))
            extra["n_band_pairs"] = int(np.sum((ar > thr) & (ar <= lim)))
            extra["pure_closed_form_max_abs"] = float(np.nanmax(np.abs(M - pure)))
            if P is not None:
                extra["pyspi_minus_pure_max_abs"] = float(np.nanmax(np.abs(P - pure)))
    if uid in ("misc.lmfit.Lasso", "misc.lmfit.ElasticNet") and run["path"] == "kernel":
        l1 = 1.0 if uid.endswith("Lasso") else 0.5
        Pn = X.shape[0]
        act = [enet_sklearn_tol(X[i], X[j], 1.0, l1, return_params=True)[1]["coef"] != 0
               for i in range(Pn) for j in range(Pn) if i != j]
        C0 = np.cov(X, ddof=0)
        extra["n_active"] = int(np.sum(act))
        extra["n_abs_cov0_gt_alpha_l1"] = int(np.sum(np.abs(offdiag(C0)) > l1))
        extra["n_offdiag"] = len(act)
    detail = f"{FN_SRC[name]} on {run['path']} data ({VERS})"
    notes = "M[i,j] = f(predictor x_i, target x_j); diag NaN"
    if uid == "misc.lmfit.Ridge" and run.get("tag") == "closedform":
        R = corr_matrix(X)
        M = ridge_closed_form(R, T)
        np.fill_diagonal(M, np.nan)
        detail = f"{SELF}:ridge_closed_form ((T-1)/T)(1 - r^2 (T^2-1)/T^2), r = numpy.corrcoef ({VERS})"
        kind = "analytic"
    elif uid == "misc.lmfit.Lasso" and run["path"] != "kernel":
        # analytic: |r|(T-1)/T < 1 = alpha  ->  w = 0  ->  MSE = mean(z_j^2) = (T-1)/T
        lasso = pair_matrix(X, lambda a, b: K.lasso_1d_mse(a, b, 1.0))
        extra["lasso_1d_mse_vs_exit_max_abs"] = float(np.nanmax(np.abs(lasso - M)))
        M = np.full_like(M, c)
        np.fill_diagonal(M, np.nan)
        detail = f"analytic (T-1)/T (KKT |r|(T-1)/T <= alpha=1); cross-checked with refs/_kernels_basic_misc.py:lasso_1d_mse ({VERS})"
        kind = "analytic"
    else:
        kind = "independent-impl"
    return dict(value=M, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind=kind,
                reference_detail=detail, notes=notes, **extra)


def h_pure(run, X, P):
    """ElasticNet against the pure closed-form minimiser (no tol exit): fails exactly where pairs
    fall in the band 0.5T/(T-1) < |r| <= 0.5T/((T-1)(1-sqrt(2e-4)))."""
    T = run["T"]
    M = pair_matrix(X, lambda a, b: K.elasticnet_1d_mse(a, b, 1.0, 0.5))
    R = corr_matrix(X)
    ar = np.abs(R)
    lim = 0.5 * T / ((T - 1) * (1 - np.sqrt(2e-4)))
    thr = 0.5 * T / (T - 1)
    band = (ar > thr) & (ar <= lim)
    np.fill_diagonal(band, False)
    extra = dict(n_band_pairs=int(band.sum()))
    if P is not None:
        d = np.abs(P - M)
        extra["max_abs_in_band"] = float(d[band].max()) if band.any() else 0.0
        extra["max_abs_outside_band"] = float(np.nanmax(np.where(band, 0.0, d)))
    if band.any():
        k = np.argwhere(band)[0]
        extra["band_example"] = f"pair ({k[0]},{k[1]}) r={R[k[0], k[1]]:.6f} pure={M[k[0], k[1]]:.9f}"
    return dict(value=M, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"refs/_kernels_basic_misc.py:elasticnet_1d_mse (exact minimiser, no solver exit) ({VERS})",
                notes="pure closed form; a mismatch is expected only for band pairs (sklearn tol exit)", **extra)


def h_sgd_ols(run, X, P):
    T = run["T"]
    M = pair_matrix(X, K.ols_mse)
    extra = {}
    if P is not None:
        ex = offdiag(P - M)
        extra.update(excess_max=float(ex.max()), excess_min=float(ex.min()),
                     excess_median=float(np.median(ex)),
                     n_above_trivial=int(np.sum(offdiag(P) > (T - 1) / T)),
                     n_offdiag=int(ex.size), asym_max=asym(P))
    return dict(value=M, criterion="tolerance", rtol=0.0, atol=1.0 / (10 * T), reference_kind="independent-impl",
                reference_detail=f"refs/_kernels_basic_misc.py:ols_mse on {run['path']} data ({VERS})",
                notes="documented 'ordinary least squares fit'; atol = 1/(10T) = one tenth of the null-pair mean r^2",
                **extra)


def h_sgd_opt(run, X, P):
    T = run["T"]
    M = pair_matrix(X, K.ols_mse)
    extra = {}
    flag = 0
    if P is not None:
        ex = offdiag(P - M)
        flag = int(ex.min() >= -1e-12)
        extra.update(excess_max=float(ex.max()), excess_min=float(ex.min()),
                     n_above_trivial=int(np.sum(offdiag(P) > (T - 1) / T)),
                     max_value=float(offdiag(P).max()), trivial_bound=(T - 1) / T, asym_max=asym(P))
    return dict(value=M, criterion="optimality", pass_flag=flag, reference_kind="independent-impl",
                reference_detail=f"refs/_kernels_basic_misc.py:ols_mse; pass iff A >= OLS - 1e-12 ({VERS})",
                notes="OLS is the in-sample minimiser of the squared loss; excess = SGD - OLS", **extra)


def h_br_evidence(run, X, P):
    M = np.full((X.shape[0],) * 2, np.nan)
    edges = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i != j:
                M[i, j], info = br_evidence_mse(X[i], X[j])
                edges += info["at_grid_edge"]
    FP = pair_matrix(X, lambda a, b: K.bayesian_ridge_mse(a, b, max_iter=100000, tol=1e-14))
    extra = dict(grid_edge_pairs=int(edges), evidence_vs_fixedpoint_max_abs=float(np.nanmax(np.abs(M - FP))))
    flag = 0
    if P is not None:
        d = np.abs(offdiag(P - M))
        flag = int(d.max() <= 1e-4)
        sd = offdiag(P - M)
        extra.update(max_abs_vs_evidence_opt=float(d.max()), max_abs_vs_converged_fixedpoint=float(np.nanmax(np.abs(P - FP))),
                     signed_min=float(sd.min()), signed_max=float(sd.max()))
    return dict(value=M, criterion="optimality", pass_flag=flag, reference_kind="independent-impl",
                reference_detail=f"{SELF}:br_evidence_mse (direct maximisation of log evidence + Gamma(1e-6) hyperpriors, grid + Nelder-Mead) ({VERS})",
                notes="pass iff |pyspi - MSE at evidence maximiser| <= 1e-4 (quantifies sklearn's tol=1e-3 early stop)",
                **extra)


def h_direction(run, X, P):
    """Raw-scale direction: is A[i, j] the fit with predictor i and target j?"""
    uid = run["unit_id"]
    name, f = UNIT_FN[uid]
    M = pair_matrix(X, f)
    extra = dict(n_diag_pairs=0)
    flag = 0
    if P is not None:
        idx = [(i, j) for i in range(M.shape[0]) for j in range(M.shape[0])
               if i != j and abs(np.log(M[i, j]) - np.log(M[j, i])) > np.log(1.01)]
        agree = [abs(np.log(P[i, j]) - np.log(M[i, j])) < abs(np.log(P[i, j]) - np.log(M[j, i])) for i, j in idx]
        extra.update(n_diag_pairs=len(idx), agreement=float(np.mean(agree)) if agree else float("nan"),
                     max_rel_to_oriented=float(np.nanmax(np.abs(P - M) / M)),
                     max_rel_to_transposed=float(np.nanmax(np.abs(P - M.T) / M.T)),
                     asym_max=asym(P),
                     var0=json.dumps([float(v) for v in X.var(axis=1)]))
        flag = int(len(idx) > 0 and all(agree))
    return dict(value=M, criterion="qualitative", pass_flag=flag, reference_kind="independent-impl",
                reference_detail=f"{FN_SRC[name]} on raw data; pass iff A[i,j] is closer (log scale) to f(x_i -> x_j) than to f(x_j -> x_i) on every pair where they differ by > 1% ({VERS})",
                notes="direction = which index is the target; on raw data the target variance sets the scale", **extra)


# ---- probe runs
def h_params(run, X, P):
    """sklearn fitted attributes exported by probes/lmfit_params.py vs the modelled values."""
    uid = run["unit_id"]
    Pn = X.shape[0]
    if uid in ("misc.lmfit.Lasso", "misc.lmfit.ElasticNet"):
        l1 = 1.0 if uid.endswith("Lasso") else 0.5
        V = np.full((Pn, Pn, 3), np.nan)
        for i in range(Pn):
            for j in range(Pn):
                if i != j:
                    _, p = enet_sklearn_tol(X[i], X[j], 1.0, l1, return_params=True)
                    V[i, j] = (p["n_iter"], p["coef"], p["intercept"])
        det = f"{SELF}:enet_sklearn_tol params (n_iter_, coef_, intercept_)"
        notes = "n_iter 0 = initial duality-gap exit (w = 0); 2 = one update to the exact minimiser plus the stopping check"
        atol = 1e-12
    else:  # BayesianRidge
        V = np.full((Pn, Pn, 3), np.nan)
        for i in range(Pn):
            for j in range(Pn):
                if i != j:
                    _, p = K.bayesian_ridge_mse(X[i], X[j], return_params=True)
                    V[i, j] = (p["n_iter"], p["alpha"], p["lambda_"])
        det = "refs/_kernels_basic_misc.py:bayesian_ridge_mse(return_params=True) (n_iter_, alpha_, lambda_)"
        notes = "same recursion, own code; checks the iteration count as well as the fixed point reached"
        atol = 1e-12
    extra = {}
    if P is not None and P.shape == V.shape:
        extra["max_abs_n_iter"] = float(np.nanmax(np.abs(P[..., 0] - V[..., 0])))
        extra["n_iter_values"] = json.dumps(sorted({int(v) for v in P[..., 0][~np.isnan(P[..., 0])]}))
    return dict(value=V, criterion="exact", rtol=1e-8, atol=atol, reference_kind="independent-impl",
                reference_detail=f"{det} on z-scored data ({VERS})", notes=notes, **extra)


def h_grid(run, x, P):
    uid, T = run["unit_id"], run["T"]
    kind = "enet" if uid.endswith("ElasticNet") else "br"
    rs = r_grid(kind)
    name, f = UNIT_FN[uid]
    V = np.full((rs.size, 3), np.nan)
    for k, r in enumerate(rs):
        Z = zscore(grid_pair(x, r), axis=1, ddof=1)  # pyspi's default Data() re-z-scores
        V[k] = (np.corrcoef(Z)[0, 1], f(Z[0], Z[1]), f(Z[1], Z[0]))
    extra = {}
    c = (T - 1) / T
    if P is not None:
        a = P[:, 1]
        extra["value_at_r1"] = float(a[-1])
        if kind == "enet":
            const = np.abs(a - c) <= 1e-12
            extra.update(last_r_constant=float(P[const, 0].max()), first_r_nonconstant=float(P[~const, 0].min()),
                         jump_at_switch=float(c - a[~const][np.argmin(P[~const, 0])]),
                         exact_threshold=0.5 * T / (T - 1), tol_exit_bound=0.5 * T / ((T - 1) * (1 - np.sqrt(2e-4))))
            pure = np.array([K.elasticnet_1d_mse(*zscore(grid_pair(x, r), axis=1, ddof=1), 1.0, 0.5) for r in rs])
            extra["pyspi_minus_pure_max_abs"] = float(np.max(np.abs(a - pure)))
        else:
            d = np.diff(a)
            nit = np.array([K.bayesian_ridge_mse(*zscore(grid_pair(x, r), axis=1, ddof=1), return_params=True)[1]["n_iter"] for r in rs])
            inc = np.flatnonzero(d > 1e-15)
            extra.update(monotone_nonincreasing=bool(np.all(d <= 1e-15)), max_increase=float(d.max()),
                         n_increases=int(inc.size),
                         increases_at_n_iter_change=int(np.sum(nit[inc] != nit[inc + 1])),
                         increase_r=json.dumps([round(float(rs[k]), 4) for k in inc]),
                         n_iter_range=json.dumps([int(nit.min()), int(nit.max())]),
                         above_ols=bool(np.all(a >= np.array([K.ols_mse(*zscore(grid_pair(x, r), axis=1, ddof=1)) for r in rs]) - 1e-12)))
    return dict(value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{FN_SRC[name]} on a synthetic r grid (u = z(x0), y = r u + sqrt(1-r^2) z(x1 orth u)) ({VERS})",
                notes="columns: sample r, A[0,1] (predict y from u), A[1,0]", **extra)


def h_determinism(run, x, P):
    flag, extra = 0, {}
    if P is not None:
        d = max(float(np.nanmax(np.abs(P[k] - P[0]))) for k in range(1, P.shape[0]))
        flag = int(d == 0.0)
        extra["max_abs_between_global_seeds"] = d
    return dict(value=np.array(0.0), criterion="qualitative", pass_flag=flag, reference_kind="same-lineage",
                reference_detail="pyspi vs pyspi self-consistency: LinearModel('SGDRegressor') outputs after three global seeds compared with each other "
                                 "(expected equal because misc.py:121-122 passes random_state=42); value is a placeholder 0.0, pass_flag decided from pyspi alone",
                notes="probe recomputes LinearModel('SGDRegressor') after np.random/random seeds 42, 0, 12345; same-lineage self-consistency, not a correctness reference", **extra)


def h_seedsweep(run, x, P):
    X = prep(x, "pipeline")
    M = pair_matrix(X, K.ols_mse)
    flag, extra = 0, {}
    if P is not None:
        ex = P - M[None]
        off = ~np.eye(M.shape[0], dtype=bool)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            spread = (np.nanmax(P, axis=0) - np.nanmin(P, axis=0))[off]
        flag = int(np.nanmin(ex) >= -1e-12)
        extra.update(n_seeds=int(P.shape[0]), seed_spread_max=float(spread.max()), seed_spread_median=float(np.median(spread)),
                     excess_min=float(np.nanmin(ex)), excess_max=float(np.nanmax(ex)),
                     null_r2_scale=1.0 / run["T"])
    return dict(value=np.array(0.0), criterion="qualitative", pass_flag=flag, reference_kind="same-lineage",
                reference_detail="probes/lmfit_sgd.py:seed_sweep (sklearn SGDRegressor random_state 0..19, same library) vs refs/_kernels_basic_misc.py:ols_mse",
                notes="measures seed sensitivity only; pass iff every seed's MSE >= OLS - 1e-12", **extra)


HANDLERS = {"scratch": h_exact, "closedform": h_exact, "recursion": h_exact, "exitmodel": h_exact,
            "constant": h_exact, "detrend": h_exact, "raw": h_exact, "pure": h_pure,
            "ols": h_sgd_ols, "optimality": h_sgd_opt, "evidence": h_br_evidence, "direction": h_direction,
            "params": h_params, "grid": h_grid, "determinism": h_determinism, "seedsweep": h_seedsweep}
PROBE_TAGS = {"grid", "determinism", "seedsweep"}  # handlers that take the raw x


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    runs = json.loads(MANIFEST.read_text())["runs"]
    n = 0
    for run in runs:
        rid = run["run_id"]
        if only and only not in rid:
            continue
        tag = rid.split("|")[-1]
        run = dict(run, tag=tag)
        x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
        if run.get("processes") is not None:
            x = x[list(run["processes"])]
        P = load_pyspi(rid)
        if tag in PROBE_TAGS:
            out = HANDLERS[tag](run, x, P)
        else:
            X = prep(x, "pipeline" if tag == "params" else run["path"])
            out = HANDLERS[tag](run, X, P)
        write(rid, **out)
        n += 1
    print(f"refs/lmfit.py: wrote {n} reference files")


if __name__ == "__main__":
    main()
