#!/usr/bin/env python3
"""References for the phi family (COMPARISON venv: numpy/scipy only; no pyspi, no phi_native).

Units: infotheory.phi.{star,Geo}_norm-{0,1} (IntegratedInformation, partition [1, 2], tau = 1).
For every run of ``audit/checks/manifest/phi.json`` this script writes ``out/ref/<run_id>.npz``
in the compare.py contract.  The run_id tag (last ``|`` field) selects the check.

Independent pieces written here (no import of pyspi or of phi_native):

* ``cov3``        past/present covariances exactly as the Gaussian model needs them (past window
                  x[:, :T-1], present x[:, 1:], each centred by its own mean, denominator T-2).
* ``StarRef``     Phi* = I - max_beta tilde-I(beta) (Oizumi et al. 2016 PLoS CB, Eqs. 8, 9, 20 and the
                  Gaussian Methods section), derived from scratch: E_x[q(y|x)^beta] is a Gaussian integral
                  (completing the square), giving
                    tilde-I(beta) = 1/2 log|Sigma_X| + 1/2 log|Q^-1| + beta/2 tr(Sigma_Y D^-1)
                                    - beta^2/2 tr(Sigma_Y G Q G') - beta/2 tr(D^-1 S),
                  Q^-1 = Sigma_X^-1 + beta M, M = A_D' D^-1 A_D, G = D^-1 A_D, S = E[(y - A_D x)(y - A_D x)'],
                  A_D = diag(a_k), D = diag(d_k) the per-channel regression of y_k on x_k; and its derivative
                    d tilde-I/d beta = 1/2 tr(Q M) + 1/2 tr(Sigma_Y D^-1) - beta tr(Sigma_Y G Q G')
                                       + beta^2/2 tr(Sigma_Y G Q M Q G') - 1/2 tr(D^-1 S).
                  tilde-I is concave in beta (minus a log-partition function plus a linear term), so the
                  derivative is decreasing and its root, found by scipy.optimize.brentq on [0, hi], is the
                  unique maximiser.  ``matlab_IsId`` is a line-by-line transcription of
                  PhiToolbox utility/I_s_I_s_d.m (objective and analytic derivative) used to cross-check the
                  derivation (agreement reported as ref_matlab_transcription_maxabs).
* ``geo_ref``     Phi_G = min over diagonal A' of 1/2 log|Sigma_E + (A - A') Sigma_X (A - A')'| / |Sigma_E|
                  (Oizumi, Tsuchiya & Amari 2016 PNAS, Eqs. 17-21) by a vectorised 2-D grid (81 x 81 over
                  [-R, R]^2) followed by BFGS with the analytic gradient (Eq. 19 residual
                  -(Sigma_E'^-1 (A - A') Sigma_X)_kk) from the best grid points, diag(A) and 0, then a
                  Nelder-Mead polish.
* normalisation   norm-1 divides by (N_c - 1) * min_k 1/2 log(2 pi e Sigma_X[k, k]) (N_c = 2): the
                  PhiToolbox rule (phi_star_Gauss.m / phi_G_Gauss_AL.m normalization == 1).

The Octave reference (octave/phi.m) runs pyspi's vendored PhiToolbox copy (git 76ab2bb^, extracted to
the session scratchpad; it is not in the scratch-probe archive, re-extract it with
``git archive 76ab2bb^ pyspi/lib/PhiToolbox | tar -x -C <dir>``, see audit/PATHS.md) on the SAME input
array pyspi received (the ``input`` stored by
pyspi_outputs.py), with Z = [1 2], params.tau = 1, options.type_of_dist = 'Gauss',
options.type_of_phi = 'star'|'Geo', options.normalization = 0|1 (the struct the pre-eb9d471 pyspi
wrapper passed through oct2py).

Tags: indep, ratio, pop, pop-details, nullpop, mc, details, geo-details, conv, bounds, null, sym, mirror,
octave, degenerate, exactdelay, invariance, negative, detrend (see the chk_* functions).

Run under the comparison venv:  python audit/checks/refs/phi.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.signal

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_infotheory as K  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "phi"
OUT = CHECKS / "out" / "ref"
PYS = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / FAMILY
OCTAVE_M = CHECKS / "octave" / "phi.m"
TOOLBOX = Path("/tmp/claude-1000/-media-dan-Data-git-pyspi/5085b751-c4f7-4c35-9512-0b49510da359/"
               "scratchpad/phitoolbox/pyspi/lib/PhiToolbox")
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"
ME = "audit/checks/refs/phi.py"
LOG2PIE = math.log(2 * math.pi * math.e)
R_MC = 300


# ============================================================================ helpers
def zs(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(axis=-1, keepdims=True)
    s = np.sqrt(((x - m) ** 2).sum(axis=-1, keepdims=True) / (x.shape[-1] - 1))
    return (x - m) / s


def prep(x, path):
    x = np.asarray(x, dtype=np.float64)
    if path == "kernel":
        return x.copy()
    if path == "pipeline":
        return zs(x)
    if path == "pipeline-detrend":
        return zs(scipy.signal.detrend(x, axis=1))
    raise ValueError(path)


def cov3(z2, tau=1):
    """(Sigma_X, Sigma_XY, Sigma_Y): past = z[:, :T-tau], present = z[:, tau:], separate means,
    denominator T - tau - 1; Sigma_XY = E[past present'] (the orientation phi_native uses)."""
    z2 = np.asarray(z2, dtype=np.float64)
    T = z2.shape[1]
    Xp = z2[:, :T - tau]
    Yp = z2[:, tau:]
    Xp = Xp - Xp.mean(axis=1, keepdims=True)
    Yp = Yp - Yp.mean(axis=1, keepdims=True)
    den = T - tau - 1
    return Xp @ Xp.T / den, Xp @ Yp.T / den, Yp @ Yp.T / den


def slogdet(C):
    s, l = np.linalg.slogdet(C)
    return l if s > 0 else (-np.inf if s == 0 else np.nan)


def hmin(CX):
    """min_k Gaussian differential entropy of the PAST marginal k (nats); (N_c - 1) = 1."""
    return float(min(0.5 * (LOG2PIE + math.log(CX[k, k])) for k in range(CX.shape[0])))


def gauss_mi(CX, CXY, CY):
    """I(X_{t-1}; X_t) = 1/2 log |Sigma_Y| / |Sigma_Y - Sigma_YX Sigma_X^-1 Sigma_XY|."""
    SE = CY - CXY.T @ np.linalg.solve(CX, CXY)
    return 0.5 * (slogdet(CY) - slogdet(SE))


def gauss_si(CX, CXY, CY):
    """Stochastic interaction sum_k H(y_k|x_k) - H(Y|X) from the same covariances (nats)."""
    SE = CY - CXY.T @ np.linalg.solve(CX, CXY)
    d = np.diag(CY) - np.diag(CXY) ** 2 / np.diag(CX)
    return 0.5 * (np.log(d).sum() - slogdet(SE))


# ============================================================================ Phi* (from scratch)
class StarRef:
    def __init__(self, CX, CXY, CY):
        self.CX, self.CXY, self.CY = CX, CXY, CY
        n = CX.shape[0]
        self.n = n
        c = np.diag(CXY).copy()                     # Cov(x_k(t-1), x_k(t))
        a = c / np.diag(CX)                         # per-channel regression coefficient
        d = np.diag(CY) - a * c                     # per-channel residual variance
        self.AD, self.Dinv = np.diag(a), np.diag(1.0 / d)
        self.M = self.AD.T @ self.Dinv @ self.AD
        self.G = self.Dinv @ self.AD
        self.CXi = np.linalg.inv(CX)
        S = CY - self.AD @ CXY - CXY.T @ self.AD.T + self.AD @ CX @ self.AD.T
        self.trDS = float(np.trace(self.Dinv @ S))
        self.trYD = float(np.trace(CY @ self.Dinv))
        self.ldX = slogdet(CX)
        self.I = gauss_mi(CX, CXY, CY)

    def Is(self, beta):
        Qi = self.CXi + beta * self.M
        Q = np.linalg.inv(Qi)
        return (0.5 * self.ldX + 0.5 * slogdet(Qi) + 0.5 * beta * self.trYD
                - 0.5 * beta ** 2 * np.trace(self.CY @ self.G @ Q @ self.G.T) - 0.5 * beta * self.trDS)

    def dIs(self, beta):
        Q = np.linalg.inv(self.CXi + beta * self.M)
        GQ = self.G @ Q
        return (0.5 * np.trace(Q @ self.M) + 0.5 * self.trYD - beta * np.trace(self.CY @ GQ @ self.G.T)
                + 0.5 * beta ** 2 * np.trace(self.CY @ GQ @ self.M @ Q @ self.G.T) - 0.5 * self.trDS)

    def solve(self):
        f = self.dIs
        lo, hi = 0.0, 2.0
        if f(lo) <= 0:
            self.beta = 0.0
        else:
            while f(hi) > 0 and hi < 1e8:
                lo, hi = hi, hi * 2.0
            self.beta = scipy.optimize.brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
        self.Istar = self.Is(self.beta)
        self.phi = self.I - self.Istar
        return self.phi

    def solve_matlab(self):
        """brentq on the derivative transcribed from utility/I_s_I_s_d.m (second output), objective
        from the same transcription; returns (phi, beta)."""
        f = lambda b: matlab_IsId(b, self.CX, self.CXY, self.CY)[1]
        lo, hi = 0.0, 2.0
        if f(lo) <= 0:
            b = 0.0
        else:
            while f(hi) > 0 and hi < 1e8:
                lo, hi = hi, hi * 2.0
            b = scipy.optimize.brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
        return self.I - matlab_IsId(b, self.CX, self.CXY, self.CY)[0], b


def matlab_IsId(beta, CX, CXY, CY):
    """Transcription of PhiToolbox phi_star_Gauss.m (setup, atomic partition) + utility/I_s_I_s_d.m."""
    N = CX.shape[0]
    Cov_cond = lambda A, B, C: A - B @ np.linalg.solve(C, B.T)          # Cov_cond.m: X - XY/Y*XY'
    X_D = np.zeros((N, N)); YX_D = np.zeros((N, N)); C_D_cond = np.zeros((N, N))
    for k in range(N):
        M = [k]
        X_D[np.ix_(M, M)] = CX[np.ix_(M, M)]
        YX_D[np.ix_(M, M)] = CXY[np.ix_(M, M)].T
        C_D_cond[np.ix_(M, M)] = Cov_cond(CY[np.ix_(M, M)], CXY[np.ix_(M, M)].T, CX[np.ix_(M, M)])
    Cov_X_inv = np.linalg.inv(CX)
    mr = lambda A, B: np.linalg.solve(B.T, A.T).T                         # A / B
    C_D_beta1_inv = mr(mr(np.linalg.solve(X_D, YX_D.T), C_D_cond) @ YX_D, X_D)
    S_left = mr(np.linalg.solve(C_D_cond.T, YX_D), X_D)
    S_right = mr(np.linalg.solve(X_D, YX_D.T), C_D_cond)
    c = 0.5 * N
    Q_inv = Cov_X_inv + beta * C_D_beta1_inv
    norm_t = 0.5 * slogdet(Q_inv) + 0.5 * slogdet(CX)
    R = beta * np.linalg.inv(C_D_cond) - beta ** 2 * mr(S_left, Q_inv) @ S_right
    I_s = norm_t + 0.5 * np.trace(CY @ R) - beta * c
    Q_d = -mr(np.linalg.solve(Q_inv, C_D_beta1_inv), Q_inv)
    R_d = np.linalg.inv(C_D_cond) - mr(beta * S_left * 2, Q_inv) @ S_right - beta * S_left * beta @ Q_d @ S_right
    I_s_d = 0.5 * (-np.trace(Q_inv @ Q_d) + np.trace(CY @ R_d)) - c
    return I_s, I_s_d


# ============================================================================ Phi_G (from scratch)
def geo_parts(CX, CXY, CY):
    A = np.linalg.solve(CX, CXY).T
    SE = CY - CXY.T @ np.linalg.solve(CX, CXY)
    return A, 0.5 * (SE + SE.T)


def geo_f(a, A, SE, CX):
    Dl = A - np.diag(a)
    return 0.5 * (slogdet(SE + Dl @ CX @ Dl.T) - slogdet(SE))


def geo_grad(a, A, SE, CX):
    Dl = A - np.diag(a)
    S = SE + Dl @ CX @ Dl.T
    return -np.diag(np.linalg.solve(S, Dl @ CX))


def geo_ref(CX, CXY, CY, starts_only=False):
    A, SE = geo_parts(CX, CXY, CY)
    f = lambda a: geo_f(a, A, SE, CX)
    g = lambda a: geo_grad(a, A, SE, CX)
    starts = [np.diag(A).copy(), np.zeros(2)]
    if not starts_only:
        R = max(20.0, 3.0 * float(np.abs(A).max()))
        grid = np.linspace(-R, R, 81)
        a1, a2 = np.meshgrid(grid, grid, indexing="ij")
        d11, d12, d21, d22 = A[0, 0] - a1, A[0, 1], A[1, 0], A[1, 1] - a2
        # S = SE + Dl CX Dl' for every grid point (2 x 2, explicit)
        r0 = (d11 * CX[0, 0] + d12 * CX[1, 0], d11 * CX[0, 1] + d12 * CX[1, 1])
        r1 = (d21 * CX[0, 0] + d22 * CX[1, 0], d21 * CX[0, 1] + d22 * CX[1, 1])
        s00 = SE[0, 0] + r0[0] * d11 + r0[1] * d12
        s01 = SE[0, 1] + r0[0] * d21 + r0[1] * d22
        s11 = SE[1, 1] + r1[0] * d21 + r1[1] * d22
        F = np.log(s00 * s11 - s01 ** 2)
        for idx in np.argsort(F, axis=None)[:4]:
            i, j = np.unravel_index(idx, F.shape)
            starts.append(np.array([grid[i], grid[j]]))
    best = None
    for s in starts:
        r = scipy.optimize.minimize(f, s, jac=g, method="BFGS", options=dict(gtol=1e-12, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    if not starts_only:
        r = scipy.optimize.minimize(f, best.x, method="Nelder-Mead",
                                    options=dict(xatol=1e-12, fatol=1e-16, maxiter=20000))
        if r.fun < best.fun:
            best = r
        r = scipy.optimize.minimize(f, best.x, jac=g, method="BFGS", options=dict(gtol=1e-13, maxiter=2000))
        if r.fun < best.fun:
            best = r
    return float(best.fun), np.asarray(best.x), float(np.abs(g(best.x)).max())


# ============================================================================ pair -> matrix
def unit_of(uid):
    tail = uid.split(".")[-1]
    p, n = tail.split("_norm-")
    return p, int(n)


def ref_pair(z2, phitype, norm, details=False):
    CX, CXY, CY = cov3(z2)
    if phitype == "star":
        r = StarRef(CX, CXY, CY)
        v = r.solve()
        extra = r
    else:
        v, a, gmax = geo_ref(CX, CXY, CY)
        extra = (a, gmax)
    if norm == 1:
        v = v / hmin(CX)
    return (v, extra) if details else v


def ref_matrix(z, phitype, norm):
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            A[i, j] = A[j, i] = ref_pair(z[[i, j]], phitype, norm)
    return A


def load_pyspi(rid, key="value"):
    with np.load(PYS / f"{rid}.npz") as zf:
        return np.asarray(zf[key])


def has_pyspi(rid):
    return (PYS / f"{rid}.npz").is_file()


def off(A):
    A = np.asarray(A)
    return A[~np.eye(A.shape[0], dtype=bool)]


def upper(A):
    A = np.asarray(A)
    return A[np.triu_indices(A.shape[0], 1)]


def symdiff(A):
    A = np.asarray(A, dtype=np.float64)
    d = np.abs(A - A.T)
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def rng_for(rid):
    return np.random.default_rng(zlib.crc32(rid.encode()) + 20261015)


def tol_for(phitype):
    # optimiser tolerance: star L-BFGS-B pgtol 1e-5 (phi error ~ g^2 / |I''|); Geo AL stop |d logdet| < 1e-10
    return 1e-7 if phitype == "star" else 1e-6


def save(rid, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in kw.items():
        clean[k] = np.asarray(v)
    np.savez(OUT / f"{rid}.npz", **clean)


def detail(rid, kind="indep"):
    return f"{ME}:chk_{kind}; {VERS}"


# ============================================================================ checks
def chk_indep(run, x):
    """Every pair (i < j, mirrored) vs the from-scratch optimum on the same preprocessing.
    star: brentq root of the analytic derivative; Geo: multi-start global minimum.  norm-1: / hmin."""
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    z = prep(x, run["path"])
    V = ref_matrix(z, phitype, norm)
    S = load_pyspi(rid).astype(np.float64)
    gap = upper(S) - upper(V)
    zin = load_pyspi(rid, "input").astype(np.float64)
    kw = {}
    if phitype == "Geo":
        star_rid = rid.replace("Geo_norm", "star_norm")
        if has_pyspi(star_rid):
            Ss = load_pyspi(star_rid).astype(np.float64)
            kw["n_pairs_geo_gt_star"] = int(np.sum(upper(S) > upper(Ss)))
    save(rid, value=V, criterion="tolerance", rtol=0.0, atol=tol_for(phitype),
         reference_kind="independent-impl",
         reference_detail=detail(rid) + (" (StarRef.solve: brentq on d tilde-I/d beta)" if phitype == "star"
                                         else " (geo_ref: grid + BFGS + Nelder-Mead global minimum)"),
         notes=f"pyspi - ref signed gap over {gap.size} pairs; atol = optimiser tolerance; norm-{norm}",
         signed_gap_min=float(np.nanmin(gap)), signed_gap_max=float(np.nanmax(gap)),
         n_gap_gt_1e8=int(np.sum(gap > 1e-8)), n_gap_gt_1e6=int(np.sum(gap > 1e-6)),
         input_vs_own_zscore_maxabs=float(np.abs(zin - z).max()),
         pyspi_sym_maxabs=symdiff(S), n_pairs=int(gap.size), **kw)


def chk_ratio(run, x):
    """norm-1 == pyspi norm-0 (same dataset/T/path) / min past-window Gaussian entropy (from scratch)."""
    rid = run["run_id"]
    phitype, _ = unit_of(run["unit_id"])
    rid0 = rid.replace("_norm-1|", "_norm-0|").replace("|ratio", "|indep")
    S0 = load_pyspi(rid0).astype(np.float64)
    z = prep(x, run["path"])
    P = z.shape[0]
    V = np.full((P, P), np.nan)
    H = []
    for i in range(P):
        for j in range(i + 1, P):
            h = hmin(cov3(z[[i, j]])[0])
            H.append(h)
            V[i, j] = S0[i, j] / h
            V[j, i] = S0[j, i] / h
    save(rid, value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="same-lineage",
         reference_detail=detail(rid, "ratio") + f"; numerator = pyspi {rid0}",
         notes="identity check of the normalisation step: divisor (N_c-1)*min_k 1/2 log(2 pi e var_k(past)), "
               "var with denominator T-2; numerator is pyspi's own norm-0 output (same-lineage)",
         divisor_min=float(min(H)), divisor_max=float(max(H)))


def var2_lag_cov(A1, A2, Sg):
    """Lag-0 and lag-1 autocovariances of x_t = A1 x_{t-1} + A2 x_{t-2} + e_t, Var(e) = Sg, from the
    companion form s_t = [x_t; x_{t-1}] (Var(s) = [[G0, G1], [G1', G0]], G1 = E[x_t x_{t-1}'])."""
    n = A1.shape[0]
    F = np.block([[A1, A2], [np.eye(n), np.zeros((n, n))]])
    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = Sg
    V = scipy.linalg.solve_discrete_lyapunov(F, Q)
    V = 0.5 * (V + V.T)
    return V[:n, :n].copy(), V[:n, n:].copy()


def pop_cov(dataset="gauss_ar1_pair"):
    """Population (Cov_X, Cov_XY = E[X(t-1) X(t)'], Cov_Y, truth dict) of the lag-1 Gaussian model."""
    d = make_data.load(dataset, 1024)
    if dataset == "gauss_ar1_pair":
        SX = np.asarray(d["Sigma_X"], dtype=np.float64)
        return SX, np.asarray(d["Sigma_XY"], dtype=np.float64).T.copy(), SX.copy(), d
    if dataset == "var2_bivariate":
        G0, G1 = var2_lag_cov(np.asarray(d["A1"], dtype=np.float64), np.asarray(d["A2"], dtype=np.float64),
                              np.asarray(d["Sigma"], dtype=np.float64))
        return G0, G1.T.copy(), G0.copy(), d
    raise ValueError(dataset)


def deriv_poly_degree(r):
    """Smallest polynomial degree that reproduces dI/dbeta * det(Q^-1)^2 on 40 beta points to 1e-9 rel."""
    bs = np.linspace(0.05, 4.0, 40)
    y = np.array([r.dIs(b) * np.linalg.det(r.CXi + b * r.M) ** 2 for b in bs])
    for deg in range(0, 9):
        c = np.polynomial.polynomial.polyfit(bs, y, deg)
        res = np.abs(np.polynomial.polynomial.polyval(bs, c) - y).max() / np.abs(y).max()
        if res < 1e-9:
            return deg, c
    return -1, None


def chk_pop(run, x, octave=None):
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    ds = run["dataset"]
    CX, CXY, CY, d = pop_cov(ds)
    h = hmin(CX)
    kw = {}
    if ds == "var2_bivariate":
        kw["Gamma0_vs_truth_maxabs"] = float(np.abs(CX - np.asarray(d["Gamma0"])).max())
    if phitype == "star":
        r = StarRef(CX, CXY, CY)
        r.solve()
        v = np.array([r.phi / h if norm else r.phi, r.I])
        if ds == "gauss_ar1_pair":
            kw["I_analytic_SigmaX_SigmaE"] = 0.5 * (slogdet(d["Sigma_X"]) - slogdet(d["Sigma_E"]))
        pm, bm = r.solve_matlab()
        kw.update(matlab_brentq_phi_minus_ref=float(pm - r.phi), matlab_brentq_beta=float(bm))
        deg, coef = deriv_poly_degree(r)
        bs = np.linspace(-0.0, 6.0, 121)
        dd = np.array([r.dIs(b) for b in bs])
        mt = max(max(abs(matlab_IsId(b, CX, CXY, CY)[0] - r.Is(b)), abs(matlab_IsId(b, CX, CXY, CY)[1] - r.dIs(b)))
                 for b in (0.25, 0.5, 1.0, 2.0, 4.0))
        bnd = scipy.optimize.minimize_scalar(lambda b: -r.Is(b), bounds=(-0.5, 50), method="bounded",
                                             options=dict(xatol=1e-12))
        kw.update(beta_ref=r.beta, I_ref=r.I,
                  dIs_at_beta1=r.dIs(1.0), deriv_numerator_poly_degree=deg,
                  deriv_decreasing_on_0_6=int(np.all(np.diff(dd) < 0)),
                  matlab_transcription_maxabs=mt, bounded_maximiser_phi_diff=float((r.I + bnd.fun) - r.phi))
        detail_s = f"StarRef.solve (brentq on d tilde-I/d beta) on the {ds} population lag-1 covariances"
    else:
        phi, a, gmax = geo_ref(CX, CXY, CY)
        v = np.array([phi / h if norm else phi])
        kw.update(a_ref_1=float(a[0]), a_ref_2=float(a[1]), eq19_residual_at_ref=gmax,
                  I_pop=gauss_mi(CX, CXY, CY), SI_pop=gauss_si(CX, CXY, CY))
        detail_s = f"geo_ref on the {ds} population lag-1 covariances"
    if norm:
        kw["divisor_hmin"] = h
    if octave is not None and rid in octave:
        kw["octave_phi"] = octave[rid]
        kw["octave_minus_ref"] = float(octave[rid] - v[0])
    save(rid, value=v, criterion="tolerance", rtol=0.0, atol=1e-8, reference_kind="independent-impl",
         reference_detail=f"{ME}:chk_pop ({detail_s}); {VERS}",
         notes="value = [phi, I] for star, [phi] for Geo; norm-1 divides phi by min past entropy", **kw)


def chk_nullpop(run, x):
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    A = np.diag([0.5, 0.3])
    vals = []
    for SE in (np.eye(2), np.array([[1.0, 0.5], [0.5, 1.0]])):
        SX = scipy.linalg.solve_discrete_lyapunov(A, SE)
        SX = 0.5 * (SX + SX.T)
        CX, CXY, CY = SX, (A @ SX).T.copy(), SX.copy()
        v = ref_pair_cov(CX, CXY, CY, phitype)
        vals.append(v / hmin(CX) if norm else v)
    save(rid, value=np.array(vals), criterion="tolerance", rtol=0.0, atol=1e-8,
         reference_kind="independent-impl", reference_detail=detail(rid, "nullpop"),
         notes="(a) A = diag(0.5, 0.3), Sigma_E = I: no coupling, Phi* = Phi_G = 0 analytically; "
               "(b) same A with Sigma_E = [[1, .5], [.5, 1]] (instantaneous noise correlation only)",
         ref_b=float(vals[1]))


def ref_pair_cov(CX, CXY, CY, phitype):
    if phitype == "star":
        return StarRef(CX, CXY, CY).solve()
    return geo_ref(CX, CXY, CY)[0]


def chk_mc(run, x):
    """Monte Carlo of tilde-I(beta) (Eq. 20 definition) at beta = 0.5, 1, 2 on the D15 population:
    joint draws (x, y), inner E_x'[q(y|x')^beta] by 2-D Gauss-Hermite quadrature over x' ~ N(0, Sigma_X)
    (60 x 60 nodes; 40 x 40 reported as a stability check), outer expectation by Monte Carlo."""
    rid = run["run_id"]
    CX, CXY, CY, d = pop_cov()
    A = np.asarray(d["A"], dtype=np.float64)
    SE = np.asarray(d["Sigma_E"], dtype=np.float64)
    rng = rng_for(rid)
    n = 200_000
    LX = np.linalg.cholesky(CX)
    LE = np.linalg.cholesky(SE)
    xs = rng.standard_normal((n, 2)) @ LX.T
    ys = xs @ A.T + rng.standard_normal((n, 2)) @ LE.T
    c = np.diag(CXY)
    a = c / np.diag(CX)
    dd = np.diag(CY) - a * c

    def logq(y, xx):          # log q(y|x) = sum_k log N(y_k; a_k x_k, d_k); broadcasting
        return (-0.5 * np.log(2 * np.pi * dd) - 0.5 * (y - a * xx) ** 2 / dd).sum(axis=-1)

    def inner(beta, m):
        g, w = np.polynomial.hermite_e.hermegauss(m)
        w = w / w.sum()
        G1, G2 = np.meshgrid(g, g, indexing="ij")
        nodes = np.stack([G1.ravel(), G2.ravel()], axis=1) @ LX.T          # x' = L g
        lw = np.log(np.outer(w, w).ravel())
        out = np.empty(n)
        for s in range(0, n, 4000):
            yy = ys[s:s + 4000, None, :]
            L = beta * logq(yy, nodes[None, :, :]) + lw[None, :]
            mx = L.max(axis=1, keepdims=True)
            out[s:s + 4000] = (mx[:, 0] + np.log(np.exp(L - mx).sum(axis=1)))
        return out

    est, se, stab = [], [], []
    for beta in (0.5, 1.0, 2.0):
        lin = inner(beta, 60)
        terms = -lin + beta * logq(ys, xs)
        est.append(terms.mean())
        se.append(terms.std(ddof=1) / math.sqrt(n))
        stab.append(float(np.abs(inner(beta, 40) - lin).max()))
    est, se = np.array(est), np.array(se)
    S = load_pyspi(rid).astype(np.float64)
    r = StarRef(CX, CXY, CY)
    closed = np.array([r.Is(b) for b in (0.5, 1.0, 2.0)])
    save(rid, value=est, se=1.5 * se, criterion="statistical", reference_kind="independent-impl",
         reference_detail=f"{ME}:chk_mc (Monte Carlo n = {n}, Gauss-Hermite 60x60 inner integral); {VERS}",
         notes="se key = 1.5 x Monte Carlo SE, so compare's 2-se rule is the 3-MC-SE rule of the proposed check; "
               "pyspi value = -i_s_objective(beta) captured from phi_star_gauss",
         mc_se_max=float(se.max()), gh40_vs_gh60_maxabs=max(stab),
         z_max=float(np.max(np.abs(S - est) / se)),
         closed_form_minus_pyspi_maxabs=float(np.abs(closed - S).max()))


def chk_details(run, x):
    """star: pyspi's beta and L-BFGS-B iteration count per pair vs the brentq optimum; also what an
    explicit-gradient L-BFGS-B (scipy defaults, jac = analytic derivative) does from beta = 1."""
    rid = run["run_id"]
    _, norm = unit_of(run["unit_id"])
    D = load_pyspi(rid).astype(np.float64)
    if run["path"] == "kernel" and run["run_id"].endswith("|pop-details"):
        CX, CXY, CY, _ = pop_cov(run["dataset"])
        covs = [((0, 0), (CX, CXY, CY))]
    else:
        z = prep(x, run["path"])
        P = z.shape[0]
        covs = [((i, j), cov3(z[[i, j]])) for i in range(P) for j in range(i + 1, P)]
    gaps, rel, nit0, gap_nit0, gap_nitpos, b_ref, b_py, g1, lb_nit0, lb_gap, mat_d = ([] for _ in range(11))
    for (i, j), (CX, CXY, CY) in covs:
        r = StarRef(CX, CXY, CY)
        phi = r.solve()
        h = hmin(CX) if norm else 1.0
        py_phi, py_I, py_beta, py_nit = D[i, j]
        gap = py_phi - phi / h
        gaps.append(gap)
        rel.append(abs(gap) / abs(phi / h) if phi != 0 else np.nan)
        nit0.append(py_nit == 0)
        (gap_nit0 if py_nit == 0 else gap_nitpos).append(gap)
        b_ref.append(r.beta)
        pm, bm = r.solve_matlab()
        mat_d.append(abs(pm - phi))
        b_py.append(py_beta)
        g1.append(abs(r.dIs(1.0)))
        lb = scipy.optimize.minimize(lambda b: -r.Is(float(b[0])), np.array([1.0]),
                                     jac=lambda b: np.array([-r.dIs(float(b[0]))]), method="L-BFGS-B")
        lb_nit0.append(lb.nit == 0)
        lb_gap.append((r.I + lb.fun) - phi)
    gaps = np.array(gaps)
    ok = bool(np.all(gaps >= -1e-12) and np.all(gaps <= 1e-7))
    rel = np.array(rel, dtype=float)
    save(rid, value=np.array(gaps), criterion="optimality", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"{ME}:chk_details (StarRef brentq optimum; scipy L-BFGS-B with analytic jac); {VERS}",
         notes="pass if every pyspi - ref gap lies in [-1e-12, 1e-7]; value = signed gaps (not compared)",
         n_pairs=len(gaps), gap_min=float(gaps.min()), gap_max=float(gaps.max()),
         rel_gap_max=float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0,
         n_pyspi_nit0=int(np.sum(nit0)),
         gap_max_nit0=float(max(gap_nit0)) if gap_nit0 else 0.0,
         gap_max_nitpos=float(max(gap_nitpos)) if gap_nitpos else 0.0,
         beta_ref_min=float(min(b_ref)), beta_ref_max=float(max(b_ref)),
         beta_pyspi_minus_ref_maxabs=float(np.max(np.abs(np.array(b_py) - np.array(b_ref)))),
         dIs_at_1_max_when_nit0=float(max([g for g, k in zip(g1, nit0) if k], default=0.0)),
         analytic_grad_lbfgsb_n_nit0=int(np.sum(lb_nit0)),
         analytic_grad_lbfgsb_gap_max=float(max(lb_gap)),
         matlab_brentq_vs_ref_maxabs=float(max(mat_d)))


def chk_geo_details(run, x):
    """Geo: Eq. 19 stationarity residual at pyspi's A_p, AL iterations and the gap to the global minimum."""
    rid = run["run_id"]
    _, norm = unit_of(run["unit_id"])
    D = load_pyspi(rid).astype(np.float64)
    z = prep(x, run["path"])
    P = z.shape[0]
    gaps, res, iters, offd, apart = [], [], [], [], []
    for i in range(P):
        for j in range(i + 1, P):
            CX, CXY, CY = cov3(z[[i, j]])
            phi, a, _ = geo_ref(CX, CXY, CY)
            h = hmin(CX) if norm else 1.0
            py_phi, a1, a2, od, it = D[i, j]
            A, SE = geo_parts(CX, CXY, CY)
            gaps.append(py_phi - phi / h)
            res.append(float(np.abs(geo_grad(np.array([a1, a2]), A, SE, CX)).max()))
            iters.append(it)
            offd.append(od)
            apart.append(float(np.abs(np.array([a1, a2]) - a).max()))
    gaps = np.array(gaps)
    ok = bool(np.all(gaps >= -1e-7) and np.all(gaps <= 1e-6))
    k = int(np.argmax(gaps))
    save(rid, value=gaps, criterion="optimality", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"{ME}:chk_geo_details (geo_ref global minimum; Eq. 19 residual); {VERS}",
         notes="pass if every pyspi - global-min gap lies in [-1e-7, 1e-6]; value = signed gaps (not compared)",
         n_pairs=len(gaps), gap_min=float(gaps.min()), gap_max=float(gaps.max()),
         n_gap_gt_1e6=int(np.sum(gaps > 1e-6)), n_gap_gt_1e8=int(np.sum(gaps > 1e-8)),
         eq19_residual_at_worst=res[k], eq19_residual_max=float(max(res)),
         al_iterations_at_worst=float(iters[k]), al_iterations_max=float(max(iters)),
         offdiag_Ap_max=float(max(offd)), a_distance_at_worst=apart[k])


def simulate_ar1(A, SE, T, rng, burn=1000):
    L = np.linalg.cholesky(SE)
    e = rng.standard_normal((T + burn, 2)) @ L.T
    x = np.zeros((T + burn, 2))
    for t in range(1, T + burn):
        x[t] = A @ x[t - 1] + e[t]
    return x[burn:].T


def simulate_var2(A1, A2, SE, T, rng, burn=1000):
    L = np.linalg.cholesky(SE)
    e = rng.standard_normal((T + burn, 2)) @ L.T
    x = np.zeros((T + burn, 2))
    for t in range(2, T + burn):
        x[t] = A1 @ x[t - 1] + A2 @ x[t - 2] + e[t]
    return x[burn:].T


def chk_conv(run, x):
    """Statistical: pyspi on the dataset draw vs the Monte Carlo distribution (R = 300 fresh draws of the
    generating model (D15 AR(1) or D5 VAR(2)) at the same T, z-scored, from-scratch estimator);
    the population value of the lag-1 Gaussian model is reported alongside."""
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    ds = run["dataset"]
    d = make_data.load(ds, run["T"])
    if ds == "gauss_ar1_pair":
        A = np.asarray(d["A"]); SE = np.asarray(d["Sigma_E"])
        sim = lambda rng: simulate_ar1(A, SE, run["T"], rng)
    else:
        A1 = np.asarray(d["A1"]); A2 = np.asarray(d["A2"]); SE = np.asarray(d["Sigma"])
        sim = lambda rng: simulate_var2(A1, A2, SE, run["T"], rng)
    CXp, CXYp, CYp, _ = pop_cov(ds)
    pop = ref_pair_cov(CXp, CXYp, CYp, phitype)
    if norm:
        pop = pop / (0.5 * LOG2PIE)          # pipeline z-scores: unit-variance past marginals
    rng = rng_for(rid)
    vals = []
    for _ in range(R_MC):
        z = zs(sim(rng))
        CX, CXY, CY = cov3(z)
        v = StarRef(CX, CXY, CY).solve() if phitype == "star" else geo_ref(CX, CXY, CY, starts_only=True)[0]
        vals.append(v / hmin(CX) if norm else v)
    vals = np.array(vals)
    S = load_pyspi(rid).astype(np.float64)
    V = np.full((2, 2), np.nan); V[0, 1] = V[1, 0] = vals.mean()
    SEm = np.full((2, 2), np.nan); SEm[0, 1] = SEm[1, 0] = vals.std(ddof=1)
    save(rid, value=V, se=SEm, criterion="statistical", reference_kind="independent-impl",
         reference_detail=f"{ME}:chk_conv (R = {R_MC} draws of the {ds} generating model, burn-in 1000); {VERS}",
         notes="value = Monte Carlo mean of the from-scratch estimator at this T; se = Monte Carlo SD",
         population=pop, pyspi_minus_population=float(S[0, 1] - pop), mc_mean=float(vals.mean()),
         mc_sd=float(vals.std(ddof=1)), mc_mean_minus_population=float(vals.mean() - pop))


def chk_bounds(run, x):
    """0 <= Phi* <= I(X_{t-1}; X_t); 0 <= Phi_G <= min(I, SI) (all from the same sample covariances)."""
    rid = run["run_id"]
    phitype, _ = unit_of(run["unit_id"])
    S = load_pyspi(rid).astype(np.float64)
    z = prep(x, run["path"])
    P = z.shape[0]
    lows, highs, V = [], [], np.full((P, P), np.nan)
    over_I, over_SI = [], []
    for i in range(P):
        for j in range(i + 1, P):
            CX, CXY, CY = cov3(z[[i, j]])
            I = gauss_mi(CX, CXY, CY)
            ub = I if phitype == "star" else min(I, gauss_si(CX, CXY, CY))
            V[i, j] = V[j, i] = ub
            lows.append(S[i, j])
            highs.append(S[i, j] - ub)
            over_I.append(S[i, j] - I)
            if phitype == "Geo":
                over_SI.append(S[i, j] - gauss_si(CX, CXY, CY))
    eps = 1e-12 if phitype == "star" else 1e-9
    ok = bool(min(lows) >= -eps and max(highs) <= eps)
    kw = {"max_minus_SI": float(max(over_SI))} if over_SI else {}
    save(rid, value=V, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{ME}:chk_bounds (gauss_mi, gauss_si from the same covariances); {VERS}",
         notes=f"pass if every pair has -{eps:g} <= phi <= upper bound + {eps:g}; value = upper bound",
         phi_min=float(min(lows)), max_minus_upper=float(max(highs)), max_minus_I=float(max(over_I)), **kw)


def chk_null(run, x):
    """iid_gaussian: population Phi = 0; mean over the 10 pairs vs the Monte Carlo distribution of that
    mean on fresh iid N(0, 1) data at the same T (from-scratch estimator)."""
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    S = load_pyspi(rid).astype(np.float64)
    m = float(np.nanmean(upper(S)))
    rng = rng_for(rid)
    P, T = x.shape
    means = []
    for _ in range(R_MC):
        z = zs(rng.standard_normal((P, T)))
        vals = []
        for i in range(P):
            for j in range(i + 1, P):
                CX, CXY, CY = cov3(z[[i, j]])
                v = StarRef(CX, CXY, CY).solve() if phitype == "star" else geo_ref(CX, CXY, CY, starts_only=True)[0]
                vals.append(v / hmin(CX) if norm else v)
        means.append(np.mean(vals))
    means = np.array(means)
    mu, sd = float(means.mean()), float(means.std(ddof=1))
    zval = (m - mu) / sd
    ok = bool(abs(zval) <= 2 and np.nanmin(upper(S)) >= -1e-12)
    save(rid, value=np.full(S.shape, mu), criterion="qualitative", pass_flag=int(ok),
         reference_kind="analytic", reference_detail=f"{ME}:chk_null (R = {R_MC} iid replicates); {VERS}",
         notes="population value 0; pass if |mean over pairs - MC mean| <= 2 MC SD and no value < -1e-12",
         mean_over_pairs=m, mc_mean=mu, mc_sd=sd, z=zval, T_times_mean=m * T,
         min_value=float(np.nanmin(upper(S))))


def chk_sym(run, x):
    """Unmirrored Directed.multivariate (both orders computed): A[i, j] vs A[j, i] (analytic symmetry of
    Phi under relabelling the two parts) within optimiser tolerance."""
    rid = run["run_id"]
    phitype, _ = unit_of(run["unit_id"])
    U = load_pyspi(rid).astype(np.float64)
    save(rid, value=U.T.copy(), criterion="tolerance", rtol=0.0, atol=tol_for(phitype),
         reference_kind="analytic", reference_detail=f"{ME}:chk_sym (transpose of pyspi's own unmirrored matrix)",
         notes="value = transpose of the unmirrored pyspi matrix; checks bivariate(i, j) == bivariate(j, i)",
         unmirrored_sym_maxabs=symdiff(U))


def chk_mirror(run, x):
    """The returned matrix: off-diagonal bitwise symmetric (Undirected mirror), so the known coupling
    direction (var2: x0 -> x1; delayed_noise: x0 leads) leaves no trace in A."""
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    S = load_pyspi(rid).astype(np.float64)
    o = ~np.eye(S.shape[0], dtype=bool)
    bit = bool(np.array_equal(S[o], S.T[o]))
    z = prep(x, run["path"])
    V = ref_matrix(z, phitype, norm)
    save(rid, value=V, criterion="qualitative", pass_flag=int(bit and np.array_equal(S, S.T, equal_nan=True)),
         reference_kind="analytic", reference_detail=f"{ME}:chk_mirror; {VERS}",
         notes="pass if A[i, j] == A[j, i] bitwise off the diagonal (np.array_equal(A, A.T, equal_nan=True))",
         sym_maxabs=symdiff(S), A01=float(S[0, 1]), A10=float(S[1, 0]),
         ref_vs_pyspi_maxabs=float(np.nanmax(np.abs(S - V))))


def chk_degenerate(run, x):
    """shared_envelope: x2 is a bit-identical copy of x0 (pair (0, 2)); canonical Gaussian I is +inf and
    Phi is undefined.  Pass = the documented behaviour (trace stage) is reproduced on the Calculator path."""
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    S = load_pyspi(rid).astype(np.float64)
    T = run["T"]
    v = S[0, 2]
    if phitype == "star" or T <= 1024:
        expected = "NaN"
        ok = bool(np.isnan(v))
    else:
        expected = "finite negative (about -ln(2)/2)"
        ok = bool(np.isfinite(v) and abs(v - (-math.log(2) / 2)) < 1e-3)
    z = prep(x, run["path"])
    gaps = []
    V = np.full(S.shape, np.nan)
    for i in range(4):
        for j in range(i + 1, 4):
            if (i, j) == (0, 2):
                continue
            V[i, j] = V[j, i] = ref_pair(z[[i, j]], phitype, norm)
            gaps.append(S[i, j] - V[i, j])
    save(rid, value=V, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{ME}:chk_degenerate; {VERS}",
         notes=f"pair (0,2) is a bit-identical copy; canonical +inf / undefined; expected (trace) {expected}",
         value_02=float(v), x0_x2_bitidentical_after_zscore=int(np.array_equal(z[0], z[2])),
         other_pairs_gap_maxabs=float(np.nanmax(np.abs(gaps))), expected=expected)


def chk_exactdelay(run, x):
    """Exact one-step-delay pair [u(t+1), u(t)]: Cov(Y|X) singular, canonical I = +inf."""
    rid = run["run_id"]
    S = load_pyspi(rid).astype(np.float64)
    u = np.asarray(x[0], dtype=np.float64)
    pair = np.stack([u[1:], u[:-1]])
    eig = {}
    for path in ("pipeline", "kernel"):
        CX, CXY, CY = cov3(prep(pair, path))
        SE = CY - CXY.T @ np.linalg.solve(CX, CXY)
        eig[path] = float(np.linalg.eigvalsh(0.5 * (SE + SE.T)).min())
    ok = bool(np.all(np.isfinite(S)))
    save(rid, value=S, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{ME}:chk_exactdelay; {VERS}",
         notes="value = [Re, Im] z-scored then [Re, Im] raw; canonical +inf; pass = documented finite output reproduced",
         zscored_re=float(S[0]), zscored_im=float(S[1]), raw_re=float(S[2]), raw_im=float(S[3]),
         min_eig_cond_cov_zscored=eig["pipeline"], min_eig_cond_cov_raw=eig["kernel"])


def chk_invariance(run, x):
    """scaled_offset (raw, kernel path) vs the from-scratch optimum on the unscaled bivariate_gaussian rows:
    Phi (norm-0) is invariant to per-channel affine maps."""
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    base = np.asarray(make_data.load("bivariate_gaussian", run["T"])["x"], dtype=np.float64)
    V = ref_matrix(base, phitype, norm)
    S = load_pyspi(rid).astype(np.float64)
    Vs = ref_matrix(prep(x, run["path"]), phitype, norm)       # same optimum computed on the scaled rows
    g = upper(S) - upper(V)
    save(rid, value=V, criterion="tolerance", rtol=0.0, atol=tol_for(phitype), reference_kind="analytic",
         reference_detail=f"{ME}:chk_invariance (from-scratch optimum on bivariate_gaussian raw rows); {VERS}",
         notes="analytic invariance under x -> s x + c per channel",
         signed_gap_min=float(g.min()), signed_gap_max=float(g.max()),
         ref_scaled_vs_ref_unscaled_maxabs=float(np.nanmax(np.abs(Vs - V))))


def chk_negative(run, x):
    """norm-1 on raw scaled_offset (normalise=False): divisor = min past entropy can be < 0."""
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    z = prep(x, run["path"])
    V = ref_matrix(z, phitype, norm)
    S = load_pyspi(rid).astype(np.float64)
    H = hmin(cov3(z[[0, 1]])[0])
    save(rid, value=V, criterion="tolerance", rtol=0.0, atol=tol_for(phitype), reference_kind="independent-impl",
         reference_detail=f"{ME}:chk_negative; {VERS}",
         notes="from-scratch optimum / min past entropy on the raw rows; negative where the divisor is negative",
         pyspi_01=float(S[0, 1]), divisor_01=H, n_negative=int(np.sum(upper(S) < 0)))


# ============================================================================ Octave
def run_octave(runs):
    """One octave-cli process for every octave-tag run (data jobs) and every pop run (covariance jobs)."""
    EXCH.mkdir(parents=True, exist_ok=True)
    jobs, where = [], {}
    jid = 0
    for run in runs:
        tag = run["run_id"].split("|")[-1]
        phitype, norm = unit_of(run["unit_id"])
        pt = 1 if phitype == "star" else 2
        if tag == "octave" and has_pyspi(run["run_id"]):
            z = load_pyspi(run["run_id"], "input").astype(np.float64)
            P = z.shape[0]
            for i in range(P):
                for j in range(i + 1, P):
                    jid += 1
                    np.savetxt(EXCH / f"job_{jid}.csv", z[[i, j]], delimiter=",", fmt="%.17g")
                    jobs.append([jid, 1, pt, norm, 1])
                    where[jid] = (run["run_id"], i, j)
        elif tag == "pop":
            CX, CXY, CY, _ = pop_cov(run["dataset"])
            jid += 1
            np.savetxt(EXCH / f"job_{jid}.csv", np.vstack([CX, CXY, CY]), delimiter=",", fmt="%.17g")
            jobs.append([jid, 2, pt, norm, 1])
            where[jid] = (run["run_id"], None, None)
    if not jobs:
        return {}, {}
    np.savetxt(EXCH / "jobs.csv", np.array(jobs), delimiter=",", fmt="%d")
    outf = EXCH / "out.csv"
    if outf.exists():
        outf.unlink()
    p = subprocess.run(["/usr/bin/octave-cli", "--no-gui", "-q", str(OCTAVE_M), str(TOOLBOX),
                        str(EXCH / "jobs.csv"), str(outf)], capture_output=True, text=True, timeout=1800)
    (EXCH / "octave_stdout.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"octave failed: {p.stderr[-2000:]}")
    res = np.loadtxt(outf, delimiter=",", ndmin=2)
    mats, popv = {}, {}
    for row in res:
        rid, i, j = where[int(row[0])]
        if i is None:
            popv[rid] = float(row[1])
            continue
        if rid not in mats:
            P = load_pyspi(rid, "input").shape[0]
            mats[rid] = {k: np.full((P, P), np.nan) for k in ("phi", "I", "beta", "phig", "betag", "offd")}
        for k, col in zip(("phi", "I", "beta", "phig", "betag", "offd"), range(1, 7)):
            mats[rid][k][i, j] = mats[rid][k][j, i] = row[col]
    return mats, popv


def chk_octave(run, x, mats):
    rid = run["run_id"]
    phitype, norm = unit_of(run["unit_id"])
    m = mats[rid]
    S = load_pyspi(rid).astype(np.float64)
    z = load_pyspi(rid, "input").astype(np.float64)
    R = ref_matrix(z, phitype, norm)
    kw = dict(octave_minus_indep_ref_max=float(np.nanmax(upper(m["phi"]) - upper(R))),
              octave_minus_indep_ref_min=float(np.nanmin(upper(m["phi"]) - upper(R))),
              pyspi_minus_octave_max=float(np.nanmax(upper(S) - upper(m["phi"]))),
              pyspi_minus_octave_min=float(np.nanmin(upper(S) - upper(m["phi"]))))
    if phitype == "star":
        kw.update(octave_gradobj_minus_indep_ref_maxabs=float(np.nanmax(np.abs(upper(m["phig"]) - upper(R)))),
                  octave_beta_min=float(np.nanmin(m["beta"])), octave_beta_max=float(np.nanmax(m["beta"])))
    else:
        kw.update(octave_offdiag_Ap_max=float(np.nanmax(m["offd"])))
    save(rid, value=m["phi"], criterion="tolerance", rtol=0.0, atol=1e-7 if phitype == "star" else 1e-8,
         reference_kind="same-lineage",
         reference_detail=("octave/phi.m: GNU Octave 8.4.0 running pyspi's vendored PhiToolbox "
                           "(git 76ab2bb^:pyspi/lib/PhiToolbox) phi_comp(X, [1 2], params.tau = 1, options."
                           f"type_of_phi = '{phitype}', type_of_dist = 'Gauss', normalization = {norm}) on "
                           "pyspi's own input array"),
         notes="authors' MATLAB code that phi_native ports, hence same-lineage under the independence rule; "
               "star: fminunc with "
               "finite differences (vendored), phig = fminunc with GradObj on (analytic gradient of I_s_I_s_d.m)",
         **kw)


# ============================================================================ main
HANDLERS = {
    "indep": chk_indep, "detrend": chk_indep, "ratio": chk_ratio, "nullpop": chk_nullpop, "mc": chk_mc,
    "details": chk_details, "pop-details": chk_details, "geo-details": chk_geo_details, "conv": chk_conv,
    "bounds": chk_bounds, "null": chk_null, "sym": chk_sym, "mirror": chk_mirror,
    "degenerate": chk_degenerate, "exactdelay": chk_exactdelay, "invariance": chk_invariance,
    "negative": chk_negative,
}


def main():
    m = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    runs = m["runs"]
    only = set(sys.argv[1:])
    if only:
        runs = [r for r in runs if any(o in r["run_id"] for o in only)]
    mats, popv = run_octave(runs)
    n_ok = n_err = 0
    for run in runs:
        rid = run["run_id"]
        tag = rid.split("|")[-1]
        if not has_pyspi(rid):
            print(f"SKIP {rid}: no pyspi output", file=sys.stderr)
            continue
        try:
            x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
            if run.get("processes") is not None:
                x = x[list(run["processes"])]
            if tag == "pop":
                chk_pop(run, x, popv)
            elif tag == "octave":
                chk_octave(run, x, mats)
            else:
                HANDLERS[tag](run, x)
            n_ok += 1
        except Exception as exc:
            import traceback
            n_err += 1
            print(f"ERROR {rid}: {exc!r}\n{traceback.format_exc()}", file=sys.stderr)
    print(f"phi refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
