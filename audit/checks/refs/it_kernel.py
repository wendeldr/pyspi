#!/usr/bin/env python3
"""References for the it_kernel family (COMPARISON venv: numpy/scipy only, no pyspi, no JIDT).

For every run of ``audit/checks/manifest/it_kernel.json`` this script writes
``audit/checks/out/ref/<run_id>.npz`` in the compare.py contract. The run_id tag (last ``|``
field) selects the check.

All values are in BITS (JIDT's kernel calculators report bits: entropy -<log2 p>, MI and TE
<log2 ratio>). Box-kernel conventions are those of refs/_kernels_infotheory.py (closed max-norm
box |dx_d| <= w_d, the query point counts itself, half-width w_d = width * N-1 SD of each column
passed to JIDT when NORMALISE is on, p = count / (N prod 2 w_d)); the tie-grid run re-establishes
the closed-box and self-inclusion rules against JIDT on data whose counts change exactly at
|dx| = w. Widths: 0.5 for je/ce/cce/xme/di/si, 0.25 for mi/tlmi/te (units.yaml identifiers).

numpy                 exact (rtol 1e-10, atol 1e-12): from-scratch box-kernel composition of
                      _kernels_infotheory.entropy_box_kernel / mi_box_kernel / te_box_kernel on the
                      same preprocessing (pipeline: z-score ddof=1; kernel: raw rows, JIDT widths
                      follow the SD; pipeline-detrend: scipy.signal.detrend then z-score).
truth*, null          statistical / qualitative: Monte Carlo mean and SD of the same box-kernel
                      estimator under the dataset's generating model at this T (fast KD-tree
                      counting, fast_* below), with the Gaussian population value and, where it
                      has a closed numerical form, the large-sample box-kernel limit stored as
                      extras. Box-kernel estimates of differential entropy/MI/TE are biased for
                      continuous data; the population comparison is convergence information only.
direction, sym-fresh, bound, degenerate, frame, ...   see the function of the same name.

Run under the comparison venv:  python audit/checks/refs/it_kernel.py [run_id substrings ...]
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.signal
from scipy.spatial import cKDTree
from scipy.stats import norm

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_infotheory as K  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "it_kernel"
OUT = CHECKS / "out" / "ref"
PYS = CHECKS / "out" / "pyspi"
LN2 = math.log(2.0)
HN = 0.5 * math.log2(2 * math.pi * math.e)            # differential entropy of N(0,1), bits = 2.0470956
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"
KREF = "refs/_kernels_infotheory.py"
W_CFG = {"je": 0.5, "ce": 0.5, "cce": 0.5, "xme1": 0.5, "xme10": 0.5, "di": 0.5, "si": 0.5,
         "mi": 0.25, "tlmi": 0.25, "te": 0.25}


# ============================================================================ helpers
def zs(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(axis=-1, keepdims=True)
    s = np.sqrt(((x - m) ** 2).sum(axis=-1, keepdims=True) / (x.shape[-1] - 1))
    return (x - m) / s


def prep(x, path):
    if path == "kernel":
        return np.asarray(x, dtype=np.float64)
    if path == "pipeline":
        return zs(x)
    if path == "pipeline-detrend":
        return zs(scipy.signal.detrend(np.asarray(x, dtype=np.float64), axis=1))
    raise ValueError(path)


def directed(fn, z):
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = fn(z[i], z[j])
    return A


def undirected(fn, z):
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            A[i, j] = A[j, i] = fn(z[i], z[j])
    return A


def load_pyspi(rid):
    with np.load(PYS / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def off(A):
    A = np.asarray(A)
    return A[~np.eye(A.shape[-1], dtype=bool)]


def rng_for(rid):
    return np.random.default_rng(zlib.crc32(rid.encode()) + 20261015)


# ============================================================================ estimator engines
# Exact engine: the shared kernels (chunked all-pairs comparisons, |dx| <= w, JIDT rule).
def ex_H(X, w):
    return K.entropy_box_kernel(K._as2d(X), w)


def ex_MI(a, b, w):
    return K.mi_box_kernel(a, b, w)


def ex_TE(a, b, w, k=1):
    return K.te_box_kernel(a, b, k=k, width=w)


# Fast engine for Monte Carlo only: KD-tree counts in coordinates scaled by the half-widths
# (|dx_d / w_d| <= 1, closed, self included). Identical counts to the exact engine except at
# exact |dx| == w ties, which have probability 0 for the continuous simulated data.
def _w(X, width):
    X = K._as2d(X)
    return width * np.sqrt(((X - X.mean(0)) ** 2).sum(0) / (X.shape[0] - 1))


def _cnt(X, w):
    Xs = K._as2d(X) / w
    return cKDTree(Xs).query_ball_point(Xs, r=1.0, p=np.inf, return_length=True).astype(np.float64)


def fa_H(X, w):
    X = K._as2d(X)
    if X.shape[1] == 0:
        return 0.0
    wd = _w(X, w)
    c = _cnt(X, wd)
    return float(-np.mean(np.log2(c / (X.shape[0] * np.prod(2 * wd)))))


def fa_MI(a, b, w):
    a, b = K._as2d(a), K._as2d(b)
    wa, wb = _w(a, w), _w(b, w)
    ca, cb = _cnt(a, wa), _cnt(b, wb)
    cj = _cnt(np.hstack([a, b]), np.concatenate([wa, wb]))
    return float(np.mean(np.log2(a.shape[0] * cj / (ca * cb))))


def fa_TE(a, b, w, k=1):
    assert k == 1
    a, b = np.ravel(a), np.ravel(b)
    past, nxt, src = b[:-1, None], b[1:, None], a[:-1, None]
    wp, ws = _w(past, w), _w(src, w)
    cp = _cnt(past, wp)
    cnp = _cnt(np.hstack([past, nxt]), np.concatenate([wp, wp]))
    cps = _cnt(np.hstack([past, src]), np.concatenate([wp, ws]))
    cnps = _cnt(np.hstack([past, nxt, src]), np.concatenate([wp, wp, ws]))
    return float(np.mean(np.log2((cnps / cps) / (cnp / cp))))


EXACT = {"H": ex_H, "MI": ex_MI, "TE": ex_TE}
FAST = {"H": fa_H, "MI": fa_MI, "TE": fa_TE}


# ============================================================================ pyspi compositions (bits)
def ce_(E, Y, X, w):
    """H(Y | X) = H([Y, X]) - H(X) (pyspi JIDTBase._compute_conditional_entropy, infotheory.py:180-193)."""
    return E["H"](np.hstack([K._as2d(Y), K._as2d(X)]), w) - E["H"](X, w)


def u_je(a, b, w, E):
    return E["H"](np.stack([a, b], axis=1), w)


def u_ce(a, b, w, E):            # A[i,j] = H(x_j | x_i)   (a = x_i)
    return ce_(E, b, a, w)


def u_mi(a, b, w, E):
    return E["MI"](a, b, w)


def u_tlmi(a, b, w, E):          # I(x_i(t-1); x_j(t)), pyspi src[:-1], targ[1:]
    return E["MI"](np.ravel(a)[:-1], np.ravel(b)[1:], w)


def u_te(a, b, w, E):            # TE(x_i -> x_j), k = l = 1
    return E["TE"](a, b, w)


def u_xme(a, b, w, E, k):
    Yf, Xp = K.xme_embedding(a, b, k)
    return ce_(E, Yf, Xp, w)


def xme_lags(a, b, w, E, lags):
    """H(y_t | x_{t-L}, L in lags) on t = max(lags) .. T-1 (the 'intended' variant with lags 1..k)."""
    a, b = np.ravel(a), np.ravel(b)
    T, m = a.size, max(lags)
    Xp = np.stack([a[m - L:T - L] for L in lags], axis=1)
    return ce_(E, b[m:, None], Xp, w)


def cce_terms(a, b, w, E, n):
    out = []
    for i in range(1, n + 1):
        Yf, Yp, Xp = K.cce_embedding(a, b, i)
        out.append(ce_(E, Yf, np.hstack([Yp, Xp]), w))
    return out


def u_cce(a, b, w, E, n=5):
    return float(sum(cce_terms(a, b, w, E, n)))


def di_rate_term(b, w, E, n):
    """sum_m H(Y^m)/m with Y^m = makeDelayEmbeddingVector(targ, m) (infotheory.py:493-499)."""
    return float(sum(E["H"](K.embed(np.ravel(b), m), w) / m for m in range(1, n + 1)))


def di_massey_target(b, w, E, n):
    """sum_m H(Y_t | Y_{t-1..t-m+1}) on the CCE sample of term m (Massey 1990 eq. 6 target part)."""
    s = 0.0
    for m in range(1, n + 1):
        Yf, Yp, _ = K.cce_embedding(b, b, m)
        s += ce_(E, Yf, Yp, w) if Yp.shape[1] else E["H"](Yf, w)
    return s


def u_di(a, b, w, E, n=5):
    return di_rate_term(b, w, E, n) - u_cce(a, b, w, E, n)


def u_di_massey(a, b, w, E, n=5):
    return di_massey_target(b, w, E, n) - u_cce(a, b, w, E, n)


def u_si(a, b, w, E, tau=1):
    x, y = K._as2d(a), K._as2d(b)
    xy = np.hstack([x, y])
    return (ce_(E, x[tau:], x[:-tau], w) + ce_(E, y[tau:], y[:-tau], w) - ce_(E, xy[tau:], xy[:-tau], w))


UFN = {
    "je": (undirected, u_je), "ce": (directed, u_ce), "mi": (undirected, u_mi), "tlmi": (directed, u_tlmi),
    "te": (directed, u_te), "xme1": (directed, lambda a, b, w, E: u_xme(a, b, w, E, 1)),
    "xme10": (directed, lambda a, b, w, E: u_xme(a, b, w, E, 10)), "cce": (directed, u_cce),
    "di": (directed, u_di), "si": (undirected, u_si),
}
UNIT_KEY = {
    "infotheory.je.kernel": "je", "infotheory.ce.kernel": "ce", "infotheory.mi.kernel": "mi",
    "infotheory.tlmi.kernel": "tlmi", "infotheory.te.kernel_k-1_kt-1_l-1_lt-1": "te",
    "infotheory.xme.kernel_k1": "xme1", "infotheory.xme.kernel_k10": "xme10", "infotheory.cce.kernel": "cce",
    "infotheory.di.kernel": "di", "infotheory.si.kernel": "si", "infotheory.est.kernel": "est",
}
KFUN = {"je": "entropy_box_kernel([z_i, z_j], 0.5)", "ce": "entropy_box_kernel([z_i,z_j]) - entropy_box_kernel(z_i) (0.5)",
        "mi": "mi_box_kernel(z_i, z_j, 0.25)", "tlmi": "mi_box_kernel(z_i[:-1], z_j[1:], 0.25)",
        "te": "te_box_kernel(z_i, z_j, k=1, width=0.25)", "xme1": "xme_embedding(k=1) + box H (0.5)",
        "xme10": "xme_embedding(k=10, lags 1..9) + box H (0.5)", "cce": "cce_embedding m=1..5 + box H (0.5)",
        "di": "sum_m H(embed(z_j, m))/m - cce (0.5)", "si": "box H(x_t|x_t-1)+H(y_t|y_t-1)-H(xy_t|xy_t-1) (0.5)"}


def stat_matrix(key, z, E=EXACT, w=None):
    kind, fn = UFN[key]
    w = W_CFG[key] if w is None else w
    return kind(lambda a, b: fn(a, b, w, E), z)


# ============================================================================ simulators (Monte Carlo)
def sim_var(A_list, Sigma, T, R, rng, burn=1000):
    A_list = [np.asarray(A, dtype=np.float64) for A in A_list]
    P = A_list[0].shape[0]
    L = np.linalg.cholesky(np.asarray(Sigma, dtype=np.float64))
    n = T + burn
    E = rng.standard_normal((n, R, P)) @ L.T
    X = np.zeros((n, R, P))
    for t in range(len(A_list), n):
        acc = E[t].copy()
        for l, A in enumerate(A_list, start=1):
            acc += X[t - l] @ A.T
        X[t] = acc
    return np.transpose(X[burn:], (1, 2, 0))           # (R, P, T)


def sim_pair(rho, T, R, rng):
    a = rng.standard_normal((R, T))
    b = rho * a + math.sqrt(1 - rho ** 2) * rng.standard_normal((R, T))
    return np.stack([a, b], axis=1)


def sim_delayed(T, R, rng, delays=(0, 1, 3, 5), noise=0.1):
    m = max(delays)
    base = rng.standard_normal((R, T + m))
    out = np.empty((R, len(delays), T))
    out[:, 0] = base[:, m:]
    for k, d in enumerate(delays[1:], start=1):
        out[:, k] = base[:, m - d:m - d + T] + noise * rng.standard_normal((R, T))
    return out


def mc(fn, sims):
    vals = np.array([fn(zs(s)) for s in sims])
    return vals.mean(axis=0), vals.std(axis=0, ddof=1)


def r_mc(key, T):
    """Replicates sized by cost (fast engine)."""
    heavy = key in ("cce", "di", "xme10")
    if T <= 256:
        return 200 if not heavy else 100
    return 100 if not heavy else 40


# ============================================================================ population values and box limits
def h_box1(eps):
    """Large-sample limit of the 1-D box-kernel entropy of N(0,1) with half-width eps (bits):
    -int phi(x) log2[(Phi(x+eps) - Phi(x-eps)) / (2 eps)] dx."""
    # symmetric integrand; P = sf(x - eps) - sf(x + eps) keeps precision in the upper tail
    f = lambda x: -norm.pdf(x) * math.log2((norm.sf(x - eps) - norm.sf(x + eps)) / (2 * eps))
    return 2.0 * scipy.integrate.quad(f, 0.0, 9.0, limit=200)[0]


def h_box2(rho, e1, e2, n_gh=60, n_gl=64):
    """Large-sample limit of the 2-D box-kernel entropy of a standard bivariate normal with
    correlation rho, half-widths e1, e2 (bits): -E log2[P_rect(x, y) / (4 e1 e2)],
    P_rect = int_{x-e1}^{x+e1} phi(u) [Phi((y+e2-rho u)/s) - Phi((y-e2-rho u)/s)] du, s = sqrt(1-rho^2).
    Outer expectation by probabilists' Gauss-Hermite, inner integral by Gauss-Legendre."""
    a, wa = np.polynomial.hermite_e.hermegauss(n_gh)
    wa = wa / math.sqrt(2 * math.pi)
    g, wg = np.polynomial.legendre.leggauss(n_gl)
    s = math.sqrt(1 - rho ** 2)
    X = a[:, None]
    Y = rho * a[:, None] + s * a[None, :]
    U = X[:, :, None] + e1 * g[None, None, :]                      # (n, 1, gl) broadcast over y
    Pr = (e1 * wg * norm.pdf(U) * (norm.cdf((Y[:, :, None] + e2 - rho * U) / s)
                                   - norm.cdf((Y[:, :, None] - e2 - rho * U) / s))).sum(-1)
    Pr = np.maximum(Pr, 1e-300)       # far-tail nodes (weights < 1e-30) can underflow to 0
    return float(-(wa[:, None] * wa[None, :] * np.log2(Pr / (4 * e1 * e2))).sum())


def h_box_uniform(eps):
    """Box limit for the z-scored uniform (support [-sqrt3, sqrt3]), half-width eps (bits)."""
    a = math.sqrt(3.0)
    f = lambda u: -(1 / (2 * a)) * math.log2(((min(u + eps, a) - max(u - eps, -a)) / (2 * a)) / (2 * eps))
    return scipy.integrate.quad(f, -a, a, points=[-a + eps, a - eps], limit=200)[0]


def var_gammas(A_list, Sigma, H):
    """Gamma(h) = E[x_{t+h} x_t^T], h = 0..H, for a stable VAR(p) (companion Lyapunov)."""
    A_list = [np.asarray(A, dtype=np.float64) for A in A_list]
    P, p = A_list[0].shape[0], len(A_list)
    F = np.zeros((P * p, P * p))
    F[:P] = np.hstack(A_list)
    if p > 1:
        F[P:, :-P] = np.eye(P * (p - 1))
    Q = np.zeros_like(F)
    Q[:P, :P] = Sigma
    S = scipy.linalg.solve_discrete_lyapunov(F, Q)
    G = [S[:P, :P]]
    for h in range(1, p):
        G.append(S[:P, h * P:(h + 1) * P])
    while len(G) <= H:
        h = len(G)
        G.append(sum(A_list[l] @ G[h - 1 - l] for l in range(p)))
    return G[:H + 1]


def block_cov(G, items):
    n = len(items)
    C = np.empty((n, n))
    for a, (ca, la) in enumerate(items):
        for b, (cb, lb) in enumerate(items):
            h = lb - la
            C[a, b] = G[h][ca, cb] if h >= 0 else G[-h][cb, ca]
    return C


def cond_var(C, target=0):
    rest = [i for i in range(C.shape[0]) if i != target]
    if not rest:
        return C[target, target]
    c = C[target, rest]
    return C[target, target] - c @ np.linalg.solve(C[np.ix_(rest, rest)], c)


def hb(v):
    """Gaussian differential entropy in bits for variance v."""
    return 0.5 * math.log2(2 * math.pi * math.e * v)


def pop_te1(G, src, tgt):
    v_r = cond_var(block_cov(G, [(tgt, 0), (tgt, 1)]))
    v_f = cond_var(block_cov(G, [(tgt, 0), (tgt, 1), (src, 1)]))
    return 0.5 * math.log2(v_r / v_f)


def pop_tlmi(G, src, tgt):
    r = G[1][tgt, src] / math.sqrt(G[0][src, src] * G[0][tgt, tgt])
    return -0.5 * math.log2(1 - r ** 2)


def pop_xme1(G, src, tgt):
    r = G[1][tgt, src] / math.sqrt(G[0][src, src] * G[0][tgt, tgt])
    return hb(1 - r ** 2)


def pop_cce(G, src, tgt, n=5):
    s = 0.0
    for m in range(1, n + 1):
        items = [(tgt, 0)] + [(tgt, L) for L in range(1, m)] + [(src, L) for L in range(0, m)]
        s += hb(cond_var(block_cov(G, items)) / G[0][tgt, tgt])
    return s


def pop_si(G, Sigma_E):
    vv = [cond_var(block_cov(G, [(v, 0), (v, 1)])) for v in range(2)]
    return 0.5 * math.log2(vv[0] * vv[1] / np.linalg.det(Sigma_E))


def pop_di_excess_univariate(gamma, n=5):
    """Population B = sum_m h(Y^m)/m - sum_m h(Y_m | Y^{m-1}) (bits) for a stationary Gaussian scalar."""
    b = 0.0
    for m in range(1, n + 1):
        Tm = scipy.linalg.toeplitz(gamma[:m])
        b += 0.5 * (m * math.log2(2 * math.pi * math.e) + np.linalg.slogdet(Tm)[1] / LN2) / m
        b -= hb(cond_var(Tm[::-1, ::-1]))
    return b


# ============================================================================ writers
def write(rid, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{rid}.npz", **{k: np.asarray(v) for k, v in kw.items()})


def write_exact(rid, value, kind, detail, notes="", rtol=1e-10, atol=1e-12, **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), criterion="exact", rtol=rtol, atol=atol,
          reference_kind=kind, reference_detail=detail, notes=notes, units="bits", **extra)


def write_stat(rid, value, se, detail, pop_matrix, S, R, notes="", **extra):
    pop = np.asarray(pop_matrix, dtype=np.float64)
    write(rid, value=np.asarray(value, dtype=np.float64), se=np.asarray(se, dtype=np.float64),
          criterion="statistical", reference_kind="analytic", reference_detail=detail, notes=notes, units="bits",
          pyspi_minus_population_maxabs=float(np.nanmax(np.abs(S - pop))),
          mc_mean_minus_population_maxabs=float(np.nanmax(np.abs(np.asarray(value) - pop))),
          mc_replicates=R, **extra)


def write_qual(rid, ok, detail, notes="", kind="analytic", **extra):
    write(rid, value=np.asarray(np.nan), criterion="qualitative", pass_flag=int(bool(ok)),
          reference_kind=kind, reference_detail=detail, notes=notes, units="bits", **extra)


PATH_NOTE = {"pipeline": "rows z-scored ddof=1", "kernel": "raw rows (Data normalise=False); JIDT widths follow each column's N-1 SD",
             "pipeline-detrend": "scipy.signal.detrend(axis=1) then z-score ddof=1"}


# ============================================================================ checks
def chk_numpy(run, key, x, ds, T, path):
    z = prep(x, path)
    V = stat_matrix(key, z)
    extra = {}
    if key == "xme10":                                         # intended 10-lag variant (off-by-one size)
        V10 = directed(lambda a, b: xme_lags(a, b, 0.5, EXACT, list(range(1, 11))), z)
        extra = dict(ten_lag_variant_maxabs_diff=float(np.nanmax(np.abs(V - V10))),
                     ten_lag_01=float(V10[0, 1]), ten_lag_10=float(V10[1, 0]))
    if key == "te":                                            # next-value box with its own SD (variant)
        def te_own(a, b):
            a, b = np.ravel(a), np.ravel(b)
            past, nxt, src = b[:-1], b[1:], a[:-1]
            wp, wn, ws = (0.25 * np.std(v, ddof=1) for v in (past, nxt, src))
            dp = np.abs(past[:, None] - past[None, :]) <= wp
            dn = np.abs(nxt[:, None] - nxt[None, :]) <= wn
            dsrc = np.abs(src[:, None] - src[None, :]) <= ws
            cp, cnp, cps, cnps = dp.sum(1), (dp & dn).sum(1), (dp & dsrc).sum(1), (dp & dn & dsrc).sum(1)
            return float(np.mean(np.log2((cnps / cps) / (cnp / cp))))
        Vo = directed(te_own, z)
        extra = dict(own_next_width_variant_maxabs_diff=float(np.nanmax(np.abs(V - Vo))))
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:{KFUN[key]}; {VERS}", notes=PATH_NOTE[path], **extra)


def chk_equals_d2(run, key, x, ds, T, path):
    d2 = make_data.load("bivariate_gaussian", T)["x"]
    z = zs(d2) if path == "pipeline" else np.asarray(d2, dtype=np.float64)
    V = stat_matrix(key, z)
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:{KFUN[key]} on the {'z-scored' if path == 'pipeline' else 'raw'} bivariate_gaussian rows (same draw); {VERS}",
                notes="scaled_offset = scale*D2 + offset; z-scoring (pipeline) or SD-scaled box widths (kernel path, MI) remove the affine map")


def chk_shift(run, key, x, ds, T, path):
    d2 = make_data.load("bivariate_gaussian", T)["x"]
    base = stat_matrix(key, zs(d2))
    sd = np.asarray(x, dtype=np.float64).std(axis=1, ddof=1)
    if key == "je":
        V = base + np.log2(sd[:, None] * sd[None, :])
        rule = "je_raw(i,j) = je(z-scored D2)(i,j) + log2(sd_i sd_j)"
    else:
        V = base + np.log2(sd[None, :] * np.ones_like(sd)[:, None])
        rule = "ce_raw(i->j) = ce(z-scored D2)(i->j) + log2(sd_j)"
    np.fill_diagonal(V, np.nan)
    write_exact(run["run_id"], V, "analytic",
                f"{rule}: box counts are invariant when widths follow the SD (JIDT NORMALISE=true), the volume prod 2*0.5*sd_d scales; numpy {KREF}:{KFUN[key]}; {VERS}",
                notes="kernel path on raw scaled_offset (NORMALISE=true inside JIDT)",
                **{f"log2_sd_x{p}": float(np.log2(sd[p])) for p in range(sd.size)})


def chk_bound(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    vals = off(S)
    hi = math.log2(T)
    ok = bool(np.isfinite(vals).all() and vals.min() >= -1e-12 and vals.max() <= hi + 1e-12)
    write_qual(run["run_id"], ok, f"volume-1 box on z-scored data (W-0.5): 0 <= {key} <= log2 T = {hi:.4f}",
               pyspi_min=float(vals.min()), pyspi_max=float(vals.max()), log2T=hi)


def _pop_pair(key, rho):
    hj = math.log2(2 * math.pi * math.e) + 0.5 * math.log2(1 - rho ** 2)
    if key == "je":
        return hj, h_box2(rho, 0.5, 0.5), "log2(2 pi e) + 0.5 log2(1-rho^2)"
    if key == "ce":
        return hj - HN, h_box2(rho, 0.5, 0.5) - h_box1(0.5), "0.5 log2(2 pi e (1-rho^2))"
    if key == "mi":
        return -0.5 * math.log2(1 - rho ** 2), 2 * h_box1(0.25) - h_box2(rho, 0.25, 0.25), "-0.5 log2(1-rho^2)"
    raise KeyError(key)


def chk_truth_pair(run, key, x, ds, T, path):
    p = run["processes"]
    pair = 0 if p == [0, 1] else 1
    rho = float(ds["rho"][pair])
    pv, bl, form = _pop_pair(key, rho)
    pop = np.full((2, 2), pv)
    box = np.full((2, 2), bl)
    np.fill_diagonal(pop, np.nan)
    np.fill_diagonal(box, np.nan)
    R = r_mc(key, T)
    sims = sim_pair(rho, T, R, rng_for(run["run_id"]))
    m, s = mc(lambda z: stat_matrix(key, z, FAST), sims)
    S = load_pyspi(run["run_id"])
    write_stat(run["run_id"], m, s,
               f"Monte Carlo mean/SD of the box-kernel estimator ({R} draws of N2(rho={rho}), fast KD-tree counts); population {form} (Cover & Thomas 2006 ch. 8); box-kernel large-sample limit by quadrature (h_box2)",
               pop, S, R, notes="convergence information: box-kernel estimates are biased for continuous data",
               rho=rho, population_value=pv, box_limit=bl, pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]),
               pyspi_minus_box_limit=float(np.nanmax(np.abs(S - box))), mc_mean_01=float(m[0, 1]), mc_sd_01=float(s[0, 1]))


def chk_truth_iid(run, key, x, ds, T, path):
    P = x.shape[0]
    popv, boxv, form = {
        "je": (2 * HN, 2 * h_box1(0.5), "2 h(N(0,1)) = log2(2 pi e)"),
        "xme1": (HN, h_box1(0.5), "h(N(0,1)): source independent of target"),
        "xme10": (HN, h_box1(0.5), "h(N(0,1)): source independent of target"),
        "cce": (5 * HN, 5 * h_box1(0.5), "5 h(N(0,1)) for independent white pairs (every term = h)"),
        "di": (0.0, 0.0, "0 (independent white pairs; both DI forms)"),
    }[key]
    pop = np.full((P, P), popv)
    np.fill_diagonal(pop, np.nan)
    R = r_mc(key, T)
    sims = rng_for(run["run_id"]).standard_normal((R, P, T))
    m, s = mc(lambda z: stat_matrix(key, z, FAST), sims)
    S = load_pyspi(run["run_id"])
    extra = {}
    if key in ("cce", "di"):
        z = zs(x)
        terms = cce_terms(z[0], z[1], 0.5, EXACT, 5)
        extra = {f"cce_term{m_}_01": float(t) for m_, t in enumerate(terms, start=1)}
        extra["term_population"] = HN
        if key == "di":
            extra["massey_form_01"] = float(u_di_massey(z[0], z[1], 0.5, EXACT))
            extra["rate_term_target1"] = float(di_rate_term(z[1], 0.5, EXACT, 5))
            extra["massey_target_term_target1"] = float(di_massey_target(z[1], 0.5, EXACT, 5))
    if key == "xme10":
        z = zs(x)
        Yf, Xp = K.xme_embedding(z[0], z[1], 10)
        extra = dict(frac_count1_joint=_frac1(np.hstack([Yf, Xp]), 0.5), frac_count1_past=_frac1(Xp, 0.5))
    write_stat(run["run_id"], m, s,
               f"Monte Carlo mean/SD of the box-kernel estimator ({R} iid N(0,1) draws, fast KD-tree counts); population {form}",
               pop, S, R, notes="convergence information: box-kernel estimates are biased for continuous data",
               population_value=popv, box_limit=boxv, pyspi_mean=float(np.nanmean(off(S))),
               mc_mean=float(np.nanmean(off(m))), **extra)


def _frac1(X, w):
    X = K._as2d(X)
    wd = _w(X, w)
    c = np.concatenate([K._box_match(X, wd, r).sum(1) for r in K._chunks(X.shape[0])])
    return float((c == 1).mean())


def chk_null(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    sym = key in ("mi", "si")
    def mean_stat(A):
        return (A[np.triu_indices(A.shape[0], 1)] if sym else off(A)).mean()
    stat = float(mean_stat(S))
    R = r_mc(key, T)
    sims = rng_for(run["run_id"]).standard_normal((R,) + x.shape)
    m, s = mc(lambda z: mean_stat(stat_matrix(key, z, FAST)), sims)
    vals = S[np.triu_indices(S.shape[0], 1)] if sym else off(S)
    ok = abs(stat - m) <= 2 * s
    write_qual(run["run_id"], ok,
               f"mean over pairs vs the Monte Carlo distribution of that mean for the box-kernel estimator on iid N(0,1) data ({R} draws); population 0",
               notes="documents the positive null bias of the resubstitution box estimator (not a pass/fail on 0)",
               pyspi_mean=stat, mc_mean=float(m), mc_sd=float(s), population=0.0, z=float((stat - m) / s),
               pyspi_min=float(vals.min()), pyspi_max=float(vals.max()))


def chk_truth_model(run, key, x, ds, T, path):
    """Gaussian VAR / delayed-noise population values (bits) vs the box-kernel sampling distribution."""
    S = load_pyspi(run["run_id"])
    name = run["dataset"]
    procs = run.get("processes")
    if name == "delayed_noise":
        pop = np.array([[np.nan, 0.5 * math.log2(101)], [0.0, np.nan]])
        R = r_mc(key, T)
        sims = sim_delayed(T, R, rng_for(run["run_id"]))[:, procs]
        form = "x1(t+1) = x0(t) + 0.1 e: I(x0(t); x1(t+1)) = 0.5 log2(101); reverse 0"
    else:
        if name == "var2_bivariate":
            A_list, Sig = [ds["A1"], ds["A2"]], ds["Sigma"]
        else:
            A_list, Sig = [ds["A"]], ds["Sigma_E"]
        G = var_gammas(A_list, Sig, 6)
        f = {"tlmi": pop_tlmi, "te": pop_te1, "xme1": pop_xme1, "cce": pop_cce}.get(key)
        if key == "si":
            v = pop_si(G, Sig)
            pop = np.array([[np.nan, v], [v, np.nan]])
        else:
            pop = np.array([[np.nan, f(G, 0, 1)], [f(G, 1, 0), np.nan]])
        R = r_mc(key, T)
        sims = sim_var(A_list, Sig, T, R, rng_for(run["run_id"]))
        form = {"tlmi": "-0.5 log2(1 - corr(x_i(t-1), x_j(t))^2) from Gamma(1)",
                "te": "0.5 log2(var(y_t|y_t-1) / var(y_t|y_t-1, x_t-1)) from companion-form Lyapunov Gamma(h)",
                "xme1": "h(x_j(t) | x_i(t-1)) = 0.5 log2(2 pi e (1 - r^2)), standardised",
                "si": "0.5 log2(prod_v var(x_v(t)|x_v(t-1)) / det Sigma_E) (Ay 2015)",
                "cce": "sum_m h(y_t | y lags 1..m-1, x lags 0..m-1), standardised"}[key]
    m, s = mc(lambda z: stat_matrix(key, z, FAST), sims)
    write_stat(run["run_id"], m, s, f"Monte Carlo mean/SD of the box-kernel estimator ({R} draws of the generating model, fast KD-tree counts); Gaussian population {form}",
               pop, S, R, notes="convergence information: box-kernel estimates are biased for continuous data",
               pop_01=float(pop[0, 1]), pop_10=float(pop[1, 0]), pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]),
               mc_mean_01=float(m[0, 1]), mc_mean_10=float(m[1, 0]),
               pyspi_diff_01_minus_10=float(S[0, 1] - S[1, 0]), pop_diff_01_minus_10=float(pop[0, 1] - pop[1, 0]))


def chk_gauss_block(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    G = var_gammas([ds["A"]], ds["Sigma_E"], 12)
    pop5 = np.array([[np.nan, pop_cce(G, 0, 1, 5)], [pop_cce(G, 1, 0, 5), np.nan]])
    pop10 = np.array([[np.nan, pop_cce(G, 0, 1, 10)], [pop_cce(G, 1, 0, 10), np.nan]])
    z = zs(x)
    t01 = cce_terms(z[0], z[1], 0.5, EXACT, 5)
    G1 = [hb(cond_var(block_cov(G, [(1, 0)] + [(1, L) for L in range(1, m_)] + [(0, L) for L in range(0, m_)])) / G[0][1, 1])
          for m_ in range(1, 6)]
    ok = bool((off(S) < off(pop5)).all())
    write_qual(run["run_id"], ok,
               "Gaussian population H(Y^5||X^5) (bits, standardised, companion-form Lyapunov); pass = kernel value below the population (count saturation in 4-10 dims documented)",
               pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]), pop5_01=float(pop5[0, 1]), pop5_10=float(pop5[1, 0]),
               pop10_01=float(pop10[0, 1]), pop10_10=float(pop10[1, 0]),
               **{f"kernel_term{m_}_01": float(t) for m_, t in enumerate(t01, start=1)},
               **{f"pop_term{m_}_01": float(g) for m_, g in enumerate(G1, start=1)})


def chk_nonlinear(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    z = zs(x)
    cross = [S[i, j] for i in range(4) for j in range(4) if i != j and {i, j} not in ({0, 1}, {2, 3})]
    pear01 = float(np.corrcoef(z[0], z[1])[0, 1])
    pear23 = float(np.corrcoef(z[2], z[3])[0, 1])
    ok = S[0, 1] > 5 * max(cross) and S[2, 3] > 5 * max(cross)
    write_qual(run["run_id"], ok, "x1 = x0^2 + 0.05 noise and (cos th, sin th) are dependent with Pearson ~ 0: MI(0,1), MI(2,3) must exceed 5x every cross pair",
               pyspi_01=float(S[0, 1]), pyspi_23=float(S[2, 3]), pyspi_cross_max=float(max(cross)),
               gaussian_mi_bits_01=float(-0.5 * math.log2(1 - pear01 ** 2)), gaussian_mi_bits_23=float(-0.5 * math.log2(1 - pear23 ** 2)))


def chk_direction(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    name = run["dataset"]
    extra = {}
    if key == "ce":
        z = zs(x)
        h1 = np.array([ex_H(z[p], 0.5) for p in range(z.shape[0])])
        anti = S - S.T
        expect = h1[None, :] - h1[:, None]                  # A[i,j] - A[j,i] = H(x_j) - H(x_i)
        err = float(np.nanmax(np.abs(anti - expect)))
        ok = err <= 1e-10
        rule = "A[i,j] = H(x_j|x_i) = JE(i,j) - H(x_i): the antisymmetric part equals H(x_j) - H(x_i) (box 1-D entropies of the z-scored rows)"
        if name == "nonlinear_dependence":
            ok = ok and S[0, 1] < S[1, 0]
            rule += "; nonlinear_dependence: A[0,1] = H(x1|x0) < A[1,0] = H(x0|x1)"
        extra = dict(antisym_identity_err=err, **{f"H1d_x{p}": float(h1[p]) for p in range(h1.size)})
    elif key in ("cce", "xme1", "xme10"):
        ok = S[0, 1] < S[1, 0]
        rule = "A[0,1] < A[1,0] (conditioning on the driver x0 lowers the target's entropy)"
    elif key == "tlmi" and name == "var2_bivariate":
        G = var_gammas([ds["A1"], ds["A2"]], ds["Sigma"], 2)
        p01, p10 = pop_tlmi(G, 0, 1), pop_tlmi(G, 1, 0)
        ok = np.sign(S[0, 1] - S[1, 0]) == np.sign(p01 - p10)
        rule = "sign(A[0,1]-A[1,0]) equals the Gaussian population ordering of I(x_i(t-1); x_j(t))"
        extra = dict(pop_01=p01, pop_10=p10)
    else:
        ok = S[0, 1] > S[1, 0]
        rule = "A[0,1] > A[1,0] (x0 drives x1)"
    if name == "delayed_noise":
        extra.update({f"dn_A0{k}": float(S[0, k]) for k in range(1, S.shape[0])})
        extra.update({f"dn_A{k}0": float(S[k, 0]) for k in range(1, S.shape[0])})
    write_qual(run["run_id"], bool(ok), f"ground truth {name}: {rule}", A01=float(S[0, 1]), A10=float(S[1, 0]),
               max_abs_A_minus_AT=float(np.nanmax(np.abs(S - S.T))), **extra)


def chk_sym_fresh(run, key, x, ds, T, path):
    D = load_pyspi(run["run_id"])
    V = stat_matrix(key, zs(x))
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:{KFUN[key]} (symmetric by construction) vs pyspi bivariate(i,j) on a fresh Data for every ordered pair (no cache, no mirror); {VERS}",
                notes="tests that the estimator itself is symmetric in its arguments",
                fresh_max_abs_D_minus_DT=float(np.nanmax(np.abs(D - D.T))))


def chk_degenerate(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    z = zs(x)
    fr_j, fr_p = [], []
    for (i, j) in [(0, 1), (1, 0)]:
        Yf, Xp = K.xme_embedding(z[i], z[j], 10)
        fr_j.append(_frac1(np.hstack([Yf, Xp]), 0.5))
        fr_p.append(_frac1(Xp, 0.5))
    vals = off(S)
    ok = min(fr_j) > 0.9 and float(np.abs(vals).max()) < 0.05
    write_qual(run["run_id"], ok, "count saturation of the 10-D joint / 9-D past boxes (W-0.5) from the numpy counts; pass = >90% of joint counts are 1 and every |value| < 0.05 bits",
               kind="independent-impl", frac_count1_joint_min=float(min(fr_j)), frac_count1_past_min=float(min(fr_p)),
               frac_count1_joint_max=float(max(fr_j)), frac_count1_past_max=float(max(fr_p)),
               pyspi_absmax=float(np.abs(vals).max()), h_true=HN)


def chk_k2_numpy(run, key, x, ds, T, path):
    z = zs(x)
    V = directed(lambda a, b: u_xme(a, b, 0.5, EXACT, 2), z)
    k1 = directed(lambda a, b: u_xme(a, b, 0.5, EXACT, 1), z)
    k2_true = directed(lambda a, b: xme_lags(a, b, 0.5, EXACT, [1, 2]), z)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:xme_embedding(k=2) (lag set {{1}}) + box H (0.5); {VERS}",
                k1_01=float(k1[0, 1]), k1_10=float(k1[1, 0]), k2_01=float(V[0, 1]), k2_10=float(V[1, 0]),
                k2_minus_k1_maxabs=float(np.nanmax(np.abs(V - k1))),
                two_lag_variant_01=float(k2_true[0, 1]), two_lag_variant_10=float(k2_true[1, 0]))


def chk_k2_equals_k1_shifted(run, key, x, ds, T, path):
    z = zs(x)
    V = directed(lambda a, b: u_xme(a[1:], b[1:], 0.5, EXACT, 1), z)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:xme_embedding(k=1) on the z-scored rows with the first sample dropped; {VERS}",
                notes="history_length=2 uses the same lag set {1} as history_length=1, on samples t = 2..T-1")


def chk_n10(run, key, x, ds, T, path):
    z = zs(x)
    f = u_cce if key == "cce" else u_di
    V = directed(lambda a, b: f(a, b, 0.5, EXACT, 10), z)
    V5 = directed(lambda a, b: f(a, b, 0.5, EXACT, 5), z)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}: box-kernel {key} with n=10 (documented history length); {VERS}",
                notes="documented history length 10 (supplement 1.4.7/1.4.8) via probe; configured value is 5",
                n10_01=float(V[0, 1]), n10_10=float(V[1, 0]), n5_01=float(V5[0, 1]), n5_10=float(V5[1, 0]))


def chk_configured_n(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    write_qual(run["run_id"], S.tolist() == [5.0, 5.0],
               "pyspi/statistics/infotheory.py CausalEntropy/DirectedInfo default n=5; config.yaml passes no n; supplement 1.4.7/1.4.8 document 10",
               kind="literature-only", cce_n=float(S[0]), di_n=float(S[1]), documented_n=10)


def chk_massey_identity(run, key, x, ds, T, path):
    z = zs(x)
    S = load_pyspi(run["run_id"])
    M = directed(lambda a, b: u_di_massey(a, b, 0.5, EXACT), z)
    B = np.array([[np.nan, di_rate_term(z[1], 0.5, EXACT, 5) - di_massey_target(z[1], 0.5, EXACT, 5)],
                  [di_rate_term(z[0], 0.5, EXACT, 5) - di_massey_target(z[0], 0.5, EXACT, 5), np.nan]])
    err = float(np.nanmax(np.abs(S - (M + B))))
    write_qual(run["run_id"], err <= 1e-10 and np.nanmin(B) > 0,
               "pyspi di = Massey DI_5 (box kernel, sum_m H(Y_m|Y^{m-1}) - CCE) + B(target), B = sum_m H(Y^m)/m - sum_m H(Y_m|Y^{m-1}) (sample identity); pass = identity to 1e-10 and B > 0",
               kind="independent-impl", identity_maxabs=err, massey_01=float(M[0, 1]), massey_10=float(M[1, 0]),
               pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]), B_target1=float(B[0, 1]), B_target0=float(B[1, 0]))


def indep_pair(x_var2, x_iid):
    """Duplicate of probes/it_kernel_probes.py:indep_pair (no import of the probe module)."""
    return np.stack([np.asarray(x_iid[0], dtype=np.float64), np.asarray(x_var2[0], dtype=np.float64)])


def chk_indep_numpy(run, key, x, ds, T, path):
    X = zs(indep_pair(x, make_data.load("iid_gaussian", T)["x"]))
    V = directed(lambda a, b: u_di(a, b, 0.5, EXACT), X)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}: box-kernel di (pyspi form) on [iid row 0, var2 row 0]; {VERS}")


def chk_indep_gap(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    X = zs(indep_pair(x, make_data.load("iid_gaussian", T)["x"]))
    M = directed(lambda a, b: u_di_massey(a, b, 0.5, EXACT), X)
    Bt = di_rate_term(X[1], 0.5, EXACT, 5) - di_massey_target(X[1], 0.5, EXACT, 5)
    Bs = di_rate_term(X[0], 0.5, EXACT, 5) - di_massey_target(X[0], 0.5, EXACT, 5)
    a1, a2, s2 = float(ds["A1"][0, 0]), float(ds["A2"][0, 0]), float(ds["Sigma"][0, 0])
    G = var_gammas([np.array([[a1]]), np.array([[a2]])], np.array([[s2]]), 6)
    Bpop = pop_di_excess_univariate(np.array([g[0, 0] for g in G]))
    ok = abs((S[0, 1] - M[0, 1]) - Bt) <= 1e-10 and Bt > 0
    write_qual(run["run_id"], ok,
               "independent white source -> AR(2) target (var2_bivariate x0): Massey DI = 0 in population; pyspi form adds B(target); box-kernel B also carries the dimension-dependent bias of the 1..5-D box entropies (white-target B shown for comparison)",
               kind="independent-impl", pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]), massey_box_01=float(M[0, 1]),
               massey_box_10=float(M[1, 0]), B_box_target_ar2=float(Bt), B_box_target_white=float(Bs),
               B_population_gaussian_ar2=float(Bpop), B_population_white=0.0,
               it_gaussian_run="infotheory.di.gaussian|var2_bivariate|4096|pipeline|indep-truth")


def te_forwarding_ref(run, key, x, ds, T, path):
    z = zs(x)
    V1 = stat_matrix("te", z)
    V = np.stack([V1] * 5)
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:te_box_kernel(k=1, width=0.25) for every (l_history, k_tau, l_tau) in [(1,1,1),(3,1,1),(1,4,1),(1,1,2),(3,4,2)]: the kernel branch forwards only k_HISTORY; {VERS}",
                notes="pass = all five pyspi matrices equal the k=l=1 numpy value (parameters ignored)")


def chk_forwarding_flags(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    exp = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    ok = S.tolist() == exp
    write_qual(run["run_id"], ok,
               "[#variants with identifier te_kernel_W-0.25_k-1 (5), k_HISTORY (1), setProperty raises for l_HISTORY/k_TAU/l_TAU (1,1,1), auto-embed identifier ends _k-max-4 (1), k_HISTORY after auto-embed (1), auto minus fixed value A[0,1] (0)]",
               kind="literature-only", **{f"flag{i}": float(v) for i, v in enumerate(S)})


def _cfg_width(ident):
    return 0.5 if "W-0.5" in ident else 0.25


def chk_frame(run, key, x, ds, T, path):
    z = zs(x)
    w = W_CFG[key]
    Vc = stat_matrix(key, z, EXACT, w)
    V25 = Vc if w == 0.25 else stat_matrix(key, z, EXACT, 0.25)
    V = np.stack([Vc, V25, Vc])
    S = load_pyspi(run["run_id"])
    write_exact(run["run_id"], V, "independent-impl",
                f"slices [Calculator, CalculatorFrame, pickle round trip] vs numpy {KREF}:{KFUN[key]} at widths [{w}, 0.25, {w}]; {VERS}",
                notes="pass = the CalculatorFrame slice equals the WIDTH-0.25 numpy value (copy path resets KERNEL_WIDTH to JIDT's default) and Calculator/pickle equal the configured width",
                calc_01=float(S[0, 0, 1]), frame_01=float(S[1, 0, 1]), pickle_01=float(S[2, 0, 1]),
                frame_minus_calc_maxabs=float(np.nanmax(np.abs(S[1] - S[0]))),
                numpy_w025_01=float(V25[0, 1]))


# ----- estimator layer -------------------------------------------------------------------
def lag_matrix(x, maxlag=4):
    """Duplicate of probes/it_kernel_probes.py:lag_matrix."""
    x = np.asarray(x, dtype=np.float64)
    P, T = x.shape
    return np.stack([x[p, maxlag - L:T - L] for p in range(P) for L in range(maxlag + 1)], axis=1)


def direct_tasks(P):
    """Duplicate of probes/it_kernel_probes.py:direct_tasks."""
    ncol = 5 * P
    tasks = []
    for w in (0.5, 0.25):
        for d in range(1, min(10, ncol) + 1):
            tasks.append(("H", tuple(range(d)), (), w))
        tasks.append(("MI", (0,), (5 % ncol,), w))
        tasks.append(("MI", (0, 1), (ncol - 1, ncol - 2), w))
        tasks.append(("TE", (0,), (1 % P,), w, 1))
        tasks.append(("TE", (1 % P,), (0,), w, 1))
        tasks.append(("TE", (0,), (1 % P,), w, 2))
    return tasks


def chk_direct_jidt(run, key, x, ds, T, path):
    M = lag_matrix(x)
    vals, vals0, nats = [], [], []
    for t in direct_tasks(x.shape[0]):
        kind, A, B, w = t[0], t[1], t[2], t[3]
        if kind == "H":
            X = M[:, list(A)]
            v = K.entropy_box_kernel(X, w)
            sd0 = X.std(axis=0, ddof=0)                  # ddof=0 variant: half-widths w * population SD
            vals0.append(K.entropy_box_kernel(X / sd0, w, normalise=False) + float(np.sum(np.log2(sd0))))
        elif kind == "MI":
            v = K.mi_box_kernel(M[:, list(A)], M[:, list(B)], w)
            vals0.append(np.nan)
        else:
            v = K.te_box_kernel(x[A[0]], x[B[0]], k=t[4], width=w)
            vals0.append(np.nan)
        vals.append(v)
    V = np.array(vals)
    S = load_pyspi(run["run_id"])
    hm = np.array([t[0] == "H" for t in direct_tasks(x.shape[0])])
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:entropy_box_kernel (d = 1..10) / mi_box_kernel (1+1, 2+2) / te_box_kernel (k = 1, 2) on the raw lag matrix, widths 0.5 and 0.25, NORMALISE (N-1 SD), bits; {VERS}",
                notes="JIDT kernel calculators driven directly (default NORMALISE=true)",
                n_tasks=int(V.size), ddof0_entropy_maxabs_diff=float(np.nanmax(np.abs(S[hm] - np.array(vals0)[hm]))),
                nats_entropy_maxabs_diff=float(np.nanmax(np.abs(S[hm] - V[hm] * LN2))))


GRID = [0.0, 1.0, 2.0, 3.0, 10.0]
GRID_B = [0.0, 2.0, 1.0, 3.0, 4.0]
GRID2 = [[a, b] for a in (0.0, 1.0, 2.0) for b in (0.0, 2.0, 5.0)]
TE_S = [0.0, 1, 2, 1, 0, 3, 2, 1, 0, 1]
TE_T = [1.0, 0, 1, 2, 1, 0, 3, 2, 1, 0]
TENTH_GRID_N = 31


def grid_tasks():
    """Duplicate of probes/it_kernel_probes.py:grid_tasks."""
    q = list(np.arange(TENTH_GRID_N) * 0.1)
    return [
        ("H", GRID, 1.0 - 1e-6), ("H", GRID, 1.0), ("H", GRID, 1.0 + 1e-6), ("H", GRID, 2.0),
        ("H", GRID2, 1.0 - 1e-6), ("H", GRID2, 1.0), ("H", GRID2, 2.0),
        ("H", [0.0, 100.0], 0.5),
        ("MI", (GRID, GRID), 1.0), ("MI", (GRID, GRID_B), 1.0), ("MI", (GRID, GRID_B), 1.0 - 1e-6),
        ("TE", (TE_S, TE_T), 1.0), ("TE", (TE_S, TE_T), 1.0 - 1e-6),
        ("H", q, 0.1), ("H", q, 0.1 + 1e-12),
    ]


def _open_H(X, w):
    X = K._as2d(np.asarray(X, dtype=np.float64))
    N = X.shape[0]
    c = np.all(np.abs(X[:, None, :] - X[None, :, :]) < w, axis=2).sum(1)
    return float(-np.mean(np.log2(c / (N * (2 * w) ** X.shape[1]))))


def _noself_H(X, w):
    X = K._as2d(np.asarray(X, dtype=np.float64))
    N = X.shape[0]
    c = np.all(np.abs(X[:, None, :] - X[None, :, :]) <= w, axis=2).sum(1) - 1
    with np.errstate(divide="ignore"):
        return float(-np.mean(np.log2(c / ((N - 1) * (2 * w) ** X.shape[1]))))


def chk_tie_grid(run, key, x, ds, T, path):
    vals, opens, noself = [], [], []
    for kind, args, w in grid_tasks():
        if kind == "H":
            X = np.asarray(args, dtype=np.float64)
            vals.append(K.entropy_box_kernel(X, w, normalise=False))
            opens.append(_open_H(X, w))
            noself.append(_noself_H(X, w))
        elif kind == "MI":
            vals.append(K.mi_box_kernel(np.asarray(args[0]), np.asarray(args[1]), w, normalise=False))
            opens.append(np.nan); noself.append(np.nan)
        else:
            vals.append(K.te_box_kernel(np.asarray(args[0]), np.asarray(args[1]), k=1, width=w, normalise=False))
            opens.append(np.nan); noself.append(np.nan)
    V = np.array(vals)
    S = load_pyspi(run["run_id"])
    hm = np.isfinite(np.array(opens))
    write_exact(run["run_id"], V, "analytic",
                "closed max-norm box counts (|dx| <= w), self included, NORMALISE=false (absolute widths), p = C/(N (2w)^d), bits; grids [0,1,2,3,10], 3x3 product grid, [0,100], k*0.1 (k=0..30, counts change at |dx| = w by float rounding), MI and TE grids",
                notes="H(GRID, w=1) closed counts [2,3,3,2,1] -> 2.2879430946; open-box and no-self variants recorded as extras",
                rtol=0.0, atol=1e-13,
                open_box_maxabs_diff=float(np.nanmax(np.abs(S[hm] - np.array(opens)[hm]))),
                no_self_maxabs_diff_finite=float(np.nanmax(np.abs((S[hm] - np.array(noself)[hm])[np.isfinite(np.array(noself)[hm])]))),
                no_self_isolated_point_H=float(noself[7]),
                H_grid_w1=float(S[1]), H_grid_w1_open=float(opens[1]), H_isolated=float(S[7]),
                H_tenth_w=float(S[13]), H_tenth_w_open=float(opens[13]))


def chk_props(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    nan = np.isnan
    ok = (S[0] == 42 and S[1] == 0 and nan(S[2]) and S[3] == 42 and S[4] == 0 and nan(S[5])
          and nan(S[6]) and nan(S[7]) and nan(S[8]) and S[9] == 0.5 and S[10] == 0.25 and S[11] == 0.25
          and S[12] == 1 and S[13] == 1 and S[14] == 0 and S[16] == 0 and S[17] != 0
          and S[18] == 0 and S[20] == 0 and S[21] != 0)
    names = ["ent_NOISE_SEED", "ent_NOISE_LEVEL", "ent_BIAS_CORRECTION", "mi_NOISE_SEED", "mi_NOISE_LEVEL", "mi_BIAS_CORRECTION",
             "te_NOISE_SEED", "te_NOISE_LEVEL", "te_BIAS_CORRECTION", "ent_KERNEL_WIDTH", "mi_KERNEL_WIDTH", "te_KERNEL_WIDTH",
             "ent_NORMALISE", "te_NORMALISE", "dH_seed_only", "dH_noise1e-8_seed42", "dH_bias_correction", "dH_noise0.01_seed42",
             "dMI_seed_only", "dMI_noise1e-8_seed42", "dMI_bias_correction", "dMI_noise0.01_seed42"]
    write_qual(run["run_id"], bool(ok),
               "JIDT getProperty after pyspi _setup (nan = unknown property) and value perturbation: seed alone and BIAS_CORRECTION change nothing, a noise level changes the value (seeded noise path live but unused)",
               kind="same-lineage", **{n: float(v) for n, v in zip(names, S)})


def chk_copy_props(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ids = ["je", "ce", "cce", "xme_k1", "xme_k10", "di", "si", "mi", "tlmi", "te"]
    ok = True
    for r, ident in enumerate(ids):
        row = S[r]
        if r < 7:
            ok &= row[0] == 0.5 and row[1] == 0.25 and row[2] == 0.5 and np.isnan(row[3:6]).all()
        else:
            ok &= (row[:6] == 0.25).all()
        ok &= row[6] == 42 and row[7] == -1
    write_qual(run["run_id"], bool(ok),
               "KERNEL_WIDTH of _entropy_calc / _calc on [original, copy.deepcopy, pickle round trip], NOISE_SEED [original, deepcopy]: the deepcopy reads JIDT's defaults (0.25, 'NONE'), pickle re-applies _setup",
               kind="same-lineage", **{f"{ident}_deepcopy_width": float(S[r, 1]) for r, ident in enumerate(ids)},
               **{f"{ident}_pickle_width": float(S[r, 2]) for r, ident in enumerate(ids)},
               deepcopy_noise_seed_is_NONE=float((S[:, 7] == -1).all()))


def chk_force_all(run, key, x, ds, T, path):
    z = zs(x)
    mi = K.mi_box_kernel(z[0], z[1], 0.25)
    te01 = K.te_box_kernel(z[0], z[1], 1, 0.25)
    te10 = K.te_box_kernel(z[1], z[0], 1, 0.25)
    V = np.array([mi, mi, te01, te01, te10, te10])
    S = load_pyspi(run["run_id"])
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:mi_box_kernel / te_box_kernel (all-pairs comparisons) for both the bin-search and FORCE_KERNEL_COMPARE_TO_ALL slots; {VERS}",
                bins_minus_all_maxabs=float(np.max(np.abs(S[0::2] - S[1::2]))))


RAW_IDS = ["je", "ce", "mi", "tlmi", "te"]


def chk_raw_array(run, key, x, ds, T, path):
    xa, xb = np.asarray(x[2], dtype=np.float64), np.asarray(x[3], dtype=np.float64)
    za, zb = zs(xa[None])[0], zs(xb[None])[0]
    vals = []
    for u in RAW_IDS:
        fn = UFN[u][1]
        w = W_CFG[u]
        vals.append(fn(za, xb, w, EXACT))            # raw call: first array z-scored, second appended raw
        vals.append(fn(za, zb, w, EXACT))            # Data call: both z-scored
    V = np.array(vals)
    S = load_pyspi(run["run_id"])
    sdb = float(xb.std(ddof=1))
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}: [raw bivariate(x_a, x_b), Data bivariate(i, j)] per identifier, raw = f(z(x_a), x_b raw) (Data.add_process z-scores only the first array); {VERS}",
                notes="rows 2, 3 of raw scaled_offset; order je, ce, mi, tlmi, te",
                log2_sd_b=math.log2(sdb), je_raw_minus_data=float(S[0] - S[1]), ce_raw_minus_data=float(S[2] - S[3]),
                mi_raw_minus_data=float(S[4] - S[5]), tlmi_raw_minus_data=float(S[6] - S[7]), te_raw_minus_data=float(S[8] - S[9]))


def chk_entropy_1d_numpy(run, key, x, ds, T, path):
    z = zs(x)
    V = np.array([[ex_H(z[p], w) for p in range(z.shape[0])] for w in (0.5, 0.25)])
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:entropy_box_kernel(z_p, 0.5 / 0.25); {VERS}")


def chk_entropy_1d_truth(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    uniform = run["dataset"] == "nonlinear_dependence"
    P = S.shape[1]
    R = 400
    rng = rng_for(run["run_id"])
    sims = rng.uniform(-1, 1, (R, T)) if uniform else rng.standard_normal((R, T))
    vals = np.array([[fa_H(zs(s[None])[0], w) for w in (0.5, 0.25)] for s in sims])
    m, s = vals.mean(0), vals.std(0, ddof=1)
    Mv = np.repeat(m[:, None], P, axis=1)
    Sv = np.repeat(s[:, None], P, axis=1)
    if uniform:
        popv = math.log2(2 * math.sqrt(3.0))
        box = [h_box_uniform(0.5), h_box_uniform(0.25)]
        L = 2 * math.sqrt(3.0)                          # _kernels_infotheory selftest (4) edge formula
        edge = [math.log2(L) + 2 * e * (1 - LN2) / (L * LN2) for e in (0.5, 0.25)]
        form = (f"z-scored U(-1,1): h = log2(2 sqrt 3) = 1.79248; box limit by quadrature, equal to the edge formula "
                f"log2 L + 2w(1 - ln2)/(L ln2) of _kernels_infotheory selftest (4): {edge[0]:.6f} / {edge[1]:.6f}")
    else:
        popv = HN
        box = [h_box1(0.5), h_box1(0.25)]
        form = "N(0,1): h = 0.5 log2(2 pi e) = 2.04710; box limit -int phi log2[(Phi(x+e)-Phi(x-e))/2e]"
    pop = np.full_like(S, popv)
    write_stat(run["run_id"], Mv, Sv, f"Monte Carlo mean/SD of the 1-D box entropy ({R} draws, fast KD-tree counts); {form}",
               pop, S, R, notes="rows: width 0.5 (configured), 0.25; convergence information",
               population_value=popv, box_limit_w05=box[0], box_limit_w025=box[1],
               pyspi_mean_w05=float(S[0].mean()), pyspi_mean_w025=float(S[1].mean()),
               mc_mean_w05=float(m[0]), mc_mean_w025=float(m[1]))


CHECKS_BY_TAG = {
    ("*", "numpy"): chk_numpy,
    ("*", "equals-d2"): chk_equals_d2,
    ("*", "shift"): chk_shift,
    ("*", "bound"): chk_bound,
    ("*", "truth-iid"): chk_truth_iid,
    ("*", "null"): chk_null,
    ("*", "truth"): chk_truth_model,
    ("mi", "nonlinear"): chk_nonlinear,
    ("cce", "gauss-block"): chk_gauss_block,
    ("*", "direction"): chk_direction,
    ("*", "sym-fresh"): chk_sym_fresh,
    ("xme10", "degenerate"): chk_degenerate,
    ("xme1", "k2-numpy"): chk_k2_numpy,
    ("xme1", "k2-equals-k1-shifted"): chk_k2_equals_k1_shifted,
    ("*", "n10"): chk_n10,
    ("cce", "configured-n"): chk_configured_n,
    ("di", "massey-identity"): chk_massey_identity,
    ("di", "indep-numpy"): chk_indep_numpy,
    ("di", "indep-gap"): chk_indep_gap,
    ("te", "forwarding"): te_forwarding_ref,
    ("te", "forwarding-flags"): chk_forwarding_flags,
    ("*", "frame"): chk_frame,
    ("est", "direct-jidt"): chk_direct_jidt,
    ("est", "tie-grid"): chk_tie_grid,
    ("est", "props"): chk_props,
    ("est", "copy-props"): chk_copy_props,
    ("est", "force-all"): chk_force_all,
    ("est", "raw-array"): chk_raw_array,
    ("est", "entropy-1d-numpy"): chk_entropy_1d_numpy,
    ("est", "entropy-1d-truth"): chk_entropy_1d_truth,
    ("est", "entropy-1d-uniform"): chk_entropy_1d_truth,
}


def _truth_dispatch(run, key, x, ds, T, path):
    if run["dataset"] == "bivariate_gaussian":
        return chk_truth_pair(run, key, x, ds, T, path)
    return chk_truth_model(run, key, x, ds, T, path)


CHECKS_BY_TAG[("*", "truth-p01")] = _truth_dispatch
CHECKS_BY_TAG[("*", "truth-p23")] = _truth_dispatch


def run_check(run):
    rid, ds_name, T, path = run["run_id"], run["dataset"], int(run["T"]), run["path"]
    tag = rid.split("|")[-1]
    key = UNIT_KEY[run["unit_id"]]
    ds = make_data.load(ds_name, T)
    x = np.asarray(ds["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    fn = CHECKS_BY_TAG.get((key, tag)) or CHECKS_BY_TAG.get(("*", tag))
    if fn is None:
        raise KeyError(f"no reference for {rid}")
    return fn(run, key, x, ds, T, path)


CALIB_RUN = "infotheory.est.kernel|iid_gaussian|1024|pipeline|entropy-1d-truth"


def calibration_summary():
    """Per-entry z = |pyspi - MC mean| / MC SD over every statistical run of the family (reads
    out/pyspi and out/ref): count of entries beyond 2 and 3 SD vs the 4.55% / 0.27% expected if
    the pyspi value were a typical draw from the box estimator's sampling distribution, plus the
    binomial upper-tail probability P(X >= beyond_2sd) for X ~ Bin(entries, 0.0455). Entries are
    not independent (shared draws within a run), so the tail probability is descriptive."""
    from scipy.stats import binom
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    zs_all, n_runs = [], 0
    for run in man["runs"]:
        rp, rr = PYS / f"{run['run_id']}.npz", OUT / f"{run['run_id']}.npz"
        if not (rp.is_file() and rr.is_file()):
            continue
        with np.load(rr) as zr:
            if str(zr["criterion"]) != "statistical":
                continue
            V, se = zr["value"], zr["se"]
        S = load_pyspi(run["run_id"])
        m = ~np.isnan(S) & ~np.isnan(V)
        zs_all.extend(np.abs(S[m] - V[m]) / se[m])
        n_runs += 1
    z = np.array(zs_all)
    n2 = int((z > 2).sum())
    return {"statistical_runs": n_runs, "entries": int(z.size), "beyond_2sd": n2,
            "expected_beyond_2sd": round(0.0455 * z.size, 2), "beyond_3sd": int((z > 3).sum()),
            "expected_beyond_3sd": round(0.0027 * z.size, 2), "max_z": float(z.max()),
            "binom_p_ge_beyond_2sd": float(binom.sf(n2 - 1, z.size, 0.0455))}


def calibration():
    print(json.dumps(calibration_summary()))


def attach_calibration():
    """Store the family calibration as extra keys (calib_*) on CALIB_RUN's reference npz so that
    compare.py copies them into results.json (as ref_calib_*). Run after every reference exists."""
    rr = OUT / f"{CALIB_RUN}.npz"
    if not rr.is_file():
        return
    c = calibration_summary()
    with np.load(rr) as zr:
        kw = {k: zr[k] for k in zr.files if not k.startswith("calib_")}
    kw.update({f"calib_{k}": v for k, v in c.items()})
    kw["notes"] = (str(kw.get("notes", "")) + " calib_* keys: family-wide calibration over every "
                   "statistical run (refs/it_kernel.py:calibration_summary), not specific to this run.").strip()
    write(CALIB_RUN, **kw)
    print("calibration attached to " + CALIB_RUN + ": " + json.dumps(c))


def main():
    if "--calibration" in sys.argv:
        return calibration()
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    t_all = time.perf_counter()
    for run in man["runs"]:
        rid = run["run_id"]
        if only and not any(o in rid for o in only):
            continue
        t0 = time.perf_counter()
        try:
            run_check(run)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        dt = time.perf_counter() - t0
        if dt > 20:
            print(f"slow {dt:.1f}s {rid}", file=sys.stderr)
    attach_calibration()
    print(f"it_kernel references: {n_ok} written, {n_err} errors in {time.perf_counter() - t_all:.0f}s")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
