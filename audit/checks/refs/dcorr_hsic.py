"""References for the dcorr_hsic family (distance.dcorr.biased-{False,True}, distance.hsic.biased-{False,True}).
COMPARISON venv: no hyppo, no pyspi, no sklearn in the value path.

Estimators (the quantity pyspi stores, written from the papers):
* dcorr        bias-corrected dCor^2 R*_n (Szekely & Rizzo 2014, Ann. Stat. 42:2382, U-centring), signed, NOT square-rooted
               -> _kernels_distance.dcorr(x, y, biased=False)["hyppo"]
* dcorr_biased sample dCor_n (Szekely, Rizzo & Bakirov 2007, Ann. Stat. 35:2769, Sec. 2.2, double centring), square-rooted
               -> _kernels_distance.dcorr(x, y, biased=True)["hyppo"]
* hsic         HSIC_u(K,L) / sqrt(HSIC_u(K,K) HSIC_u(L,L)), HSIC_u = Song et al. 2012 (JMLR 13:1393) eq. 5 (this file:
               hsic_stats), Gaussian kernels exp(-gamma d^2) with gamma = 1/(2 med^2), med = median off-diagonal |x_a - x_b|
               of the FIRST argument only (hyppo convention, _kernels_distance.median_gamma), applied to both variables
* hsic_biased  sqrt(HSIC_b(K,L) / sqrt(HSIC_b(K,K) HSIC_b(L,L))), HSIC_b = tr(KHLH)/n^2 (Gretton et al. 2005/2008),
               same kernel convention (_kernels_distance.hsic_biased)
pyspi stores A[i, j] = A[j, i] = f(x_min(i,j), x_max(i,j)) (Undirected mirror, pyspi/base.py:132-138); the
reference matrices use the same rule (sym_matrix) except for the probe runs, which return the unmirrored
f(x_i, x_j) (full_matrix).

Second references: dcor==0.7 (external package, never imports hyppo) for both dcorr units; R dHSIC 2.2
(r/dcorr_hsic.R, CSV exchange under out/exchange/dcorr_hsic/) for the raw biased HSIC; Gretton et al. 2007 eq. (3)
by brute force over distinct index tuples for the unbiased HSIC at T = 64.

Population values (bivariate normal, unit variances, correlation rho):
* dCor^2(rho): Szekely, Rizzo & Bakirov 2007 Theorem 7(ii) closed form (dcor2_closed), cross-checked by an
  independent quadrature of the three-term dCov^2 decomposition (dcor2_quad: folded-normal conditional means) and
  by a large-n Monte Carlo of the U-statistic (dcor package, method avl).
* normalised HSIC with sigma = population median of |X - X'| = sqrt(2) Phi^-1(0.75): closed form from
  E[exp(-z'Az/2)] = det(I + A Sigma)^-1/2 (hsic_pop_closed), cross-checked by direct 2-D quadrature of each
  expectation (hsic_pop_quad) and by a Monte Carlo over independent quadruples (hsic_pop_mc).

Monte Carlo SDs (truth, null, detected runs) use fresh draws with fixed seeds and are cached in
out/exchange/dcorr_hsic/mc_cache.json (key includes MC_VERSION).

Run tags (last field of run_id): scratch, dcor, dhsic, eq3, affine, unmirrored, reversed, null, null-trend,
truth-p01, truth-p23, popcheck, detected, detrend, trend-contrast, scale. See ``REFS``.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402

import make_data  # noqa: E402
import _kernels_distance as kd  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PYSPI = CHECKS / "out" / "pyspi"
EXCHANGE = CHECKS / "out" / "exchange" / "dcorr_hsic"
FAMILY = "dcorr_hsic"
MC_VERSION = "v2"
KD = "refs/_kernels_distance.py:"
ME = "refs/dcorr_hsic.py:"
SIGMA_POP = float(np.sqrt(2.0) * stats.norm.ppf(0.75))

UNIT = {"distance.dcorr.biased-False": "dcorr", "distance.dcorr.biased-True": "dcorr_biased",
        "distance.hsic.biased-False": "hsic", "distance.hsic.biased-True": "hsic_biased"}


# ------------------------------------------------------------------------------------------- helpers
def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(axis=-1, keepdims=True)
    return (x - m) / x.std(axis=-1, ddof=1, keepdims=True)


def lstsq_detrend(x):
    """Least-squares line removal per row (np.linalg.lstsq on [1, t]); scipy.signal.detrend('linear') removes the same line."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[1]
    A = np.stack([np.ones(T), np.arange(T, dtype=np.float64)], axis=1)
    coef = np.linalg.lstsq(A, x.T, rcond=None)[0]
    return x - (A @ coef).T


def prep(x, path):
    if path == "kernel":
        return np.asarray(x, dtype=np.float64)
    if path == "pipeline":
        return zscore(x)
    if path == "pipeline-detrend":
        return zscore(lstsq_detrend(x))
    raise ValueError(path)


def load_x(run, dataset=None, T=None, processes="run"):
    x = np.asarray(make_data.load(dataset or run["dataset"], T or run["T"])["x"], dtype=np.float64)
    procs = run.get("processes") if processes == "run" else processes
    return x[list(procs)] if procs is not None else x


def pyspi_npz(rid):
    with np.load(OUT_PYSPI / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64), np.asarray(z["input"], dtype=np.float64)


def asym(S):
    D = np.abs(S - S.T)
    return float(np.nanmax(D)) if np.isfinite(D).any() else float("nan")


def save(rid, value, criterion, kind, detail, notes="", **extra):
    payload = dict(value=np.asarray(value, dtype=np.float64), criterion=np.array(criterion),
                   reference_kind=np.array(kind), reference_detail=np.array(detail), notes=np.array(notes))
    for k, v in extra.items():
        payload[k] = np.array(v) if isinstance(v, str) else np.asarray(v, dtype=np.float64)
    np.savez(OUT_REF / f"{rid}.npz", **payload)


def pairs_upper(P):
    return [(i, j) for i in range(P) for j in range(i + 1, P)]


# --------------------------------------------------------------------------------------- estimators
def _gram(x, gamma):
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.exp(-gamma * (x[:, None] - x[None, :]) ** 2)


def hsic_u_from_gram(K, L):
    """Unbiased HSIC, Song, Smola, Gretton, Bedo & Borgwardt 2012 (JMLR 13:1393-1434) eq. (5):
    with K~, L~ the Gram matrices with zeroed diagonal,
    HSIC_u = [tr(K~L~) + 1'K~1 1'L~1 / ((n-1)(n-2)) - 2/(n-2) 1'K~L~1] / (n(n-3)).
    The reductions are done in np.longdouble (x87 80-bit on this machine): in float64 the three sums of near-one Gram
    entries cancel badly when one kernel is almost constant (raw scaled_offset with x's bandwidth 50x y's scale lost
    1.1e-12 absolute; scratchpad probe dcorr_hsic_check/probe_ld.py, which audit/checks/archive_probes.py packs into
    audit/checks/probes/scratch_probes_2026-10-15.tar.xz under the same relative path)."""
    n = K.shape[0]
    Kt, Lt = np.asarray(K, dtype=np.longdouble).copy(), np.asarray(L, dtype=np.longdouble).copy()
    np.fill_diagonal(Kt, 0.0); np.fill_diagonal(Lt, 0.0)
    kr, lr = Kt.sum(1), Lt.sum(1)
    return float((np.sum(Kt * Lt) + Kt.sum() * Lt.sum() / ((n - 1) * (n - 2)) - np.longdouble(2.0) / (n - 2) * np.dot(kr, lr))
                 / (n * (n - 3)))


def hsic_b_from_gram(K, L):
    """Biased HSIC_b = tr(K H L H) / n^2 = sum((HKH) * (HLH)) / n^2 (Gretton et al. 2005, 1/n^2 scaling)."""
    n = K.shape[0]
    K, L = np.asarray(K, dtype=np.longdouble), np.asarray(L, dtype=np.longdouble)   # same reason as hsic_u_from_gram
    Kc = K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
    Lc = L - L.mean(0, keepdims=True) - L.mean(1, keepdims=True) + L.mean()
    return float(np.sum(Kc * Lc) / n ** 2)


def hsic_stats(x, y, gamma_x=None, gamma_y=None):
    """Raw and normalised HSIC_b / HSIC_u with Gaussian kernels. Default: gamma from x only for BOTH kernels (hyppo
    convention, kd.median_gamma mirrors hyppo/tools/common.py:321-343)."""
    gx = kd.median_gamma(x) if gamma_x is None else gamma_x
    gy = gx if gamma_y is None else gamma_y
    K, L = _gram(x, gx), _gram(y, gy)
    out = dict(gamma_x=gx, gamma_y=gy)
    for tag, f in (("b", hsic_b_from_gram), ("u", hsic_u_from_gram)):
        hxy, hxx, hyy = f(K, L), f(K, K), f(L, L)
        out.update({f"h{tag}_xy": hxy, f"h{tag}_xx": hxx, f"h{tag}_yy": hyy,
                    f"norm_{tag}": hxy / np.sqrt(hxx * hyy) if hxx * hyy > 0 else 0.0})
    out["stored_b"] = float(np.sqrt(out["norm_b"])) if out["norm_b"] > 0 else 0.0   # hyppo: sqrt, 0 if covar <= 0
    out["stored_u"] = float(out["norm_u"])
    return out


def hsic_stored_fast(x, y, biased):
    """Same stored quantity as hsic_stats(...)['stored_b' | 'stored_u'] (x-only median gamma for both kernels), computing
    only the estimator needed; used for the Monte Carlo spreads."""
    g = kd.median_gamma(x)
    K, L = _gram(x, g), _gram(y, g)
    if biased:
        f = hsic_b_from_gram
        nb = f(K, L) / np.sqrt(f(K, K) * f(L, L))
        return float(np.sqrt(nb)) if nb > 0 else 0.0
    f = hsic_u_from_gram
    return float(f(K, L) / np.sqrt(f(K, K) * f(L, L)))


def per_variable_gamma(x, y):
    return kd.median_gamma(x), kd.median_gamma(y)


def stat(ident, x, y, hsic_gammas="x"):
    """The quantity pyspi stores for f(x, y)."""
    if ident == "dcorr":
        return kd.dcorr(x, y, biased=False)["hyppo"]
    if ident == "dcorr_biased":
        return kd.dcorr(x, y, biased=True)["hyppo"]
    if hsic_gammas == "x":
        h = hsic_stats(x, y)
    else:
        h = hsic_stats(x, y, *per_variable_gamma(x, y))
    return h["stored_u"] if ident == "hsic" else h["stored_b"]


def sym_matrix(f, X):
    """A[i,j] = A[j,i] = f(x_min, x_max) (pyspi's Undirected mirror)."""
    P = X.shape[0]
    V = np.full((P, P), np.nan)
    for i, j in pairs_upper(P):
        V[i, j] = V[j, i] = f(X[i], X[j])
    return V


def full_matrix(f, X):
    P = X.shape[0]
    V = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = f(X[i], X[j])
    return V


def detail_of(ident):
    return {"dcorr": KD + "dcorr(biased=False)['hyppo'] (U-centred bias-corrected dCor^2, Szekely & Rizzo 2014 Sec. 3; no sqrt)",
            "dcorr_biased": KD + "dcorr(biased=True)['hyppo'] (double-centred dCor_n, Szekely, Rizzo & Bakirov 2007 Sec. 2.2; sqrt)",
            "hsic": ME + "hsic_stats norm_u (Song et al. 2012 eq. 5 HSIC_u, normalised, Gaussian kernel gamma = 1/(2 med_x^2) for both)",
            "hsic_biased": ME + "hsic_stats stored_b = sqrt(HSIC_b(K,L)/sqrt(HSIC_b(K,K) HSIC_b(L,L))) (" + KD
            + "median_gamma from x for both kernels)"}[ident]


# ---------------------------------------------------------------------------- Gretton 2007 eq. (3)
def hsic_u_eq3(K, L):
    """Gretton, Fukumizu, Teo, Song, Schoelkopf & Smola 2007 (NIPS 20) eq. (3), brute force over distinct index tuples:
    HSIC_u = (m)_2^-1 sum_{(i,j)} k_ij l_ij + (m)_4^-1 sum_{(i,j,q,r)} k_ij l_qr - 2 (m)_3^-1 sum_{(i,j,q)} k_ij l_iq,
    (m)_k = m!/(m-k)!, sums over tuples of pairwise distinct indices. O(m^4) masks; for m <= 64 only."""
    m = K.shape[0]
    off = ~np.eye(m, dtype=bool)
    a = b = c = 0.0
    idx = np.arange(m)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            a += K[i, j] * L[i, j]
            keep = (idx != i) & (idx != j)
            M = off & keep[:, None] & keep[None, :]           # q != r, q, r not in {i, j}
            b += K[i, j] * L[M].sum()
            c += K[i, j] * L[i, keep].sum()                  # q not in {i, j}
    m2, m3, m4 = m * (m - 1), m * (m - 1) * (m - 2), m * (m - 1) * (m - 2) * (m - 3)
    return a / m2 + b / m4 - 2.0 * c / m3


# ------------------------------------------------------------------------------- population values
def dcor2_closed(rho):
    """Szekely, Rizzo & Bakirov 2007 Theorem 7(ii) (arXiv 0803.4101 p.18): bivariate normal, correlation rho,
    dCor^2 = [rho asin rho + sqrt(1-rho^2) - rho asin(rho/2) - sqrt(4-rho^2) + 1] / [1 + pi/3 - sqrt 3]."""
    r = abs(rho)
    return (r * np.arcsin(r) + np.sqrt(1 - r * r) - r * np.arcsin(r / 2) - np.sqrt(4 - r * r) + 1) / (1 + np.pi / 3 - np.sqrt(3))


def _e_abs_prod(r):
    """E|U||V| for standard bivariate normal with correlation r, by 1-D quadrature over u of |u| E[|V| | U=u]
    (V|U=u ~ N(r u, 1-r^2); folded-normal mean s sqrt(2/pi) exp(-m^2/2s^2) + m (1 - 2 Phi(-m/s)))."""
    if abs(r) >= 1:
        return 1.0                                          # E U^2
    s = np.sqrt(1 - r * r)

    def g(u):
        m = r * u
        return abs(u) * (s * np.sqrt(2 / np.pi) * np.exp(-m * m / (2 * s * s)) + m * (1 - 2 * stats.norm.cdf(-m / s))) * stats.norm.pdf(u)
    return integrate.quad(g, -np.inf, np.inf, epsabs=1e-14, epsrel=1e-13, limit=200)[0]


def dcor2_quad(rho):
    """dCov^2(X,Y) = E|X-X'||Y-Y'| + E|X-X'| E|Y-Y'| - 2 E|X-X'||Y-Y''| (Szekely et al. 2007 Thm 7 proof route, but
    each expectation by quadrature): X-X', Y-Y' are N(0,2) with corr rho; X-X', Y-Y'' with corr rho/2."""
    def dcov2(r):
        return 2 * _e_abs_prod(r) + 2 * (2 / np.pi) - 2 * 2 * _e_abs_prod(r / 2)
    return dcov2(rho) / dcov2(1.0)


def hsic_pop_closed(rho, sigma=SIGMA_POP):
    """Population HSIC for (X, Y) standard bivariate normal, both kernels exp(-d^2/(2 sigma^2)):
    E[k l] + E[k] E[l] - 2 E[k(X,X') l(Y,Y'')] with E[exp(-z'Az/2)] = det(I + A Sigma)^-1/2; returns
    (HSIC(rho), HSIC(1), normalised = HSIC(rho)/HSIC(1))."""
    def h(r):
        A = np.eye(2) / sigma ** 2
        t1 = np.linalg.det(np.eye(2) + A @ (2 * np.array([[1, r], [r, 1]]))) ** -0.5
        t2 = (1 + 2 / sigma ** 2) ** -1
        t3 = np.linalg.det(np.eye(2) + A @ np.array([[2, r], [r, 2]])) ** -0.5
        return t1 + t2 - 2 * t3
    a, b = h(rho), h(1.0)
    return a, b, a / b


def _gauss2_expect(c, sigma, box=14.0):
    """E[exp(-(U^2 + V^2)/(2 sigma^2))] for (U, V) ~ N(0, [[2, c], [c, 2]]) by 2-D quadrature of integrand x density."""
    S = np.array([[2.0, c], [c, 2.0]])
    Si, det = np.linalg.inv(S), np.linalg.det(S)
    norm = 1.0 / (2 * np.pi * np.sqrt(det))

    def f(v, u):
        q = Si[0, 0] * u * u + 2 * Si[0, 1] * u * v + Si[1, 1] * v * v
        return np.exp(-(u * u + v * v) / (2 * sigma ** 2) - q / 2) * norm
    return integrate.dblquad(f, -box, box, -box, box, epsabs=1e-13, epsrel=1e-12)[0]


def _gauss1_expect(scale2, sigma):
    """E[exp(-scale2 U^2 / (2 sigma^2))], U ~ N(0, 2), by 1-D quadrature."""
    f = lambda u: np.exp(-scale2 * u * u / (2 * sigma ** 2)) * stats.norm.pdf(u, scale=np.sqrt(2.0))
    return integrate.quad(f, -np.inf, np.inf, epsabs=1e-14, epsrel=1e-13)[0]


def hsic_pop_quad(rho, sigma=SIGMA_POP):
    """Same population quantity as hsic_pop_closed, each expectation by direct numerical integration (no det identity).
    HSIC(1) = HSIC(X, X): term1 = E[exp(-2 (X-X')^2/(2 sigma^2))] (1-D), term3 uses corr 1/2 (2-D)."""
    t2 = _gauss1_expect(1.0, sigma) ** 2
    a = _gauss2_expect(2 * rho, sigma) + t2 - 2 * _gauss2_expect(rho, sigma)
    b = _gauss1_expect(2.0, sigma) + t2 - 2 * _gauss2_expect(1.0, sigma)
    return a, b, a / b


def hsic_pop_mc(rho, n=2_000_000, reps=10, sigma=SIGMA_POP, seed=11):
    """Monte Carlo over independent quadruples (X1,Y1),...,(X4,Y4) of h = k12 l12 + k12 l34 - 2 k12 l13 (an unbiased
    estimator of population HSIC), for rho and for the (X, X) normaliser; returns normalised mean and se."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        z = rng.standard_normal((4, 2, n))
        xs = z[:, 0]
        ys = rho * z[:, 0] + np.sqrt(1 - rho * rho) * z[:, 1]
        k = lambda a, b: np.exp(-(a - b) ** 2 / (2 * sigma ** 2))
        k12 = k(xs[0], xs[1])
        num = np.mean(k12 * k(ys[0], ys[1]) + k12 * k(ys[2], ys[3]) - 2 * k12 * k(ys[0], ys[2]))
        den = np.mean(k12 * k12 + k12 * k(xs[2], xs[3]) - 2 * k12 * k(xs[0], xs[2]))
        vals.append(num / den)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(reps))


def population(ident, rho):
    """Population value of the STORED quantity: dcorr dCor^2, dcorr_biased dCor, hsic normalised HSIC,
    hsic_biased sqrt(normalised HSIC)."""
    if ident == "dcorr":
        return float(dcor2_closed(rho))
    if ident == "dcorr_biased":
        return float(np.sqrt(dcor2_closed(rho)))
    v = hsic_pop_closed(rho)[2]
    return float(v if ident == "hsic" else np.sqrt(v))


# ----------------------------------------------------------------------------------- Monte Carlo
_MC_PATH = EXCHANGE / "mc_cache_v2.json"
_MC = None


def _mc_cache():
    global _MC
    if _MC is None:
        try:
            _MC = json.loads(_MC_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _MC = {}
    return _MC


def _mc_store(key, val):
    c = _mc_cache()
    c[key] = val
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    tmp = _MC_PATH.with_name(_MC_PATH.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(c, indent=1, sort_keys=True))
    os.replace(tmp, _MC_PATH)


def mc_stat_fn(ident):
    """Fast statistic for Monte Carlo spread on z-scored inputs. dcorr units use the dcor package's O(n log n)
    1-D algorithm (method 'avl'); hsic units use hsic_stats (O(n^2))."""
    import dcor
    if ident == "dcorr":
        return lambda x, y: float(dcor.u_distance_correlation_sqr(x, y, method="avl"))
    if ident == "dcorr_biased":
        return lambda x, y: float(dcor.distance_correlation(x, y, method="avl"))
    return lambda x, y: hsic_stored_fast(x, y, biased=ident == "hsic_biased")


def mc_reps(ident, T):
    if ident.startswith("hsic"):
        return 200 if T <= 256 else 100
    return 400 if T <= 1024 else 200


def mc_pair(ident, rho, T):
    """Distribution of the stored statistic for one bivariate-normal pair at length T (z-scored as in the pipeline)."""
    key = f"{MC_VERSION}|pair|{ident}|{rho}|{T}"
    c = _mc_cache()
    if key not in c:
        f = mc_stat_fn(ident)
        rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
        vals = []
        for _ in range(mc_reps(ident, T)):
            z = rng.standard_normal((2, T))
            x, y = z[0], rho * z[0] + np.sqrt(1 - rho * rho) * z[1]
            xz, yz = zscore(np.vstack([x, y]))
            vals.append(f(xz, yz))
        vals = np.array(vals)
        _mc_store(key, dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)), n=int(vals.size),
                            frac_neg=float((vals < 0).mean())))
    return _mc_cache()[key]


def mc_null(ident, T, P=5):
    """Distribution of the MEAN over the P(P-1)/2 pairs of P iid N(0,1) series (and of single pair values)."""
    key = f"{MC_VERSION}|null|{ident}|{T}|{P}"
    c = _mc_cache()
    if key not in c:
        f = mc_stat_fn(ident)
        rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))
        means, singles = [], []
        for _ in range(mc_reps(ident, T)):
            X = zscore(rng.standard_normal((P, T)))
            v = [f(X[i], X[j]) for i, j in pairs_upper(P)]
            means.append(np.mean(v)); singles.extend(v)
        means, singles = np.array(means), np.array(singles)
        _mc_store(key, dict(mean_of_mean=float(means.mean()), sd_of_mean=float(means.std(ddof=1)),
                            pair_mean=float(singles.mean()), pair_sd=float(singles.std(ddof=1)),
                            pair_frac_neg=float((singles < 0).mean()), pair_min=float(singles.min()), n=int(means.size)))
    return _mc_cache()[key]


def perm_null(ident, x, y, reps=200, seed=7):
    """Permutation null of f(x, y): y permuted in time, x fixed (so the hsic bandwidth stays that of x)."""
    key = f"{MC_VERSION}|perm|{ident}|{hashlib.sha1(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()}|{reps}"
    c = _mc_cache()
    if key not in c:
        f = mc_stat_fn(ident)
        rng = np.random.default_rng(seed)
        v = np.array([f(x, y[rng.permutation(y.size)]) for _ in range(reps)])
        _mc_store(key, dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), max=float(v.max()), n=reps))
    return _mc_cache()[key]


# --------------------------------------------------------------------------------------------- R
def run_r(jobs):
    """jobs: list of (rid, X (P, T)) -> {rid: list of per-ordered-pair dicts} via r/dcorr_hsic.R."""
    if not jobs:
        return {}
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    with open(EXCHANGE / "jobs.csv", "w") as f:
        f.write("job\n")
        for k, (rid, X) in enumerate(jobs):
            f.write(f"{k}\n")
            np.savetxt(EXCHANGE / f"in_{k}.csv", np.asarray(X).T, delimiter=",", fmt="%.17g")
    env = dict(os.environ, R_LIBS_USER=os.environ.get("R_LIBS_USER", "/home/dan/R/audit-lib"))
    r = subprocess.run(["Rscript", str(CHECKS / "r" / "dcorr_hsic.R"), str(EXCHANGE)], env=env,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"Rscript failed: {r.stderr}")
    print(r.stdout.strip())
    out = {}
    for k, (rid, X) in enumerate(jobs):
        rows = np.genfromtxt(EXCHANGE / f"out_{k}.csv", delimiter=",", names=True)
        out[rid] = [{n: float(row[n]) for n in rows.dtype.names} for row in np.atleast_1d(rows)]
    return out


# ------------------------------------------------------------------------------------------ checks
def chk_scratch(rid, run, S):
    ident = UNIT[run["unit_id"]]
    x = load_x(run)
    X = prep(x, run["path"])
    V = sym_matrix(lambda a, b: stat(ident, a, b), X)
    _, xin = pyspi_npz(rid)
    extra = dict(rtol=1e-10, atol=1e-12, pyspi_max_asym=asym(S), input_maxabs_diff=float(np.max(np.abs(xin - X))))
    notes = "stored matrix vs f(x_min, x_max); both orientations compared"
    if ident.startswith("dcorr"):
        # which canonical form the library value equals, and the other form's value
        D2 = sym_matrix(lambda a, b: kd.dcorr(a, b, biased=ident == "dcorr_biased")["dcor2"], X)
        D1 = sym_matrix(lambda a, b: kd.dcorr(a, b, biased=ident == "dcorr_biased")["dcor"], X)
        extra.update(maxabs_vs_dcor2=float(np.nanmax(np.abs(S - D2))), maxabs_vs_dcor=float(np.nanmax(np.abs(S - D1))))
        if X.shape[0] == 4:
            extra.update(pyspi_01=S[0, 1], pyspi_23=S[2, 3], dcor_01=D1[0, 1], dcor2_01=D2[0, 1], dcor_23=D1[2, 3], dcor2_23=D2[2, 3])
        if ident == "dcorr":
            extra["n_negative_offdiag"] = float(np.sum(S[~np.isnan(S)] < 0) / 2)
            extra["min_offdiag"] = float(np.nanmin(S))
        notes += "; maxabs_vs_dcor2 / maxabs_vs_dcor: distance to the squared / unsquared canonical form"
    else:
        H = {p: hsic_stats(X[p[0]], X[p[1]]) for p in pairs_upper(X.shape[0])}
        Hpv = {p: hsic_stats(X[p[0]], X[p[1]], *per_variable_gamma(X[p[0]], X[p[1]])) for p in H}
        Hrev = {p: hsic_stats(X[p[1]], X[p[0]]) for p in H}
        key = "stored_u" if ident == "hsic" else "stored_b"
        raw = "hu_xy" if ident == "hsic" else "hb_xy"
        norm = "norm_u" if ident == "hsic" else "norm_b"
        PV = np.full_like(S, np.nan); RV = np.full_like(S, np.nan); RAW = np.full_like(S, np.nan)
        NRM = np.full_like(S, np.nan); REP = np.full_like(S, np.nan)
        for (i, j), h in H.items():
            PV[i, j] = PV[j, i] = Hpv[(i, j)][key]
            RV[i, j] = RV[j, i] = Hrev[(i, j)][key]
            RAW[i, j] = RAW[j, i] = h[raw]
            NRM[i, j] = NRM[j, i] = h[norm]
            REP[i, j] = REP[j, i] = kd.hyppo_hsic_replica(X[i], X[j], biased=ident == "hsic_biased")
        rel = lambda A: float(np.nanmax(np.abs(S - A) / np.abs(A)))
        extra.update(maxabs_vs_hyppo_replica=float(np.nanmax(np.abs(S - REP))),
                     maxabs_vs_raw_hsic=float(np.nanmax(np.abs(S - RAW))),
                     maxabs_vs_normalised=float(np.nanmax(np.abs(S - NRM))),
                     max_ratio_stored_over_raw=float(np.nanmax(S / RAW)), min_ratio_stored_over_raw=float(np.nanmin(S / RAW)),
                     maxabs_vs_per_variable_median=float(np.nanmax(np.abs(S - PV))), maxrel_vs_per_variable_median=rel(PV),
                     maxabs_order_swap=float(np.nanmax(np.abs(S - RV))), maxrel_order_swap=rel(RV))
        if X.shape[0] == 4:
            extra.update(pyspi_01=S[0, 1], pyspi_23=S[2, 3], raw_01=RAW[0, 1], raw_23=RAW[2, 3],
                         normalised_01=NRM[0, 1], normalised_23=NRM[2, 3], per_variable_01=PV[0, 1], per_variable_23=PV[2, 3],
                         swapped_01=RV[0, 1], swapped_23=RV[2, 3])
        if ident == "hsic":
            extra["n_negative_offdiag"] = float(np.sum(S[~np.isnan(S)] < 0) / 2)
        notes += ("; raw = unnormalised HSIC (" + ("Song eq. 5 HSIC_u" if ident == "hsic" else "tr(KHLH)/n^2")
                  + "), normalised = raw/sqrt(raw_xx raw_yy); per_variable = separate median per variable; swapped = f(x_max, x_min); "
                  "hyppo_replica = " + KD + "hyppo_hsic_replica (1 - K/max K into Dcorr, sklearn arithmetic)")
    save(rid, V, "exact", "independent-impl", detail_of(ident), notes, **extra)


def chk_dcor(rid, run, S):
    import dcor
    ident = UNIT[run["unit_id"]]
    X = prep(load_x(run), run["path"])
    if ident == "dcorr":
        f = lambda a, b: float(dcor.u_distance_correlation_sqr(a, b, method="naive"))
        other = lambda a, b: float(np.sqrt(max(dcor.u_distance_correlation_sqr(a, b, method="naive"), 0.0)))
        det = "dcor==0.7 dcor.u_distance_correlation_sqr(method='naive') (bias-corrected dCor^2)"
        on = "sqrt(max(u_distance_correlation_sqr, 0))"
    else:
        f = lambda a, b: float(dcor.distance_correlation(a, b, method="naive"))
        other = lambda a, b: float(dcor.distance_correlation_sqr(a, b, method="naive"))
        det = "dcor==0.7 dcor.distance_correlation(method='naive') (V-statistic dCor_n, unsquared)"
        on = "distance_correlation_sqr"
    V = sym_matrix(f, X)
    O = sym_matrix(other, X)
    save(rid, V, "exact", "external-pkg", det, f"other form {on} recorded as maxabs_vs_other_form (must NOT match)",
         rtol=1e-10, atol=1e-12, maxabs_vs_other_form=float(np.nanmax(np.abs(S - O))), pyspi_max_asym=asym(S))


_R_OUT = {}


def chk_dhsic(rid, run, S):
    ident = UNIT[run["unit_id"]]
    rows = _R_OUT[rid]
    P = S.shape[0]
    by = {(int(r["i"]) - 1, int(r["j"]) - 1): r for r in rows}
    V = np.full((P, P), np.nan); DEF = np.full((P, P), np.nan); RAWB = np.full((P, P), np.nan)
    FULL = np.full((P, P), np.nan)
    for i, j in pairs_upper(P):
        r = by[(i, j)]
        if ident == "hsic_biased":
            nb = r["hb_xy"] / np.sqrt(r["hb_xx"] * r["hb_yy"])
            V[i, j] = V[j, i] = np.sqrt(nb)
        else:
            V[i, j] = V[j, i] = r["hu_xy"] / np.sqrt(r["hu_xx"] * r["hu_yy"])
        dn = r["def_xy"] / np.sqrt(r["def_xx"] * r["def_yy"])
        DEF[i, j] = DEF[j, i] = np.sqrt(dn) if ident == "hsic_biased" else dn
        RAWB[i, j] = RAWB[j, i] = r["hb_xy"]
    for (i, j), r in by.items():
        FULL[i, j] = (np.sqrt(r["hb_xy"] / np.sqrt(r["hb_xx"] * r["hb_yy"])) if ident == "hsic_biased"
                      else r["hu_xy"] / np.sqrt(r["hu_xx"] * r["hu_yy"]))
    X = prep(load_x(run), run["path"])
    loc = {p: hsic_stats(X[p[0]], X[p[1]]) for p in pairs_upper(P)}
    raw_diff = max(abs(loc[p]["hb_xy"] - by[p]["hb_xy"]) for p in loc)
    sig_diff = max(abs(np.sqrt(1 / (2 * loc[p]["gamma_x"])) - by[p]["sigma_x"]) for p in loc)
    bw_ratio = [by[p]["bw_x"] / by[p]["sigma_x"] for p in loc]
    extra = dict(rtol=1e-10, atol=1e-12, maxabs_raw_hsic_b_vs_python=raw_diff, maxabs_sigma_vs_python=sig_diff,
                 raw_hsic_b_r_max=float(np.nanmax(RAWB)), stored_over_raw_max=float(np.nanmax(S / RAWB)),
                 maxabs_vs_dhsic_default=float(np.nanmax(np.abs(S - DEF))),
                 dhsic_default_bw_over_sigma_min=float(min(bw_ratio)), dhsic_default_bw_over_sigma_max=float(max(bw_ratio)),
                 unmirrored_asym_max=asym(FULL))
    if P == 4:
        extra.update(pyspi_01=S[0, 1], pyspi_23=S[2, 3], dhsic_default_01=DEF[0, 1], dhsic_default_23=DEF[2, 3],
                     raw_hsic_b_01=RAWB[0, 1], raw_hsic_b_23=RAWB[2, 3])
    if ident == "hsic_biased":
        kind, det = "external-pkg", ("dHSIC==2.2 dhsic(kernel='gaussian.fixed', bandwidth=median(dist(x)) for both) via "
                                     "r/dcorr_hsic.R; value = sqrt(dhsic(x,y)/sqrt(dhsic(x,x) dhsic(y,y)))")
    else:
        kind, det = "independent-impl", ("r/dcorr_hsic.R hsic_u (R re-implementation of Song et al. 2012 eq. 5) on dHSIC==2.2 "
                                         "gaussian_grammat_rcpp Gram matrices, bandwidth = median(dist(x)) for both; normalised")
    save(rid, V, "exact", kind, det, "dhsic_default = dHSIC's own defaults (per-variable median bandwidth sqrt(median(d^2)/2)), "
         "normalised the same way; recorded, not compared", **extra)


def chk_eq3(rid, run, S):
    X = prep(load_x(run), run["path"])
    P = X.shape[0]

    def f(a, b):
        g = kd.median_gamma(a)
        K, L = _gram(a, g), _gram(b, g)
        return hsic_u_eq3(K, L) / np.sqrt(hsic_u_eq3(K, K) * hsic_u_eq3(L, L))
    V = sym_matrix(f, X)
    song = sym_matrix(lambda a, b: hsic_stats(a, b)["norm_u"], X)
    save(rid, V, "exact", "independent-impl", ME + "hsic_u_eq3 (Gretton et al. 2007 NIPS eq. 3, brute force over distinct "
         "index tuples), normalised; gamma = " + KD + "median_gamma(x) for both", "",
         rtol=1e-10, atol=1e-12, maxabs_eq3_vs_song_eq5=float(np.nanmax(np.abs(V - song))))


def chk_affine(rid, run, S):
    ident = UNIT[run["unit_id"]]
    xb = load_x(run, dataset="bivariate_gaussian")
    X = prep(xb, run["path"])
    V = sym_matrix(lambda a, b: stat(ident, a, b), X)
    save(rid, V, "exact", "analytic", detail_of(ident) + " evaluated on bivariate_gaussian (same draw; scaled_offset = scale*D2 + offset)",
         "affine invariance: " + ("dCor is invariant to per-variable scale and shift, so even raw (kernel) scaled_offset equals D2"
                                  if ident.startswith("dcorr") else "z-scoring removes the per-variable affine map"),
         rtol=1e-10, atol=1e-12, pyspi_max_asym=asym(S))


def chk_unmirrored(rid, run, S):
    ident = UNIT[run["unit_id"]]
    X = prep(load_x(run), run["path"])
    V = full_matrix(lambda a, b: stat(ident, a, b), X)
    extra = dict(rtol=1e-10, atol=1e-12, unmirrored_max_asym=asym(S))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.abs(S - S.T) / np.abs(S.T)
    extra["unmirrored_max_rel_asym"] = float(np.nanmax(R))
    notes = "probe probes/dcorr_hsic_probes.py:unmirrored: A[i,j] = bivariate(x = process i, y = process j), no mirror"
    if ident.startswith("hsic") and X.shape[0] == 4:
        PV = sym_matrix(lambda a, b: stat(ident, a, b, hsic_gammas="per_variable"), X)
        for i, j in pairs_upper(4):
            extra[f"A{i}{j}"] = S[i, j]; extra[f"A{j}{i}"] = S[j, i]; extra[f"perVar{i}{j}"] = PV[i, j]
        if run["dataset"] == "scaled_offset":
            D2 = sym_matrix(lambda a, b: stat(ident, a, b), zscore(load_x(run, dataset="bivariate_gaussian")))
            for i, j in pairs_upper(4):
                extra[f"zscored{i}{j}"] = D2[i, j]
            notes += "; zscoredIJ = value on z-scored rows (the pipeline value); perVarIJ = per-variable median bandwidth"
    save(rid, V, "exact", "independent-impl", detail_of(ident) + " on every ordered pair (unmirrored)", notes, **extra)


def chk_reversed(rid, run, S):
    ident = UNIT[run["unit_id"]]
    X = prep(load_x(run), run["path"])                       # processes [3,2,1,0]
    V = sym_matrix(lambda a, b: stat(ident, a, b), X)
    # same pair in the ORIGINAL order (x = lower original index) for comparison
    Xo = prep(load_x(run, processes=None), run["path"])
    orig = run["processes"]
    O = np.full_like(V, np.nan)
    for a, b in pairs_upper(4):
        oa, ob = orig[a], orig[b]
        lo, hi = min(oa, ob), max(oa, ob)
        O[a, b] = O[b, a] = stat(ident, Xo[lo], Xo[hi])
    D = np.abs(S - O)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rl = D / np.abs(O)
    extra = dict(rtol=1e-10, atol=1e-12, maxabs_vs_original_order=float(np.nanmax(D)),
                 maxrel_vs_original_order=float(np.nanmax(Rl)))
    for a, b in pairs_upper(4):
        extra[f"rev_{orig[a]}{orig[b]}"] = S[a, b]; extra[f"orig_{min(orig[a], orig[b])}{max(orig[a], orig[b])}"] = O[a, b]
    save(rid, V, "exact", "independent-impl", detail_of(ident) + " on the reversed process order",
         "processes [3,2,1,0]: the stored value of pair (p,q) is f(x_q, x_p) in the original labels; "
         "maxabs_vs_original_order = |stored(reversed) - stored(original order)| over pairs", **extra)


def chk_null(rid, run, S):
    ident = UNIT[run["unit_id"]]
    T = run["T"]
    mc = mc_null(ident, T)
    vals = S[np.triu_indices(S.shape[0], 1)]
    m = float(vals.mean())
    z = (m - mc["mean_of_mean"]) / mc["sd_of_mean"]
    biased = ident.endswith("biased")
    ok = abs(z) <= 2 and ((vals > 0).all() if biased else True)
    rule = "|mean over 10 pairs - MC mean| <= 2 MC SD of that mean" + (" and every value > 0" if biased else "")
    save(rid, 0.0, "qualitative", "analytic", "population value 0 under independence; finite-T null distribution by Monte Carlo ("
         + ME + "mc_null, " + str(mc["n"]) + " reps of 5 iid N(0,1) series, z-scored, " + ("dcor avl" if ident.startswith("dcorr") else "hsic_stats") + ")",
         rule, pass_flag=float(ok), pyspi_mean=m, pyspi_min=float(vals.min()), pyspi_max=float(vals.max()),
         pyspi_frac_negative=float((vals < 0).mean()), mc_mean_of_mean=mc["mean_of_mean"], mc_sd_of_mean=mc["sd_of_mean"],
         mc_pair_sd=mc["pair_sd"], mc_pair_frac_negative=mc["pair_frac_neg"], mc_pair_min=mc["pair_min"], z=z)


def chk_null_trend(rid, run, S):
    ident = UNIT[run["unit_id"]]
    Ts = (64, 256, 1024, 4096) if ident.startswith("dcorr") else (64, 256, 1024)
    means, sds, mins, mc_means, mc_sds = [], [], [], [], []
    for T in Ts:
        r2 = f"{run['unit_id']}|iid_gaussian|{T}|pipeline|null"
        A, _ = pyspi_npz(r2)
        v = A[np.triu_indices(A.shape[0], 1)]
        means.append(v.mean()); sds.append(v.std(ddof=1)); mins.append(v.min())
        mc = mc_null(ident, T); mc_means.append(mc["pair_mean"]); mc_sds.append(mc["pair_sd"])
    lT = np.log(np.array(Ts, float))
    biased = ident.endswith("biased")
    if biased:
        slope = float(np.polyfit(lT, np.log(means), 1)[0])
        mc_slope = float(np.polyfit(lT, np.log(mc_means), 1)[0])
        ok = all(m > 0 for m in mins) and all(np.diff(means) < 0)
        rule = "every null value > 0 and the mean over pairs strictly decreasing in T"
    else:
        slope = float(np.polyfit(lT, np.log(np.abs(sds)), 1)[0])
        mc_slope = float(np.polyfit(lT, np.log(mc_sds), 1)[0])
        ok = all(np.diff(sds) < 0) and abs(slope - mc_slope) < 0.35
        rule = ("pyspi spread (SD over the 10 null pairs) strictly decreasing in T and its log-log slope within 0.35 of the "
                "Monte Carlo slope")
    extra = {f"mean_T{T}": m for T, m in zip(Ts, means)}
    extra.update({f"sd_T{T}": s for T, s in zip(Ts, sds)})
    extra.update({f"min_T{T}": s for T, s in zip(Ts, mins)})
    extra.update({f"mc_pair_mean_T{T}": s for T, s in zip(Ts, mc_means)})
    extra.update({f"mc_pair_sd_T{T}": s for T, s in zip(Ts, mc_sds)})
    save(rid, 0.0, "qualitative", "analytic", ME + "mc_null across T; log-log slope of " + ("mean" if biased else "SD") + " vs T",
         rule, pass_flag=float(ok), loglog_slope=slope, mc_loglog_slope=mc_slope, **extra)


def chk_truth(rid, run, S):
    ident = UNIT[run["unit_id"]]
    T = run["T"]
    rho = 0.3 if run["processes"] == [0, 1] else 0.8
    pop = population(ident, rho)
    mc = mc_pair(ident, rho, T)
    v = S[0, 1]
    V = np.array([[np.nan, pop], [pop, np.nan]])
    SE = np.array([[np.nan, mc["sd"]], [mc["sd"], np.nan]])
    save(rid, V, "statistical", "analytic",
         {"dcorr": "Szekely, Rizzo & Bakirov 2007 Thm 7(ii) dCor^2(rho) (" + ME + "dcor2_closed)",
          "dcorr_biased": "sqrt of Szekely, Rizzo & Bakirov 2007 Thm 7(ii) dCor^2(rho) (" + ME + "dcor2_closed)",
          "hsic": ME + "hsic_pop_closed normalised HSIC(rho)/HSIC(1), sigma = sqrt(2) Phi^-1(0.75) (quadrature-checked)",
          "hsic_biased": "sqrt of " + ME + "hsic_pop_closed normalised HSIC(rho)/HSIC(1) (quadrature-checked)"}[ident]
         + f"; se = Monte Carlo SD of the estimator at T ({mc['n']} reps, " + ME + "mc_pair)",
         "value = population; pass |pyspi - population| <= 2 SD_MC(T); mc_mean = finite-T expectation",
         se=SE, rho=rho, population=pop, pyspi=v, mc_mean=mc["mean"], mc_sd=mc["sd"], mc_bias=mc["mean"] - pop,
         z_vs_population=(v - pop) / mc["sd"], z_vs_mc_mean=(v - mc["mean"]) / mc["sd"], abs_err=abs(v - pop))


def chk_popcheck(rid, run, S):
    ident = UNIT[run["unit_id"]]
    out = {}
    ok = True
    if ident.startswith("dcorr"):
        import dcor
        for rho in (0.3, 0.8):
            c, q = float(dcor2_closed(rho)), float(dcor2_quad(rho))
            key = f"{MC_VERSION}|bigmc|dcor|{rho}"
            if key not in _mc_cache():
                rng = np.random.default_rng(int(rho * 1000) + 5)
                vals = []
                for _ in range(10):
                    z = rng.standard_normal((2, 2 ** 16))
                    vals.append(float(dcor.u_distance_correlation_sqr(z[0], rho * z[0] + np.sqrt(1 - rho * rho) * z[1], method="avl")))
                _mc_store(key, dict(mean=float(np.mean(vals)), se=float(np.std(vals, ddof=1) / np.sqrt(len(vals)))))
            mc = _mc_cache()[key]
            out.update({f"closed_{rho}": c, f"quad_{rho}": q, f"bigmc_{rho}": mc["mean"], f"bigmc_se_{rho}": mc["se"]})
            ok &= abs(c - q) <= 1e-9 and abs(mc["mean"] - c) <= 3 * mc["se"]
        out["closed_rho1"] = float(dcor2_closed(1.0))
        det = ME + "dcor2_closed (Thm 7(ii)) vs dcor2_quad (quadrature of E|X-X'||Y-Y'| terms) vs n=65536 U-statistic MC (dcor avl, 10 reps)"
        rule = "|closed - quad| <= 1e-9 and |bigMC - closed| <= 3 se"
    else:
        for rho in (0.3, 0.8):
            c, q = hsic_pop_closed(rho), hsic_pop_quad(rho)
            key = f"{MC_VERSION}|bigmc|hsic|{rho}"
            if key not in _mc_cache():
                m, se = hsic_pop_mc(rho)
                _mc_store(key, dict(mean=m, se=se))
            mc = _mc_cache()[key]
            out.update({f"closed_{rho}": c[2], f"quad_{rho}": q[2], f"closed_raw_{rho}": c[0], f"quad_raw_{rho}": q[0],
                        f"bigmc_{rho}": mc["mean"], f"bigmc_se_{rho}": mc["se"]})
            ok &= abs(c[2] - q[2]) <= 1e-8 * max(1.0, abs(c[2])) and abs(c[0] - q[0]) <= 1e-9 and abs(mc["mean"] - c[2]) <= 3 * mc["se"]
        out["closed_raw_rho1"] = hsic_pop_closed(1.0)[1]
        out["sigma_pop"] = SIGMA_POP
        det = (ME + "hsic_pop_closed (det(I + A Sigma)^-1/2) vs hsic_pop_quad (2-D quadrature of each expectation) vs "
               "quadruple Monte Carlo (10 x 2e6), sigma = sqrt(2) Phi^-1(0.75)")
        rule = "|closed - quad| <= 1e-8 (normalised) and 1e-9 (raw), |MC - closed| <= 3 se"
    save(rid, 0.0, "qualitative", "analytic", det, rule + "; independent of the pyspi value", pass_flag=float(ok), **out)


def chk_detected(rid, run, S):
    ident = UNIT[run["unit_id"]]
    X = prep(load_x(run), run["path"])
    T = run["T"]
    nul = mc_null(ident, T)
    extra = {}
    ok = True
    for i, j in ((0, 1), (2, 3)):
        pn = perm_null(ident, X[i], X[j])
        v = S[i, j]
        zp = (v - pn["mean"]) / pn["sd"]
        zi = (v - nul["pair_mean"]) / nul["pair_sd"]
        ok &= zp > 5 and zi > 5 and v > pn["max"]
        extra.update({f"pyspi_{i}{j}": v, f"pearson_{i}{j}": float(np.corrcoef(X[i], X[j])[0, 1]),
                      f"perm_mean_{i}{j}": pn["mean"], f"perm_sd_{i}{j}": pn["sd"], f"perm_max_{i}{j}": pn["max"],
                      f"z_perm_{i}{j}": zp, f"z_iid_null_{i}{j}": zi})
    save(rid, 0.0, "qualitative", "literature-only",
         "dependence detection where Pearson ~ 0 (Szekely et al. 2007 Thm 3: dCor = 0 iff independent; Gretton et al. 2005 "
         "Thm 4 for HSIC with a characteristic kernel); null by 200 time permutations of y (" + ME + "perm_null) and the iid "
         "Gaussian Monte Carlo null (" + ME + "mc_null)",
         "pairs (0,1) x1 = x0^2 + noise and (2,3) circle: value > max of 200 permutations and z > 5 against both nulls",
         pass_flag=float(ok), **extra)


def chk_detrend(rid, run, S):
    ident = UNIT[run["unit_id"]]
    xb = load_x(run, dataset="bivariate_gaussian")
    X = zscore(lstsq_detrend(xb))
    V = sym_matrix(lambda a, b: stat(ident, a, b), X)
    save(rid, V, "tolerance", "analytic", detail_of(ident) + " on lstsq-detrended, z-scored bivariate_gaussian (same draw)",
         "detrend(D2 + slope t) = detrend(D2); pyspi detrends with scipy.signal.detrend", rtol=1e-10, atol=1e-12)


def chk_trend_contrast(rid, run, S):
    ident = UNIT[run["unit_id"]]
    xb = zscore(load_x(run, dataset="bivariate_gaussian"))
    D2 = sym_matrix(lambda a, b: stat(ident, a, b), xb)
    extra = {}
    ok = True
    for i, j in ((1, 2), (1, 3), (2, 3)):            # both rows trended (slopes 0.05, -0.02, 0.1)
        extra[f"pyspi_{i}{j}"] = S[i, j]; extra[f"d2_{i}{j}"] = D2[i, j]
        # (1,2), (1,3) are independent in D2: require an inflation of at least 0.3. (2,3) is the rho = 0.8 pair, whose
        # D2 value (0.76 on the root scale) is within 0.3 of the bound 1, so only an increase is required there.
        ok &= S[i, j] > D2[i, j] + (0.3 if (i, j) != (2, 3) else 0.0)
    save(rid, 0.0, "qualitative", "analytic", "trended rows without detrending vs the untrended draw (" + detail_of(ident) + ")",
         "default pipeline does not detrend: pairs of trended rows read as strongly dependent (independent pairs (1,2), (1,3): "
         "value > D2 value + 0.3; dependent pair (2,3): value > D2 value, since D2 + 0.3 exceeds the bound 1 on the root scale)",
         pass_flag=float(ok), **extra)


def chk_scale(rid, run, S):
    ident = UNIT[run["unit_id"]]
    other_unit = run["unit_id"].replace("biased-True", "biased-False")
    tag = f"{other_unit}|{run['dataset']}|{run['T']}|{run['path']}|scratch"
    U, _ = pyspi_npz(tag)
    extra = {}
    ok = True
    for i, j in ((0, 1), (2, 3)):
        b, u = S[i, j], U[i, j]
        extra.update({f"biased_{i}{j}": b, f"biased_sq_{i}{j}": b * b, f"unbiased_{i}{j}": u,
                      f"absdiff_biased_unbiased_{i}{j}": abs(b - u), f"absdiff_biasedsq_unbiased_{i}{j}": abs(b * b - u)})
        ok &= abs(b * b - u) < 0.25 * abs(b - u)
    save(rid, 0.0, "qualitative", "independent-impl", "pyspi biased vs pyspi unbiased on the same draw (" + tag + ")",
         "scale mismatch: biased^2 is close to unbiased (|b^2 - u| < 0.25 |b - u|) on the dependent pairs",
         pass_flag=float(ok), **extra)


def chk_gamma(rid, run, S):
    """Probe probes/dcorr_hsic_probes.py:gamma_used returns the gamma hyppo applied to y's kernel for each ordered pair."""
    X = prep(load_x(run), run["path"])
    P = X.shape[0]
    V = np.full((P, P), np.nan); PV = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = kd.median_gamma(X[i])
                PV[i, j] = kd.median_gamma(X[j])
    extra = dict(rtol=1e-8, atol=0.0, maxrel_vs_y_own_median=float(np.nanmax(np.abs(S - PV) / PV)))
    for i in range(P):
        extra[f"gamma_x{i}"] = float(kd.median_gamma(X[i]))
        extra[f"sigma_x{i}"] = float(np.sqrt(1 / (2 * kd.median_gamma(X[i]))))
    save(rid, V, "tolerance", "independent-impl",
         KD + "median_gamma(x_i) (median of off-diagonal |x_a - x_b| of the FIRST argument, gamma = 1/(2 med^2))",
         "value[i, j] = gamma of x = process i, the claim that hyppo uses x's bandwidth for y's kernel; "
         "maxrel_vs_y_own_median = distance to the per-variable alternative gamma(x_j); rtol 1e-8 covers the log recovery",
         **extra)


REFS = {"gamma": chk_gamma, "scratch": chk_scratch, "dcor": chk_dcor, "dhsic": chk_dhsic, "eq3": chk_eq3, "affine": chk_affine,
        "unmirrored": chk_unmirrored, "reversed": chk_reversed, "null": chk_null, "null-trend": chk_null_trend,
        "truth-p01": chk_truth, "truth-p23": chk_truth, "popcheck": chk_popcheck, "detected": chk_detected,
        "detrend": chk_detrend, "trend-contrast": chk_trend_contrast, "scale": chk_scale}


def main():
    OUT_REF.mkdir(parents=True, exist_ok=True)
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    index = json.loads((OUT_PYSPI / "index" / f"{FAMILY}.json").read_text())
    runs = [r for r in man["runs"] if index.get(r["run_id"], {}).get("status") == "ok"]
    rjobs = [(r["run_id"], prep(load_x(r), r["path"])) for r in runs if r["run_id"].rsplit("|", 1)[1] == "dhsic"]
    _R_OUT.update(run_r(rjobs))
    n_ok = n_err = 0
    for r in runs:
        rid = r["run_id"]
        tag = rid.rsplit("|", 1)[1]
        try:
            S, _ = pyspi_npz(rid)
            REFS[tag](rid, r, S)
            n_ok += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
            n_err += 1
    print(f"refs/dcorr_hsic.py: {n_ok} references written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
