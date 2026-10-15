#!/usr/bin/env python3
"""References for the it_kraskov family (COMPARISON venv: numpy/scipy/sklearn only; no pyspi, no JIDT).

For every run of ``audit/checks/manifest/it_kraskov.json`` this script writes
``audit/checks/out/ref/<run_id>.npz`` in the compare.py contract.  The last ``|`` field of the
run_id (the tag) selects the check.

Estimator reference ("JIDT-exact KSG1", independent of JIDT):
  * columns z-scored with the N-1 std (_kernels_infotheory.zscore_jidt; MatrixUtils.normalise),
  * the 1e-8 Gaussian jitter regenerated with a pure-Python port of java.util.Random
    (48-bit LCG, nextGaussian polar method; seeded with NOISE_SEED=42 at every
    finaliseAddObservations, rows in order, var1/source columns then var2/dest columns then
    cond columns: JIDT a9b6b3e MutualInfoMultiVariateCommon.java:626-642,
    ConditionalMutualInfoMultiVariateCommon.java:766-785),
  * then _kernels_infotheory.mi_ksg1 / cmi_ksg1 with normalise=False (cKDTree max norm, k = 4,
    strict marginal counts, Theiler exclusion |s - t| <= W, psi(N) unadjusted),
  * guard: NaN when N <= k + 2W (JIDT throws, pyspi returns NaN).
  TE: _kernels_infotheory.te_embedding (JIDT sample layout) + the CMI above.  Auto-embedding:
  from-scratch MAX_CORR_AIS search (k = 1..10, tau = 1..4, tau loop broken at k = 1, strict
  maximum of the KSG1 AIS without bias correction; one failing candidate -> NaN for the pair).
  DCE window: from-scratch explicit lag-loop ACF, W = int(2 sum_{k=0}^{T-1} r_i(k) r_j(k)).

Tags (see the function of the same name, prefixed ``chk_``): ksg, nojitter, sklearn, truth, null,
direction, invariance, ordered, threads, seeds, copy, copyprops, embed, fixed, direct, semantics,
props, lagloop, bartlett, pop, growth, detrend, and (added after review) copyk6, copypropsk6,
reversal, guard.

Run under the comparison venv:  python audit/checks/refs/it_kraskov.py [run_id substrings ...]
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
from scipy.special import digamma

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_infotheory as K  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "it_kraskov"
OUT = CHECKS / "out" / "ref"
PYS = CHECKS / "out" / "pyspi"
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"
KREF = "refs/_kernels_infotheory.py"
ME = "refs/it_kraskov.py"
KNN = 4
SEED = 42
NOISE = 1e-8
W_LIST = (0, 1, 2, 5, 20, 85)          # mirrors probes/it_kraskov_probes.py:W_LIST
JIDT = "JIDT a9b6b3e (jar built 2024-04-21)"


# ============================================================================ java.util.Random
class JavaRandom:
    """Pure-Python java.util.Random: 48-bit LCG (0x5DEECE66D, 0xB), nextDouble = 53-bit,
    nextGaussian = Marsaglia polar method with the cached second deviate (JDK Random.java)."""
    MASK = (1 << 48) - 1

    def __init__(self, seed):
        self.s = (int(seed) ^ 0x5DEECE66D) & self.MASK
        self.have = False
        self.nng = 0.0

    def _next(self, bits):
        self.s = (self.s * 0x5DEECE66D + 0xB) & self.MASK
        r = self.s >> (48 - bits)
        return r - (1 << 32) if r >= (1 << 31) else r

    def next_double(self):
        return ((self._next(26) << 27) + self._next(27)) * (1.0 / (1 << 53))

    def next_gaussian(self):
        if self.have:
            self.have = False
            return self.nng
        while True:
            v1 = 2.0 * self.next_double() - 1.0
            v2 = 2.0 * self.next_double() - 1.0
            s = v1 * v1 + v2 * v2
            if 0.0 < s < 1.0:
                break
        m = math.sqrt(-2.0 * math.log(s) / s)
        self.nng = v2 * m
        self.have = True
        return v1 * m


def jidt_prep(blocks, seed=SEED, level=NOISE):
    """z-score every column (N-1 std), then add the seeded jitter row by row, block by block."""
    out = [K.zscore_jidt(b) for b in blocks]
    if seed is None:
        return out
    rng = JavaRandom(seed)
    N = out[0].shape[0]
    widths = [b.shape[1] for b in out]
    noise = np.empty((N, sum(widths)))
    flat = [rng.next_gaussian() for _ in range(N * sum(widths))]
    noise[:] = np.asarray(flat).reshape(N, sum(widths))
    res, c = [], 0
    for b, w in zip(out, widths):
        res.append(b + level * noise[:, c:c + w])
        c += w
    return res


# ============================================================================ estimators
def mi_jidt(x, y, W=0, seed=SEED, k=KNN):
    x, y = K._as2d(x), K._as2d(y)
    N = x.shape[0]
    if N <= k + 2 * W:
        return float("nan")
    X, Y = jidt_prep([x, y], seed)
    return K.mi_ksg1(X, Y, k=k, theiler=W, normalise=False)


def cmi_jidt(x, y, z, W=0, seed=SEED, k=KNN):
    x, y, z = K._as2d(x), K._as2d(y), K._as2d(z)
    N = x.shape[0]
    if N <= k + 2 * W:
        return float("nan")
    X, Y, Z = jidt_prep([x, y, z], seed)
    return K.cmi_ksg1(X, Y, Z, k=k, theiler=W, normalise=False)


def te_jidt(src, targ, emb=(1, 1, 1, 1), W=0, seed=SEED):
    k, kt, l, lt = emb
    sp, dn, dp = K.te_embedding(src, targ, k, kt, l, lt, 1)
    return cmi_jidt(sp, dn, dp, W=W, seed=seed)


class Guard(Exception):
    pass


def ais_jidt(y, k, tau, W=0, seed=SEED):
    y = np.ravel(y)
    s = (k - 1) * tau
    n = y.size - s - 1
    if n <= KNN + 2 * W:
        raise Guard(f"AIS candidate k={k} tau={tau}: N={n} <= {KNN}+2*{W}")
    past = K.embed(y, k, tau, start=s, n=n)
    nxt = y[s + 1:s + 1 + n]
    return mi_jidt(past, nxt, W=W, seed=seed)


def select_embedding(y, W=0, seed=SEED, kmax=10, taumax=4):
    """MAX_CORR_AIS (ActiveInfoStorageCalculatorViaMutualInfo.java a9b6b3e :579-615)."""
    best, bk, bt = -np.inf, 1, 1
    for k in range(1, kmax + 1):
        for tau in range(1, taumax + 1):
            a = ais_jidt(y, k, tau, W, seed)
            if a > best:
                best, bk, bt = a, k, tau
            if k == 1:
                break
    return bk, bt


def te_auto_jidt(src, targ, W=0, seed=SEED, cache=None, key=None):
    """(value, (k, kt, l, lt)); NaN value and None embedding when any AIS candidate trips the guard."""
    try:
        if cache is not None and (key[1], W) in cache:
            k, kt = cache[(key[1], W)]
        else:
            k, kt = select_embedding(targ, W, seed)
            if cache is not None:
                cache[(key[1], W)] = (k, kt)
        if cache is not None and (key[0], W) in cache:
            l, lt = cache[(key[0], W)]
        else:
            l, lt = select_embedding(src, W, seed)
            if cache is not None:
                cache[(key[0], W)] = (l, lt)
    except Guard:
        return float("nan"), None
    return te_jidt(src, targ, (k, kt, l, lt), W, seed), (k, kt, l, lt)


# ============================================================================ DCE window (from scratch)
def acf_lagloop(x):
    x = np.ravel(np.asarray(x, dtype=np.float64))
    x = (x - x.mean()) / x.std()
    T = x.size
    den = float(np.dot(x, x))
    return np.array([float(np.dot(x[:T - k], x[k:])) / den for k in range(T)])


def window_raw(z):
    P = z.shape[0]
    acfs = [acf_lagloop(z[i]) for i in range(P)]
    Wr = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            Wr[i, j] = Wr[j, i] = 2.0 * float(np.dot(acfs[i], acfs[j]))
    return Wr


def window_raw_fast(z):
    """Same rule through the kernel's np.correlate ACF (used inside Monte Carlo loops only)."""
    P = z.shape[0]
    acfs = [K.acf_pyspi(z[i]) for i in range(P)]
    Wr = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            Wr[i, j] = Wr[j, i] = 2.0 * float(np.dot(acfs[i], acfs[j]))
    return Wr


def window_int(Wr):
    Wi = np.zeros(Wr.shape, dtype=np.int64)
    m = np.isfinite(Wr)
    Wi[m] = np.trunc(Wr[m]).astype(np.int64)
    return Wi


# ============================================================================ unit definitions
#   key, DCE, undirected
UNIT = {
    "infotheory.mi.kraskov": ("mi", False, True),
    "infotheory.mi.kraskov_DCE": ("mi", True, True),
    "infotheory.tlmi.kraskov": ("tlmi", False, False),
    "infotheory.tlmi.kraskov_DCE": ("tlmi", True, False),
    "infotheory.te.kraskov_k-1_kt-1_l-1_lt-1": ("te1", False, False),
    "infotheory.te.kraskov_DCE_k-1_kt-1_l-1_lt-1": ("te1", True, False),
    "infotheory.te.kraskov_DCE_k-2_kt-1_l-1_lt-1": ("te2", True, False),
    "infotheory.te.kraskov_auto-MAX_CORR_AIS_kmax-10_taumax-4": ("auto", False, False),
    "infotheory.te.kraskov_DCE_auto-MAX_CORR_AIS_kmax-10_taumax-4": ("auto", True, False),
}
KDESC = {"mi": "I(x_i; x_j)", "tlmi": "I(x_i(t); x_j(t+1)) on (x_i[:-1], x_j[1:])",
         "te1": "TE(x_i -> x_j), k=kt=l=lt=1, u=1", "te2": "TE(x_i -> x_j), k=2, kt=l=lt=1, u=1",
         "auto": "TE(x_i -> x_j), (k,kt,l,lt) by MAX_CORR_AIS k<=10 tau<=4"}


def pair_value(key, a, b, W, seed, cache=None, ij=None, emb_out=None):
    if key == "mi":
        return mi_jidt(a, b, W, seed)
    if key == "tlmi":
        return mi_jidt(a[:-1], b[1:], W, seed)
    if key == "te1":
        return te_jidt(a, b, (1, 1, 1, 1), W, seed)
    if key == "te2":
        return te_jidt(a, b, (2, 1, 1, 1), W, seed)
    if key == "auto":
        v, emb = te_auto_jidt(a, b, W, seed, cache, ij)
        if emb_out is not None:
            emb_out[ij] = emb
        return v
    raise KeyError(key)


def unit_matrix(unit, z, seed=SEED, fast_window=False, force_directed=False, emb_out=None):
    key, dce, und = UNIT[unit]
    P = z.shape[0]
    Wi = window_int(window_raw_fast(z) if fast_window else window_raw(z)) if dce else np.zeros((P, P), int)
    A = np.full((P, P), np.nan)
    cache = {}
    for i in range(P):
        for j in range(P):
            if i == j or (und and not force_directed and j < i):
                continue
            A[i, j] = pair_value(key, z[i], z[j], int(Wi[i, j]), seed, cache, (i, j), emb_out)
            if und and not force_directed:
                A[j, i] = A[i, j]
    return A, Wi


# ============================================================================ data helpers
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


def load_pyspi(rid):
    with np.load(PYS / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def off(A):
    return A[~np.eye(A.shape[-1], dtype=bool)]


def rng_for(rid):
    return np.random.default_rng(zlib.crc32(rid.encode()) + 20261015)


def maxabs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")


# ============================================================================ writers
def write(rid, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{rid}.npz", **{k: np.asarray(v) for k, v in kw.items()})


def write_tol(rid, value, kind, detail, notes="", rtol=1e-10, atol=1e-12, crit="tolerance", **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), criterion=crit, rtol=rtol, atol=atol,
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


def write_stat(rid, value, se, detail, population, S, R, notes="", **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), se=np.asarray(se, dtype=np.float64),
          criterion="statistical", reference_kind="independent-impl", reference_detail=detail, notes=notes,
          population_kind="analytic",
          population=np.asarray(population, dtype=np.float64),
          pyspi_minus_population_maxabs=maxabs(S, population),
          mc_mean_minus_population_maxabs=maxabs(value, population),
          z_max=float(np.nanmax(np.abs(S - value) / se)), mc_replicates=R, **extra)


def write_qual(rid, ok, detail, notes="", kind="analytic", **extra):
    write(rid, value=np.asarray(np.nan), criterion="qualitative", pass_flag=int(bool(ok)),
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


EST_DETAIL = (f"{ME}:mi_jidt/cmi_jidt/te_jidt/te_auto_jidt (N-1 z-score, java.util.Random(42) jitter port, "
              f"{KREF}:mi_ksg1/cmi_ksg1 normalise=False, k=4, Theiler |s-t|<=W, guard N<=k+2W -> NaN); "
              f"DCE window {ME}:window_raw (explicit lag loop); {VERS}")


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
    return np.transpose(X[burn:], (1, 2, 0))


def sim_pair(rho, T, R, rng):
    a = rng.standard_normal((R, T))
    b = rho * a + math.sqrt(1 - rho ** 2) * rng.standard_normal((R, T))
    return np.stack([a, b], axis=1)


def sim_delayed01(T, R, rng, noise=0.1):
    base = rng.standard_normal((R, T + 1))
    return np.stack([base[:, 1:], base[:, :T] + noise * rng.standard_normal((R, T))], axis=1)


def simulator(ds_name, ds, procs):
    if ds_name == "bivariate_gaussian":
        rho = float(ds["rho"][0 if procs == [0, 1] else 1])
        return lambda T, R, rng: sim_pair(rho, T, R, rng)
    if ds_name == "gauss_ar1_pair":
        return lambda T, R, rng: sim_var([ds["A"]], ds["Sigma_E"], T, R, rng)
    if ds_name == "var2_bivariate":
        return lambda T, R, rng: sim_var([ds["A1"], ds["A2"]], ds["Sigma"], T, R, rng)
    if ds_name == "delayed_noise" and procs == [0, 1]:
        return lambda T, R, rng: sim_delayed01(T, R, rng)
    if ds_name == "iid_gaussian":
        P = 5 if procs is None else len(procs)
        return lambda T, R, rng: rng.standard_normal((R, P, T))
    raise KeyError((ds_name, procs))


# ============================================================================ population values
def autocov(ds_name, ds, H):
    if ds_name == "gauss_ar1_pair":
        return make_data.var_autocov([ds["A"]], ds["Sigma_E"], H)
    if ds_name == "var2_bivariate":
        return make_data.var_autocov([ds["A1"], ds["A2"]], ds["Sigma"], H)
    raise KeyError(ds_name)


def block_cov(G, items):
    """Cov of the scalars x_comp(t - lag) for items [(comp, lag)]; G[h] = E[x_{t+h} x_t^T]."""
    n = len(items)
    C = np.empty((n, n))
    for a, (ca, la) in enumerate(items):
        for b, (cb, lb) in enumerate(items):
            h = lb - la
            C[a, b] = G[h][ca, cb] if h >= 0 else G[-h][cb, ca]
    return C


def cond_var(C):
    if C.shape[0] == 1:
        return C[0, 0]
    c = C[0, 1:]
    return C[0, 0] - c @ np.linalg.solve(C[1:, 1:], c)


def pop_te(G, src, tgt, k, l):
    """Gaussian TE(src -> tgt) = 0.5 ln(Var(y_{t+1}|y past k) / Var(y_{t+1}|y past k, x past l)),
    u = 1, delays 1 (Schreiber 2000 eq. 4 for Gaussian processes, Barnett et al. 2009)."""
    r = [(tgt, 0)] + [(tgt, L) for L in range(1, k + 1)]
    f = r + [(src, L) for L in range(1, l + 1)]
    return 0.5 * math.log(cond_var(block_cov(G, r)) / cond_var(block_cov(G, f)))


def population(unit, ds_name, ds, procs, T):
    key, dce, und = UNIT[unit]
    pop = np.full((2, 2), np.nan)
    if ds_name == "bivariate_gaussian":
        v = float(ds["true_mi_nats"][0 if procs == [0, 1] else 1])
        pop[0, 1] = pop[1, 0] = v
        return pop, f"-0.5 ln(1-rho^2) = true_mi_nats ({v:.6f})"
    if ds_name == "delayed_noise":
        big = 0.5 * math.log(101.0)
        pop[0, 1], pop[1, 0] = big, 0.0
        return pop, "x1(t+1) = x0(t) + 0.1 e: 0 -> 1 = 0.5 ln(101) = 2.307573, 1 -> 0 = 0 (x0 white)"
    G = autocov(ds_name, ds, 30)
    for i in range(2):
        for j in range(2):
            if i == j:
                continue
            if key == "mi":
                r = G[0][i, j] / math.sqrt(G[0][i, i] * G[0][j, j])
                pop[i, j] = -0.5 * math.log(1 - r * r)
            elif key == "tlmi":
                r = G[1][j, i] / math.sqrt(G[0][i, i] * G[0][j, j])     # corr(x_i(t), x_j(t+1))
                pop[i, j] = -0.5 * math.log(1 - r * r)
            elif key == "te1":
                pop[i, j] = pop_te(G, i, j, 1, 1)
            elif key == "te2":
                pop[i, j] = pop_te(G, i, j, 2, 1)
    det = {"mi": "-0.5 ln(1 - rho_ij(0)^2) from Gamma(0)", "tlmi": "-0.5 ln(1 - corr(x_i(t), x_j(t+1))^2) from Gamma(1)",
           "te1": "Gaussian TE k=l=1 from the VAR autocovariances", "te2": "Gaussian TE k=2, l=1 from the VAR autocovariances"}[key]
    return pop, det


def pop_tau(ds_name, ds, H=400):
    """Population Bartlett factor tau = 1 + 2 sum_{k>=1} rho_0(k) rho_1(k) for the (0,1) pair."""
    if ds_name == "iid_gaussian":
        return 1.0
    G = autocov(ds_name, ds, H)
    r0 = G[:, 0, 0] / G[0, 0, 0]
    r1 = G[:, 1, 1] / G[0, 1, 1]
    return 1.0 + 2.0 * float(np.sum(r0[1:] * r1[1:]))


# ============================================================================ checks: estimator
N_OFFSET = {"mi": 0, "tlmi": 1, "te1": 1, "te2": 2}     # KSG sample count N = T - offset


def chk_ksg(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    A, Wi = unit_matrix(unit, z)
    S = load_pyspi(rid)
    key, dce, und = UNIT[unit]
    extra = dict(sym_maxabs_pyspi=maxabs(S, S.T), nan_count_ref=int(np.isnan(off(A)).sum()))
    if dce:
        extra["W_max"] = int(Wi.max())
        extra["W_min_offdiag"] = int(off(Wi).min())
        if key in N_OFFSET:
            N = z.shape[1] - N_OFFSET[key]
            extra["guard_N"] = N
            extra["guard_margin_min"] = int(N - (KNN + 2 * off(Wi).max()))    # > 0: guard N <= k + 2W not tripped
    if key != "mi" and S.shape[0] == 2:
        extra.update(A01=float(S[0, 1]), A10=float(S[1, 0]))
    write_tol(rid, A, "independent-impl", EST_DETAIL,
              notes=f"{KDESC[key]}{' with Theiler window W_ij' if dce else ''}; path {run['path']}", **extra)


def chk_nojitter(run, ds, x, z):
    """The shared kernel itself (no jitter at all) vs pyspi: residual quoted."""
    rid = run["run_id"]
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            A[i, j] = A[j, i] = K.mi_ksg1(z[i], z[j], k=KNN)
    S = load_pyspi(rid)
    write_tol(rid, A, "independent-impl", f"{KREF}:mi_ksg1(k=4, normalise=True, noise_level=0) (cKDTree Chebyshev); {VERS}",
              notes="no jitter at all: on tie-free data the 1e-8 jitter moves no neighbour count", residual=maxabs(S, A))


def chk_sklearn(run, ds, x, z):
    import sklearn
    rid = run["run_id"]
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            A[i, j] = A[j, i] = K.mi_ksg1_sklearn(z[i], z[j], k=KNN, seed=0)
    S = load_pyspi(rid)
    own = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            own[i, j] = own[j, i] = mi_jidt(z[i], z[j])
    write_tol(rid, A, "external-pkg",
              f"sklearn=={sklearn.__version__} feature_selection.mutual_info_regression(n_neighbors=4, random_state=0) via {KREF}:mi_ksg1_sklearn",
              notes="sklearn: KSG1 with KDTree chebyshev, scale(with_mean=False), its own 1e-10 noise, clipped at 0; "
                    "pairs chosen with clearly positive MI so the clip is inactive",
              rtol=0.0, atol=1e-12, residual=maxabs(S, A), sklearn_vs_own_ref=maxabs(own, A))


def _mc(fn, sims):
    vals = np.array([fn(s) for s in sims])
    return vals.mean(axis=0), vals.std(axis=0, ddof=1)


def chk_truth(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    T, procs = int(run["T"]), run.get("processes")
    pop, det = population(unit, run["dataset"], ds, procs, T)
    R = 100 if T >= 4096 else 200
    sims = simulator(run["dataset"], ds, procs)(T, R, rng_for(rid))
    m, s = _mc(lambda q: unit_matrix(unit, zs(q), seed=None, fast_window=True)[0], sims)
    S = load_pyspi(rid)
    write_stat(rid, m, s, f"population {det}; reference = Monte Carlo mean/SD of the same KSG1 estimator "
                          f"({ME}:unit_matrix without jitter) over {R} draws of the generating model at T={T}",
               pop, S, R, pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]), population_01=float(pop[0, 1]),
               population_10=float(pop[1, 0]), mc_mean_01=float(m[0, 1]), mc_mean_10=float(m[1, 0]),
               mc_sd_01=float(s[0, 1]), mc_sd_10=float(s[1, 0]))


def chk_null(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    key, dce, und = UNIT[unit]
    T = int(run["T"])
    S = load_pyspi(rid)
    P = S.shape[0]
    sel = (lambda A: A[np.triu_indices(P, 1)]) if und else off
    vals = sel(S)
    stat = float(np.nanmean(vals))
    R = 40 if key == "auto" else 200
    sims = rng_for(rid).standard_normal((R, P, T))
    m, s = _mc(lambda q: float(np.nanmean(sel(unit_matrix(unit, zs(q), seed=None, fast_window=True)[0]))), sims)
    ok = abs(stat - m) <= 2 * s
    write_qual(rid, ok, f"mean over {'unordered' if und else 'ordered'} pairs vs Monte Carlo distribution of that mean "
                        f"({R} iid N(0,1) draws, same estimator without jitter); independence => population 0",
               kind="independent-impl", population_kind="analytic",
               pyspi_mean=stat, mc_mean=float(m), mc_sd=float(s), z=float((stat - m) / s),
               pyspi_min=float(np.nanmin(vals)), pyspi_max=float(np.nanmax(vals)),
               frac_negative=float(np.mean(vals < 0)), mc_replicates=R)


def chk_direction(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    key = UNIT[unit][0]
    S = load_pyspi(rid)
    name = run["dataset"]
    extra = {}
    if name == "gauss_ar1_pair":
        G = autocov(name, ds, 30)
        if key == "tlmi":
            p01 = -0.5 * math.log(1 - (G[1][1, 0] / math.sqrt(G[0][0, 0] * G[0][1, 1])) ** 2)
            p10 = -0.5 * math.log(1 - (G[1][0, 1] / math.sqrt(G[0][0, 0] * G[0][1, 1])) ** 2)
        elif key in ("te1", "auto"):
            p01, p10 = pop_te(G, 0, 1, 1, 1), pop_te(G, 1, 0, 1, 1)
        else:
            p01, p10 = pop_te(G, 0, 1, 2, 1), pop_te(G, 1, 0, 2, 1)
        expect_01 = p01 > p10
        extra.update(population_01=p01, population_10=p10)
        rule = f"A[0,1] {'>' if expect_01 else '<'} A[1,0] (Gaussian population values for this quantity; A = [[0.5,0.3],[0.2,0.5]])"
    else:
        expect_01 = True
        rule = {"delayed_noise": "A[0,1] > A[1,0] (x0 leads x1 by one sample)",
                "var2_bivariate": "A[0,1] > A[1,0] (x0 -> x1 only, A1[1,0] = 0.5)",
                "logistic_maps_chaotic": "A[0,1] > A[1,0] (x -> y only, beta_yx = 0.1, beta_xy = 0)"}[name]
    ok = (S[0, 1] > S[1, 0]) if expect_01 else (S[1, 0] > S[0, 1])
    if name == "delayed_noise":
        P = S.shape[0]
        for k in range(1, P):
            extra[f"dn_A0{k}"] = float(S[0, k])
            extra[f"dn_A{k}0"] = float(S[k, 0])
    write_qual(rid, ok, f"ground truth {name}: {rule}; A[i,j] read as source i -> target j", A01=float(S[0, 1]),
               A10=float(S[1, 0]), asym_maxabs=maxabs(S, S.T), **extra)


def chk_invariance(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    d2 = make_data.load("bivariate_gaussian", int(run["T"]))["x"]
    A, _ = unit_matrix(unit, zs(d2))
    write_tol(rid, A, "independent-impl", EST_DETAIL + "; applied to the z-scored bivariate_gaussian rows (same draw)",
              notes="scaled_offset = scale * D2 + offset: after z-scoring the input equals D2's up to round-off")


def chk_ordered(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    A, _ = unit_matrix(unit, z, force_directed=True)
    S = load_pyspi(rid)
    write_tol(rid, A, "independent-impl", EST_DETAIL + "; every ordered pair bivariate(i, j) with x = row i, y = row j",
              notes="probe ordered_mi: Directed.multivariate (all ordered pairs, no mirror)",
              ordered_asym_maxabs_pyspi=maxabs(S, S.T), ordered_asym_maxabs_ref=maxabs(A, A.T))


def chk_threads(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    S = load_pyspi(rid)                                     # (3, P, P): default, NUM_THREADS=1, default again
    A, _ = unit_matrix(unit, z)
    d_thr, d_rep = maxabs(S[0], S[1]), maxabs(S[0], S[2])
    d_ref = max(maxabs(S[k], A) for k in range(3))
    ok = d_thr <= 1e-13 and d_rep == 0.0 and d_ref <= 1e-12
    write_qual(rid, ok, "pyspi default NUM_THREADS (= availableProcessors) vs NUM_THREADS=1 vs a repeat; "
                        f"pass if |default - 1 thread| <= 1e-13, repeat bit-identical and all within 1e-12 of {ME}:unit_matrix",
               kind="same-lineage", default_vs_1thread=d_thr, repeat_diff=d_rep, max_vs_ref=d_ref)


def chk_seeds(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    V = np.stack([unit_matrix(unit, z, seed=s)[0] for s in (42, 43, 44)])
    S = load_pyspi(rid)
    spread = float(np.nanmax(np.nanmax(S, axis=0) - np.nanmin(S, axis=0)))
    write_tol(rid, V, "independent-impl", EST_DETAIL + "; NOISE_SEED 42, 43, 44 (layer order)",
              notes="the jitter stream is regenerated per seed; on tied data the value is a function of the seed",
              seed_spread_pyspi=spread, seed42_vs_43=maxabs(S[0], S[1]), seed42_vs_44=maxabs(S[0], S[2]))


def chk_copy(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    A, _ = unit_matrix(unit, z)
    S = load_pyspi(rid)                                     # (4, P, P): Calculator, CalculatorFrame, deepcopy, pickle
    V = np.stack([A] * 4)
    write_tol(rid, V, "independent-impl", EST_DETAIL + "; the configured SPI, expected on every copy path",
              notes="layers: Calculator, CalculatorFrame (deepcopy, calculator.py:559), copy.deepcopy(spi), pickle round trip",
              calculator_vs_ref=maxabs(S[0], A), frame_vs_ref=maxabs(S[1], A), deepcopy_vs_ref=maxabs(S[2], A),
              pickle_vs_ref=maxabs(S[3], A), frame_vs_calculator=maxabs(S[1], S[0]))


def chk_copyprops(run, ds, x, z):
    rid = run["run_id"]
    S = load_pyspi(rid)                                     # (3, 5): orig, deepcopy, pickle x (k, NOISE_SEED, k_HISTORY, l_HISTORY, AUTO)
    same = [bool(np.array_equal(S[r], S[0], equal_nan=True)) for r in (1, 2)]
    write_qual(rid, all(same), "getProperty read-back: copies must keep (k, NOISE_SEED, k_HISTORY, l_HISTORY, AUTO_EMBED_METHOD) of the original",
               kind="same-lineage", original=S[0].tolist(), deepcopy=S[1].tolist(), pickled=S[2].tolist(),
               deepcopy_keeps=int(same[0]), pickle_keeps=int(same[1]))


def chk_embed(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    emb = {}
    A, Wi = unit_matrix(unit, z, emb_out=emb)
    P = z.shape[0]
    V = np.full((P, P, 5), np.nan)
    for (i, j), e in emb.items():
        V[i, j, 0] = A[i, j]
        if e is not None:
            V[i, j, 1:] = e
    S = load_pyspi(rid)
    agree = int(np.array_equal(S[..., 1:], V[..., 1:], equal_nan=True))
    write_tol(rid, V, "independent-impl", EST_DETAIL + f"; {ME}:select_embedding (from-scratch MAX_CORR_AIS)",
              notes="layers [TE, k_HISTORY, k_TAU, l_HISTORY, l_TAU] per ordered pair (i -> j)",
              embedding_identical=agree, W_max=int(Wi.max()))


def chk_fixed(run, ds, x, z):
    rid, unit = run["run_id"], run["unit_id"]
    A, _ = unit_matrix(unit, z)
    S = load_pyspi(rid)                                     # (2, P, P): auto, fixed with the chosen embedding
    write_tol(rid, np.stack([A, A]), "independent-impl", EST_DETAIL + "; both layers must equal the from-scratch auto-embedding TE",
              notes="layer 1 = pyspi fixed-embedding TransferEntropy with the (k, kt, l, lt) JIDT chose for that pair",
              fixed_minus_auto_pyspi=maxabs(S[0], S[1]))


# ============================================================================ checks: estimator unit
def chk_direct(run, ds, x, z):
    rid = run["run_id"]
    a, b, c = (np.asarray(x[i], dtype=np.float64) for i in range(3))
    V = np.array([[mi_jidt(a, b, W) for W in W_LIST], [cmi_jidt(a, b, c, W) for W in W_LIST]])
    write_tol(rid, V, "independent-impl", EST_DETAIL + "; rows MI(x0;x1), CMI(x0;x1|x2); columns DYN_CORR_EXCL in " + str(W_LIST),
              notes="JIDT KSG1 calculators driven directly with k=4, NOISE_SEED=42 on the raw rows")


def semantics_series():
    t = np.arange(40, dtype=np.float64)
    ph = 0.37 * t + 0.011 * t ** 2
    return np.sin(ph) + 0.1 * np.cos(3.1 * t), np.cos(1.3 * ph) + 0.05 * t / 40.0


def mi_strict_window(x, y, W, seed=SEED):
    """Alternative semantics |s - t| < W excluded (self always excluded) for contrast."""
    X, Y = jidt_prep([K._as2d(x), K._as2d(y)], seed)
    N = X.shape[0]
    t = np.arange(N)
    dt = np.abs(t[:, None] - t[None, :])
    excl = (dt < W) | (dt == 0)
    Dx = np.abs(X - X.T)
    Dy = np.abs(Y - Y.T)
    Dj = np.where(excl, np.inf, np.maximum(Dx, Dy))
    eps = np.sort(Dj, axis=1)[:, KNN - 1]
    nx = ((Dx < eps[:, None]) & ~excl).sum(1)
    ny = ((Dy < eps[:, None]) & ~excl).sum(1)
    return float(digamma(KNN) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N))


def chk_semantics(run, ds, x, z):
    rid = run["run_id"]
    u0, u1 = semantics_series()
    rows = []
    alt = []
    for W in range(7):
        rows.append([W, 31 - 2 * W, 31 - 2 * W, W + 1, 1.0, 1.0, mi_jidt(u0, u1, W)])
        alt.append(mi_strict_window(u0, u1, W))
    V = np.array(rows, dtype=np.float64)
    S = load_pyspi(rid)
    write_tol(rid, V, "analytic",
              "expected under 'excluded iff |s-t| <= W' (KdTree.java a9b6b3e:752, :1125; UnivariateNearestNeighbourSearcher.java:278-346; "
              "guard N <= k+2W MutualInfoCalculatorMultiVariateKraskov.java:298): count of a 32-point ramp around t=15 = 31-2W, "
              f"nearest admissible |s-t| = W+1, raise at N=4+2W, compute at N=5+2W; MI column from {ME}:mi_jidt",
              notes="columns: W, univariate count, KdTree count, min |s-t| of 4 NN, raises at 4+2W, computes at 5+2W, KSG1 MI (T=40)",
              alt_strict_lt_counts=[31 if W == 0 else 33 - 2 * W for W in range(7)], alt_strict_lt_mi=alt,
              mi_pyspi_minus_alt_maxabs=maxabs(S[:, 6], np.array(alt)))


def chk_props(run, ds, x, z):
    rid = run["run_id"]
    S = load_pyspi(rid)
    ncpu, nt_mi, nt_cmi, bias_mi, bias_te, fresh_none, seed_after, te_nt_none = S.tolist()
    ok = nt_mi == ncpu and nt_cmi == ncpu and bias_mi == 1 and bias_te == 1 and fresh_none == 1 and seed_after == 42
    write_qual(rid, ok, "MutualInfoCalculatorMultiVariateKraskov.java a9b6b3e:133 and ConditionalMutualInfoCalculatorMultiVariateKraskov.java:155 "
                        "numThreads = availableProcessors(); BIAS_CORRECTION not a property (getProperty -> null); NOISE_SEED unset on a fresh calculator, 42 after pyspi _setup",
               kind="literature-only", available_processors=ncpu, num_threads_mi=nt_mi, num_threads_cmi=nt_cmi,
               bias_correction_is_none_mi=bias_mi, bias_correction_is_none_te=bias_te, fresh_noise_seed_none=fresh_none,
               noise_seed_after_setup=seed_after, te_num_threads_is_none=te_nt_none)


# ============================================================================ checks: Theiler window unit
def chk_lagloop(run, ds, x, z):
    rid = run["run_id"]
    Wr = window_raw(z)
    S = load_pyspi(rid)
    Wi_ref = window_int(Wr)
    Wi_py = window_int(S)
    write_tol(rid, Wr, "independent-impl", f"{ME}:window_raw (explicit lag-loop biased ACF of each z-scored row, 2 sum_k r_i(k) r_j(k), k=0..T-1); {VERS}",
              notes=f"path {run['path']}; data.theiler holds the raw value, DYN_CORR_EXCL = int(raw)",
              int_mismatch=int((Wi_ref != Wi_py)[~np.eye(len(S), dtype=bool)].sum()), W_int_max=int(Wi_py.max()),
              W_int_min=int(off(Wi_py).min()), raw_over_T_max=float(np.nanmax(S) / int(run["T"])))


def chk_bartlett(run, ds, x, z):
    rid = run["run_id"]
    P = z.shape[0]
    V = np.full((P, P), np.nan)
    B = np.full((P, P), np.nan)
    Bt = np.full((P, P), np.nan)
    T = z.shape[1]
    for i in range(P):
        for j in range(i + 1, P):
            B[i, j] = B[j, i] = K.theiler_window_bartlett(z[i], z[j], max_lag=None)
            Bt[i, j] = Bt[j, i] = K.theiler_window_bartlett(z[i], z[j], max_lag=T // 5)
            V[i, j] = V[j, i] = B[i, j] + 1.0
    S = load_pyspi(rid)
    iu = np.triu_indices(P, 1)
    write_tol(rid, V, "independent-impl", f"{KREF}:theiler_window_bartlett(max_lag=None) + 1 (Bartlett 1946 factor 1 + 2 sum_(k>=1) r_i r_j, all lags); {VERS}",
              notes="candidate: pyspi raw = Bartlett + 1 (lag-0 term counted twice)",
              pairs=[list(map(int, p)) for p in zip(*iu)], pyspi_raw=S[iu].tolist(), bartlett_all_lags=B[iu].tolist(),
              bartlett_lags_to_T_over_5=Bt[iu].tolist(), pyspi_int=window_int(S)[iu].tolist(),
              raw_minus_bartlett=(S[iu] - B[iu]).tolist())


def pop_cross_sq(ds_name, ds, H=400):
    """C = sum_{m=-inf}^{inf} rho_01(m)^2 for the (0,1) pair (0 for independent channels).

    For lags k beyond the correlation scale, E[r_0(k) r_1(k)] ~ Cov(r_0(k), r_1(k)) ~ (T-k)/T^2 * C
    (Bartlett 1946 covariance of lagged products of a Gaussian pair), so the all-lag sum
    2 sum_{k>=1} r_0(k) r_1(k) carries an extra 2 sum_k (T-k)/T^2 C -> C that does not vanish as T grows:
    E[raw] -> tau + 1 + C."""
    if ds_name == "iid_gaussian":
        return 0.0
    G = autocov(ds_name, ds, H)
    s = math.sqrt(G[0, 0, 0] * G[0, 1, 1])
    return float(np.sum((G[:, 0, 1] / s) ** 2) + np.sum((G[1:, 1, 0] / s) ** 2))


def chk_pop(run, ds, x, z):
    rid = run["run_id"]
    T = int(run["T"])
    tau = pop_tau(run["dataset"], ds)
    C = pop_cross_sq(run["dataset"], ds)
    R = 300
    sims = simulator(run["dataset"], ds, run.get("processes"))(T, R, rng_for(rid))
    m, s = _mc(lambda q: window_raw_fast(zs(q)), sims)
    S = load_pyspi(rid)
    pop = np.full(S.shape, tau + 1.0)
    np.fill_diagonal(pop, np.nan)
    write_stat(rid, m, s, f"population Bartlett tau = 1 + 2 sum_(k>=1) rho_0(k) rho_1(k) = {tau:.6f} (VAR autocovariances; white noise 1); "
                          f"expected raw = tau + 1; reference = Monte Carlo mean/SD of the pyspi rule over {R} draws at T={T}",
               pop, S, R, tau_population=tau, raw_pyspi_01=float(S[0, 1]), int_pyspi_01=int(S[0, 1]), mc_mean_01=float(m[0, 1]),
               mc_sd_01=float(s[0, 1]), mc_excess_over_tau_plus_1=float(m[0, 1] - tau - 1.0),
               cross_sq_C=C, large_T_limit_tau_plus_1_plus_C=tau + 1.0 + C)


def chk_growth(run, ds, x, z):
    rid = run["run_id"]
    T = int(run["T"])
    S = load_pyspi(rid)
    f0 = float(ds["f0"])
    k = np.arange(T)
    a2 = (0.5 / (0.5 + 0.05 ** 2)) ** 2
    approx = 2.0 * float(np.sum((1 - k / T) ** 2 * np.cos(2 * np.pi * f0 * k) ** 2)) * a2
    ratio = off(S) / approx
    ok = bool(np.all(np.abs(ratio - 1) < 0.05)) and abs(approx / T - 1 / 3) < 0.03
    write_qual(rid, ok, "sinusoid + 0.05 noise: biased ACF ~ a (1-k/T) cos(w k), a = 0.5/(0.5+0.0025); raw window ~ 2 a^2 sum_k (1-k/T)^2 cos^2(w k) ~ T/3; "
                        "pass if every pair within 5% of that value and value/T within 0.03 of 1/3",
               raw_min=float(np.nanmin(off(S))), raw_max=float(np.nanmax(off(S))), approx=approx, approx_over_T=approx / T,
               raw_over_T_max=float(np.nanmax(off(S)) / T), ratio_min=float(ratio.min()), ratio_max=float(ratio.max()))


def chk_detrend(run, ds, x, z):
    rid = run["run_id"]
    T = int(run["T"])
    S_det = load_pyspi(rid)
    S_raw = load_pyspi(rid.replace("|pipeline-detrend|detrend", "|pipeline|lagloop"))
    if run["dataset"] == "trending":
        tr = [i for i, s in enumerate(ds["slope"]) if s != 0]
        sub = S_raw[np.ix_(tr, tr)]
        ok = np.nanmax(S_det) <= 3.0 and np.nanmin(off(sub)) >= 50.0
        rule = (f"trending (slopes {np.round(ds['slope'], 3).tolist()}): W(pipeline) >= 50 on every pair of the trending "
                f"processes {tr}, W(pipeline-detrend) <= 3 (white-noise level) on every pair")
    else:
        ok = np.nanmax(S_det) >= 0.1 * T
        rule = "cointegrated: random-walk components keep the detrended window large (max >= 0.1 T)"
    write_qual(rid, ok, rule, raw_pipeline_min=float(np.nanmin(off(S_raw))), raw_pipeline_max=float(np.nanmax(S_raw)),
               raw_detrend_min=float(np.nanmin(off(S_det))), raw_detrend_max=float(np.nanmax(S_det)))


# ============================================================================ checks added after review
def chk_copyk6(run, ds, x, z):
    """MutualInfo(kraskov, prop_k=6) on every copy path; k = 4 value quoted for the lost-prop_k reading."""
    rid = run["run_id"]
    P = z.shape[0]
    A6 = np.full((P, P), np.nan)
    A4 = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(i + 1, P):
            A6[i, j] = A6[j, i] = mi_jidt(z[i], z[j], k=6)
            A4[i, j] = A4[j, i] = mi_jidt(z[i], z[j], k=4)
    S = load_pyspi(rid)                                     # (4, P, P): Calculator, CalculatorFrame, deepcopy, pickle
    write_tol(rid, np.stack([A6] * 4), "independent-impl", EST_DETAIL.replace("k=4", "k=6") + "; MI with k=6 expected on every copy path",
              notes="probe copy_paths_mi_k6: MutualInfo(estimator='kraskov', prop_k=6) (identifier mi_kraskov_NN-6, not in config.yaml)",
              calculator_vs_k6=maxabs(S[0], A6), frame_vs_k6=maxabs(S[1], A6), deepcopy_vs_k6=maxabs(S[2], A6),
              pickle_vs_k6=maxabs(S[3], A6), frame_vs_k4=maxabs(S[1], A4), deepcopy_vs_k4=maxabs(S[2], A4),
              ref_k6_01=float(A6[0, 1]), ref_k4_01=float(A4[0, 1]), pyspi_frame_01=float(S[1, 0, 1]),
              pyspi_deepcopy_01=float(S[2, 0, 1]))


chk_copypropsk6 = chk_copyprops


def chk_reversal(run, ds, x, z):
    """Directed TLMI: A(reversed x) = A(x).T up to the jitter assignment (samples enter in reverse order)."""
    rid, unit = run["run_id"], run["unit_id"]
    A, _ = unit_matrix(unit, z)
    Ar, _ = unit_matrix(unit, np.ascontiguousarray(z[:, ::-1]))
    S = load_pyspi(rid)                                     # (2, P, P): A(x), A(x reversed)
    write_tol(rid, np.stack([A, Ar]), "independent-impl", EST_DETAIL + "; layer 1 on the time-reversed rows",
              notes="I(y_i(t); y_j(t+1)) with y = reversed x uses the sample pairs (x_j(s), x_i(s+1)), so A_rev = A.T except for "
                    "the jitter, which follows the sample order",
              asym_maxabs_pyspi=maxabs(S[0], S[0].T), rev_T_minus_orig_pyspi=maxabs(S[1].T, S[0]),
              rev_T_minus_orig_ref=maxabs(Ar.T, A), rev_T_minus_orig_ref_nojitter=maxabs(
                  unit_matrix(unit, np.ascontiguousarray(z[:, ::-1]), seed=None)[0].T, unit_matrix(unit, z, seed=None)[0]))


def chk_guard(run, ds, x, z):
    """Numeric dyn_corr_excl at Wc - 1 (computes) and Wc (JIDT guard N <= 4 + 2W raises; pyspi returns NaN)."""
    rid, unit = run["run_id"], run["unit_id"]
    key = UNIT[unit][0]
    T = z.shape[1]
    N = T - N_OFFSET[key]
    Wc = int(math.ceil((N - KNN) / 2.0))
    P = z.shape[0]
    V = np.full((2, P, P), np.nan)
    for li, W in enumerate((Wc - 1, Wc)):
        for i in range(P):
            for j in range(P):
                if i != j:
                    V[li, i, j] = pair_value(key, z[i], z[j], W, SEED)
    S = load_pyspi(rid)
    write_tol(rid, V, "independent-impl", EST_DETAIL + f"; numeric window W = {Wc - 1} (layer 0) and {Wc} (layer 1, guard N={N} <= 4+2W -> NaN)",
              notes="probe guard_numeric: the SPI rebuilt with dyn_corr_excl=W (infotheory.py:217-220)",
              guard_N=N, W_computes=Wc - 1, W_trips=Wc, nan_pyspi_layer0=int(np.isnan(off(S[0])).sum()),
              nan_pyspi_layer1=int(np.isnan(off(S[1])).sum()))


TAGS = {k[4:]: v for k, v in globals().items() if k.startswith("chk_")}


def run_check(run):
    tag = run["run_id"].split("|")[-1]
    ds = make_data.load(run["dataset"], int(run["T"]))
    x = np.asarray(ds["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    z = prep(x, run["path"])
    return TAGS[tag](run, ds, x, z)


def main():
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    only = sys.argv[1:]
    n_ok = n_err = 0
    for run in man["runs"]:
        rid = run["run_id"]
        if only and not any(o in rid for o in only):
            continue
        try:
            run_check(run)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    print(f"{FAMILY} references: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
