#!/usr/bin/env python3
"""References for the it_kozachenko family (COMPARISON venv: numpy/scipy only, no pyspi, no JIDT).

For every run of ``audit/checks/manifest/it_kozachenko.json`` this script writes
``audit/checks/out/ref/<run_id>.npz`` in the compare.py contract.  The run_id tag (last ``|``
field) selects the check.  Three entropy back-ends are used, all built on
``_kernels_infotheory.entropy_kozachenko`` (cKDTree, Euclidean norm, Kraskov 2004 eq. 20, k = 1):

kl     canonical Kozachenko-Leonenko estimate, no jitter, exact nearest neighbours.
       Criterion ``tolerance`` with atol = ATOL_KL: pyspi carries JIDT's 1e-8 jitter, so the
       difference is the jitter effect; every kl run stores the jitter residual measured on the
       reference side (|jrand - kl|) as ``ref_jitter_residual_maxabs``.
jrand  the same canonical estimate on the input plus JIDT's jitter, reproduced exactly:
       ``observations[r][c] += 1e-8 * random.nextGaussian()`` with ``java.util.Random`` seeded
       with 42 at every setObservations call (EntropyCalculatorMultiVariateCommon.java a9b6b3e7
       :200-211; pyspi sets NOISE_SEED=42, infotheory.py:91).  java.util.Random is a documented
       48-bit LCG (Java SE API: Random.next / nextDouble / nextGaussian, polar Box-Muller); it is
       reimplemented below in pure Python (JavaRandom).  Criterion ``exact`` (rtol 1e-10,
       atol 1e-10: compositions sum up to 20 entropies of magnitude <= 15).  Independent of JIDT:
       only the jitter convention is mirrored, like the ddof conventions elsewhere.
jidt   jrand plus JIDT's nearest-neighbour dispatch (EuclideanUtils.java v1.6.1 :69-75): exact
       ("fast") search for <= 2000 rows, and for > 2000 rows a vectorised port of
       computeMinEuclideanDistancesNaive (:85-110, early exit compares the running SQUARED sum
       with the un-squared minimum).  A port of the delegate's code, so ``same-lineage``: used
       only to show WHY T = 4096 values differ from kl/jrand, never to score.

Statistical runs (truth*, null): the reference value is the population value of the quantity
for the dataset's generating model, se = SD of a Monte Carlo of the canonical (kl) estimator of
the same composition at the same T; the MC mean (population + estimator bias) is stored as
``ref_mc_mean`` and ``ref_z_vs_mc_mean`` = max |pyspi - MC mean| / SD.

Compositions mirror pyspi/statistics/infotheory.py (HEAD a42ebc1) including the column order
of every JIDT call, because the jitter layout depends on it:
  je  A[a,b] = H([z_b, z_a]) for a < b (Directed.multivariate visits (i=b, j=a) first; the
      joint-entropy cache stores both cells; base.py:85-94, infotheory.py:159-177)
  ce  A[i,j] = H([z_max(i,j), z_min(i,j)]) - H(z_i)                    (infotheory.py:246-250)
  xme H([X_lags, y_t]) - H(X_lags), lags 1, 2..k-1                     (infotheory.py:405-426)
  cce sum_i H([Yf, Yp, Xp]) - H([Yp, Xp])                               (infotheory.py:439-454)
  di  sum_i H(embed(y, i)) / i - cce                                     (infotheory.py:487-510)
  si  [H([x_t,x_t-1]) - H(x_t-1)] + [..y..] - [H([x_t,y_t,x_t-1,y_t-1]) - H([x_t-1,y_t-1])],
      A[a,b] = si(z_a, z_b) for a < b, mirrored                         (infotheory.py:524-534)
Population values and simulators come from refs/it_gaussian.py (var_gammas, block_cov,
cond_var, pop_cce, pop_xme_delayed, pop_di_excess_univariate, sim_*).

Run under the comparison venv:  python audit/checks/refs/it_kozachenko.py [run_id substrings]
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
import scipy.signal
import scipy.stats
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_infotheory as K  # noqa: E402
import it_gaussian as G  # noqa: E402  (population helpers and simulators only)

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "it_kozachenko"
OUT = CHECKS / "out" / "ref"
PYS = CHECKS / "out" / "pyspi"
H1 = 0.5 * (1.0 + math.log(2.0 * math.pi))          # 1.4189385332046727
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"
KREF = "refs/_kernels_infotheory.py:entropy_kozachenko"
ME = "refs/it_kozachenko.py"
NOISE = 1e-8
ATOL_KL = 5e-4          # jitter allowance: measured jitter residual (ref_jitter_residual_maxabs) max 2.7e-4 nat over the 30 kl runs at T <= 1024; 1.1e-3 at T = 4096 (cce/di iid runs, which fail by ~2.6 nat anyway)
EXACT = dict(rtol=1e-10, atol=1e-10)
DLIST = (1, 2, 3, 5, 6, 10)
SCALES = (0.1, 10.0, 1000.0)
NPROC = 8


# ============================================================================ java.util.Random
class JavaRandom:
    """java.util.Random (48-bit LCG, multiplier 0x5DEECE66D, addend 0xB; nextGaussian = polar
    Box-Muller returning the second value on the next call).  Java SE API specification of
    Random.setSeed / next / nextDouble / nextGaussian.  StrictMath.log/sqrt vs math.log/sqrt can
    differ in the last ulp (stream agrees with the JVM to <= 4.4e-16 absolute)."""
    MULT, ADD, MASK = 0x5DEECE66D, 0xB, (1 << 48) - 1

    def __init__(self, seed):
        self.seed = (seed ^ self.MULT) & self.MASK
        self.have = False
        self.nng = 0.0

    def _next(self, bits):
        self.seed = (self.seed * self.MULT + self.ADD) & self.MASK
        return self.seed >> (48 - bits)

    def next_double(self):
        return ((self._next(26) << 27) + self._next(27)) * 2.0 ** -53

    def next_gaussian(self):
        if self.have:
            self.have = False
            return self.nng
        while True:
            v1 = 2 * self.next_double() - 1
            v2 = 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        mul = math.sqrt(-2 * math.log(s) / s)
        self.nng = v2 * mul
        self.have = True
        return v1 * mul


_STREAM = np.empty(0)


def java_gauss(n, seed=42):
    """First n values of new Random(seed).nextGaussian() (cached; every JIDT call reseeds)."""
    global _STREAM
    if _STREAM.size < n:
        r = JavaRandom(seed)
        m = max(n, 2 * _STREAM.size, 50000)
        _STREAM = np.array([r.next_gaussian() for _ in range(m)])
    return _STREAM[:n]


def jitter(X, level=NOISE):
    N, d = X.shape
    return X + level * java_gauss(N * d).reshape(N, d)


# ============================================================================ entropy back-ends
def _as2d(X):
    X = np.asarray(X, dtype=np.float64)
    return X[:, None] if X.ndim == 1 else X


def kl_from_r(r, d):
    """Kraskov 2004 eq. 20, k = 1, Euclidean unit ball (same constant as the kernel)."""
    N = r.size
    with np.errstate(divide="ignore"):
        s = d * np.mean(np.log(2.0 * r))
    return float(s + digamma(N) - digamma(1) + 0.5 * d * math.log(math.pi / 4.0) - gammaln(d / 2.0 + 1.0))


def nn_naive_jidt(X):
    """Port of EuclideanUtils.computeMinEuclideanDistancesNaive (JIDT v1.6.1 :85-110), vectorised
    over the query points t while the candidate loop t2 stays sequential (the running minimum
    makes it order-dependent).  For each t2: running squared sum over dimensions in order, the
    loop stops after the first dimension at which the sum is >= the UN-squared minimum m[t];
    sqrt of that (partial) sum is accepted if < m[t]."""
    X = _as2d(X)
    N, D = X.shape
    m = np.full(N, np.inf)
    idx = np.arange(N)
    for t2 in range(N):
        C = np.cumsum((X - X[t2]) ** 2, axis=1)
        ge = C >= m[:, None]
        k = np.where(ge.any(axis=1), ge.argmax(axis=1), D - 1)
        v = np.sqrt(C[idx, k])
        acc = (v < m) & (idx != t2)
        m = np.where(acc, v, m)
    return m


_CACHE = {}


def H(X, mode="kl", level=NOISE):
    """Entropy (nats) of the (N, d) sample X; mode kl | jrand | jidt (see module docstring)."""
    X = _as2d(X)
    N, d = X.shape
    if d == 0:
        return 0.0
    key = (mode, level, X.shape, hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest())
    if key in _CACHE:
        return _CACHE[key]
    if mode == "kl":
        v = K.entropy_kozachenko(X)
    elif mode == "jrand":
        v = K.entropy_kozachenko(jitter(X, level))
    elif mode == "jidt":
        Xj = jitter(X, level)
        if N <= 2000 or d == 1:      # fast routine = exact search; d = 1 naive loop always completes
            v = K.entropy_kozachenko(Xj)
        else:
            v = kl_from_r(nn_naive_jidt(Xj), d)
    else:
        raise ValueError(mode)
    _CACHE[key] = v
    return v


def Hf(mode, level=NOISE):
    return lambda X: H(X, mode, level)


# ============================================================================ compositions (pyspi column order)
def col(v):
    return np.asarray(v, dtype=np.float64).reshape(-1, 1)


def je_mat(z, h):
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for a in range(P):
        for b in range(a + 1, P):
            A[a, b] = A[b, a] = h(np.hstack([col(z[b]), col(z[a])]))
    return A


def ce_mat(z, h):
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                hi, lo = max(i, j), min(i, j)
                A[i, j] = h(np.hstack([col(z[hi]), col(z[lo])])) - h(col(z[i]))
    return A


def xme_val(src, targ, k, h):
    Yf, Xp = K.xme_embedding(src, targ, k)
    return h(np.hstack([Xp, Yf])) - h(Xp)


def cce_val(src, targ, n, h):
    tot = 0.0
    for i in range(1, n + 1):
        Yf, Yp, Xp = K.cce_embedding(src, targ, i)
        XYp = np.hstack([Yp, Xp])
        tot += h(np.hstack([Yf, XYp])) - h(XYp)
    return tot


def rates_val(targ, n, h):
    return sum(h(K.embed(np.ravel(targ), i)) / i for i in range(1, n + 1))


def massey_target_val(targ, n, h):
    """sum_i H(y_t | y_{t-1..t-i+1}) on the CCE sample of term i (Massey/Kramer H(Y^n))."""
    s = 0.0
    for i in range(1, n + 1):
        Yf, Yp, _ = K.cce_embedding(targ, targ, i)
        s += h(np.hstack([Yf, Yp])) - h(Yp)
    return s


def di_val(src, targ, n, h, form="pyspi"):
    c = cce_val(src, targ, n, h)
    if form == "pyspi":
        return rates_val(targ, n, h) - c
    return massey_target_val(targ, n, h) - c


def si_val(x, y, h):
    x, y = col(x), col(y)
    xy = np.hstack([x, y])

    def ce(Xf, Xp):
        return h(np.hstack([Xf, Xp])) - h(Xp)
    return ce(x[1:], x[:-1]) + ce(y[1:], y[:-1]) - ce(xy[1:], xy[:-1])


def directed(fn, z):
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = fn(z[i], z[j])
    return A


def upper_mirror(fn, z):
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    for a in range(P):
        for b in range(a + 1, P):
            A[a, b] = A[b, a] = fn(z[a], z[b])
    return A


UNIT_KEY = {"infotheory.je.kozachenko": "je", "infotheory.ce.kozachenko": "ce",
            "infotheory.xme.kozachenko_k1": "xme1", "infotheory.xme.kozachenko_k10": "xme10",
            "infotheory.cce.kozachenko": "cce", "infotheory.di.kozachenko": "di",
            "infotheory.si.kozachenko": "si", "infotheory.est.kozachenko": "est"}
FDESC = {"je": "je_mat", "ce": "ce_mat", "xme1": "xme_val(k=1)", "xme10": "xme_val(k=10)",
         "cce": "cce_val(n=5)", "di": "di_val(n=5, form='pyspi')", "si": "si_val (upper triangle, mirrored)"}


def stat_matrix(key, z, h):
    if key == "je":
        return je_mat(z, h)
    if key == "ce":
        return ce_mat(z, h)
    if key == "xme1":
        return directed(lambda a, b: xme_val(a, b, 1, h), z)
    if key == "xme10":
        return directed(lambda a, b: xme_val(a, b, 10, h), z)
    if key == "cce":
        return directed(lambda a, b: cce_val(a, b, 5, h), z)
    if key == "di":
        return directed(lambda a, b: di_val(a, b, 5, h), z)
    if key == "si":
        return upper_mirror(lambda a, b: si_val(a, b, h), z)
    raise KeyError(key)


# ============================================================================ helpers
def zs(x):
    return scipy.stats.zscore(np.asarray(x, dtype=np.float64), axis=-1, nan_policy="omit", ddof=1)


def prep(x, path):
    if path == "kernel":
        return np.asarray(x, dtype=np.float64)
    if path == "pipeline":
        return zs(x)
    if path == "pipeline-detrend":
        return zs(scipy.signal.detrend(np.asarray(x, dtype=np.float64), axis=1))
    raise ValueError(path)


def load_pyspi(rid, key="value"):
    with np.load(PYS / f"{rid}.npz") as z:
        return np.asarray(z[key], dtype=np.float64)


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def sym_maxabs(S):
    return float(np.nanmax(np.abs(S - S.T))) if S.ndim == 2 and S.shape[0] == S.shape[1] else float("nan")


def rng_for(rid):
    return np.random.default_rng(zlib.crc32(rid.encode()) + 20261015)


def design(x, d):
    """Duplicate of probes/it_kozachenko_probes.py:design (no import of the probe module)."""
    x = np.asarray(x, dtype=np.float64)
    P, T = x.shape
    maxlag = -(-d // P) - 1
    cols = [x[p, maxlag - L:T - L] for L in range(maxlag + 1) for p in range(P)]
    return np.stack(cols[:d], axis=1)


def nn_configs(T):
    """Duplicate of probes/it_kozachenko_probes.py:nn_configs."""
    out = []
    for d in (1, 2, 3, 5, 10):
        for N in (2000, 2001, T - 9):
            out.append((d, N, 1.0))
    out += [(2, T - 9, 10.0), (10, T - 9, 0.05)]
    return out


def shift_dataset(x):
    """Duplicate of probes/it_kozachenko_probes.py:shift_dataset."""
    u, v1, v2 = (np.asarray(x[k], dtype=np.float64) for k in range(3))
    return np.stack([u, np.roll(u, 9) + 0.1 * v1, np.roll(u, 10) + 0.1 * v2])


def indep_pair(x_var2, x_iid):
    """Duplicate of probes/it_kozachenko_probes.py:indep_pair."""
    return np.stack([np.asarray(x_iid[0], dtype=np.float64), np.asarray(x_var2[0], dtype=np.float64)])


# ============================================================================ writers
def write(rid, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{rid}.npz", **{k: np.asarray(v) for k, v in kw.items()})


def write_exact(rid, value, kind, detail, notes="", rtol=EXACT["rtol"], atol=EXACT["atol"], crit="exact", **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), criterion=crit, rtol=rtol, atol=atol,
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


def write_qual(rid, ok, detail, notes="", kind="independent-impl", **extra):
    write(rid, value=np.asarray(np.nan), criterion="qualitative", pass_flag=int(bool(ok)),
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


def write_stat(rid, pop, mc_vals, S, detail, notes="", canon=None, **extra):
    m = np.nanmean(mc_vals, axis=0)
    s = np.nanstd(mc_vals, axis=0, ddof=1)
    pop = np.asarray(pop, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_mc = np.nanmax(np.abs(S - m) / s)
        z_pop = np.nanmax(np.abs(S - pop) / s)
    write(rid, value=pop, se=s, criterion="statistical", reference_kind="analytic", reference_detail=detail,
          notes=notes, mc_mean=m, mc_sd=s, mc_replicates=int(len(mc_vals)),
          mc_bias_maxabs=float(np.nanmax(np.abs(m - pop))),
          pyspi_minus_population_maxabs=float(np.nanmax(np.abs(S - pop))),
          z_vs_population=float(z_pop), z_vs_mc_mean=float(z_mc), **extra)
    if canon is not None:   # canonical KL composition on the SAME data: separates estimator sampling from pyspi
        c = np.asarray(canon, dtype=np.float64)
        with np.load(OUT / f"{rid}.npz") as z0:
            payload = {k: z0[k] for k in z0.files}
        payload.update(canonical_on_data=c, pyspi_minus_canonical_on_data_maxabs=np.asarray(float(np.nanmax(np.abs(S - c)))),
                       z_population_of_canonical_on_data=np.asarray(float(np.nanmax(np.abs(c - pop) / s))))
        np.savez(OUT / f"{rid}.npz", **payload)


# ============================================================================ Monte Carlo (pool)
def _mc_worker(args):
    fn, sim = args
    return fn(sim)


def mc(fn, sims):
    ctx = mp.get_context("fork")
    with ctx.Pool(NPROC) as pool:
        vals = pool.map(_mc_worker, [(fn, s) for s in sims], chunksize=max(1, len(sims) // (4 * NPROC)))
    return np.array(vals)


_MC_FN = {}


def _mc_call(tag, sim):
    return _MC_FN[tag](sim)


class MCF:
    """Picklable-by-fork callable: looks the function up in _MC_FN in the forked worker."""
    def __init__(self, tag, fn):
        self.tag = tag
        _MC_FN[tag] = fn

    def __call__(self, sim):
        return _MC_FN[self.tag](sim)


# ============================================================================ checks: shared composition runs
def comp_detail(key, mode):
    what = {"kl": f"{KREF} (canonical KL, no jitter, exact NN)",
            "jrand": f"{KREF} on input + 1e-8 * java.util.Random(42).nextGaussian() stream ({ME}:JavaRandom)",
            "jidt": f"{ME}:H(mode='jidt') = jitter replica + port of JIDT EuclideanUtils naive NN routine for > 2000 rows"}[mode]
    return f"{ME}:{FDESC[key]} with {what}; {VERS}"


def chk_comp(run, key, x, ds, T, path, mode):
    rid = run["run_id"]
    z = prep(x, path)
    S = load_pyspi(rid)
    zin = load_pyspi(rid, "input")
    V = stat_matrix(key, z, Hf(mode))
    extra = dict(sym_maxabs=sym_maxabs(S), input_maxabs_diff=float(np.max(np.abs(zin - z))),
                 maxabs_vs_transposed=float(np.nanmax(np.abs(S - V.T))))
    pathnote = {"pipeline": "rows z-scored (scipy.stats.zscore ddof=1, nan_policy omit)", "kernel": "raw rows",
                "pipeline-detrend": "scipy.signal.detrend(axis=1) then z-score ddof=1"}[path]
    if mode == "kl":
        J = stat_matrix(key, z, Hf("jrand"))
        extra["jitter_residual_maxabs"] = float(np.nanmax(np.abs(J - V)))
        extra["pyspi_minus_jrand_maxabs"] = float(np.nanmax(np.abs(S - J)))
        write_exact(rid, V, "independent-impl", comp_detail(key, mode), crit="tolerance", rtol=0.0, atol=ATOL_KL,
                    notes=f"{pathnote}; tolerance = jitter allowance ATOL_KL (measured jitter residual stored)", **extra)
    elif mode == "jrand":
        if T > 2000:
            K0 = stat_matrix(key, z, Hf("kl"))
            extra["kl_vs_jrand_maxabs"] = float(np.nanmax(np.abs(K0 - V)))
        write_exact(rid, V, "independent-impl", comp_detail(key, mode), notes=pathnote, **extra)
    else:
        K0 = stat_matrix(key, z, Hf("kl"))
        extra["canonical_kl_maxabs_diff"] = float(np.nanmax(np.abs(S - K0)))
        extra.update({f"pyspi_{i}{j}": float(S[i, j]) for i in range(min(2, S.shape[0])) for j in range(min(2, S.shape[0])) if i != j})
        extra.update({f"kl_{i}{j}": float(K0[i, j]) for i in range(min(2, S.shape[0])) for j in range(min(2, S.shape[0])) if i != j})
        write_exact(rid, V, "same-lineage", comp_detail(key, mode),
                    notes=f"{pathnote}; explains the > 2000-row values through JIDT's naive NN routine (port of the delegate's code)", **extra)


def chk_noiseoff(run, key, x, ds, T, path):
    z = prep(x, path)
    V = stat_matrix(key, z, Hf("kl"))
    S = load_pyspi(run["run_id"])
    write_exact(run["run_id"], V, "independent-impl", comp_detail(key, "kl"),
                notes="pyspi SPI with NOISE_LEVEL_TO_ADD=0 (probe spi_noiseoff) vs canonical KL: composition isolated from the jitter",
                sym_maxabs=sym_maxabs(S))


# ----- statistical truth --------------------------------------------------------------
def pop_di_ar(Gm, src, tgt, n=5, form="pyspi"):
    gam = np.array([g[tgt, tgt] for g in Gm]) / Gm[0][tgt, tgt]
    if form == "pyspi":
        t = sum(K.gaussian_entropy(scipy.linalg.toeplitz(gam[:m])) / m for m in range(1, n + 1))
    else:
        t = sum(G.h_gauss(G.cond_var(scipy.linalg.toeplitz(gam[:m])[::-1, ::-1])) for m in range(1, n + 1))
    return t - G.pop_cce(Gm, src, tgt, n)


def chk_truth(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    rng = rng_for(rid)
    dsn = run["dataset"]
    R = int(run.get("_R", 200))
    hk = Hf("kl")
    if dsn == "bivariate_gaussian":
        pair = 0 if run["processes"] == [0, 1] else 1
        rho, mi = float(ds["rho"][pair]), float(ds["true_mi_nats"][pair])
        e = {"je": 2 * H1 - mi, "ce": H1 - mi, "si": mi, "cce": 5 * (H1 - mi), "di": 5 * mi}[key]
        pop = np.array([[np.nan, e], [e, np.nan]])
        sims = G.sim_pair(rho, T, R, rng)
        detail = {"je": "2 H1 + 0.5 ln(1 - rho^2)", "ce": "H1 + 0.5 ln(1 - rho^2)", "si": "-0.5 ln(1 - rho^2) (white pair)",
                  "cce": "5 (H1 + 0.5 ln(1-rho^2)) (white pair)", "di": "5 * (-0.5 ln(1-rho^2)) (white pair)"}[key]
        detail = f"population {detail}, rho={rho}, H1 = 0.5(1+ln 2pi) (Cover & Thomas Thm 8.4.1)"
    elif dsn == "gauss_ar1_pair":
        A, SE = np.asarray(ds["A"]), np.asarray(ds["Sigma_E"])
        Gm = G.var_gammas([A], SE, 6)
        if key == "si":
            SX, SXY = np.asarray(ds["Sigma_X"]), np.asarray(ds["Sigma_XY"])
            vv = [SX[i, i] - SXY[i, i] ** 2 / SX[i, i] for i in range(2)]
            e = 0.5 * math.log(vv[0] * vv[1] / np.linalg.det(SE))
            pop = np.array([[np.nan, e], [e, np.nan]])
            detail = "population SI = 0.5 ln(prod_v var(x_v(t+1)|x_v(t)) / det Sigma_E) (Ay 2015; Barrett & Seth 2011)"
        elif key == "cce":
            pop = np.array([[np.nan, G.pop_cce(Gm, 0, 1)], [G.pop_cce(Gm, 1, 0), np.nan]])
            detail = "population sum_m h(y_t | y lags 1..m-1, x lags 0..m-1), Gamma(h) from the VAR(1) Lyapunov solution, unit-variance target"
        elif key == "di":
            pop = np.array([[np.nan, pop_di_ar(Gm, 0, 1)], [pop_di_ar(Gm, 1, 0), np.nan]])
            detail = "population pyspi-form DI: sum_m h(Y^m)/m (Toeplitz of the target autocorrelation) - population cce"
        else:
            raise KeyError(rid)
        sims = G.sim_var([A], SE, T, R, rng)
    elif dsn == "delayed_noise":
        procs = run["processes"]
        k = 1 if key == "xme1" else 10
        pop = G.pop_xme_delayed(k, procs)
        sims = G.sim_delayed(T, R, rng)[:, procs]
        detail = f"population H(x_j(t) | x_i lags {'{1}' if k == 1 else '{1..9}'}) for delayed_noise: H1 + 0.5 ln(1 - 1/1.01) if the delay difference is a used lag else H1 (refs/it_gaussian.py:pop_xme_delayed)"
    elif dsn == "iid_gaussian" and key == "cce":
        pop = np.full((2, 2), 5 * H1)
        np.fill_diagonal(pop, np.nan)
        sims = rng.standard_normal((R, 2, T))
        detail = "population 5 H1 for independent white processes"
    else:
        raise KeyError(rid)
    fn = MCF(rid, lambda s, key=key: stat_matrix(key, zs(s), hk))
    vals = mc(fn, list(sims))
    write_stat(rid, pop, vals, S, f"{detail}; se = SD of {R} Monte Carlo draws of {ME}:{FDESC[key]} with canonical {KREF}",
               canon=stat_matrix(key, zs(x), hk))


def chk_est_truth(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    pop = np.array([d * H1 for d in DLIST])
    R = int(run.get("_R", 200))
    rng = rng_for(rid)
    sims = [rng.standard_normal(x.shape) for _ in range(R)]
    fn = MCF(rid, lambda s: np.array([H(design(s, d), "kl") for d in DLIST]))
    vals = mc(fn, sims)
    sd = vals.std(axis=0, ddof=1)
    write_stat(rid, pop, vals, S, f"population d * 0.5 ln(2 pi e) for N(0, I_d) (iid_gaussian rows and lags, raw); se = SD of {R} draws of {KREF} on the same design of fresh iid N(0,1) rows",
               rows=np.array([design(x, d).shape[0] for d in DLIST]), dlist=np.array(DLIST),
               z_by_d=(S - pop) / sd, canon=np.array([H(design(x, d), "kl") for d in DLIST]))


def chk_null_si(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    iu = np.triu_indices(S.shape[0], 1)
    vals = S[iu]
    R = int(run.get("_R", 100))
    rng = rng_for(rid)
    sims = [rng.standard_normal(x.shape) for _ in range(R)]
    hk = Hf("kl")
    fn = MCF(rid, lambda s: stat_matrix("si", zs(s), hk)[np.triu_indices(s.shape[0], 1)].mean())
    mcv = mc(fn, sims)
    m, s = float(mcv.mean()), float(mcv.std(ddof=1))
    stat = float(vals.mean())
    write_qual(rid, abs(stat - m) <= 2 * s, f"mean of the {vals.size} pair values vs the MC distribution of that mean ({R} iid N(0,1) draws, canonical KL); population SI = 0",
               kind="analytic", pyspi_mean=stat, mc_mean=m, mc_sd=s, z=(stat - m) / s, pyspi_min=float(vals.min()),
               pyspi_max=float(vals.max()), n_negative=int((vals < 0).sum()), n_pairs=int(vals.size))


# ----- est (kernel path probes) ---------------------------------------------------------
def chk_est_direct(run, key, x, ds, T, path, mode):
    rid = run["run_id"]
    S = load_pyspi(rid)
    level = 0.0 if mode == "jidt" else NOISE          # the naive-routine probe runs with noise off
    V = np.array([H(design(x, d), mode, level) for d in DLIST])
    extra = dict(rows=np.array([design(x, d).shape[0] for d in DLIST]), dlist=np.array(DLIST))
    if mode != "kl":
        Kv = np.array([H(design(x, d), "kl") for d in DLIST])
        extra["jitter_residual_by_d"] = V - Kv if mode == "jrand" else np.array([H(design(x, d), "jrand") for d in DLIST]) - Kv
        extra["pyspi_minus_canonical_by_d"] = S - Kv
    else:
        extra["pyspi_minus_canonical_by_d"] = S - V
    kind = "same-lineage" if mode == "jidt" else "independent-impl"
    note = {"kl": "JIDT calculator with NOISE_LEVEL_TO_ADD=0 vs canonical KL on design(x, d) (rows x_p(t-L))",
            "jrand": "JIDT calculator with pyspi's properties (1e-8 jitter, NOISE_SEED=42) vs canonical KL on the jittered input",
            "jidt": "JIDT calculator, noise off, vs the port of JIDT's naive NN routine (> 2000 rows)"}[mode]
    detail = f"{ME}:H(mode='{mode}') on {ME}:design(x, d), d in {DLIST}; {VERS}"
    write_exact(rid, V, kind, detail, notes=note, **extra)


def chk_nn(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    pos = 0
    ok = True
    wrong, maxratio_lo, all_gt1, d1_wrong, port_ok = [], [], True, 0, True
    fast_ok, disp_ok = True, True
    for d, N, s in nn_configs(T):
        X = s * design(x, d)[:N]
        truth = cKDTree(X).query(X, k=2, p=2)[0][:, 1]
        naive = S[pos:pos + N]; pos += N
        disp = S[pos:pos + N]; pos += N
        if N <= 2001:
            fast = S[pos:pos + N]; pos += N
            fast_ok &= bool(np.allclose(fast, truth, rtol=1e-12, atol=0))
        disp_ok &= bool(np.array_equal(disp, naive if N > 2000 else (fast if N <= 2001 else disp)))
        w = naive < truth * (1 - 1e-12)
        wrong.append(int(w.sum()))
        if w.any():
            all_gt1 &= bool((truth[w] > 1).all() and (naive[w] > 1 - 1e-12).all())
            maxratio_lo.append(float((naive[w] / truth[w]).min()))
        else:
            maxratio_lo.append(1.0)
        ok &= bool((naive <= truth * (1 + 1e-12)).all())
        if d == 1:
            d1_wrong += int(w.sum())
        if N > 2000 and d > 1:
            port_ok &= bool(np.allclose(nn_naive_jidt(X), naive, rtol=1e-13, atol=0))
    ok = ok and fast_ok and disp_ok and all_gt1 and d1_wrong == 0 and port_ok and pos == S.size
    cfg = np.array(nn_configs(T))
    write_qual(rid, ok, "scipy cKDTree exact Euclidean NN distances vs JIDT EuclideanUtils naive/dispatch/fast routines called directly; port of the naive routine as a mechanism check",
               notes="pass = fast exact; dispatch = fast for N <= 2000 and naive for N > 2000; naive <= truth, wrong only where truth > 1 with value in (1, truth); d = 1 never wrong; port reproduces naive",
               configs=cfg, wrong_counts=np.array(wrong), min_ratio_wrong=np.array(maxratio_lo),
               fast_exact=int(fast_ok), dispatch_ok=int(disp_ok), wrong_only_above_1=int(all_gt1),
               d1_wrong=d1_wrong, port_reproduces_naive=int(port_ok))


def chk_scale(run, key, x, ds, T, path):
    V = np.array([d * math.log(c) for d in (1, 2, 5) for c in SCALES])
    S = load_pyspi(run["run_id"])
    write_exact(run["run_id"], V, "analytic", "H(c X) - H(X) = d ln c (Kozachenko-Leonenko estimate is scale-equivariant: every NN distance scales by c)",
                notes="X = design(x, d)[:2000] (fast routine), noise off, d in (1, 2, 5), c in (0.1, 10, 1000)",
                maxabs=float(np.max(np.abs(S - V))))


def chk_scale_naive(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    V = np.array([d * math.log(c) for d in (1, 2, 5) for c in SCALES])
    port = []
    for d in (1, 2, 5):
        X = design(x, d)[:T - 9]
        h0 = H(X, "jidt", level=0.0) if d > 1 else H(X, "kl")
        for c in SCALES:
            port.append((H(c * X, "jidt", level=0.0) if d > 1 else H(c * X, "kl")) - h0)
    port = np.array(port)
    dev = S - V
    ok = np.max(np.abs(dev[:3])) <= 1e-10 and np.max(np.abs(S - port)) <= 1e-9 and np.max(np.abs(dev[3:])) > 0.01
    write_qual(rid, ok, f"analytic d ln c, and {ME}:nn_naive_jidt port (same-lineage) for the mechanism",
               kind="analytic", notes="X = design(x, d)[:T-9] (> 2000 rows -> naive routine), noise off; pass = d = 1 obeys d ln c, d >= 2 deviates, port reproduces every value",
               pyspi=S, analytic=V, deviation_from_analytic=dev, port_maxabs=float(np.max(np.abs(S - port))))


def chk_determinism(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    v = H(design(x, 2), "jrand")
    ok = bool(np.all(S == S[0])) and abs(S[0] - v) <= 1e-10
    write_qual(run["run_id"], ok, f"bit-identical repeats on one calculator and on a fresh one; value = {ME}:H(mode='jrand')",
               values=S, jrand=v, jrand_diff=float(S[0] - v))


def chk_ties(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    xr = np.round(x[0] * 2.0) / 2.0
    vt = H(xr, "jrand")
    vc = H(x[0], "jrand")
    ok = np.isneginf(S[1]) and S[0] < -10 and abs(S[0] - vt) <= 1e-6 and abs(S[2] - vc) <= 1e-10
    write_qual(run["run_id"], ok, f"row 0 rounded to 0.5 steps ({np.unique(xr).size} distinct values): JIDT default jitter vs noise off; {ME}:H(mode='jrand') replica",
               kind="analytic", pyspi_tied=float(S[0]), pyspi_tied_noiseoff=float(S[1]), pyspi_continuous=float(S[2]),
               jrand_tied=vt, jrand_continuous=vc, distinct_values=int(np.unique(xr).size))


def chk_properties(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ok = (S[0] == 1.0 and S[1] == 42.0 and S[2] == 1.0 and S[3] == 1.0 and abs(S[4] - 1e-8) < 1e-20
          and abs(S[5] - math.log(10)) <= 1e-12)
    write_qual(run["run_id"], ok, "JIDT getProperty on pyspi's calculator; H(10 x) - H(x) = ln 10 (noise off) shows NORMALISE is not applied",
               kind="literature-only", normalise_reads_true=float(S[0]), noise_seed=float(S[1]), bias_correction_is_none=float(S[2]),
               superclass_is_common=float(S[3]), noise_level=float(S[4]), shift_noiseoff=float(S[5]), shift_default=float(S[6]),
               ln10=math.log(10), shift_default_minus_ln10=float(S[6] - math.log(10)))


def chk_labels(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ok = bool(np.all(S[:, 0] == 1.0) and np.all(S[:, 1] == 0.0))
    write_qual(run["run_id"], ok, "pyspi/statistics/infotheory.py:53-55 else-branch labels kozachenko 'linear'; supplement/GitBook/config.yaml keywords say 'nonlinear'",
               kind="literature-only", notes="rows je, ce, cce, di, si, xme_k1, xme_k10; columns [linear, nonlinear]; pass = all carry 'linear' and none 'nonlinear' (the candidate)")


# ----- je / ce / si specials ------------------------------------------------------------
def chk_order(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    z = zs(x)
    h = Hf("jrand")
    if key == "je":
        V = np.array([h(np.hstack([col(z[1]), col(z[0])]))] * 2 + [h(np.hstack([col(z[0]), col(z[1])])), h(np.hstack([col(z[1]), col(z[0])]))])
    else:
        V = np.array([si_val(z[0], z[1], h)] * 2 + [si_val(z[0], z[1], h), si_val(z[1], z[0], h)])
    write_exact(rid, V, "independent-impl", f"{ME}:{'je' if key == 'je' else 'si_val'} with the jitter replica in each call's column order",
                notes="[A01, A10, fresh bivariate(0,1), fresh bivariate(1,0)]: matrix exactly symmetric; the two direct calls differ only by the jitter layout",
                A01_minus_A10=float(S[0] - S[1]), direct_order_diff=float(S[2] - S[3]))


def chk_noise_sens(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    z = zs(x)
    a = je_mat(z, Hf("jrand"))
    b = je_mat(z, Hf("jrand", 1e-6))
    move = np.abs(S[1] - S[0])
    nd = np.array([np.unique(np.round(r, 9)).size for r in z])
    extra = {f"move_{i}{j}": float(move[i, j]) for i in range(z.shape[0]) for j in range(i + 1, z.shape[0])}
    extra.update({f"je_{i}{j}": float(S[0, i, j]) for i in range(z.shape[0]) for j in range(i + 1, z.shape[0])})
    if run["dataset"] == "shared_envelope":
        tied = [(0, 1), (0, 2), (1, 2)]
        cont = [(0, 3), (1, 3), (2, 3)]
        ok = all(move[i, j] > 1.0 for i, j in tied) and all(move[i, j] < 1e-3 for i, j in cont)
        rule = "pairs among rows 0-2 (near-tied) move by > 1 nat when the jitter goes 1e-8 -> 1e-6; pairs with row 3 by < 1e-3"
    else:
        ok = bool(np.nanmax(move) < 1e-3)
        rule = "logistic_maps: the period-4 row is tied in 1-D only; je in 2-D stays jitter-insensitive (< 1e-3)"
    ok = ok and np.nanmax(np.abs(S[0] - a)) <= 1e-6 and np.nanmax(np.abs(S[1] - b)) <= 1e-6
    write_qual(rid, ok, f"{rule}; replica {ME}:H(mode='jrand', level=1e-8 / 1e-6) reproduces both matrices to <= 1e-6",
               kind="analytic", distinct_values_1e9=nd, replica_maxabs_default=float(np.nanmax(np.abs(S[0] - a))),
               replica_maxabs_1e6=float(np.nanmax(np.abs(S[1] - b))), **extra)


def chk_je_shift(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    scale = np.abs(np.asarray(ds["scale"], dtype=np.float64))
    V = np.log(scale[:, None] * scale[None, :])
    np.fill_diagonal(V, np.nan)
    d2 = make_data.load("bivariate_gaussian", T)["x"]
    canon = je_mat(np.asarray(x), Hf("kl")) - je_mat(np.asarray(d2), Hf("kl"))
    rep_ = je_mat(np.asarray(x), Hf("jrand")) - je_mat(np.asarray(d2), Hf("jrand"))
    iso = np.abs(np.log(scale[:, None] / scale[None, :])) < 1e-12
    write_exact(rid, canon, "independent-impl",
                f"{ME}:je_mat with canonical {KREF} on the RAW scaled_offset rows minus on the raw bivariate_gaussian rows (same draw); analytic population shift ln(s_i s_j) stored",
                crit="tolerance", rtol=0.0, atol=1e-5,
                notes="pyspi je_kozachenko normalise=False. A z-scoring (NORMALISE) estimator would give 0 everywhere; the KL estimate shifts by ln(s_i s_j) only up to anisotropy (Euclidean NN changes under unequal per-axis scales), so the analytic value is reported, not used; atol 1e-5 = jitter allowance at d = 2",
                analytic_shift=V, pyspi_maxabs_vs_analytic=float(np.nanmax(np.abs(S - V))),
                canonical_maxabs_vs_analytic=float(np.nanmax(np.abs(canon - V))),
                min_abs_shift=float(np.nanmin(np.abs(S))), replica_maxabs_vs_pyspi=float(np.nanmax(np.abs(rep_ - S))),
                isotropic_pairs=int(iso[~np.eye(len(scale), dtype=bool)].sum()))


def chk_dir(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    dsn = run["dataset"]
    lower_is_driven = key in ("cce", "xme1", "xme10", "ce")
    pairs = []
    if dsn == "var2_bivariate":
        pairs = [(0, 1)]
        why = "x0 -> x1"
    elif dsn == "delayed_noise":
        dl = (0, 1, 3, 5)
        lags = {"xme1": {1}, "xme10": set(range(1, 10)), "cce": set(range(1, 5)), "di": set(range(1, 5))}[key]
        pairs = [(a, b) for a in range(4) for b in range(4) if a != b and (dl[b] - dl[a]) in lags]
        why = f"x_b = x0 delayed by d_b; source a leads target b by d_b - d_a in the lag window {sorted(lags)}"
    elif dsn == "logistic_maps_bidir":
        pairs = [(1, 0)]
        why = "y -> x (beta_xy = 0.1) stronger than x -> y (0.02)"
    elif dsn == "anm":
        pairs = [(2, 3)]
        why = "x2 -> x3 = tanh(2 x2) + 0.2 e: H(x3 | x2) < H(x2 | x3)"
    elif dsn == "nonlinear_dependence":
        pairs = [(0, 1)]
        why = "x1 = x0^2 + 0.05 e, x0 ~ U(-1, 1): x1 is nearly a function of x0, x0 given x1 keeps the sign ambiguity, so H(x1 | x0) < H(x0 | x1); (x2, x3) on the unit circle recorded only"
    res = [bool(S[a, b] < S[b, a]) if lower_is_driven else bool(S[a, b] > S[b, a]) for a, b in pairs]
    extra = {f"A{a}{b}": float(S[a, b]) for a, b in pairs}
    extra.update({f"A{b}{a}": float(S[b, a]) for a, b in pairs})
    rule = ("A[driver, driven] < A[driven, driver]" if lower_is_driven else "A[driver, driven] > A[driven, driver]")
    # semantic ordering asserted only where the record states it (cce on logistic_maps_bidir: recorded, not asserted:
    # cce is dominated by the target's own predictability, not by the coupling strength)
    asserted = not (key == "cce" and dsn == "logistic_maps_bidir")
    # orientation (the record's proposed check): A[i,j] equals the (source i, target j) replica, not its transpose
    z = zs(x)
    V = stat_matrix(key, z, Hf("jrand"))
    extra["maxabs_vs_replica"] = float(np.nanmax(np.abs(S - V)))
    extra["maxabs_vs_transposed_replica"] = float(np.nanmax(np.abs(S - V.T)))
    if dsn == "anm":
        extra["A01_cubic"], extra["A10_cubic"] = float(S[0, 1]), float(S[1, 0])
    if dsn == "nonlinear_dependence":
        extra["A23_circle"], extra["A32_circle"] = float(S[2, 3]), float(S[3, 2])
    ok = extra["maxabs_vs_replica"] <= 1e-10 and extra["maxabs_vs_transposed_replica"] > 0.01 and (all(res) or not asserted)
    write_qual(rid, ok, f"orientation: {ME}:{FDESC[key]} jitter replica (source i, target j) vs its transpose; ground truth {dsn}: {why}; rule {rule} for pairs {pairs} ({'asserted' if asserted else 'recorded only'})",
               kind="independent-impl", n_pairs=len(pairs), n_agree=int(sum(res)), ordering_asserted=int(asserted),
               sym_maxabs=sym_maxabs(S), **extra)


# ----- xme specials -----------------------------------------------------------------------
def chk_k2(run, key, x, ds, T, path):
    z = zs(x)
    h = Hf("jrand")
    V = directed(lambda a, b: xme_val(a, b, 2, h), z)
    k1 = directed(lambda a, b: xme_val(a, b, 1, h), z)
    S = load_pyspi(run["run_id"])
    write_exact(run["run_id"], V, "independent-impl", f"{ME}:xme_val(k=2) (lag set {{1}}, samples t = 2..T-1) with the jitter replica; {VERS}",
                k2_01=float(S[0, 1]), k2_10=float(S[1, 0]), k1_01=float(k1[0, 1]), k1_10=float(k1[1, 0]),
                k2_minus_k1_maxabs=float(np.nanmax(np.abs(S - k1))))


def chk_shift_rep(run, key, x, ds, T, path):
    z = zs(shift_dataset(x))
    h = Hf("jrand")
    V = np.stack([directed(lambda a, b: xme_val(a, b, 1, h), z), directed(lambda a, b: xme_val(a, b, 10, h), z)])
    write_exact(run["run_id"], V, "independent-impl", f"{ME}:xme_val(k = 1, 10) on [u, u(t-9)+0.1v1, u(t-10)+0.1v2] (z-scored) with the jitter replica; {VERS}")


def lag_design_val(src, targ, lags, h, start=10):
    """H(y_t | x_{t-L}, L in lags) on t = start..T-1 (pyspi's k=10 sample: targ[10:]) with columns [lags..., y_t]."""
    src, targ = np.ravel(src), np.ravel(targ)
    T, m = src.size, start
    Xp = np.stack([src[m - L:T - L] for L in lags], axis=1)
    return h(np.hstack([Xp, targ[m:, None]])) - h(Xp)


def chk_shift_pattern(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    k1, k10 = S
    z = zs(shift_dataset(x))
    hk = Hf("kl")
    c9 = [lag_design_val(z[0], z[j], list(range(1, 10)), hk) for j in (1, 2)]
    c10 = [lag_design_val(z[0], z[j], list(range(1, 11)), hk) for j in (1, 2)]
    ok = (abs(k10[0, 1] - c9[0]) < ATOL_KL and abs(k10[0, 2] - c9[1]) < ATOL_KL and abs(k10[0, 2] - c10[1]) > 0.5
          and k10[0, 2] - k10[0, 1] > 0.5 and abs(k1[0, 1] - k1[0, 2]) < 0.2)
    write_qual(run["run_id"], ok, f"lag-set identification on [u, u(t-9)+0.1v1, u(t-10)+0.1v2]: canonical {KREF} of H(y_t | x lags 1..9) and of H(y_t | x lags 1..10) on the same z-scored rows",
               kind="independent-impl",
               notes="pass = history_length=10 equals the 9-lag design (lags 1..9) for both targets and differs from the 10-lag design for the lag-10 target; u -> lag-9 copy reduced, u -> lag-10 copy not; history_length=1 sees neither. Population (Gaussian) values -0.889 / 1.419 are not reached by KL in d=10 (MC bias +1.5 nat, xme_k10 truth runs)",
               k1_u_to_d9=float(k1[0, 1]), k1_u_to_d10=float(k1[0, 2]), k10_u_to_d9=float(k10[0, 1]), k10_u_to_d10=float(k10[0, 2]),
               canon_9lag_d9=float(c9[0]), canon_9lag_d10=float(c9[1]), canon_10lag_d9=float(c10[0]), canon_10lag_d10=float(c10[1]),
               H1=H1, pop_d9_k10=H1 + 0.5 * math.log(1 - 1 / 1.01))


def chk_threshold(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    Tps = list(range(2008, 2015))
    canon, port = [], []
    for Tp in Tps:
        z = zs(np.asarray(x)[:2, :Tp])
        canon.append(xme_val(z[0], z[1], 10, Hf("kl")))
        port.append(xme_val(z[0], z[1], 10, Hf("jidt")))
    canon, port = np.array(canon), np.array(port)
    rows = np.array(Tps) - 10
    below = rows <= 2000
    ok = (np.max(np.abs(S[below] - canon[below])) < 1e-3 and np.min(np.abs(S[~below] - canon[~below])) > 0.5
          and np.max(np.abs(S - port)) <= 1e-9)
    write_qual(run["run_id"], ok, f"canonical {KREF} (independent) and {ME}:H(mode='jidt') port (same-lineage) of xme_k10(0 -> 1) on truncations T' = 2008..2014",
               notes="pass = agreement while every KL estimate has <= 2000 rows, a jump of > 0.5 nat once it has > 2000 rows, port reproduces all",
               Tprime=np.array(Tps), rows=rows, pyspi=S, canonical=canon, port=port,
               jump_at_2001_rows=float(S[3] - canon[3]), port_maxabs=float(np.max(np.abs(S - port))))


# ----- cce / di specials --------------------------------------------------------------------
def chk_n10(run, key, x, ds, T, path):
    z = zs(x)
    h = Hf("jrand")
    if key == "cce":
        V = directed(lambda a, b: cce_val(a, b, 10, h), z)
        V5 = directed(lambda a, b: cce_val(a, b, 5, h), z)
    else:
        V = directed(lambda a, b: di_val(a, b, 10, h), z)
        V5 = directed(lambda a, b: di_val(a, b, 5, h), z)
    write_exact(run["run_id"], V, "independent-impl", f"{ME}:{'cce_val' if key == 'cce' else 'di_val'}(n=10) with the jitter replica; {VERS}",
                notes="documented history length 10 (supplement 1.4.7 / 1.4.8) via probe; configured value is 5",
                n10_01=float(V[0, 1]), n10_10=float(V[1, 0]), n5_01=float(V5[0, 1]), n5_10=float(V5[1, 0]))


def chk_configured_n(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    write_qual(run["run_id"], S.tolist() == [5.0, 5.0], "pyspi/statistics/infotheory.py:435 / :483 default n=5; config.yaml passes no n; supplement 1.4.7 / 1.4.8 and GitBook document 10",
               kind="literature-only", cce_n=float(S[0]), di_n=float(S[1]), documented_n=10)


def _var2_B_pop(ds, tgt):
    Gm = G.var_gammas([ds["A1"], ds["A2"]], ds["Sigma"], 6)
    gam = np.array([g[tgt, tgt] for g in Gm])
    return G.pop_di_excess_univariate(gam)


def chk_massey(run, key, x, ds, T, path):
    rid = run["run_id"]
    z = zs(x)
    S = load_pyspi(rid)
    V = directed(lambda a, b: di_val(a, b, 5, Hf("kl"), "massey"), z)
    hj = Hf("jrand")
    Bk = [rates_val(z[t], 5, hj) - massey_target_val(z[t], 5, hj) for t in range(2)]
    Bg = [K.di_block_term(z[t]) for t in range(2)]
    extra = dict(massey_01=float(V[0, 1]), massey_10=float(V[1, 0]), pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]),
                 B_kl_target1=float(Bk[1]), B_kl_target0=float(Bk[0]), B_gauss_plugin_target1=float(Bg[1]),
                 B_gauss_plugin_target0=float(Bg[0]))
    if run["dataset"] == "var2_bivariate":
        extra.update(B_population_target1=float(_var2_B_pop(ds, 1)), B_population_target0=float(_var2_B_pop(ds, 0)))
    write_exact(rid, V, "independent-impl", f"{ME}:di_val(form='massey') = sum_m [H(Y_t|Y^(m-1)) - H(Y_t|Y^(m-1),X^m)] (Massey 1990 eq. 6 / Kramer 1998), n=5, canonical {KREF}",
                crit="tolerance", rtol=0.0, atol=ATOL_KL,
                notes="canonical directed information; the mismatch is the block term B(target) = sum_m H(Y^m)/m - H(Y^5)", **extra)


def chk_massey_identity(run, key, x, ds, T, path):
    rid = run["run_id"]
    z = zs(x)
    S = load_pyspi(rid)
    h = Hf("jrand")
    M = directed(lambda a, b: di_val(a, b, 5, h, "massey"), z)
    B = np.array([rates_val(z[t], 5, h) - massey_target_val(z[t], 5, h) for t in range(z.shape[0])])
    V = M + B[None, :]
    err = float(np.nanmax(np.abs(S - V)))
    extra = {f"B_kl_target{t}": float(B[t]) for t in range(z.shape[0])}
    extra.update({f"B_gauss_plugin_target{t}": float(K.di_block_term(z[t])) for t in range(z.shape[0])})
    extra.update({f"massey_{i}{j}": float(M[i, j]) for i in range(z.shape[0]) for j in range(z.shape[0]) if i != j})
    gap = offdiag(S - M)                         # pyspi DI - Massey DI on the same data (= B of the target)
    extra.update(gap_min=float(gap.min()), gap_max=float(gap.max()), gap_maxabs=float(np.max(np.abs(gap))),
                 gap_mean=float(gap.mean()), n_cells=int(gap.size))
    if run["dataset"] == "iid_gaussian":
        extra["B_population"] = 0.0
        extra["B_gauss_plugin_maxabs"] = float(max(abs(K.di_block_term(z[t])) for t in range(z.shape[0])))
        extra["trace_expectation_gap_le_0p15"] = int(extra["gap_maxabs"] <= 0.15)
    elif run["dataset"] == "gauss_ar1_pair":
        Gm = G.var_gammas([np.asarray(ds["A"])], np.asarray(ds["Sigma_E"]), 6)
        for t in range(2):
            extra[f"B_population_target{t}"] = float(G.pop_di_excess_univariate(np.array([g[t, t] for g in Gm])))
    write_qual(rid, err <= 1e-10, f"pyspi di = Massey DI_5 + B(target) with every entropy from {ME}:H(mode='jrand') (sample identity); population B from refs/it_gaussian.py:pop_di_excess_univariate where the target autocovariance is known",
               notes="pass = sample identity; gap_* = pyspi - Massey over the off-diagonal cells (the KL estimate of B per target), B_gauss_plugin_* = Gaussian plug-in of B (refs/_kernels_infotheory.py:di_block_term)",
               identity_maxabs=err, **extra)


def chk_indep_rep(run, key, x, ds, T, path):
    X = indep_pair(x, make_data.load("iid_gaussian", T)["x"])
    z = zs(X)
    V = directed(lambda a, b: di_val(a, b, 5, Hf("jrand")), z)
    M = directed(lambda a, b: di_val(a, b, 5, Hf("kl"), "massey"), z)
    write_exact(run["run_id"], V, "independent-impl", f"{ME}:di_val(form='pyspi') with the jitter replica on [iid row 0, var2 row 0]; {VERS}",
                massey_kl_01=float(M[0, 1]), massey_kl_10=float(M[1, 0]), B_gauss_plugin_target1=float(K.di_block_term(z[1])))


def chk_indep_truth(run, key, x, ds, T, path):
    rid = run["run_id"]
    a1, a2, s2 = float(ds["A1"][0, 0]), float(ds["A2"][0, 0]), float(ds["Sigma"][0, 0])
    B = _var2_B_pop(ds, 0)
    pop = np.array([[np.nan, B], [0.0, np.nan]])
    R = int(run.get("_R", 150))
    rng = rng_for(rid)
    tgt = G.sim_var([np.array([[a1]]), np.array([[a2]])], np.array([[s2]]), T, R, rng)[:, 0]
    src = rng.standard_normal((R, T))
    sims = list(np.stack([src, tgt], axis=1))
    hk = Hf("kl")
    fn = MCF(rid, lambda s: stat_matrix("di", zs(s), hk))
    vals = mc(fn, sims)
    S = load_pyspi(rid)
    write_stat(rid, pop, vals, S, canon=stat_matrix("di", zs(indep_pair(x, make_data.load("iid_gaussian", T)["x"])), hk),
               detail=f"population pyspi-form DI for an independent white source and the AR(2) target x0 of var2_bivariate (a1={a1}, a2={a2}): B = sum_m h(Y^m)/m - h(Y^5) (Massey DI = 0); se = SD of {R} MC draws of {ME}:di_val with canonical KL",
               B_population=B, population_massey=0.0)


# ----- copy (CalculatorFrame) path ----------------------------------------------------------
def chk_copy_props(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ids = ["je", "ce", "cce", "di", "si", "xme_k1", "xme_k10"]
    ok = bool(np.all(S[:, 0] == 42.0) and np.all(S[:, 1] == -1.0) and np.all(S[:, 2] == 42.0)
              and np.all(np.abs(S[:, 3:] - 1e-8) < 1e-20))
    extra = {f"{i}_seed_orig_deepcopy_pickle": S[r, :3] for r, i in enumerate(ids)}
    write_qual(run["run_id"], ok,
               "JIDT getProperty on _entropy_calc of every configured kozachenko SPI: original, copy.deepcopy (JIDTBase.__deepcopy__, pyspi/statistics/infotheory.py:77-82, used by CalculatorFrame.init_from_list, pyspi/calculator.py:559) and pickle round trip (__setstate__ -> _getcalc -> _setup, infotheory.py:71-75, 134)",
               kind="literature-only",
               notes="columns NOISE_SEED [orig, deepcopy, pickle] then NOISE_LEVEL_TO_ADD [orig, deepcopy, pickle]; 'NONE' read as -1; pass = deepcopy reads NOISE_SEED 'NONE' (unseeded jitter) with the 1e-8 noise still on, original and pickle read 42",
               deepcopy_noise_seed_is_NONE=float(np.all(S[:, 1] == -1.0)), pickle_noise_seed_is_42=float(np.all(S[:, 2] == 42.0)),
               noise_level_all=float(np.max(S[:, 3:])), **extra)


def chk_frame(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)                      # [Calculator, CalculatorFrame A, CalculatorFrame B, pickle]
    z = zs(x)
    J = stat_matrix(key, z, Hf("jrand"))
    K0 = stat_matrix(key, z, Hf("kl"))
    tol = 1e-6 if run["dataset"] == "shared_envelope" else 1e-10   # near-tied rows: replica tolerance as in chk_noise_sens
    calc_rep = float(np.nanmax(np.abs(S[0] - J)))
    pick_rep = float(np.nanmax(np.abs(S[3] - J)))
    fa_calc = float(np.nanmax(np.abs(S[1] - S[0])))
    fb_calc = float(np.nanmax(np.abs(S[2] - S[0])))
    fa_fb = float(np.nanmax(np.abs(S[1] - S[2])))
    ok = calc_rep <= tol and pick_rep <= tol and min(fa_calc, fb_calc) > 1e-12 and fa_fb > 1e-12
    P = S.shape[1]
    extra = dict(calc_minus_replica_maxabs=calc_rep, pickle_minus_replica_maxabs=pick_rep,
                 frameA_minus_calc_maxabs=fa_calc, frameB_minus_calc_maxabs=fb_calc, frameA_minus_frameB_maxabs=fa_fb,
                 calc_minus_kl_maxabs=float(np.nanmax(np.abs(S[0] - K0))),
                 frame_minus_kl_maxabs=float(max(np.nanmax(np.abs(S[1] - K0)), np.nanmax(np.abs(S[2] - K0)))),
                 calc_01=float(S[0, 0, 1]), frameA_01=float(S[1, 0, 1]), frameB_01=float(S[2, 0, 1]), pickle_01=float(S[3, 0, 1]),
                 calc_10=float(S[0, 1, 0]), frameA_10=float(S[1, 1, 0]), frameB_10=float(S[2, 1, 0]))
    if P > 3:
        extra.update(calc_03=float(S[0, 0, 3]), frameA_03=float(S[1, 0, 3]), frameB_03=float(S[2, 0, 3]))
    write_qual(rid, ok, f"{ME}:{FDESC[key]} with the seeded jitter replica (java.util.Random(42)) and canonical KL on the same z-scored rows; pyspi Calculator, two CalculatorFrame objects and a pickle round trip on the same data",
               notes=f"pass = Calculator and pickle slices equal the seeded replica (<= {tol:g}) while both CalculatorFrame slices differ from it and from each other (the deepcopy calculator reads NOISE_SEED 'NONE', so its 1e-8 jitter is unseeded)",
               **extra)


# ----- ce ordering by rho ------------------------------------------------------------------
def chk_rho_order(run, key, x, ds, T, path):
    rid = run["run_id"]
    S = load_pyspi(rid)
    z = zs(x)
    mode = "jidt" if T > 2000 else "jrand"
    V = stat_matrix(key, z, Hf(mode))
    pairs = [tuple(int(v) for v in p) for p in np.asarray(ds["pairs"])]
    rho = [float(r) for r in np.asarray(ds["rho"])]
    lo, hi = pairs[int(np.argmin(rho))], pairs[int(np.argmax(rho))]
    cells_hi = [S[hi[0], hi[1]], S[hi[1], hi[0]]]
    cells_lo = [S[lo[0], lo[1]], S[lo[1], lo[0]]]
    rep = float(np.nanmax(np.abs(S - V)))
    ok = max(cells_hi) < min(cells_lo) and rep <= 1e-10
    pop = {r: H1 + 0.5 * math.log(1 - r * r) for r in rho}
    write_qual(rid, ok, f"population H1 + 0.5 ln(1 - rho^2) orders the rho = {max(rho)} pair below the rho = {min(rho)} pair; replica {ME}:ce_mat with H(mode='{mode}')",
               kind="analytic", notes="pass = both cells of the high-rho pair below both cells of the low-rho pair, and pyspi equals the replica (<= 1e-10)",
               ce_high_rho_cells=np.array(cells_hi), ce_low_rho_cells=np.array(cells_lo),
               pop_high_rho=pop[max(rho)], pop_low_rho=pop[min(rho)], margin=float(min(cells_lo) - max(cells_hi)),
               replica_maxabs=rep)


# ============================================================================ dispatch
def run_check(run):
    rid, T, path = run["run_id"], int(run["T"]), run["path"]
    tag = rid.split("|")[-1]
    key = UNIT_KEY[run["unit_id"]]
    ds = make_data.load(run["dataset"], T)
    x = np.asarray(ds["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    if key == "est":
        fn = {"direct-noiseoff": lambda: chk_est_direct(run, key, x, ds, T, path, "kl"),
              "direct-jrand": lambda: chk_est_direct(run, key, x, ds, T, path, "jrand"),
              "direct-naive-noiseoff": lambda: chk_est_direct(run, key, x, ds, T, path, "jidt"),
              "truth": lambda: chk_est_truth(run, key, x, ds, T, path),
              "nn-routines": lambda: chk_nn(run, key, x, ds, T, path),
              "scale-law": lambda: chk_scale(run, key, x, ds, T, path),
              "scale-law-naive": lambda: chk_scale_naive(run, key, x, ds, T, path),
              "determinism": lambda: chk_determinism(run, key, x, ds, T, path),
              "ties": lambda: chk_ties(run, key, x, ds, T, path),
              "properties": lambda: chk_properties(run, key, x, ds, T, path),
              "labels": lambda: chk_labels(run, key, x, ds, T, path),
              "copy-props": lambda: chk_copy_props(run, key, x, ds, T, path)}[tag]
        return fn()
    if tag in ("kl", "jrand", "jidt"):
        return chk_comp(run, key, x, ds, T, path, tag)
    table = {"noiseoff": chk_noiseoff, "order": chk_order, "ties": chk_noise_sens, "shift-d2": chk_je_shift,
             "direction": chk_dir, "k2-jrand": chk_k2, "shift-jrand": chk_shift_rep, "shift-pattern": chk_shift_pattern,
             "threshold": chk_threshold, "n10-jrand": chk_n10, "configured-n": chk_configured_n, "massey": chk_massey,
             "massey-identity": chk_massey_identity, "indep-jrand": chk_indep_rep, "indep-truth": chk_indep_truth,
             "null": chk_null_si, "frame": chk_frame, "rho-order": chk_rho_order}
    if tag.endswith("-order") and tag not in table:
        return chk_order(run, key, x, ds, T, path)
    if tag.startswith("truth"):
        return chk_truth(run, key, x, ds, T, path)
    return table[tag](run, key, x, ds, T, path)


def main():
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    only = set(sys.argv[1:])
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
