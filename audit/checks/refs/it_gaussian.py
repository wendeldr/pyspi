#!/usr/bin/env python3
"""References for the it_gaussian family (COMPARISON venv: numpy/scipy only, no pyspi, no JIDT).

For every run of ``audit/checks/manifest/it_gaussian.json`` this script writes
``audit/checks/out/ref/<run_id>.npz`` in the compare.py contract.  The run_id tag
(last ``|`` field) selects the check:

numpy                 from-scratch closed forms of _kernels_infotheory on the same preprocessing
                      (pipeline: z-score ddof=1; kernel: raw; pipeline-detrend: scipy detrend then
                      z-score).  exact (rtol 1e-10, atol 1e-12).
ols                   gc: 0.5 ln(RSS_r/RSS_f) by OLS with intercept (kernel gaussian_te_ols). exact.
identity-*            pyspi-vs-pyspi analytic identities (same-lineage).
truth*                statistical: the reference value is the Monte Carlo mean of the numpy
                      estimator under the dataset's generating model at this T (population value +
                      finite-sample bias), se = Monte Carlo SD (R replicates); the analytic population
                      value is stored as ``ref_population`` with pyspi - population.
null                  qualitative: mean over ordered/unordered pairs vs the Monte Carlo distribution of
                      that mean on iid data (analytic chi-square mean df/(2N) reported).
direction             qualitative on ground-truth coupling (x0 drives x1).
others                see the function of the same name below.

Run under the comparison venv:  python audit/checks/refs/it_gaussian.py
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
import scipy.signal

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_infotheory as K  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "it_gaussian"
OUT = CHECKS / "out" / "ref"
PYS = CHECKS / "out" / "pyspi"
H1 = 0.5 * (1.0 + math.log(2.0 * math.pi))          # 1.4189385332046727
R_MC = 300                                             # Monte Carlo replicates for statistical runs
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"
KREF = "refs/_kernels_infotheory.py"


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
    return A[~np.eye(A.shape[-1], dtype=bool)] if A.ndim == 2 else A


def rng_for(rid):
    return np.random.default_rng(zlib.crc32(rid.encode()) + 20261014)


# ============================================================================ estimators (as pyspi computes them)
def ais_select(x, kmax=10, taumax=2):
    """JIDT MAX_CORR_AIS (ActiveInfoStorageCalculatorViaMutualInfo.java a9b6b3e :579-615):
    k = 1..kmax, tau = 1..taumax (tau loop broken after k = 1); score = Gaussian MI(past
    embedding, next) - k/(2N) with N = T - (k-1)tau - 1 the candidate's own sample count
    (ActiveInfoStorageCalculatorGaussian :152-170, ChiSquareMeasurementDistribution mean df/(2N),
    df = k * 1); strict '>' keeps the first maximum."""
    x = np.ravel(x)
    T = x.size
    best, bk, bt = -np.inf, 1, 1
    for k in range(1, kmax + 1):
        for tau in range(1, taumax + 1):
            s = (k - 1) * tau
            n = T - s - 1
            past = K.embed(x, k, tau, start=s, n=n)
            nxt = x[s + 1:s + 1 + n]
            score = K.gaussian_mi(past, nxt) - k / (2.0 * n)
            if score > best:
                best, bk, bt = score, k, tau
            if k == 1:
                break
    return bk, bt


def te_auto(src, targ):
    k, kt = ais_select(targ)
    l, lt = ais_select(src)
    return K.gaussian_te(src, targ, k=k, k_tau=kt, l=l, l_tau=lt, delay=1), (k, kt, l, lt)


def ste(src, targ, m=3):
    """Symbolic TE (Staniek & Lehnertz 2008, PRL 100:158101, eq. 2): ordinal patterns of
    order m, delay 1 (stable argsort of [x_{t-m+1}, ..., x_t]); plug-in probabilities of
    (s^y_{t+1}, s^y_t, s^x_t); nats."""
    def sym(v):
        E = K.embed(v, m)[:, ::-1]                       # oldest first
        r = np.argsort(np.argsort(E, axis=1, kind="stable"), axis=1, kind="stable")
        return (r * (m ** np.arange(m))).sum(axis=1)
    sx, sy = sym(src), sym(targ)
    yn, yp, xp = sy[1:], sy[:-1], sx[:-1]
    def H(*cols):
        _, c = np.unique(np.stack(cols, axis=1), axis=0, return_counts=True)
        p = c / c.sum()
        return -(p * np.log(p)).sum()
    # TE = H(yn, yp) - H(yp) - H(yn, yp, xp) + H(yp, xp)
    return H(yn, yp) - H(yp) - H(yn, yp, xp) + H(yp, xp)


STAT = {
    "je": (undirected, lambda a, b: K.gaussian_joint_entropy(a, b)),
    "ce": (directed, lambda a, b: K.gaussian_conditional_entropy(b, a)),   # A[i,j] = H(x_j | x_i)
    "mi": (undirected, K.gaussian_mi),
    "tlmi": (directed, K.gaussian_tlmi),
    "gc1": (directed, lambda a, b: K.gaussian_te(a, b)),
    "gca": (directed, lambda a, b: te_auto(a, b)[0]),
    "sym1": (directed, lambda a, b: K.gaussian_te(a, b, k=1)),
    "sym10": (directed, lambda a, b: K.gaussian_te(a, b, k=10)),
    "xme1": (directed, lambda a, b: K.gaussian_xme(a, b, 1)),
    "xme10": (directed, lambda a, b: K.gaussian_xme(a, b, 10)),
    "cce": (directed, lambda a, b: K.gaussian_cce(a, b, 5)),
    "di": (directed, lambda a, b: K.gaussian_di(a, b, 5, "pyspi")),
    "si": (undirected, K.gaussian_si),
}
UNIT_KEY = {
    "infotheory.je.gaussian": "je", "infotheory.ce.gaussian": "ce", "infotheory.mi.gaussian": "mi",
    "infotheory.tlmi.gaussian": "tlmi", "infotheory.te.gaussian_k-1_kt-1_l-1_lt-1": "gc1",
    "infotheory.te.gaussian_auto-MAX_CORR_AIS_kmax-10_taumax-2": "gca",
    "infotheory.xme.gaussian_k1": "xme1", "infotheory.xme.gaussian_k10": "xme10",
    "infotheory.cce.gaussian": "cce", "infotheory.di.gaussian": "di", "infotheory.si.gaussian": "si",
    "infotheory.est.gaussian": "est",
}
KFUN = {"je": "gaussian_joint_entropy", "ce": "gaussian_conditional_entropy(x_j, x_i)", "mi": "gaussian_mi",
        "tlmi": "gaussian_tlmi", "gc1": "gaussian_te(k=l=1)", "gca": "it_gaussian.py:ais_select + gaussian_te",
        "sym1": "gaussian_te(k=1, l=1)", "sym10": "gaussian_te(k=10, l=1)", "xme1": "gaussian_xme(k=1)",
        "xme10": "gaussian_xme(k=10)", "cce": "gaussian_cce(n=5)", "di": "gaussian_di(n=5, form='pyspi')",
        "si": "gaussian_si"}


def stat_matrix(key, z):
    kind, fn = STAT[key]
    return kind(fn, z)


# ============================================================================ generators (Monte Carlo)
def sim_var(A_list, Sigma, T, R, rng, burn=1000):
    A_list = [np.asarray(A, dtype=np.float64) for A in A_list]
    P = A_list[0].shape[0]
    L = np.linalg.cholesky(np.asarray(Sigma, dtype=np.float64))
    p = len(A_list)
    n = T + burn
    E = rng.standard_normal((n, R, P)) @ L.T
    X = np.zeros((n, R, P))
    for t in range(p, n):
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


# ============================================================================ population values
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
        G.append(S[:P, h * P:(h + 1) * P])                # E[x_t x_{t-h}^T]
    while len(G) <= H:
        h = len(G)
        G.append(sum(A_list[l] @ G[h - 1 - l] for l in range(p)))
    return G[:H + 1]


def block_cov(G, items):
    """Covariance of the stacked scalars x_{comp}(t - lag) for items = [(comp, lag), ...]."""
    n = len(items)
    C = np.empty((n, n))
    for a, (ca, la) in enumerate(items):
        for b, (cb, lb) in enumerate(items):
            h = lb - la                                  # (t - la) - (t - lb)
            C[a, b] = G[h][ca, cb] if h >= 0 else G[-h][cb, ca]
    return C


def cond_var(C, target=0):
    rest = [i for i in range(C.shape[0]) if i != target]
    if not rest:
        return C[target, target]
    c = C[target, rest]
    return C[target, target] - c @ np.linalg.solve(C[np.ix_(rest, rest)], c)


def h_gauss(v):
    return 0.5 * math.log(2 * math.pi * math.e * v)


def pop_te1(G, src, tgt):
    v_r = cond_var(block_cov(G, [(tgt, 0), (tgt, 1)]))
    v_f = cond_var(block_cov(G, [(tgt, 0), (tgt, 1), (src, 1)]))
    return 0.5 * math.log(v_r / v_f)


def pop_cce(G, src, tgt, n=5):
    s = 0.0
    for m in range(1, n + 1):
        items = [(tgt, 0)] + [(tgt, L) for L in range(1, m)] + [(src, L) for L in range(0, m)]
        s += h_gauss(cond_var(block_cov(G, items)) / G[0][tgt, tgt])
    return s


def pop_di_excess_univariate(gamma, n=5):
    """B = sum_m h(Y^m)/m - sum_m h(Y_m | Y^{m-1}) for a stationary scalar process with
    autocovariance gamma(h) (scale cancels)."""
    b = 0.0
    for m in range(1, n + 1):
        Tm = scipy.linalg.toeplitz(gamma[:m])
        b += K.gaussian_entropy(Tm) / m
        b -= h_gauss(cond_var(Tm[::-1, ::-1]))          # Y_m given the m-1 previous
    return b


# ============================================================================ writers
def write(rid, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in kw.items():
        payload[k] = np.asarray(v) if not isinstance(v, str) else np.asarray(v)
    np.savez(OUT / f"{rid}.npz", **payload)


def write_exact(rid, value, kind, detail, notes="", rtol=1e-10, atol=1e-12, **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), criterion="exact", rtol=rtol, atol=atol,
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


def write_stat(rid, value, se, detail, population, S, notes="", **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), se=np.asarray(se, dtype=np.float64),
          criterion="statistical", reference_kind="analytic", reference_detail=detail, notes=notes,
          population=np.asarray(population, dtype=np.float64),
          pyspi_minus_population_maxabs=float(np.nanmax(np.abs(S - population))),
          mc_mean_minus_population_maxabs=float(np.nanmax(np.abs(np.asarray(value) - population))),
          mc_replicates=R_MC, **extra)


def write_qual(rid, ok, detail, notes="", kind="analytic", **extra):
    write(rid, value=np.asarray(np.nan), criterion="qualitative", pass_flag=int(bool(ok)),
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


# ============================================================================ checks
def chk_numpy(run, key, x):
    z = prep(x, run["path"])
    V = stat_matrix(key, z)
    note = {"pipeline": "rows z-scored ddof=1", "kernel": "raw rows (Data normalise=False)",
            "pipeline-detrend": "scipy.signal.detrend(axis=1) then z-score ddof=1"}[run["path"]]
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:{KFUN[key]}; {VERS}", notes=note)


def ref_value(key, rid, x, path):
    """Recompute the reference matrix of another (numpy) run of the same data."""
    return stat_matrix(key, prep(x, path))


def run_check(run):
    rid, ds_name, T, path = run["run_id"], run["dataset"], int(run["T"]), run["path"]
    tag = rid.split("|")[-1]
    unit = run["unit_id"]
    key = UNIT_KEY.get(unit)
    if unit == "infotheory.te.symbolic":
        key = "sym10" if tag.startswith("k10") else "sym1"
        tag = tag.split("-", 1)[1]
    ds = make_data.load(ds_name, T)
    x = np.asarray(ds["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]

    if tag == "numpy":
        return chk_numpy(run, key, x)
    fn = CHECKS_BY_TAG.get((key, tag)) or CHECKS_BY_TAG.get(("*", tag))
    if fn is None:
        raise KeyError(f"no reference for {rid}")
    return fn(run, key, x, ds, T, path)


# ----- je / ce / mi -------------------------------------------------------------
def chk_equals_d2(run, key, x, ds, T, path):
    d2 = make_data.load("bivariate_gaussian", T)["x"]
    V = stat_matrix(key, zs(d2))
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:{KFUN[key]} on z-scored bivariate_gaussian rows (same draw); {VERS}",
                notes="scaled_offset = scale*D2 + offset; after z-scoring the pipeline input equals D2's")


def chk_identity_mi(run, key, x, ds, T, path):
    mi = load_pyspi(f"infotheory.mi.gaussian|{run['dataset']}|{T}|pipeline|numpy")
    V = (2 * H1 - mi) if key == "je" else (H1 - mi)
    S = load_pyspi(run["run_id"])
    write_exact(run["run_id"], V, "same-lineage",
                f"identity {'je = 2*H1' if key == 'je' else 'ce = H1'} - pyspi mi_gaussian (same Data), H1 = 0.5(1+ln 2pi)",
                notes="pyspi-vs-pyspi identity under z-scoring", sym_maxabs=float(np.nanmax(np.abs(S - S.T))))


def chk_truth_pair(run, key, x, ds, T, path):
    """bivariate_gaussian pair (serially iid, contemporaneous correlation rho)."""
    p = run["processes"]
    pair = 0 if p == [0, 1] else 1
    rho = float(ds["rho"][pair])
    mi_true = float(ds["true_mi_nats"][pair])
    pop_entry = {"je": 2 * H1 - mi_true, "ce": H1 - mi_true, "mi": mi_true, "cce": 5 * (H1 - mi_true),
                 "di": 5 * mi_true, "si": mi_true}[key]
    pop = np.full((2, 2), pop_entry)
    np.fill_diagonal(pop, np.nan)
    sims = sim_pair(rho, T, R_MC, rng_for(run["run_id"]))
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    S = load_pyspi(run["run_id"])
    detail = {"je": "(1+ln 2pi) + 0.5 ln(1-rho^2) (Cover & Thomas Thm 8.4.1)",
              "ce": "0.5 ln(2 pi e (1-rho^2))", "mi": "-0.5 ln(1-rho^2) (Cover & Thomas Ex. 8.5.1)",
              "cce": "5 (H1 + 0.5 ln(1-rho^2)) for serially white pairs",
              "di": "5 * (-0.5 ln(1-rho^2)) for serially white pairs (both DI forms)",
              "si": "-0.5 ln(1-rho^2): SI of serially white pairs = contemporaneous MI (Ay 2015 def.)"}[key]
    write_stat(run["run_id"], m, s, f"population {detail}, rho={rho}; MC mean/SD of {KREF}:{KFUN[key]} over {R_MC} draws",
               pop, S, rho=rho, analytic_bias_mi=1 / (2 * T))


def chk_order(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    sd = x.std(axis=1, ddof=1)
    expect = np.log(sd[None, :] / sd[:, None])          # A[i,j] - A[j,i] = ln(sd_j / sd_i)
    diff = S - S.T
    err = np.nanmax(np.abs(diff - expect))
    rev = directed(lambda a, b: K.gaussian_conditional_entropy(a, b), x)   # H(x_i | x_j) orientation
    fwd = directed(lambda a, b: K.gaussian_conditional_entropy(b, a), x)
    ok = err <= 1e-10 and np.nanmax(np.abs(S - fwd)) <= 1e-10 and np.nanmax(np.abs(S - rev)) > 0.1
    write_qual(run["run_id"], ok, "A[i,j]-A[j,i] = ln(sd_j/sd_i) (ddof=1 SDs of raw scaled_offset rows); A = H(x_j|x_i) not H(x_i|x_j)",
               antisym_err=float(err), A01=float(S[0, 1]), A10=float(S[1, 0]), expected_A01_minus_A10=float(expect[0, 1]),
               maxabs_vs_reversed_orientation=float(np.nanmax(np.abs(S - rev))), ln10=math.log(10))


def chk_invariance(run, key, x, ds, T, path):
    d2 = make_data.load("bivariate_gaussian", T)["x"]
    V = stat_matrix(key, d2)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:{KFUN[key]} on the unscaled bivariate_gaussian rows (raw); {VERS}",
                notes="MI is invariant to per-channel affine maps: Data(normalise=False) on scaled_offset must equal MI of D2")


def chk_null(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    vals = S[np.triu_indices(S.shape[0], 1)] if key in ("mi", "si") else off(S)
    stat = float(vals.mean())
    rng = rng_for(run["run_id"])
    sims = rng.standard_normal((R_MC,) + x.shape)
    def mean_stat(z):
        A = stat_matrix(key, z)
        return (A[np.triu_indices(A.shape[0], 1)] if key in ("mi", "si") else off(A)).mean()
    m, s = mc(mean_stat, sims)
    df, N = {"mi": (1, T), "gc1": (1, T - 1), "si": (3, T - 1)}[key]
    ok = abs(stat - m) <= 2 * s and (vals >= -1e-12).all()
    write_qual(run["run_id"], ok, f"mean over pairs vs MC distribution of that mean on iid N(0,1) data ({R_MC} draws); analytic chi-square({df})/(2N) mean",
               pyspi_mean=stat, mc_mean=float(m), mc_sd=float(s), analytic_mean=df / (2 * N),
               pyspi_min=float(vals.min()), pyspi_max=float(vals.max()), z=float((stat - m) / s))


def chk_linear_only(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ksg01 = K.mi_ksg1(x[0], x[1], k=4)
    ksg23 = K.mi_ksg1(x[2], x[3], k=4)
    ok = S[0, 1] < 0.01 and S[2, 3] < 0.01 and ksg01 > 0.5 and ksg23 > 0.5
    write_qual(run["run_id"], ok, f"{KREF}:mi_ksg1 (k=4) as dependence witness; pass if Gaussian MI < 0.01 nats while KSG MI > 0.5 nats",
               kind="independent-impl", pyspi_01=float(S[0, 1]), pyspi_23=float(S[2, 3]), ksg_01=float(ksg01), ksg_23=float(ksg23))


# ----- tlmi ----------------------------------------------------------------------
def chk_tlmi_truth(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    if run["dataset"] == "delayed_noise":
        pop = np.array([[np.nan, 0.5 * math.log(101)], [0.0, np.nan]])
        sims = sim_delayed(T, R_MC, rng_for(run["run_id"]))[:, [0, 1]]
        detail = "x1(t+1) = x0(t) + 0.1 e: I(x0(t); x1(t+1)) = 0.5 ln(101); reverse 0"
    else:
        SX, SXY = ds["Sigma_X"], ds["Sigma_XY"]
        pop = np.full((2, 2), np.nan)
        for i in range(2):
            for j in range(2):
                if i != j:
                    r = SXY[j, i] / math.sqrt(SX[i, i] * SX[j, j])
                    pop[i, j] = -0.5 * math.log(1 - r ** 2)
        sims = sim_var([ds["A"]], ds["Sigma_E"], T, R_MC, rng_for(run["run_id"]))
        detail = "rho(x_i(t), x_j(t+1)) = Sigma_XY[j,i]/sqrt(Sigma_X[i,i]Sigma_X[j,j]) (Sigma_XY = A Sigma_X)"
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    write_stat(run["run_id"], m, s, f"population {detail}; MC mean/SD of {KREF}:gaussian_tlmi ({R_MC} draws)", pop, S)


# ----- direction (all directed units) ----------------------------------------------
def chk_direction(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    if key in ("cce", "xme1", "xme10"):
        ok = S[0, 1] < S[1, 0]
        rule = "A[0,1] < A[1,0] (conditioning on the driver x0 lowers the target's entropy)"
    else:
        ok = S[0, 1] > S[1, 0]
        rule = "A[0,1] > A[1,0] (x0 drives x1)"
    extra = {}
    if run["dataset"] == "delayed_noise" and key in ("gca",):
        extra = {f"dn_A0{k}": float(S[0, k]) for k in (1, 2, 3)}
        extra.update({f"dn_A{k}0": float(S[k, 0]) for k in (1, 2, 3)})
    write_qual(run["run_id"], ok, f"ground truth {run['dataset']}: {rule}", A01=float(S[0, 1]), A10=float(S[1, 0]),
               ratio=float(S[0, 1] / S[1, 0]) if S[1, 0] != 0 else float("inf"), **extra)


# ----- gc ------------------------------------------------------------------------
def chk_ols(run, key, x, ds, T, path):
    z = prep(x, path)
    V = directed(lambda a, b: K.gaussian_te_ols(a, b), z)
    S = load_pyspi(run["run_id"])
    F = 2 * V
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:gaussian_te_ols = 0.5 ln(RSS_r/RSS_f), OLS with intercept (Geweke 1982 F / 2); {VERS}",
                notes="pyspi value / Geweke F = 0.5", geweke_F_01=float(F[0, 1]), geweke_F_10=float(F[1, 0]),
                pyspi_over_F_01=float(S[0, 1] / F[0, 1]), pyspi_over_F_10=float(S[1, 0] / F[1, 0]),
                cov_vs_ols_maxabs=float(np.nanmax(np.abs(directed(lambda a, b: K.gaussian_te(a, b), z) - V))))


def chk_gc_truth(run, key, x, ds, T, path):
    G = var_gammas([ds["A1"], ds["A2"]], ds["Sigma"], 2)
    pop = np.array([[np.nan, pop_te1(G, 0, 1)], [pop_te1(G, 1, 0), np.nan]])
    sims = sim_var([ds["A1"], ds["A2"]], ds["Sigma"], T, R_MC, rng_for(run["run_id"]))
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    S = load_pyspi(run["run_id"])
    write_stat(run["run_id"], m, s, f"population TE k=l=1 of the VAR(2) from companion-form Lyapunov Gamma(0), Gamma(1); MC mean/SD of {KREF}:gaussian_te ({R_MC} draws)",
               pop, S, pop_01=float(pop[0, 1]), pop_10=float(pop[1, 0]),
               pop_k2l2_01=float(K._var2_te_truth(ds["A1"], ds["A2"], ds["Sigma"], 0, 1)))


# ----- gc auto ---------------------------------------------------------------------
def chk_params(run, key, x, ds, T, path):
    z = prep(x, path)
    P = z.shape[0]
    V = np.full((P, P, 4), np.nan)
    sel = {}
    for p in range(P):
        sel[p] = ais_select(z[p])
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = [sel[j][0], sel[j][1], sel[i][0], sel[i][1]]
    extra = {f"sel_k_x{p}": sel[p][0] for p in range(P)}
    extra.update({f"sel_tau_x{p}": sel[p][1] for p in range(P)})
    write_exact(run["run_id"], V, "independent-impl",
                "refs/it_gaussian.py:ais_select (MAX_CORR_AIS: Gaussian AIS - k/(2N) per candidate N, k<=10, tau<=2, strict >)",
                notes="[k_HISTORY, k_TAU, l_HISTORY, l_TAU] per ordered pair (i source, j target)", rtol=0.0, atol=0.0, **extra)


# ----- te_symbolic ------------------------------------------------------------------
def chk_identity_gc1(run, key, x, ds, T, path):
    V = load_pyspi(f"infotheory.te.gaussian_k-1_kt-1_l-1_lt-1|{run['dataset']}|{T}|pipeline|numpy")
    write_exact(run["run_id"], V, "same-lineage", "pyspi gc_gaussian_k-1_kt-1_l-1_lt-1 on the same Data (tests/README.md bug 1)",
                notes="bitwise equality required (rtol = atol = 0)", rtol=0.0, atol=0.0)


def chk_monotone(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])                       # (2, P, P): pyspi on x and on exp(x)
    P = x.shape[0]
    st = directed(lambda a, b: ste(a, b), x)
    st_exp = directed(lambda a, b: ste(a, b), np.exp(x))
    ste_change = float(np.nanmax(np.abs(st - st_exp)))
    rel = float(np.nanmax(np.abs(S[0] - S[1]) / np.abs(S[0])))
    gauss_on_x = directed(lambda a, b: K.gaussian_te(a, b), zs(x))
    ok = ste_change == 0.0 and rel > 1e-3 and np.nanmax(np.abs(S[0] - gauss_on_x)) < 1e-10
    extra = {}
    for i in range(P):
        for j in range(P):
            if i != j and (i, j) in ((0, 1), (1, 0), (2, 3), (3, 2)):
                extra[f"ste_{i}{j}"] = float(st[i, j])
                extra[f"pyspi_{i}{j}"] = float(S[0, i, j])
                extra[f"pyspi_exp_{i}{j}"] = float(S[1, i, j])
    write_qual(run["run_id"], ok, "refs/it_gaussian.py:ste (Staniek & Lehnertz 2008 eq. 2, m=3, delay 1) on x and exp(x)",
               kind="independent-impl", notes="pass = STE invariant under exp (ordinal), pyspi te_symbolic changes (Gaussian) and equals Gaussian TE on x",
               ste_change_under_exp=ste_change, pyspi_max_rel_change_under_exp=rel, **extra)


# ----- xme ----------------------------------------------------------------------------
def pop_xme_delayed(k, procs, delays=(0, 1, 3, 5)):
    lags = [1] + list(range(2, k))
    var = [1.0] + [1.01] * (len(delays) - 1)
    P = len(procs)
    A = np.full((P, P), np.nan)
    for a, i in enumerate(procs):
        for b, j in enumerate(procs):
            if a == b:
                continue
            L = delays[j] - delays[i]
            if L in lags:
                r2 = 1.0 / (var[i] * var[j])
                A[a, b] = H1 + 0.5 * math.log(1 - r2)
            else:
                A[a, b] = H1
    return A


def chk_xme_truth(run, key, x, ds, T, path):
    k = 1 if key == "xme1" else 10
    procs = run["processes"]
    pop = pop_xme_delayed(k, procs)
    sims = sim_delayed(T, R_MC, rng_for(run["run_id"]))[:, procs]
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    S = load_pyspi(run["run_id"])
    write_stat(run["run_id"], m, s, f"population H(x_j(t)|x_i lags {'{1}' if k == 1 else '{1..9}'}) for delayed_noise (z-scored): H1 + 0.5 ln(1 - rho^2) if d_j - d_i is a used lag else H1; MC mean/SD ({R_MC} draws)",
               pop, S)


def shift_dataset(x):
    """Duplicate of probes/it_gaussian_probes.py:shift_dataset (no import of the probe module)."""
    u, v1, v2 = x[0], x[1], x[2]
    return np.stack([u, np.roll(u, 9) + 0.1 * v1, np.roll(u, 10) + 0.1 * v2])


def chk_shift_numpy(run, key, x, ds, T, path):
    z = zs(shift_dataset(x))
    V = np.stack([directed(lambda a, b, kk=kk: K.gaussian_xme(a, b, kk), z) for kk in (1, 2, 10)])
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:gaussian_xme(k = 1, 2, 10) on [u, u(t-9)+0.1v1, u(t-10)+0.1v2]; {VERS}")


def chk_shift_pattern(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])                       # (3, 3, 3): k1, k2, k10
    k1, k2, k10 = S
    if key == "xme10":
        ok = k10[0, 1] < 0 and abs(k10[0, 2] - H1) < 0.05
        rule = "history_length=10: u -> u(t-9) strongly negative (lag 9 used), u -> u(t-10) ~ H1 (lag 10 not used)"
    else:
        ok = abs(k1[0, 1] - H1) < 0.05 and abs(k1[0, 2] - H1) < 0.05 and abs(k2[0, 1] - k1[0, 1]) < 0.01
        rule = "history_length=1 (lag 1 only): both ~ H1; history_length=2 ~ history_length=1"
    write_qual(run["run_id"], ok, rule, k1_u_to_d9=float(k1[0, 1]), k1_u_to_d10=float(k1[0, 2]),
               k2_u_to_d9=float(k2[0, 1]), k10_u_to_d9=float(k10[0, 1]), k10_u_to_d10=float(k10[0, 2]), H1=H1,
               analytic_d9_k10=H1 + 0.5 * math.log(1 - 1 / 1.01))


def chk_k2_numpy(run, key, x, ds, T, path):
    z = zs(x)
    V = directed(lambda a, b: K.gaussian_xme(a, b, 2), z)
    k1 = directed(lambda a, b: K.gaussian_xme(a, b, 1), z)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:gaussian_xme(k=2) (lag set {{1}}); {VERS}",
                k1_01=float(k1[0, 1]), k1_10=float(k1[1, 0]), k2_01=float(V[0, 1]), k2_10=float(V[1, 0]),
                k2_minus_k1_maxabs=float(np.nanmax(np.abs(V - k1))))


def chk_k2_equals_k1_shifted(run, key, x, ds, T, path):
    z = zs(x)
    V = directed(lambda a, b: K.gaussian_xme(a[1:], b[1:], 1), z)
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:gaussian_xme(k=1) on the z-scored rows with the first sample dropped; {VERS}",
                notes="history_length=2 uses the same lag set {1} as history_length=1, on samples t = 2..T-1")


# ----- cce / di ----------------------------------------------------------------------
def chk_cce_truth(run, key, x, ds, T, path):
    G = var_gammas([ds["A"]], ds["Sigma_E"], 6)
    pop = np.array([[np.nan, pop_cce(G, 0, 1)], [pop_cce(G, 1, 0), np.nan]])
    sims = sim_var([ds["A"]], ds["Sigma_E"], T, R_MC, rng_for(run["run_id"]))
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    S = load_pyspi(run["run_id"])
    write_stat(run["run_id"], m, s, f"population sum_m h(y_t | y lags 1..m-1, x lags 0..m-1) from Gamma(h) = A^h Sigma_X, z-scored; MC mean/SD ({R_MC} draws)",
               pop, S)


def chk_n10(run, key, x, ds, T, path):
    z = zs(x)
    if key == "cce":
        V = directed(lambda a, b: K.gaussian_cce(a, b, 10), z)
        V5 = directed(lambda a, b: K.gaussian_cce(a, b, 5), z)
    else:
        V = directed(lambda a, b: K.gaussian_di(a, b, 10, "pyspi"), z)
        V5 = directed(lambda a, b: K.gaussian_di(a, b, 5, "pyspi"), z)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:{'gaussian_cce' if key == 'cce' else 'gaussian_di'}(n=10); {VERS}",
                notes="documented history length 10 (supplement 1.4.7/1.4.8) via probe; configured value is 5",
                n10_01=float(V[0, 1]), n10_10=float(V[1, 0]), n5_01=float(V5[0, 1]), n5_10=float(V5[1, 0]))


def chk_configured_n(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ok = S.tolist() == [5.0, 5.0]
    write_qual(run["run_id"], ok, "pyspi/statistics/infotheory.py:435/:483 default n=5; config.yaml passes no n; supplement 1.4.7/1.4.8 document 10",
               kind="literature-only", cce_n=float(S[0]), di_n=float(S[1]), documented_n=10)


def chk_massey(run, key, x, ds, T, path):
    z = zs(x)
    V = directed(lambda a, b: K.gaussian_di(a, b, 5, "massey"), z)
    S = load_pyspi(run["run_id"])
    B = np.array([[np.nan, K.di_block_term(z[1])], [K.di_block_term(z[0]), np.nan]])
    write(run["run_id"], value=V, criterion="tolerance", rtol=1e-6, atol=1e-8, reference_kind="independent-impl",
          reference_detail=f"{KREF}:gaussian_di(form='massey') = sum_m [H(Y_t|Y^(m-1)) - H(Y_t|Y^(m-1),X^m)] (Massey 1990 eq. 6 / Kramer 1998), n=5",
          notes="canonical directed information; a mismatch here quantifies the block-entropy deviation",
          massey_01=float(V[0, 1]), massey_10=float(V[1, 0]), pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]),
          block_term_target1=float(B[0, 1]), block_term_target0=float(B[1, 0]))


def chk_massey_identity(run, key, x, ds, T, path):
    z = zs(x)
    S = load_pyspi(run["run_id"])
    V = directed(lambda a, b: K.gaussian_di(a, b, 5, "massey") + K.di_block_term(b), z)
    err = float(np.nanmax(np.abs(S - V)))
    write_qual(run["run_id"], err <= 1e-10, f"pyspi di = Massey DI_5 + B(target), B = {KREF}:di_block_term (sample identity)",
               kind="independent-impl", identity_maxabs=err)


def indep_pair(x_var2, x_iid):
    """Duplicate of probes/it_gaussian_probes.py:indep_pair."""
    return np.stack([np.asarray(x_iid[0], dtype=np.float64), np.asarray(x_var2[0], dtype=np.float64)])


def chk_indep_numpy(run, key, x, ds, T, path):
    X = indep_pair(x, make_data.load("iid_gaussian", T)["x"])
    z = zs(X)
    V = directed(lambda a, b: K.gaussian_di(a, b, 5, "pyspi"), z)
    M = directed(lambda a, b: K.gaussian_di(a, b, 5, "massey"), z)
    write_exact(run["run_id"], V, "independent-impl", f"{KREF}:gaussian_di(form='pyspi') on [iid row 0, var2 row 0]; {VERS}",
                massey_01=float(M[0, 1]), massey_10=float(M[1, 0]))


def chk_indep_truth(run, key, x, ds, T, path):
    a1, a2, s2 = float(ds["A1"][0, 0]), float(ds["A2"][0, 0]), float(ds["Sigma"][0, 0])
    G = var_gammas([np.array([[a1]]), np.array([[a2]])], np.array([[s2]]), 6)
    gamma = np.array([g[0, 0] for g in G])
    B = pop_di_excess_univariate(gamma)
    pop = np.array([[np.nan, B], [0.0, np.nan]])
    rng = rng_for(run["run_id"])
    tgt = sim_var([np.array([[a1]]), np.array([[a2]])], np.array([[s2]]), T, R_MC, rng)[:, 0]
    src = rng.standard_normal((R_MC, T))
    sims = np.stack([src, tgt], axis=1)
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    S = load_pyspi(run["run_id"])
    write_stat(run["run_id"], m, s, f"population pyspi-form DI for an independent white source and the AR(2) target x0 of var2_bivariate (a1={a1}, a2={a2}): B = sum_m h(Y^m)/m - h(Y^5) (Massey DI = 0); MC mean/SD ({R_MC} draws)",
               pop, S, population_massey=0.0, B_population=B)


# ----- si -------------------------------------------------------------------------------
def chk_si_truth(run, key, x, ds, T, path):
    SX, SXY, SE = ds["Sigma_X"], ds["Sigma_XY"], ds["Sigma_E"]
    vv = [SX[i, i] - SXY[i, i] ** 2 / SX[i, i] for i in range(2)]
    si = 0.5 * math.log(vv[0] * vv[1] / np.linalg.det(SE))
    pop = np.array([[np.nan, si], [si, np.nan]])
    sims = sim_var([ds["A"]], SE, T, R_MC, rng_for(run["run_id"]))
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    S = load_pyspi(run["run_id"])
    write_stat(run["run_id"], m, s, f"population SI = 0.5 ln(prod_v var(x_v(t+1)|x_v(t)) / det Sigma_E) (Ay 2015); MC mean/SD ({R_MC} draws)", pop, S)


# ----- estimator layer -------------------------------------------------------------------
def jidt_tasks(P):
    """Duplicate of probes/it_gaussian_probes.py:jidt_tasks."""
    ncol = 5 * P
    tasks = []
    for d in range(1, min(10, ncol) + 1):
        tasks.append(("H", tuple(range(d)), (), ()))
    tasks.append(("H", tuple(range(0, ncol, 2))[:10], (), ()))
    for d in range(1, min(6, ncol - 1) + 1):
        tasks.append(("MI", (0,), tuple(range(1, 1 + d)), ()))
    tasks.append(("MI", (0, 1), (ncol - 1, ncol - 2), ()))
    for d in range(1, min(5, ncol - 2) + 1):
        tasks.append(("CMI", (0,), (1,), tuple(range(2, 2 + d))))
    tasks.append(("CMI", (0, 2), (1,), (ncol - 1,)))
    return tasks


def lag_matrix(x, maxlag=4):
    P, T = x.shape
    return np.stack([x[p, maxlag - L:T - L] for p in range(P) for L in range(maxlag + 1)], axis=1)


def chk_direct_jidt(run, key, x, ds, T, path):
    M = lag_matrix(x)
    vals, vals0 = [], []
    for kind, A, B, C in jidt_tasks(x.shape[0]):
        a, b, c = M[:, list(A)], M[:, list(B)] if B else None, M[:, list(C)] if C else None
        if kind == "H":
            vals.append(K.gaussian_joint_entropy(a))
            vals0.append(K.gaussian_entropy(K._cov(a, ddof=0)))
        elif kind == "MI":
            vals.append(K.gaussian_mi(a, b))
            vals0.append(np.nan)
        else:
            vals.append(K.gaussian_cmi(a, b, c))
            vals0.append(np.nan)
    V = np.array(vals)
    S = load_pyspi(run["run_id"])
    hmask = np.array([t[0] == "H" for t in jidt_tasks(x.shape[0])])
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:gaussian_joint_entropy / gaussian_mi / gaussian_cmi (slogdet of the N-1 covariance) on the raw lag matrix; {VERS}",
                notes="JIDT Gaussian entropy/MI/CMI calculators driven directly with pyspi's properties",
                n_tasks=int(V.size), max_dim=int(max(len(t[1]) + len(t[2]) + len(t[3]) for t in jidt_tasks(x.shape[0]))),
                ddof0_entropy_maxabs_diff=float(np.nanmax(np.abs(S[hmask] - np.array(vals0)[hmask]))))


def chk_entropy_constant(run, key, x, ds, T, path):
    V = np.full(x.shape[0], H1)
    write_exact(run["run_id"], V, "analytic", "0.5(1+ln 2pi) = 1.4189385332046727: Gaussian entropy of a ddof=1 z-scored series with ddof=1 covariance",
                ddof0_would_give=H1 + 0.5 * math.log((T - 1) / T))


def chk_normalise(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ok = S[:3].tolist() == [1.0, 0.0, 0.0] and abs(S[3] - math.log(10)) <= 1e-12
    write_qual(run["run_id"], ok, "JIDT getProperty('NORMALISE'): entropy true, MI false, TE false; Gaussian entropy not normalised: H(10x)-H(x) = ln 10",
               kind="analytic", entropy_normalise=float(S[0]), mi_normalise=float(S[1]), te_normalise=float(S[2]),
               H_shift=float(S[3]), ln10=math.log(10), H_raw=float(S[4]),
               H_raw_numpy=float(K.gaussian_joint_entropy(x[0])))


def chk_labels(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])                       # rows: te_symbolic_k-1, te_symbolic_k-10, gc_gaussian_k-1
    ok = S[0].tolist() == [0.0, 1.0, 0.0] and S[1].tolist() == [0.0, 1.0, 0.0] and S[2].tolist() == [1.0, 0.0, 0.0]
    write_qual(run["run_id"], ok, "pyspi/statistics/infotheory.py:45-52 (symbolic branch returns before the 'linear' label) vs :56-57",
               kind="literature-only", notes="columns [linear, symbolic, nonlinear]",
               sym_k1_linear=float(S[0, 0]), sym_k1_symbolic=float(S[0, 1]), gc_k1_linear=float(S[2, 0]))


def chk_jar_provenance(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    ok = S[0] == 1.0 and S[1] == 1.0 and S[2] == 1.0
    write_qual(run["run_id"], ok, "JIDT tag v1.6.1 has no infodynamics/measures/continuous/EntropyCalculatorMultiVariateCommon.java (added in upstream commit 3d1f3fe, 2024-04-20); pass = the bundled jar has it as the Gaussian entropy superclass and BIAS_CORRECTION is not an entropy-calculator property",
               kind="literature-only", superclass_is_common=float(S[0]), common_class_loads=float(S[1]),
               entropy_bias_correction_is_none=float(S[2]), entropy_noise_seed=float(S[3]))


DEGEN_IDS = ["je_gaussian", "ce_gaussian", "mi_gaussian", "tlmi_gaussian", "cce_gaussian", "di_gaussian",
             "si_gaussian_k-1", "xme_gaussian_k1", "xme_gaussian_k10", "gc_gaussian_k-1_kt-1_l-1_lt-1",
             "gc_gaussian_k-max-10_tau-max-2", "te_symbolic_k-1_kt-1_l-1_lt-1", "te_symbolic_k-10_kt-1_l-1_lt-1"]


def chk_degenerate(run, key, x, ds, T, path):
    S = load_pyspi(run["run_id"])
    z = zs(x)
    # independent expectation: does the numpy covariance each SPI needs fail Cholesky?
    def pd(M):
        try:
            np.linalg.cholesky(K._cov(M))
            return True
        except np.linalg.LinAlgError:
            return False
    _, xp9 = K.xme_embedding(z[0], z[3], 10)
    lag9_pd = pd(xp9)
    pair_pd = pd(np.stack([z[0], z[2]], axis=1))
    expect_raise = {"je_gaussian": True, "ce_gaussian": True, "cce_gaussian": True, "di_gaussian": True,
                    "si_gaussian_k-1": True, "xme_gaussian_k10": not lag9_pd}
    rows = {ident: S[k] for k, ident in enumerate(DEGEN_IDS)}
    ok = True
    for ident in DEGEN_IDS:
        r = rows[ident]
        if expect_raise.get(ident, False):
            ok &= r[0] == 1.0 and r[4] == 1.0
        else:
            ok &= r[0] == 0.0
    ok &= np.isposinf(rows["mi_gaussian"][3]) and rows["gc_gaussian_k-1_kt-1_l-1_lt-1"][3] == 0.0
    ok &= np.isfinite(rows["tlmi_gaussian"][3]) and np.isfinite(rows["xme_gaussian_k1"][3])
    calc = S[len(DEGEN_IDS)]
    ok &= calc[0] == 1.0 and calc[3] == 1.0 and calc[1] == 2.0
    extra = {f"raised_{i.split('_')[0]}{'_k10' if 'k10' in i or 'k-10' in i else ''}{'_auto' if 'max' in i else ''}": float(rows[i][0])
             for i in DEGEN_IDS}
    write_qual(run["run_id"], ok, "numpy Cholesky of the same covariances (pair (0,2) bit-identical; 9-lag source covariance of noise-free narrowband rows)",
               kind="independent-impl", pair02_cov_pd=float(pair_pd), lag9_cov_row0_pd=float(lag9_pd),
               mi_02=float(rows["mi_gaussian"][3]), gc1_02=float(rows["gc_gaussian_k-1_kt-1_l-1_lt-1"][3]),
               gca_02=float(rows["gc_gaussian_k-max-10_tau-max-2"][3]),
               calculator_je_all_nan=float(calc[0]), calculator_mi_posinf=float(calc[1]), calculator_je_report_failed=float(calc[3]),
               **extra)


CHECKS_BY_TAG = {
    ("je", "equals-d2"): chk_equals_d2,
    ("*", "identity-mi"): chk_identity_mi,
    ("*", "truth-p01"): None, ("*", "truth-p23"): None,      # dispatched below
    ("ce", "order"): chk_order,
    ("mi", "invariance"): chk_invariance,
    ("*", "null"): chk_null,
    ("mi", "linear-only"): chk_linear_only,
    ("tlmi", "truth"): chk_tlmi_truth,
    ("*", "direction"): chk_direction,
    ("gc1", "ols"): chk_ols,
    ("gc1", "truth"): chk_gc_truth,
    ("gca", "params"): chk_params,
    ("sym1", "identity-gc1"): chk_identity_gc1,
    ("sym1", "monotone"): chk_monotone,
    ("*", "shift-numpy"): chk_shift_numpy,
    ("*", "shift-pattern"): chk_shift_pattern,
    ("xme1", "k2-numpy"): chk_k2_numpy,
    ("xme1", "k2-equals-k1-shifted"): chk_k2_equals_k1_shifted,
    ("cce", "truth"): chk_cce_truth,
    ("*", "n10"): chk_n10,
    ("cce", "configured-n"): chk_configured_n,
    ("di", "massey"): chk_massey,
    ("di", "massey-identity"): chk_massey_identity,
    ("di", "indep-numpy"): chk_indep_numpy,
    ("di", "indep-truth"): chk_indep_truth,
    ("si", "truth"): chk_si_truth,
    ("est", "entropy-constant"): chk_entropy_constant,
    ("est", "direct-jidt"): chk_direct_jidt,
    ("est", "normalise"): chk_normalise,
    ("est", "degenerate"): chk_degenerate,
    ("est", "jar-provenance"): chk_jar_provenance,
    ("sym1", "labels"): chk_labels,
}


def _truth_dispatch(run, key, x, ds, T, path):
    if run["dataset"] == "bivariate_gaussian":
        return chk_truth_pair(run, key, x, ds, T, path)
    if key in ("xme1", "xme10"):
        return chk_xme_truth(run, key, x, ds, T, path)
    if key == "tlmi":
        return chk_tlmi_truth(run, key, x, ds, T, path)
    raise KeyError(run["run_id"])


CHECKS_BY_TAG[("*", "truth-p01")] = _truth_dispatch
CHECKS_BY_TAG[("*", "truth-p23")] = _truth_dispatch
CHECKS_BY_TAG[("*", "truth-p03")] = _truth_dispatch


# ----- review follow-up: scaled_offset shifts, normalise invariance, cce null -------------
def chk_entropy_shift(run, key, x, ds, T, path):
    scale = np.asarray(ds["scale"], dtype=np.float64)
    V = np.log(np.abs(scale))
    S = load_pyspi(run["run_id"])
    write_exact(run["run_id"], V, "analytic",
                "H(s x + c) - H(x) = ln|s| for the ddof=1 Gaussian entropy (covariance scales by s^2, offset removed by the mean)",
                notes="pyspi cached univariate entropies (ce_gaussian, normalise=False): scaled_offset row p minus bivariate_gaussian row p, same draw",
                **{f"shift_x{p}": float(S[p]) for p in range(S.size)},
                **{f"ln_scale_x{p}": float(V[p]) for p in range(V.size)})


def chk_norm_invariance(run, key, x, ds, T, path):
    z = zs(x)
    ref = np.stack([stat_matrix(k, z) for k in ("mi", "tlmi", "gc1")])
    V = np.stack([ref, ref])
    S = load_pyspi(run["run_id"])
    diff = np.abs(S[0] - S[1])
    write_exact(run["run_id"], V, "independent-impl",
                f"{KREF}:gaussian_mi / gaussian_tlmi / gaussian_te(k=l=1) on the z-scored rows, for both the normalise=True and the normalise=False pyspi slots; {VERS}",
                notes="slots [normalise True, normalise False] x [mi, tlmi, gc1]; Gaussian MI/CMI/TE are invariant to per-channel affine maps",
                pyspi_true_vs_false_maxabs=float(np.nanmax(diff)),
                pyspi_true_vs_false_mi=float(np.nanmax(diff[0])), pyspi_true_vs_false_tlmi=float(np.nanmax(diff[1])),
                pyspi_true_vs_false_gc1=float(np.nanmax(diff[2])))


def chk_je_shift(run, key, x, ds, T, path):
    scale = np.abs(np.asarray(ds["scale"], dtype=np.float64))
    V = np.log(scale[:, None] * scale[None, :])
    np.fill_diagonal(V, np.nan)
    write_exact(run["run_id"], V, "analytic",
                "je(s_i x_i + c_i, s_j x_j + c_j) - je(x_i, x_j) = ln(s_i s_j) (det of the ddof=1 covariance scales by (s_i s_j)^2)",
                notes="pyspi je_gaussian normalise=False on scaled_offset minus on the unscaled bivariate_gaussian rows (same draw)")


def chk_cce_iid(run, key, x, ds, T, path):
    pop = np.full((2, 2), 5 * H1)
    np.fill_diagonal(pop, np.nan)
    sims = rng_for(run["run_id"]).standard_normal((R_MC, 2, T))
    m, s = mc(lambda z: stat_matrix(key, z), sims)
    S = load_pyspi(run["run_id"])
    write_stat(run["run_id"], m, s,
               f"population 5 H1 = 5 * 0.5(1+ln 2pi) = {5 * H1:.7f} for independent white Gaussian processes (every conditional entropy equals the marginal H1); MC mean/SD of {KREF}:gaussian_cce(n=5) over {R_MC} iid N(0,1) draws",
               pop, S, pop_entry=5 * H1, pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]),
               mc_mean_01=float(m[0, 1]), mc_sd_01=float(s[0, 1]))


CHECKS_BY_TAG[("est", "entropy-shift")] = chk_entropy_shift
CHECKS_BY_TAG[("est", "norm-invariance")] = chk_norm_invariance
CHECKS_BY_TAG[("je", "shift-d2")] = chk_je_shift
CHECKS_BY_TAG[("cce", "truth-iid")] = chk_cce_iid


def main():
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in man["runs"]:
        rid = run["run_id"]
        if only and not any(o in rid for o in only):
            continue
        if not (PYS / f"{rid}.npz").is_file() and not rid.endswith("|numpy"):
            # references that read pyspi's own value need the pyspi side first
            pass
        try:
            run_check(run)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    print(f"it_gaussian references: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
