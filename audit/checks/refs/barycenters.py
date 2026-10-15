"""References for the barycenters family (distance.bary.{euclidean,dtw,sgddtw,softdtw}). COMPARISON venv only.

Never imports tslearn (the library pyspi delegates to). Building blocks: numpy, scipy.optimize.minimize (L-BFGS-B, the
optimiser tslearn's softdtw_barycenter also calls), and the shared kernels in _kernels_distance (DTW dynamic programme
and path backtrack, soft-DTW forward/backward recursions, DBA of Petitjean et al. 2011 with tslearn's init and stopping
rule mirrored). The stochastic subgradient mean (SSG, Schultz & Jain 2018, arXiv:1701.06393v1 Alg. 3; Alg. 1 there
is the batch subgradient mean) is written here from the paper with tslearn 0.9.0's step schedule and stopping rule mirrored (tslearn/barycenters/dba.py:838-884) and driven by
numpy.random.RandomState(seed), the legacy stream that np.random.seed(seed) (harness.seed_all) resets.

Conventions mirrored from pyspi (cited in the family note): preprocessing per manifest path (kernel = raw, pipeline =
z-score ddof=1, pipeline-detrend = least-squares linear detrend then z-score), Directed.multivariate computes the pair
{a < b} once from z[[b, a]] in the order a = 0.., b = a+1.. and mirrors it (base.py:85-94, distance.py:247-263),
statistic = nanmean / nanmax of b or b**2.

Run ids are "<unit>|<dataset>|<T>|<path>|<tag>[|<identifier>][|p<procs>][|seed<s>]"; the tag selects the check.
Results are cached in out/exchange/barycenters/ref_cache.json keyed by the run spec, the tag's version and (for
references that read pyspi's own output) the pyspi npz bytes. Usage: python barycenters.py [--only TAG ...] [--force]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))
import make_data  # noqa: E402
import _kernels_distance as kd  # noqa: E402

FAMILY = "barycenters"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / FAMILY
CACHE_FILE = EXCH / "ref_cache.json"
EU, DT, SG, SD = ("distance.bary." + m for m in ("euclidean", "dtw", "sgddtw", "softdtw"))
SELF = "refs/barycenters.py"
VERSION = {"zero": 1, "formula": 1, "scratch": 1, "detrend": 1, "offset": 1, "invariance": 1, "popstat": 2,
           "sym": 1, "cold": 1, "dba": 1, "dba-bc": 1, "opt": 3, "xx": 1, "vs-eucl": 1, "ssg": 2, "ssg-bc": 1,
           "det": 1, "rngdep": 1, "cache": 1, "order": 1, "seeds50": 1, "vs-dba": 1, "sdtw": 1, "sdtw-bc": 1,
           "obj0": 1, "fd": 1, "gamma": 1}
PYSPI_DEPENDENT = {"popstat", "invariance", "sym", "cold", "opt", "vs-eucl", "det", "rngdep", "order", "vs-dba", "gamma", "fd"}


# ------------------------------------------------------------------------------------------------ helpers
def load_x(run):
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    return x


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)


def detrend_ls(x):
    """Least-squares linear detrend of each row, closed form: slope = sum (t - tbar)(x - xbar) / sum (t - tbar)^2."""
    x = np.asarray(x, float)
    t = np.arange(x.shape[1], dtype=float)
    tc = t - t.mean()
    slope = (x - x.mean(1, keepdims=True)) @ tc / (tc @ tc)
    return x - x.mean(1, keepdims=True) - slope[:, None] * tc[None, :]


def prep(x, path):
    if path == "kernel":
        return np.array(x, float)
    if path == "pipeline":
        return zscore(x)
    if path == "pipeline-detrend":
        return zscore(detrend_ls(x))
    raise ValueError(path)


def parse_ident(ident):
    head, rest = ident.split("_", 1)
    mode, stat = rest.rsplit("_", 1)
    return mode, head == "bary-sq", stat


def stat_of(b, ident):
    _, sq, stat = parse_ident(ident)
    v = np.asarray(b, float).ravel()
    v = v ** 2 if sq else v
    return float(np.nanmean(v) if stat == "mean" else np.nanmax(v))


def pairs(P):
    return [(a, c) for a in range(P) for c in range(a + 1, P)]


def matrix_from_pairs(P, vals):
    M = np.full((P, P), np.nan)
    for (a, c), v in zip(pairs(P), vals):
        M[a, c] = M[c, a] = v
    return M


def pyspi_value(rid):
    p = OUT_PY / f"{rid}.npz"
    with np.load(p) as z:
        return np.asarray(z["value"], float)


def asym(A):
    A = np.asarray(A, float)
    d = np.abs(A - A.T)
    return float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")


def rid_of(unit, ds, T, path, tag, ident=None):
    r = f"{unit}|{ds}|{T}|{path}|{tag}"
    return r + (f"|{ident}" if ident else "")


# ------------------------------------------------------------------------------------------- algorithms
def frechet_dtw(b, X):
    """DTW Frechet cost as tslearn's _mm_assignment reports it: mean_k DTW(b, x_k)^2 (unit weights / sum)."""
    return float(np.mean([kd.dtw_dp(b, xk) ** 2 for xk in X]))


def dba_from(X, init, max_iter=30, tol=1e-5):
    """kd.dba with an explicit initial barycenter (same assignment/update/stopping), for multi-start runs."""
    X = np.asarray(X, float)
    z = np.array(init, float)
    prev, costs = np.inf, []
    for _ in range(max_iter):
        num, cnt, cost = np.zeros_like(z), np.zeros_like(z), 0.0
        for xk in X:
            dist, path = kd.dtw_path(z, xk)
            cost += dist ** 2
            p = np.array(path)
            np.add.at(num, p[:, 0], xk[p[:, 1]])
            np.add.at(cnt, p[:, 0], 1.0)
        cost /= len(X)
        costs.append(cost)
        z = num / cnt
        if abs(prev - cost) < tol or prev < cost:
            break
        prev = cost
    return z, costs


def ssg(X, rng, max_iter=30, eta0=0.05, eta1=0.005, tol=1e-5, early_stop=True):
    """Stochastic subgradient mean (Schultz & Jain 2018, Pattern Recognit. 74:340; arXiv:1701.06393v1 Alg. 3) for unit weights:
    per sample k (random order each epoch), b <- b - 2 eta (V_k b - W_k x_k) with V_k = path valence, W_k x_k = sum of
    the x_k samples aligned to each b index. tslearn conventions mirrored (dba.py:838-884): init = pointwise mean;
    eta decreases by (eta0 - eta1)/n after each sample of epoch 0 only; the cost tested at the end of an epoch is the
    pre-update DTW^2 of the LAST processed sample times weights[0]/sum(weights) = 1/n; break when its change is < tol
    or it increases. Returns (b, epochs, stop reason)."""
    X = np.asarray(X, float)
    n, T = X.shape
    b = X.mean(0)
    eta = eta0
    prev = np.inf
    reason = "max_iter"
    it = -1
    for it in range(max_iter):
        for idx in rng.permutation(n):
            dist, path = kd.dtw_path(b, X[idx])
            cost = dist ** 2 * (1.0 / n)
            p = np.array(path)
            v = np.bincount(p[:, 0], minlength=T).astype(float)
            wx = np.bincount(p[:, 0], weights=X[idx][p[:, 1]], minlength=T)
            delta = v * b - wx
            b = b - (2.0 * eta / 1.0) * delta
            if it == 0:
                eta -= (eta0 - eta1) / n
        if early_stop:
            if abs(prev - cost) < tol:
                reason = "tol"
                break
            if prev < cost:
                reason = "increase"
                break
            prev = cost
    return b, it + 1, reason


def softdtw_bary(X, gamma=1.0, tol=1e-3, maxiter=50, init=None):
    """Soft-DTW barycenter (Cuturi & Blondel 2017, Sec. 4) with tslearn's settings mirrored (softdtw.py:38-117):
    unit weights (objective is a sum), init = pointwise mean, scipy L-BFGS-B with tol (= ftol = gtol) and maxiter.
    Objective/gradient from _kernels_distance.soft_dtw_barycenter_objective."""
    X = np.asarray(X, float)
    z0 = X.mean(0) if init is None else np.array(init, float)
    res = minimize(lambda z: kd.soft_dtw_barycenter_objective(z, X, gamma), z0, method="L-BFGS-B", jac=True,
                   tol=tol, options=dict(maxiter=maxiter))
    return res.x, res


def pair_barys(x, path, fn):
    """fn(z[[b, a]]) for every pair a < b, in Directed.multivariate order."""
    z = prep(x, path)
    return [fn(z[[c, a]]) for a, c in pairs(z.shape[0])]


# ---------------------------------------------------------------------------------------------- handlers
def h_zero(run):
    return dict(value=np.array(0.0), criterion="tolerance", rtol=0.0, atol=1e-12, reference_kind="analytic",
                reference_detail=f"{SELF}:h_zero; mean_t (z_i + z_j)/2 = (mean z_i + mean z_j)/2 = 0 for z-scored rows",
                notes="bary_euclidean_mean on the default (and detrend) pipeline: analytic value 0; max_abs is max|pyspi|")


def h_formula(run):
    x = load_x(run)
    T = x.shape[1]
    r = np.corrcoef(x)
    V = (1 + r) * (T - 1) / (2 * T)
    np.fill_diagonal(V, np.nan)
    return dict(value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="analytic",
                reference_detail=f"{SELF}:h_formula; mean_t((z_i+z_j)/2)^2 = (2(T-1) + 2 r (T-1))/(4T) with sum z^2 = T-1",
                notes="bary-sq_euclidean_mean = (1 + r)(T - 1)/(2T), r = numpy.corrcoef of the raw rows")


def h_scratch(run, path=None):
    x = load_x(run)
    ident = run["identifier"]
    vals = pair_barys(x, path or run["path"], lambda X: stat_of(X.mean(0), ident))
    return dict(value=matrix_from_pairs(x.shape[0], vals), criterion="exact", rtol=1e-10, atol=1e-12,
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_scratch; numpy z-score (ddof=1) / raw, b = (z_i + z_j)/2, nanmean/nanmax",
                notes="from-scratch Euclidean barycenter statistics")


def h_detrend(run):
    out = h_scratch(run)
    out.update(criterion="tolerance", rtol=1e-9, atol=1e-12,
               reference_detail=f"{SELF}:h_detrend; closed-form least-squares linear detrend, z-score ddof=1, (z_i+z_j)/2",
               notes="pipeline-detrend vs from-scratch detrend + z-score; rtol 1e-9 allows the lstsq-vs-closed-form rounding")
    return out


def h_offset(run):
    x = load_x(run)
    m = x.mean(1)
    vals = [(m[a] + m[c]) / 2 for a, c in pairs(x.shape[0])]
    d = make_data.load(run["dataset"], run["T"])
    return dict(value=matrix_from_pairs(x.shape[0], vals), criterion="exact", rtol=1e-10, atol=1e-12,
                reference_kind="analytic",
                reference_detail=f"{SELF}:h_offset; normalise=False: mean_t (x_i + x_j)/2 = (mean x_i + mean x_j)/2",
                notes="kernel path (no z-score): the mean statistic is the average of the two series' offsets",
                design_offsets=np.asarray(d["offset"], float), sample_means=m)


def h_invariance(run):
    ref_rid = rid_of(EU, "bivariate_gaussian", run["T"], "pipeline", "scratch", run["identifier"])
    return dict(value=pyspi_value(ref_rid), criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="same-lineage",
                reference_detail=f"{SELF}:h_invariance; pyspi output on bivariate_gaussian ({ref_rid})",
                notes="SELF-CONSISTENCY (pyspi vs pyspi): scaled_offset is bivariate_gaussian's draw with per-row scale and "
                      "offset; establishes invariance only")


def h_popstat(run):
    d = make_data.load(run["dataset"], run["T"])
    C = np.asarray(d["cov_true"], float)
    sd = np.sqrt(np.diag(C))
    rho = C / np.outer(sd, sd)
    T = run["T"]
    V = (1 + rho) / 2 * (T - 1) / T
    m = 6   # unordered pairs among 4 processes
    # two-sided Bonferroni critical value for alpha = 0.05 over the 2m = 12 ordered off-diagonal entries that
    # compare.py tests: Phi^-1(1 - 0.05/24) = 2.865 (equivalently alpha = 0.025 over the m = 6 unordered pairs;
    # the alpha = 0.05 value over 6 pairs, Phi^-1(1 - 0.05/12), would be 2.638)
    zb = 2.865
    se = (1 - rho ** 2) / (2 * np.sqrt(T)) * (T - 1) / T * zb / 2.0
    np.fill_diagonal(V, np.nan)
    np.fill_diagonal(se, np.nan)
    se_raw = se * 2.0 / zb
    A = pyspi_value(run["run_id"])
    raw_max_z = float(np.nanmax(np.abs(A - V) / se_raw))
    return dict(value=V, criterion="statistical", se=se, reference_kind="analytic",
                raw_max_z=raw_max_z, se_inflation=zb / 2.0,
                reference_detail=f"{SELF}:h_popstat; E -> (1 + rho)/2 (T-1)/T, sd(r) ~ (1 - rho^2)/sqrt(T) (Fisher 1915 delta method)",
                notes=(f"population check; se inflated by {zb}/2 so the 2-se rule is a two-sided Bonferroni test at alpha 0.05 over the {2 * m} ordered off-diagonal entries "
                       f"(alpha 0.025 over {m} unordered pairs); "
                       "compare.py max_z is on the inflated se, raw_max_z = max|pyspi - E| / se_raw (unscaled delta-method se)"))


def h_sym(run):
    A = pyspi_value(run["run_id"])
    a = asym(A)
    return dict(value=A.T, criterion="qualitative", pass_flag=int(a == 0.0), reference_kind="same-lineage",
                reference_detail=f"{SELF}:h_sym; pyspi A vs A.T",
                notes="SELF-CONSISTENCY (pyspi vs pyspi): symmetric matrix means A[i,j] carries no direction",
                max_abs_asym=a, min_offdiag=float(np.nanmin(A)), max_offdiag=float(np.nanmax(A)))


def bary_fn(mode, seed=42):
    if mode == "euclidean":
        return lambda X: X.mean(0)
    if mode == "dtw":
        return lambda X: kd.dba(X)[0]
    if mode == "sgddtw":
        return lambda X: ssg(X, np.random.RandomState(seed))[0]
    if mode == "softdtw":
        return lambda X: softdtw_bary(X)[0]
    raise ValueError(mode)


def h_cold(run):
    """Ordered-pair cold-cache matrix: M[i, j] = statistic of the barycenter of z[[i, j]] (fresh RandomState(seed) per
    ordered pair for sgddtw, matching the probe's np.random.seed before each call)."""
    x = load_x(run)
    z = prep(x, run["path"])
    ident = run["identifier"]
    fn = bary_fn(parse_ident(ident)[0], run["seed"])
    P = z.shape[0]
    V = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = stat_of(fn(z[[i, j]]), ident)
    M = pyspi_value(run["run_id"])
    mode = parse_ident(ident)[0]
    tol = dict(euclidean=(1e-10, 1e-12), dtw=(1e-9, 1e-10), sgddtw=(1e-9, 1e-10), softdtw=(1e-6, 1e-8))[mode]
    return dict(value=V, criterion="tolerance", rtol=tol[0], atol=tol[1], reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_cold; from-scratch barycenter per ORDERED pair",
                notes="cold-cache bivariate(i, j) on fresh Data objects; max_abs_asym_pyspi = max|M - M.T| shows input-order dependence",
                max_abs_asym_pyspi=asym(M), max_abs_asym_ref=asym(V))


def h_dba(run):
    x = load_x(run)
    ident = run["identifier"]
    res = pair_barys(x, run["path"], lambda X: (kd.dba(X), X.mean(0)))
    vals = [stat_of(b, ident) for (b, _), _ in res]
    iters = [len(c) for (_, c), _ in res]
    stalled = [int(np.max(np.abs(b - b0)) == 0.0) for (b, _), b0 in res]
    return dict(value=matrix_from_pairs(x.shape[0], vals), criterion="tolerance", rtol=1e-9, atol=1e-10,
                reference_kind="independent-impl",
                reference_detail="refs/_kernels_distance.py:dba (Petitjean et al. 2011; tslearn init/stopping mirrored)",
                notes="from-scratch DBA, max_iter 30, tol 1e-5; iterations = number of assignment steps run",
                iterations=np.array(iters), stalled_at_init=np.array(stalled),
                max_abs_stat=float(np.max(np.abs(vals))))


def h_dba_bc(run):
    x = load_x(run)
    res = pair_barys(x, run["path"], kd.dba)
    return dict(value=np.stack([b for b, _ in res]), criterion="tolerance", rtol=1e-9, atol=1e-10,
                reference_kind="independent-impl",
                reference_detail="refs/_kernels_distance.py:dba",
                notes="full DBA barycenters per pair (rows a<b, computed from z[[b, a]])",
                iterations=np.array([len(c) for _, c in res]))


def _backtrack_ties(b, xk):
    """Number of backtrack steps on the optimal path where two or three predecessors tie exactly."""
    A = kd.dtw_acc(b, xk)[1:, 1:]
    ties = 0
    i, j = A.shape[0] - 1, A.shape[1] - 1
    while (i, j) != (0, 0):
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        c = np.array([A[i - 1, j - 1], A[i - 1, j], A[i, j - 1]])
        k = int(np.argmin(c))
        ties += int(np.sum(c == c[k]) > 1)
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    return ties


def h_opt_dtw(run):
    """DBA optimality: F(b) = mean_k DTW(b, x_k)^2 at pyspi's barycenter vs the Euclidean init, each input series, and a
    multi-start from-scratch DBA (inits: each input series, 6 perturbed means b0 + 0.5 N(0,1), up to 100 iterations).
    pass_flag: F(pyspi) <= F(Euclidean init) (1 + 1e-12) for every pair (the MM descent property)."""
    x = load_x(run)
    z = prep(x, run["path"])
    B = pyspi_value(run["run_id"])
    F_py, F_eu, F_in, F_best, diag0, ties0, pert = [], [], [], [], [], [], []
    for r, (a, c) in enumerate(pairs(z.shape[0])):
        X = z[[c, a]]
        b0 = X.mean(0)
        F_py.append(frechet_dtw(B[r], X))
        F_eu.append(frechet_dtw(b0, X))
        F_in.append(min(frechet_dtw(X[0], X), frechet_dtw(X[1], X)))
        rng = np.random.default_rng(1000 + r)
        inits = [X[0], X[1]] + [b0 + 0.5 * rng.standard_normal(b0.size) for _ in range(6)]
        best = min(F_py[-1], F_in[-1])
        for init in inits:
            bb, _ = dba_from(X, init, max_iter=100)
            best = min(best, frechet_dtw(bb, X))
        F_best.append(best)
        # local diagnostics at the Euclidean init (open question: genuine local minimum or tie artefact?)
        paths = [kd.dtw_path(b0, xk)[1] for xk in X]
        diag0.append(int(all(all(i == j for i, j in p) for p in paths)))
        ties0.append(sum(_backtrack_ties(b0, xk) for xk in X))
        rngp = np.random.default_rng(2000 + r)
        dfs = []
        for _ in range(10):
            d = rngp.standard_normal(b0.size)
            d /= np.linalg.norm(d)
            for eps in (1e-3, 1e-2):
                dfs.append(frechet_dtw(b0 + eps * d, X) - F_eu[-1])
        pert.append(float(min(dfs)))
    F_py, F_eu, F_best = map(np.array, (F_py, F_eu, F_best))
    ok = bool(np.all(F_py <= F_eu * (1 + 1e-12)))
    return dict(value=np.array(0.0), criterion="optimality", pass_flag=int(ok), reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_opt_dtw; F from refs/_kernels_distance.py:dtw_dp; multi-start {SELF}:dba_from",
                notes="pass: F(pyspi) <= F(Euclidean init) for every pair; rel_gap_best = (F(pyspi) - best multi-start)/best",
                F_pyspi=F_py, F_euclid_init=F_eu, F_best_input_series=np.array(F_in), F_best_multistart=F_best,
                rel_gap_best=(F_py - F_best) / F_best, rel_drop_from_init=(F_eu - F_py) / F_eu,
                init_paths_diagonal=np.array(diag0), init_backtrack_ties=np.array(ties0),
                init_min_dF_perturbation=np.array(pert))


def h_xx(run):
    """Analytic: the barycenter of two identical series is that series (DBA, SSG: zero (sub)gradient on the diagonal
    path). For softdtw the analytic contrast does not hold (soft-DTW is not a divergence); there the reference is the
    from-scratch gamma = 1 barycenter (tolerance) and the deviation from x is reported."""
    x = load_x(run)
    ident = run["identifier"]
    mode = parse_ident(ident)[0]
    xs = x[0]
    if mode == "softdtw":
        b, res = softdtw_bary(np.stack([xs, xs]))
        B = pyspi_value(run["run_id"])
        return dict(value=b[None, :], criterion="tolerance", rtol=1e-6, atol=1e-6, reference_kind="independent-impl",
                    reference_detail=f"{SELF}:softdtw_bary (gamma=1, L-BFGS-B tol 1e-3, maxiter 50)",
                    notes="softdtw barycenter of (x, x); analytic Frechet mean x does NOT hold for soft-DTW",
                    var_x=float(np.var(xs)), var_b_pyspi=float(np.var(B)), max_abs_b_minus_x_pyspi=float(np.max(np.abs(B[0] - xs))),
                    max_x=float(xs.max()), max_b_pyspi=float(B.max()), nit=int(res.nit))
    V = np.full((2, 2), stat_of(xs, ident))
    np.fill_diagonal(V, np.nan)
    return dict(value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="analytic",
                reference_detail=f"{SELF}:h_xx; barycenter(x, x) = x",
                notes=f"{mode} barycenter of two identical series; value = statistic of x itself (kernel path, raw x)")


def h_vs_eucl(run):
    T, ds = run["T"], run["dataset"]
    A = pyspi_value(run["run_id"])
    E = pyspi_value(rid_of(EU, ds, T, "pipeline", "scratch", "bary_euclidean_max"))
    iu = np.triu_indices(A.shape[0], 1)
    gt, eq = A[iu] > E[iu], A[iu] == E[iu]
    if ds.startswith("delayed"):
        ok = bool(gt.all())
        rule = "delayed copies: bary_dtw_max > bary_euclidean_max for every pair"
    else:
        ok = bool(eq.all())
        rule = "narrowband: bary_dtw_max == bary_euclidean_max for every pair (DBA stays at its Euclidean init)"
    return dict(value=E, criterion="qualitative", pass_flag=int(ok), reference_kind="same-lineage",
                reference_detail=f"{SELF}:h_vs_eucl; pyspi bary_euclidean_max on the same data",
                notes=rule, dtw_max=A[iu], euclid_max=E[iu], frac_greater=float(gt.mean()), frac_equal=float(eq.mean()))


def ssg_matrix(z, ident, rng):
    """Simulate Directed.multivariate for sgddtw: pairs in loop order, one shared RNG stream."""
    out, info = [], []
    for a, c in pairs(z.shape[0]):
        b, ep, why = ssg(z[[c, a]], rng)
        out.append(b)
        info.append((ep, why))
    return out, info


def h_ssg(run):
    x = load_x(run)
    z = prep(x, run["path"])
    ident = run["identifier"]
    B, info = ssg_matrix(z, ident, np.random.RandomState(run["seed"]))
    return dict(value=matrix_from_pairs(z.shape[0], [stat_of(b, ident) for b in B]), criterion="tolerance",
                rtol=1e-9, atol=1e-10, reference_kind="independent-impl",
                reference_detail=f"{SELF}:ssg (Schultz & Jain 2018, arXiv:1701.06393v1 Alg. 3, tslearn schedule/stopping mirrored), RandomState({run['seed']})",
                notes="global-RNG stream simulated over pairs in Directed.multivariate order",
                epochs=np.array([e for e, _ in info]),
                stopped_on_increase=np.array([int(w == "increase") for _, w in info]))


def h_ssg_bc(run):
    x = load_x(run)
    z = prep(x, run["path"])
    B, info = ssg_matrix(z, run["identifier"], np.random.RandomState(run["seed"]))
    return dict(value=np.stack(B), criterion="tolerance", rtol=1e-9, atol=1e-10, reference_kind="independent-impl",
                reference_detail=f"{SELF}:ssg", notes="full SSG barycenters per pair (rows a<b)",
                epochs=np.array([e for e, _ in info]))


def h_det(run):
    S = pyspi_value(run["run_id"])
    d = float(np.nanmax(np.abs(S[0] - S[1])))
    return dict(value=S, criterion="qualitative", pass_flag=int(d == 0.0), reference_kind="same-lineage",
                reference_detail=f"{SELF}:h_det; two seed_all({run['seed']}) computations in one process",
                notes="SELF-CONSISTENCY (pyspi vs pyspi): determinism under the harness seed", max_abs_repeat_diff=d)


def h_rngdep(run):
    xc = zscore(np.asarray(make_data.load("iid_gaussian", 64)["x"], float))
    rng = np.random.RandomState(run["seed"])
    ssg_matrix(xc, "bary_sgddtw_mean", rng)           # the earlier RNG consumer
    x = load_x(run)
    z = prep(x, run["path"])
    ident = run["identifier"]
    B, _ = ssg_matrix(z, ident, rng)
    V = matrix_from_pairs(z.shape[0], [stat_of(b, ident) for b in B])
    plain = pyspi_value(rid_of(SG, run["dataset"], run["T"], run["path"], "ssg", ident))
    after = pyspi_value(run["run_id"])
    return dict(value=V, criterion="tolerance", rtol=1e-9, atol=1e-10, reference_kind="independent-impl",
                reference_detail=f"{SELF}:ssg after simulating the consumer's draws on RandomState({run['seed']})",
                notes="value after another sgddtw SPI consumed the global RNG; max_abs_change_vs_plain compares with the plain seed-42 run",
                max_abs_change_vs_plain=float(np.nanmax(np.abs(after - plain))),
                max_rel_change_vs_plain=float(np.nanmax(np.abs(after - plain) / np.abs(plain))))


def h_cache(run):
    x = load_x(run)
    z = prep(x, run["path"])
    ident = run["identifier"]
    B, _ = ssg_matrix(z, ident, np.random.RandomState(run["seed"]))
    V = matrix_from_pairs(z.shape[0], [stat_of(b, ident) for b in B])
    return dict(value=np.concatenate([V.ravel(), [0.0]]), criterion="tolerance", rtol=1e-9, atol=1e-10,
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:ssg with RandomState({run['seed']}) (barycenter drawn ONCE, by the first SPI)",
                notes="shared Data: bary_sgddtw_mean then this identifier; last element = RNG advanced during the second SPI (expected 0)")


def h_seeds50(run):
    x = load_x(run)
    z = prep(x, run["path"])
    X = z[[1, 0]]
    rows = [ssg(X, np.random.RandomState(s))[0] for s in range(50)]
    R = np.stack(rows)
    st = {k: np.array([stat_of(r, k) for r in R]) for k in
          ("bary_sgddtw_mean", "bary_sgddtw_max", "bary-sq_sgddtw_mean", "bary-sq_sgddtw_max")}
    B0 = X.mean(0)
    Bd = kd.dba(X)[0]
    return dict(value=R, criterion="tolerance", rtol=1e-9, atol=1e-10, reference_kind="independent-impl",
                reference_detail=f"{SELF}:ssg with RandomState(s), s = 0..49",
                notes="seed spread of the sgddtw barycenter for one pair; sd_* = SD over 50 seeds",
                sd_mean=float(st["bary_sgddtw_mean"].std(ddof=1)), sd_max=float(st["bary_sgddtw_max"].std(ddof=1)),
                sd_sq_mean=float(st["bary-sq_sgddtw_mean"].std(ddof=1)), sd_sq_max=float(st["bary-sq_sgddtw_max"].std(ddof=1)),
                range_max=float(np.ptp(st["bary_sgddtw_max"])), range_mean=float(np.ptp(st["bary_sgddtw_mean"])),
                euclid_max=float(B0.max()), dba_max=float(Bd.max()), avg_max=float(st["bary_sgddtw_max"].mean()))


def h_vs_dba(run):
    """Qualitative (cap 5): objective of pyspi's sgddtw barycenter vs the from-scratch DBA barycenter, an SSG run of 30
    full epochs without early stopping (RandomState(seed) per pair), and the Euclidean init.
    pass_flag: F(sgddtw) < F(Euclidean init) for every pair (the subgradient run descends)."""
    x = load_x(run)
    z = prep(x, run["path"])
    B = pyspi_value(run["run_id"])
    F_sg, F_db, F_30, F_eu, dist = [], [], [], [], []
    for r, (a, c) in enumerate(pairs(z.shape[0])):
        X = z[[c, a]]
        bd = kd.dba(X)[0]
        b30 = ssg(X, np.random.RandomState(run["seed"]), early_stop=False)[0]
        F_sg.append(frechet_dtw(B[r], X)); F_db.append(frechet_dtw(bd, X))
        F_30.append(frechet_dtw(b30, X)); F_eu.append(frechet_dtw(X.mean(0), X))
        dist.append(float(np.sqrt(np.mean((B[r] - bd) ** 2))))
    F_sg, F_db, F_30, F_eu = map(np.array, (F_sg, F_db, F_30, F_eu))
    return dict(value=np.array(0.0), criterion="qualitative", pass_flag=int(np.all(F_sg < F_eu)),
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_vs_dba; F = mean_k DTW^2 via refs/_kernels_distance.py:dtw_dp",
                notes="pass: sgddtw lowers the DTW Frechet cost below the Euclidean init for every pair; gaps relative to DBA reported",
                F_sgddtw=F_sg, F_dba=F_db, F_ssg30=F_30, F_euclid=F_eu,
                rel_gap_sgd_vs_dba=(F_sg - F_db) / F_db, rel_gap_ssg30_vs_dba=(F_30 - F_db) / F_db,
                rms_sgd_minus_dba=np.array(dist))


def h_sdtw(run):
    x = load_x(run)
    ident = run["identifier"]
    res = pair_barys(x, run["path"], softdtw_bary)
    return dict(value=matrix_from_pairs(x.shape[0], [stat_of(b, ident) for b, _ in res]), criterion="tolerance",
                rtol=1e-6, atol=1e-8, reference_kind="independent-impl",
                reference_detail=f"{SELF}:softdtw_bary (kernel objective + scipy L-BFGS-B tol 1e-3 maxiter 50)",
                notes="statistics of the from-scratch soft-DTW barycenter; optimiser-path sensitivity allowed at rtol 1e-6",
                nit=np.array([r.nit for _, r in res]), success=np.array([int(r.success) for _, r in res]))


def h_sdtw_bc(run):
    x = load_x(run)
    res = pair_barys(x, run["path"], softdtw_bary)
    return dict(value=np.stack([b for b, _ in res]), criterion="tolerance", rtol=1e-6, atol=1e-6,
                reference_kind="independent-impl", reference_detail=f"{SELF}:softdtw_bary",
                notes="full soft-DTW barycenters per pair", nit=np.array([r.nit for _, r in res]))


def h_obj0(run):
    x = load_x(run)
    rows = []
    for X in pair_barys(x, run["path"], lambda X: X):
        f, g = kd.soft_dtw_barycenter_objective(X.mean(0), X)
        rows.append(np.concatenate([[f], g]))
    return dict(value=np.stack(rows), criterion="tolerance", rtol=1e-9, atol=1e-9, reference_kind="independent-impl",
                reference_detail="refs/_kernels_distance.py:soft_dtw_barycenter_objective (Cuturi & Blondel 2017 Alg. 1-2)",
                notes="soft-DTW barycenter objective and gradient at the Euclidean init: [f, grad...] per pair")


def h_fd(run):
    x = load_x(run)
    rows = []
    h = 1e-6
    for X in pair_barys(x, run["path"], lambda X: X):
        z0 = X.mean(0)
        f = kd.soft_dtw_barycenter_objective(z0, X)[0]
        g = np.empty_like(z0)
        for t in range(z0.size):
            e = np.zeros_like(z0); e[t] = h
            g[t] = (kd.soft_dtw_barycenter_objective(z0 + e, X)[0] - kd.soft_dtw_barycenter_objective(z0 - e, X)[0]) / (2 * h)
        rows.append(np.concatenate([[f], g]))
    return dict(value=np.stack(rows), criterion="tolerance", rtol=1e-5, atol=1e-5, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_fd; central differences (h = 1e-6) of the kernel soft-DTW objective",
                notes="tslearn's analytic gradient at the init vs finite differences")


def h_opt_sdtw(run):
    """Soft-DTW optimality at pyspi's returned barycenter b: gradient of the kernel objective at b vs at the init b0, and
    a tight re-optimisation from b (L-BFGS-B, ftol = gtol = 1e-12, maxiter 300/150/500 at T = 64/256/1024, which lets
    it reach scipy's own convergence test; reopt_converged and reopt_max_grad record whether it did). At T = 4096 no
    re-optimisation (cost): an Armijo backtracking line search along -grad gives only a lower bound on the gap.
    pass_flag (declared before running): relative objective gap (f(b) - f_best)/|f_best| <= 1e-3 for every pair."""
    x = load_x(run)
    z = prep(x, run["path"])
    B = pyspi_value(run["run_id"])
    T = z.shape[1]
    maxit = {64: 300, 256: 150, 1024: 500}.get(T, 0)
    gaps, gr, f_b, f_star, dmean, dmax, nits, conv, gfin = [], [], [], [], [], [], [], [], []
    for r, (a, c) in enumerate(pairs(z.shape[0])):
        X = z[[c, a]]
        b = B[r]
        f0, g0 = kd.soft_dtw_barycenter_objective(X.mean(0), X)
        fb, gb = kd.soft_dtw_barycenter_objective(b, X)
        gr.append(float(np.max(np.abs(gb)) / np.max(np.abs(g0))))
        if maxit:
            res = minimize(lambda q: kd.soft_dtw_barycenter_objective(q, X), b, method="L-BFGS-B", jac=True,
                           options=dict(maxiter=maxit, ftol=1e-12, gtol=1e-12))
            fs, bs = float(min(res.fun, fb)), res.x
            nits.append(int(res.nit)); conv.append(int(res.success)); gfin.append(float(np.max(np.abs(res.jac))))
        else:
            step, fs, bs = 1.0, fb, b
            gn2 = float(gb @ gb)
            for _ in range(40):
                cand = b - step * gb
                fc = kd.soft_dtw_barycenter_objective(cand, X)[0]
                if fc <= fb - 1e-4 * step * gn2:
                    fs, bs = fc, cand
                    break
                step *= 0.5
            nits.append(0); conv.append(0); gfin.append(float("nan"))
        f_b.append(fb); f_star.append(fs)
        gaps.append((fb - fs) / abs(fs))
        dmean.append(float(abs(np.mean(bs) - np.mean(b))))
        dmax.append(float(abs(np.max(bs) - np.max(b))))
    gaps = np.array(gaps)
    return dict(value=np.array(0.0), criterion="optimality", pass_flag=int(np.all(gaps <= 1e-3)),
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_opt_sdtw; refs/_kernels_distance.py:soft_dtw_barycenter_objective",
                notes=("pass: relative objective gap to a tight re-optimisation <= 1e-3 for every pair" if maxit else
                       "T=4096: gap is a LOWER BOUND from one Armijo line search along -grad (no re-optimisation)"),
                rel_gap=gaps, grad_ratio=np.array(gr), f_pyspi=np.array(f_b), f_best=np.array(f_star),
                abs_change_mean_stat=np.array(dmean), abs_change_max_stat=np.array(dmax), reopt_nit=np.array(nits),
                reopt_converged=np.array(conv), reopt_max_grad=np.array(gfin))


def h_gamma(run):
    x = load_x(run)
    xs = x[0]
    X = np.stack([xs, xs])
    dev, var, nit = [], [], []
    for g in (1.0, 0.1, 0.01):
        b, res = softdtw_bary(X, gamma=g)
        dev.append(float(np.max(np.abs(b - xs)))); var.append(float(np.var(b))); nit.append(int(res.nit))
    B = pyspi_value(run["run_id"])[0]
    ok = bool(var[0] < np.var(xs) and dev[0] > dev[1] > dev[2])
    return dict(value=np.array(0.0), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
                reference_detail=f"{SELF}:h_gamma; Frechet mean of (x, x) under a divergence is x; soft-DTW gamma sweep via {SELF}:softdtw_bary",
                notes="pass: gamma=1 barycenter of (x, x) shrinks (var(b) < var(x)) and max|b - x| decreases as gamma -> 0",
                gammas=np.array([1.0, 0.1, 0.01]), max_abs_b_minus_x=np.array(dev), var_b=np.array(var),
                var_x=float(np.var(xs)), nit=np.array(nit), pyspi_max_abs_b_minus_x=float(np.max(np.abs(B - xs))),
                pyspi_var_b=float(np.var(B)))


HANDLERS = {"zero": h_zero, "formula": h_formula, "scratch": h_scratch, "detrend": h_detrend, "offset": h_offset,
            "invariance": h_invariance, "popstat": h_popstat, "sym": h_sym, "cold": h_cold, "dba": h_dba,
            "dba-bc": h_dba_bc, "xx": h_xx, "vs-eucl": h_vs_eucl, "ssg": h_ssg, "ssg-bc": h_ssg_bc, "det": h_det,
            "rngdep": h_rngdep, "cache": h_cache, "seeds50": h_seeds50, "vs-dba": h_vs_dba, "sdtw": h_sdtw,
            "sdtw-bc": h_sdtw_bc, "obj0": h_obj0, "fd": h_fd, "gamma": h_gamma, "order": h_cold}


def handler_for(run, tag):
    if tag == "opt":
        return h_opt_dtw if run["unit_id"] == DT else h_opt_sdtw
    return HANDLERS[tag]


def cache_key(run, tag):
    h = hashlib.sha256(json.dumps({k: run.get(k) for k in sorted(run)}, sort_keys=True).encode())
    h.update(str(VERSION[tag]).encode())
    if tag in PYSPI_DEPENDENT:
        p = OUT_PY / f"{run['run_id']}.npz"
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT_REF.mkdir(parents=True, exist_ok=True)
    EXCH.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.is_file() else {}
    runs = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())["runs"]
    n_ok = n_err = n_skip = 0
    for run in runs:
        rid = run["run_id"]
        tag = rid.split("|")[4]
        if a.only and tag not in a.only:
            continue
        target = OUT_REF / f"{rid}.npz"
        try:
            key = cache_key(run, tag)
        except Exception:
            key = None
        if not a.force and key and cache.get(rid) == key and target.is_file():
            n_skip += 1
            continue
        t0 = time.perf_counter()
        try:
            out = handler_for(run, tag)(run)
        except FileNotFoundError as exc:
            print(f"SKIP {rid}: pyspi output missing ({exc})", file=sys.stderr)
            n_err += 1
            continue
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
            n_err += 1
            continue
        out = {k: (np.asarray(v) if not isinstance(v, str) else np.asarray(v)) for k, v in out.items()}
        out["ref_time_s"] = np.asarray(time.perf_counter() - t0)
        np.savez(target, **out)
        cache[rid] = key
        CACHE_FILE.write_text(json.dumps(cache, indent=0, sort_keys=True))
        n_ok += 1
        print(f"ok {rid} ({float(out['ref_time_s']):.1f} s)", file=sys.stderr)
    print(f"barycenters refs: {n_ok} written, {n_skip} cached, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
