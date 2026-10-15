"""hhg_mgc family references (COMPARISON venv): distance.hhg and distance.mgc.

pyspi computes, for every ordered pair (i, j) of processes, hyppo's HHG / MGC statistic of
x = process i and y = process j as (T, 1) arrays (pyspi/statistics/distance.py:66-77, 100-111),
with distance matrices from sklearn's dot-product Euclidean distances (hyppo/tools/common.py:499
-> sklearn/metrics/pairwise.py:403-421). HHG is Directed (both orders stored); MGC is Undirected
(upper triangle copied over the lower: A[i, j] = A[j, i] = MGC(x_min(i,j), x_max(i,j))).

Nothing here imports hyppo, sklearn or scipy.stats.multiscale_graphcorr. Building blocks:
  * _kernels_distance._pdist (exact |x_a - x_b|), _sk_sqdist (sklearn's float64 arithmetic
    -2 x x^T + x^2 + x^2, clip, zero diagonal, MIRRORED not imported), hhg (Heller et al. 2013
    from scratch, used to validate hhg_from_dist), dcorr_from_dist (U-centred dCor^2).
  * hhg_from_dist (this file): Heller, Heller & Gorfine 2013 Biometrika 100:503-510, Sec. 2:
    S(i,j) = (N)(A12 A21 - A11 A22)^2 / (A1. A2. A.1 A.2), N = n - 2, '<=' indicators over
    k not in {i, j}, S = 0 for a zero margin, T = sum_{i != j} S(i,j); counts from a 2-D
    cumulative histogram of per-row max-ranks (O(n^2) per row), integer arithmetic.
  * mgc_from_dist (this file): Shen, Priebe & Vogelstein 2020 JASA 115:280-291, Sec. 3.1 eq. (6)
    (column centring with 1/(n-1), zero diagonal; column-wise minimal (dense) ranks), local
    dCov^{k,l} = sum A^k o B^l' - S_A(k) S_B(l) / c, local variances likewise, dCorr = cov /
    sqrt(var var); Sec. 4.1 / Corollary 3 thresholding tau = 2 F^-1_Beta(s(s-3)/4 - 1/2, same)
    (1 - 0.02/s) - 1, largest connected component, smoothed maximum. Constants selectable:
      impl  : c = n^2, s = n - 1, region >= ceil(0.02 max(m, n')) min(m, n')  (what SciPy does)
      paper : c = n(n - 1), s = n, region >= 2n                                (Shen 2020 text)
      rmgc  : c = n^2, s = n - 1, region >= 2 min(m, n')                       (neurodata/r-mgc)
    Connected components use the scipy.ndimage.label primitive (4-connectivity, raster label
    order; first label on size ties), as the trace stage proposed; beta.ppf from scipy.stats.
  * mgc_global: the global-scale analytic value sum A_ab B_ba / sqrt(sum A_ab A_ba sum B_ab B_ba).
  * dcor2_gauss: population dCor^2 of a bivariate normal (Szekely, Rizzo & Bakirov 2007 Ann.
    Stat. 35:2769, Theorem 7): [r asin r + sqrt(1-r^2) - r asin(r/2) - sqrt(4-r^2) + 1] /
    [1 + pi/3 - sqrt 3].
  * HHG null mean (derived here): under independence each table is hypergeometric given its
    margins, so E[S | margins > 0] = N/(N-1); a margin is 0 iff j is i's nearest or farthest
    neighbour in x or in y, P(no zero margin) = ((n-3)/(n-1))^2, hence
    E[T] = n(n-1) ((n-3)/(n-1))^2 (n-2)/(n-3) = n(n-2)(n-3)/(n-1).

Run tags (last field of the run_id; '-pXY' suffix = process subset):
  def          from-scratch statistic on EXACT distances of the rows pyspi receives (kernel: raw;
               pipeline: own ddof=1 z-score); tolerance (HHG rtol 1e-12; MGC atol 1e-10)
  skdist       same, on sklearn-mirrored distances (isolates the estimator on tied data)
  exact        tied data (logistic_maps, shared_envelope) vs the EXACT-distance definition:
               tests the sklearn-rounding candidate (a fail confirms it)
  hyppo-exact  probe: the delegate fed exact distances vs the exact-distance reference
               (attribution aid; the pyspi side is same-lineage, the reference is independent)
  copies       probe rows (x0, x0, 2x0+1, x1): analytic N(N-2)(N-3) (HHG) / 1 (MGC) on the copy
               pairs, from-scratch value on the pairs with x1; exact
  dup          shared_envelope processes (0, 2) (bit-identical copy): analytic, exact
  dir          pipeline matrix vs from-scratch per orientation; extra keys max|A - A.T|
  ordered      probe Directed.multivariate (MGC, no mirror) vs from-scratch per ordered pair
  null         statistical: HHG analytic E[T] = n(n-2)(n-3)/(n-1) with Monte Carlo SD; MGC Monte
               Carlo mean/SD of the from-scratch statistic on iid N(0,1) pairs; se = SD *
               z_{1-0.025/m}/2 (Bonferroni over m unique pairs)
  perm         qualitative: per-pair 200-permutation null of the from-scratch statistic (exact distances, kernel
               rows); iid_gaussian: KS of the pyspi percentiles vs U(0,1) and no Bonferroni upper-tail rejection;
               gauss_ar1_pair: upper-tail p < 0.025 (detection). HHG also: pooled permutation mean vs analytic E[T]
  order        qualitative: dependent pairs above the family-wise null band, independent pairs
               inside it; bivariate_gaussian also rho=0.8 pair > rho=0.3 pair. Band = empirical
               quantiles q(0.025/m), q(1-0.025/m) of 2000 Monte Carlo draws (mc_quantile_band);
               a first draft used mean +- z SD, which under-covers HHG's right-skewed null
  scaleinv     scaled_offset (pipeline) vs from-scratch on the z-scored bivariate_gaussian draw
  trend        trending through the plain pipeline vs from-scratch on own z-scored rows
  detrend      trending through pipeline-detrend vs from-scratch on own detrended + z-scored rows
  trendqual    qualitative: (1,2), (1,3) (independent, both trending) above the null band
               (mc_quantile_band, m = 6) without detrend and inside it after detrend
  global       MGC analytic global-scale identity (column-centred dCorr^{n,n}); exact
  udcorr       MGC vs U-centred bias-corrected dCor^2 (_kernels_distance.dcorr_from_dist):
               related quantity, Shen 2020 Sec. 3.2 says equal up to O(1/n): atol = 1/n
  optscale     probe exporting SciPy's mgc_dict['opt_scale'] and global map value vs the
               from-scratch optimal scale (sklearn-mirrored distances); exact
  paper, rmgc  MGC with the paper's / r-mgc's constants (sklearn-mirrored distances so only the
               constants differ); paper atol 2e-3 (trace expectation for n^2 vs n(n-1)),
               rmgc atol 1e-10 (only the region rule differs)
  pop          statistical: population dCor^2(rho) of the dataset's Gaussian pairs (0 for cross
               pairs replaced by the Monte Carlo null mean), se = Monte Carlo SD at the same T
               times the Bonferroni factor
  smalln       qualitative: MGC raises ValueError at n = 4 (scipy/_mgc.py:331-333)

Writes out/ref/<run_id>.npz per the compare.py contract. Monte Carlo results are cached in
out/exchange/hhg_mgc/mc_cache.json (delete it to recompute).
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np
import scipy
from scipy import ndimage, stats

import make_data
from _kernels_distance import _pdist, _sk_sqdist, dcorr_from_dist, hhg as kernel_hhg

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / "hhg_mgc"
MANIFEST = CHECKS / "manifest" / "hhg_mgc.json"
SELF = "audit/checks/refs/hhg_mgc.py"
KD = "audit/checks/refs/_kernels_distance.py"
MC_SEED = 20261015


# ------------------------------------------------------------------ preprocessing
def zscore(X):
    X = np.asarray(X, float)
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)


def detrend(X):
    X = np.asarray(X, float)
    T = X.shape[1]
    A = np.column_stack([np.ones(T), np.arange(T, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)
    return X - (A @ coef).T


def preprocess(X, path):
    if path == "kernel":
        return np.asarray(X, float)
    if path == "pipeline":
        return zscore(X)
    if path == "pipeline-detrend":
        return zscore(detrend(X))
    raise ValueError(path)


def derive(kind, x):
    """Mirror of probes/hhg_mgc_probes.py derive()."""
    x = np.asarray(x, dtype=np.float64)
    if kind == "copies":
        return np.vstack([x[0], x[0].copy(), 2.0 * x[0] + 1.0, x[1]])
    raise ValueError(kind)


def dist(z, mode):
    if mode == "exact":
        return _pdist(z)
    if mode == "sk":
        return np.sqrt(_sk_sqdist(z))
    raise ValueError(mode)


# ------------------------------------------------------------------------- HHG
def hhg_from_dist(dx, dy):
    """Heller et al. 2013 T from (n, n) distance matrices (row i = centre point)."""
    n = dx.shape[0]
    N = n - 2
    S = np.zeros((n, n))
    for i in range(n):
        rx = np.searchsorted(np.sort(dx[i]), dx[i], side="right").astype(np.int64)   # #{k: d_ik <= d_ij}
        ry = np.searchsorted(np.sort(dy[i]), dy[i], side="right").astype(np.int64)
        H = np.bincount(rx * (n + 1) + ry, minlength=(n + 1) ** 2).reshape(n + 1, n + 1)
        C = H.cumsum(0).cumsum(1)
        t11 = C[rx, ry] - 2                       # k = i and k = j satisfy both '<='
        t12 = (rx - 2) - t11
        t21 = (ry - 2) - t11
        t22 = N - t11 - t12 - t21
        den = (t11 + t12) * (t21 + t22) * (t11 + t21) * (t12 + t22)
        num = N * (t12 * t21 - t11 * t22) ** 2
        s = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        s[i] = 0.0
        S[i] = s
    mask = ~np.eye(n, dtype=bool)
    return float(np.sum(S[mask]))


def hhg_null_mean(n):
    return n * (n - 2) * (n - 3) / (n - 1)


# ------------------------------------------------------------------------- MGC
def dense_col_rank(D):
    """0-based minimal (dense) rank of D[:, j] within column j."""
    order = np.argsort(D, axis=0, kind="stable")
    Ds = np.take_along_axis(D, order, 0)
    inc = np.zeros(D.shape, dtype=np.int64)
    inc[1:] = np.diff(Ds, axis=0) > 0
    dr = np.cumsum(inc, axis=0)
    R = np.empty_like(dr)
    np.put_along_axis(R, order, dr, 0)
    return R


def col_centre(D):
    n = D.shape[0]
    A = D - D.sum(0, keepdims=True) / (n - 1)
    np.fill_diagonal(A, 0.0)
    return A


def _cum2(rk, rl, w, mk, ml):
    H = np.bincount(rk * ml + rl, weights=w, minlength=mk * ml).reshape(mk, ml)
    return H.cumsum(0).cumsum(1)


def local_corr(Dx, Dy, const="n2"):
    n = Dx.shape[0]
    A, B = col_centre(Dx), col_centre(Dy)
    RA, RB = dense_col_rank(Dx), dense_col_rank(Dy)
    mx, my = int(RA.max()) + 1, int(RB.max()) + 1
    c = float(n) ** 2 if const == "n2" else float(n) * (n - 1)
    cov = _cum2(RA.ravel(), RB.T.ravel(), (A * B.T).ravel(), mx, my)
    SA = np.bincount(RA.ravel(), A.ravel(), minlength=mx).cumsum()
    SB = np.bincount(RB.ravel(), B.ravel(), minlength=my).cumsum()
    cov = cov - np.outer(SA, SB) / c
    vx = np.diag(_cum2(RA.ravel(), RA.T.ravel(), (A * A.T).ravel(), mx, mx)) - SA ** 2 / c
    vy = np.diag(_cum2(RB.ravel(), RB.T.ravel(), (B * B.T).ravel(), my, my)) - SB ** 2 / c
    with np.errstate(all="ignore"):
        C = cov / np.sqrt(np.outer(vx, vy))
    C[C > 1] = 1.0
    C[vx <= 0, :] = 0.0
    C[:, vy <= 0] = 0.0
    return C


VARIANTS = {"impl": ("n2", "n-1", "impl"), "paper": ("nn1", "n", "2n"), "rmgc": ("n2", "n-1", "2min")}


def mgc_from_dist(Dx, Dy, variant="impl", return_scale=False):
    const, samp, region = VARIANTS[variant]
    n = Dx.shape[0]
    C = local_corr(Dx, Dy, const)
    m, nn = C.shape
    g = C[m - 1, nn - 1]
    out = (g, (m, nn))
    if m > 1 and nn > 1:
        s = n - 1 if samp == "n-1" else n
        sh = s * (s - 3) / 4 - 0.5
        tau = stats.beta.ppf(1 - 0.02 / s, sh, sh) * 2 - 1
        sig = C > max(tau, g)
        if sig.sum() > 0:
            lab, _ = ndimage.label(sig)
            cnt = np.bincount(lab.ravel())
            R = lab == (int(np.argmax(cnt[1:])) + 1)
            need = {"impl": math.ceil(0.02 * max(m, nn)) * min(m, nn), "2n": 2 * n,
                    "2min": 2 * min(m, nn)}[region]
            if R.sum() >= need:
                mc = C[R].max()
                if mc >= g:
                    idx = int(np.flatnonzero(((C >= mc) & R).ravel()).max())
                    out = (mc, (idx // nn + 1, idx % nn + 1))
    return out if return_scale else out[0]


def mgc_global(Dx, Dy):
    A, B = col_centre(Dx), col_centre(Dy)
    return float(np.sum(A * B.T) / np.sqrt(np.sum(A * A.T) * np.sum(B * B.T)))


def dcor2_gauss(r):
    num = r * np.arcsin(r) + np.sqrt(1 - r * r) - r * np.arcsin(r / 2) - np.sqrt(4 - r * r) + 1
    return float(num / (1 + np.pi / 3 - np.sqrt(3)))


# -------------------------------------------------------------------- matrices
def pair_matrix(Z, fn, dmode, undirected):
    P = Z.shape[0]
    D = [dist(z, dmode) for z in Z]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            if undirected and i > j:
                continue
            A[i, j] = fn(D[i], D[j])
            if undirected:
                A[j, i] = A[i, j]
    return A


def stat_fn(unit, variant="impl"):
    if unit == "distance.hhg":
        return hhg_from_dist
    return lambda dx, dy: mgc_from_dist(dx, dy, variant)


def unit_matrix(unit, Z, dmode, variant="impl", ordered=False):
    return pair_matrix(Z, stat_fn(unit, variant), dmode, undirected=(unit == "distance.mgc" and not ordered))


# ----------------------------------------------------------------- Monte Carlo
_MC = None


def _mc_cache():
    global _MC
    if _MC is None:
        f = EXCH / "mc_cache.json"
        _MC = json.loads(f.read_text()) if f.is_file() else {}
    return _MC


def _mc_save():
    EXCH.mkdir(parents=True, exist_ok=True)
    (EXCH / "mc_cache.json").write_text(json.dumps(_MC, indent=1, sort_keys=True))


def mc(unit, T, rho, reps):
    """Monte Carlo of the from-scratch statistic on z-scored Gaussian pairs (rho, T)."""
    key = f"{unit}|T{T}|rho{rho:.3f}|reps{reps}"
    cache = _mc_cache()
    if key in cache:
        return cache[key]
    rng = np.random.default_rng([MC_SEED, T, int(round(rho * 1000)) + 1000, 1 if unit == "distance.hhg" else 2])
    fn = stat_fn(unit)
    vals = []
    t0 = time.time()
    for _ in range(reps):
        a = rng.standard_normal(T)
        b = rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal(T)
        z = zscore(np.vstack([a, b]))
        vals.append(fn(dist(z[0], "exact"), dist(z[1], "exact")))
    v = np.asarray(vals)
    res = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), reps=reps, frac_neg=float((v < 0).mean()),
               q_lo=float(np.quantile(v, 0.005)), q_hi=float(np.quantile(v, 0.995)), seconds=time.time() - t0)
    cache[key] = res
    _mc_save()
    return res


def mc_quantile_band(unit, T, m, reps=2000):
    """Family-wise null band [q(0.025/m), q(1 - 0.025/m)] from the EMPIRICAL quantiles of a Monte Carlo of the
    from-scratch statistic on z-scored iid N(0,1) pairs. This is the unconditional null for GAUSSIAN marginals only.
    HHG's null MEAN n(n-2)(n-3)/(n-1) is distribution-free, but its spread is not: 1-D distance ranks depend on the
    spacings, and the scratchpad probe hhg_mgc_check/p_null_marginals.py (n=64, 3000 draws per law; packed under the
    same relative path into audit/checks/probes/scratch_probes_2026-10-15.tar.xz by audit/checks/archive_probes.py)
    gives null SD 923 (Gaussian), 1022 (exponential), 1055 (uniform), 996 (Cauchy), KS Gaussian vs uniform p = 0.007. For
    non-Gaussian pairs (anm, nonlinear_dependence) the band is therefore an approximation, which the conditional
    per-pair permutation null (ref_perm) does not need. The empirical quantiles replace a mean +- z SD band, which is
    wrong for HHG's right-skewed null. Values cached in out/exchange/hhg_mgc/null_<unit>_T<T>_<reps>.npy."""
    f = EXCH / f"null_{unit}_T{T}_{reps}.npy"
    if f.is_file():
        v = np.load(f)
    else:
        rng = np.random.default_rng([MC_SEED, T, 7, 1 if unit == "distance.hhg" else 2])
        fn = stat_fn(unit)
        v = np.empty(reps)
        for r in range(reps):
            z = zscore(rng.standard_normal((2, T)))
            v[r] = fn(dist(z[0], "exact"), dist(z[1], "exact"))
        EXCH.mkdir(parents=True, exist_ok=True)
        np.save(f, v)
    lo, hi = np.quantile(v, [0.025 / m, 1 - 0.025 / m])
    return float(lo), float(hi), v


def mc_reps(unit, T):
    if unit == "distance.hhg":
        return {64: 400, 256: 300, 1024: 40}[T]
    return {64: 400, 256: 300, 1024: 150, 4096: 25}[T]


def bonf(m):
    return float(stats.norm.ppf(1 - 0.025 / m))


# -------------------------------------------------------------------- helpers
def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_REF / f"{rid}.npz", **{k: np.asarray(v) for k, v in kw.items()})


def load_pyspi(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None, None
    with np.load(p) as z:
        return np.asarray(z["value"], float), (np.asarray(z["input"], float) if "input" in z.files else None)


def asym(A):
    return float(np.nanmax(np.abs(A - A.T))) if A.ndim == 2 else float("nan")


def tol_for(unit):
    return dict(rtol=1e-12, atol=0.0) if unit == "distance.hhg" else dict(rtol=0.0, atol=1e-10)


def detail(unit, dmode, extra=""):
    f = f"{SELF}:hhg_from_dist (Heller, Heller & Gorfine 2013 Sec. 2)" if unit == "distance.hhg" else \
        f"{SELF}:mgc_from_dist (Shen, Priebe & Vogelstein 2020 eq. 6, Sec. 4.1; SciPy constants n^2, n-1, ceil(0.02 max) min)"
    d = {"exact": f"exact distances {KD}:_pdist", "sk": f"sklearn-mirrored distances sqrt({KD}:_sk_sqdist)"}[dmode]
    return f"{f}; {d}; numpy {np.__version__}, scipy {scipy.__version__}{('; ' + extra) if extra else ''}"


def pair_list(P):
    return [(i, j) for i in range(P) for j in range(i + 1, P)]


# ------------------------------------------------------------------ run kinds
def ref_def(run, unit, x, dmode, kind="independent-impl", note=""):
    Z = preprocess(x, run["path"])
    V = unit_matrix(unit, Z, dmode)
    S, xin = load_pyspi(run["run_id"])
    extra = {}
    if xin is not None and xin.shape == Z.shape:
        extra["input_max_abs"] = float(np.max(np.abs(xin - Z)))
    if S is not None:
        extra["pyspi_asym"] = asym(S)
    extra["ref_asym"] = asym(V)
    save(run["run_id"], value=V, criterion="tolerance", reference_kind=kind,
         reference_detail=detail(unit, dmode), notes=note or f"{run['path']} rows, {dmode} distances", **tol_for(unit), **extra)


def ref_hyppo_exact(run, unit, x):
    Z = preprocess(x, run["path"])
    V = unit_matrix(unit, Z, "exact", ordered=True)
    save(run["run_id"], value=V, criterion="tolerance", reference_kind="independent-impl",
         reference_detail=detail(unit, "exact"), **tol_for(unit),
         notes="pyspi side is the DELEGATE (hyppo HHG / scipy multiscale_graphcorr) fed exact |x_a - x_b| "
               "distance matrices for every ordered pair (probe hyppo_exact); isolates the distance inputs")


def ref_copies(run, unit, x):
    Z = preprocess(derive("copies", x), run["path"])
    n = Z.shape[1]
    V = unit_matrix(unit, Z, "exact", ordered=(unit == "distance.hhg"))
    a = float(n * (n - 2) * (n - 3)) if unit == "distance.hhg" else 1.0
    for i in range(3):
        for j in range(3):
            if i != j:
                V[i, j] = a
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="analytic",
         reference_detail=f"identical-ranking analytic value {'N(N-2)(N-3)' if unit == 'distance.hhg' else '1'} on rows "
                          f"(x0, x0, 2x0+1); from-scratch {detail(unit, 'exact')} on the pairs with x1",
         analytic_value=a,
         notes="HHG: for identical rankings S(i,j) = n-2 unless j is i's nearest or farthest neighbour (zero margin), "
               "so T = n(n-2)(n-3); MGC: every local correlation of x with itself is 1")


def ref_dup(run, unit, x):
    n = x.shape[1]
    a = float(n * (n - 2) * (n - 3)) if unit == "distance.hhg" else 1.0
    V = np.array([[np.nan, a], [a, np.nan]])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="analytic",
         reference_detail="shared_envelope x2 is bit-identical to x0: HHG n(n-2)(n-3) / MGC 1",
         analytic_value=a)


def ref_dir(run, unit, x, ordered=False):
    Z = preprocess(x, run["path"])
    V = unit_matrix(unit, Z, "exact", ordered=ordered)
    S, _ = load_pyspi(run["run_id"])
    extra = dict(ref_asym=asym(V))
    if S is not None:
        extra["pyspi_asym"] = asym(S)
        extra["pyspi_asym_rel"] = float(np.nanmax(np.abs(S - S.T) / np.abs(S)))
    save(run["run_id"], value=V, criterion="tolerance", reference_kind="independent-impl",
         reference_detail=detail(unit, "exact"), **tol_for(unit), **extra,
         notes=("Directed.multivariate (no Undirected mirror) via probe 'ordered'; " if ordered else "") +
               "value[i, j] = statistic(x = process i, y = process j)" +
               ("" if unit == "distance.hhg" or ordered else " for i < j, mirrored"))


def null_band(unit, T, rho=0.0):
    r = mc(unit, T, rho, mc_reps(unit, T))
    mean = hhg_null_mean(T) if (unit == "distance.hhg" and rho == 0.0) else r["mean"]
    return mean, r["sd"], r


def ref_null(run, unit, x):
    T, P = x.shape[1], x.shape[0]
    mean, sd, r = null_band(unit, T)
    m = P * (P - 1) // 2
    z = bonf(m)
    V = np.full((P, P), mean); np.fill_diagonal(V, np.nan)
    SE = np.full((P, P), sd * z / 2)
    S, _ = load_pyspi(run["run_id"])
    extra = dict(mc_mean=r["mean"], mc_sd=sd, mc_reps=r["reps"], mc_frac_neg=r["frac_neg"], bonferroni_z=z)
    if unit == "distance.hhg":
        extra.update(analytic_null_mean=hhg_null_mean(T), mc_mean_minus_analytic_in_se=(r["mean"] - hhg_null_mean(T)) / (sd / np.sqrt(r["reps"])))
    if S is not None:
        off = S[~np.isnan(S)]
        extra.update(pyspi_mean=float(off.mean()), pyspi_min=float(off.min()), pyspi_max=float(off.max()),
                     pyspi_frac_neg=float((off < 0).mean()), pyspi_asym=asym(S))
        if unit == "distance.hhg":
            extra["pyspi_mean_over_analytic"] = float(off.mean() / hhg_null_mean(T))
    kind = "analytic" if unit == "distance.hhg" else "independent-impl"
    save(run["run_id"], value=V, criterion="statistical", se=SE, reference_kind=kind,
         reference_detail=(f"E[T] = n(n-2)(n-3)/(n-1) (derived in {SELF} docstring); SD from Monte Carlo of {SELF}:hhg_from_dist"
                           if unit == "distance.hhg" else f"Monte Carlo mean/SD of {SELF}:mgc_from_dist on iid N(0,1) pairs")
         + f"; se = SD z_(1-0.025/{m})/2", **extra)


def ref_order(run, unit, x):
    T, P = x.shape[1], x.shape[0]
    S, _ = load_pyspi(run["run_id"])
    m = P * (P - 1) // 2
    lo, hi, nullv = mc_quantile_band(unit, T, m)
    ds = run["dataset"]
    dep = [(0, 1), (2, 3)]
    ok = True
    for (i, j) in pair_list(P):
        for (a, b) in ((i, j), (j, i)):
            v = S[a, b]
            if (i, j) in dep:
                ok &= bool(v > hi)
            else:
                ok &= bool(lo <= v <= hi)
    extra = {}
    if ds == "bivariate_gaussian":
        ok &= bool(S[2, 3] > S[0, 1] and S[3, 2] > S[1, 0])
    for (i, j) in pair_list(P):
        extra[f"pyspi_{i}{j}"] = float(S[i, j])
        extra[f"null_upper_tail_{i}{j}"] = float((nullv >= S[i, j]).mean())
    save(run["run_id"], value=np.array(0.0), criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"null band = empirical quantiles q(0.025/{m}), q(1-0.025/{m}) of {nullv.size} Monte Carlo draws of "
                          f"{SELF}:{'hhg_from_dist' if unit == 'distance.hhg' else 'mgc_from_dist'} on z-scored iid N(0,1) pairs "
                          f"(mc_quantile_band)", band_lo=lo, band_hi=hi, null_reps=nullv.size,
         notes=f"pass = pairs (0,1),(2,3) of {ds} above the band (both orders), other pairs inside"
               + ("; and the rho=0.8 pair (2,3) above the rho=0.3 pair (0,1)" if ds == "bivariate_gaussian" else ""),
         **extra)


PERM_REPS = 200


def perm_null(unit, x, i, j, T, reps=PERM_REPS):
    """Conditional permutation null of the from-scratch statistic for the pair (row i, row j) of x: y = row j is
    permuted (indices) `reps` times, distances are exact. Cached in out/exchange/hhg_mgc/perm_<unit>_<tag>.npy."""
    tag = f"{hashlib.sha256(np.ascontiguousarray(x[[i, j]]).tobytes()).hexdigest()[:12]}_{i}{j}_T{T}_{reps}"
    f = EXCH / f"perm_{unit}_{tag}.npy"
    if f.is_file():
        return np.load(f)
    rng = np.random.default_rng([MC_SEED, T, 11, i, j, 1 if unit == "distance.hhg" else 2])
    fn = stat_fn(unit)
    dx, dy = dist(x[i], "exact"), dist(x[j], "exact")
    v = np.empty(reps)
    for r in range(reps):
        p = rng.permutation(T)
        v[r] = fn(dx, dy[np.ix_(p, p)])
    EXCH.mkdir(parents=True, exist_ok=True)
    np.save(f, v)
    return v


def ref_perm(run, unit, x):
    """Per-pair 200-permutation null (trace proposed_checks: distance.hhg [4], distance.mgc [5]). Kernel rows,
    exact distances. For each unordered pair: mid-rank percentile u of pyspi's A[i,j] in its own permutation null and
    the one-sided upper-tail permutation p = (1 + #{null >= obs}) / (reps + 1).
    iid_gaussian (independent, exchangeable): pass = KS test of the m percentiles against U(0,1) has p >= 0.05 AND no
      pair has upper-tail p < 0.05/m (Bonferroni). HHG also reports the pooled permutation-null mean against the
      analytic E[T] = n(n-2)(n-3)/(n-1), which is exact conditionally on the data for continuous rows (the x margins
      are fixed with exactly two zero-margin j per i, the y margins are uniformly random under the permutation).
    gauss_ar1_pair (contemporaneously dependent, Sigma_X off-diagonal != 0): pass = upper-tail p < 0.025. Permutation
      breaks the AR(1) autocorrelation, so this is a detection check only, not a calibrated test for serially
      dependent data."""
    T, P = x.shape[1], x.shape[0]
    S, _ = load_pyspi(run["run_id"])
    pairs = pair_list(P)
    m = len(pairs)
    us, ps, extra = [], [], {}
    allnull = []
    for (i, j) in pairs:
        v = perm_null(unit, x, i, j, T)
        allnull.append(v)
        obs = float(S[i, j])
        u = float(((v < obs).sum() + 0.5 * (v == obs).sum()) / v.size)
        pu = float((1 + (v >= obs).sum()) / (v.size + 1))
        us.append(u); ps.append(pu)
        extra[f"pct_{i}{j}"] = u
        extra[f"p_upper_{i}{j}"] = pu
        extra[f"pyspi_{i}{j}"] = obs
        extra[f"perm_mean_{i}{j}"] = float(v.mean())
        if unit == "distance.mgc":
            extra[f"perm_frac_neg_{i}{j}"] = float((v < 0).mean())
    allnull = np.concatenate(allnull)
    extra.update(perm_reps=PERM_REPS, n_pairs=m, min_p_upper=float(min(ps)), pct_mean=float(np.mean(us)), perm_null_mean=float(allnull.mean()),
                 perm_null_sd=float(allnull.std(ddof=1)))
    if unit == "distance.hhg":
        E = hhg_null_mean(T)
        extra.update(analytic_null_mean=E,
                     perm_mean_minus_analytic_in_se=float((allnull.mean() - E) / (allnull.std(ddof=1) / np.sqrt(allnull.size))))
    else:
        offs = np.array([S[i, j] for (i, j) in pairs])
        extra.update(perm_frac_neg=float((allnull < 0).mean()), pyspi_frac_neg=float((offs < 0).mean()))
    if run["dataset"] == "gauss_ar1_pair":
        ok = all(p < 0.025 for p in ps)
        crit = "pass = upper-tail permutation p < 0.025 (dependent pair; detection only, permutation ignores autocorrelation)"
    else:
        ks = stats.kstest(us, "uniform")
        extra.update(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        ok = bool(ks.pvalue >= 0.05 and min(ps) >= 0.05 / m)
        crit = f"pass = KS of the {m} percentiles vs U(0,1) p >= 0.05 and every upper-tail p >= 0.05/{m}"
    fname = "hhg_from_dist" if unit == "distance.hhg" else "mgc_from_dist"
    save(run["run_id"], value=np.array(0.0), criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"{SELF}:ref_perm, {PERM_REPS} permutations of y per pair of {SELF}:{fname} on exact distances "
                          f"({KD}:_pdist); scipy.stats.kstest {scipy.__version__}",
         notes=crit, **extra)


def ref_scaleinv(run, unit, x):
    xb = make_data.load("bivariate_gaussian", run["T"])["x"]
    Z = zscore(xb)
    V = unit_matrix(unit, Z, "exact")
    save(run["run_id"], value=V, criterion="tolerance", reference_kind="independent-impl",
         reference_detail=detail(unit, "exact", "on the z-scored bivariate_gaussian draw scaled_offset was built from"),
         **tol_for(unit))


def ref_trendqual(run, unit, x):
    T = run["T"]
    S_det, _ = load_pyspi(run["run_id"])
    rid_plain = run["run_id"].replace("|pipeline-detrend|trendqual", "|pipeline|trend")
    S_plain, _ = load_pyspi(rid_plain)
    m = 6
    lo, hi, _ = mc_quantile_band(unit, T, m)
    ok = True
    for (i, j) in ((1, 2), (1, 3)):
        ok &= bool(S_plain[i, j] > hi and S_plain[j, i] > hi)
        ok &= bool(lo <= S_det[i, j] <= hi and lo <= S_det[j, i] <= hi)
    extra = {f"plain_{i}{j}": float(S_plain[i, j]) for (i, j) in pair_list(4)}
    extra.update({f"detrended_{i}{j}": float(S_det[i, j]) for (i, j) in pair_list(4)})
    save(run["run_id"], value=np.array(0.0), criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"null band = empirical quantiles q(0.025/6), q(1-0.025/6) of {SELF}:mc_quantile_band", band_lo=lo, band_hi=hi,
         notes=f"pass = pairs (1,2),(1,3) (independent in bivariate_gaussian, both trending) above the band in {rid_plain} and inside it after detrend",
         **extra)


def ref_global(run, unit, x):
    Z = preprocess(x, run["path"])
    V = pair_matrix(Z, mgc_global, "exact", undirected=True)
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:mgc_global: sum A_ab B_ba / sqrt(sum A_ab A_ba sum B_ab B_ba), column-centred A, B (Shen 2020 eq. 6), exact distances",
         notes="the global-scale value dCorr^{n,n}; equals the statistic whenever SciPy's optimal scale is (n, n)")


def ref_udcorr(run, unit, x):
    Z = preprocess(x, run["path"])
    n = Z.shape[1]
    V = pair_matrix(Z, lambda a, b: dcorr_from_dist(a, b, biased=False)[0], "exact", undirected=True)
    G = pair_matrix(Z, mgc_global, "exact", undirected=True)
    save(run["run_id"], value=V, criterion="tolerance", rtol=0.0, atol=1.0 / n, reference_kind="independent-impl",
         reference_detail=f"{KD}:dcorr_from_dist(biased=False) (Szekely & Rizzo 2014 U-centred dCor^2), exact distances; "
                          "tolerance 1/n from Shen 2020 Sec. 3.2 ('dCorr^{n,n} ~ unbiased dCorr up to O(1/n)')",
         global_minus_udcorr_max_abs=float(np.nanmax(np.abs(G - V))),
         notes="RELATED quantity: U-centring vs MGC's column centring; the exact identity is the 'global' run")


def ref_optscale(run, unit, x):
    Z = preprocess(x, run["path"])
    P = Z.shape[0]
    D = [dist(z, "sk") for z in Z]
    V = np.full((3, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                _, (k, l) = mgc_from_dist(D[i], D[j], return_scale=True)
                V[0, i, j], V[1, i, j] = k, l
                V[2, i, j] = mgc_global(D[i], D[j])
    S, _ = load_pyspi(run["run_id"])
    extra = {}
    if S is not None:
        glob = (S[0] == T_of(Z)) & (S[1] == T_of(Z))
        extra["pyspi_pairs_global"] = int(np.sum(glob))
        extra["pyspi_pairs_total"] = int(P * (P - 1))
        loc = [f"({i},{j}):({int(S[0, i, j])},{int(S[1, i, j])})" for i in range(P) for j in range(P) if i != j and not glob[i, j]]
        extra["pyspi_local_scales"] = "; ".join(loc) if loc else "none"
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:mgc_from_dist(return_scale=True) and mgc_global on sklearn-mirrored distances; "
                          "layer 0/1 = optimal scale (k, l) 1-based, layer 2 = global map value", **extra)


def T_of(Z):
    return Z.shape[1]


def ref_variant(run, unit, x, variant):
    Z = preprocess(x, run["path"])
    V = unit_matrix(unit, Z, "sk", variant=variant)
    Vi = unit_matrix(unit, Z, "sk", variant="impl")
    atol = 2e-3 if variant == "paper" else 1e-10
    const = {"paper": "c = n(n-1), tau with s = n, region >= 2n (Shen 2020 Sec. 3.1, Corollary 3, Sec. 4.1)",
             "rmgc": "c = n^2, s = n-1, region >= 2 min(m, n') (neurodata/r-mgc MGCLocalCorr.R:167, MGCSampleStat.R:99,172)"}[variant]
    save(run["run_id"], value=V, criterion="tolerance", rtol=0.0, atol=atol, reference_kind="independent-impl",
         reference_detail=f"{SELF}:mgc_from_dist(variant='{variant}'): {const}; sklearn-mirrored distances",
         variant_minus_impl_max_abs=float(np.nanmax(np.abs(V - Vi))),
         notes=f"quantifies pyspi (SciPy constants) against the {variant} constants; a fail is the size of that deviation")


def ref_pop(run, unit, x):
    T = x.shape[1]
    procs = run.get("processes") or [0, 1, 2, 3]
    rhos = {(0, 1): 0.3, (2, 3): 0.8}
    P = len(procs)
    V = np.full((P, P), np.nan)
    SE = np.full((P, P), np.nan)
    m = P * (P - 1) // 2
    z = bonf(m)
    extra = {}
    for a in range(P):
        for b in range(P):
            if a == b:
                continue
            key = tuple(sorted((procs[a], procs[b])))
            rho = rhos.get(key, 0.0)
            r = mc(unit, T, rho, mc_reps(unit, T))
            V[a, b] = dcor2_gauss(rho) if rho > 0 else r["mean"]
            SE[a, b] = r["sd"] * z / 2
            if rho > 0:
                extra[f"pop_dcor2_rho{rho}"] = dcor2_gauss(rho)
                extra[f"mc_mean_rho{rho}"] = r["mean"]
                extra[f"mc_sd_rho{rho}"] = r["sd"]
    save(run["run_id"], value=V, criterion="statistical", se=SE, reference_kind="analytic",
         reference_detail=f"{SELF}:dcor2_gauss (Szekely, Rizzo & Bakirov 2007 Thm 7) for the rho pairs, Monte Carlo null mean "
                          f"for cross pairs; SD from Monte Carlo of {SELF}:mgc_from_dist at T={T}; se = SD z_(1-0.025/{m})/2",
         **extra)


def ref_smalln(run, unit, x):
    S, _ = load_pyspi(run["run_id"])
    ok = S is not None and S[0] == 1.0 and np.isfinite(S[1])
    xs = np.asarray(x[:2, :5], float)
    v5 = mgc_from_dist(dist(xs[0], "sk"), dist(xs[1], "sk"))
    save(run["run_id"], value=np.array(0.0), criterion="qualitative", pass_flag=int(bool(ok)), reference_kind="analytic",
         reference_detail="scipy/stats/_mgc.py:331-333 documents ValueError for n < 5; pass = raised at n = 4 and finite at n = 5",
         ref_value_n5=float(v5), pyspi_value_n5=float(S[1]) if S is not None else float("nan"))


# ------------------------------------------------------------------ dispatch
def dispatch(run):
    rid = run["run_id"]
    unit = run["unit_id"]
    tag = rid.split("|")[4].split("-p")[0]
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], float)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    if tag == "def":
        return ref_def(run, unit, x, "exact")
    if tag == "skdist":
        return ref_def(run, unit, x, "sk", note="kernel rows, sklearn-mirrored distances (estimator isolated from the distance arithmetic)")
    if tag == "exact":
        return ref_def(run, unit, x, "exact", note="TIED data vs the exact-distance definition (sklearn-rounding candidate)")
    if tag == "hyppo-exact":
        return ref_hyppo_exact(run, unit, x)
    if tag == "copies":
        return ref_copies(run, unit, x)
    if tag == "dup":
        return ref_dup(run, unit, x)
    if tag == "dir":
        return ref_dir(run, unit, x)
    if tag == "ordered":
        return ref_dir(run, unit, x, ordered=True)
    if tag == "null":
        return ref_null(run, unit, x)
    if tag == "order":
        return ref_order(run, unit, x)
    if tag == "perm":
        return ref_perm(run, unit, x)
    if tag == "scaleinv":
        return ref_scaleinv(run, unit, x)
    if tag in ("trend", "detrend"):
        return ref_def(run, unit, x, "exact", note=f"trending rows through {run['path']} (own preprocessing)")
    if tag == "trendqual":
        return ref_trendqual(run, unit, x)
    if tag == "global":
        return ref_global(run, unit, x)
    if tag == "udcorr":
        return ref_udcorr(run, unit, x)
    if tag == "optscale":
        return ref_optscale(run, unit, x)
    if tag in ("paper", "rmgc"):
        return ref_variant(run, unit, x, tag)
    if tag == "pop":
        return ref_pop(run, unit, x)
    if tag == "smalln":
        return ref_smalln(run, unit, x)
    raise ValueError(f"unknown tag {tag!r} in {rid}")


def selftest():
    rng = np.random.default_rng(3)
    ok = True
    for n in (12, 40):
        a, b = rng.standard_normal(n), rng.standard_normal(n)
        b2 = a ** 2 + 0.1 * b
        for y in (b, b2):
            k = kernel_hhg(a, y)
            f = hhg_from_dist(_pdist(a), _pdist(y))
            good = abs(k - f) <= 1e-12 * abs(k)
            ok &= good
            print(f"hhg n={n}: kernel {k!r} fast {f!r} {'ok' if good else 'MISMATCH'}")
        # ties: rounded data
        ar, yr = np.round(a, 1), np.round(b2, 1)
        k, f = kernel_hhg(ar, yr), hhg_from_dist(_pdist(ar), _pdist(yr))
        ok &= abs(k - f) <= 1e-12 * abs(k)
        print(f"hhg ties n={n}: kernel {k!r} fast {f!r}")
        g1 = mgc_global(_pdist(a), _pdist(b))
        C = local_corr(_pdist(a), _pdist(b))
        ok &= abs(C[-1, -1] - g1) < 1e-12
        print(f"mgc global n={n}: map corner {C[-1, -1]!r} analytic {g1!r}")
    print("population dCor^2: rho 0.3", dcor2_gauss(0.3), "rho 0.8", dcor2_gauss(0.8), "rho 1", dcor2_gauss(1.0))
    for n in (64, 256):
        print("HHG analytic null mean", n, hhg_null_mean(n))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main(argv):
    if "--selftest" in argv:
        sys.exit(0 if selftest() else 1)
    runs = json.loads(MANIFEST.read_text())["runs"]
    only = [a for a in argv if not a.startswith("--")]
    n_ok = n_err = 0
    for run in runs:
        if only and not any(o in run["run_id"] for o in only):
            continue
        t0 = time.time()
        try:
            dispatch(run)
            n_ok += 1
            print(f"ref ok {run['run_id']} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as exc:
            import traceback
            n_err += 1
            print(f"ref ERROR {run['run_id']}: {exc!r}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    print(f"hhg_mgc refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
