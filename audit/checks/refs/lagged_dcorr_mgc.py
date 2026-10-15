"""lagged_dcorr_mgc family references (COMPARISON venv): distance.dcorrx.maxlag-{1,10}, distance.mgcx.maxlag-{1,10}.

pyspi stores A[i, j] = DcorrX/MGCX(max_lag=L).statistic(x=z[i], y=z[j]) (pyspi/statistics/distance.py:126-131,
146-151 at HEAD a42ebc1). hyppo 0.5.2 computes (time_series/_utils.py:85-108):
    term_0 = Dcorr/MGC of the raw series (line 93; weight 1),
    term_l = (n-l)/n * Dcorr/MGC(compute_distance="euclidean").statistic(D_x[l:n, l:n], D_y[0:n-l, 0:n-l])  (lines 98-102),
    stat   = sum_l term_l (line 106).
Because the test object keeps compute_distance="euclidean" (line 92), Dcorr.statistic / MGC.statistic re-apply
Euclidean pairwise distances to the ROWS of the sliced distance matrices (independence/dcorr.py:144-147,
mgc.py:154-157): a distance of distances. hyppo's own docstring (time_series/dcorrx.py:69-72) states the
series-slice formula sum_l (n-l)/n Dcorr_n(x[l:n], y[0:n-l]) (Mehta et al. 2019; Shen et al. 2024 Alg. Step 3).

Nothing here imports hyppo, sklearn, or scipy.stats.multiscale_graphcorr except the 'scipy' runs, which are
labelled same-lineage. Building blocks:
  * _kernels_distance.hyppo_dcorrx_replica (as implemented, sklearn-mirrored arithmetic) and
    _kernels_distance.dcorrx_mehta (canonical: U-centred dCor^2 of the lagged SERIES slices).
  * mgc_local_corr / mgc_stat (this file): MGC from scratch after Shen, Priebe & Vogelstein 2020 (JASA 115:280,
    Sec. 3.1: column centring with 1/(n-1), zero diagonal, column-wise rank truncation, local covariance minus
    the product of truncated sums / n^2) and Vogelstein et al. 2019 (eLife 8:e41690, Alg. 1: beta-approximation
    threshold, largest connected component, smoothed maximum), with SciPy's constants (sample size n-1 in the
    threshold, 1 - 0.02/s percentile, region >= ceil(0.02 max(k, l)) min(k, l)); minimal-dense column ranks
    (the rank convention that reproduces SciPy's map shape on tied data, see the mgcmap runs).
    numpy + scipy.ndimage.label + scipy.stats.beta.ppf only.

Run tags (last field of the run_id; '-pXYZ' = process subset):
  replica    pyspi vs the as-implemented replica (distance of distances), tolerance atol = rtol = 1e-10
  mehta      pyspi vs the canonical series-slice statistic: DESIGNED TO FAIL; the size of the gap is the finding
  lagterms   probe: hyppo's own per-lag weighted terms + statistic vs the replica's; extra keys quantify the
             per-lag gap to the Mehta terms and the true-delay terms on delayed_noise
  dir        probe A - A.T vs the generating truth (direction); extra keys: Mehta's A - A.T on the same pairs
  null       statistical: Bonferroni band over the m ordered pairs from the empirical Monte Carlo quantiles of the
             replica on iid N(0,1) pairs at the same T (the null is right-skewed; T=1024 borrows the standardised
             T=256 shape and rescales it with the T=1024 Monte Carlo mean and SD)
  neg        qualitative: negative values present on iid data (refutes the 'unsigned' label)
  sum        qualitative: max > 1 and |A| <= sum_l (n-l)/n (weighted SUM, not an average)
  scaleinv   scaled_offset through the pipeline vs the replica on the z-scored bivariate_gaussian draw
  smalln     qualitative: failure threshold T >= L+3 (dcorrx, ZeroDivisionError) / T >= L+5 (mgcx, ValueError)
  multireal  R = 2 realisations: replica with 2-D samples (each time point an R-vector)
  scipy      mgcx only, SAME-LINEAGE: scipy.stats.multiscale_graphcorr on the same distance-of-distance inputs
  lag0       mgcx only: MGCX(max_lag=0) per ordered pair and the pyspi mgc SPI vs from-scratch MGC
  mgcmap     mgcx only: SciPy's compiled local-correlation map vs mgc_local_corr
  calls      mgcx only, qualitative: L+3 multiscale_graphcorr calls per pair (L+1 needed), same value, RNG untouched

Writes out/ref/<run_id>.npz per the compare.py contract. Monte Carlo samples are cached in
out/exchange/lagged_dcorr_mgc/mc_*.npy (delete them to recompute).
"""
import json
import math
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np
import scipy
from scipy import ndimage, signal, stats

import make_data
import _kernels_distance as KD

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / "lagged_dcorr_mgc"
MANIFEST = CHECKS / "manifest" / "lagged_dcorr_mgc.json"
SELF = "audit/checks/refs/lagged_dcorr_mgc.py"
KDP = "audit/checks/refs/_kernels_distance.py"
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"
MC_SEED = 20261015
TOL = dict(rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------------ preprocessing
def zscore(X):
    X = np.asarray(X, float)
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)


def preprocess(X, path):
    if path == "kernel":
        return np.asarray(X, float)
    if path == "pipeline":
        return zscore(X)
    if path == "pipeline-detrend":
        return zscore(signal.detrend(np.asarray(X, float), axis=1))
    raise ValueError(path)


def sk_dist(z):
    """Euclidean distances with sklearn's float64 arithmetic (mirrored in _kernels_distance._sk_sqdist)."""
    return np.sqrt(KD._sk_sqdist(z))


# --------------------------------------------------------------------------- MGC
def dense_col_rank(D):
    """0-based minimal dense rank of D[i, j] among column j (ties share a rank, no gaps)."""
    order = np.argsort(D, axis=0, kind="stable")
    srt = np.take_along_axis(D, order, axis=0)
    step = np.zeros(D.shape, dtype=np.int64)
    step[1:] = (np.diff(srt, axis=0) > 0)
    R = np.empty(D.shape, dtype=np.int64)
    np.put_along_axis(R, order, np.cumsum(step, axis=0), axis=0)
    return R


def mgc_local_corr(Dx, Dy):
    """Local correlation map C[k, l] (Shen, Priebe & Vogelstein 2020, Sec. 3.1).
    A_ij = Dx_ij - sum_a Dx_aj / (n-1), A_ii = 0 (likewise B). Truncation: A_ij enters scale k when
    rank_col(Dx)_ij <= k, B_ji enters scale l when rank_col(Dy)_ji <= l. Local cov = sum A_ij B_ji 1{.} -
    (sum_{rank<=k} A)(sum_{rank<=l} B) / n^2; local var likewise on (A, A^T); corr = cov / sqrt(var var),
    clipped above at 1, 0 where a variance is <= 0."""
    n = Dx.shape[0]
    A = Dx - Dx.sum(axis=0, keepdims=True) / (n - 1)
    B = Dy - Dy.sum(axis=0, keepdims=True) / (n - 1)
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(B, 0.0)
    RA, RB = dense_col_rank(Dx), dense_col_rank(Dy)
    kx, ky = int(RA.max()) + 1, int(RB.max()) + 1

    def lcov(a, ra, b, rb, k, l):
        H = np.bincount((ra * l + rb.T).ravel(), weights=(a * b.T).ravel(), minlength=k * l).reshape(k, l)
        H = H.cumsum(axis=0).cumsum(axis=1)
        sa = np.bincount(ra.ravel(), weights=a.ravel(), minlength=k).cumsum()
        sb = np.bincount(rb.ravel(), weights=b.ravel(), minlength=l).cumsum()
        return H - np.outer(sa, sb) / n ** 2

    cov = lcov(A, RA, B, RB, kx, ky)
    vx = np.diag(lcov(A, RA, A, RA, kx, kx))
    vy = np.diag(lcov(B, RB, B, RB, ky, ky))
    with np.errstate(all="ignore"):
        C = cov / np.sqrt(np.outer(vx, vy))
    C[C > 1] = 1.0
    C[vx <= 0, :] = 0.0
    C[:, vy <= 0] = 0.0
    return C


def mgc_stat(Dx, Dy):
    """Sample MGC (Vogelstein et al. 2019 Alg. 1 with SciPy's constants): threshold
    tau = max(2 F^-1_Beta(1 - 0.02/s; s(s-3)/4 - 1/2, same) - 1, global), s = n - 1; largest 4-connected
    component of {C > tau}; if its area >= ceil(0.02 max(k, l)) min(k, l) and its maximum >= the global
    value C[k-1, l-1], return that maximum, else the global value."""
    n = Dx.shape[0]
    C = mgc_local_corr(Dx, Dy)
    k, l = C.shape
    g = float(C[k - 1, l - 1])
    if k == 1 or l == 1:
        return g
    s = n - 1
    sh = s * (s - 3) / 4 - 0.5
    tau = max(2.0 * stats.beta.ppf(1 - 0.02 / s, sh, sh) - 1.0, g)
    sig = C > tau
    if not sig.any():
        return g
    lab, _ = ndimage.label(sig)
    cnt = np.bincount(lab.ravel())
    R = lab == (int(np.argmax(cnt[1:])) + 1)
    if R.sum() >= math.ceil(0.02 * max(k, l)) * min(k, l):
        mc = float(C[R].max())
        if mc >= g:
            return mc
    return g


def scipy_mgc(Dx, Dy):
    from scipy.stats import multiscale_graphcorr
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return float(multiscale_graphcorr(Dx, Dy, compute_distance=None, reps=0).statistic)


# ---------------------------------------------------------------- lagged statistics
def mgcx_terms(a, b, L, stat=mgc_stat, dist_of_dist=True):
    """Weighted per-lag MGC terms. dist_of_dist=True: hyppo's construction (lag l >= 1 on Euclidean distances
    between rows of the sliced distance matrices); False: canonical (the slices used as distance matrices)."""
    dx, dy = sk_dist(a), sk_dist(b)
    n = dx.shape[0]
    terms = [stat(dx, dy)]
    for j in range(1, L + 1):
        sx, sy = dx[j:n, j:n], dy[0:n - j, 0:n - j]
        if dist_of_dist:
            sx, sy = sk_dist(sx), sk_dist(sy)
        terms.append((n - j) * stat(sx, sy) / n)
    return terms


def dcorrx_terms(a, b, L, canonical=False):
    if canonical:
        return KD.dcorrx_mehta(a, b, L)[1]
    return KD.hyppo_dcorrx_replica(a, b, L)[1]


def terms_fn(kind, L, variant):
    """variant: replica | mehta | scipy."""
    if kind == "dcorrx":
        return lambda a, b: dcorrx_terms(a, b, L, canonical=(variant == "mehta"))
    if variant == "scipy":
        return lambda a, b: mgcx_terms(a, b, L, stat=scipy_mgc)
    return lambda a, b: mgcx_terms(a, b, L, dist_of_dist=(variant != "mehta"))


_MEMO = {}


def pair_terms(kind, L, variant, Z, key):
    """(L+1, P, P) weighted per-lag terms for every ordered pair (i, j): x = Z[i], y = Z[j]."""
    mk = (kind, L, variant, key)
    if mk in _MEMO:
        return _MEMO[mk]
    P = Z.shape[0]
    fn = terms_fn(kind, L, variant)
    out = np.full((L + 1, P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                out[:, i, j] = fn(Z[i], Z[j])
    _MEMO[mk] = out
    return out


def total(terms):
    return terms.sum(axis=0)


# ------------------------------------------------------------------------ helpers
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
    return float(np.nanmax(np.abs(A - A.T)))


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def unit_kind(unit):
    kind = "dcorrx" if ".dcorrx." in unit else "mgcx"
    return kind, int(unit.rsplit("-", 1)[1])


def replica_kind(kind):
    """dcorrx replica (_kernels_distance.hyppo_dcorrx_replica) mirrors hyppo's compute_stat AND hyppo's _dcorr
    arithmetic line by line (its docstring says 'mirrored'), so it is labelled same-lineage. The mgcx replica keeps
    the mirrored lag slicing but computes MGC from scratch (mgc_stat, Shen, Priebe & Vogelstein 2020), and that
    part is tested against SciPy's compiled map, so it stays independent-impl."""
    return "same-lineage" if kind == "dcorrx" else "independent-impl"


def detail(kind, variant, extra=""):
    if kind == "dcorrx":
        base = {"replica": f"{KDP}:hyppo_dcorrx_replica (hyppo/time_series/_utils.py:85-108 mirrored: lag-0 U-dCor^2 of the series, "
                           "lag l>=1 U-dCor^2 of Euclidean distances between rows of D_x[l:,l:], D_y[:n-l,:n-l], weights (n-l)/n, sum; sklearn arithmetic)",
                "mehta": f"{KDP}:dcorrx_mehta (Mehta et al. 2019 arXiv:1907.02088 / Shen et al. 2024 TMLR Alg. Step 3: "
                         "sum_l (n-l)/n U-dCor^2(x[l:], y[:n-l]) on the series slices)"}[variant]
    else:
        base = {"replica": f"{SELF}:mgcx_terms(dist_of_dist=True) with {SELF}:mgc_stat (from-scratch MGC, Shen, Priebe & Vogelstein 2020 Sec. 3.1, "
                           "Vogelstein et al. 2019 Alg. 1, SciPy constants) on hyppo's distance-of-distance lag inputs (sklearn arithmetic)",
                "mehta": f"{SELF}:mgcx_terms(dist_of_dist=False) with {SELF}:mgc_stat: Shen et al. 2024 TMLR Alg. Step 3 with MGC on the "
                         "lagged series slices (D_x[l:,l:], D_y[:n-l,:n-l] used as distance matrices)",
                "scipy": f"{SELF}:mgcx_terms(stat=scipy_mgc): scipy.stats.multiscale_graphcorr(compute_distance=None, reps=0) on the "
                         "distance-of-distance lag inputs (SAME-LINEAGE: hyppo MGC delegates to this function)"}[variant]
    return f"{base}; {VERS}" + (f"; {extra}" if extra else "")


def run_data(run):
    ds = make_data.load(run["dataset"], run["T"])
    x = np.asarray(ds["x"], float)
    procs = run.get("processes")
    if procs is not None:
        x = x[list(procs)]
    return ds, x, procs


def bonf(m):
    return float(stats.norm.ppf(1 - 0.025 / m))


# -------------------------------------------------------------------- Monte Carlo
def mc_values(kind, L, T, reps):
    """Replica statistic for `reps` ordered pairs of iid N(0,1) series (z-scored, ddof=1); cached as .npy."""
    f = EXCH / f"mc_{kind}_L{L}_T{T}_r{reps}_s{MC_SEED}.npy"
    if f.is_file():
        return np.load(f)
    rng = np.random.default_rng([MC_SEED, L, T, 0 if kind == "dcorrx" else 1])
    fn = terms_fn(kind, L, "replica")
    v = np.empty(reps)
    for r in range(reps):
        z = zscore(rng.standard_normal((2, T)))
        v[r] = float(np.sum(fn(z[0], z[1])))
    EXCH.mkdir(parents=True, exist_ok=True)
    np.save(f, v)
    return v


# T <= 256: enough replicates for the empirical 0.025/m quantiles (m = 20 for P = 5, 6 for P = 3).
# T = 1024 (dcorrx only): mean and SD from a smaller Monte Carlo at 1024, SHAPE (standardised quantiles)
# from the 8000-replicate T = 256 sample (n U-dCov^2 has a fixed limit law, so the standardised shape
# stabilises; the skewness at 256 and 1024 is reported for comparison).
MC_REPS = {("dcorrx", 1): {64: 8000, 256: 8000, 1024: 200}, ("dcorrx", 10): {64: 8000, 256: 8000, 1024: 80},
           ("mgcx", 1): {64: 2400, 256: 2400}, ("mgcx", 10): {64: 2400, 256: 2400}}
SHAPE_T = 256


def null_band(kind, L, T, m):
    tail = 0.025 / m
    v = mc_values(kind, L, T, MC_REPS[(kind, L)][T])
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    shape_T = T if T <= SHAPE_T else SHAPE_T
    w = v if shape_T == T else mc_values(kind, L, shape_T, MC_REPS[(kind, L)][shape_T])
    zw = (w - w.mean()) / w.std(ddof=1)
    zlo, zhi = np.quantile(zw, [tail, 1 - tail])
    return dict(lo=mean + sd * float(zlo), hi=mean + sd * float(zhi), mean=mean, sd=sd, reps=int(v.size),
                skew=float(stats.skew(v, bias=False)), shape_T=shape_T, shape_reps=int(w.size),
                shape_skew=float(stats.skew(w, bias=False)), frac_neg=float((v < 0).mean()),
                normal_lo=mean - bonf(m) * sd, normal_hi=mean + bonf(m) * sd, n_beyond=int(((zw < zlo) | (zw > zhi)).sum()))


# ------------------------------------------------------------------- truth: direction
def expected_dir(ds, run, L, P):
    E = np.zeros((P, P))
    name = run["dataset"]

    def claim(tgt, src):          # src -> tgt: A[tgt, src] (future of tgt, past of src) larger
        E[tgt, src], E[src, tgt] = 1.0, -1.0

    if name == "delayed_noise":
        d = np.asarray(ds["delay"])
        if run.get("processes") is not None:
            d = d[list(run["processes"])]
        for a in range(P):
            for b in range(P):
                if d[a] > d[b] and d[a] - d[b] <= L:       # x_a = x_b delayed by d_a - d_b: b leads a
                    claim(a, b)
    elif name in ("var2_bivariate", "logistic_maps_chaotic"):
        claim(1, 0)
    elif name == "var2_chain":
        claim(1, 0)
        claim(2, 1)
    elif name == "gauss_ar1_pair":
        claim(0, 1)               # A = [[.5,.3],[.2,.5]]: |E x0(t+h) x1(t)| > |E x1(t+h) x0(t)| for h >= 1
    else:
        raise ValueError(name)
    return E


# ----------------------------------------------------------------------- references
def ref_replica(run, kind, L, x, ds):
    path = run["path"]
    Z = preprocess(x, path)
    S, xin = load_pyspi(run["run_id"])
    terms = pair_terms(kind, L, "replica", Z, (run["dataset"], run["T"], path, str(run.get("processes"))))
    V = total(terms)
    extra = {}
    if S is not None:
        extra["pyspi_asym"] = asym(S)
        extra["pyspi_min"] = float(np.nanmin(S))
        extra["pyspi_max"] = float(np.nanmax(S))
    if xin is not None and xin.shape == Z.shape:
        extra["own_preproc_vs_pyspi_input_maxabs"] = float(np.max(np.abs(xin - Z)))
    save(run["run_id"], value=V, criterion="tolerance", reference_kind=replica_kind(kind),
         reference_detail=detail(kind, "replica", f"rows preprocessed by {SELF}:preprocess({path})"),
         ref_asym=asym(V), notes=f"as-implemented quantity (distance of distances for lags >= 1), path {path}", **TOL, **extra)


def ref_scipy(run, kind, L, x, ds):
    Z = preprocess(x, run["path"])
    V = total(pair_terms(kind, L, "scipy", Z, (run["dataset"], run["T"], run["path"], str(run.get("processes")))))
    Vr = total(pair_terms(kind, L, "replica", Z, (run["dataset"], run["T"], run["path"], str(run.get("processes")))))
    save(run["run_id"], value=V, criterion="tolerance", reference_kind="same-lineage",
         reference_detail=detail(kind, "scipy"), scipy_vs_scratch_maxabs=float(np.nanmax(np.abs(V - Vr))),
         notes="same-lineage second opinion (scipy 1.18.1 in the comparison venv vs 1.17.1 in pyspi's venv)", **TOL)


def ref_mehta(run, kind, L, x, ds):
    path = run["path"]
    Z = preprocess(x, path)
    key = (run["dataset"], run["T"], path, str(run.get("processes")))
    Tm = pair_terms(kind, L, "mehta", Z, key)
    Tr = pair_terms(kind, L, "replica", Z, key)
    V, Vr = total(Tm), total(Tr)
    S, _ = load_pyspi(run["run_id"])
    extra = {"replica_minus_mehta_maxabs": float(np.nanmax(np.abs(Vr - V))),
             "lag0_replica_minus_mehta_maxabs": float(np.nanmax(np.abs(Tr[0] - Tm[0]))),
             "perlag_replica_minus_mehta_maxabs": [float(np.nanmax(np.abs(Tr[l] - Tm[l]))) for l in range(L + 1)],
             "mehta_min": float(np.nanmin(V)), "mehta_max": float(np.nanmax(V))}
    if S is not None:
        D = S - V
        extra["pyspi_minus_mehta_maxabs"] = float(np.nanmax(np.abs(D)))
        extra["pyspi_minus_mehta_mean"] = float(np.nanmean(D))
        extra["pyspi_minus_mehta_maxrel"] = float(np.nanmax(np.abs(D) / np.maximum(np.abs(V), 1e-12)))
        extra["pyspi_minus_mehta_min"] = float(np.nanmin(D))
        extra["pyspi_minus_mehta_max"] = float(np.nanmax(D))
        extra["frac_pyspi_above_mehta"] = float(np.nanmean(D[np.isfinite(D)] > 0))
        extra["pyspi_max"] = float(np.nanmax(S))
        ia, ib = np.unravel_index(np.nanargmax(np.abs(D)), D.shape)
        extra["maxgap_pair_target_source"] = f"({ia},{ib})"
        extra["maxgap_pyspi_terms"] = [float(Tr[l][ia, ib]) for l in range(L + 1)]
        extra["maxgap_mehta_terms"] = [float(Tm[l][ia, ib]) for l in range(L + 1)]
        iu = np.triu_indices(S.shape[0], 1)
        sp, sm = np.sign((S - S.T)[iu]), np.sign((V - V.T)[iu])
        extra["ordering_agreement"] = float(np.mean(sp == sm))
        extra["pyspi_asym"] = asym(S)
    extra["mehta_asym"] = asym(V)
    save(run["run_id"], value=V, criterion="tolerance", reference_kind="independent-impl",
         reference_detail=detail(kind, "mehta"),
         notes="DESIGNED TO FAIL: canonical lagged statistic on the series slices; the gap is the distance-of-distances deviation",
         **TOL, **extra)


def ref_lagterms(run, kind, L, x, ds):
    path = run["path"]
    Z = preprocess(x, path)
    key = (run["dataset"], run["T"], path, str(run.get("processes")))
    Tr = pair_terms(kind, L, "replica", Z, key)
    Tm = pair_terms(kind, L, "mehta", Z, key)
    V = np.concatenate([Tr, total(Tr)[None]], axis=0)
    S, _ = load_pyspi(run["run_id"])
    extra = {}
    if S is not None:
        St = S[:L + 1]
        extra["perlag_pyspi_minus_mehta_maxabs"] = [float(np.nanmax(np.abs(St[l] - Tm[l]))) for l in range(L + 1)]
        extra["perlag_mehta_maxabs"] = [float(np.nanmax(np.abs(Tm[l]))) for l in range(L + 1)]
        extra["perlag_pyspi_maxabs"] = [float(np.nanmax(np.abs(St[l]))) for l in range(L + 1)]
        extra["perlag_pyspi_minus_mehta_min"] = [float(np.nanmin(St[l] - Tm[l])) for l in range(L + 1)]
        extra["perlag_pyspi_minus_mehta_max"] = [float(np.nanmax(St[l] - Tm[l])) for l in range(L + 1)]
        Dt = S[L + 1] - total(Tm)
        ia, ib = np.unravel_index(np.nanargmax(np.abs(Dt)), Dt.shape)
        extra["total_pyspi_minus_mehta_min"] = float(np.nanmin(Dt))
        extra["total_pyspi_minus_mehta_max"] = float(np.nanmax(Dt))
        extra["maxgap_pair_target_source"] = f"({ia},{ib})"
        extra["maxgap_pyspi_terms"] = [float(St[l][ia, ib]) for l in range(L + 1)]
        extra["maxgap_mehta_terms"] = [float(Tm[l][ia, ib]) for l in range(L + 1)]
        extra["stat_minus_sum_of_terms_maxabs"] = float(np.nanmax(np.abs(S[L + 1] - St.sum(axis=0))))
        extra["stat_minus_mean_of_terms_maxabs"] = float(np.nanmax(np.abs(S[L + 1] - St.mean(axis=0))))
        if run["dataset"] == "delayed_noise":
            d = np.asarray(ds["delay"])
            pr = list(run["processes"]) if run.get("processes") is not None else list(range(len(d)))
            d = d[pr]
            src = int(np.argmin(d))
            rows = [(a, int(d[a] - d[src])) for a in range(len(pr)) if a != src and 0 < d[a] - d[src] <= L]
            if rows:
                extra["truedelay_pairs_target_source_lag"] = [f"({pr[a]},{pr[src]}) lag {lag}" for a, lag in rows]
                extra["truedelay_pyspi_term"] = [float(S[lag, a, src]) for a, lag in rows]
                extra["truedelay_mehta_term"] = [float(Tm[lag, a, src]) for a, lag in rows]
                extra["truedelay_pyspi_over_mehta"] = [float(S[lag, a, src] / Tm[lag, a, src]) for a, lag in rows]
                extra["truedelay_reverse_pyspi_term"] = [float(S[lag, src, a]) for a, lag in rows]
    save(run["run_id"], value=V, criterion="tolerance", reference_kind=replica_kind(kind),
         reference_detail=detail(kind, "replica", "stacked weighted per-lag terms l = 0..L and their sum"),
         notes="slices 0..L: weighted lag terms of hyppo compute_stat; slice L+1: the statistic; extra keys compare with the Mehta terms",
         **TOL, **extra)


def ref_dir(run, kind, L, x, ds):
    path = run["path"]
    Z = preprocess(x, path)
    P = Z.shape[0]
    key = (run["dataset"], run["T"], path, str(run.get("processes")))
    Vr = total(pair_terms(kind, L, "replica", Z, key))
    Vm = total(pair_terms(kind, L, "mehta", Z, key))
    E = expected_dir(ds, run, L, P)
    S, _ = load_pyspi(run["run_id"])
    m = E > 0
    extra = {"claims_target_source": [f"({t},{s})" for t, s in zip(*np.nonzero(m))],
             "replica_antisym_at_claims": [float(v) for v in (Vr - Vr.T)[m]],
             "mehta_antisym_at_claims": [float(v) for v in (Vm - Vm.T)[m]],
             "mehta_direction_agreement": float(np.mean(np.sign((Vm - Vm.T)[m]) > 0)) if m.any() else float("nan")}
    if S is not None:
        extra["pyspi_antisym_at_claims"] = [float(v) for v in S[m]]
        extra["pyspi_max_abs_antisym"] = float(np.nanmax(np.abs(S)))
    save(run["run_id"], value=Vr - Vr.T, criterion="direction", expected_sign=E, reference_kind="analytic",
         reference_detail=f"generating truth of {run['dataset']} (data/CONVENTIONS.md; gauss_ar1_pair: A=[[.5,.3],[.2,.5]], population lagged "
                          "Gaussian dCor^2 sums favour 1->0 by 0.060 (L=1) / 0.172 (L=10)); value = replica A - A.T for magnitude",
         notes="pyspi value is A - A.T; expected +1 at [target, source] (A[i,j] = past of j -> future of i)", **extra)


def ref_null(run, kind, L, x, ds):
    T = run["T"]
    S, _ = load_pyspi(run["run_id"])
    P = x.shape[0]
    m = P * (P - 1)
    # Bonferroni band [lo, hi] = Monte Carlo null quantiles at 0.025/m and 1 - 0.025/m (the nulls are right-skewed,
    # so neither a symmetric normal band nor a three-moment fit is adequate); pass <=> lo <= pyspi <= hi.
    nb = null_band(kind, L, T, m)
    V = np.full((P, P), 0.5 * (nb["lo"] + nb["hi"]))
    np.fill_diagonal(V, np.nan)
    se = np.full((P, P), (nb["hi"] - nb["lo"]) / 4)
    extra = {"mc_mean": nb["mean"], "mc_sd": nb["sd"], "mc_skew": nb["skew"], "mc_reps": nb["reps"],
             "mc_frac_negative": nb["frac_neg"], "band_lo": nb["lo"], "band_hi": nb["hi"], "band_tail": 0.025 / m,
             "shape_T": nb["shape_T"], "shape_reps": nb["shape_reps"], "shape_skew": nb["shape_skew"],
             "normal_band_lo": nb["normal_lo"], "normal_band_hi": nb["normal_hi"]}
    if S is not None:
        o = offdiag(S)
        extra.update(pyspi_mean=float(o.mean()), pyspi_min=float(o.min()), pyspi_max=float(o.max()),
                     pyspi_n_negative=int((o < 0).sum()), pyspi_n=int(o.size), pyspi_asym=asym(S))
    pop = ("population value 0: the U-centred numerator has mean 0 for independent exchangeable samples" if kind == "dcorrx"
           else "population dependence 0; the sample MGC (smoothed maximum) has a small positive Monte Carlo mean")
    save(run["run_id"], value=V, se=se, criterion="statistical", reference_kind="analytic",
         reference_detail=f"null: Monte Carlo of {detail(kind, 'replica')} on iid N(0,1) pairs (z-scored), seed {MC_SEED}; {pop}",
         notes="value/se encode the Bonferroni band [band_lo, band_hi] over the m ordered pairs: empirical Monte Carlo quantiles "
               "(T=1024: standardised T=256 quantiles rescaled by the T=1024 Monte Carlo mean and SD)", **extra)


def ref_neg(run, kind, L, x, ds):
    S, _ = load_pyspi(run["run_id"])
    o = offdiag(S)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int((o < 0).any()), reference_kind="analytic",
         reference_detail="sign of the unbiased / MGC estimator under independence (population 0, estimator takes both signs)",
         pyspi_min=float(o.min()), pyspi_n_negative=int((o < 0).sum()), pyspi_n=int(o.size),
         notes="pass = at least one negative off-diagonal value, i.e. the 'unsigned' label and the glossary's [0,1] range do not hold")


def ref_sum(run, kind, L, x, ds):
    S, _ = load_pyspi(run["run_id"])
    T = run["T"]
    bound = float(sum((T - j) / T for j in range(L + 1)))
    Z = preprocess(x, run["path"])
    Vr = total(pair_terms(kind, L, "replica", Z, (run["dataset"], T, run["path"], str(run.get("processes")))))
    o = offdiag(S)
    ok = bool(o.max() > 1.0 and np.all(np.abs(o) <= bound + 1e-12))
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail="bound of the implemented weighted sum: |A| <= sum_l (n-l)/n since every term is in [-1, 1]",
         pyspi_max=float(o.max()), pyspi_min=float(o.min()), bound=bound, max_over_Lplus1=float(o.max() / (L + 1)),
         replica_maxabs_diff=float(np.nanmax(np.abs(S - Vr))), pyspi_asym=asym(S),
         notes="pass = a value above 1 (impossible for an average of correlations in [-1,1]) and every value within the sum bound")


def ref_scaleinv(run, kind, L, x, ds):
    bg = np.asarray(make_data.load("bivariate_gaussian", run["T"])["x"], float)
    if run.get("processes") is not None:
        bg = bg[list(run["processes"])]
    Z = zscore(bg)
    V = total(pair_terms(kind, L, "replica", Z, ("bivariate_gaussian", run["T"], "pipeline", str(run.get("processes")))))
    S, xin = load_pyspi(run["run_id"])
    extra = {"pyspi_asym": asym(S)} if S is not None else {}
    if xin is not None:
        extra["pyspi_input_vs_zscored_bivariate_maxabs"] = float(np.max(np.abs(xin - Z)))
    save(run["run_id"], value=V, criterion="tolerance", reference_kind=replica_kind(kind),
         reference_detail=detail(kind, "replica", "on the z-scored bivariate_gaussian draw (scaled_offset = scale * D2 + offset)"),
         notes="positive scale/offset invariance through pyspi's z-score", **TOL, **extra)


def ref_smalln(run, kind, L, x, ds):
    S, _ = load_pyspi(run["run_id"])
    need, code = (L + 3, 1) if kind == "dcorrx" else (L + 5, 2)
    ns, codes = S[:, 0].astype(int), S[:, 1].astype(int)
    ok = bool(np.all(np.where(ns < need, codes == code, codes == 0)))
    vals = []
    for n, c, v in S:
        if c == 0 and n >= need + 1:
            vals.append(abs(v - float(np.sum(terms_fn(kind, L, "replica")(x[0, :int(n)], x[1, :int(n)])))))
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=("hyppo dcorr.py:262-281 U-centring divides by n-2 and (n-1)(n-2): a lag slice with < 3 samples raises "
                           "ZeroDivisionError, so T >= L+3" if kind == "dcorrx" else
                           "scipy/stats/_mgc.py:331-333 raises ValueError for n < 5 samples in any lag slice, so T >= L+5"),
         threshold_T=need, pyspi_rows_n=[int(v) for v in ns], pyspi_rows_code=[int(v) for v in codes],
         replica_maxabs_diff_above_threshold=float(max(vals)) if vals else float("nan"),
         notes="codes: 0 ok, 1 ZeroDivisionError, 2 ValueError, 3 IndexError, 4 other; pass = raises exactly below the threshold")


def ref_multireal(run, kind, L, x, ds):
    P, T = x.shape
    h = T // 2
    x3 = np.stack([x[:, :h], x[:, h:2 * h]], axis=-1)       # (P, T/2, 2): each time point is a 2-vector
    V = np.full((P, P), np.nan)
    Vavg = np.full((P, P), np.nan)
    fn = terms_fn(kind, L, "replica")
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = float(np.sum(fn(x3[i], x3[j])))
                Vavg[i, j] = 0.5 * sum(float(np.sum(fn(x3[i, :, r], x3[j, :, r]))) for r in range(2))
    S, _ = load_pyspi(run["run_id"])
    extra = {"pyspi_minus_realisation_average_maxabs": float(np.nanmax(np.abs(S - Vavg)))} if S is not None else {}
    save(run["run_id"], value=V, criterion="tolerance", reference_kind=replica_kind(kind),
         reference_detail=detail(kind, "replica", "with (T/2, 2) samples: Euclidean distances between 2-vectors (one per realisation)"),
         notes="R > 1: hyppo treats z[i] (T, R) as T samples in R dimensions; NOT an average over realisations", **TOL, **extra)


def ref_lag0(run, kind, L, x, ds):
    Z = preprocess(x, run["path"])
    P = Z.shape[0]
    M = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                M[i, j] = mgc_stat(sk_dist(Z[i]), sk_dist(Z[j]))
    S, _ = load_pyspi(run["run_id"])
    extra = {"ref_mgc_asym": asym(M)}
    if S is not None:
        extra["mgcx_lag0_minus_mgc_unit_maxabs"] = float(np.nanmax(np.abs(S[0] - S[1])))
        extra["pyspi_mgc_unit_asym"] = asym(S[1])
        extra["pyspi_lag0_asym"] = asym(S[0])
    save(run["run_id"], value=np.stack([M, M]), criterion="tolerance", reference_kind="independent-impl",
         reference_detail=f"{SELF}:mgc_stat on sklearn-mirrored distances of the series, every ordered pair; {VERS}",
         notes="slice 0: MGCX(max_lag=0) = lag-0 term of mgcx; slice 1: pyspi mgc SPI (distance.mgc); both vs from-scratch MGC",
         **TOL, **extra)


def ref_mgcmap(run, kind, L, x, ds):
    Z = preprocess(x, run["path"])
    C = mgc_local_corr(sk_dist(Z[0]), sk_dist(Z[1]))
    S, _ = load_pyspi(run["run_id"])
    extra = {"map_shape": list(C.shape)}
    if S is not None:
        extra["pyspi_map_shape"] = list(S.shape)
    save(run["run_id"], value=C, criterion="tolerance", reference_kind="independent-impl",
         reference_detail=f"{SELF}:mgc_local_corr (Shen, Priebe & Vogelstein 2020 Sec. 3.1; dense column ranks); {VERS}",
         notes="tests SciPy's compiled _local_correlations (scipy/stats/_stats.pyx, not shipped in the wheel)", **TOL, **extra)


def ref_calls(run, kind, L, x, ds):
    S, _ = load_pyspi(run["run_id"])
    n_full, n_stat, dmax, rng_ok, t_full, t_stat = [float(v) for v in S]
    ok = (n_full == L + 3) and (n_stat == L + 1) and dmax == 0.0 and rng_ok == 1.0
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail="hyppo mgcx.py:106-118 + _utils.py:111-126: compute_stat makes L+1 multiscale_graphcorr calls; "
                          "compute_scale_at_lag -> MGC.test makes 1 (mgc.py:254) + 1 via perm_test's observed statistic (base.py:130, common.py:730)",
         calls_per_pair=n_full, calls_needed=n_stat, value_change=dmax, rng_untouched=rng_ok,
         time_full_s=t_full, time_compute_stat_s=t_stat, overhead_fraction=(t_full - t_stat) / t_full,
         notes="pass = L+3 calls where L+1 are needed, identical statistic, global RNG unchanged")


REFS = {"replica": ref_replica, "scipy": ref_scipy, "mehta": ref_mehta, "lagterms": ref_lagterms, "dir": ref_dir,
        "null": ref_null, "neg": ref_neg, "sum": ref_sum, "scaleinv": ref_scaleinv, "smalln": ref_smalln,
        "multireal": ref_multireal, "lag0": ref_lag0, "mgcmap": ref_mgcmap, "calls": ref_calls}


def main():
    runs = json.loads(MANIFEST.read_text())["runs"]
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in runs:
        rid = run["run_id"]
        if only and not any(o in rid for o in only):
            continue
        tag = rid.split("|")[-1].split("-p")[0]
        kind, L = unit_kind(run["unit_id"])
        try:
            ds, x, _ = run_data(run)
            t0 = time.time()
            REFS[tag](run, kind, L, x, ds)
            n_ok += 1
            dt = time.time() - t0
            if dt > 5:
                print(f"  {rid}: {dt:.1f} s", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr, flush=True)
            n_err += 1
    print(f"lagged_dcorr_mgc refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
