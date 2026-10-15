"""pdist family references (COMPARISON venv): distance.pdist.{euclidean, cityblock,
cosine, chebyshev, canberra, braycurtis}.

pyspi computes ``sklearn.metrics.pairwise_distances(X (P, T), metric)``
(pyspi/statistics/distance.py:38-40) on the rows of the (optionally detrended,
then ddof=1 z-scored) data.  Independent references here use numpy/scipy
primitives only (never sklearn, never scipy.spatial), except the explicitly
same-lineage ``cdist`` runs.

Run tags (last field of the run_id) and what each reference does:

  def              from-definition distance (_kernels_basic_misc.pdist_metric) on the rows
                   pyspi receives: raw (kernel), own ddof=1 z-score (pipeline), own linear
                   detrend + z-score (pipeline-detrend); exact rtol = atol = 1e-12
  cdist            scipy.spatial.distance.cdist(z, z, metric) in this venv: SAME-LINEAGE
                   (pyspi's chebyshev/canberra/braycurtis/cityblock reach scipy's C++ code)
  ident            analytic identities on z-scored rows: cosine = 1 - r,
                   euclidean = sqrt(2 (T-1) (1 - r)), r = from-scratch Pearson of the raw rows
  scaleinv         scaled_offset (pipeline) vs from-definition distance on the z-scored
                   bivariate_gaussian draw it was built from (positive scale + offset)
  detrendinv       trending (pipeline-detrend) vs from-definition distance on the detrended,
                   z-scored bivariate_gaussian draw (trending = bivariate + slope * t)
  trend-nodetrend  trending through the plain pipeline vs from-definition distance on the
                   z-scored trending rows (exact), with the change against the untrended draw
  perm             pyspi on a joint time permutation (probe) vs from-definition distance on
                   the unpermuted z-scored rows (tolerance rtol 1e-10)
  sym, sym-wide64  qualitative: NaN diagonal and max|A - A.T| from pyspi's own matrix;
                   bit-exact symmetry expected except euclidean (last-bit, REVIEW M2).
                   pyspi is judged against itself (A vs A.T): labelled same-lineage
                   (self-consistency)
  pop, pop-negated statistical: analytic population value for a Gaussian pair with the
                   dataset's true rho; se = Monte-Carlo sd (2000 z-scored Gaussian pairs at the
                   same T and rho) times z_{1-0.025/m}/2 (Bonferroni over the m unique pairs),
                   so |pyspi - value| <= 2 se is a family-wise 5 % test
  delayed-null     qualitative: every entry of delayed_noise (lag-0 independent pairs) lies
                   inside the Bonferroni 95 % band of the Monte-Carlo rho = 0 null
  dup              shared_envelope: x2 is a bit-identical copy of x0, so d(x0, x2) = 0 by
                   definition (exact atol 1e-12)
  near             rows x0 + eps x1 (eps 1e-3 .. 1e-9): numerically stable references
                   (direct difference; cosine as |u/|u| - v/|v||^2 / 2); tolerance rtol 1e-6
  edge             kernel path on (x, -x) and rows with shared exact zeros (canberra 0/0 := 0)
  signed-affine    qualitative: (x, -x, 2x+1, x1) through the pipeline (canberra, braycurtis);
                   pyspi's entries are tested against analytic constants (canberra(x,-x) = T,
                   braycurtis(x,-x) = +inf, d(x, 2x+1) = 0): labelled analytic
  signed-warn      qualitative: same input, warnings raised inside multivariate();
                   a pyspi-side observation only: labelled same-lineage (self-consistency)
  calc-downstream  qualitative: Calculator.compute + _get_correlation_df(rmmin=True) with +inf;
                   a pyspi-side observation only: labelled same-lineage (self-consistency)
  contam           qualitative: chebyshev on contaminated vs the clean pair

Writes out/ref/<run_id>.npz per the compare.py contract.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np
import scipy
from scipy import integrate, special, stats

import make_data
from _kernels_basic_misc import pdist_metric as _pdist_metric_kernel

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
SELF = "audit/checks/refs/pdist.py"
KERNEL = "audit/checks/refs/_kernels_basic_misc.py:pdist_metric"

PERM_SEED = 20261014          # mirrors probes/pdist_derived.py
MIX_SEED = 7
NEAR_EPS = (1e-3, 1e-5, 1e-7, 1e-9)
MC_N = 2000
MC_SEED = 9090


# ------------------------------------------------------------------ primitives
def pdist_metric(X, metric):
    with np.errstate(divide="ignore", invalid="ignore"):
        D = _pdist_metric_kernel(X, metric)
    np.fill_diagonal(D, np.nan)
    return D


def zscore(X):
    """ddof=1 z-score per row (pyspi/data.py:195 uses scipy.stats.zscore(axis=1, ddof=1))."""
    X = np.asarray(X, float)
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)


def detrend(X):
    """Least-squares removal of a + b t per row (scipy.signal.detrend type='linear')."""
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


def pearson(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    G = Xc @ Xc.T
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def derive(kind, x):
    """Mirror of probes/pdist_derived.py derive()."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[1]
    if kind == "permuted":
        return x[:, np.random.default_rng(PERM_SEED).permutation(T)]
    if kind == "wide64":
        return np.random.default_rng(MIX_SEED).standard_normal((64, x.shape[0])) @ x
    if kind == "negated":
        return x * np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    if kind == "near_identical":
        return np.vstack([x[0]] + [x[0] + e * x[1] for e in NEAR_EPS])
    if kind == "edge":
        a = x[1].copy(); b = x[2].copy()
        a[::4] = 0.0; b[::4] = 0.0
        return np.vstack([x[0], -x[0], a, b])
    if kind == "signed_affine":
        return np.vstack([x[0], -x[0], 2.0 * x[0] + 1.0, x[1]])
    raise ValueError(kind)


def stable_metric(X, metric):
    """Cancellation-free forms for near-identical rows."""
    if metric == "cosine":
        U = X / np.linalg.norm(X, axis=1, keepdims=True)
        D = 0.5 * np.sum((U[:, None, :] - U[None, :, :]) ** 2, -1)
        np.fill_diagonal(D, np.nan)
        return D
    return pdist_metric(X, metric)      # every other definition is already a direct difference


# ------------------------------------------------------------ population values
def chebyshev_pop(rho, T):
    """E max_t |X_t - Y_t|, X - Y ~ N(0, 2(1 - rho)) iid: int_0^inf 1 - erf(t / (2 sqrt(1-rho)))^T dt."""
    s = 2.0 * np.sqrt(1.0 - rho)
    f = lambda t: 1.0 - special.erf(t / s) ** T
    return integrate.quad(f, 0, np.inf, limit=400)[0]


def canberra_pop_per_sample(rho):
    """E |X-Y| / (|X|+|Y|) for a unit-variance Gaussian pair: depends on the polar angle only,
    whose density is sqrt(1-rho^2) / (2 pi (1 - rho sin 2 th))."""
    g = lambda th: abs(np.cos(th) - np.sin(th)) / (abs(np.cos(th)) + abs(np.sin(th)))
    f = lambda th: np.sqrt(1 - rho ** 2) / (2 * np.pi * (1 - rho * np.sin(2 * th)))
    pts = [k * np.pi / 4 for k in range(1, 8)]
    return integrate.quad(lambda th: g(th) * f(th), 0, 2 * np.pi, points=pts, limit=400)[0]


def population(metric, rho, T):
    if metric == "euclidean":
        return np.sqrt(2 * (T - 1) * (1 - rho))
    if metric == "cityblock":
        return T * 2 * np.sqrt((1 - rho) / np.pi)
    if metric == "cosine":
        return 1 - rho
    if metric == "chebyshev":
        return chebyshev_pop(rho, T)
    if metric == "canberra":
        return T * canberra_pop_per_sample(rho)
    if metric == "braycurtis":
        return np.sqrt((1 - rho) / (1 + rho))
    raise ValueError(metric)


_MC = {}


def mc_sample(rho, T):
    """MC_N independent z-scored Gaussian pairs (rho, T); returns {metric: (MC_N,) values}."""
    key = (round(float(rho), 12), int(T))
    if key in _MC:
        return _MC[key]
    rng = np.random.default_rng([MC_SEED, int(T), int(round(1000 * rho)) + 1000])
    out = {m: [] for m in ("euclidean", "cityblock", "cosine", "chebyshev", "canberra", "braycurtis")}
    chunk = 250
    for _ in range(MC_N // chunk):
        a = rng.standard_normal((chunk, T))
        b = rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal((chunk, T))
        za, zb = zscore(a), zscore(b)
        d = za - zb
        out["euclidean"].append(np.sqrt(np.sum(d ** 2, 1)))
        out["cityblock"].append(np.sum(np.abs(d), 1))
        out["cosine"].append(1 - np.sum(za * zb, 1) / np.sqrt(np.sum(za ** 2, 1) * np.sum(zb ** 2, 1)))
        out["chebyshev"].append(np.max(np.abs(d), 1))
        den = np.abs(za) + np.abs(zb)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["canberra"].append(np.sum(np.where(den > 0, np.abs(d) / den, 0.0), 1))
        out["braycurtis"].append(np.sum(np.abs(d), 1) / np.sum(np.abs(za + zb), 1))
    res = {m: np.concatenate(v) for m, v in out.items()}
    _MC[key] = res
    return res


def corr_truth(ds, T, negated=False):
    d = make_data.load(ds, T)
    P = d["x"].shape[0]
    if "cov_true" in d:
        C = np.asarray(d["cov_true"], float)
        R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    else:
        R = np.eye(P)                       # iid_gaussian: independent processes
    if negated:
        s = np.array([1.0, -1.0, 1.0, -1.0])
        R = R * np.outer(s, s)
    return R


# ----------------------------------------------------------------- npz writing
def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in kw.items():
        clean[k] = np.asarray(v)
    np.savez(OUT_REF / f"{rid}.npz", **clean)


def load_pyspi(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], float)


def pyspi_status(rid):
    f = OUT_PY / "index" / "pdist.json"
    if not f.is_file():
        return {}
    return json.loads(f.read_text()).get(rid, {})


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


# ---------------------------------------------------------------- run handlers
def ref_def(run, metric, x):
    Z = preprocess(x, run["path"])
    V = pdist_metric(Z, metric)
    note = {"kernel": "raw rows (Data normalise=False)", "pipeline": "own ddof=1 z-score",
            "pipeline-detrend": "own lstsq linear detrend then ddof=1 z-score"}[run["path"]]
    extra = {}
    S = load_pyspi(run["run_id"])
    if S is not None:
        extra["max_asym_pyspi"] = float(np.nanmax(np.abs(S - S.T)))
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:ref_def -> {KERNEL} ({metric}); {note}",
         notes="from-definition numpy distance, no sklearn / scipy.spatial", **extra)


def ref_cdist(run, metric, x):
    from scipy.spatial.distance import cdist
    Z = preprocess(x, run["path"])
    V = cdist(Z, Z, metric)
    np.fill_diagonal(V, np.nan)
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="same-lineage",
         reference_detail=f"{SELF}:ref_cdist -> scipy=={scipy.__version__} scipy.spatial.distance.cdist(z, z, '{metric}')",
         notes="same-lineage delegate (scipy.spatial); recorded only as same-lineage")


def ref_ident(run, metric, x):
    R = pearson(np.asarray(x, float))
    T = x.shape[1]
    if metric == "cosine":
        V = 1 - R; formula = "1 - r"
    else:
        V = np.sqrt(np.maximum(2 * (T - 1) * (1 - R), 0.0)); formula = "sqrt(2 (T-1) (1 - r))"
    np.fill_diagonal(V, np.nan)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:ref_ident; {formula} with r = from-scratch Pearson of the raw rows (ddof=1 z-scores: |z|^2 = T-1)",
         notes="analytic identity on z-scored rows")


def ref_scaleinv(run, metric, x):
    xb = make_data.load("bivariate_gaussian", run["T"])["x"]
    V = pdist_metric(zscore(xb), metric)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:ref_scaleinv; from-definition distance on z-scored bivariate_gaussian (scaled_offset = scale*x + offset, scale > 0)",
         notes="invariance to positive per-channel scale and offset through z-scoring")


def ref_detrendinv(run, metric, x):
    xb = make_data.load("bivariate_gaussian", run["T"])["x"]
    V = pdist_metric(zscore(detrend(xb)), metric)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:ref_detrendinv; from-definition distance on detrended z-scored bivariate_gaussian (trending = x + slope t)",
         notes="pipeline-detrend removes the linear trend exactly")


def ref_trend_nodetrend(run, metric, x):
    V = pdist_metric(zscore(x), metric)
    xb = make_data.load("bivariate_gaussian", run["T"])["x"]
    U = pdist_metric(zscore(xb), metric)
    m = ~np.isnan(V)
    change = np.abs(V[m] - U[m]) / np.abs(U[m])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:ref_trend_nodetrend -> {KERNEL} on z-scored trending rows",
         notes="contrast: without detrend the trend enters the distance",
         max_rel_change_vs_untrended=float(change.max()), pair01_trending=float(V[0, 1]), pair01_untrended=float(U[0, 1]),
         pair13_trending=float(V[1, 3]), pair13_untrended=float(U[1, 3]))


def ref_perm(run, metric, x):
    V = pdist_metric(zscore(x), metric)                 # unpermuted rows
    extra = {}
    S = load_pyspi(run["run_id"])
    base = load_pyspi(f"distance.pdist.{metric}|bivariate_gaussian|{run['T']}|pipeline|def")
    if S is not None and base is not None:
        extra["max_abs_vs_pyspi_unpermuted"] = float(np.nanmax(np.abs(S - base)))
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:ref_perm; from-definition distance on the UNPERMUTED z-scored rows; pyspi side permutes time jointly (default_rng({PERM_SEED}))",
         notes="invariance to a joint reordering of samples", **extra)


def ref_sym(run, metric, x, wide=False):
    S = load_pyspi(run["run_id"])
    X = derive("wide64", x) if wide else x
    V = pdist_metric(zscore(X), metric)
    if S is None:
        save(run["run_id"], value=V, criterion="qualitative", pass_flag=0, reference_kind="same-lineage",
             reference_detail=f"{SELF}:ref_sym", notes="pyspi output missing")
        return
    diag_nan = bool(np.all(np.isnan(np.diag(S))))
    asym = np.abs(S - S.T)
    max_asym = float(np.nanmax(asym))
    scale = float(np.nanmax(np.abs(S[np.isfinite(S)])))
    n_asym = int(np.nansum(asym > 0) // 2)
    if metric == "euclidean":
        ok = max_asym <= 1e-12 * scale          # last-bit reciprocal asymmetry allowed (REVIEW M2)
        rule = "max|A-A.T| <= 1e-12 * max|A| (last-bit allowed, REVIEW M2)"
    else:
        ok = max_asym == 0.0
        rule = "max|A-A.T| == 0 (bit-exact)"
    m = ~np.isnan(S) & ~np.isnan(V)
    def_rel = float(np.max(np.abs(S[m] - V[m]) / np.maximum(np.abs(V[m]), 1e-300))) if m.any() else 0.0
    save(run["run_id"], value=S.T, criterion="qualitative", pass_flag=int(ok and diag_nan),
         reference_kind="same-lineage",
         reference_detail=f"{SELF}:ref_sym; self-consistency (pyspi's matrix vs its own transpose); symmetric metric d(u,v) = d(v,u); rule {rule}; NaN diagonal",
         notes="pass = NaN diagonal and symmetry rule holds; def_max_rel = from-definition agreement",
         max_asym=max_asym, max_rel_asym=max_asym / scale, n_asym_pairs=n_asym, diag_nan=int(diag_nan),
         def_max_rel=def_rel, P=int(S.shape[0]))


def ref_pop(run, metric, x, negated=False):
    T = run["T"]
    R = corr_truth(run["dataset"], T, negated=negated)
    P = R.shape[0]
    m_pairs = P * (P - 1) // 2
    c = stats.norm.ppf(1 - 0.05 / (2 * m_pairs)) / 2.0
    V = np.full((P, P), np.nan); SE = np.full((P, P), np.nan)
    MCm = np.full((P, P), np.nan); SD = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            rho = float(R[i, j])
            s = mc_sample(rho, T)[metric]
            V[i, j] = population(metric, rho, T)
            SD[i, j] = s.std(ddof=1); MCm[i, j] = s.mean()
            SE[i, j] = SD[i, j] * c
    extra = {}
    S = load_pyspi(run["run_id"])
    if S is not None:
        mk = ~np.isnan(S)
        extra["max_z_raw"] = float(np.max(np.abs(S[mk] - V[mk]) / SD[mk]))
        extra["max_rel_err"] = float(np.max(np.abs(S[mk] - V[mk]) / np.abs(V[mk])))
    mk = ~np.eye(P, dtype=bool)
    extra["max_bias_over_sd"] = float(np.max(np.abs(MCm[mk] - V[mk]) / SD[mk]))
    extra["rho_pairs"] = np.array(sorted({round(float(r), 6) for r in offdiag(R)}))
    if metric == "braycurtis":
        extra["max_value_pop"] = float(np.nanmax(V))
    save(run["run_id"], value=V, criterion="statistical", se=SE, reference_kind="analytic",
         reference_detail=(f"{SELF}:ref_pop; population value for a Gaussian pair (population({metric}, rho, T)); "
                           f"se = MC sd ({MC_N} z-scored Gaussian pairs) x z_(1-0.025/{m_pairs})/2 = sd x {c:.4f}"),
         notes=("rho from dataset truth (cov_true -> correlation" + (", rows 1 and 3 negated" if negated else "") +
                "); |pyspi - value| <= 2 se is a Bonferroni family-wise 5 % test over the unique pairs"),
         bonferroni_factor=c, **extra)


def ref_delayed(run, metric, x):
    S = load_pyspi(run["run_id"])
    T = run["T"]
    s = mc_sample(0.0, T)[metric]
    P = x.shape[0]
    m_pairs = P * (P - 1) // 2
    lo, hi = np.quantile(s, [0.025 / m_pairs, 1 - 0.025 / m_pairs])
    V = np.full((P, P), population(metric, 0.0, T)); np.fill_diagonal(V, np.nan)
    if S is None:
        save(run["run_id"], value=V, criterion="qualitative", pass_flag=0, reference_kind="analytic",
             reference_detail=f"{SELF}:ref_delayed", notes="pyspi output missing")
        return
    vals = offdiag(S)
    ok = bool(np.all((vals >= lo) & (vals <= hi)))
    save(run["run_id"], value=V, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{SELF}:ref_delayed; Monte-Carlo rho=0 null ({MC_N} z-scored Gaussian pairs, T={T}), Bonferroni 95 % band over {m_pairs} pairs",
         notes="pass = every delayed-copy pair lies inside the independence band (lag-0 metric cannot see delayed coupling)",
         band_lo=float(lo), band_hi=float(hi), null_value=float(population(metric, 0.0, T)),
         d_x0_x1=float(S[0, 1]), d_x0_x2=float(S[0, 2]), d_x0_x3=float(S[0, 3]))


def ref_dup(run, metric, x):
    Z = zscore(x)
    V = pdist_metric(Z, metric)
    assert np.array_equal(Z[0], Z[2])
    V[0, 2] = V[2, 0] = 0.0
    extra = {}
    S = load_pyspi(run["run_id"])
    if S is not None:
        extra["pyspi_d02"] = float(S[0, 2]); extra["pyspi_d20"] = float(S[2, 0])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:ref_dup; x2 is a bit-identical copy of x0 (shared_envelope), so d(z0, z2) = 0 by definition; other pairs from {KERNEL}",
         notes="identical rows must give exactly 0", **extra)


def ref_near(run, metric, x):
    X = derive("near_identical", x)
    V = stable_metric(zscore(X), metric)
    extra = {"ref_d0k": V[0, 1:].copy()}
    S = load_pyspi(run["run_id"])
    if S is not None:
        extra["pyspi_d0k"] = S[0, 1:].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            extra["rel_err_d0k"] = np.abs(S[0, 1:] - V[0, 1:]) / np.abs(V[0, 1:])
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-6, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:ref_near; rows z(x0 + eps x1), eps in {NEAR_EPS}; direct-difference distances (cosine via |u/|u| - v/|v||^2 / 2)",
         notes="numerical accuracy for near-identical channels", **extra)


def ref_edge(run, metric, x):
    X = derive("edge", x)
    V = pdist_metric(X, metric)
    extra = {"ref_d01": float(V[0, 1]), "ref_d23": float(V[2, 3]), "T_len": int(X.shape[1])}
    S = load_pyspi(run["run_id"])
    if S is not None:
        extra["pyspi_d01"] = float(S[0, 1]); extra["pyspi_d23"] = float(S[2, 3])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:ref_edge; kernel rows (x, -x, a, b) with a, b zero at t % 4 == 0; {KERNEL} (canberra 0/0 := 0)",
         notes="edge inputs: canberra(x,-x) = T; braycurtis(x,-x) = +inf", **extra)


def ref_signed_affine(run, metric, x):
    S = load_pyspi(run["run_id"])
    T = run["T"]
    X = derive("signed_affine", x)
    V = pdist_metric(zscore(X), metric)
    if S is None:
        save(run["run_id"], value=V, criterion="qualitative", pass_flag=0, reference_kind="analytic",
             reference_detail=f"{SELF}:ref_signed_affine", notes="pyspi output missing")
        return
    if metric == "braycurtis":
        ok = (np.isposinf(S[0, 1]) and np.isposinf(S[1, 0]) and abs(S[0, 2]) < 1e-12
              and (np.isposinf(S[1, 2]) or S[1, 2] > 1e10) and np.isfinite(S[0, 3]))
        rule = "S01 = S10 = +inf; S02 < 1e-12; S12 = +inf or > 1e10 (decided by z-score rounding); S03 finite"
    else:
        ok = (S[0, 1] == T and S[1, 0] == T and abs(S[0, 2]) < 1e-9 and abs(S[1, 2] - T) < 1e-9 and S[0, 3] <= T)
        rule = "S01 = S10 = T exactly; S02 ~ 0; S12 ~ T; all <= T"
    save(run["run_id"], value=V, criterion="qualitative", pass_flag=int(bool(ok)), reference_kind="analytic",
         reference_detail=f"{SELF}:ref_signed_affine; rows (x, -x, 2x+1, x1) through the pipeline; rule {rule}",
         notes="signed z-scored input outside the non-negative domain of the metric",
         S01=float(S[0, 1]), S10=float(S[1, 0]), S02=float(S[0, 2]), S12=float(S[1, 2]), S03=float(S[0, 3]),
         ref_S12=float(V[1, 2]), T_len=int(T))


def ref_signed_warn(run, metric, x):
    S = load_pyspi(run["run_id"])
    if S is None:
        save(run["run_id"], value=np.zeros(6), criterion="qualitative", pass_flag=0, reference_kind="same-lineage",
             reference_detail=f"{SELF}:ref_signed_warn", notes="pyspi output missing")
        return
    n_w, n_rt = int(S[0]), int(S[1])
    if metric == "braycurtis":
        ok = n_rt == 0 and np.isposinf(S[2])
        rule = "no RuntimeWarning while S01 = +inf (silent)"
    else:
        ok = n_rt == 0
        rule = "no RuntimeWarning"
    save(run["run_id"], value=S, criterion="qualitative", pass_flag=int(bool(ok)), reference_kind="same-lineage",
         reference_detail=f"{SELF}:ref_signed_warn; probes/pdist_derived.py:signed_affine_warnings; rule {rule}",
         notes="pass = reproduces the trace observation that the extreme values are produced silently",
         n_warnings=n_w, n_runtime_warnings=n_rt, S01=float(S[2]), S12=float(S[5]))


def ref_calc_downstream(run, metric, x):
    S = load_pyspi(run["run_id"])
    if S is None:
        save(run["run_id"], value=np.zeros((4, 3)), criterion="qualitative", pass_flag=0, reference_kind="same-lineage",
             reference_detail=f"{SELF}:ref_calc_downstream", notes="pyspi output missing")
        return
    C = S[:3]
    n_nonfinite = int(S[3, 2])
    ok = np.isposinf(S[3, 0]) and n_nonfinite == 0
    save(run["run_id"], value=S, criterion="qualitative", pass_flag=int(bool(ok)), reference_kind="same-lineage",
         reference_detail=(f"{SELF}:ref_calc_downstream; probes/pdist_derived.py:calc_downstream "
                           "(Calculator.compute + _get_correlation_df(rmmin=True), pyspi/calculator.py:361-366, 427-458)"),
         notes="pass = +inf survives compute and _rmmin but the rank-based SPI-SPI correlation stays finite",
         bc01_after_compute=float(S[3, 0]), bc01_after_rmmin=float(S[3, 1]), n_nonfinite_corr=n_nonfinite,
         spearman_euc_bc=float(C[0, 2]), spearman_can_bc=float(C[1, 2]))


def ref_contam(run, metric, x):
    S = load_pyspi(run["run_id"])
    d = make_data.load("contaminated", run["T"])
    V = pdist_metric(zscore(d["x"]), metric)
    clean = pdist_metric(zscore(d["x_clean"]), metric)
    if S is None:
        save(run["run_id"], value=V, criterion="qualitative", pass_flag=0, reference_kind="analytic",
             reference_detail=f"{SELF}:ref_contam", notes="pyspi output missing")
        return
    ratio = float(S[0, 1] / clean[0, 1])
    save(run["run_id"], value=V, criterion="qualitative", pass_flag=int(ratio > 2.0), reference_kind="analytic",
         reference_detail=f"{SELF}:ref_contam; chebyshev on z-scored contaminated vs z-scored x_clean (bivariate_gaussian pair (2,3), rho 0.8)",
         notes="pass = outliers inflate the single-sample maximum by more than 2x",
         pyspi_contaminated=float(S[0, 1]), clean_value=float(clean[0, 1]), ratio=ratio,
         def_contaminated=float(V[0, 1]))


HANDLERS = {
    "def": ref_def, "cdist": ref_cdist, "ident": ref_ident, "scaleinv": ref_scaleinv,
    "detrendinv": ref_detrendinv, "trend-nodetrend": ref_trend_nodetrend, "perm": ref_perm,
    "sym": ref_sym, "sym-wide64": lambda r, m, x: ref_sym(r, m, x, wide=True),
    "pop": ref_pop, "pop-negated": lambda r, m, x: ref_pop(r, m, x, negated=True),
    "delayed-null": ref_delayed, "dup": ref_dup, "near": ref_near, "edge": ref_edge,
    "signed-affine": ref_signed_affine, "signed-warn": ref_signed_warn,
    "calc-downstream": ref_calc_downstream, "contam": ref_contam,
}


def main():
    man = json.loads((CHECKS / "manifest" / "pdist.json").read_text())
    n = 0
    for run in man["runs"]:
        tag = run["run_id"].split("|")[-1]
        metric = run["identifier"][len("pdist_"):]
        x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], float)
        if run.get("processes") is not None:
            x = x[list(run["processes"])]
        HANDLERS[tag](run, metric, x)
        n += 1
    print(f"refs/pdist.py: wrote {n} reference files")


if __name__ == "__main__":
    main()
