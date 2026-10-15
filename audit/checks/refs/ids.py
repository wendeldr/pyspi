"""ids references (COMPARISON venv: numpy/scipy only; no pyspi, no pyspi.lib.ids).

Unit misc.ids = InterDependenceScore (pyspi/statistics/misc.py:178-200) -> vendored
pyspi/lib/ids/numpy_dependence.py:compute_IDS_numpy (Radhakrishnan, Jain, Uhler & Lander 2025,
PNAS 122(34):e2509860122).  References:

* ``_kernels_basic_misc.ids`` (shared kernel: transform, explicit Pearson via dot / norms, max over
  the term blocks; mirrors numpy_dependence.py:16-23, 76-92 without calling np.corrcoef);
* ``eq6_ids`` below: an own per-pair transcription of the paper, Eq. 6 feature map
  phi_m(x) = exp(-x^2 / (2 B^2)) x^m / (sqrt(m!) B^m), m = 0..k-1, Eq. 11 correlations, Eq. 12
  IDS_p = ((1/k^2) sum |rho|^p)^(1/p), IDS_inf = max; pyspi's ``bandwidth`` b is the coefficient of
  x^2, so B = sqrt(1 / (2 b)) (b = 0.5 <-> B = 1);
* analytic cases (identical / sign-flipped / affinely transformed copies -> 1; z-scoring invariance);
* population IDS-max of a standard bivariate Gaussian by 2-D Gauss-Hermite quadrature;
* own Monte Carlo of the finite-sample statistic (null: iid N(0,1); truth: bivariate Gaussian
  with the D2 rho; random-walk null; AR(1) null as extra keys) and per-pair permutation nulls;
* the RNG side-effect probes are judged against the legacy MT19937 stream
  (np.random.RandomState(seed), a deterministic function of the seed, identical in both venvs).

Run tags (last field of run_id):
  kids              exact vs _kernels_basic_misc.ids on the prepared input (+ Eq. 6 key)
  invariance        scaled_offset (pipeline) vs ids of z-scored bivariate_gaussian rows (analytic)
  signflip          probe with row 1 negated vs ids of the unflipped prepared rows (analytic)
  identical(-pair)  analytic: every off-diagonal entry 1
  eq6-bw*           probe InterDependenceScore(bandwidth=b) vs eq6_ids with B = sqrt(1/(2b))
  pnorm1/2          probe InterDependenceScore(pnorm=p) vs eq6_ids Eq. 12 with p
  minmax08          probe: rows min-max scaled to [0, 8] (paper's reported setting) then the
                    kernel path, vs eq6_ids(B = 1) on the same scaled rows
  paper-minmax08    pyspi default (z-score pipeline) vs the paper's reported setting
                    (min-max [0, 8], B = 1, k = 6, IDS_inf); EXPECTED TO FAIL, quantifies the gap
  truth-p01/p23     statistical: |pyspi - MC mean| <= 2 MC sd (rho 0.3 / 0.8); population value
                    by quadrature as extra key
  null              statistical: iid N(0,1), MC mean / sd at T; se inflated familywise over pairs
  null-randomwalk   statistical: independent random walks (cointegrated x2, x3) vs random-walk MC
  structure         qualitative: symmetric, dependent pairs above and independent pairs below the
                    familywise permutation bound
  zero-lag          qualitative: delayed_noise (lagged copies) all pairs below the permutation bound
  hidden-dependence statistical: constructed dependent pair with population IDS-max = 0 at k = 6;
                    extra keys: own MC (300 draws) of IDS-max and Corr(g(X), g(Y)) under that law
  pnorm-unsupported qualitative: non-default pnorm (3, "2") gives an all-NaN Calculator block
  rng-*             qualitative: global-RNG side effects of importing pyspi.lib.ids
  symmetry          qualitative: max |A - A^T| over every square output of this family

Run: python audit/checks/refs/ids.py
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))

import make_data  # noqa: E402
import _kernels_basic_misc as K  # noqa: E402

FAMILY = "ids"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
MC_SEED = 20261015
N_MC = 2000
N_PERM = 1000
D2_RHO = {(0, 1): 0.3, (2, 3): 0.8}

_spec = importlib.util.spec_from_file_location("ids_probes", CHECKS / "probes" / "ids_probes.py")
PR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PR)          # numpy-only top level (pyspi imported lazily inside functions)


# ------------------------------------------------------------------ preprocessing (own)
def zscore1(x):
    x = np.asarray(x, float)
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def detrend_linear(x):
    x = np.asarray(x, float)
    T = x.shape[1]
    A = np.column_stack([np.ones(T), np.arange(T, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, x.T, rcond=None)
    return x - (A @ coef).T


def prepare(x, path):
    if path == "kernel":
        return np.asarray(x, float)
    if path == "pipeline":
        return zscore1(x)
    if path == "pipeline-detrend":
        return zscore1(detrend_linear(x))
    raise ValueError(path)


def minmax08(x):
    x = np.asarray(x, float)
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return 8.0 * (x - lo) / (hi - lo)


# ------------------------------------------------------------------ own transcription of the paper
def eq6_features(x, B, k=6):
    """Eq. 6: phi_m(x) = exp(-x^2/(2 B^2)) x^m / (sqrt(m!) B^m), m = 0..k-1 -> (k, T)."""
    x = np.asarray(x, float)
    return np.stack([np.exp(-x**2 / (2.0 * B**2)) * x**m / (math.sqrt(math.factorial(m)) * B**m)
                     for m in range(k)])


def eq11_corr(a, b):
    """Eq. 11 sample version: Cov / sqrt(Var Var), 0 when a variance is 0 (code convention)."""
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b)) / den if den > 0 else 0.0


def eq6_ids(X, bandwidth=0.5, k=6, p="max"):
    """Per-pair IDS_p (Eq. 12) on the rows of X as given; returns (P, P) with NaN diagonal."""
    X = np.asarray(X, float)
    P = X.shape[0]
    B = math.sqrt(1.0 / (2.0 * bandwidth))
    F = [eq6_features(X[i], B, k) for i in range(P)]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            R = np.abs(np.array([[eq11_corr(F[i][m], F[j][l]) for l in range(k)] for m in range(k)]))
            A[i, j] = R.max() if p == "max" else float(np.mean(R**p) ** (1.0 / p))
    return A


def kids(X, **kw):
    return nandiag(K.ids(X, **kw))


def ids_pair(a, b):
    """IDS-max of one pair (a, b given as already-prepared rows); kernel mirror."""
    return float(K.ids(np.stack([a, b]))[0, 1])


# ------------------------------------------------------------------ population value (quadrature)
def population_ids_gauss(rho, n_nodes=120, k=6):
    """IDS-max of (X, Y) standard bivariate normal with correlation rho, features of Eq. 6 with
    B = 1, by 2-D probabilists' Gauss-Hermite quadrature (X = u, Y = rho u + sqrt(1-rho^2) v)."""
    from numpy.polynomial.hermite_e import hermegauss
    t, w = hermegauss(n_nodes)
    w = w / w.sum()
    U, V = np.meshgrid(t, t, indexing="ij")
    W = np.outer(w, w)
    Xq, Yq = U, rho * U + math.sqrt(1 - rho**2) * V
    fx = eq6_features(Xq.ravel(), 1.0, k).reshape(k, *Xq.shape)
    fy = eq6_features(Yq.ravel(), 1.0, k).reshape(k, *Yq.shape)
    ex = np.array([np.sum(W * fx[m]) for m in range(k)])
    ey = np.array([np.sum(W * fy[m]) for m in range(k)])
    vx = np.array([np.sum(W * fx[m] ** 2) for m in range(k)]) - ex**2
    vy = np.array([np.sum(W * fy[m] ** 2) for m in range(k)]) - ey**2
    C = np.array([[(np.sum(W * fx[m] * fy[l]) - ex[m] * ey[l]) / math.sqrt(vx[m] * vy[l])
                   for l in range(k)] for m in range(k)])
    idx = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    return float(np.abs(C).max()), tuple(int(i) for i in idx)


# ------------------------------------------------------------------ Monte Carlo (own)
_MC = {}
KIND_ID = {"iid": 1, "gauss": 2, "walk": 3, "ar1": 4}


def mc(kind, T, n=N_MC, z=True, **kw):
    """Finite-sample IDS-max distribution of one pair under a generating model (cached)."""
    key = (kind, T, n, z, tuple(sorted(kw.items())))
    if key in _MC:
        return _MC[key]
    rng = np.random.default_rng([MC_SEED, T, KIND_ID[kind], int(1000 * kw.get("rho", 0)),
                                 int(1000 * kw.get("phi", 0)), int(z)])
    vals = np.empty(n)
    for r in range(n):
        if kind == "iid":
            x = rng.standard_normal((2, T))
        elif kind == "gauss":
            rho = kw["rho"]
            u, v = rng.standard_normal((2, T))
            x = np.stack([u, rho * u + math.sqrt(1 - rho**2) * v])
        elif kind == "walk":
            x = np.cumsum(rng.standard_normal((2, T)), axis=1)
        elif kind == "ar1":
            phi = kw["phi"]
            e = rng.standard_normal((2, T + 500))
            x = lfilter([1.0], [1.0, -phi], e, axis=1)[:, 500:]
        else:
            raise ValueError(kind)
        vals[r] = ids_pair(*(zscore1(x) if z else x))
    _MC[key] = vals
    return vals


_MCH = {}


def mc_hidden(T, z, coef, scale, eps, sup_g, n=300):
    """Finite-sample IDS-max (kernel mirror) and sample Corr(g(X), g(Y)) under the constructed
    dependent law phi(x)phi(y)(1 + eps g(x)g(y)); own rejection sampler (same law as
    probes/ids_probes.py:hidden_sample, independent seeds)."""
    key = (T, z, n)
    if key in _MCH:
        return _MCH[key]
    rng = np.random.default_rng([MC_SEED, T, 77, int(z)])
    bound = 1.0 + eps * sup_g**2
    ids_v, cg = np.empty(n), np.empty(n)
    for r in range(n):
        parts, got = [], 0
        while got < T:
            xy = rng.standard_normal((2, 4 * T))
            acc = (1.0 + eps * PR.hidden_g(xy[0], coef, scale) * PR.hidden_g(xy[1], coef, scale)) / bound
            keep = rng.uniform(size=4 * T) < acc
            parts.append(xy[:, keep])
            got += int(keep.sum())
        smp = np.concatenate(parts, axis=1)[:, :T]
        ids_v[r] = ids_pair(*(zscore1(smp) if z else smp))
        gs = PR.hidden_g(smp, coef, scale)
        cg[r] = np.corrcoef(gs)[0, 1]
    _MCH[key] = (ids_v, cg)
    return _MCH[key]


def perm_bound(a, b, m, seed):
    """Per-pair permutation (1 - 0.05/m) quantile of IDS-max (shuffle b; kernel mirror)."""
    rng = np.random.default_rng([MC_SEED, seed])
    vals = np.array([ids_pair(a, b[rng.permutation(b.size)]) for _ in range(N_PERM)])
    return float(np.quantile(vals, 1 - 0.05 / max(m, 1)))


# ------------------------------------------------------------------ io helpers
def nandiag(A):
    A = np.array(A, float, copy=True)
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        np.fill_diagonal(A, np.nan)
    return A


def pyspi_value(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], float)


def sym_max(A):
    if A is None or A.ndim != 2 or A.shape[0] != A.shape[1]:
        return float("nan")
    d = np.abs(A - A.T)
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def offd(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in kw.items():
        if v is None:
            continue
        arrays[k] = np.array(v) if isinstance(v, str) else np.asarray(v)
    np.savez(OUT_REF / f"{rid}.npz", **arrays)


def fw_factor(m):
    """se inflation so that |d| <= 2 se is a two-sided familywise 5 % test over m entries."""
    return float(ndtri(1 - 0.05 / (2 * max(m, 1))) / 2.0)


def raw_input(run):
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], float)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    return x


EXACT = dict(criterion="exact", rtol=1e-10, atol=1e-12)
KDETAIL = "refs/_kernels_basic_misc.py:ids (transform + explicit dot/norm Pearson + max over term blocks)"
EQ6DETAIL = "refs/ids.py:eq6_ids (Radhakrishnan et al. 2025 PNAS Eq. 6, 11, 12; per-pair loops)"


# ------------------------------------------------------------------ checks
def chk_kids(run, rid, tag):
    x = raw_input(run)
    Xp = prepare(x, run["path"])
    V = kids(Xp)
    S = pyspi_value(rid)
    extra = {}
    E = eq6_ids(Xp)
    extra["eq6_vs_kernel_maxabs"] = float(np.nanmax(np.abs(E - V)))
    if S is not None:
        extra["pyspi_sym_max"] = sym_max(S)
        extra["pyspi_offdiag_mean"] = float(np.nanmean(offd(S)))
    notes = "pyspi vs kernel mirror on the same prepared rows"
    if run["dataset"] == "scaled_offset" and run["path"] == "kernel":
        Vb = kids(zscore1(make_data.load("bivariate_gaussian", run["T"])["x"]))
        if S is not None:
            extra["kernel_minus_bivariate_zscored_maxabs"] = float(np.nanmax(np.abs(S - Vb)))
        notes += "; raw scaled rows (scale 0.2/2/5/10): IDS is not scale invariant at the kernel level"
    if run["dataset"] == "trending" and S is not None:
        extra["pairs_01_23_12_13"] = np.array([S[0, 1], S[2, 3], S[1, 2], S[1, 3]])
    save(rid, value=V, reference_kind="independent-impl", reference_detail=KDETAIL,
         notes=notes, **EXACT, **extra)


def chk_invariance(run, rid, tag):
    xb = make_data.load("bivariate_gaussian", run["T"])["x"]
    V = kids(zscore1(xb))
    save(rid, value=V, reference_kind="analytic",
         reference_detail="z-scoring removes scale/offset: ids(zscore(s x + c)) = ids(zscore(x)); "
                          "value = _kernels_basic_misc.ids(zscore(bivariate_gaussian))",
         notes="scaled_offset = scale * D2 + offset, compared with ids of the unscaled D2 rows", **EXACT)


def chk_signflip(run, rid, tag):
    x = raw_input(run)
    V = kids(prepare(x, run["path"]))          # UNflipped rows
    save(rid, value=V, reference_kind="analytic",
         reference_detail="sign-flip invariance (features odd/even in x, |rho| unchanged); value = "
                          "_kernels_basic_misc.ids of the unflipped prepared rows",
         notes="probe negates row 1 before pyspi", **EXACT)


def chk_identical(run, rid, tag):
    P = 4 if tag == "identical" else 2
    V = np.ones((P, P))
    np.fill_diagonal(V, np.nan)
    S = pyspi_value(rid)
    extra = {}
    if S is not None:
        extra["pyspi_minus_1_maxabs"] = float(np.nanmax(np.abs(S - 1.0)))
    save(rid, value=V, reference_kind="analytic",
         reference_detail="rows identical up to sign / positive affine map -> z-scored rows equal up to sign "
                          "-> the k = l term has |rho| = 1 -> IDS-max = 1",
         notes="identical: [x0, x0, -x0, 2.5 x0 - 1]; identical-pair: shared_envelope x2 is a bit copy of x0",
         **EXACT, **extra)


def chk_eq6(run, rid, tag):
    x = raw_input(run)
    Xp = prepare(x, run["path"])
    if tag.startswith("eq6-bw"):
        bw = float(tag[len("eq6-bw"):])
        V = eq6_ids(Xp, bandwidth=bw)
        extra = dict(bandwidth=bw, B_paper=math.sqrt(1 / (2 * bw)),
                     kernel_vs_eq6_maxabs=float(np.nanmax(np.abs(kids(Xp, bandwidth=bw) - V))))
        detail = EQ6DETAIL + f"; B = sqrt(1/(2*{bw}))"
    else:
        p = int(tag[len("pnorm"):])
        V = eq6_ids(Xp, p=p)
        extra = dict(p=p, kernel_vs_eq12_maxabs=float(np.nanmax(np.abs(kids(Xp, pnorm=p) - V))))
        detail = EQ6DETAIL + f"; Eq. 12 with p = {p}"
    save(rid, value=V, reference_kind="independent-impl", reference_detail=detail,
         notes="probe constructs InterDependenceScore with a non-config parameter", **EXACT, **extra)


def chk_minmax(run, rid, tag):
    x = raw_input(run)
    Xm = minmax08(x)
    V = eq6_ids(Xm, bandwidth=0.5)
    S = pyspi_value(rid)
    extra = {}
    if tag == "paper-minmax08":
        notes = ("EXPECTED TO FAIL: pyspi default (z-score, ddof=1) vs the paper's reported setting "
                 "(min-max [0, 8], B = 1, k = 6, IDS_inf); max_abs is the size of the preprocessing effect")
        if S is not None:
            extra["pyspi_pairs_01_23"] = np.array([S[0, 1], S[2, 3]]) if S.shape[0] >= 4 else np.array([S[0, 1]])
            extra["paper_pairs_01_23"] = np.array([V[0, 1], V[2, 3]]) if V.shape[0] >= 4 else np.array([V[0, 1]])
    else:
        notes = "probe min-max scales rows to [0, 8] and calls pyspi on the kernel path (no z-score)"
    save(rid, value=V, reference_kind="independent-impl",
         reference_detail=EQ6DETAIL + "; paper main text: min-max to [0, 8], B = 1, k = 6, IDS_inf",
         notes=notes, **EXACT, **extra)


def chk_truth(run, rid, tag):
    pair = (0, 1) if tag.endswith("p01") else (2, 3)
    rho = D2_RHO[pair]
    T = run["T"]
    vals = mc("gauss", T, rho=rho)
    pop, arg = population_ids_gauss(rho)
    V = np.array([[np.nan, vals.mean()], [vals.mean(), np.nan]])
    se = np.full((2, 2), vals.std(ddof=1))
    S = pyspi_value(rid)
    extra = dict(population_ids=pop, population_argmax_kl=np.array(arg), mc_mean=float(vals.mean()),
                 mc_sd=float(vals.std(ddof=1)), rho=rho, n_mc=N_MC)
    if S is not None:
        extra["pyspi_value"] = float(S[0, 1])
        extra["pyspi_minus_population"] = float(S[0, 1] - pop)
        extra["mc_mean_minus_population"] = float(vals.mean() - pop)
    save(rid, value=V, se=se, criterion="statistical", reference_kind="analytic",
         reference_detail="own MC of the finite-sample statistic (z-scored bivariate normal draws, "
                          "_kernels_basic_misc.ids) + population IDS-max by 2-D Gauss-Hermite (120 nodes)",
         notes="|pyspi - MC mean| <= 2 MC sd; population value reported as extra key", **extra)


def chk_null(run, rid, tag):
    T = run["T"]
    S = pyspi_value(rid)
    P = S.shape[0] if S is not None else 5
    m = P * (P - 1) // 2
    kind = "walk" if tag == "null-randomwalk" else "iid"
    vals = mc(kind, T)
    V = np.full((P, P), vals.mean())
    np.fill_diagonal(V, np.nan)
    se = np.full((P, P), vals.std(ddof=1) * fw_factor(m))
    extra = dict(mc_mean=float(vals.mean()), mc_sd=float(vals.std(ddof=1)),
                 mc_mean_times_sqrtT=float(vals.mean() * math.sqrt(T)),
                 mc_q95=float(np.quantile(vals, 0.95)), n_mc=N_MC, fw_factor=fw_factor(m))
    if S is not None:
        o = S[np.triu_indices(P, 1)]
        extra.update(pyspi_mean=float(o.mean()), pyspi_max=float(o.max()),
                     pyspi_mean_times_sqrtT=float(o.mean() * math.sqrt(T)),
                     pyspi_frac_above_q95=float(np.mean(o > np.quantile(vals, 0.95))))
    if kind == "walk":
        iid = mc("iid", T)
        extra["iid_mc_mean"] = float(iid.mean())
        extra["walk_over_iid_mean_ratio"] = float(vals.mean() / iid.mean())
        notes = "independent random walks (cointegrated x2, x3) vs MC of independent random walks (z-scored)"
    else:
        notes = "iid N(0,1) null: finite-sample positive bias of a max over 36 |corr|"
        if T == 1024:
            for phi in (0.5, 0.9, 0.99):
                a = mc("ar1", T, n=300, phi=phi)
                extra[f"ar1_phi{phi}_mc_mean"] = float(a.mean())
                extra[f"ar1_phi{phi}_mc_q95"] = float(np.quantile(a, 0.95))
    save(rid, value=V, se=se, criterion="statistical", reference_kind="independent-impl",
         reference_detail="own Monte Carlo (numpy default_rng) of _kernels_basic_misc.ids on independent "
                          f"{kind} pairs, z-scored ddof=1; se = MC sd x familywise factor over {m} pairs",
         notes=notes, **extra)


STRUCT = {  # dependent pairs (undirected) per dataset
    "nonlinear_dependence": [(0, 1), (2, 3)],
    "anm": [(0, 1), (2, 3)],
    "bivariate_gaussian": [(0, 1), (2, 3)],
    "trending": [(0, 1), (2, 3)],
    "delayed_noise": [],
}


def chk_structure(run, rid, tag):
    x = raw_input(run)
    Xp = prepare(x, run["path"])
    S = pyspi_value(rid)
    P = Xp.shape[0]
    dep = STRUCT[run["dataset"]]
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    ind = [p for p in pairs if p not in dep]
    bounds = {p: perm_bound(Xp[p[0]], Xp[p[1]], len(pairs), seed=100 * p[0] + p[1]) for p in pairs}
    ok_sym = sym_max(S) <= 1e-15
    ok_dep = all(S[p] > bounds[p] for p in dep)
    ok_ind = all(S[p] <= bounds[p] for p in ind)
    ok_nan = bool(np.isnan(np.diag(S)).all())
    flag = ok_sym and ok_dep and ok_ind and ok_nan
    extra = dict(sym_max=sym_max(S), dependent_pairs=np.array(dep if dep else [[-1, -1]]),
                 dependent_values=np.array([S[p] for p in dep]) if dep else np.array([np.nan]),
                 dependent_min_margin=float(min(S[p] - bounds[p] for p in dep)) if dep else float("nan"),
                 independent_max=float(max(S[p] for p in ind)),
                 independent_max_over_bound=float(max(S[p] / bounds[p] for p in ind)),
                 perm_bound_median=float(np.median(list(bounds.values()))),
                 ok_sym=float(ok_sym), ok_dep=float(ok_dep), ok_ind=float(ok_ind))
    if run["dataset"] == "trending":
        Sp = pyspi_value(rid.replace("|pipeline-detrend|structure", "|pipeline|kids"))
        if Sp is not None:
            extra["no_detrend_independent_max"] = float(max(Sp[p] for p in ind))
    save(rid, value=np.nan, criterion="qualitative", pass_flag=int(flag), reference_kind="analytic",
         reference_detail="known dependence structure of the dataset + per-pair permutation bound "
                          f"(own, {N_PERM} shuffles, quantile 1 - 0.05/{len(pairs)}, _kernels_basic_misc.ids)",
         notes="pass: max|A-A^T| <= 1e-15, NaN diagonal, dependent pairs above and independent pairs below "
               "their permutation bound" + ("; delayed_noise: lagged copies only, so every pair independent "
                                           "at lag 0" if not dep else ""), **extra)


def chk_hidden(run, rid, tag):
    T = run["T"]
    coef, scale, eps, sup_g, ortho_gh = PR.hidden_construction()
    # independent verification of the orthogonality by a fine trapezoid rule (not Gauss-Hermite)
    xs = np.linspace(-16, 16, 640001)
    phi = np.exp(-0.5 * xs**2) / math.sqrt(2 * math.pi)
    g = PR.hidden_g(xs, coef, scale)
    F = eq6_features(xs, 1.0)
    ortho_trap = float(max(abs(np.trapezoid(phi * g * F[m], xs)) for m in range(6)))
    mean_trap = float(np.trapezoid(phi * g, xs))
    norm_trap = float(np.trapezoid(phi * g * g, xs))
    smp = PR.hidden_sample(T, run.get("seed", 42))
    gs = PR.hidden_g(smp, coef, scale)
    r_g = float(np.corrcoef(gs)[0, 1])
    S = pyspi_value(rid)
    z = run["path"] == "pipeline"
    Xp = zscore1(smp) if z else smp
    vals = mc("iid", T, z=z)
    hid = mc_hidden(T, z, coef, scale, eps, sup_g)
    V = np.array([[np.nan, vals.mean()], [vals.mean(), np.nan]])
    se = np.full((2, 2), vals.std(ddof=1))
    extra = dict(eps=eps, sup_g=sup_g, ortho_gauss_hermite=ortho_gh, ortho_trapezoid=ortho_trap,
                 mc_hidden_mean=float(hid[0].mean()), mc_hidden_sd=float(hid[0].std(ddof=1)),
                 mc_hidden_q95=float(np.quantile(hid[0], 0.95)), mc_hidden_n=int(hid[0].size),
                 mc_hidden_corr_g_mean=float(hid[1].mean()), mc_null_sd=float(vals.std(ddof=1)),
                 mc_hidden_minus_null_mean_z=float((hid[0].mean() - vals.mean())
                                                   / math.sqrt(hid[0].var(ddof=1) / hid[0].size
                                                               + vals.var(ddof=1) / vals.size)),
                 g_mean_trapezoid=mean_trap, g_norm2_trapezoid=norm_trap,
                 population_ids=0.0, population_corr_gX_gY=eps * norm_trap,
                 sample_corr_gX_gY=r_g, sample_corr_gX_gY_z=r_g * math.sqrt(T),
                 kernel_mirror_on_sample=ids_pair(*Xp), mc_null_mean=float(vals.mean()),
                 mc_null_q95=float(np.quantile(vals, 0.95)))
    if S is not None:
        extra["pyspi_value"] = float(S[0, 1])
        extra["pyspi_minus_kernel_mirror"] = float(S[0, 1] - ids_pair(*Xp))
    save(rid, value=V, se=se, criterion="statistical", reference_kind="analytic",
         reference_detail="density phi(x)phi(y)(1 + eps g(x)g(y)) with g orthogonal to 1 and to the six "
                          "Eq. 6 features (B = 1) under N(0,1): population Corr(phi_k(X), phi_l(Y)) = 0 for all "
                          "k, l, so population IDS_p = 0 although Corr(g(X), g(Y)) = eps; probes/ids_probes.py:"
                          "hidden_construction; null MC of iid N(0,1) pairs",
         notes="pass = pyspi indistinguishable from the independence null although the pair is dependent", **extra)


def chk_pnorm_unsupported(run, rid, tag):
    S = pyspi_value(rid)
    flag = 0
    extra = {}
    if S is not None and S.shape == (len(PR.PNORM_CASES), len(PR.PNORM_FIELDS)):
        rows = {f"{pn}_P{P}": S[i] for i, (pn, P) in enumerate(PR.PNORM_CASES)}
        ok = []
        for (pn, P), r in zip(PR.PNORM_CASES, S):
            d = dict(zip(PR.PNORM_FIELDS, r))
            exp_err = "err_ValueError" if P != 6 else "err_AssertionError"
            ok.append(d["table_block_all_nan"] == 1 and d["report_failed"] == 1 and d[exp_err] == 1
                      and d["direct_ndim"] == 4)
        flag = int(all(ok))
        extra = {k.replace("'", ""): v for k, v in rows.items()}
        extra["fields"] = ",".join(PR.PNORM_FIELDS)
    save(rid, value=np.nan, criterion="qualitative", pass_flag=flag, reference_kind="analytic",
         reference_detail="numpy_dependence.py:83-92 handles only 'max', 1, 2; expected: compute_IDS returns a "
                          "(6, P, 6, P) array; Calculator.compute fill_diagonal (calculator.py:332) raises ValueError "
                          "for P != 6, the table store (calculator.py:335) raises AssertionError for P = 6; the except "
                          "(336-339) blanks the block and the MetricReport status is failed (342-346)",
         notes="pass = traced failure mode reproduced for pnorm 3 and '2' at P = 4 and 6", **extra)


def _rs(seed, n=2):
    return np.random.RandomState(seed).random_sample(n)


def chk_rng(run, rid, tag):
    S = pyspi_value(rid)
    r123, r1717, r5 = _rs(123), _rs(1717), _rs(5)
    extra = {}
    flag = 0
    if tag == "rng-import":
        d = dict(zip(PR.RNG_IMPORT_KEYS, S))
        checks = dict(
            baseline_seed123=d["draw_seed123_no_import"] == r123[0],
            pyspi_calculator_import_no_reseed=d["draw_after_import_pyspi_calculator_only"] == r123[0],
            reseeded_to_1717=d["draw_after_import_seeded123"] == r1717[0],
            state_eq_1717=d["state_eq_seed1717_after_import"] == 1.0,
            ids_consumes_no_rng=d["draw_after_ids_compute"] == r1717[1],
            ids_deterministic=d["ids_diff_seed1_vs_seed2"] == 0.0)
        extra.update(d)
    elif tag == "rng-calculator":
        d = dict(zip(PR.RNG_CALC_KEYS, S))
        checks = dict(
            first_Calculator_reseeds_1717=d["draw_after_first_Calculator_seeded123"] == r1717[0],
            state_eq_1717=d["state_eq_seed1717_after_first_Calculator"] == 1.0,
            second_Calculator_keeps_user_seed=d["draw_after_second_Calculator_seeded5"] == r5[0])
        extra.update(d)
    else:
        d = dict(zip(PR.RNG_CROSS_KEYS, S))
        checks = dict(
            user_seed_before_Calculator_ignored=(d["mcd_with_ids_seed1_before"] == d["mcd_with_ids_seed2_before"]
                                                 == d["mcd_noids_seed1717_before"]),
            mcd_depends_on_seed_without_ids=d["mcd_noids_seed1_before"] != d["mcd_noids_seed2_before"],
            ids_compute_leaves_stream=d["mcd_idsfirst_seed7_after"] == d["mcd_noids_seed7_after"])
        extra.update(d)
        extra["mcd_spread_noids_seed1_vs_seed2"] = float(abs(d["mcd_noids_seed1_before"] - d["mcd_noids_seed2_before"]))
    flag = int(all(checks.values()))
    extra.update({f"ok_{k}": float(v) for k, v in checks.items()})
    extra.update(rs123_0=r123[0], rs1717_0=r1717[0], rs1717_1=r1717[1], rs5_0=r5[0])
    save(rid, value=np.nan, criterion="qualitative", pass_flag=flag, reference_kind="analytic",
         reference_detail="legacy MT19937 stream np.random.RandomState(seed).random_sample (deterministic in "
                          "the seed); probes/ids_probes.py runs each case in a fresh interpreter",
         notes="pass = the traced side effect is reproduced exactly (pyspi/lib/ids/numpy_dependence.py:11-12 "
               "reseeds the global RNG to 1717 on first import); it is not a statement that the behaviour is "
               "desirable", **extra)


def chk_symmetry(run, rid, tag, manifest):
    worst, n_sq, n_nonzero, worst_rid = 0.0, 0, 0, ""
    for r in manifest["runs"]:
        S = pyspi_value(r["run_id"])
        if S is None or S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 2:
            continue
        n_sq += 1
        s = sym_max(S)
        n_nonzero += s > 0
        if s > worst:
            worst, worst_rid = s, r["run_id"]
    flag = int(worst <= 1e-15)
    save(rid, value=np.nan, criterion="qualitative", pass_flag=flag, reference_kind="analytic",
         reference_detail="undirected: A = A^T; no mirroring (misc.py:194-200), symmetry from np.corrcoef",
         notes="pass = max|A - A^T| <= 1e-15 over every square pyspi output of the family",
         sym_max_all=worst, n_square_outputs=n_sq, n_not_bitwise_symmetric=int(n_nonzero),
         worst_run=worst_rid)


def main():
    manifest = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_err = 0
    for run in manifest["runs"]:
        rid = run["run_id"]
        tag = rid.split("|")[-1]
        try:
            if tag == "kids":
                chk_kids(run, rid, tag)
            elif tag == "invariance":
                chk_invariance(run, rid, tag)
            elif tag == "signflip":
                chk_signflip(run, rid, tag)
            elif tag in ("identical", "identical-pair"):
                chk_identical(run, rid, tag)
            elif tag.startswith("eq6-bw") or tag.startswith("pnorm") and tag != "pnorm-unsupported":
                chk_eq6(run, rid, tag)
            elif tag in ("minmax08", "paper-minmax08"):
                chk_minmax(run, rid, tag)
            elif tag.startswith("truth"):
                chk_truth(run, rid, tag)
            elif tag.startswith("null"):
                chk_null(run, rid, tag)
            elif tag in ("structure", "zero-lag"):
                chk_structure(run, rid, tag)
            elif tag == "hidden-dependence":
                chk_hidden(run, rid, tag)
            elif tag == "pnorm-unsupported":
                chk_pnorm_unsupported(run, rid, tag)
            elif tag.startswith("rng-"):
                chk_rng(run, rid, tag)
            elif tag == "symmetry":
                continue
            else:
                raise ValueError(f"unknown tag {tag}")
            n_ok += 1
        except Exception as exc:
            n_err += 1
            import traceback
            print(f"ERROR {rid}: {exc!r}\n{traceback.format_exc()}", file=sys.stderr)
    for run in manifest["runs"]:          # after everything else: reads every pyspi output of the family
        if run["run_id"].endswith("|symmetry"):
            chk_symmetry(run, run["run_id"], "symmetry", manifest)
            n_ok += 1
    print(f"ids refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
