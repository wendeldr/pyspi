"""References for the cdt_pairwise family (causal.anm, causal.cds, causal.reci, causal.igci): COMPARISON venv.

Writes ``audit/checks/out/ref/<run_id>.npz`` for every run of ``manifest/cdt_pairwise.json`` (contract in
``compare.py``). Building blocks are the self-tested kernels in ``_kernels_distance.py`` (cdt_fast_hsic_gamma,
anm_replica, gp_fit_hyperparameters, gp_posterior_mean, cds_ref, reci_ref, igci_ref; none imports cdt) plus the
from-scratch canonical references defined here:

* ``gp_noise_fit`` / ``anm_canonical``: Hoyer, Janzing, Mooij, Peters & Schoelkopf 2009 (NIPS 21) Sec. 4-5 ANM
  procedure: GP regression with a Gaussian (RBF) kernel PLUS independent Gaussian noise (C*RBF + White), type-II
  ML hyperparameters (L-BFGS-B, R&W 2006 eq. 5.9 gradient), residual = y - posterior mean of f, then the HSIC
  statistic m*HSIC_b on all m samples (no 200-point subsample) with the Gretton et al. 2008 gamma p-value
  (``hsic_gamma_test``, written from memory of hsicTestGamma.m; the decision uses the statistic, lower = causal).
* ``reci_canonical``: Bloebaum et al. 2018 (AISTATS) Sec. 4-5 as recalled: both variables min-max scaled, MON family
  a*x^n + b with n in 2..9 selected on the training half, test MSE averaged over 20 random 50/50 splits.
* ``cds_variant``: Fonollosa 2016 Sec. 2.1/2.3 with sfactor = 3 and the sample variance (ddof = 1).

pyspi-side inputs: when a check isolates the estimator (anm replica runs), the reference reads the exact array
pyspi fed to the estimator from ``out/pyspi/<run_id>.npz['input']`` (anm amplifies last-bit input changes by ~1e7,
see the scipy118-drift run); every other run preprocesses the raw dataset itself (scipy.stats.zscore ddof=1, scipy
signal.detrend), and the max |difference| to pyspi's input is stored as ``input_max_abs_vs_pyspi``.
The sklearn GaussianProcessRegressor is imported only in the ``scipy118-drift`` run (labelled same-lineage).
Runs that compare pyspi with pyspi (igci ``antisym``: value = -A.T of pyspi's own matrix; anm ``random-state``:
determinism of pyspi's score) and the static code-property probes (anm ``patch-global``/``labels``, igci ``labels``:
pyspi introspection vs a hard-coded expected flag vector) are labelled same-lineage; the ``invariance`` runs of cds
and reci evaluate the independent replica on the unscaled draw and are labelled independent-impl.
"""
import json
import os
import sys
import warnings
from pathlib import Path

# The check run's pyspi GP fits are reproduced bit for bit only with a single-thread OpenBLAS (pyspi/__init__.py:4 sets
# OMP_NUM_THREADS=1 for BLAS libraries loaded after pyspi); anm amplifies the last-bit differences of a threaded BLAS to
# ~1e-6 (see notes/cdt_pairwise.md), so the reference sets one thread before numpy/scipy are imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_distance as kd  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.signal import detrend  # noqa: E402
from scipy.stats import gamma as gamma_dist  # noqa: E402
from scipy.stats import zscore  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
MANIFEST = CHECKS / "manifest" / "cdt_pairwise.json"
ME = "refs/cdt_pairwise.py"
ALPHA = 1e-10                         # sklearn GaussianProcessRegressor default jitter (_gpr.py:215)
LOG_B = (np.log(1e-5), np.log(1e5))   # sklearn default hyperparameter bounds (kernels.py:1236, 1511)
TRUTH = {"anm": [(0, 1), (2, 3)], "var2_bivariate": [(0, 1)], "logistic_maps_chaotic": [(0, 1)]}

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------- helpers
def raw(run):
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
    return x if run.get("processes") is None else x[list(run["processes"])]


def prep(x, path):
    """pyspi Data.set_data (data.py:184-195): optional scipy.signal.detrend(axis=1), then zscore(ddof=1)."""
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return x
    if path == "pipeline-detrend":
        x = detrend(x, axis=1)
    return zscore(x, axis=1, nan_policy="omit", ddof=1)


def pyspi(rid):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64), np.asarray(z["input"], dtype=np.float64)


def pairwise(f, z):
    P = z.shape[0]
    B = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                B[i, j] = f(z[i], z[j])
    return B


def truth_sign(run, P):
    s = np.zeros((P, P))
    procs = run.get("processes") or list(range(P))
    for c, e in TRUTH.get(run["dataset"], []):
        if c in procs and e in procs:
            s[procs.index(c), procs.index(e)], s[procs.index(e), procs.index(c)] = 1, -1
    return s


def truth_agreement(D, sign):
    m = sign != 0
    return float(np.mean(np.sign(D[m]) == sign[m])) if m.any() else float("nan")


def zs(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / v.std()


def relmax(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m]))) if m.any() else 0.0


def write(rid, value, criterion, kind, detail, notes="", **extra):
    d = dict(value=np.asarray(value, dtype=np.float64), criterion=np.array(criterion),
             reference_kind=np.array(kind), reference_detail=np.array(detail), notes=np.array(notes))
    for k, v in extra.items():
        d[k] = np.asarray(v)
    OUT_REF.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_REF / f"{rid}.npz", **d)


# ------------------------------------------------------------------------------- anm references
def anm_residual(x, y, c, l):
    """r = posterior mean - y for the sklearn-default GP (C*RBF, alpha 1e-10) at fixed (c, l) (kd.gp_posterior_mean)."""
    x, y = np.ravel(x), np.ravel(y)
    return kd.gp_posterior_mean(x, y, c, l, ALPHA) - y


def lml_crbf(x, y, c, l):
    """log marginal likelihood of the sklearn-default model C*RBF + alpha I (R&W 2006 eq. 5.8), sklearn RBF arithmetic."""
    x, y = np.ravel(x), np.ravel(y)
    cf = cho_factor(c * np.exp(-0.5 * kd._sqdiff_scaled(x, l)) + ALPHA * np.eye(len(x)), lower=True)
    return float(-0.5 * y @ cho_solve(cf, y) - np.log(np.diag(cf[0])).sum() - 0.5 * len(x) * np.log(2 * np.pi))


def pyspi_ref(rid):
    """The reference value already written for another run (used to cross-reference sibling runs)."""
    with np.load(OUT_REF / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def anm_ownfit(x, y):
    c, l = kd.gp_fit_hyperparameters(np.ravel(x), np.ravel(y), ALPHA)
    return c, l, anm_residual(x, y, c, l)


def gp_noise_fit(x, y):
    """Type-II ML for k = c exp(-(x-x')^2 / (2 l^2)) + w delta (sklearn-style bounds 1e-5..1e5, + 1e-10 jitter);
    L-BFGS-B on log(c, l, w) from two deterministic starts (w = 1 and w = 0.1), best log marginal likelihood kept.
    Returns (c, l, w, residual y - f_hat) with f_hat = K_f (K_f + (w + jitter) I)^-1 y."""
    x, y = np.ravel(x), np.ravel(y)
    n = len(x)
    S = (x[:, None] - x[None, :]) ** 2

    def nlml(th):
        c, l, w = np.exp(th)
        Kf = c * np.exp(-0.5 * S / l ** 2)
        cf = cho_factor(Kf + (w + ALPHA) * np.eye(n), lower=True)
        a = cho_solve(cf, y)
        W = np.outer(a, a) - cho_solve(cf, np.eye(n))
        g = -0.5 * np.array([np.sum(W * Kf), np.sum(W * Kf * S / l ** 2), w * np.trace(W)])
        f = 0.5 * y @ a + np.log(np.diag(cf[0])).sum() + 0.5 * n * np.log(2 * np.pi)
        return f, g
    best = None
    for start in ([0.0, 0.0, 0.0], [0.0, 0.0, np.log(0.1)]):
        r = minimize(nlml, np.array(start), jac=True, method="L-BFGS-B", bounds=[LOG_B] * 3)
        if best is None or r.fun < best.fun:
            best = r
    c, l, w = np.exp(best.x)
    Kf = c * np.exp(-0.5 * S / l ** 2)
    f_hat = Kf @ cho_solve(cho_factor(Kf + (w + ALPHA) * np.eye(n), lower=True), y)
    return c, l, w, y - f_hat


def hsic_gamma_test(X, Y):
    """m*HSIC_b with cdt/Gretton median-heuristic bandwidths and the gamma-approximation p-value
    (Gretton et al. 2008, NIPS 20, hsicTestGamma.m as recalled). Returns (statistic, p_value)."""
    X, Y = np.ravel(X), np.ravel(Y)
    m = len(X)

    def rbf(v):
        H = (v[:, None] - v[None, :]) ** 2
        d = H[np.triu_indices(m, 1)]
        deg2 = 0.5 * np.median(d[d > 0])
        return np.exp(-H / (2 * deg2))
    K, L = rbf(X), rbf(Y)
    Kc, Lc = kd._double_centre(K), kd._double_centre(L)
    stat = np.sum(Kc * Lc) / m
    v = (Kc * Lc / 6) ** 2
    var = (v.sum() - np.trace(v)) / m / (m - 1)
    var = 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3) * var
    K0, L0 = K - np.diag(np.diag(K)), L - np.diag(np.diag(L))
    muX, muY = K0.sum() / m / (m - 1), L0.sum() / m / (m - 1)
    mH = (1 + muX * muY - muX - muY) / m
    al, bet = mH ** 2 / var, var * m / mH
    return float(stat), float(gamma_dist.sf(stat, al, scale=bet))


def anm_canonical(x, y):
    """Hoyer et al. 2009 score for x -> y: (statistic, p-value) of HSIC(residual, x), full sample."""
    _, _, _, r = gp_noise_fit(x, y)
    return hsic_gamma_test(zs(r), zs(x))


# ------------------------------------------------------------------------- cds / reci references
def cds_variant(x, y, ffactor=3, ddof=1, maxdev=3, minc=12):
    """Fonollosa 2016 Sec. 2.1/2.3 with sfactor = ffactor (paper: 3) and var_x with ddof (paper: 'sample variance'):
    CDS = sqrt(mean_k var_x(P[:, k])), same trimmed standardisation and minc as cdt (kd.cds_ref)."""
    x, y = np.ravel(x), np.ravel(y)
    xd = kd._cds_discretize(x, ffactor, maxdev)
    vals = np.arange(-ffactor * maxdev, ffactor * maxdev + 1)
    pyx = []
    for a in np.unique(xd):
        sel = xd == a
        if sel.sum() > minc:
            yx = kd._cds_discretize((y[sel] - y[sel].mean()) / y.std(), ffactor, maxdev, norm=False)
            p = np.array([(yx == v).sum() for v in vals], float)
            pyx.append(p / p.sum())
    if len(pyx) <= ddof:
        return 0.0
    pyx = np.array(pyx)
    return float(np.sqrt(np.mean(np.var(pyx, axis=0, ddof=ddof))))


def cds_bins(x, ffactor=2, maxdev=3, minc=12):
    """Qualifying x-bins (count > minc) of cdt's discretisation of x (CDS.py:164)."""
    xd = kd._cds_discretize(np.ravel(x), ffactor, maxdev)
    u, c = np.unique(xd, return_counts=True)
    return xd, [a for a, n in zip(u, c) if n > minc]


def cds_pmfs(x, y, ffactor=2, maxdev=3, minc=12):
    xd, q = cds_bins(x, ffactor, maxdev, minc)
    vals = np.arange(-ffactor * maxdev, ffactor * maxdev + 1)
    out = []
    for a in q:
        sel = xd == a
        yx = kd._cds_discretize((y[sel] - y[sel].mean()) / y.std(), ffactor, maxdev, norm=False)
        out.append(np.array([(yx == v).sum() for v in vals], float))
    return out


def reci_canonical(x, y, n_splits=20, seed=0):
    """Bloebaum et al. 2018 Sec. 4-5 (recalled): min-max scale, MON family a*x^n + b (n = 2..9) chosen by training
    MSE on a random half, test MSE on the other half, averaged over n_splits splits."""
    s = lambda v: (v - v.min()) / (v.max() - v.min())
    x, y = s(np.ravel(x)), s(np.ravel(y))
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        tr, te = p[: n // 2], p[n // 2:]
        best = None
        for k in range(2, 10):
            Z = np.column_stack([np.ones(n), x ** k])
            beta = np.linalg.lstsq(Z[tr], y[tr], rcond=None)[0]
            mtr = np.mean((Z[tr] @ beta - y[tr]) ** 2)
            if best is None or mtr < best[0]:
                best = (mtr, np.mean((Z[te] @ beta - y[te]) ** 2))
        out.append(best[1])
    return float(np.mean(out))


# ------------------------------------------------------------------------------- inline inputs
def spike_input(T=512, seed=2026):
    """Mirror of probes/cdt_pairwise_probes.py:spike_input (same construction, same generator)."""
    rng = np.random.default_rng(seed)
    x = 1e-3 * rng.standard_normal(T)
    levels = np.repeat(np.array([-6.0, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]), 2)
    pos = rng.choice(T, 24, replace=False)
    x[pos] = levels * (1 + 0.01 * rng.standard_normal(24))
    return np.vstack([x, rng.standard_normal(T)])


def cubic_pair(T, seed):
    """Mirror of probes/cdt_pairwise_probes.py:cubic_pair: X ~ U(0, 1), Y = X^3."""
    u = np.random.default_rng(seed).uniform(0.0, 1.0, T)
    return np.vstack([u, u ** 3])


IGCI_CUBIC = -((np.log(3) - 2) + np.log(1 / np.sqrt(12)) - 0.5 * np.log(1 / 7 - 1 / 16))   # -C^G = 0.88320


# ======================================================================================== runs
def run_anm(run, tag):
    rid, T = run["run_id"], run["T"]
    S, inp = pyspi(rid)
    P = S.shape[-1]
    if tag == "theta":
        z = prep(raw(run), run["path"])
        V = np.full((2, P, P), np.nan)
        gap = np.full((P, P), np.nan)
        for i in range(P):
            for j in range(P):
                if i != j:
                    V[:, i, j] = np.log(kd.gp_fit_hyperparameters(z[i], z[j], ALPHA))
                    gap[i, j] = lml_crbf(z[i], z[j], *np.exp(V[:, i, j])) - lml_crbf(z[i], z[j], *np.exp(S[:, i, j]))
        d = np.abs(S - V)
        write(rid, V, "tolerance", "independent-impl", f"{ME}:run_anm[theta] -> _kernels_distance.gp_fit_hyperparameters "
              "(numpy/scipy L-BFGS-B on the R&W 2006 eq. 5.9 gradient, start log theta = 0, bounds 1e-5..1e5)",
              "log theta = (log constant, log length_scale) of sklearn's fit vs an independent fit of the same model; "
              "atol 1e-3 in log space (0.1 % in the parameter): both optimisers stop on L-BFGS-B's default ftol, so end "
              "points agree only to the flatness of the objective", rtol=0.0, atol=1e-3,
              max_abs_dlog_constant=np.nanmax(d[0]), max_abs_dlog_length_scale=np.nanmax(d[1]),
              pyspi_max_length_scale=float(np.nanmax(np.exp(S[1]))),
              pyspi_n_length_scale_at_lower_bound=int(np.sum(np.exp(S[1]) <= 1e-5 * (1 + 1e-9))),
              lml_ref_minus_sklearn_max=float(np.nanmax(gap)), lml_ref_minus_sklearn_min=float(np.nanmin(gap)),
              n_pairs_ref_lml_better_by_1e3=int(np.nansum(gap > 1e-3)))
    elif tag == "replica-theta":
        th, _ = pyspi(rid.replace("replica-theta", "theta"))
        V = pairwise(lambda a, b: np.nan, inp)
        Vp = V.copy()
        zr = prep(raw(run), run["path"])
        for i in range(P):
            for j in range(P):
                if i != j:
                    c, l = np.exp(th[:, i, j])
                    V[i, j] = kd.anm_replica(inp[i], inp[j], c, l)
                    Vp[i, j] = kd.anm_replica(zr[i], zr[j], c, l)
        write(rid, V, "exact", "independent-impl", f"{ME}:run_anm[replica-theta] -> _kernels_distance.anm_replica at "
              "sklearn's fitted theta (exported by the sibling |theta probe run), on pyspi's own estimator input",
              "isolates GP posterior mean + cdt FastHsicTestGamma from the optimiser; replica imports neither sklearn nor cdt",
              input_max_abs_vs_pyspi=float(np.abs(zr - inp).max()), ref_prep_max_rel=relmax(Vp, V))
    elif tag == "replica-projectenv":
        A, _ = pyspi(rid.replace("replica-projectenv", "replica-theta"))
        write(rid, A, "exact", "independent-impl", "probes/cdt_pairwise_probes.py:anm_replica_projectenv -> "
              "_kernels_distance.anm_replica run in the project venv at sklearn's theta; value = pyspi's SPI matrix from "
              "the sibling |replica-theta run", "same replica as |replica-theta, but evaluated with pyspi's own scipy 1.17.1 "
              "LAPACK and thread setting; separates replica arithmetic from the comparison venv's scipy 1.18.1 build",
              comparison_venv_replica_max_rel=relmax(pyspi_ref(rid.replace("replica-projectenv", "replica-theta")), A))
    elif tag == "replica-ownfit":
        V = pairwise(lambda a, b: kd.anm_replica(a, b, *kd.gp_fit_hyperparameters(a.ravel(), b.ravel(), ALPHA)), inp)
        th_rid = rid.replace("replica-ownfit", "theta")
        lml_note = {}
        if (OUT_PY / f"{th_rid}.npz").is_file():
            th, _ = pyspi(th_rid)
            g = pairwise(lambda a, b: np.nan, inp)
            for i in range(P):
                for j in range(P):
                    if i != j:
                        g[i, j] = (lml_crbf(inp[i], inp[j], *kd.gp_fit_hyperparameters(inp[i], inp[j], ALPHA))
                                   - lml_crbf(inp[i], inp[j], *np.exp(th[:, i, j])))
            k = np.unravel_index(np.nanargmax(np.abs(S - V) / S), S.shape)
            lml_note = dict(lml_ref_minus_sklearn_max=float(np.nanmax(g)), worst_pair=list(map(int, k)),
                            lml_ref_minus_sklearn_worst_pair=float(g[k]))
        write(rid, V, "tolerance", "independent-impl", f"{ME}:run_anm[replica-ownfit] -> _kernels_distance.anm_replica with "
              "its own gp_fit_hyperparameters (no sklearn, no cdt)", "rtol 1e-5: the score moves ~1e-6 relative for a "
              "one-ulp change of the fitted constant (see the scipy118-drift run), so optimiser end points that agree "
              "to ~1e-5 in log theta cannot give better agreement", rtol=1e-5, atol=0.0, max_rel=relmax(S, V), **lml_note)
    elif tag == "collapse":
        z = prep(raw(run), run["path"])
        V = np.full((3, P, P), np.nan)
        for i in range(P):
            for j in range(P):
                if i != j:
                    c, l, r = anm_ownfit(z[i], z[j])
                    y = z[j]
                    V[:, i, j] = [l, r.std() / y.std(), np.median(r / (-(ALPHA / c) * y))]
        ok = bool(np.nanmax(S[1]) < 1e-8 and np.nanmax(V[1]) < 1e-8)
        write(rid, V, "qualitative", "independent-impl", f"{ME}:run_anm[collapse] -> own C*RBF fit (alpha 1e-10, no noise "
              "kernel), residual = gp_posterior_mean - y", "pass = the residual collapses (std(r)/std(y) < 1e-8) both in "
              "sklearn's fit (probe) and in the independent fit, i.e. the collapse is a property of the model "
              "specification, not of sklearn", pass_flag=int(ok),
              pyspi_max_length_scale=float(np.nanmax(S[0])), ref_max_length_scale=float(np.nanmax(V[0])),
              pyspi_max_resid_ratio=float(np.nanmax(S[1])), ref_max_resid_ratio=float(np.nanmax(V[1])),
              pyspi_median_ratio_range=[float(np.nanmin(S[2])), float(np.nanmax(S[2]))],
              ref_median_ratio_range=[float(np.nanmin(V[2])), float(np.nanmax(V[2]))])
    elif tag == "rawhsic":
        H = pairwise(lambda a, b: kd.cdt_fast_hsic_gamma(zs(b), zs(a)), inp)
        m = ~np.isnan(S)
        rel = np.abs(S[m] - H[m]) / H[m]
        ok = bool(np.median(rel) < 0.1 and rel.max() < 0.3)
        extra = dict(median_rel=float(np.median(rel)), max_rel=float(rel.max()),
                     pyspi_max_abs_asym=float(np.nanmax(np.abs(S - S.T))),
                     pyspi_max_rel_asym=float(np.nanmax(np.abs(S - S.T) / np.abs(S))))
        if run["dataset"] == "bivariate_gaussian":
            k = (0, 1) if P == 2 else (2, 3)
            extra.update(pyspi_A_rho08_pair=float(S[k]), ref_H_rho08_pair=float(H[k]), m_used=min(T, 200))
        write(rid, H, "qualitative", "independent-impl", f"{ME}:run_anm[rawhsic] -> _kernels_distance.cdt_fast_hsic_gamma "
              "(m*HSIC_b, 200-point stride subsample, median heuristic) of the standardised raw pair, no regression",
              "pass = median |A - H|/H < 0.1 and max < 0.3 (the record's criterion): anm behaves as a dependence score "
              "between z_i and z_j", pass_flag=int(ok), **extra)
    elif tag == "direction":
        z = prep(raw(run), run["path"])
        B, Pv = np.full((P, P), np.nan), np.full((P, P), np.nan)
        for i in range(P):
            for j in range(P):
                if i != j:
                    B[i, j], Pv[i, j] = anm_canonical(z[i], z[j])
        sign = truth_sign(run, P)
        D = B.T - B
        extra = dict(ref_truth_agreement=truth_agreement(D, sign), pyspi_truth_agreement=truth_agreement(S, sign))
        procs = run.get("processes") or list(range(P))
        for c, e in TRUTH["anm"]:
            if c in procs and e in procs:
                a, b = procs.index(c), procs.index(e)
                extra.update({f"ref_score_{c}to{e}": B[a, b], f"ref_score_{e}to{c}": B[b, a],
                              f"ref_p_{c}to{e}": Pv[a, b], f"ref_p_{e}to{c}": Pv[b, a],
                              f"pyspi_D_{c}{e}": S[a, b]})
        write(rid, D, "direction", "independent-impl", f"{ME}:run_anm[direction] -> anm_canonical (Hoyer et al. 2009: "
              "GP C*RBF + White by type-II ML, full-sample HSIC + gamma p-value); value = B.T - B",
              "pyspi value = A.T - A from the probe (cdt predict_proba form; > 0 means i -> j); expected sign = ground "
              "truth (x0 -> x1 cubic, x2 -> x3 tanh)", expected_sign=sign, **extra)
    elif tag == "subsample":
        s200, sfull, can200, canfull = (np.full((P, P), np.nan) for _ in range(4))
        for i in range(P):
            for j in range(P):
                if i != j:
                    c, l, r = anm_ownfit(inp[i], inp[j])
                    s200[i, j] = kd.cdt_fast_hsic_gamma(zs(r), zs(inp[i]))
                    sfull[i, j] = kd.cdt_fast_hsic_gamma(zs(r), zs(inp[i]), maxpnt=T)
                    rc = gp_noise_fit(inp[i], inp[j])[3]
                    can200[i, j] = kd.cdt_fast_hsic_gamma(zs(rc), zs(inp[i]))
                    canfull[i, j] = kd.cdt_fast_hsic_gamma(zs(rc), zs(inp[i]), maxpnt=T)
        ok = relmax(S, s200) < 1e-5
        write(rid, s200, "qualitative", "independent-impl", f"{ME}:run_anm[subsample] -> own-fit residual, "
              "cdt_fast_hsic_gamma with maxpnt=200 (pyspi) and maxpnt=T (full sample); canonical C*RBF+White residual "
              "likewise", "pass = the 200-point replica reproduces pyspi (rel < 1e-5); the subsample effect is reported "
              "in the extra keys (full-sample statistic is m*HSIC_b with m = T, so it is ~T/200 larger under dependence)",
              pass_flag=int(ok), max_rel_vs_pyspi=relmax(S, s200),
              pyspi_01=float(S[0, 1]), pyspi_10=float(S[1, 0]), full_01=float(sfull[0, 1]), full_10=float(sfull[1, 0]),
              full_over_200_ratio=[float(np.nanmin(sfull / s200)), float(np.nanmax(sfull / s200))],
              decision_200=float(np.sign(s200[1, 0] - s200[0, 1])), decision_full=float(np.sign(sfull[1, 0] - sfull[0, 1])),
              canonical_200_01=float(can200[0, 1]), canonical_200_10=float(can200[1, 0]),
              canonical_full_01=float(canfull[0, 1]), canonical_full_10=float(canfull[1, 0]))
    elif tag == "orientation":
        z = prep(raw(run), run["path"])
        B = pairwise(lambda a, b: kd.anm_replica(a, b), z)
        D = B.T - B
        sign = truth_sign(run, P)
        write(rid, D, "direction", "independent-impl", f"{ME}:run_anm[orientation] -> _kernels_distance.anm_replica with "
              "its own GP fit; value = B.T - B", "orientation check on a lagged direction dataset (outside the "
              "contemporaneous ANM assumption): expected sign = sign of the reference of the same estimator; ground-truth "
              "recovery reported separately", expected_sign=np.sign(np.nan_to_num(D)),
              pyspi_truth_agreement=truth_agreement(S, sign), ref_truth_agreement=truth_agreement(D, sign),
              pyspi_D01=float(S[0, 1]), ref_D01=float(D[0, 1]))
    elif tag == "random-state":
        V = np.zeros_like(S)
        np.fill_diagonal(V, np.nan)
        write(rid, V, "exact", "same-lineage", "pyspi-vs-pyspi determinism (self-consistency): probe "
              "probes/cdt_pairwise_probes.py:anm_random_state recomputes pyspi's score with random_state 7 and None and "
              "returns max |score(rs) - score(42)|; the zero expectation comes from sklearn/gaussian_process/_gpr.py:322-333 "
              "(rng used only for restarts; n_restarts_optimizer = 0)", "probe value = max |score(rs) - score(42)| over rs in "
              "{7, None}; must be 0",
              rtol=0.0, atol=0.0)
    elif tag == "patch-global":
        ok = bool(np.all(S == 1.0))
        write(rid, np.ones(3), "qualitative", "same-lineage", "static code-property probe (pyspi/cdt introspection in "
              "the project venv vs a hard-coded expected flag vector read from pyspi/statistics/causal.py:24, the class-body "
              "assignment ANM.anm_score = corrected_anm_score)", "probe flags [ANM.anm_score is pyspi's function, cdt ANM().anm_score "
              "returns pyspi's value, cdt's unpatched gp.predict(x) - y broadcasts to (T, T)]; pass = all three hold",
              pass_flag=int(ok), flags=S)
    elif tag == "labels":
        ok = bool(np.array_equal(S, [1.0, 0.0, 1.0]))
        write(rid, np.array([1.0, 0.0, 1.0]), "qualitative", "same-lineage", "static code-property probe (pyspi "
              "introspection vs a hard-coded expected flag vector read from pyspi/statistics/causal.py:15 and "
              "pyspi/config.yaml:326-334)", "probe flags ['linear' in class labels, 'nonlinear' in class labels, "
              "'nonlinear' in config labels]; pass = [1, 0, 1], i.e. the label mismatch is present", pass_flag=int(ok),
              flags=S)
    elif tag == "scipy118-drift":
        import scipy
        import sklearn
        from sklearn.gaussian_process import GaussianProcessRegressor
        th, _ = pyspi(rid.replace("scipy118-drift", "theta"))
        V, Vfix, Vulp = (np.full((P, P), np.nan) for _ in range(3))
        dth = []
        for i in range(P):
            for j in range(P):
                if i == j:
                    continue
                xi, yj = inp[i][:, None], inp[j][:, None]
                gp = GaussianProcessRegressor(random_state=42).fit(xi, yj)
                r = gp.predict(xi).reshape(-1, 1) - yj
                V[i, j] = kd.cdt_fast_hsic_gamma(zs(r), zs(xi))
                dth.append(np.abs(gp.kernel_.theta - th[:, i, j]).max())
                c, l = np.exp(th[:, i, j])
                Vfix[i, j] = kd.anm_replica(inp[i], inp[j], c, l)
                Vulp[i, j] = kd.anm_replica(inp[i], inp[j], np.nextafter(c, np.inf), l)
        write(rid, V, "tolerance", "same-lineage", f"{ME}:run_anm[scipy118-drift] -> sklearn=={sklearn.__version__} "
              f"GaussianProcessRegressor(random_state=42) under scipy=={scipy.__version__} (comparison venv) + "
              "_kernels_distance.cdt_fast_hsic_gamma", "pyspi runs scipy 1.17.1 (DEPENDENCY_MIGRATION.md SciPy row: "
              "1.18.1 changed anm by 1.25e-6 relative). Extra keys decompose the drift: theta change of the optimiser, "
              "replica at pyspi's theta under scipy 1.18 linear algebra, and the score change for a one-ulp change of c",
              rtol=1e-5, atol=0.0, max_rel=relmax(V, S), max_abs_dlog_theta=float(max(dth)),
              fixed_theta_scipy118_max_rel=relmax(Vfix, S), one_ulp_constant_max_rel=relmax(Vulp, S))
    elif tag in ("scale", "detrend"):
        z = prep(raw(run), run["path"])
        V = pairwise(lambda a, b: kd.anm_replica(a, b), z)
        bg, _ = pyspi("causal.anm|bivariate_gaussian|256|pipeline|replica-theta")
        write(rid, V, "tolerance", "independent-impl", f"{ME}:run_anm[{tag}] -> _kernels_distance.anm_replica (own fit) on "
              "the reference's own preprocessing of the raw dataset", "rtol 1e-5 as for replica-ownfit; extra keys "
              "compare with pyspi on bivariate_gaussian (same draw, normalise=True)", rtol=1e-5, atol=0.0,
              max_rel=relmax(S, V), input_max_abs_vs_pyspi=float(np.abs(z - inp).max()),
              max_rel_vs_bivariate_gaussian_pipeline=relmax(S, bg))
    else:
        raise KeyError(tag)


def run_cds(run, tag):
    rid, T = run["run_id"], run["T"]
    S, inp = pyspi(rid)
    if tag in ("spike-zero",):
        x = prep(spike_input(T), run["path"])
    else:
        x = prep(raw(run), run["path"])
    P = x.shape[0]
    imax = float(np.abs(x - inp).max()) if inp.shape == x.shape and tag not in ("spike-zero",) else float("nan")
    ref = lambda z: pairwise(lambda a, b: kd.cds_ref(a, b), z)
    det = f"{ME}:run_cds -> _kernels_distance.cds_ref (numpy, cdt CDS.py:56-195 conventions, ffactor 2, maxdev 3, minc 12)"
    if tag == "exact":
        write(rid, ref(x), "exact", "independent-impl", det, input_max_abs_vs_pyspi=imax)
    elif tag == "paper-variant":
        A = ref(x)
        V = pairwise(lambda a, b: cds_variant(a, b, 3, 1), x)
        Vf = pairwise(lambda a, b: cds_variant(a, b, 3, 0), x)
        Vd = pairwise(lambda a, b: cds_variant(a, b, 2, 1), x)
        iu = np.triu_indices(P, 1)
        flips = int(np.sum(np.sign(A[iu] - A.T[iu]) != np.sign(V[iu] - V.T[iu])))
        write(rid, V, "qualitative", "independent-impl", f"{ME}:cds_variant (Fonollosa 2016 Sec. 2.1/2.3, sfactor 3, "
              "sample variance ddof 1)", "pass = every unordered pair keeps the ordering of A[i,j] vs A[j,i]; extra keys "
              "give the size of each change", pass_flag=int(flips == 0), n_order_flips=flips, n_pairs=len(iu[0]),
              cdt_ref_max_abs_vs_pyspi=float(np.nanmax(np.abs(A - S))),
              variant_median_rel=float(np.nanmedian(np.abs(V - S) / S)), variant_max_rel=relmax(V, S),
              ffactor3_only_max_rel=relmax(Vf, S), ddof1_only_max_rel=relmax(Vd, S))
    elif tag in ("direction", "orientation"):
        B = ref(x)
        D = B.T - B
        sign = truth_sign(run, P)
        exp = sign if tag == "direction" else np.sign(np.nan_to_num(D))
        write(rid, D, "direction", "independent-impl", det + "; value = B.T - B",
              "pyspi value = A.T - A (probe; > 0 means i -> j). direction: expected sign = ground truth; orientation: "
              "expected sign = the reference of the same estimator (lagged dataset outside the method's assumptions)",
              expected_sign=exp, ref_truth_agreement=truth_agreement(D, sign), pyspi_truth_agreement=truth_agreement(S, sign),
              pyspi_D01=float(S[0, 1]), ref_D01=float(D[0, 1]), **({"pyspi_D23": float(S[2, 3]), "ref_D23": float(D[2, 3])}
                                                                  if P == 4 else {}))
    elif tag in ("zero-predicate", "spike-zero"):
        B = ref(x)
        count_part = np.zeros((P, P), bool)
        full = np.zeros((P, P), bool)
        nq = []
        for i in range(P):
            q = cds_bins(x[i])[1]
            nq.append(len(q))
            for j in range(P):
                if i == j:
                    continue
                pm = cds_pmfs(x[i], x[j])
                count_part[i, j] = len(q) <= 1
                full[i, j] = len(q) <= 1 or all(np.array_equal(p, pm[0]) for p in pm)
        off = ~np.eye(P, dtype=bool)
        zero_ok = bool(np.all(S[count_part & off] == 0.0))
        iff_ok = bool(np.array_equal((np.abs(S) < 1e-15)[off], full[off]))
        match = bool(np.allclose(S[off], B[off], rtol=1e-10, atol=1e-12))
        write(rid, B, "qualitative", "independent-impl", det + "; predicate from cdt's binning (CDS.py:69-85, 164)",
              "pass = (A[i,j] == 0.0 exactly wherever process i has <= 1 x-bin with > 12 samples) and (|A[i,j]| < 1e-15 "
              "iff that count condition holds or all qualifying conditional pmfs are identical) and A matches cds_ref",
              pass_flag=int(zero_ok and iff_ok and match), qualifying_bins_per_process=np.array(nq),
              n_exact_zero=int(np.sum(S[off] == 0.0)), n_count_part=int(count_part[off].sum()),
              n_full_predicate=int(full[off].sum()), zero_ok=int(zero_ok), iff_ok=int(iff_ok), matches_ref=int(match))
    elif tag == "invariance":
        bg = np.asarray(make_data.load("bivariate_gaussian", T)["x"], float)
        write(rid, ref(bg), "exact", "independent-impl", det + " on bivariate_gaussian (the same draw before scale/offset); "
              "the expectation of equality is the positive affine-invariance argument",
              "positive affine invariance: pyspi on raw scaled_offset (normalise=False) vs the reference on the unscaled "
              "draw", ref_on_scaled_max_abs=float(np.nanmax(np.abs(ref(x) - ref(bg)))))
    elif tag == "detrend":
        bg = np.asarray(make_data.load("bivariate_gaussian", T)["x"], float)
        write(rid, ref(x), "exact", "independent-impl", det + " on scipy.signal.detrend + zscore(ddof=1) of trending",
              "EEG path (detrend then z-score); extra key = change relative to the trend-free draw",
              input_max_abs_vs_pyspi=imax, max_abs_vs_trend_free=float(np.nanmax(np.abs(S - ref(bg)))))
    else:
        raise KeyError(tag)


def run_reci(run, tag):
    rid, T = run["run_id"], run["T"]
    S, inp = pyspi(rid)
    x = prep(raw(run), run["path"])
    P = x.shape[0]
    imax = float(np.abs(x - inp).max()) if inp.shape == x.shape else float("nan")
    ref = lambda z: pairwise(kd.reci_ref, z)
    det = f"{ME}:run_reci -> _kernels_distance.reci_ref (numpy lstsq on [1, x^3] after min-max scaling, in-sample MSE)"
    if tag == "exact":
        write(rid, ref(x), "exact", "independent-impl", det, input_max_abs_vs_pyspi=imax)
    elif tag in ("direction", "orientation"):
        B = pairwise(reci_canonical, x) if tag == "direction" else ref(x)
        D = B.T - B
        sign = truth_sign(run, P)
        exp = sign if tag == "direction" else np.sign(np.nan_to_num(D))
        extra = dict(ref_truth_agreement=truth_agreement(D, sign), pyspi_truth_agreement=truth_agreement(S, sign),
                     pyspi_D01=float(S[0, 1]), ref_D01=float(D[0, 1]))
        if P == 4:
            extra.update(pyspi_D23=float(S[2, 3]), ref_D23=float(D[2, 3]))
        if tag == "direction":
            Bc = ref(x)
            extra.update(cdt_form_D01=float(Bc[1, 0] - Bc[0, 1]), **({"cdt_form_D23": float(Bc[3, 2] - Bc[2, 3])} if P == 4 else {}))
        write(rid, D, "direction", "independent-impl",
              (f"{ME}:reci_canonical (Bloebaum et al. 2018: MON n = 2..9, held-out MSE, 20 splits)" if tag == "direction"
               else det) + "; value = B.T - B",
              "pyspi value = A.T - A (probe; > 0 means i -> j). direction: expected sign = ground truth; orientation: "
              "expected sign = the reference of the same estimator", expected_sign=exp, **extra)
    elif tag == "invariance":
        bg = np.asarray(make_data.load("bivariate_gaussian", T)["x"], float)
        write(rid, ref(bg), "exact", "independent-impl", det + " on bivariate_gaussian (the same draw before scale/offset); "
              "the expectation of equality is the positive affine-invariance argument",
              "positive affine invariance (min-max scaling removes scale and offset)")
    elif tag == "detrend":
        bg = np.asarray(make_data.load("bivariate_gaussian", T)["x"], float)
        write(rid, ref(x), "exact", "independent-impl", det + " on scipy.signal.detrend + zscore(ddof=1) of trending",
              "EEG path (detrend then z-score); extra key = change relative to the trend-free draw",
              input_max_abs_vs_pyspi=imax, max_abs_vs_trend_free=float(np.nanmax(np.abs(S - ref(bg)))))
    else:
        raise KeyError(tag)


def run_igci(run, tag):
    rid, T = run["run_id"], run["T"]
    S, inp = pyspi(rid)
    x = prep(raw(run), run["path"])
    P = x.shape[0]
    imax = float(np.abs(x - inp).max()) if inp.shape == x.shape else float("nan")
    ref = lambda z: pairwise(kd.igci_ref, z)
    det = (f"{ME}:run_igci -> _kernels_distance.igci_ref (Daniusis et al. 2010 eq. 12 1-spacing entropy, Gaussian "
           "reference: standardise ddof 0; value S(z_i) - S(z_j) = -C_{i->j})")
    if tag == "exact":
        write(rid, ref(x), "exact", "independent-impl", det, input_max_abs_vs_pyspi=imax,
              pyspi_min=float(np.nanmin(S)), pyspi_max=float(np.nanmax(S)))
    elif tag == "cubic-analytic":
        vals = np.array([kd.igci_ref(*cubic_pair(T, 1000 + s)) for s in range(20)])
        se = vals.std(ddof=1) / np.sqrt(20)
        V = np.array([[np.nan, IGCI_CUBIC], [-IGCI_CUBIC, np.nan]])
        write(rid, V, "statistical", "analytic", "-C^G_{X->Y} = -(E log|f'(X)| + log sd X - log sd Y) for X ~ U(0,1), "
              "Y = X^3 (Daniusis et al. 2010 Sec. 3.2/3.5) = 0.88320; se = sd over 20 seeds / sqrt(20) of the "
              "independent igci_ref on the same inline inputs", "pyspi value = mean over seeds 1000..1019 of the pyspi "
              "matrix; positive A[x, y] establishes 'positive = i -> j'",
              se=np.array([[np.nan, se], [se, np.nan]]), ref_mean=float(vals.mean()), ref_se=float(se),
              analytic=float(IGCI_CUBIC))
    elif tag == "antisym":
        write(rid, -S.T, "exact", "same-lineage", "pyspi-vs-pyspi self-consistency: value = -A.T of pyspi's own "
              "matrix; the antisymmetry expectation follows from S(z_i) - S(z_j) (IGCI.py:141)",
              "value = -A.T from pyspi itself; rtol = atol = 0, i.e. A[j, i] == -A[i, j] bit for bit", rtol=0.0, atol=0.0,
              max_abs_A_plus_AT=float(np.nanmax(np.abs(S + S.T))))
    elif tag in ("direction", "orientation"):
        B = ref(x)
        sign = truth_sign(run, P)
        exp = sign if tag == "direction" else np.sign(np.nan_to_num(B))
        extra = dict(ref_truth_agreement=truth_agreement(B, sign), pyspi_truth_agreement=truth_agreement(S, sign),
                     pyspi_A01=float(S[0, 1]), ref_A01=float(B[0, 1]))
        if P == 4:
            extra.update(pyspi_A23=float(S[2, 3]), ref_A23=float(B[2, 3]))
        write(rid, B, "direction", "independent-impl", det, "pyspi matrix itself (antisymmetric, > 0 means i -> j). "
              "direction: expected sign = ground truth; orientation: expected sign = the reference", expected_sign=exp, **extra)
    elif tag == "raw-bivariate":
        u, v = cubic_pair(1024, 0)
        a = kd.igci_ref(zscore(u, ddof=1), v)
        b = kd.igci_ref(zscore(v, ddof=1), u)
        both = kd.igci_ref(zscore(u, ddof=1), zscore(v, ddof=1)) + kd.igci_ref(zscore(v, ddof=1), zscore(u, ddof=1))
        write(rid, np.array([[np.nan, a], [b, np.nan]]), "exact", "independent-impl", det + " on (zscore(first), raw "
              "second), the input parse_bivariate builds via Data.add_process (base.py:33-36, data.py:230-238)",
              "pyspi value from spi.bivariate(u, v) and spi.bivariate(v, u) with raw arrays", pyspi_gap=float(S[0, 1] + S[1, 0]),
              ref_gap=float(a + b), ref_gap_both_zscored=float(both))
    elif tag == "labels":
        ok = bool(np.array_equal(S, [1.0, 0.0, 1.0]))
        write(rid, np.array([1.0, 0.0, 1.0]), "qualitative", "same-lineage", "static code-property probe (pyspi "
              "introspection vs a hard-coded expected flag vector read from pyspi/statistics/causal.py:56-60, base.py:146-150, "
              "config.yaml:359-367)", "probe flags ['unsigned' in class labels, issigned(), 'unsigned' in config labels]; "
              "pass = [1, 0, 1]: the output is declared unsigned (signed values: see the |exact runs' pyspi_min)",
              pass_flag=int(ok), flags=S)
    else:
        raise KeyError(tag)


def main():
    runs = json.loads(MANIFEST.read_text())["runs"]
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in runs:
        rid = run["run_id"]
        if only and not any(o in rid for o in only):
            continue
        if not (OUT_PY / f"{rid}.npz").is_file():
            print(f"SKIP {rid}: no pyspi output")
            continue
        tag = rid.split("|")[-1]
        for pre in ("p01-", "p23-"):
            tag = tag[len(pre):] if tag.startswith(pre) else tag
        unit = run["unit_id"].split(".")[1]
        try:
            {"anm": run_anm, "cds": run_cds, "reci": run_reci, "igci": run_igci}[unit](run, tag)
            n_ok += 1
        except Exception as exc:  # report and continue
            import traceback
            traceback.print_exc()
            print(f"ERROR {rid}: {exc!r}")
            n_err += 1
    print(f"cdt_pairwise references: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
