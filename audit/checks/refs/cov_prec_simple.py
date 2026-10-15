"""cov_prec_simple references (COMPARISON venv: numpy/scipy only, no sklearn, no pyspi).

Units: basic.cov.{EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance} and
basic.prec.{same}. pyspi delegates every unit to sklearn.covariance (fit on the
(T, P) array with all defaults; covariance_ / precision_ = scipy pinvh of it,
pyspi/statistics/basic.py:31-54). The references below never import sklearn:

* estimator kernels from _kernels_basic_misc (empirical_cov = Xc Xc^T / T,
  ledoit_wolf = Ledoit & Wolf 2004 eq. 14 with the per-sample bbar^2 sum,
  oas(paper=False) = sklearn's variant without the 2/p terms, oas(paper=True) =
  Chen et al. 2010 Eq. 23, shrunk_cov = 0.9 S + 0.1 tr(S)/p I);
* precision = numpy LU inverse (np.linalg.inv), not the eigh/pinvh route pyspi
  takes; the pinvh mirror (_kernels_basic_misc.precision) and np.cov(bias=True)
  are reported as extra keys;
* preprocessing re-implemented here: z-score with ddof=1 (pyspi/data.py:184-195),
  linear least-squares detrend for pipeline-detrend (scipy.signal.detrend type
  'linear' semantics).

Run tags (last field of run_id):
  ref          estimator on the prepared input vs pyspi (exact for cov, 1e-10 for prec)
  sq-ref       squared identifier vs the square of the reference (+ bitwise vs square of
               pyspi's own plain output)
  doc-corr     cov vs plain Pearson r (the documented 'equivalent to correlation');
               EXPECTED TO FAIL, records the size of the (1-s)(T-1)/T factor
  doc-pcorr    prec vs partial correlation -Theta_ij/sqrt(Theta_ii Theta_jj) (the documented
               quantity); EXPECTED TO FAIL, records the sign/range gap
  eq23         OAS vs Chen et al. Eq. 23 (2/p kept); EXPECTED TO FAIL, quantifies the
               omission (also *-eq23 on the P = 19 stacks)
  truth/null   statistical: pyspi vs the Monte Carlo mean of the same estimator applied to
               N_MC draws from the dataset's population model (cov_true); se = MC sd,
               inflated so the 2-se rule is a familywise 5% test over the P(P-1)/2 pairs.
               LW/OAS 'null' runs are exact zeros (shrinkage clipped to 1).
  sym          qualitative: symmetry (bitwise for cov, <= 1e-12 max|A| for prec) AND
               value agreement with the reference
  singular     shared_envelope (x2 == x0): prec_EmpiricalCovariance vs numpy SVD pinv
  avgref       average-referenced input (rank P-1): pseudo-inverse with the null
               eigenvalue removed explicitly
  stackA/B     P = 19 arrays stacked from independent datasets (EEG-like P)
  shrink       fitted sklearn shrinkage_ (probe) vs the kernel shrinkage
  mutation*    qualitative: order independence of outputs under the in-place diagonal
               mutation of the cached estimator; flags reported as extra keys
  scaling      pipeline prec = D1 Theta_kernel D1 with D1 = diag(sd ddof=1)

Run: python audit/checks/refs/cov_prec_simple.py   (reads manifest/cov_prec_simple.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))

import make_data  # noqa: E402
import _kernels_basic_misc as K  # noqa: E402

FAMILY = "cov_prec_simple"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EPS = np.finfo(np.float64).eps
N_MC = 1000
MC_SEED = 20261014

# must equal probes/cov_prec_simple_probes.py:STACKS
STACKS = {
    "stackA": ["iid_gaussian", "bivariate_gaussian", "sparse_precision", "gauss_ar1_pair", "var2_chain"],
    "stackB": ["delayed_noise", "narrowband_f0-0.1", "narrowband_f0-0.25", "var2_bivariate",
               "bivariate_gaussian", "var2_chain"],
}
DETAIL_EST = {
    "EmpiricalCovariance": "_kernels_basic_misc.empirical_cov (Xc Xc^T / T)",
    "LedoitWolf": "_kernels_basic_misc.ledoit_wolf (Ledoit & Wolf 2004 eq. 14, literal bbar^2)",
    "OAS": "_kernels_basic_misc.oas(paper=False) (Chen et al. 2010 Eq. 23 without the 2/p terms)",
    "ShrunkCovariance": "_kernels_basic_misc.shrunk_cov(0.1) (0.9 S + 0.1 tr(S)/p I)",
}


# ------------------------------------------------------------------ preprocessing
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


def stack(name, T):
    return np.vstack([np.asarray(make_data.load(d, T)["x"], float) for d in STACKS[name]])


def run_input(run):
    tag = run["run_id"].split("|")[-1]
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], float)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    for name in STACKS:
        if name in tag:
            x = stack(name, run["T"])
    if "avgref" in tag:
        x = x - x.mean(axis=0, keepdims=True)
    return x


# ------------------------------------------------------------------ estimators
def estimate(X, est, paper=False):
    if est == "EmpiricalCovariance":
        return K.empirical_cov(X), 0.0
    if est == "LedoitWolf":
        return K.ledoit_wolf(X)
    if est == "OAS":
        return K.oas(X, paper=paper)
    if est == "ShrunkCovariance":
        return K.shrunk_cov(X, 0.1)
    raise ValueError(est)


def nandiag(A):
    A = np.array(A, float, copy=True)
    np.fill_diagonal(A, np.nan)
    return A


def parse(run):
    _, kind, est = run["unit_id"].split(".")
    ident = run.get("identifier") or ""
    return kind, est, ("-sq_" in ident)


def pyspi_value(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], float)


def sym_max(A):
    if A is None or A.ndim != 2:
        return float("nan")
    d = np.abs(A - A.T)
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in kw.items():
        if isinstance(v, str):
            arrays[k] = np.array(v)
        elif v is None:
            continue
        else:
            arrays[k] = np.asarray(v)
    np.savez(OUT_REF / f"{rid}.npz", **arrays)


def fw_factor(P):
    """Inflate se so that '|d| <= 2 se' is a two-sided familywise 5% test over P(P-1)/2 pairs."""
    m = max(P * (P - 1) // 2, 1)
    return float(ndtri(1 - 0.05 / (2 * m)) / 2.0)


# ------------------------------------------------------------------ reference matrices
def ref_matrix(X, kind, est, paper=False):
    C, s = estimate(X, est, paper=paper)
    if kind == "cov":
        return C, s, {}
    try:
        Theta = np.linalg.inv(C)
    except np.linalg.LinAlgError:   # exactly singular (duplicated channel): caller substitutes a pinv
        Theta = np.full_like(C, np.nan)
        return Theta, s, dict(cond=float("inf"), max_rel_inv_vs_pinvh_mirror=float("nan"))
    extra = dict(cond=float(np.linalg.cond(C)),
                 max_rel_inv_vs_pinvh_mirror=float(np.max(np.abs(Theta - K.precision(C)) /
                                                          np.maximum(np.abs(Theta), 1e-300))))
    return Theta, s, extra


def prec_rtol(cond):
    """Accuracy floor of a computed inverse: relative error ~ cond * eps. Use 1e-10 unless
    the conditioning makes that unreachable for two backward-stable inverses."""
    return max(1e-10, 50.0 * cond * EPS)


# ------------------------------------------------------------------ Monte Carlo truth
_MC_CACHE = {}


def population(ds, T):
    d = make_data.load(ds, T)
    if ds in ("bivariate_gaussian", "iid_gaussian"):
        P = d["x"].shape[0]
        Sigma = d["cov_true"] if "cov_true" in d else np.eye(P)
        return Sigma, np.zeros(P)
    if ds == "scaled_offset":
        return d["cov_true"], d["offset"]
    if ds == "sparse_precision":
        return d["cov_true"], np.zeros(d["x"].shape[0])
    raise ValueError(ds)


def mc_stats(ds, T, path, kind, est):
    key = (ds, T, path, est)
    if key not in _MC_CACHE:
        Sigma, mu = population(ds, T)
        L = np.linalg.cholesky(Sigma)
        rng = np.random.default_rng(MC_SEED + T)
        covs, precs = [], []
        for _ in range(N_MC):
            x = L @ rng.standard_normal((Sigma.shape[0], T)) + mu[:, None]
            C, _s = estimate(prepare(x, path), est)
            covs.append(C)
            precs.append(np.linalg.inv(C))
        covs, precs = np.array(covs), np.array(precs)
        _MC_CACHE[key] = dict(cov=(covs.mean(0), covs.std(0, ddof=1)),
                              prec=(precs.mean(0), precs.std(0, ddof=1)), Sigma=Sigma,
                              samples=dict(cov=covs, prec=precs))
    return _MC_CACHE[key]


def population_limit(Sigma, path, kind):
    """T -> infinity limit (every shrinkage -> 0 on Gaussian data with dependence)."""
    M = Sigma
    if path != "kernel":
        d = np.sqrt(np.diag(Sigma))
        M = Sigma / np.outer(d, d)
    return M if kind == "cov" else np.linalg.inv(M)


# ------------------------------------------------------------------ run handlers
def handle(run):
    rid = run["run_id"]
    tag = rid.split("|")[-1]
    kind, est, squared = parse(run)
    path, T, ds = run["path"], run["T"], run["dataset"]
    S = pyspi_value(rid)
    if S is None:
        return "skip"   # no pyspi output yet (pyspi side failed or not run); compare reports it
    common = dict(pyspi_sym_max_abs=sym_max(S) if S is not None else float("nan"))

    # ---- fitted shrinkage readout
    if tag.endswith("shrink"):
        X = prepare(run_input(run), path)
        _, s = estimate(X, est)
        extra = dict(P=X.shape[0], T=T, s_kernel=s)
        notes = f"pyspi fitted shrinkage_ vs {DETAIL_EST[est]}"
        if est == "OAS":
            _, r23 = estimate(X, est, paper=True)
            extra.update(rho_eq23=r23, rho_rel_diff=(s / r23 - 1.0) if r23 > 0 else float("nan"),
                         offdiag_factor_rel_diff=((1 - s) / (1 - r23) - 1.0) if r23 < 1 else float("nan"))
            notes += "; rho_eq23 = Chen et al. 2010 Eq. 23 with 2/p; offdiag_factor_rel_diff = (1-rho_skl)/(1-rho_eq23)-1"
        save(rid, value=np.array(s), criterion="tolerance", rtol=1e-10, atol=1e-13,
             reference_kind="independent-impl",
             reference_detail=f"refs/cov_prec_simple.py:handle[shrink] -> {DETAIL_EST[est]}",
             notes=notes, **extra, **common)
        return

    # ---- mutation / aliasing flags
    if tag == "mutation":
        f = S
        names = ["order_indep_fwd", "order_indep_rev", "cov_returned_is_cache", "cache_cov_diag_all_nan",
                 "prec_returned_is_cache", "cache_prec_diag_all_nan", "score_nan_after_prec",
                 "mahalanobis_nan_frac_after_prec", "score_nan_after_cov_only"]
        extra = {n: float(v) for n, v in zip(names, f)}
        ok = int(f[0] == 1 and f[1] == 1)
        save(rid, value=np.array(np.nan), criterion="qualitative", pass_flag=ok, reference_kind="same-lineage",
             reference_detail="probes/cov_prec_simple_probes.py:mutation (pyspi vs pyspi self-consistency); pass = every cov/cov-sq/prec/prec-sq "
                              "output bitwise identical whether computed on a fresh Data or after the other three "
                              "on a shared Data (both orders)",
             notes="flags describe the in-place np.fill_diagonal on the cached sklearn estimator "
                   "(pyspi/statistics/basic.py:49-50)", **extra)
        return
    if tag == "mutation-others":
        names = [f"{e}_{o}" for e in ("EllipticEnvelope", "MinCovDet", "GraphicalLasso", "GraphicalLassoCV")
                 for o in ("fwd", "rev")]
        extra = {n: float(v) for n, v in zip(names, S)}
        save(rid, value=np.array(np.nan), criterion="qualitative", pass_flag=int(np.all(S == 1)),
             reference_kind="same-lineage",
             reference_detail="probes/cov_prec_simple_probes.py:mutation_others (pyspi vs pyspi self-consistency); pass = outputs order independent "
                              "for estimators of other families sharing Estimators._from_cache",
             notes="open question 5 of the trace stage", **extra)
        return

    X = prepare(run_input(run), path)
    P = X.shape[0]

    # ---- documented-claim comparisons (expected to fail)
    if tag == "doc-corr":
        r = nandiag(np.corrcoef(X))
        C, s, _ = ref_matrix(X, kind, est)
        ratio = offdiag(S) / offdiag(r)
        save(rid, value=r, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
             reference_detail="np.corrcoef of the prepared input (documented: 'correlation and covariance "
                              "matrices are equivalent', supplement 1.1.1 p.3-4)",
             notes="EXPECTED TO FAIL: pyspi = (1-s)(T-1)/T * r, not r",
             ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
             expected_ratio=float((1 - s) * (T - 1) / T), s_kernel=s, **common)
        return
    if tag == "doc-pcorr":
        Theta, s, extra = ref_matrix(X, "prec", est)
        dg = np.sqrt(np.outer(np.diag(Theta), np.diag(Theta)))
        pc = nandiag(-Theta / dg)
        off = ~np.eye(P, dtype=bool)
        big = off & (np.abs(pc) > 1e-3)
        opp = float(np.mean(np.sign(S[big]) == -np.sign(pc[big]))) if big.any() else float("nan")
        renorm = np.nanmax(np.abs(-S[off] / dg[off] - pc[off]))
        kw = dict(frac_opposite_sign=opp, max_abs_pyspi=float(np.nanmax(np.abs(S))),
                  max_abs_pcorr=float(np.nanmax(np.abs(pc))), max_abs_renormalised_minus_pcorr=float(renorm),
                  s_kernel=s)
        if ds == "bivariate_gaussian":
            kw.update(pyspi_23=float(S[2, 3]), pcorr_23=float(pc[2, 3]), pyspi_01=float(S[0, 1]),
                      pcorr_01=float(pc[0, 1]))
        save(rid, value=pc, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
             reference_detail="partial correlation -Theta_ij/sqrt(Theta_ii Theta_jj) (Whittaker 1990) from "
                              f"np.linalg.inv of {DETAIL_EST[est]} (documented: 'equivalent to the partial "
                              "correlation ... within [-1, 1]', supplement 1.1.2 p.4)",
             notes="EXPECTED TO FAIL: pyspi returns Theta_ij, the unnormalised precision entry",
             **kw, **extra, **common)
        return

    # ---- statistical truth (Monte Carlo of the population model through the same estimator)
    if tag in ("truth", "null") and not (tag == "null" and est in ("LedoitWolf", "OAS")):
        mc = mc_stats(ds, T, path, kind, est)
        mean, sd = mc[kind]
        lim = population_limit(mc["Sigma"], path, kind)
        f = fw_factor(P)
        dev_lim = np.abs(offdiag(S) - offdiag(lim))
        reps = mc["samples"][kind]
        off = ~np.eye(P, dtype=bool)
        zmax_rep = np.max(np.abs(reps[:, off] - mean[off]) / sd[off], axis=1)
        zmax_obs = float(np.max(np.abs(S[off] - mean[off]) / sd[off]))
        p_fw = float(np.mean(zmax_rep >= zmax_obs))
        save(rid, value=nandiag(mean), se=nandiag(sd * f), criterion="statistical",
             reference_kind="independent-impl",
             reference_detail=f"Monte Carlo (N={N_MC}, seed {MC_SEED}+T) of the population model "
                              f"({ds} cov_true, Gaussian iid) through {DETAIL_EST[est]} on the '{path}' input; "
                              "value = MC mean, se = MC sd x familywise factor",
             notes="pass = |pyspi - E_MC| <= 2 * se_fw for every pair; max_dev_vs_limit = distance to the "
                   "T -> infinity limit (correlation / covariance / its inverse); mc_familywise_p = fraction of MC "
                   "replicates whose max_pairs |dev|/sd is >= the observed one",
             fw_factor=f, max_dev_vs_limit=float(dev_lim.max()), max_dev_over_sd=zmax_obs,
             mc_familywise_p=p_fw, **common)
        return
    if tag == "null":   # LW / OAS: shrinkage clipped to 1 -> exact zeros
        _, s = estimate(X, est)
        save(rid, value=nandiag(np.zeros((P, P))), criterion="exact", rtol=1e-10, atol=0.0,
             reference_kind="analytic",
             reference_detail="analytic: b^2 = min(bbar^2, d^2) (LW) / min(num/den, 1) (OAS) clips the shrinkage "
                              "to 1, so the estimate is mu I and every off-diagonal is exactly 0.0",
             notes="kernel shrinkage on the same input reported as s_kernel", s_kernel=s, **common)
        return

    # ---- pipeline precision scaling identity
    if tag == "scaling":
        xraw = run_input(run)
        Theta_k = np.linalg.inv(K.empirical_cov(xraw))
        D1 = np.diag(xraw.std(axis=1, ddof=1))
        D0 = np.diag(xraw.std(axis=1, ddof=0))
        V = D1 @ Theta_k @ D1
        save(rid, value=nandiag(V), criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
             reference_detail="analytic: pipeline Theta = inv((T-1)/T R) = D1 Theta_kernel D1, D1 = diag(sd ddof=1)",
             notes="T/(T-1) D0 Theta_kernel D0 is the same matrix (D1^2 = T/(T-1) D0^2)",
             pyspi_01=float(S[0, 1]), theta_kernel_01=float(Theta_k[0, 1]),
             d0_form_01=float((T / (T - 1) * D0 @ Theta_k @ D0)[0, 1]), precision_true_01=-0.4, **common)
        return

    # ---- OAS Eq. 23 (expected to fail)
    if tag.endswith("eq23"):
        V, r23, _ = ref_matrix(X, kind, est, paper=True)
        Vs, rs, _ = ref_matrix(X, kind, est, paper=False)
        rel = np.abs(offdiag(Vs) - offdiag(V)) / np.abs(offdiag(V))
        save(rid, value=nandiag(V), criterion="tolerance", rtol=1e-10, atol=1e-12,
             reference_kind="independent-impl",
             reference_detail="_kernels_basic_misc.oas(paper=True): Chen, Wiesel, Eldar & Hero 2010 IEEE TSP 58:5016 "
                              "Eq. 23 with the 2/p terms" + (" -> np.linalg.inv" if kind == "prec" else ""),
             notes="EXPECTED TO FAIL: sklearn drops the 2/p terms (_shrunk_covariance.py:70-71, 83-85)",
             P=P, rho_eq23=r23, rho_sklearn_variant=rs, rho_rel_diff=rs / r23 - 1 if r23 > 0 else float("nan"),
             offdiag_median_rel_diff=float(np.median(rel)), offdiag_max_rel_diff=float(rel.max()),
             pyspi_vs_variant_max_abs=float(np.nanmax(np.abs(S - nandiag(Vs)))), **common)
        return

    # ---- standard reference (ref, sq-ref, sym, singular, avgref, stacks)
    V, s, extra = ref_matrix(X, kind, est)
    rtol, atol, crit = 1e-10, 1e-12, "exact"
    detail = f"refs/cov_prec_simple.py:ref_matrix -> {DETAIL_EST[est]}"
    rk = "independent-impl"
    if kind == "prec":
        crit = "tolerance"
        detail += " -> np.linalg.inv (LU)"
        rtol = prec_rtol(extra["cond"])
        atol = 1e-12 * float(np.max(np.abs(V)))
    if kind == "cov" and est == "EmpiricalCovariance" and path == "kernel":
        extra["max_abs_vs_np_cov_bias"] = float(np.max(np.abs(V - np.cov(X, bias=True))))
    if path != "kernel":
        extra["expected_offdiag_factor"] = float((1 - s) * (T - 1) / T) if kind == "cov" else float("nan")
    w = np.linalg.eigvalsh(K.empirical_cov(X))
    extra.update(min_eig_S=float(w[0]), max_eig_S=float(w[-1]), s_kernel=s, P=P)

    if kind == "prec" and est == "EmpiricalCovariance" and (tag == "singular" or "avgref" in tag):
        C = K.empirical_cov(X)
        if tag == "singular":
            V = np.linalg.pinv(C)
            detail = "numpy.linalg.pinv (SVD, rtol = max(M,N) eps) of the empirical covariance"
        else:   # rank P-1 by construction: drop exactly one eigenvalue
            wv, U = np.linalg.eigh(C)
            V = (U[:, 1:] / wv[1:]) @ U[:, 1:].T
            detail = "Moore-Penrose pseudo-inverse with the single structural null eigenvalue removed (eigh)"
        rk = "analytic"
        rtol, atol = 1e-10, 1e-12 * float(np.max(np.abs(V)))
        crit = "tolerance"
        extra["cond"] = float(w[-1] / abs(w[0])) if w[0] != 0 else float("inf")
        try:
            extra["inv_LU_max_abs"] = float(np.max(np.abs(np.linalg.inv(C))))
        except np.linalg.LinAlgError:
            extra["inv_LU_max_abs"] = float("inf")
    if S is not None and S.ndim == 2 and tag in ("singular",) and P >= 3:
        extra["pyspi_02"] = float(S[0, 2])

    notes = ""
    if squared:
        V = V ** 2
        plain = pyspi_value(rid.replace(f"|{tag}", "|ref"))
        if plain is not None:
            extra["bitwise_equal_square_of_plain"] = int(np.array_equal(np.square(plain), S, equal_nan=True))
        notes = "squared identifier: reference is the elementwise square"
    V = nandiag(V)

    if tag == "sym":
        m = ~np.isnan(S) & ~np.isnan(V)
        val_ok = bool(np.allclose(S[m], V[m], rtol=rtol, atol=atol))
        smax = sym_max(S)
        sym_ok = (smax == 0.0) if kind == "cov" else (smax <= 1e-12 * float(np.nanmax(np.abs(S))))
        save(rid, value=V, criterion="qualitative", pass_flag=int(val_ok and sym_ok), reference_kind=rk,
             reference_detail=detail + "; symmetry rule: bitwise for cov, <= 1e-12 max|A| for prec (REVIEW M2)",
             notes="undirected: A[i,j] must equal A[j,i] and the reference value",
             value_ok=int(val_ok), sym_ok=int(sym_ok),
             value_max_abs_diff=float(np.max(np.abs(S[m] - V[m]))),
             sym_nonzero=int(np.sum(np.abs(S - S.T) > 0)), **extra, **common)
        return
    save(rid, value=V, criterion=crit, rtol=rtol, atol=atol, reference_kind=rk, reference_detail=detail,
         notes=notes, **extra, **common)


def main():
    m = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_err = n_skip = 0
    for run in m["runs"]:
        try:
            if handle(run) == "skip":
                n_skip += 1
                print(f"REF SKIP {run['run_id']}: no pyspi output", file=sys.stderr)
                continue
            n_ok += 1
        except Exception as exc:  # report and continue; a missing reference shows up in compare
            n_err += 1
            print(f"REF ERROR {run['run_id']}: {exc!r}", file=sys.stderr)
    print(f"{FAMILY} references: {n_ok} written, {n_skip} skipped (no pyspi output), {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
