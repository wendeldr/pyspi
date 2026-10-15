"""References for the preprocessing_base family (COMPARISON venv: no pyspi, no jpype).

Units: preproc.detrend, preproc.zscore, preproc.two_array_api, base.pair_handling.

Every run in ``audit/checks/manifest/preprocessing_base.json`` is dispatched on the tag
(the last ``|`` field of its run_id) to one function below, which writes
``audit/checks/out/ref/<run_id>.npz`` in the compare.py contract.

Building blocks (independent of pyspi and of the libraries pyspi delegates to here):
* ``ols_detrend``: closed-form least-squares residual r = x - xbar - b (t - tbar),
  b = sum (t - tbar)(x - xbar) / sum (t - tbar)^2 (Bendat & Piersol 2010, trend removal;
  pyspi delegates to scipy.signal.detrend, which solves the same problem by lstsq).
* ``zs``: (x - mean) / sqrt(sum (x - mean)^2 / (T - 1)) (pyspi delegates to scipy.stats.zscore, ddof=1).
* SPI values on preprocessed arrays come from the shared kernels: _kernels_basic_misc
  (spearman, ridge_mse), _kernels_distance (dtw_dp), _kernels_infotheory (gaussian_mi, gaussian_te).
* Infrastructure checks of base.pair_handling are analytic (toy SPI, call order) or compare
  pyspi with itself under a transformation (transpose, visiting order, variant order); those
  are labelled same-lineage (pyspi vs pyspi).

Run: <comparison venv python> audit/checks/refs/preprocessing_base.py
"""
import json
import sys
from pathlib import Path

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))

import numpy as np  # noqa: E402

import make_data  # noqa: E402
import _kernels_basic_misc as kb  # noqa: E402
import _kernels_distance as kd  # noqa: E402
import _kernels_infotheory as ki  # noqa: E402

FAMILY = "preprocessing_base"
OUT = CHECKS / "out" / "ref"
PYSPI = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / FAMILY
ME = "refs/preprocessing_base.py"


# --------------------------------------------------------------------------- building blocks
def ols_detrend(X):
    """Row-wise closed-form OLS linear-trend residual."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    T = X.shape[1]
    t = np.arange(T, dtype=np.float64)
    tc = t - t.mean()
    Xc = X - X.mean(axis=1, keepdims=True)
    b = (Xc @ tc) / (tc @ tc)
    return Xc - b[:, None] * tc[None, :]


def zs(X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Xc = X - X.mean(axis=1, keepdims=True)
    return Xc / np.sqrt((Xc ** 2).sum(axis=1, keepdims=True) / (X.shape[1] - 1))


def nan_diag(A):
    A = np.array(A, dtype=np.float64, copy=True)
    np.fill_diagonal(A, np.nan)
    return A


def pair_matrix(Z, f, symmetric=False):
    P = Z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j or (symmetric and j < i):
                continue
            A[i, j] = f(Z[i], Z[j])
            if symmetric:
                A[j, i] = A[i, j]
    return A


def cov_mle(Z):
    return nan_diag(kb.empirical_cov(Z, ddof=0))


SPI_FROM_Z = {
    "cov_EmpiricalCovariance": (lambda Z: cov_mle(Z), "_kernels_basic_misc.empirical_cov(ddof=0) (1/T MLE)"),
    "spearmanr": (lambda Z: pair_matrix(Z, kb.spearman, True), "_kernels_basic_misc.spearman"),
    "mi_gaussian": (lambda Z: pair_matrix(Z, ki.gaussian_mi, True), "_kernels_infotheory.gaussian_mi"),
    "dtw": (lambda Z: pair_matrix(Z, kd.dtw_dp, True), "_kernels_distance.dtw_dp (squared local cost, sqrt)"),
}
PAIR_FN = {
    "dtw": kd.dtw_dp,
    "lmfit_Ridge": lambda a, b: kb.ridge_mse(a, b, alpha=1.0),
    "cov_EmpiricalCovariance": lambda a, b: float(np.mean((a - a.mean()) * (b - b.mean()))),
    "spearmanr": kb.spearman,
    "gc_gaussian_k-1_kt-1_l-1_lt-1": lambda a, b: ki.gaussian_te(a, b, k=1, k_tau=1, l=1, l_tau=1, delay=1),
}


def raw(run):
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    return x


def bg(T):
    return np.asarray(make_data.load("bivariate_gaussian", T)["x"], dtype=np.float64)


def pyspi_value(rid):
    with np.load(PYSPI / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def sidecar(rid):
    p = EXCH / (rid.replace("|", "__") + ".json")
    return json.loads(p.read_text()) if p.is_file() else {}


def save(rid, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in kw.items():
        if isinstance(v, (dict, list)) and k not in ("value", "se", "expected_sign"):
            v = json.dumps(v, default=float)
        clean[k] = np.asarray(v)
    np.savez(OUT / f"{rid}.npz", **clean)


def tag_of(rid):
    return rid.split("|")[4]


# =========================================================================== preproc.detrend
def r_detonly(run):
    x = raw(run)
    V = ols_detrend(x)
    save(run["run_id"], value=V, criterion="tolerance", rtol=0.0, atol=1e-12 * float(np.abs(x).max()),
         reference_kind="independent-impl",
         reference_detail=f"{ME}:ols_detrend (closed-form OLS residual, numpy); pyspi side Data(x, normalise=False, detrend=True)",
         notes="detrend only; atol = 1e-12 * max|x|",
         ref_max_abs_row_mean=float(np.abs(V.mean(axis=1)).max()),
         ref_max_abs_corr_with_t=float(np.abs([np.corrcoef(v, np.arange(v.size))[0, 1] for v in V]).max()))


def r_order_full(run):
    x = raw(run)
    save(run["run_id"], value=zs(ols_detrend(x)), criterion="exact", rtol=1e-12, atol=1e-12,
         reference_kind="independent-impl",
         reference_detail=f"{ME}:zs(ols_detrend(x)) (numpy; order detrend -> z-score ddof=1)",
         notes="Data(x, detrend=True) full array")


def r_trunc(run):
    x = raw(run)
    xt = x[: x.shape[0] - 1, : x.shape[1] - 7]
    save(run["run_id"], value=zs(ols_detrend(xt)), criterion="exact", rtol=1e-12, atol=1e-12,
         reference_kind="independent-impl",
         reference_detail=f"{ME}:zs(ols_detrend(x[:P-1, :T-7])) (numpy; truncate -> detrend -> z-score ddof=1)",
         notes="Data(x, detrend=True, n_processes=P-1, n_observations=T-7); the reference applies truncation first")


def r_inv_data(run):
    T = run["T"]
    if run["unit_id"] == "preproc.detrend":
        V = zs(ols_detrend(bg(T)))
        det = "zs(ols_detrend(bivariate_gaussian x)): trending = bivariate_gaussian + slope*t, OLS detrending annihilates a + b t"
    else:
        V = zs(bg(T))
        det = "zs(bivariate_gaussian x): scaled_offset = scale*bivariate_gaussian + offset (scale > 0), z-scoring is invariant"
    save(run["run_id"], value=V, criterion="tolerance", rtol=0.0, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{ME}:r_inv_data; {det}", notes="pyspi Data array of the transformed dataset vs numpy preprocessing of the untransformed draw")


def r_inv_spi(run):
    ident = run["identifier"]
    T = run["T"]
    Z = zs(ols_detrend(bg(T))) if run["unit_id"] == "preproc.detrend" else zs(bg(T))
    f, det = SPI_FROM_Z[ident]
    save(run["run_id"], value=f(Z), criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{ME}:r_inv_spi; {det} on numpy-preprocessed bivariate_gaussian (same draw)",
         notes=f"{ident} through the {run['path']} path on {run['dataset']} must equal the SPI of the untransformed draw")


def r_nodetrend(run):
    ident = run["identifier"]
    S = pyspi_value(run["run_id"])
    f, det = SPI_FROM_Z[ident]
    V = f(zs(ols_detrend(bg(run["T"]))))
    d = float(np.nanmax(np.abs(S - V)))
    save(run["run_id"], value=V, criterion="qualitative", pass_flag=int(d > 0.05), reference_kind="analytic",
         reference_detail=f"{ME}:r_nodetrend; {det}",
         notes="control: without detrending, the trending dataset must NOT reproduce the detrended bivariate_gaussian SPI (pass = differs by > 0.05)",
         max_abs_diff_without_detrend=d)


def r_order_hist_data(run):
    x = raw(run)
    r = ols_detrend(x)
    c = x.std(axis=1, ddof=1) / r.std(axis=1, ddof=1)
    hist = ols_detrend(zs(x))
    S = pyspi_value(run["run_id"])
    ratio = S / hist
    save(run["run_id"], value=c[:, None] * hist, criterion="exact", rtol=1e-12, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{ME}:r_order_hist_data; zscore(detrend(x)) = c * detrend(zscore(x)), c = std(x)/std(resid) (ddof=1)",
         notes="current order (pyspi) vs historical order times the per-row constant",
         c_per_row=c.tolist(), pyspi_over_hist_ratio_row_median=np.median(ratio, axis=1).tolist(),
         pyspi_over_hist_ratio_row_spread=(ratio.max(axis=1) - ratio.min(axis=1)).tolist())


def r_order_hist_spis(run):
    x = raw(run)
    Zc = zs(ols_detrend(x))
    H = ols_detrend(zs(x))
    c = x.std(axis=1, ddof=1) / ols_detrend(x).std(axis=1, ddof=1)
    V = np.array([[cov_mle(Zc), cov_mle(H)],
                  [pair_matrix(Zc, kb.spearman, True), pair_matrix(H, kb.spearman, True)]])
    S = pyspi_value(run["run_id"])
    off = ~np.eye(x.shape[0], dtype=bool)
    cov_ratio = (S[0, 0] / S[0, 1])[off]
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{ME}:r_order_hist_spis; numpy cov (1/T) and _kernels_basic_misc.spearman on zs(ols_detrend(x)) and ols_detrend(zs(x))",
         notes="axes: [cov_EmpiricalCovariance, spearmanr] x [current order, historical order]",
         c_per_row=c.tolist(),
         cov_current_minus_hist_max_abs=float(np.nanmax(np.abs(S[0, 0] - S[0, 1]))),
         cov_ratio_vs_c_outer_max_rel_dev=float(np.max(np.abs(cov_ratio / np.outer(c, c)[off] - 1))),
         spearman_current_minus_hist_max_abs=float(np.nanmax(np.abs(S[1, 0] - S[1, 1]))))


def r_flat(run):
    S = pyspi_value(run["run_id"])
    sc = sidecar(run["run_id"])
    names = sc.get("replacements", [])
    accepted = [n for n, row in zip(names, S) if row[0] == 0]
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(len(accepted) == 0), reference_kind="analytic",
         reference_detail=f"{ME}:r_flat; a constant or exactly linear channel has an identically zero OLS residual, so its z-score is undefined and the channel should be rejected",
         notes="pass = every replacement rejected by Data(detrend=True); fail = some flat/linear channel accepted as a unit-variance artefact; "
               "the outcome depends on the other rows too (scipy detrend solves all P rows in one lstsq), so flat-P3/flat-P2 repeat the run on the first 3/2 rows",
         accepted_with_detrend=accepted,
         stored_std=[None if np.isnan(r[1]) else round(float(r[1]), 6) for r in S],
         stored_n_unique=[None if np.isnan(r[2]) else int(r[2]) for r in S],
         stored_lag1_autocorr=[None if np.isnan(r[3]) else round(float(r[3]), 4) for r in S],
         raised_without_detrend=[int(r[4]) for r in S], messages=sc.get("messages", []),
         P=int(raw(run).shape[0]))


def r_swallow(run):
    S = pyspi_value(run["run_id"])
    sc = sidecar(run["run_id"])
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(S[5] == 0), reference_kind="analytic",
         reference_detail=f"{ME}:r_swallow; a detrend failure (inf input) should be reported, not ignored",
         notes="pass = Data(normalise=False, detrend=True) rejects an inf sample; fail = accepted undetrended",
         inf_accepted=int(S[5]), inf_still_present=float(S[6]), stored_row0_slope=float(S[7]),
         normalise_true_raises=int(S[8]), stdout_tail=sc.get("inf_stdout", ""))


# =========================================================================== preproc.zscore
def r_zs(run):
    x = raw(run)
    V = zs(x)
    S = pyspi_value(run["run_id"])
    T = x.shape[1]
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-13, reference_kind="independent-impl",
         reference_detail=f"{ME}:zs ((x - mean)/std ddof=1, numpy)", notes="Data(x) default path",
         pyspi_max_abs_z=float(np.abs(S).max()), shiffler_bound=float((T - 1) / np.sqrt(T)),
         pyspi_max_abs_row_mean=float(np.abs(S.mean(axis=1)).max()),
         pyspi_max_abs_var_ddof1_minus_1=float(np.abs(S.var(axis=1, ddof=1) - 1).max()))


def r_cov_ratio(run):
    x = raw(run)
    T = x.shape[1]
    claim = tag_of(run["run_id"]) == "cov-supp-claim"
    R = nan_diag(np.corrcoef(x))
    V = R if claim else (T - 1) / T * R
    S = pyspi_value(run["run_id"])
    off = ~np.eye(x.shape[0], dtype=bool)
    kw = dict(criterion="exact", rtol=1e-12, atol=1e-14) if not claim else dict(criterion="exact")
    save(run["run_id"], value=V, reference_kind="analytic",
         reference_detail=(f"{ME}:r_cov_ratio; supplement S1.1.1 p.3 claim 'correlation and covariance matrices are equivalent' -> numpy corrcoef"
                           if claim else f"{ME}:r_cov_ratio; ddof=1 z-scores with the 1/T MLE: cov = (T-1)/T * Pearson r (numpy corrcoef)"),
         notes="cov_EmpiricalCovariance on bivariate_gaussian (pipeline)",
         pyspi_over_corr_mean=float(np.mean(S[off] / R[off])), t_minus_1_over_t=float((T - 1) / T), **kw)


def r_errmsg(run):
    S = pyspi_value(run["run_id"])
    sc = sidecar(run["run_id"])
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(S[0] == 1 and S[2] == 1), reference_kind="analytic",
         reference_detail=f"{ME}:r_errmsg; a zero-variance channel has an undefined z-score; expected an error naming the cause",
         notes="pass = constant channel rejected with a message about zero variance / constant input",
         raised=int(S[0]), message_mentions_nan=int(S[1]), message_mentions_variance=int(S[2]), message=sc.get("constant", ""))


def r_nanpolicy(run):
    S = pyspi_value(run["run_id"])
    sc = sidecar(run["run_id"])
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(S[3] == 0), reference_kind="analytic",
         reference_detail=f"{ME}:r_nanpolicy; zscore(nan_policy='omit') leaves the other rows finite, so the option would only matter if NaN rows were accepted",
         notes="pass = a NaN sample is tolerated as nan_policy='omit' suggests; fail = rejected anyway (option has no effect)",
         nan_input_raised=int(S[3]), omit_alone_leaves_other_rows_finite=int(S[4]), message=sc.get("nan", ""))


# =========================================================================== preproc.two_array_api
def r_capture(run):
    x = raw(run)
    tag = tag_of(run["run_id"])
    S = pyspi_value(run["run_id"])
    if tag.endswith("contract"):
        V = zs(x)
        det = f"{ME}:r_capture; documented contract (supplement S1 p.3): every process z-scored (ddof=1)"
        kw = dict(criterion="exact", rtol=1e-12, atol=1e-12)
    else:
        V = np.vstack([zs(x[0]), x[1:]])
        det = f"{ME}:r_capture; candidate mechanism: first array z-scored, later arrays appended raw (data.py add_process)"
        kw = dict(criterion="exact", rtol=1e-12, atol=1e-13)
    save(run["run_id"], value=V, reference_kind="analytic", reference_detail=det,
         notes="array received by a toy SPI through the direct-array API",
         pyspi_row_means=S.mean(axis=1).round(6).tolist(), pyspi_row_std_ddof1=S.std(axis=1, ddof=1).round(6).tolist(),
         later_rows_bit_identical_to_raw=int(np.array_equal(S[1:], x[1:])), **kw)


def r_two_array_spi(run):
    x = raw(run)
    ident = run["identifier"]
    a, b = zs(x[0])[0], zs(x[1])[0]
    f = PAIR_FN[ident]
    v = np.array([f(a, b), f(b, a)])
    V = np.repeat(v[:, None], 3, axis=1)
    S = pyspi_value(run["run_id"])
    dev = np.abs(S - V)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{ME}:r_two_array_spi; kernel on z-scored inputs (documented contract), both orientations",
         notes="columns: [direct raw arrays (bivariate(x, y) or multivariate([x, y])), Data path, direct z-scored arrays]; rows: (x, y), (y, x)",
         dev_direct_raw=float(dev[:, 0].max()), dev_data_path=float(dev[:, 1].max()), dev_direct_zscored=float(dev[:, 2].max()),
         pyspi_direct_raw=S[:, 0].tolist(), reference=v.tolist(), path_used=sidecar(run["run_id"]).get("path_used", ""))


def r_unset_ij(run):
    S = pyspi_value(run["run_id"])
    sc = sidecar(run["run_id"])
    ids = sc.get("identifiers", [])
    info = sc.get("info", [])
    nlog = S[:, 2] if S.shape[1] > 2 else np.zeros(len(ids))
    # returned a value with no exception, no Python warning and no logging record
    finite = [i for i, r, n in zip(ids, S, nlog) if r[0] == 1 and r[1] == 0 and n == 0]
    nonfinite = [i for i, r, n in zip(ids, S, nlog) if r[0] == 2 and r[1] == 0 and n == 0]
    # returned a value with no exception and no Python warning, but a logging record was emitted
    logged_only = [i for i, r, n in zip(ids, S, nlog) if r[0] != 0 and r[1] == 0 and n > 0]
    notimpl = sum("NotImplementedError" in str(r[2]) for r in info)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(not finite and not nonfinite), reference_kind="analytic",
         reference_detail=f"{ME}:r_unset_ij; bivariate on a 3-process Data with i = j = None is ill-posed: it should raise or warn",
         notes="pass = every identifier raised, emitted a Python warning, or emitted a logging record (level >= WARNING); "
               "fail = some identifier returned a value with none of these. logged_only lists returns reported only through the logging module",
         n_identifiers=len(ids), n_raised=int((S[:, 0] == 0).sum()), n_not_implemented=int(notimpl),
         n_python_warnings=int(S[:, 1].sum()), n_logging_records=int(np.sum(nlog)),
         silent_finite=finite, silent_nonfinite=nonfinite, logged_only=logged_only,
         logged_only_messages=[r[4][:1] if len(r) > 4 else [] for r in info if r[0] in logged_only],
         returned_values=[r[2] for r in info if r[0] in finite or r[0] in nonfinite or r[0] in logged_only])


def r_univariate(run):
    S = pyspi_value(run["run_id"])
    sc = sidecar(run["run_id"])
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(S[0] == 0), reference_kind="analytic",
         reference_detail=f"{ME}:r_univariate; parse_univariate documents direct input of a time series (base.py:5-7 comment)",
         notes="pass = a raw 1-D array is accepted by a parse_univariate-decorated method", raised=int(S[0]), error=sc.get("error") or "")


# =========================================================================== base.pair_handling
def toy_expected(P):
    Ad = np.full((P, P), np.nan)
    Au = np.full((P, P), np.nan)
    C = np.full((P, P), np.nan)
    k = 0
    for j in range(P):
        for i in range(P):
            if i == j:
                continue
            Ad[i, j] = 10 * i + j
            Au[i, j] = 10 * min(i, j) + max(i, j)
            C[i, j] = k
            k += 1
    return Ad, Au, C


def r_toy(run):
    P = raw(run).shape[0]
    Ad, Au, C = toy_expected(P)
    sc = sidecar(run["run_id"])
    save(run["run_id"], value=np.array([Ad, Au, C]), criterion="exact", rtol=0.0, atol=0.0, reference_kind="analytic",
         reference_detail=f"{ME}:toy_expected; Directed A[i,j] = 10i+j; Undirected keeps bivariate(i<j); call order j outer, i inner",
         notes="[A_directed, A_undirected, call position]", n_calls=sc.get("n_calls", -1), expected_n_calls=P * (P - 1),
         first_calls=sc.get("undirected_calls", [])[:4])


def r_calls(run):
    x = raw(run)
    P = x.shape[0]
    Z = zs(x)
    ident = run["identifier"]
    if ident == "spearmanr":
        A = pair_matrix(Z, kb.spearman, True)
        det = "_kernels_basic_misc.spearman"
    else:
        A = pair_matrix(Z, PAIR_FN[ident])
        det = "_kernels_infotheory.gaussian_te(src=z_i, targ=z_j) (A[i,j] = TE i -> j)"
    _, _, C = toy_expected(P)
    S = pyspi_value(run["run_id"])
    extra = {}
    if P == 2:
        extra = dict(pyspi_A01=float(S[0, 0, 1]), pyspi_A10=float(S[0, 1, 0]), ref_te_0to1=float(A[0, 1]), ref_te_1to0=float(A[1, 0]))
    save(run["run_id"], value=np.array([A, C]), criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{ME}:r_calls; {det} on numpy z-scores; call positions analytic (j outer, i inner)",
         notes="[A, call position]; n_calls = P(P-1)", n_calls=int(np.nanmax(S[1]) + 1), expected_n_calls=P * (P - 1), **extra)


def _asym_stats(S, ids):
    asym = {i: float(np.nanmax(np.abs(m - m.T))) if np.isfinite(m).any() else None for i, m in zip(ids, S)}
    anti = {i: float(np.nanmax(np.abs(m + m.T))) if np.isfinite(m).any() else None for i, m in zip(ids, S)}
    return asym, anti


def r_sym(run):
    S = pyspi_value(run["run_id"])
    ids = sidecar(run["run_id"]).get("identifiers", [])
    asym, anti = _asym_stats(S, ids)
    kind = run["run_id"].split("|sym-")[1].split("-")[0]
    tol = dict(rtol=0.0, atol=0.0) if kind in ("mirror", "directed") else dict(rtol=1e-12, atol=1e-12)
    nan_pat = [i for i, m in zip(ids, S) if not np.array_equal(np.isnan(m), np.isnan(m.T))]
    vals = [v for v in asym.values() if v is not None]
    save(run["run_id"], value=np.swapaxes(S, -1, -2), criterion="exact", reference_kind="same-lineage",
         reference_detail=f"{ME}:r_sym; transpose of pyspi's own matrix (symmetry definition A == A.T)",
         notes=f"{kind} class; all config identifiers of the class on iid_gaussian; mirror/directed: bit-exact required, bypass: 1e-12",
         max_abs_A_minus_AT=max(vals) if vals else float("nan"), asym_by_identifier=asym, antisym_by_identifier=anti,
         nan_pattern_asymmetric=nan_pat, **tol)


def r_upper(run):
    S = pyspi_value(run["run_id"])
    ids = sidecar(run["run_id"]).get("identifiers", [])
    V = np.stack([S[:, 0], S[:, 0]], axis=1)
    diff = {i: float(np.nanmax(np.abs(s[0] - s[1]))) for i, s in zip(ids, S)}
    save(run["run_id"], value=V, criterion="exact", rtol=0.0, atol=0.0, reference_kind="same-lineage",
         reference_detail=f"{ME}:r_upper; stored loop output A_loop (pyspi) as the reference for the upper-triangle-only build",
         notes="[A_loop, A_upper-only] per identifier; differences mean the stored value depends on the visiting order",
         differing_identifiers={k: v for k, v in diff.items() if v != 0}, n_identifiers=len(ids))


def r_grp(run):
    S = pyspi_value(run["run_id"])
    ids = sidecar(run["run_id"]).get("identifiers", [])
    diff = {i: float(np.nanmax(np.abs(S[0, k] - S[1, k]))) if np.isfinite(S[0, k]).any() else 0.0 for k, i in enumerate(ids)}
    save(run["run_id"], value=np.stack([S[0], S[0]]), criterion="exact", rtol=0.0, atol=0.0, reference_kind="same-lineage",
         reference_detail=f"{ME}:r_grp; forward-order output (pyspi) as the reference for the reversed-order output on one shared Data",
         notes="cache-sharing variants computed in config order and reversed order, seeded once",
         differing_identifiers={k: v for k, v in diff.items() if v != 0}, n_identifiers=len(ids))


def r_rng(run):
    S = pyspi_value(run["run_id"])
    save(run["run_id"], value=np.stack([S[0], S[0]]), criterion="exact", rtol=0.0, atol=0.0, reference_kind="same-lineage",
         reference_detail=f"{ME}:r_rng; seed-42 output vs seed-42 plus one earlier np.random draw",
         notes="already documented: tests/README.md:115 (sgddtw depends on the global RNG)",
         max_abs_diff=float(np.nanmax(np.abs(S[0] - S[1]))))


def r_direction(run):
    S = pyspi_value(run["run_id"])
    P = S.shape[-1]
    sign = np.zeros_like(S)
    sign[1, 0, 1] = 1.0
    sign[1, 1, 0] = -1.0
    A, B = S[0], S[2]
    d = abs(A[0, 1] - A[1, 0])
    unresolved = bool(np.isfinite(d) and d <= 1e-12 * max(abs(A[0, 1]), abs(A[1, 0]), 1e-300))
    crit = dict(criterion="qualitative", pass_flag=0) if unresolved else dict(criterion="direction")
    save(run["run_id"], value=np.full_like(S, np.nan), expected_sign=sign, reference_kind="analytic", **crit,
         direction_resolvable=int(not unresolved),
         reference_detail=(f"{ME}:r_direction; dataset construction (process 0 drives process 1: var2_bivariate A1[1,0]=0.5; delayed_noise x1 = x0(t-1)+noise; "
                           "logistic_maps_chaotic beta_yx=0.1; anm cause[k]->effect[k]) read with the glossary convention 'rows are sources' (glossary-of-terms.md:27) and 'larger = stronger'"),
         notes=("slice 1 (A - A.T) must be positive at [0,1]; slices: [A, A - A.T, isolated bivariate B, B - B.T]; "
                "|A[0,1] - A[1,0]| <= 1e-12 relative counts as no resolvable direction (qualitative fail), not as a sign"),
         A01=float(A[0, 1]), A10=float(A[1, 0]), B01=float(B[0, 1]), B10=float(B[1, 0]),
         larger_entry="A[0,1]" if A[0, 1] > A[1, 0] else ("A[1,0]" if A[1, 0] > A[0, 1] else "equal/NaN"), P=P)


DISPATCH = [
    ("detonly", r_detonly), ("order-full", r_order_full), ("trunc", r_trunc), ("inv-data", r_inv_data),
    ("inv-", r_inv_spi), ("nodetrend-", r_nodetrend), ("order-hist-data", r_order_hist_data),
    ("order-hist-spis", r_order_hist_spis), ("flat", r_flat), ("swallow", r_swallow),
    ("zs", r_zs), ("cov-", r_cov_ratio), ("errmsg", r_errmsg), ("nanpolicy", r_nanpolicy),
    ("cap-", r_capture), ("list-", r_capture), ("spi-", r_two_array_spi), ("mod-", r_unset_ij), ("univariate-", r_univariate),
    ("toy", r_toy), ("calls-", r_calls), ("sym-", r_sym), ("upper-", r_upper), ("grp-", r_grp),
    ("rng-", r_rng), ("dir-", r_direction),
]


def main():
    m = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_skip = n_err = 0
    for run in m["runs"]:
        rid = run["run_id"]
        tag = tag_of(rid)
        fn = next(f for p, f in DISPATCH if tag == p or tag.startswith(p))
        if not (PYSPI / f"{rid}.npz").is_file() and fn not in (r_detonly, r_order_full, r_trunc, r_inv_data, r_inv_spi, r_zs, r_toy):
            n_skip += 1
            print(f"skip {rid}: no pyspi output", file=sys.stderr)
            continue
        try:
            fn(run)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
    print(f"{FAMILY} refs: {n_ok} written, {n_skip} skipped, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
