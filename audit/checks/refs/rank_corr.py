"""References for the rank_corr family (basic.spearmanr, basic.kendalltau). COMPARISON venv.

No scipy.stats import: the estimators are the from-scratch kernels in
``_kernels_basic_misc`` (``spearman`` = Pearson correlation of average ranks, Spearman
1904 with mid-rank ties; ``kendall_tau_b`` = O(n^2) pair count, Kendall 1945 tau-b), the
analytic population values for a bivariate normal, and R ``stats::cor`` through
``r/rank_corr.R`` as an external package. Every run's reference is computed for BOTH
ordered pairs (V[i, j] = f(x_i, x_j), V[j, i] = f(x_j, x_i)), so both orientations of the
stored matrix are compared.

Input transformations mirror ``probes/rank_corr_probes.py`` exactly:
``tie_round`` = ``np.round(x, 1)``; ``split_realisations`` = contiguous halves.
Run tags (the last field of the run_id) select the reference; see ``REFS`` below.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np  # noqa: E402

import make_data  # noqa: E402
from _kernels_basic_misc import _rank_average, kendall_tau_b, spearman  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PYSPI = CHECKS / "out" / "pyspi"
EXCHANGE = CHECKS / "out" / "exchange" / "rank_corr"
FAMILY = "rank_corr"
KDETAIL = "refs/_kernels_basic_misc.py:"
EPS = np.finfo(np.float64).eps


# ---------------------------------------------------------------- shared helpers
def tie_round(x):
    return np.round(np.asarray(x, dtype=np.float64), 1)


def split_realisations(x, R=2):
    P, T = x.shape
    n = T // R
    return np.stack([x[:, r * n:(r + 1) * n] for r in range(R)], axis=2)


def lstsq_detrend_zscore(x):
    """Least-squares line removal per row (np.linalg.lstsq on [1, t]), then z-score ddof=1."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[1]
    A = np.stack([np.ones(T), np.arange(T, dtype=np.float64)], axis=1)
    coef = np.linalg.lstsq(A, x.T, rcond=None)[0]
    r = x - (A @ coef).T
    return (r - r.mean(axis=1, keepdims=True)) / r.std(axis=1, ddof=1, keepdims=True)


def kendall_tau_a(x, y):
    """Kendall 1938 tau-a = (n_c - n_d) / (n(n-1)/2); no tie correction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    i, j = np.triu_indices(n, 1)
    return float(np.sum(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])) / (n * (n - 1) / 2))


def spearman_untied_formula(x, y):
    """1 - 6 sum d^2 / (n(n^2-1)) on mid-ranks: exact only without ties."""
    d = _rank_average(x) - _rank_average(y)
    n = d.size
    return float(1 - 6 * np.sum(d * d) / (n * (n * n - 1)))


_CACHE = {}


def stat(name, x, y):
    key = (name, hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest(),
           hashlib.sha1(np.ascontiguousarray(y).tobytes()).hexdigest())
    if key not in _CACHE:
        f = {"spearmanr": spearman, "kendalltau": kendall_tau_b,
             "tau_a": kendall_tau_a, "spearman_untied": spearman_untied_formula}[name]
        _CACHE[key] = f(x, y)
    return _CACHE[key]


def matrix(name, X):
    P = X.shape[0]
    V = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = stat(name, X[i], X[j])
    return V


def base_stat(identifier):
    return identifier.replace("-sq", "")


def pyspi_value(rid):
    with np.load(OUT_PYSPI / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def asym(S):
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return np.nan
    D = np.abs(S - S.T)
    return float(np.nanmax(D)) if np.isfinite(D).any() else np.nan


def save(rid, value, criterion, kind, detail, notes="", **extra):
    payload = dict(value=np.asarray(value, dtype=np.float64), criterion=np.array(criterion),
                   reference_kind=np.array(kind), reference_detail=np.array(detail), notes=np.array(notes))
    for k, v in extra.items():
        payload[k] = np.array(v) if isinstance(v, str) else np.asarray(v, dtype=np.float64)
    np.savez(OUT_REF / f"{rid}.npz", **payload)


def detail_estimator(name):
    return {"spearmanr": KDETAIL + "spearman (Pearson of average ranks; Spearman 1904, mid-rank ties)",
            "kendalltau": KDETAIL + "kendall_tau_b (O(n^2) pair count; Kendall 1945 Biometrika 33:239 tau-b)"}[name]


def exact_extra(S):
    return dict(rtol=0.0, atol=1e-12, pyspi_max_asym=asym(S))


# ---------------------------------------------------------------- population values
def population(name, rho, n):
    """Exact expectation of the sample statistic under a bivariate normal.
    Spearman: Moran 1948 Biometrika 35:203, E r_S = 6/(pi(n+1)) [asin rho + (n-2) asin(rho/2)].
    Kendall: Greiner 1909 / Kendall 1949, E tau = (2/pi) asin rho (unbiased U-statistic)."""
    if name == "spearmanr":
        return 6.0 / (np.pi * (n + 1)) * (np.arcsin(rho) + (n - 2) * np.arcsin(rho / 2))
    return 2.0 / np.pi * np.arcsin(rho)


def standard_error(name, rho, n):
    """rho = 0: exact null SD (Spearman 1/sqrt(n-1); Kendall sqrt(2(2n+5)/(9n(n-1)))).
    rho != 0: Fisher-z delta method, Spearman var(z) = (1 + r_S^2/2)/(n-3) (Bonett & Wright
    2000 Psychometrika 65:23), Kendall var(z) = 0.437/(n-4) (Fieller, Hartley & Pearson 1957
    Biometrika 44:470); se_r = (1 - r^2) sqrt(var z) at the population value r."""
    if rho == 0:
        return 1 / np.sqrt(n - 1) if name == "spearmanr" else np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    if name == "spearmanr":
        r = 6 / np.pi * np.arcsin(rho / 2)
        return (1 - r * r) * np.sqrt((1 + r * r / 2) / (n - 3))
    r = 2 / np.pi * np.arcsin(rho)
    return (1 - r * r) * np.sqrt(0.437 / (n - 4))


def population_matrix(name, C, n):
    P = C.shape[0]
    V, SE = np.full((P, P), np.nan), np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = population(name, C[i, j], n)
                SE[i, j] = standard_error(name, C[i, j], n)
    return V, SE


SE_FACTOR = 1.5   # compare.py passes |pyspi - value| <= 2*se; 2 * 1.5 SE = the record's 3-SE criterion


def statistical(rid, run, S, C, label):
    name = base_stat(run["identifier"])
    n = run["T"]
    V, SE = population_matrix(name, C, n)
    z = np.abs(S - V) / SE
    extra = dict(max_z_1se=float(np.nanmax(z)), pyspi_max_asym=asym(S))
    if C.shape[0] == 4 and C[0, 1] != 0:
        extra.update(abs_err_pair01=abs(S[0, 1] - V[0, 1]), abs_err_pair23=abs(S[2, 3] - V[2, 3]),
                     se_pair01=SE[0, 1], se_pair23=SE[2, 3])
    formula = ("Moran 1948 E r_S = 6/(pi(n+1))[asin rho + (n-2) asin(rho/2)]" if name == "spearmanr"
               else "Greiner 1909 E tau = (2/pi) asin rho")
    save(rid, V, "statistical", "analytic", f"refs/rank_corr.py:population ({formula}); SE refs/rank_corr.py:standard_error",
         f"{label}; se stored = {SE_FACTOR} x analytic SE so the 2-se band is 3 SE", se=SE_FACTOR * SE, **extra)


# ---------------------------------------------------------------- R external cross-check
def run_r(jobs):
    """jobs: list of (rid, method, X (P, T)) -> {rid: P x P matrix} via r/rank_corr.R."""
    if not jobs:
        return {}
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    with open(EXCHANGE / "jobs.csv", "w") as f:
        f.write("job,method\n")
        for k, (rid, method, X) in enumerate(jobs):
            f.write(f"{k},{method}\n")
            np.savetxt(EXCHANGE / f"in_{k}.csv", np.asarray(X).T, delimiter=",", fmt="%.17g")
    env = dict(os.environ, R_LIBS_USER=os.environ.get("R_LIBS_USER", "/home/dan/R/audit-lib"))
    r = subprocess.run(["Rscript", str(CHECKS / "r" / "rank_corr.R"), str(EXCHANGE)], env=env,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"Rscript failed: {r.stderr}")
    print(r.stdout.strip())
    out = {}
    for k, (rid, method, X) in enumerate(jobs):
        M = np.loadtxt(EXCHANGE / f"out_{k}.csv", delimiter=",", ndmin=2)
        np.fill_diagonal(M, np.nan)
        out[rid] = M
    return out


def r_version():
    r = subprocess.run(["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True)
    return r.stdout.strip()


# ---------------------------------------------------------------- per-tag references
def ref_run(run, S, rversion, rmats):
    rid, ident, ds, T, path = run["run_id"], run["identifier"], run["dataset"], run["T"], run["path"]
    tag = rid.split("|")[-1]
    name = base_stat(ident)
    unit = run["unit_id"]
    d = make_data.load(ds, T)
    x = np.asarray(d["x"], dtype=np.float64)
    est = detail_estimator(name)

    if tag == "exact":
        if path == "pipeline-detrend":
            X, how = lstsq_detrend_zscore(x), "on np.linalg.lstsq-detrended, z-scored rows"
        else:
            X, how = x, "on the RAW rows (pipeline z-scoring must not change ranks)" if path == "pipeline" else "on the raw rows"
        V = matrix(name, X)
        extra = exact_extra(S)
        if ds == "trending" and path == "pipeline":
            Vd = matrix(name, lstsq_detrend_zscore(x))
            extra.update(detrended_pair01=Vd[0, 1], detrended_pair23=Vd[2, 3], raw_pair01=V[0, 1], raw_pair23=V[2, 3])
        if ds == "trending" and path == "pipeline-detrend":
            Vr = matrix(name, x)
            extra.update(raw_pair01=Vr[0, 1], raw_pair23=Vr[2, 3], detrended_pair01=V[0, 1], detrended_pair23=V[2, 3])
        save(rid, V, "exact", "independent-impl", est, f"both ordered pairs, {how}", **extra)
        return

    if tag in ("ties", "sq-ties"):
        Xr = tie_round(x)
        V = matrix(name, Xr)
        alt = "tau_a" if name == "kendalltau" else "spearman_untied"
        Va = matrix(alt, Xr)
        if tag == "sq-ties":
            V = V ** 2
        extra = exact_extra(S)
        extra.update(min_distinct_values=min(len(np.unique(r)) for r in Xr),
                     max_abs_untied_formula_vs_tiecorrected=float(np.nanmax(np.abs(Va - matrix(name, Xr)))))
        save(rid, V, "exact", "independent-impl", est + "; input np.round(x, 1)",
             f"tied input (rounded to 1 decimal in probe and reference); path {path}", **extra)
        return

    if tag == "ties-vs-untied-formula":
        Xr = tie_round(x)
        Vb = matrix(name, Xr)
        alt = "tau_a" if name == "kendalltau" else "spearman_untied"
        Va = matrix(alt, Xr)
        pairs = [(0, 1), (2, 3)]
        d_b = max(abs(S[i, j] - Vb[i, j]) for i, j in pairs)
        d_a = min(abs(S[i, j] - Va[i, j]) for i, j in pairs)
        ok = (np.nanmax(np.abs(S - Vb)) <= 1e-12) and d_a > 1e-4
        altname = "tau-a (Kendall 1938)" if name == "kendalltau" else "1 - 6 sum d^2/(n(n^2-1)) untied formula"
        save(rid, Vb, "qualitative", "independent-impl",
             f"refs/rank_corr.py:{alt} vs {est}",
             f"pass = pyspi equals the tie-corrected statistic (<= 1e-12) and differs from {altname} by > 1e-4 on the dependent pairs",
             pass_flag=int(ok), max_abs_vs_tiecorrected=float(np.nanmax(np.abs(S - Vb))),
             min_abs_vs_untied_formula_dependent_pairs=d_a, max_abs_vs_tiecorrected_dependent_pairs=d_b,
             pyspi_pair01=S[0, 1], pyspi_pair23=S[2, 3], untied_pair01=Va[0, 1], untied_pair23=Va[2, 3])
        return

    if tag == "population":
        statistical(rid, run, S, d["cov_true"], "bivariate_gaussian population value (pipeline)")
        return

    if tag == "null":
        statistical(rid, run, S, np.eye(x.shape[0]), "iid_gaussian null, population value 0")
        return

    if tag == "contemporaneous-null":
        statistical(rid, run, S, np.eye(x.shape[0]),
                    "delayed_noise: zero-lag population correlation 0 for every pair (white noise, distinct delays)")
        return

    if tag == "pipeline-eq-kernel":
        K = pyspi_value(f"{unit}|{ds}|{T}|kernel|exact")
        same = np.array_equal(S, K, equal_nan=True)
        save(rid, K, "qualitative", "same-lineage",
             f"pyspi kernel-path matrix of run {unit}|{ds}|{T}|kernel|exact",
             "pass = pipeline (z-scored) matrix bit-identical to kernel (raw) matrix: rank invariance to z-scoring",
             pass_flag=int(same), max_abs_pipeline_vs_kernel=float(np.nanmax(np.abs(S - K))))
        return

    if tag == "invariance-exact":
        xb = np.asarray(make_data.load("bivariate_gaussian", T)["x"], dtype=np.float64)
        save(rid, matrix(name, xb), "exact", "independent-impl", est + " on bivariate_gaussian (unscaled draw)",
             "scaled_offset = positive scale * bivariate_gaussian + offset: reference on the UNSCALED rows",
             **exact_extra(S))
        return

    if tag == "invariance-bitid":
        B = pyspi_value(f"{unit}|bivariate_gaussian|{T}|{path}|exact")
        same = np.array_equal(S, B, equal_nan=True)
        save(rid, B, "qualitative", "same-lineage", f"pyspi run {unit}|bivariate_gaussian|{T}|{path}|exact",
             "pass = scaled_offset matrix bit-identical to bivariate_gaussian matrix on the same path",
             pass_flag=int(same), max_abs_vs_unscaled=float(np.nanmax(np.abs(S - B))))
        return

    if tag == "direction":
        G = d["Gamma0"]
        rho0 = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])
        C = np.array([[1.0, rho0], [rho0, 1.0]])
        V = np.array([[np.nan, population(name, rho0, 10 ** 9)], [population(name, rho0, 10 ** 9), np.nan]])
        sign = np.sign(C) * (1 - np.eye(2))
        save(rid, V, "direction", "analytic",
             "Gaussian VAR(2) zero-lag correlation from Gamma0, population rank correlation "
             "(6/pi) asin(rho/2) or (2/pi) asin(rho)",
             "expected sign = sign of the zero-lag correlation for BOTH orientations (no directional content)",
             expected_sign=sign, rho_zero_lag=rho0, pyspi_max_asym=asym(S), pyspi_01=S[0, 1], pyspi_10=S[1, 0])
        return

    if tag == "monotone-only":
        null_sd = standard_error(name, 0.0, T)
        vals = np.array([S[0, 1], S[2, 3]])
        ok = bool(np.all(np.abs(vals) <= 3 * null_sd))
        # dependence witnesses: the same statistic on monotone transforms that expose the dependence
        w01 = stat(name, np.abs(x[0]), x[1])
        w23 = stat(name, x[2] ** 2, x[3] ** 2)
        save(rid, np.zeros(2), "qualitative", "analytic",
             "population rank correlation 0 for x1 = x0^2 + noise (x0 ~ U(-1,1) symmetric) and for (cos th, sin th); "
             f"band 3 x null SD; witnesses via {est}",
             "pass = |r| <= 3 null SD on both dependent pairs (rank correlation sees monotone dependence only)",
             pass_flag=int(ok), pyspi_pair01=vals[0], pyspi_pair23=vals[1], band=3 * null_sd,
             witness_abs_x0_vs_x1=w01, witness_x2sq_vs_x3sq=w23)
        return

    if tag == "sq-exact":
        V = matrix(name, x) ** 2
        save(rid, V, "exact", "independent-impl", est + " squared, on the RAW rows",
             "-sq identifier = square of the signed statistic", pyspi_min=float(np.nanmin(S)),
             pyspi_max=float(np.nanmax(S)), **exact_extra(S))
        return

    if tag == "sq-relabel":
        signed_rid = {("bivariate_gaussian", 256): "population", ("bivariate_gaussian", 1024): "exact",
                      ("iid_gaussian", 1024): "null"}[(ds, T)]
        srid = f"{unit}|{ds}|{T}|pipeline|{signed_rid}"
        Ssig = pyspi_value(srid)
        same = np.array_equal(S, Ssig ** 2, equal_nan=True)
        save(rid, Ssig ** 2, "qualitative", "same-lineage", f"pyspi signed run {srid}, squared",
             "pass = -sq matrix bit-identical to the signed matrix squared elementwise (relabel)",
             pass_flag=int(same), max_abs_sq_vs_signed_squared=float(np.nanmax(np.abs(S - Ssig ** 2))))
        return

    if tag == "directed-ties":
        Xr = tie_round(x)
        save(rid, matrix(name, Xr), "exact", "independent-impl", est + "; input np.round(x, 1)",
             "Directed.multivariate output (no Undirected mirror): each ordered pair checked separately",
             **exact_extra(S))
        return

    if tag in ("recip-ties", "recip-continuous"):
        D = np.abs(S - S.T)
        off = ~np.eye(S.shape[0], dtype=bool)
        lo = np.minimum(np.abs(S), np.abs(S.T))
        ulps = np.where(off, D / np.spacing(np.where(lo > 0, lo, 1.0)), 0.0)
        n_asym = int(np.sum((D > 0) & off) // 2)
        if tag == "recip-ties":
            ok = float(np.max(ulps)) <= 4.0
            note = "pass = reciprocal Directed calls agree to <= 4 ulp of the smaller value (tied input)"
        else:
            ok = n_asym == 0
            note = "pass = reciprocal Directed calls bit-identical on continuous input"
        save(rid, S.T, "qualitative", "same-lineage", "pyspi Directed.multivariate transposed (reciprocal calls)",
             note, pass_flag=int(ok), max_abs_reciprocal=float(np.nanmax(D[off])), max_ulp_reciprocal=float(np.max(ulps)),
             n_unordered_pairs_asym=n_asym, n_unordered_pairs=int(off.sum() // 2))
        return

    if tag in ("R-cor", "R-cor-ties"):
        M = rmats[rid]
        X = tie_round(x) if tag == "R-cor-ties" else x
        mine = matrix(name, X)
        save(rid, M, "exact", "external-pkg",
             f"r/rank_corr.R stats::cor(method='{'spearman' if name == 'spearmanr' else 'kendall'}'), {rversion}",
             "R value parsed from 17-significant-digit CSV" + ("; input np.round(x, 1)" if tag == "R-cor-ties" else ""),
             max_abs_R_vs_independent_impl=float(np.nanmax(np.abs(M - mine))), **exact_extra(S))
        return

    if tag == "labels":
        L = np.asarray(S)
        fresh, sqd, after, calc_sq, calc_plain, same_obj = L
        ok = (list(fresh[:5]) == [1, 0, 1, 0, 6]) and calc_sq[0] >= 1 and calc_sq[1] >= 1 \
            and calc_plain[0] >= 1 and calc_plain[1] >= 1 and same_obj[0] == 1 and after[0] == 2
        save(rid, L, "qualitative", "analytic",
             "expected label state from pyspi/statistics/basic.py class attribute + list += mutation",
             "rows: fresh default, squared, default-after-squared, Calculator -sq, Calculator plain, "
             "[same list object]; cols [n signed, n unsigned, has linear, has nonlinear, len]. "
             "pass = bug 4 reproduced (Calculator instances carry signed AND unsigned, shared list) "
             "and the class label 'linear' (not the documented 'nonlinear') present",
             pass_flag=int(ok), fresh_has_linear=fresh[2], fresh_has_nonlinear=fresh[3],
             calc_sq_n_signed=calc_sq[0], calc_sq_n_unsigned=calc_sq[1],
             calc_plain_n_signed=calc_plain[0], calc_plain_n_unsigned=calc_plain[1], shared_list=same_obj[0])
        return

    if tag == "call-count":
        cnt, P, half, a = np.asarray(S)
        ok = cnt == P * (P - 1) and a == 0
        save(rid, np.array([P * (P - 1), P, half, 0.0]), "qualitative", "analytic",
             "pyspi/base.py:91-93 loop over ordered pairs, :136-137 mirror",
             "pass = one multivariate call makes P(P-1) bivariate calls (P(P-1)/2 would suffice) and output exactly symmetric",
             pass_flag=int(ok), bivariate_calls=cnt, needed_calls=half, pyspi_max_asym=a)
        return

    if tag == "realisations":
        b_ndim, b_size, b_val, raised, a01 = np.asarray(S)
        x3 = split_realisations(x)
        if name == "spearmanr":
            ok = b_ndim == 2 and b_size == 16 and raised == 1
            note = ("pass = hypothesis confirmed: with R = 2 realisations spearmanr.correlation is a 4x4 matrix "
                    "and multivariate raises")
            save(rid, np.array([2, 16, np.nan, 1, np.nan]), "qualitative", "analytic",
                 "expected from scipy spearmanr column-stacking (T, R) inputs", note, pass_flag=int(ok),
                 bivariate_ndim=b_ndim, bivariate_size=b_size, multivariate_raised=raised)
        else:
            concat = stat("kendalltau", x3[0].ravel(), x3[1].ravel())
            ok = abs(b_val - concat) <= 1e-12 and abs(a01 - concat) <= 1e-12
            per_real = [stat("kendalltau", x3[0, :, r], x3[1, :, r]) for r in range(2)]
            note = ("pass = hypothesis confirmed: with R = 2 kendalltau ravels the realisations together "
                    "(value = tau_b of the pooled samples)")
            save(rid, np.array([0, 1, concat, 0, concat]), "qualitative", "independent-impl",
                 est + " on the pooled realisations", note, pass_flag=int(ok), pyspi_value=b_val,
                 pooled_tau_b=concat, realisation0_tau_b=per_real[0], realisation1_tau_b=per_real[1])
        return

    if tag == "array-input":
        v_arr, v_data, m0, s0, m1, s1 = np.asarray(S)
        ref = stat(name, x[0], x[1])
        raw_m1, raw_s1 = x[1].mean(), x[1].std(ddof=1)
        ok = v_arr == v_data and abs(v_arr - ref) <= 1e-12 and abs(m0) < 1e-12 and abs(s0 - 1) < 1e-12 \
            and abs(m1 - raw_m1) < 1e-12 and abs(s1 - raw_s1) < 1e-12
        save(rid, np.array([ref, ref, 0, 1, raw_m1, raw_s1]), "qualitative", "independent-impl",
             est + " on raw rows; raw row-1 mean/std",
             "pass = array-input value equals Data-input value and reference, while parse_bivariate z-scores row 0 "
             "and appends row 1 raw (unequal preprocessing, no effect on ranks)",
             pass_flag=int(ok), value_array_input=v_arr, value_data_input=v_data, row0_mean=m0, row0_std=s0,
             row1_mean=m1, row1_std=s1)
        return

    raise ValueError(f"no reference for tag {tag!r} ({rid})")


def main():
    manifest = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    idx_path = OUT_PYSPI / "index" / f"{FAMILY}.json"
    index = json.loads(idx_path.read_text()) if idx_path.is_file() else {}
    OUT_REF.mkdir(parents=True, exist_ok=True)
    runs = [r for r in manifest["runs"] if index.get(r["run_id"], {}).get("status") == "ok"]
    rjobs = []
    for run in runs:
        tag = run["run_id"].split("|")[-1]
        if tag in ("R-cor", "R-cor-ties"):
            x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
            X = tie_round(x) if tag == "R-cor-ties" else x
            rjobs.append((run["run_id"], "spearman" if base_stat(run["identifier"]) == "spearmanr" else "kendall", X))
    rmats = run_r(rjobs)
    rversion = r_version() if rjobs else ""
    n_err = 0
    for run in runs:
        try:
            ref_run(run, pyspi_value(run["run_id"]), rversion, rmats)
        except Exception as exc:  # report and continue; a missing reference shows up in compare
            n_err += 1
            print(f"REF ERROR {run['run_id']}: {exc!r}", file=sys.stderr)
    skipped = len(manifest["runs"]) - len(runs)
    print(f"rank_corr refs: {len(runs) - n_err} written, {n_err} errors, {skipped} runs without pyspi output")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
