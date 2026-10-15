"""References for the coint family (COMPARISON venv; never imports statsmodels or pyspi).

Units: misc.coint.johansen_order-{0,1}_ardiff-{1,10} (identifiers coint_johansen_{max_eig,trace}_stat_*)
and misc.coint.aeg (coint_aeg_tstat_trend-{c,ct}_autolag-{aic,bic}_maxlag-10). pyspi calls
statsmodels coint_johansen(z[[i, j]].T, det_order, k_ar_diff) and keeps max_eig_stat[0] /
trace_stat[0], or stattools.coint(z[i], z[j], trend, maxlag, autolag) and keeps the t-statistic
(pyspi/statistics/misc.py:62-103). Under multivariate the first call for a pair a < b is
bivariate(i=b, j=a) (base.py:85-94, :132-138) and misc.py:95 caches it for (a, b), so every stored
entry is the statistic with the HIGHER index first (Johansen column 0; aeg dependent variable y0).

Handlers, keyed by the run_id tag ("<unit>|<dataset>|<T>|<path>|<tag>"):

own        independent-impl  Johansen: _kernels_basic_misc.johansen (statsmodels placement of the
                             deterministic terms, numpy solve/eigvals) on [z_hi, z_lo]; aeg:
                             _kernels_basic_misc.engle_granger_tstat(y0 = z_hi, y1 = z_lo). Exact.
canon      independent-impl  Textbook Johansen reduced-rank regression written here (rrr_case):
                             R0, R1 = residuals of dy_t and of the (possibly augmented) y_{t-1} on the
                             unrestricted regressors, generalized symmetric eigenproblem
                             scipy.linalg.eigh(S10 S00^-1 S01, S11). Case 3 (unrestricted constant,
                             Johansen-Juselius H1) is the reference for det_order 0 (tolerance 1e-9);
                             case 4 (restricted linear trend, H*) for det_order 1 (expected to differ:
                             the gap is the measurement); case 2/5 and pre-detrended case 3 as extras.
urca       external-pkg      R urca 1.3.3 ca.jo (r/coint.R jo): ecdet "none" for det_order 0, ecdet
                             "trend" for det_order 1 (closest native case; expected to differ), every
                             ecdet x spec variant reported as extra keys.
urcadt     external-pkg      ca.jo ecdet "none" on full-sample OLS-detrended levels (det_order 1 only).
urca (aeg) external-pkg      R lm residuals + urca ur.df(type "none", selectlags "Fixed") at the lag
                             this script's lag search selects; urca's own AIC/BIC search as extras.
rev        independent-impl  aeg with the reverse orientation (y0 = z_lo): expected to differ.
stat       analytic          truth (cointegrated: only pair (0,1) cointegrated) vs statsmodels' tabulated
                             critical values (crit probe): pass iff A[0,1] rejects at 5% and is the
                             most extreme entry.
crit       literature-only   echoes the tables (statsmodels copies of MHM / MacKinnon 2010); qualitative.
cvmc       independent-impl  Monte Carlo quantiles of textbook case 3 (drifting walks) / case 5 (walks with a
                             quadratic trend) vs statsmodels' det_order 0 / 1 tables (statistical, bootstrap se).
size       independent-impl  null Monte Carlo (probe size): the per-replicate pyspi statistics vs this
                             file's kernel on the re-simulated draws (exact); rejection rate at the
                             tabulated 5% value and the Monte Carlo quantile are extra keys.
perm       independent-impl  processes reversed: kernel on the permuted data (higher permuted index first).
orderdep   same-lineage      processes reversed: pyspi's unpermuted output relabelled (expected to
                             differ for aeg: visiting-order dependence).
ident      analytic          trace - max_eig = -m ln(1 - lambda_2): pyspi max_eig + kernel term.
affine     analytic          scaled_offset (s*D2 + o) vs the kernel on bivariate_gaussian (D2).
trendinv   analytic          trending (D2 + slope t) vs the kernel on bivariate_gaussian (D2).
confound   analytic          proposition: every independent stationary pair rejects "no cointegration".
onefit / calc / collinear / degen / cplx / meta / api  qualitative or exact readings of probe outputs.

Run by run_checks.py; writes out/ref/<run_id>.npz.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import linalg as sla

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))
import make_data  # noqa: E402
import _kernels_basic_misc as K  # noqa: E402

FAMILY = "coint"
OUTP = CHECKS / "out" / "pyspi"
OUTR = CHECKS / "out" / "ref"
EXCH = CHECKS / "out" / "exchange" / FAMILY
R_SCRIPT = CHECKS / "r" / f"{FAMILY}.R"
SELF = "refs/coint.py"
URCA = "urca==1.3.3 (R 4.6.1)"
SIZE_REPS = 400                       # must equal probes/coint_probes.py SIZE_REPS


def size_seed(T):                     # must equal probes/coint_probes.py size_seed
    return 20261015 + int(T)


# ------------------------------------------------------------------ helpers
def save(rid, value, criterion, kind, detail, notes="", **extra):
    OUTR.mkdir(parents=True, exist_ok=True)
    d = dict(value=np.asarray(value, dtype=float), criterion=criterion, reference_kind=kind,
             reference_detail=detail, notes=notes)
    for k, v in extra.items():
        d[k] = np.asarray(v)
    np.savez(OUTR / f"{rid}.npz", **d)


def pys(rid):
    with np.load(OUTP / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=float)


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)


def lsq_detrend(x):
    """Least-squares removal of [1, t] per row (what scipy.signal.detrend(type='linear') computes)."""
    x = np.asarray(x, float)
    t = np.arange(x.shape[1], dtype=float)
    V = np.column_stack([np.ones_like(t), t])
    return x - (V @ np.linalg.lstsq(V, x.T, rcond=None)[0]).T


def prep(x, path):
    if path == "kernel":
        return np.asarray(x, float).copy()
    if path == "pipeline":
        return zscore(x)
    if path == "pipeline-detrend":
        return zscore(lsq_detrend(x))
    raise ValueError(path)


def load_x(run):
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], float)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    return x


def offdiag(P):
    return ~np.eye(P, dtype=bool)


def pairmat(Z, fn):
    """A[i, j] = A[j, i] = fn(Z[hi], Z[lo]) (pyspi's stored orientation), NaN diagonal."""
    P = Z.shape[0]
    A = np.full((P, P), np.nan)
    for a in range(P):
        for b in range(a + 1, P):
            A[a, b] = A[b, a] = fn(Z[b], Z[a])
    return A


def maxrel(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300)))


def maxabs(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")


def parse_ident(ident):
    if ident is None:
        return None
    if "_johansen_" in ident:
        stat = "max_eig_stat" if "max_eig_stat" in ident else "trace_stat"
        d = int(ident.split("_order-")[1].split("_")[0])
        k = int(ident.split("_ardiff-")[1])
        return dict(method="johansen", stat=stat, d=d, k=k)
    trend = ident.split("_trend-")[1].split("_")[0]
    autolag = ident.split("_autolag-")[1].split("_")[0]
    maxlag = int(ident.split("_maxlag-")[1])
    return dict(method="aeg", trend=trend, autolag=autolag, maxlag=maxlag)


# ------------------------------------------------------------------ Johansen references
def kjoh(y0, y1, d, k):
    """Shared kernel on columns [y0, y1]; returns (max_eig r=0, trace r=0, lam, n)."""
    r = K.johansen(np.vstack([y0, y1]), det_order=d, k_ar_diff=k)
    return float(r["max_eig_stat"][0]), float(r["trace_stat"][0]), r["eig"], r["nobs"]


def jstat(stat, d, k):
    idx = 0 if stat == "max_eig_stat" else 1
    return lambda a, b: kjoh(a, b, d, k)[idx]


def rrr_case(y, k, case):
    """Johansen (1995, Likelihood-Based Inference, ch. 6) reduced-rank regression, written from the
    textbook. y (2, T). Rows t = k+1 .. T-1 (0-based), dy_t = y_t - y_{t-1}, Z_t = [dy_{t-1} .. dy_{t-k}].
    case 2: restricted constant (R1 from [y_{t-1}, 1], no unrestricted constant);
    case 3: unrestricted constant (Johansen-Juselius 1990 H1);
    case 4: restricted linear trend (R1 from [y_{t-1}, t], unrestricted constant; H*);
    case 5: unrestricted constant and trend.
    lambda = generalized eigenvalues of (S10 S00^-1 S01, S11), largest two; n = T - k - 1.
    Returns (max_eig r=0, trace r=0, lam)."""
    y = np.asarray(y, float).T
    T = y.shape[0]
    dy = np.diff(y, axis=0)                      # dy[t-1] = y_t - y_{t-1}
    ts = np.arange(k + 1, T)                     # time index t of the regressand dy_t
    D0 = dy[ts - 1]
    Z = np.column_stack([dy[ts - 1 - i] for i in range(1, k + 1)]) if k else np.zeros((ts.size, 0))
    L1 = y[ts - 1]
    one = np.ones((ts.size, 1))
    tcol = ts.astype(float)[:, None]
    if case == 2:
        U, L1 = Z, np.column_stack([L1, one])
    elif case == 3:
        U = np.column_stack([one, Z])
    elif case == 4:
        U, L1 = np.column_stack([one, Z]), np.column_stack([L1, tcol])
    elif case == 5:
        U = np.column_stack([one, tcol, Z])
    else:
        raise ValueError(case)

    def res(A):
        return A - U @ np.linalg.lstsq(U, A, rcond=None)[0] if U.shape[1] else A

    R0, R1 = res(D0), res(L1)
    n = R0.shape[0]
    S00, S01, S11 = R0.T @ R0 / n, R0.T @ R1 / n, R1.T @ R1 / n
    M = S01.T @ np.linalg.solve(S00, S01)
    lam = np.sort(sla.eigh(0.5 * (M + M.T), S11, eigvals_only=True))[::-1][:2]
    return float(-n * np.log(1 - lam[0])), float(-n * np.sum(np.log(1 - lam))), lam


def canon_stat(stat, k, case, predetrend=False):
    idx = 0 if stat == "max_eig_stat" else 1

    def fn(a, b):
        y = np.vstack([a, b])
        if predetrend:
            y = lsq_detrend(y)
        return rrr_case(y, k, case)[idx]
    return fn


# ------------------------------------------------------------------ Engle-Granger references
def eg(p):
    return lambda y0, y1: K.engle_granger_tstat(y0, y1, trend=p["trend"], maxlag=p["maxlag"],
                                                autolag=p["autolag"])


def eg_lag(y0, y1, trend, maxlag, autolag):
    """Lag the statsmodels/kernel rule selects (common sample of maxlag lags, IC = -2 llf + c k,
    ties -> smallest lag); re-implemented here only to pick the ur.df Fixed lag."""
    T = y0.size
    det = {"c": [np.ones(T)], "ct": [np.ones(T), np.arange(1.0, T + 1)]}[trend]
    Xm = np.column_stack([y1] + det)
    e = y0 - Xm @ np.linalg.lstsq(Xm, y0, rcond=None)[0]
    de = np.diff(e)
    idx = np.arange(maxlag, de.size)
    dep = de[idx]
    X = np.column_stack([e[idx]] + [de[idx - i] for i in range(1, maxlag + 1)])
    best = None
    for L in range(maxlag + 1):
        Xl = X[:, :L + 1]
        u = dep - Xl @ np.linalg.lstsq(Xl, dep, rcond=None)[0]
        nn = dep.size
        llf = -nn / 2 * (np.log(2 * np.pi) + np.log(u @ u / nn) + 1)
        kk = L + 1
        ic = -2 * llf + (2 * kk if autolag == "aic" else np.log(nn) * kk)
        if best is None or (ic, L) < best:
            best = (ic, L)
    return best[1]


# ------------------------------------------------------------------ R exchange
def r_table(mode, X):
    """Run r/coint.R on X (P, T) once per distinct input; rows as dicts (numbers as float)."""
    EXCH.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()[:16]
    fin, fout = EXCH / f"{mode}_{h}_in.csv", EXCH / f"{mode}_{h}_out.csv"
    if not fout.is_file():
        np.savetxt(fin, X.T, delimiter=",", fmt="%.17g")
        env = dict(os.environ, R_LIBS_USER=os.environ.get("R_LIBS_USER", "/home/dan/R/audit-lib"))
        subprocess.run(["Rscript", str(R_SCRIPT), mode, str(fin), str(fout)], check=True, env=env,
                       capture_output=True)
    rows = []
    with open(fout) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k in ("i", "j", "K", "stat", "lam1", "lam2", "dep", "reg",
                                               "p", "tau") else v.strip()) for k, v in r.items()})
    return rows


def jo_mat(rows, P, variant, spec, Kv, typ, order="hilo"):
    A = np.full((P, P), np.nan)
    for r in rows:
        if (r["variant"], r["spec"], int(r["K"]), r["type"], r["order"]) == (variant, spec, Kv, typ, order):
            i, j = int(r["i"]), int(r["j"])
            A[i, j] = A[j, i] = r["stat"]
    return A


# ------------------------------------------------------------------ handlers
def crit_johansen(unit):
    c = pys(f"{unit}|cointegrated|256|pipeline|crit")          # rows: trace, max-eig; cols 90/95/99
    return {"trace_stat": c[0], "max_eig_stat": c[1]}


def crit_aeg(cfg):
    c = pys(f"misc.coint.aeg|cointegrated|256|pipeline|crit-{cfg}")  # rows T 64/256/1024/4096
    return {T: c[n] for n, T in enumerate((64, 256, 1024, 4096))}


def handle_johansen(run, tag, p):
    rid, unit, ds, T, path = run["run_id"], run["unit_id"], run["dataset"], run["T"], run["path"]
    stat, d, k = p["stat"], p["d"], p["k"]
    S = pys(rid)
    base = tag.split("-")[0]
    det_txt = "det_order=%d, k_ar_diff=%d, %s" % (d, k, stat)
    kdet = ("refs/_kernels_basic_misc.py:johansen (levels detrended by a degree-det_order polynomial, "
            "then dy, lagged dy and lagged levels demeaned as statsmodels vecm.py:665-684; numpy "
            "solve/eigvals), on columns [z_hi, z_lo]; " + det_txt)
    if base == "own":
        x = load_x(run)
        Z = prep(x, path)
        A = pairmat(Z, jstat(stat, d, k))
        Asw = pairmat(Z, lambda a, b: jstat(stat, d, k)(b, a))
        save(rid, A, "exact", "independent-impl", kdet,
             notes="column order [z_hi, z_lo] as pyspi's first-visited call; swap_max_rel = kernel "
                   "[z_lo, z_hi] vs [z_hi, z_lo]",
             swap_max_rel=maxrel(Asw, A), pyspi_min=float(np.nanmin(S)), pyspi_max=float(np.nanmax(S)),
             pyspi_sym_max=float(np.nanmax(np.abs(S - S.T))))
    elif base == "canon":
        Z = prep(load_x(run), path)
        if d == 0:
            A = pairmat(Z, canon_stat(stat, k, 3))
            extra = dict(case2_max_rel=maxrel(S, pairmat(Z, canon_stat(stat, k, 2))),
                         case4_max_rel=maxrel(S, pairmat(Z, canon_stat(stat, k, 4))))
            note = ("textbook case 3 (unrestricted constant) is the reference; case2/case4_max_rel = "
                    "pyspi vs the restricted-constant / restricted-trend cases")
            crit_, rt, at = "tolerance", 1e-9, 1e-9
        else:
            A = pairmat(Z, canon_stat(stat, k, 4))
            pre = pairmat(Z, canon_stat(stat, k, 3, predetrend=True))
            c5 = pairmat(Z, canon_stat(stat, k, 5))
            off = offdiag(Z.shape[0])
            extra = dict(case4_max_rel=maxrel(S, A), case4_max_abs=maxabs(S, A),
                         case4_mean_rel=float(np.mean(np.abs(S[off] - A[off]) / np.abs(A[off]))),
                         case5_max_rel=maxrel(S, c5), case5_max_abs=maxabs(S, c5),
                         case5_mean_rel=float(np.mean(np.abs(S[off] - c5[off]) / np.abs(c5[off]))),
                         predetrend_case3_max_rel=maxrel(S, pre),
                         a01_pyspi=float(S[0, 1]), a01_case4=float(A[0, 1]), a01_case5=float(c5[0, 1]),
                         a02_pyspi=float(S[0, 2]), a02_case4=float(A[0, 2]), a02_case5=float(c5[0, 2]))
            note = ("EXPECTED TO DIFFER: textbook case 4 (restricted trend) is the canonical 'constant "
                    "and linear trend' model; statsmodels det_order=1 pre-detrends the levels instead. "
                    "predetrend_case3_max_rel = pyspi vs textbook case 3 on full-sample OLS-detrended "
                    "levels (reproduces the statsmodels scheme)")
            crit_, rt, at = "tolerance", 1e-6, 1e-9
        save(rid, A, crit_, "independent-impl",
             f"{SELF}:rrr_case (scipy.linalg.eigh generalized eigenproblem); " + det_txt,
             notes=note, rtol=rt, atol=at, **extra)
    elif base in ("urca", "urcadt"):
        X = prep(load_x(run), path)
        rows = r_table("jo", X)
        P = X.shape[0]
        Kv, typ = k + 1, ("eigen" if stat == "max_eig_stat" else "trace")
        if base == "urcadt":
            ref_var = "none_dt"
        else:
            ref_var = "none" if d == 0 else "trend"
        A = jo_mat(rows, P, ref_var, "transitory", Kv, typ)
        extra = {}
        for v in ("none", "const", "trend", "none_dt", "none_ut"):
            for sp in ("transitory", "longrun"):
                if v in ("none_dt", "none_ut") and sp == "longrun":
                    continue
                M = jo_mat(rows, P, v, sp, Kv, typ)
                extra[f"maxrel_{v}_{sp}"] = maxrel(S, M)
        extra["lohi_vs_hilo_none_max_rel"] = maxrel(jo_mat(rows, P, "none", "transitory", Kv, typ, "lohi"),
                                                    jo_mat(rows, P, "none", "transitory", Kv, typ))
        best = min((k_ for k_ in extra if k_.startswith("maxrel_")), key=lambda q: extra[q])
        extra["best_variant"] = best
        expected = (d == 0 and base == "urca") or base == "urcadt"
        save(rid, A, "tolerance", "external-pkg",
             f"r/coint.R jo: {URCA} ca.jo(x = [x_hi, x_lo], type='{typ}', ecdet='"
             + ("none' on OLS-detrended levels" if ref_var == "none_dt" else ref_var + "'")
             + f", K={Kv}, spec='transitory'), r = 0 = last @teststat",
             notes=("" if expected else "EXPECTED TO DIFFER (closest native ca.jo case for det_order 1); ")
             + "maxrel_<ecdet>_<spec> = pyspi vs every ca.jo variant",
             rtol=1e-8, atol=1e-8, **extra)
    elif base == "stat":
        cv = crit_johansen(unit)[stat]
        off = offdiag(S.shape[0])
        others = [S[a, b] for a in range(4) for b in range(a + 1, 4) if (a, b) != (0, 1)]
        ok = bool(S[0, 1] > cv[1] and S[0, 1] > max(others))
        save(rid, S, "qualitative", "analytic",
             "make_data D11 is_cointegrated (only pair (0,1)) + statsmodels coint_tables c_sjt/c_sja "
             "(LeSage tables after MacKinnon-Haug-Michelis) via probe crit",
             notes="pass iff A[0,1] exceeds the tabulated 95% value and every non-cointegrated pair",
             pass_flag=int(ok), a01=float(S[0, 1]), cv95=float(cv[1]), null_max=float(max(others)),
             n_null_reject_5pct=int(sum(o > cv[1] for o in others)), n_null=len(others),
             sym_max=float(np.nanmax(np.abs(S - S.T))), offdiag_nan=int(np.isnan(S[off]).sum()))
    elif base == "perm":
        x = load_x(run)
        Z = prep(x, path)
        A = pairmat(Z, jstat(stat, d, k))
        Ao = pys(f"{unit}|cointegrated|{T}|pipeline|stat-{tag.split('-')[1]}")
        perm = list(run["processes"])
        save(rid, A, "exact", "independent-impl", kdet + "; processes reversed",
             notes="relabel_max_abs = max|pyspi(permuted) - pyspi(original)[perm][:, perm]|",
             relabel_max_abs=maxabs(S, Ao[np.ix_(perm, perm)]), sym_max=float(np.nanmax(np.abs(S - S.T))))
    elif base == "ident":
        x = load_x(run)
        Z = prep(x, path)
        ME = pys(f"{unit}|{ds}|{T}|{path}|own-me")

        def term(a, b):
            _, _, lam, n = kjoh(a, b, d, k)
            return float(-n * np.log(1 - lam[1]))
        A = ME + pairmat(Z, term)
        off = offdiag(S.shape[0])
        save(rid, A, "exact", "analytic",
             "trace(0) - max_eig(0) = -m ln(1 - lambda_2) (Johansen 1988 eq. for -2lnQ); lambda_2 from "
             "refs/_kernels_basic_misc.py:johansen; max_eig from pyspi run " + f"{unit}|{ds}|{T}|{path}|own-me",
             notes="value = pyspi max_eig_stat + kernel -m ln(1 - lambda_2)",
             min_trace_minus_maxeig=float(np.min(S[off] - ME[off])), min_maxeig=float(np.min(ME[off])))
    elif base == "confound":
        cv = crit_johansen(unit)[stat]
        off = offdiag(S.shape[0])
        v = S[off]
        m = T - k - 1
        coint01 = pys(f"{unit}|cointegrated|{T}|pipeline|stat-{tag.split('-')[1]}")[0, 1]
        ok = bool(np.all(v > cv[1]))
        save(rid, S, "qualitative", "analytic",
             "proposition: independent jointly stationary pairs reject 'no cointegration' at 5% "
             "(statsmodels tabulated value); cointegrated (0,1) from the stat run at the same T",
             notes="pass iff every off-diagonal entry exceeds the 95% value (the test reports "
                   "cointegration for unrelated stationary series)",
             pass_flag=int(ok), cv95=float(cv[1]), min_offdiag=float(v.min()), median_offdiag=float(np.median(v)),
             median_over_m=float(np.median(v) / m), cointegrated_a01=float(coint01),
             median_over_coint01=float(np.median(v) / coint01))
    elif base in ("affine", "trendinv"):
        D2 = np.asarray(make_data.load("bivariate_gaussian", T)["x"], float)
        A = pairmat(prep(D2, path), jstat(stat, d, k))
        direct = pairmat(prep(load_x(run), path), jstat(stat, d, k))
        expected_fail = base == "trendinv" and d == 0
        save(rid, A, "tolerance", "analytic",
             f"invariance: kernel on bivariate_gaussian (D2) at T={T}; {ds} = "
             + ("scale*D2 + offset" if base == "affine" else "D2 + slope*t (slopes 0, .05, -.02, .1)"),
             notes=("EXPECTED TO DIFFER (det_order 0 has no trend term); " if expected_fail else "")
             + "direct_kernel_max_rel = pyspi vs the kernel on the transformed data itself",
             rtol=1e-8, atol=1e-8, direct_kernel_max_rel=maxrel(S, direct), gap_max_abs=maxabs(S, A))
    elif base == "onefit":
        v = S
        ok = bool(v[0] == v[2] and v[1] == v[2] and v[3] == 0.0)
        save(rid, v, "qualitative", "analytic",
             "probe onefit: coint_johansen call counter around the max_eig then trace SPI on one Data",
             notes="pass iff the trace SPI adds no coint_johansen call (shared cache) and equals a fresh fit",
             pass_flag=int(ok), calls_after_maxeig=float(v[0]), calls_after_trace=float(v[1]),
             n_pairs=float(v[2]), fresh_max_abs=float(v[3]))
    elif base == "calc":
        head, M = S[0], S[1:]
        ok = bool(head[0] == 1 and head[2] == 1)
        save(rid, S, "qualitative", "analytic",
             "probe calc: Calculator.compute on shared_envelope (x2 == x0 bit-identical)",
             notes="pass iff the SPI is reported failed and its whole matrix is NaN (one duplicated "
                   "channel blanks every pair); err_code 1 'Singular matrix', 2 'not positive definite'",
             pass_flag=int(ok), failed=float(head[0]), err_code=float(head[1]),
             all_offdiag_nan=float(head[2]), n_failed_of_11=float(head[3]))
    elif base == "collinear":
        rows = S
        code, val, e1, e2 = rows[:, 3], rows[:, 4], rows[:, 5], rows[:, 6]
        ret = code == 0
        oor = ret & ((val < 0) | (e1 < 0) | (e2 < 0) | (e1 >= 1) | (e2 >= 1))
        ok = bool((~ret).any() and ret.any() and oor.any())
        save(rid, S, "qualitative", "analytic",
             "probe collinear: SPI on [z, 3z + 1] for 4 datasets x T {256, 1024} x channel {0, 1}",
             notes="proposition: the outcome is input-dependent (some raise, some return) and some returned "
                   "values are out of range (negative statistic or eigenvalue outside [0, 1))",
             pass_flag=int(ok), n_cases=int(rows.shape[0]), n_raise=int((~ret).sum()),
             n_raise_singular=int((code == 1).sum()), n_raise_not_pd=int((code == 2).sum()),
             n_return=int(ret.sum()), n_return_out_of_range=int(oor.sum()),
             min_returned=float(np.nanmin(np.where(ret, val, np.nan))) if ret.any() else float("nan"),
             max_returned=float(np.nanmax(np.where(ret, val, np.nan))) if ret.any() else float("nan"))
    elif base == "cplx":
        n, nc, im, ncw, noth = S
        ok = bool(n > 0 and nc == n and im == 0.0)
        save(rid, S, "qualitative", "analytic",
             "probe cplx: numpy.linalg.eig dtype and imaginary parts inside coint_johansen",
             notes="pass iff every eig result is complex128 with all imaginary parts exactly 0",
             pass_flag=int(ok), n_eig=float(n), n_complex=float(nc), max_imag=float(im),
             n_complex_warning=float(ncw), n_other_warning=float(noth))
    elif base == "crit":
        ok = bool(np.all(np.diff(S, axis=1) > 0))
        save(rid, S, "qualitative", "literature-only",
             "statsmodels/tsa/coint_tables.py c_sjt / c_sja (LeSage johansen.m tables after MacKinnon, "
             "Haug & Michelis); not re-read from the paper",
             notes="rows [trace, max-eig] r = 0, cols [90%, 95%, 99%]; pass iff increasing",
             pass_flag=int(ok), trace_cv95=float(S[0, 1]), maxeig_cv95=float(S[1, 1]))
    elif base == "size":
        rng = np.random.default_rng(size_seed(T))
        V = np.full(S.shape, np.nan)
        for r in range(S.shape[0]):
            w = zscore(np.cumsum(rng.standard_normal((2, T)), axis=1))
            me, tr, _, _ = kjoh(w[1], w[0], d, k)
            V[r] = (me, tr)
        cv = crit_johansen(unit)
        M = S.shape[0]
        se = np.sqrt(0.05 * 0.95 / M)
        save(rid, V, "exact", "independent-impl",
             kdet + f"; {M} independent random-walk pairs (default_rng({size_seed(T)}))",
             notes="per-replicate exactness; rej_* = rejection rate at the tabulated 95% value "
                   "(nominal 0.05, binomial se given); q95_* = Monte Carlo 95% quantile",
             rej_maxeig=float(np.mean(S[:, 0] > cv["max_eig_stat"][1])),
             rej_trace=float(np.mean(S[:, 1] > cv["trace_stat"][1])), rej_se=float(se),
             q95_maxeig=float(np.quantile(S[:, 0], 0.95)), q95_trace=float(np.quantile(S[:, 1], 0.95)),
             cv95_maxeig=float(cv["max_eig_stat"][1]), cv95_trace=float(cv["trace_stat"][1]),
             n_reps=M)
    elif base == "cvmc":
        # Which Johansen case do the tables belong to? Simulate the textbook statistic of the case the
        # table should describe, on the DGP that case's asymptotics assume, and compare quantiles.
        Tm, Mm = 4000, 2000          # asymptotic tables: T=1000 left a finite-T offset of ~+0.4 for case 5
        rng = np.random.default_rng(424242 + d)
        tt = np.arange(Tm)
        case = 3 if d == 0 else 5
        st = np.empty((Mm, 2))
        for r in range(Mm):
            e = rng.standard_normal((2, Tm)) + np.array([[0.3], [-0.2]])
            if d == 1:
                e = e + 0.001 * tt
            w = np.cumsum(e, axis=1)
            me, tr, _ = rrr_case(w, 1, case)
            st[r] = (tr, me)
        qs = (0.90, 0.95, 0.99)
        V = np.array([[np.quantile(st[:, c], q) for q in qs] for c in (0, 1)])
        boot = np.empty((200, 2, 3))
        brng = np.random.default_rng(99)
        for b in range(200):
            idx = brng.integers(0, Mm, Mm)
            boot[b] = [[np.quantile(st[idx, c], q) for q in qs] for c in (0, 1)]
        se = boot.std(axis=0, ddof=1)
        save(rid, V, "statistical", "independent-impl",
             f"{SELF}:rrr_case case {case} Monte Carlo, {Mm} reps, T={Tm}, k_ar_diff=1, DGP independent random "
             "walks with drift (0.3, -0.2)" + (" plus a linear trend 0.001 t in the increments" if d == 1 else ""),
             notes="rows [trace, max-eig] r = 0, cols [90%, 95%, 99%]; pyspi side = statsmodels tables "
                   f"(coint_tables c_sjt / c_sja, det_order {d}); se = bootstrap SD of the MC quantile",
             se=se, case=case, mc_reps=Mm, mc_T=Tm)
    elif base == "meta":
        handle_meta(run, S)
    else:
        raise KeyError(tag)


def handle_meta(run, S):
    ok = bool(list(S) == [1.0, 0.0, 0.0, 1.0, 1.0])
    save(run["run_id"], S, "qualitative", "analytic",
         "probe meta: Cointegration().labels, issigned(), and method='Johansen'/'AEG' spellings",
         notes="[has 'nonlinear', has 'linear', issigned, Johansen-spelling error code, AEG-spelling error "
               "code] (code 1 = AttributeError naming _det_order); pass iff [1, 0, 0, 1, 1]",
         pass_flag=int(ok))


def handle_aeg(run, tag, p):
    rid, unit, ds, T, path = run["run_id"], run["unit_id"], run["dataset"], run["T"], run["path"]
    S = pys(rid)
    parts = tag.split("-")
    base, cfg = parts[0], (parts[1] if len(parts) > 1 else None)
    cfg_txt = f"trend={p['trend']}, autolag={p['autolag']}, maxlag={p['maxlag']}"
    kdet = ("refs/_kernels_basic_misc.py:engle_granger_tstat (OLS of y0 on [y1, deterministic], ADF without "
            "deterministic term on the residuals, common-sample IC lag search, re-fit with the chosen lag; "
            "numpy lstsq), y0 = z_hi, y1 = z_lo; " + cfg_txt)
    f = eg(p)
    if base == "own":
        Z = prep(load_x(run), path)
        A = pairmat(Z, f)
        R = pairmat(Z, lambda a, b: f(b, a))
        save(rid, A, "exact", "independent-impl", kdet,
             notes="rev_max_abs = |EG(y0 = z_hi) - EG(y0 = z_lo)| (regression asymmetry)",
             rev_max_abs=maxabs(A, R), pyspi_sym_max=float(np.nanmax(np.abs(S - S.T))),
             pyspi_min=float(np.nanmin(S)), pyspi_max=float(np.nanmax(S)))
    elif base == "urca":
        X = prep(load_x(run), path)
        rows = r_table("df", X)
        P = X.shape[0]
        tab = {(int(r["dep"]), int(r["reg"]), r["trend"], r["sel"], int(r["p"])): r["tau"] for r in rows}
        own_sel = {(int(r["dep"]), int(r["reg"]), r["trend"], r["sel"]): (int(r["p"]), r["tau"])
                   for r in rows if r["sel"] != "Fixed"}
        A = np.full((P, P), np.nan)
        U = np.full((P, P), np.nan)
        lag_pyspi, lag_urca = [], []
        for a in range(P):
            for b in range(a + 1, P):
                L = eg_lag(X[b], X[a], p["trend"], p["maxlag"], p["autolag"])
                A[a, b] = A[b, a] = tab[(b, a, p["trend"], "Fixed", L)]
                pu, tu = own_sel[(b, a, p["trend"], p["autolag"].upper())]
                U[a, b] = U[b, a] = tu
                lag_pyspi.append(L)
                lag_urca.append(pu)
        save(rid, A, "tolerance", "external-pkg",
             f"r/coint.R df: R lm residuals + {URCA} ur.df(type='none', lags=L, selectlags='Fixed') at the "
             f"lag L chosen by {SELF}:eg_lag; y0 = x_hi; " + cfg_txt,
             notes="urca_own_sel_max_abs = pyspi vs ur.df(lags=10, selectlags=AIC/BIC) (urca searches lags "
                   "1..10 on the common sample and does not re-fit)",
             rtol=1e-8, atol=1e-8, urca_own_sel_max_abs=maxabs(S, U),
             n_lag_differs=int(sum(a_ != b_ for a_, b_ in zip(lag_pyspi, lag_urca))),
             lags_pyspi=np.array(lag_pyspi), lags_urca_own=np.array(lag_urca))
    elif base == "rev":
        Z = prep(load_x(run), path)
        R = pairmat(Z, lambda a, b: f(b, a))
        A = pairmat(Z, f)
        save(rid, R, "tolerance", "independent-impl", kdet.replace("y0 = z_hi, y1 = z_lo", "y0 = z_lo, y1 = z_hi"),
             notes="EXPECTED TO DIFFER: reverse orientation (lower index dependent); a01_hi = pyspi's "
                   "orientation EG(y0 = z1 | z0), a01_lo = EG(y0 = z0 | z1)",
             rtol=1e-9, atol=1e-9, max_abs=maxabs(S, R), kernel_vs_pyspi_max_abs=maxabs(S, A),
             a01_hi=float(A[0, 1]), a01_lo=float(R[0, 1]))
    elif base == "stat":
        cv = crit_aeg(cfg)[T]
        others = [S[a, b] for a in range(4) for b in range(a + 1, 4) if (a, b) != (0, 1)]
        ok = bool(S[0, 1] < cv[1] and S[0, 1] < min(others))
        save(rid, S, "qualitative", "analytic",
             "make_data D11 is_cointegrated (only pair (0,1)) + MacKinnon (2010) 5% value for N=2 at nobs "
             "T-1 (statsmodels adfvalues.mackinnoncrit via probe crit)",
             notes="pass iff A[0,1] is below the 5% value and below every non-cointegrated pair",
             pass_flag=int(ok), a01=float(S[0, 1]), cv5=float(cv[1]), null_min=float(min(others)),
             n_null_reject_5pct=int(sum(o < cv[1] for o in others)), n_null=len(others),
             sym_max=float(np.nanmax(np.abs(S - S.T))))
    elif base == "perm":
        Z = prep(load_x(run), path)
        A = pairmat(Z, f)
        save(rid, A, "exact", "independent-impl", kdet + "; processes reversed (y0 = higher PERMUTED index)",
             notes="the visiting rule (higher index = dependent) applied to the permuted channel order")
    elif base == "orderdep":
        Ao = pys(f"{unit}|cointegrated|{T}|pipeline|stat-{cfg}")
        perm = list(run["processes"])
        R = Ao[np.ix_(perm, perm)]
        diff = np.abs(S - R)
        save(rid, R, "tolerance", "same-lineage",
             f"pyspi run {unit}|cointegrated|{T}|pipeline|stat-{cfg} relabelled to the reversed channel order",
             notes="EXPECTED TO DIFFER: pyspi(reversed order) vs pyspi(original order) relabelled; any "
                   "difference is visiting-order dependence of the stored value",
             rtol=1e-9, atol=1e-9, max_abs=float(np.nanmax(diff)),
             n_pairs_differ=int((diff[np.triu_indices(4, 1)] > 1e-9).sum()))
    elif base in ("affine", "trendinv"):
        D2 = np.asarray(make_data.load("bivariate_gaussian", T)["x"], float)
        A = pairmat(prep(D2, path), f)
        direct = pairmat(prep(load_x(run), path), f)
        expected_fail = base == "trendinv" and p["trend"] == "c"
        save(rid, A, "tolerance", "analytic",
             f"invariance: kernel on bivariate_gaussian (D2) at T={T}; {ds} = "
             + ("scale*D2 + offset" if base == "affine" else "D2 + slope*t (slopes 0, .05, -.02, .1)"),
             notes=("EXPECTED TO DIFFER (trend 'c' has no trend term); " if expected_fail else "")
             + "direct_kernel_max_abs = pyspi vs the kernel on the transformed data itself",
             rtol=1e-8, atol=1e-8, direct_kernel_max_abs=maxabs(S, direct), gap_max_abs=maxabs(S, A))
    elif base == "degen":
        off = offdiag(S.shape[0])
        dup = np.zeros_like(off)
        dup[0, 2] = dup[2, 0] = True
        ok = bool(np.all(np.isneginf(S[dup])) and np.all(np.isfinite(S[off & ~dup])))
        save(rid, S, "qualitative", "analytic",
             "shared_envelope x2 == x0 (bit-identical): statsmodels returns -inf when R^2 >= 1 - 100 sqrt(eps) "
             "(_stattools.py:2690-2706)",
             notes="pass iff A[0,2] = A[2,0] = -inf and every other off-diagonal entry is finite",
             pass_flag=int(ok), a02=float(S[0, 2]))
    elif base == "collinear":
        code, val = S[:, 3], S[:, 4]
        ok = bool(np.all(code == 0) and np.all(np.isneginf(val)))
        save(rid, S, "qualitative", "analytic",
             "probe collinear: SPI on [z, 3z + 1] for 4 datasets x T {256, 1024} x channel {0, 1}",
             notes="pass iff every case returns -inf without raising",
             pass_flag=int(ok), n_cases=int(S.shape[0]), n_neginf=int(np.isneginf(val).sum()),
             n_raise=int((code != 0).sum()))
    elif base == "crit":
        ok = bool(np.all(np.diff(S, axis=1) > 0) and np.all(np.diff(S[:, 1]) > 0))
        save(rid, S, "qualitative", "literature-only",
             "statsmodels/tsa/adfvalues.py mackinnoncrit(N=2, regression=trend, nobs=T-1) (MacKinnon 2010 "
             "response surface); not re-read from the paper",
             notes="rows T 64/256/1024/4096, cols [1%, 5%, 10%]; pass iff increasing across levels and the "
                   "5% value rises toward its asymptote with T",
             pass_flag=int(ok), cv5_T64=float(S[0, 1]), cv5_T256=float(S[1, 1]), cv5_T1024=float(S[2, 1]),
             cv5_T4096=float(S[3, 1]))
    elif base == "size":
        rng = np.random.default_rng(size_seed(T))
        V = np.full(S.shape, np.nan)
        for r in range(S.shape[0]):
            w = zscore(np.cumsum(rng.standard_normal((2, T)), axis=1))
            V[r, 0] = f(w[1], w[0])
        cv = crit_aeg(cfg)[T]
        M = S.shape[0]
        save(rid, V, "exact", "independent-impl",
             kdet + f"; {M} independent random-walk pairs (default_rng({size_seed(T)})), y0 = walk 1",
             notes="per-replicate exactness; rej = rejection rate at the MacKinnon 5% value (nominal 0.05)",
             rej=float(np.mean(S[:, 0] < cv[1])), rej_se=float(np.sqrt(0.05 * 0.95 / M)),
             q05=float(np.quantile(S[:, 0], 0.05)), cv5=float(cv[1]), n_reps=M)
    elif base == "confound":
        cv = crit_aeg(cfg)[T]
        off = offdiag(S.shape[0])
        v = S[off]
        coint01 = pys(f"{unit}|cointegrated|{T}|pipeline|stat-{cfg}")[0, 1]
        ok = bool(np.all(v < cv[1]))
        save(rid, S, "qualitative", "analytic",
             "proposition: independent jointly stationary pairs reject 'no cointegration' at 5% "
             "(MacKinnon 2010 value); cointegrated (0,1) from the stat run at the same T",
             notes="pass iff every off-diagonal entry is below the 5% value",
             pass_flag=int(ok), cv5=float(cv[1]), max_offdiag=float(v.max()), median_offdiag=float(np.median(v)),
             median_over_sqrtT=float(np.median(v) / np.sqrt(T)), cointegrated_a01=float(coint01))
    elif base == "api":
        Z = prep(load_x(run), path)
        b01 = f(Z[0], Z[1])
        m01 = f(Z[1], Z[0])
        V = np.array([b01, m01, m01, b01])
        save(rid, V, "exact", "independent-impl", kdet,
             notes="[bivariate(D) (y0 = z0), multivariate(D)[0,1], multivariate(D)[1,0], multivariate(D'')[0,1] "
                   "after bivariate(D'') on the same Data]; reference = [EG(z0|z3), EG(z3|z0), EG(z3|z0), EG(z0|z3)]",
             entry_point_gap=float(abs(S[0] - S[1])))
    elif base == "meta":
        handle_meta(run, S)
    else:
        raise KeyError(tag)


def main():
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_err = 0
    for run in man["runs"]:
        rid = run["run_id"]
        tag = rid.split("|")[4]
        try:
            p = parse_ident(run["identifier"])
            if p is None or p["method"] == "johansen" or run["unit_id"] != "misc.coint.aeg":
                if run["unit_id"] == "misc.coint.aeg":
                    handle_meta(run, pys(rid))
                elif p is None:
                    handle_meta(run, pys(rid))
                else:
                    handle_johansen(run, tag, p)
            else:
                handle_aeg(run, tag, p)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
    print(f"refs/{FAMILY}.py: {n_ok} references written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
