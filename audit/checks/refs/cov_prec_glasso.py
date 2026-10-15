"""References for the cov_prec_glasso family (COMPARISON venv: numpy/scipy only, no sklearn import).

Units: basic.cov.GraphicalLasso, basic.prec.GraphicalLasso, basic.cov.GraphicalLassoCV,
basic.prec.GraphicalLassoCV.  The manifest audit/checks/manifest/cov_prec_glasso.json tags every
run_id with its check ("<unit>|<dataset>|<T>|<path>|<tag>"); this script writes
out/ref/<run_id>.npz for each.

Problem solved by pyspi (sklearn 1.9.1 GraphicalLasso / GraphicalLassoCV, library defaults):
    Theta = argmin_{Theta > 0}  -log det Theta + tr(S Theta) + alpha * sum_{i != j} |Theta_ij|
with S = biased (divide-by-T) covariance of the (T, P) input (sklearn empirical_covariance,
_empirical_covariance.py:104-107) and the diagonal NOT penalised (_graph_lasso.py:43-65).  This is
the Yuan & Lin 2007 form of Friedman, Hastie & Tibshirani 2008 eq. 1; R glasso(penalize.diagonal=FALSE)
solves the same problem with rho = alpha (r/cov_prec_glasso.R header).  KKT (Friedman 2008 eq. 2,
diagonal unpenalised):  W_ii = S_ii;  W_ij = S_ij + alpha sign(Theta_ij) where Theta_ij != 0;
|W_ij - S_ij| <= alpha where Theta_ij = 0;  W = Theta^-1.

Tags (criterion):
  cf       analytic P=2 closed form W12 = soft(S12, alpha), Theta = inv(W)            (tolerance 1e-10 abs)
  admm     from-scratch ADMM (Boyd et al. 2011 §6.5), off-diagonal penalty         (tolerance 1e-3 * max|ref|)
  rglasso  R glasso 1.11 (penalize.diagonal=FALSE, thr=1e-12)                        (tolerance 1e-3 * max|ref|)
  kkt      KKT residual of pyspi's full fit (probe full_fit): prec units use the kernel
           _kernels_basic_misc.glasso_kkt_residual(S, Theta, alpha) (W = inv(Theta)); cov units use
           pyspi's own W with the support/sign of pyspi's Theta                     (optimality, <= 1e-4)
  sym      bit-exact symmetry max|A - A.T| == 0                                     (qualitative)
  perm     permutation equivariance vs the unpermuted pyspi output                  (tolerance 1e-3 * max|ref|)
  fail     Calculator-style outcome (FloatingPointError -> all-NaN) vs the trace prediction (qualitative)
  range    |W_ij| <= (T-1)/T and |W_ij - S_ij| <= alpha + 1e-4                       (qualitative)
  pcorr    documented "partial correlation": sample partial correlation              (tolerance 0.05 abs)
  stat     population glasso target on sparse_precision (ADMM on (T-1)/T * cov_true) (statistical)
  cf-sq / sym-sq   -sq identifier == square of the base identifier's pyspi output    (exact)
  side     cache mutation / refit-count side effects (probe side_effects)           (qualitative)
  degen    CV: all-zero off-diagonal iff alpha_ >= alpha_max(S)                      (qualitative)
  cvsel    CV: alpha_ is the argmax of an independently recomputed contiguous 5-fold
           CV log-likelihood over sklearn's own grid (ADMM fits)                     (qualitative)
  fit      CV: probe fidelity (probe off-diagonals == plain SPI output) + alpha_ record (qualitative)
reference_kind: sym, perm, fail, side, cf-sq/sym-sq and fit compare pyspi with itself (or with the trace
prediction about pyspi's own behaviour) and are labelled same-lineage; they do not carry the verdicts, which rest on
cf (analytic), admm/kkt/cvsel (independent-impl) and rglasso (external-pkg).  range, degen, pcorr and stat compare
pyspi's output with an analytic condition evaluated on the independently recomputed S (analytic).
For GraphicalLassoCV runs alpha is the alpha_ exported by a full_fit probe run on the same
(dataset, T, path) (out/pyspi/<run_id>.npz of that probe).

Run with the comparison venv:  python audit/checks/refs/cov_prec_glasso.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_basic_misc as K  # noqa: E402

FAM = "cov_prec_glasso"
CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / FAM
ALPHA_GL = 0.01                        # sklearn GraphicalLasso default (_graph_lasso.py:522-545)
SK_TOL = 1e-4                          # sklearn tol / enet_tol defaults
SELF = "refs/cov_prec_glasso.py"


# ----------------------------------------------------------------------------- data / helpers
def preprocess(x, path):
    """Mirror pyspi/data.py:184-195 without importing pyspi: optional linear detrend (least squares
    on [1, t], as scipy.signal.detrend type='linear'), then z-score with ddof=1."""
    x = np.asarray(x, float).copy()
    if path == "pipeline-detrend":
        t = np.arange(x.shape[1], dtype=float)
        A = np.vstack([np.ones_like(t), t]).T
        coef, *_ = np.linalg.lstsq(A, x.T, rcond=None)
        x = x - (A @ coef).T
    if path in ("pipeline", "pipeline-detrend"):
        x = (x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)
    return x


def emp_S(xp):
    return K.empirical_cov(xp, ddof=0)        # np.cov(bias=1) convention (sklearn)


def offmask(P):
    return ~np.eye(P, dtype=bool)


def nandiag(A):
    A = np.array(A, float, copy=True)
    np.fill_diagonal(A, np.nan)
    return A


def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def objective(S, Theta, alpha):
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return np.inf
    off = offmask(S.shape[0])
    return float(-logdet + np.sum(S * Theta) + alpha * np.abs(Theta[off]).sum())


def admm_glasso(S, alpha, tol=1e-12, max_iter=20000, Z0=None):
    """Boyd, Parikh, Chu, Peleato & Eckstein 2011, Found. Trends Mach. Learn. 3(1) §6.5
    (sparse inverse covariance selection), with the l1 term restricted to off-diagonals and
    residual balancing of rho (§3.4.1, eq. 3.13) for the first 500 iterations, fixed rho afterwards.  Stopping: the KKT residual of the iterate Z
    (Friedman 2008 eq. 2 via _kernels_basic_misc.glasso_kkt_residual, checked every 5 iterations)
    <= tol * max(1, max|S|), i.e. scale-free; the best iterate by KKT residual is returned if the
    floating-point floor is reached first.  Returns (Theta, W, n_iter, kkt, rho); Theta = Z (exact
    zeros).  alpha == 0 returns inv(S)."""
    S = np.asarray(S, float)
    P = S.shape[0]
    if alpha == 0:
        Th = np.linalg.inv(S)
        return Th, S.copy(), 0, 0.0, 0.0
    off = offmask(P)
    Z = np.diag(1.0 / np.diag(S)) if Z0 is None else Z0.copy()
    U = np.zeros((P, P))
    rho = 1.0
    target = tol * max(1.0, float(np.max(np.abs(S))))
    best, best_Z, best_k = np.inf, Z, 0
    for k in range(max_iter):
        lam, Q = np.linalg.eigh(rho * (Z - U) - S)
        th = (lam + np.sqrt(lam * lam + 4.0 * rho)) / (2.0 * rho)
        Th = (Q * th) @ Q.T
        Zold = Z
        A = Th + U
        Z = A.copy()
        Z[off] = soft(A[off], alpha / rho)
        Z = 0.5 * (Z + Z.T)
        U = U + Th - Z
        r = np.linalg.norm(Th - Z)
        s = rho * np.linalg.norm(Z - Zold)
        if k % 5 == 4:
            try:
                kk = K.glasso_kkt_residual(S, Z, alpha)
            except np.linalg.LinAlgError:
                kk = np.inf
            if kk < best:
                best, best_Z, best_k = kk, Z.copy(), k + 1
            if kk <= target or k + 1 - best_k > 2000:
                break
        if k < 500:                       # balance rho early, then keep it fixed (guaranteed convergence)
            if r > 10 * s:
                rho *= 2.0; U /= 2.0
            elif s > 10 * r:
                rho /= 2.0; U *= 2.0
    W = np.linalg.inv(best_Z)
    return best_Z, 0.5 * (W + W.T), best_k, float(best), rho


def kkt_W(S, W, Theta, alpha):
    """Max KKT violation evaluated on a reported W (not inv(Theta)), with the support/sign of Theta."""
    P = S.shape[0]
    off = offmask(P)
    G = S - W
    nz = off & (Theta != 0)
    z = off & (Theta == 0)
    v = np.zeros_like(G)
    v[nz] = np.abs(G[nz] + alpha * np.sign(Theta[nz]))
    v[z] = np.maximum(np.abs(G[z]) - alpha, 0.0)
    v[~off] = np.abs(G[~off])
    return float(v.max())


def load_pyspi(rid):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return np.asarray(z["value"], float), np.asarray(z["input"], float)


def parse_full_fit(v):
    P = int(v[0])
    h = dict(P=P, err=int(v[1]), n_iter=float(v[2]), alpha=float(v[3]), gap=float(v[4]),
             n_cv=int(v[5]), n_splits=int(v[6]), n_warn=int(v[7]), n_direct=int(v[8]))
    o = 9
    h["W"] = v[o:o + P * P].reshape(P, P); o += P * P
    h["Theta"] = v[o:o + P * P].reshape(P, P); o += P * P
    n = h["n_cv"]
    h["cv_alphas"] = v[o:o + n]; o += n
    h["cv_mean"] = v[o:o + n]; o += n
    h["cv_splits"] = v[o:o + n * h["n_splits"]].reshape(h["n_splits"], n) if n else np.zeros((0, 0))
    return h


def save(rid, **kw):
    out = {}
    for k, v in kw.items():
        out[k] = np.array(v) if not isinstance(v, np.ndarray) else v
    np.savez(OUT_REF / f"{rid}.npz", **out)


def tagof(rid):
    return rid.split("|")[-1]


def kind_of(run):
    return "prec" if run["identifier"].startswith("prec") else "cov"


def est_of(run):
    return run["identifier"].split("_", 1)[1]


# ----------------------------------------------------------------------------- manifest context
manifest = json.loads((CHECKS / "manifest" / f"{FAM}.json").read_text())
RUNS = manifest["runs"]
INDEX = json.loads((OUT_PY / "index" / f"{FAM}.json").read_text())
FITS = {}          # (est, ds, T, path) -> parsed full_fit probe output (first ok one)
FIT_RID = {}
for r in RUNS:
    if r["probe"] and r["probe"]["function"] == "full_fit" and INDEX.get(r["run_id"], {}).get("status") == "ok":
        key = (est_of(r), r["dataset"], r["T"], r["path"])
        if key not in FITS:
            FITS[key] = parse_full_fit(load_pyspi(r["run_id"])[0])
            FIT_RID[key] = r["run_id"]
PLAIN = {}         # (identifier, ds, T, path) -> run_id of a plain (non-probe) run
for r in RUNS:
    if r["probe"] is None:
        PLAIN.setdefault((r["identifier"], r["dataset"], r["T"], r["path"]), r["run_id"])


def alpha_for(run):
    est = est_of(run)
    if est == "GraphicalLasso":
        return ALPHA_GL, "fixed 0.01 (sklearn default)"
    key = (est, run["dataset"], run["T"], run["path"])
    f = FITS[key]
    return f["alpha"], f"alpha_ from probe run {FIT_RID[key]}"


_DATA = {}


def dataset(ds, T):
    if (ds, T) not in _DATA:
        _DATA[(ds, T)] = make_data.load(ds, T)
    return _DATA[(ds, T)]


def S_of(run):
    """Independent S from the raw dataset, checked against pyspi's stored processed input."""
    x = dataset(run["dataset"], run["T"])["x"]
    xp = preprocess(x, run["path"])
    S = emp_S(xp)
    s_dev = np.nan
    if not run["probe"]:                       # plain runs store pyspi's processed input (pyspi_outputs.run_one)
        _, xin = load_pyspi(run["run_id"])
        s_dev = float(np.max(np.abs(emp_S(xin) - S)))
    return S, xp, s_dev


def max_asym(A):
    D = np.abs(A - A.T)
    return float(np.nanmax(D)) if np.isfinite(D).any() else float("nan")


def common(run, S_dev=None, **extra):
    d = dict(reference_detail=extra.pop("reference_detail"), notes=extra.pop("notes", ""))
    if S_dev is not None:
        d["S_vs_pyspi_input_maxabs"] = S_dev
    d.update(extra)
    return d


# ----------------------------------------------------------------------------- per-tag references
def ref_cf(run):
    S, _, sdev = S_of(run)
    alpha, asrc = alpha_for(run)
    assert S.shape == (2, 2)
    W = S.copy()
    W[0, 1] = W[1, 0] = soft(S[0, 1], alpha)
    Th = np.linalg.inv(W)
    if W[0, 1] == 0:
        Th[0, 1] = Th[1, 0] = 0.0
    V = W if kind_of(run) == "cov" else Th
    py, _ = load_pyspi(run["run_id"])
    save(run["run_id"], value=nandiag(V), criterion="tolerance", rtol=0.0, atol=1e-10,
         reference_kind="analytic",
         **common(run, sdev, reference_detail=f"{SELF}:ref_cf (P=2 closed form W12=soft(S12,alpha), Theta=inv(W); "
                  "KKT of Friedman 2008 eq. 2 with unpenalised diagonal)",
                  notes=f"alpha={alpha:.6g} ({asrc}); S12={S[0, 1]:.10g}; 0.95*S12={0.95 * S[0, 1]:.10g}",
                  alpha=alpha, S12=S[0, 1], start_value_0p95S12=0.95 * S[0, 1],
                  pyspi_minus_ref=float(py[0, 1] - V[0, 1]), max_asym=max_asym(py)))


def ref_admm(run):
    S, _, sdev = S_of(run)
    alpha, asrc = alpha_for(run)
    Th, W, it, r, s = admm_glasso(S, alpha)
    kk = K.glasso_kkt_residual(S, Th, alpha)
    V = W if kind_of(run) == "cov" else Th
    off = offmask(S.shape[0])
    py, _ = load_pyspi(run["run_id"])
    atol = 1e-3 * float(np.max(np.abs(V[off]))) if np.any(V[off]) else 1e-8
    zero_ref = int((Th[off] == 0).sum())
    zero_py = int((py[off] == 0).sum())
    save(run["run_id"], value=nandiag(V), criterion="tolerance", rtol=0.0, atol=atol,
         reference_kind="independent-impl",
         **common(run, sdev, reference_detail=f"{SELF}:admm_glasso (Boyd et al. 2011 §6.5, off-diagonal l1, residual "
                  "balancing, stop at KKT residual <= 1e-12 * max(1, max|S|); numpy only)",
                  notes=f"alpha={alpha:.6g} ({asrc}); ADMM iters {it}, KKT residual {kk:.2e}; zeros in ref Theta "
                  f"{zero_ref}, zeros in pyspi {kind_of(run)} {zero_py}",
                  alpha=alpha, admm_iter=it, admm_kkt=kk, ref_theta_zeros=zero_ref, pyspi_zeros=zero_py,
                  max_asym=max_asym(py)))


R_JOBS = {}


def r_name(rid, pen):
    return "j%05d_%d" % (sorted(r["run_id"] for r in RUNS).index(rid), pen)


def prepare_r(run):
    S, _, _ = S_of(run)
    alpha, _ = alpha_for(run)
    for pen in (0, 1):
        name = r_name(run["run_id"], pen)
        np.savetxt(EXCH / f"{name}_S.csv", S, delimiter=",", fmt="%.17g")
        R_JOBS[name] = dict(alpha=alpha, pen=pen)


def run_r():
    EXCH.mkdir(parents=True, exist_ok=True)
    with open(EXCH / "jobs.csv", "w") as f:
        f.write("name,alpha,penalize_diagonal,thr,maxit\n")
        for name, j in sorted(R_JOBS.items()):
            f.write(f"{name},{j['alpha']:.17g},{j['pen']},1e-12,100000\n")
    env = dict(os.environ, R_LIBS_USER="/home/dan/R/audit-lib")
    subprocess.run(["Rscript", str(CHECKS / "r" / f"{FAM}.R"), str(EXCH)], check=True, env=env)
    summ = {}
    with open(EXCH / "r_summary.csv") as f:
        hdr = f.readline().strip().replace('"', "").split(",")
        for line in f:
            vals = line.strip().replace('"', "").split(",")
            summ[vals[0]] = dict(zip(hdr, vals))
    return summ


def ref_rglasso(run, summ):
    S, _, sdev = S_of(run)
    alpha, asrc = alpha_for(run)
    name0, name1 = r_name(run["run_id"], 0), r_name(run["run_id"], 1)
    W = np.loadtxt(EXCH / f"{name0}_w.csv", delimiter=",", ndmin=2)
    Th = np.loadtxt(EXCH / f"{name0}_wi.csv", delimiter=",", ndmin=2)
    W1 = np.loadtxt(EXCH / f"{name1}_w.csv", delimiter=",", ndmin=2)
    Th1 = np.loadtxt(EXCH / f"{name1}_wi.csv", delimiter=",", ndmin=2)
    kk = K.glasso_kkt_residual(S, 0.5 * (Th + Th.T), alpha)
    V = W if kind_of(run) == "cov" else Th
    V1 = W1 if kind_of(run) == "cov" else Th1
    off = offmask(S.shape[0])
    py, _ = load_pyspi(run["run_id"])
    atol = 1e-3 * float(np.max(np.abs(V[off])))
    dpen = float(np.nanmax(np.abs(py - nandiag(V1))))
    diag_shift = float(np.max(np.abs(np.diag(W1) - np.diag(S) - alpha)))
    ver = summ[name0]["glasso_version"]
    save(run["run_id"], value=nandiag(V), criterion="tolerance", rtol=0.0, atol=atol,
         reference_kind="external-pkg",
         **common(run, sdev, reference_detail=f"R glasso=={ver} glasso(S, rho=alpha, penalize.diagonal=FALSE, thr=1e-12, "
                  f"maxit=1e5) via r/{FAM}.R ({summ[name0]['r_version']})",
                  notes=f"alpha={alpha:.6g} ({asrc}); R niter {summ[name0]['niter']}; KKT(R wi) {kk:.2e}; "
                  f"penalize.diagonal=TRUE variant differs from pyspi by {dpen:.3e} off-diagonal and has "
                  f"W_ii - S_ii - alpha = {diag_shift:.1e}",
                  alpha=alpha, r_niter=float(summ[name0]["niter"]), r_kkt=kk,
                  maxabs_vs_R_penalize_diagonal_TRUE=dpen, R_pendiag_diag_shift_minus_alpha=diag_shift,
                  max_asym=max_asym(py)))


def ref_kkt(run):
    v, xin = load_pyspi(run["run_id"])
    f = parse_full_fit(v)
    S, _, sdev = S_of(run)
    detail = (f"{SELF}:ref_kkt + _kernels_basic_misc.glasso_kkt_residual (Friedman, Hastie & Tibshirani 2008 eq. 2, "
              "unpenalised diagonal); pass iff residual <= 1e-4 (sklearn tol/enet_tol = 1e-4 on the dual gap)")
    if f["err"] != 0:
        save(run["run_id"], value=np.nan, criterion="optimality", pass_flag=0, reference_kind="independent-impl",
             **common(run, sdev, reference_detail=detail,
                      notes=f"pyspi fit raised ({'FloatingPointError' if f['err'] == 1 else 'other exception'}); "
                      "Calculator stores an all-NaN matrix; no output to certify", fit_error=f["err"]))
        return
    alpha = f["alpha"]
    W, Th = f["W"], f["Theta"]
    P = S.shape[0]
    off = offmask(P)
    kk_theta = K.glasso_kkt_residual(S, Th, alpha)
    kk_w = kkt_W(S, W, Th, alpha)
    inv_err = float(np.max(np.abs(np.linalg.inv(W) - Th)))
    inv_err_rel = inv_err / float(np.max(np.abs(Th)))
    ThA, WA, it, r, s = admm_glasso(S, alpha)
    f_py = objective(S, Th, alpha)
    f_ref = objective(S, ThA, alpha)
    kk = kk_theta if kind_of(run) == "prec" else kk_w
    diag_dev = float(np.max(np.abs(np.diag(W) - np.diag(S))))
    diag_scale = float(np.max(np.abs(np.diag(W) - (run["T"] - 1) / run["T"]))) if run["path"] != "kernel" else np.nan
    conv = bool(abs(f["gap"]) < SK_TOL)
    dW = float(np.max(np.abs(W[off] - WA[off])))
    dTh = float(np.max(np.abs(Th[off] - ThA[off])))
    note = (f"alpha={alpha:.6g}; KKT residual ({'Theta-based, W=inv(Theta)' if kind_of(run) == 'prec' else 'W-based'}) "
            f"{kk:.3e}; other form {kk_w if kind_of(run) == 'prec' else kk_theta:.3e}; sklearn n_iter {f['n_iter']:.0f}, "
            f"final dual gap {f['gap']:.3e} (tol 1e-4, converged={conv}); warnings reaching the caller {f['n_warn']} "
            f"(ConvergenceWarnings from a direct sklearn fit of the same array: {f['n_direct']}); "
            f"objective excess vs ADMM {f_py - f_ref:.3e}; max|inv(W)-Theta| {inv_err:.3e}; "
            f"max|W-W_admm| {dW:.2e}, max|Theta-Theta_admm| {dTh:.2e}")
    save(run["run_id"], value=np.nan, criterion="optimality", pass_flag=int(kk <= 1e-4),
         reference_kind="independent-impl",
         **common(run, sdev, reference_detail=detail, notes=note,
                  alpha=alpha, kkt_residual=kk, kkt_theta=kk_theta, kkt_w=kk_w, sk_n_iter=f["n_iter"],
                  sk_dual_gap=f["gap"], sk_converged=int(conv), warnings_seen=f["n_warn"],
                  convergence_warnings_direct=f["n_direct"],
                  objective_excess=f_py - f_ref, inv_err=inv_err, inv_err_rel=inv_err_rel,
                  admm_iter=it, admm_kkt=K.glasso_kkt_residual(S, ThA, alpha),
                  maxabs_W_vs_admm=dW, maxabs_Theta_vs_admm=dTh, W_diag_minus_S_diag=diag_dev,
                  W_diag_minus_Tm1_over_T=diag_scale, max_abs_theta_offdiag=float(np.max(np.abs(Th[off])))))


def ref_sym(run):
    py, _ = load_pyspi(run["run_id"])
    a = max_asym(py)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(a == 0.0),
         reference_kind="same-lineage",
         **common(run, reference_detail=f"{SELF}:ref_sym (pass iff max|A - A.T| == 0 bit-exactly; diagonal NaN)",
                  notes=f"max|A-A.T| = {a:.3e}; NaN diagonal {bool(np.isnan(np.diag(py)).all())}; "
                  f"off-diagonal NaN count {int(np.isnan(py[offmask(py.shape[0])]).sum())}",
                  max_asym=a))


def ref_perm(run):
    base = PLAIN[(run["identifier"], run["dataset"], run["T"], run["path"])]
    V, _ = load_pyspi(base)
    off = offmask(V.shape[0])
    py, _ = load_pyspi(run["run_id"])
    atol = 1e-3 * float(np.max(np.abs(V[off])))
    save(run["run_id"], value=V, criterion="tolerance", rtol=0.0, atol=atol, reference_kind="same-lineage",
         **common(run, reference_detail=f"{SELF}:ref_perm (A(x[p])[a,b] mapped to A(x)[p[a],p[b]]; reference = pyspi "
                  f"output of {base}; tolerance 1e-3*max|ref| = solver-accuracy criterion of the admm runs)",
                  notes=f"base run {base}; permutation numpy default_rng(12345).permutation(P)",
                  max_asym=max_asym(py)))


FAIL_EXPECT = {   # (estimator, dataset, T, path) -> expected outcome from the trace stage (records' candidates)
    ("GraphicalLasso", "trending", 4096, "pipeline"): "FloatingPointError",
    ("GraphicalLasso", "trending", 4096, "pipeline-detrend"): "ok",
    ("GraphicalLassoCV", "shared_envelope", 256, "pipeline"): "FloatingPointError",
    ("GraphicalLassoCV", "shared_envelope", 256, "pipeline-detrend"): "FloatingPointError",
    ("GraphicalLassoCV", "shared_envelope", 1024, "pipeline"): "FloatingPointError",
    ("GraphicalLassoCV", "narrowband_f0-0.1", 64, "pipeline"): "FloatingPointError",
}


def ref_fail(run):
    v, _ = load_pyspi(run["run_id"])
    P, err = int(v[0]), int(v[1])
    M = v[2:].reshape(P, P)
    obs = {0: "ok", 1: "FloatingPointError", 2: "other"}[err]
    key = (est_of(run), run["dataset"], run["T"], run["path"])
    exp = FAIL_EXPECT.get(key, "ok")
    src = "trace prediction" if key in FAIL_EXPECT else "no trace prediction: documented behaviour (finite matrix)"
    allnan = bool(np.isnan(M).all())
    consistent = (err == 0 and not np.isnan(M[offmask(P)]).any()) or (err != 0 and allnan)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(obs == exp and consistent),
         reference_kind="same-lineage",
         **common(run, reference_detail=f"{SELF}:ref_fail (Calculator-style outcome from probe calc_style; expected "
                  "outcome per pyspi/calculator.py:327-339 at a1f20c7 and the trace-stage records)",
                  notes=f"observed {obs} (all-NaN matrix: {allnan}); expected {exp} ({src})",
                  observed=obs, expected=exp, all_nan=int(allnan)))


def ref_range(run):
    S, xp, sdev = S_of(run)
    alpha, asrc = alpha_for(run)
    py, _ = load_pyspi(run["run_id"])
    T = run["T"]
    off = offmask(S.shape[0])
    bound = (T - 1) / T
    r = np.corrcoef(xp)
    mx = float(np.max(np.abs(py[off])))
    dS = float(np.max(np.abs(py[off] - S[off])))
    dr = float(np.max(np.abs(py[off] - r[off])))
    ok = mx <= bound and dS <= alpha + SK_TOL
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         **common(run, sdev, reference_detail=f"{SELF}:ref_range (|W_ij| <= sqrt(W_ii W_jj) = (T-1)/T for PD W with "
                  "W_ii = S_ii; KKT |W_ij - S_ij| <= alpha; allowance 1e-4 = sklearn tol)",
                  notes=f"alpha={alpha:.6g} ({asrc}); max|W_ij| {mx:.6f} vs bound {bound:.6f}; max|W-S| {dS:.3e}; "
                  f"max|W - Pearson r| {dr:.3e}",
                  alpha=alpha, max_abs_W=mx, bound=bound, max_W_minus_S=dS, max_W_minus_pearson=dr,
                  max_asym=max_asym(py)))


def ref_pcorr(run):
    S, xp, sdev = S_of(run)
    py, _ = load_pyspi(run["run_id"])
    R = np.corrcoef(xp)
    Om = np.linalg.inv(R)
    d = np.sqrt(np.diag(Om))
    pc = -Om / np.outer(d, d)
    off = offmask(S.shape[0])
    nz = off & (py != 0) & (np.abs(pc) > 1e-3)
    opp = float(np.mean(np.sign(py[nz]) == -np.sign(pc[nz]))) if nz.any() else np.nan
    mx = float(np.max(np.abs(py[off])))
    save(run["run_id"], value=nandiag(pc), criterion="tolerance", rtol=0.0, atol=0.05, reference_kind="analytic",
         **common(run, sdev, reference_detail=f"{SELF}:ref_pcorr (documented claim, supplement §1.1.2: precision "
                  "'equivalent to the partial correlation ... within [-1, 1]'; sample partial correlation "
                  "-Omega_ij/sqrt(Omega_ii Omega_jj), Omega = inv(corrcoef); atol 0.05 allows the l1 shrinkage)",
                  notes=f"fraction of nonzero entries with sign opposite to the partial correlation {opp:.3f}; "
                  f"max|prec_ij| {mx:.3f}",
                  frac_sign_opposite=opp, max_abs_prec=mx, max_asym=max_asym(py)))


def ref_stat(run):
    _, _, sdev = S_of(run)
    alpha, asrc = alpha_for(run)
    ds = dataset(run["dataset"], run["T"])
    T = run["T"]
    Sig = np.asarray(ds["cov_true"], float) * (T - 1) / T      # E[S] for the divide-by-T covariance
    ThA, WA, it, _, _ = admm_glasso(Sig, alpha)
    P = Sig.shape[0]
    # asymptotic sd of the Gaussian MLE entries (precision: (Th_ii Th_jj + Th_ij^2)/T; covariance likewise)
    if kind_of(run) == "prec":
        V = ThA
        se = np.sqrt((np.outer(np.diag(ThA), np.diag(ThA)) + ThA ** 2) / T)
    else:
        V = WA
        se = np.sqrt((np.outer(np.diag(WA), np.diag(WA)) + WA ** 2) / T)
    m = P * (P - 1) // 2
    from scipy.stats import norm
    zb = float(norm.ppf(1 - 0.05 / (2 * m)))
    py, _ = load_pyspi(run["run_id"])
    off = offmask(P)
    prec_true = np.asarray(ds["precision_true"], float)
    nn = off & (prec_true != 0)
    far = off & (prec_true == 0)
    extra = {}
    if kind_of(run) == "prec":
        extra = dict(nn_all_negative=int(bool(np.all(py[nn] < 0))),
                     min_abs_nn=float(np.min(np.abs(py[nn]))), max_abs_far=float(np.max(np.abs(py[far]))),
                     exact_zeros_far=int((py[far] == 0).sum()))
    Ssamp, _, _ = S_of(run)
    sdS = np.sqrt((np.outer(np.diag(Sig), np.diag(Sig)) + Sig ** 2) / T)
    zS = np.abs(Ssamp - Sig)[off] / sdS[off]
    extra["sample_S_max_z_sd"] = float(zS.max())
    save(run["run_id"], value=nandiag(V), criterion="statistical", se=se * zb / 2.0, reference_kind="analytic",
         **common(run, sdev, reference_detail=f"{SELF}:ref_stat (population glasso target: ADMM on (T-1)/T * cov_true "
                  "at the same alpha; se = asymptotic Gaussian-MLE sd, Bonferroni-inflated over the "
                  f"{m} unique entries: se_used = sd * {zb:.3f}/2, i.e. pass iff |z| <= {zb:.3f})",
                  notes=f"alpha={alpha:.6g} ({asrc}); population target ADMM iters {it}; the sample covariance itself "
                  f"deviates from (T-1)/T*cov_true by up to {float(zS.max()):.2f} sd (data property, no glasso involved)",
                  alpha=alpha, bonferroni_z=zb, max_asym=max_asym(py), **extra))


def ref_sq(run):
    base_id = run["identifier"].replace("-sq", "")
    base = PLAIN[(base_id, run["dataset"], run["T"], run["path"])]
    B, _ = load_pyspi(base)
    save(run["run_id"], value=np.square(B), criterion="exact", rtol=1e-15, atol=0.0, reference_kind="same-lineage",
         **common(run, reference_detail=f"{SELF}:ref_sq (np.square of the pyspi output of {base}; basic.py:51-52)",
                  notes=f"base run {base}"))


SIDE_EXPECT_OK = dict(n_fits=1, n_allnan=0, mutated=1, returned_is_cached=1, neutral=1)
SIDE_EXPECT_FAIL = dict(n_fits=4, n_allnan=4, neutral=1)


def ref_side(run):
    v, _ = load_pyspi(run["run_id"])
    obs = dict(n_fits=int(v[0]), n_allnan=int(v[1]), mutated=v[2], returned_is_cached=v[3], neutral=int(v[4]))
    exp = SIDE_EXPECT_FAIL if obs["n_allnan"] else SIDE_EXPECT_OK
    ok = all((obs[k] == e) for k, e in exp.items())
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="same-lineage",
         **common(run, reference_detail=f"{SELF}:ref_side (probe side_effects: Calculator-order compute of cov, cov-sq, "
                  "prec, prec-sq on one Data object with a counting wrapper on sklearn <Estimator>.fit; "
                  "pass iff the trace-stage candidate behaviour is reproduced)",
                  notes="observed " + json.dumps({k: (None if (isinstance(x, float) and math.isnan(x)) else x)
                                                  for k, x in obs.items()}) + "; expected " + json.dumps(exp),
                  n_fits=obs["n_fits"], n_allnan=obs["n_allnan"], diag_mutated=obs["mutated"],
                  returned_is_cached=obs["returned_is_cached"], value_neutral=obs["neutral"]))


def ref_degen(run):
    S, _, sdev = S_of(run)
    alpha, asrc = alpha_for(run)
    py, _ = load_pyspi(run["run_id"])
    off = offmask(S.shape[0])
    amax = float(np.max(np.abs(S[off])))
    n_zero = int((py[off] == 0).sum())
    n_tiny = int(((py[off] != 0) & (np.abs(py[off]) < 1e-15)).sum())
    tiny = np.abs(py[off])[(py[off] != 0) & (np.abs(py[off]) < 1e-15)]
    max_tiny = float(tiny.max()) if tiny.size else 0.0
    all_zero = (n_zero + n_tiny) == int(off.sum())
    # alpha_ is sklearn's alpha_max of its own S when CV picks the top of the grid; the independent S can
    # differ in the last bits, so compare with a relative tolerance of 1e-12
    at_max = alpha >= amax * (1 - 1e-12)
    ok = all_zero == at_max
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         **common(run, sdev, reference_detail=f"{SELF}:ref_degen (KKT: Theta off-diagonal == 0 iff alpha >= "
                  "max|S_offdiag| = alpha_max (sklearn alpha_max, _graph_lasso.py:209-226); pass iff the all-zero "
                  "output occurs exactly when alpha_ >= alpha_max)",
                  notes=f"alpha_={alpha:.6g}, alpha_max={amax:.6g}, ratio {alpha / amax:.15g}; exact zeros {n_zero}, "
                  f"|v|<1e-15 nonzero {n_tiny} of {int(off.sum())} (max tiny {max_tiny:.2e})",
                  alpha=alpha, alpha_max=amax, alpha_at_max=int(at_max), n_exact_zero=n_zero, n_tiny=n_tiny, all_zero=int(all_zero),
                  max_asym=max_asym(py)))


def kfold_contiguous(n, k=5):
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    idx = np.arange(n)
    cur = 0
    for s in sizes:
        test = idx[cur:cur + s]
        train = np.concatenate([idx[:cur], idx[cur + s:]])
        yield train, test
        cur += s


def gauss_ll(S_test, Th):
    sign, logdet = np.linalg.slogdet(Th)
    if sign <= 0:
        return -np.inf
    P = Th.shape[0]
    return float((-np.sum(S_test * Th) + logdet - P * np.log(2 * np.pi)) / 2.0)


def ref_cvsel(run):
    v, _ = load_pyspi(run["run_id"])
    f = parse_full_fit(v)
    _, xp, sdev = S_of(run)
    X = xp.T
    alphas = f["cv_alphas"]
    grid = [(i, a) for i, a in enumerate(alphas) if a > 0]                    # alpha=0 is never selectable
    grid.sort(key=lambda t: -t[1])                                           # descending, as sklearn's path
    folds = list(kfold_contiguous(X.shape[0], 5))
    scores = {}
    for tr, te in folds:
        Str = emp_S(X[tr].T)
        Ste = emp_S(X[te].T)
        Z0 = None
        for i, a in grid:
            Th, _, _, _, _ = admm_glasso(Str, a, tol=1e-10, Z0=Z0)
            Z0 = Th
            scores.setdefault(i, []).append(gauss_ll(Ste, Th))
    best, best_i = -np.inf, None
    for i, a in grid:                                                        # ties -> smaller alpha (>=)
        m = float(np.mean(scores[i]))
        if m >= best:
            best, best_i = m, i
    a_ind = float(alphas[best_i])
    i_sk = int(np.argmin(np.abs(alphas - f["alpha"])))
    gap_at_sk = best - float(np.mean(scores[i_sk]))
    sk_vs_ind = float(np.max([abs(f["cv_mean"][i] - np.mean(scores[i])) for i, _ in grid]))
    asc = np.sort(alphas[alphas > 0])
    j = int(np.searchsorted(asc, f["alpha"]))
    nb = [asc[k] for k in (j - 1, j + 1) if 0 <= k < asc.size]
    step = float(max(f["alpha"] / nb[0] if nb and nb[0] < f["alpha"] else 1.0,
                     (nb[-1] / f["alpha"]) if nb and nb[-1] > f["alpha"] else 1.0))
    ok = abs(a_ind - f["alpha"]) <= 1e-12 * max(1.0, abs(f["alpha"]))
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
         **common(run, sdev, reference_detail=f"{SELF}:ref_cvsel (contiguous KFold(5) as sklearn KFold(shuffle=False); "
                  "ADMM fits on each training fold (biased covariance centred on the training mean); test score = "
                  "Gaussian log-likelihood with the test fold centred on its own mean (_graph_lasso.py:676-700, "
                  "_empirical_covariance.py:50-55); argmax over sklearn's own 16-alpha grid, ties to the smaller "
                  "alpha (_graph_lasso.py:1041-1054)); self-consistency of the selection",
                  notes=f"sklearn alpha_={f['alpha']:.6g}; independent argmax {a_ind:.6g}; independent mean-score "
                  f"shortfall at sklearn's alpha_ {gap_at_sk:.3e}; max|sklearn mean score - independent| over grid "
                  f"{sk_vs_ind:.3e}; widest neighbour ratio at alpha_ {step:.3g}",
                  alpha_sklearn=f["alpha"], alpha_independent=a_ind, score_shortfall_at_sklearn_alpha=gap_at_sk,
                  max_score_diff_sklearn_vs_independent=sk_vs_ind, grid_neighbour_ratio=step,
                  sk_n_iter=f["n_iter"], sk_dual_gap=f["gap"]))


def ref_fit(run):
    v, _ = load_pyspi(run["run_id"])
    f = parse_full_fit(v)
    P = f["P"]
    off = offmask(P)
    checks = []
    for kind, M in (("cov", f["W"]), ("prec", f["Theta"])):
        rid = PLAIN.get((f"{kind}_GraphicalLassoCV", run["dataset"], run["T"], run["path"]))
        if rid:
            py, _ = load_pyspi(rid)
            checks.append(bool(np.array_equal(py[off], M[off])))
    ok = f["err"] == 0 and all(checks)
    conv = bool(abs(f["gap"]) < SK_TOL)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="same-lineage",
         **common(run, reference_detail=f"{SELF}:ref_fit (probe full_fit off-diagonals bit-identical to the plain SPI "
                  "runs on the same dataset/T/path; records alpha_)",
                  notes=f"alpha_={f['alpha']:.6g}; n_iter {f['n_iter']:.0f}; dual gap {f['gap']:.3e} (converged={conv}); "
                  f"bit-identical to {len(checks)} plain run(s): {all(checks)}",
                  alpha=f["alpha"], sk_n_iter=f["n_iter"], sk_dual_gap=f["gap"], sk_converged=int(conv),
                  n_plain_compared=len(checks)))


def main():
    OUT_REF.mkdir(parents=True, exist_ok=True)
    EXCH.mkdir(parents=True, exist_ok=True)
    todo = [r for r in RUNS if INDEX.get(r["run_id"], {}).get("status") == "ok"]
    rg = [r for r in todo if tagof(r["run_id"]) == "rglasso"]
    for r in rg:
        prepare_r(r)
    summ = run_r() if rg else {}
    fn = dict(cf=ref_cf, admm=ref_admm, kkt=ref_kkt, sym=ref_sym, perm=ref_perm, fail=ref_fail,
              range=ref_range, pcorr=ref_pcorr, stat=ref_stat, side=ref_side, degen=ref_degen,
              cvsel=ref_cvsel, fit=ref_fit)
    fn["cf-sq"] = ref_sq
    fn["sym-sq"] = ref_sq
    n = 0
    for r in todo:
        tag = tagof(r["run_id"])
        if tag == "rglasso":
            ref_rglasso(r, summ)
        else:
            fn[tag](r)
        n += 1
    print(f"{FAM}: wrote {n} reference files ({len(RUNS) - len(todo)} runs without pyspi output)")


if __name__ == "__main__":
    main()
