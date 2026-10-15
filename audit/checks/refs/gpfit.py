"""References for the gpfit family (pyspi/statistics/misc.py:129-147 GPModel).

Runs in the COMPARISON venv.  Every correctness reference below is numpy/scipy only and does
not import scikit-learn (the library GPModel delegates to); the single exception is the
``scipy118`` tag, which re-runs sklearn's own fit under this venv's SciPy to measure
environment sensitivity and is labelled same-lineage.

pyspi: A[i, j] = in-sample MSE of the posterior mean of sklearn's GaussianProcessRegressor
fitted with input x_i and target x_j (misc.py:145-147); kernel ConstantKernel() +
WhiteKernel() + {DotProduct | RBF}(), hyperparameters theta = log[c, w, l | sigma_0] by one
L-BFGS-B run from theta = 0 within [1e-5, 1e5] (sklearn _gpr.py:299-340), jitter 1e-10.
Every reference matrix here is oriented the same way: M[i, j] = f(input x_i -> target x_j).

Tags (last field of the run_id) and their references
  scratch     DotProduct: weight-space Bayesian linear regression (R&W 2006 §2.1 eqs. 2.8-2.11;
              Phi = [1, x], prior diag(a, 1) with a = c + sigma_0^2, noise w + 1e-10), type-II
              ML over (a, w) on the sklearn box by grid + L-BFGS-B + Nelder-Mead polish
              (blr_ml2); tolerance rtol 1e-8.
              RBF: _kernels_basic_misc.gp_fit (own L-BFGS-B single start from theta = 0,
              R&W eqs. 5.8-5.9) + gp_posterior_mean; tolerance rtol 1e-6 (same basin).
  single      DotProduct, kernel path: own single L-BFGS-B start from theta = 0 (the start sklearn uses)
              with _kernels_basic_misc.gp_fit + gp_posterior_mean (no sklearn), next to the
              global blr_ml2 optimum; separates the optimiser's stopping point from the model;
              tolerance rtol 1e-6 (as RBF scratch).
  closedform  DotProduct: analytic ((T-1)/T)(1 - r^2), r = numpy Pearson on the preprocessed
              rows; tolerance rtol 1e-4 (T = 64) / 1e-5 (T >= 256) (record's a-priori bound).
  params      probe run (probes/gpfit_probes.py:params): sklearn's fitted theta and LML;
              reference = local optimality of that theta under the independent LML
              (_kernels_basic_misc.gp_log_marginal_likelihood): projected gradient and the LML
              gain of a tight L-BFGS-B polish from it; pass iff gain <= 1e-3 for every pair.
  posterior   _kernels_basic_misc.gp_posterior_mean at the probe's theta (same dataset/T/path
              params run); exact rtol 1e-8, atol 1e-12.
  postmean    probe run (probes/gpfit_probes.py:postmean, pair input 0 -> target 1): sklearn's theta,
              LML, in-sample MSE and the posterior-mean vector f_bar; reference = the same vector
              recomputed at that theta with _kernels_basic_misc.gp_log_marginal_likelihood and
              gp_posterior_mean (theta entries are copied, so they agree trivially); exact
              rtol 1e-8, atol 1e-10 elementwise.
  ridgecf     DotProduct: analytic ridge (Bayesian linear regression posterior mean) at the
              probe's theta: [[T + w'/a, Sx], [Sx, Sxx + w']] b = [Sy, Sxy]; exact rtol 1e-8.
  identity    RBF: in-sample residual identity y - f = w' K^-1 y => MSE = w'^2 |K^-1 y|^2 / T at
              the probe's theta (theta from sklearn => same-lineage); exact rtol 1e-10, plus
              leave-one-out MSE (R&W eq. 5.12) as extra.
  multistart  (RBF extra: for best fits with l <= 0.1 x min input spacing and distinct inputs, the
              in-sample MSE vs the mechanism formula (w'/(1 + w'))^2 mean(y^2), w' = w + 1e-10)
  multistart  own multi-start ML-II (theta = 0 plus 20 log-uniform starts at every T, seed
              20261015) with the independent LML; value = MSE at the best theta;
              optimality pass iff LML(best) - LML(pyspi theta) <= 1e-3 for every pair.
  direction   orientation: A[i, j] closer (log scale) to own f(x_i -> x_j) than f(x_j -> x_i)
              where those differ by > 1%; mechanism on anm / nonlinear_dependence:
              A[cause, effect] < A[effect, cause]; DotProduct pipeline: no direction
              information expected (max|A - A^T| <= 1e-12).  qualitative.
  affine      scaled_offset (pipeline) vs pyspi's own bivariate_gaussian matrix (z-scoring
              removes scale and offset); same-lineage tolerance rtol 1e-8.
  gaussexp    RBF: analytic Gaussian expectation ((T-1)/T)(1 - r^2); tolerance rtol 0.1
              (T = 256) / 0.05 (T = 1024).
  vsridge     RBF vs the linear fit on the same pair: pass iff A_RBF <= A_Ridge + 1e-12 for
              every pair; A_Ridge = analytic ((T-1)/T)(1 - r^2 (T^2-1)/T^2), cross-checked
              against pyspi's lmfit_Ridge outputs (out/pyspi/misc.lmfit.Ridge|<ds>|<T>|pipeline|closedform).
  determinism probe: 5 recomputations (two multivariate calls, random_state 0 and 12345,
              after reseeding numpy/random); pass iff bit-identical.  same-lineage.
  warnings    probe: sklearn warnings per fit vs warnings escaping GPModel.bivariate.
  scipy118    sklearn 1.9.1 fit under SciPy 1.18.1 (this venv) vs pyspi under SciPy 1.17.1;
              same-lineage tolerance rtol 1e-6.  Both sides run SciPy's OpenBLAS on one thread
              (this script pins every BLAS to one thread, see below; pyspi/__init__.py:4 does the
              same for SciPy's OpenBLAS on the pyspi side), so the difference is the SciPy version
              and not the thread count (the thread-count effect is measured by ``crossproc``).
  crossproc   probe (probes/gpfit_probes.py:crossproc): pyspi recomputed in five fresh project-venv
              processes from the raw rows under different import orders / BLAS thread counts, vs
              the stored output of the same dataset's plain SPI run (sibling ``posterior``) computed
              inside pyspi_outputs.py; qualitative: pass iff the two 'pipeline'-order processes are
              bit-identical to the stored value and every mode stays within rtol 1e-6 of it (same
              basin).  same-lineage (pyspi vs pyspi).
  fixedhp     probe (probes/gpfit_probes.py:fixedhp): sklearn with optimizer=None, i.e. the documented
              values (c = w = 1, l | sigma_0 = 1) held fixed; reference = _kernels_basic_misc
              gp_posterior_mean at theta = 0 (no sklearn); exact rtol 1e-8.  Extras quantify the
              fixed-vs-fitted difference against pyspi's default output of the same dataset/T
              (sibling posterior | closedform | scratch run) and, for DotProduct, the identity with
              the ridge (alpha = 1) closed form.
  amplitude   probe (probes/gpfit_probes.py:amplitude): sklearn ConstantKernel()*RBF() + WhiteKernel()
              (R&W eq. 2.31), 10 restarts, random_state 0.  Reference (own numpy LML for that kernel,
              no sklearn): at l = 1e-5 (RBF Gram ~ I) the LML depends on C + w only, so the
              interpolating end of that ridge (w = 1e-5, C free) and its white-noise end (C = 1e-5,
              w free) are compared; value = LML(interpolating) - LML(white noise); qualitative: pass
              iff |value| <= 1e-3 on every pair, i.e. the amplitude does NOT remove the
              interpolating mode, it ties with pure white noise.  Extras: the same gap under pyspi's
              C + W + RBF (interpolating l = w = 1e-5 vs flat l = 1e5), own 21-start global optimum
              of C*RBF + W, and what sklearn's restarts returned.
Threads: every BLAS in this (comparison-venv) process is pinned to one thread before numpy is
imported.  With the default (64-thread) OpenBLAS the ill-conditioned shared_envelope fits move
by up to ~5e-9 relative between thread settings, while at a fixed thread count the output is
bit-identical across processes.

Run:  <comparison venv>/bin/python audit/checks/refs/gpfit.py [run_id-substring]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"   # before numpy: deterministic single-thread BLAS (see docstring)

import json
import sys
import warnings
from pathlib import Path

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS / "refs"))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import linalg as sla  # noqa: E402
from scipy import optimize  # noqa: E402
from scipy.signal import detrend  # noqa: E402
from scipy.stats import zscore  # noqa: E402

import make_data  # noqa: E402
import _kernels_basic_misc as K  # noqa: E402

FAMILY = "gpfit"
MANIFEST = CHECKS / "manifest" / f"{FAMILY}.json"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
SELF = "audit/checks/refs/gpfit.py"
KER = "refs/_kernels_basic_misc.py"
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"
LO, HI = np.log(1e-5), np.log(1e5)
ALPHA = 1e-10
MS_SEED = 20261015


# ------------------------------------------------------------------ preprocessing / io
def prep(x, path):
    """Mirror pyspi/data.py:184-197 (detrend, then zscore ddof=1) with scipy primitives."""
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return x
    if path == "pipeline":
        return zscore(x, axis=1, ddof=1)
    if path == "pipeline-detrend":
        return zscore(detrend(x, axis=1), axis=1, ddof=1)
    raise ValueError(path)


def kname(run):
    return run["unit_id"].rsplit(".", 1)[-1]


def load_npz(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], dtype=np.float64)


def sibling(run, tag):
    return "|".join(run["run_id"].split("|")[:-1] + [tag])


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def asym(A):
    return float(np.nanmax(np.abs(A - A.T)))


def pairs(P):
    return [(i, j) for i in range(P) for j in range(P) if i != j]


def write(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in kw.items():
        if isinstance(v, str):
            clean[k] = np.array(v)
        elif isinstance(v, (bool, np.bool_)):
            clean[k] = np.array(int(v))
        elif isinstance(v, (list, dict)):
            clean[k] = np.array(json.dumps(v))
        else:
            clean[k] = np.asarray(v)
    np.savez(OUT_REF / f"{rid}.npz", **clean)


# ------------------------------------------------------------------ function-space GP (kernels module)
def lml(x, y, kn, theta):
    """Independent LML and gradient (R&W eqs. 5.8-5.9) via _kernels_basic_misc; a failed
    Cholesky returns (-inf, 0) as sklearn does (_gpr.py:590-593)."""
    try:
        v, g = K.gp_log_marginal_likelihood(x, y, K._gp_params(kn, np.asarray(theta, float)))
        return v, g
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros(3)


def local_fit(x, y, kn, theta0, tight=False):
    """L-BFGS-B on -LML from theta0 within the sklearn box; default options mirror
    sklearn's call (_gpr.py:659-665); tight=True is the optimality polish."""
    def f(t):
        v, g = lml(x, y, kn, t)
        return (np.inf, np.zeros(3)) if not np.isfinite(v) else (-v, -g)
    opts = dict(ftol=1e-15, gtol=1e-11, maxiter=20000, maxfun=50000) if tight else None
    r = optimize.minimize(f, np.asarray(theta0, float), jac=True, method="L-BFGS-B",
                          bounds=[(LO, HI)] * 3, options=opts)
    return r.x, -float(r.fun), r


def scratch_fit(x, y, kn):
    """Single start from theta = 0 (sklearn's start): _kernels_basic_misc.gp_fit.
    The ``except LinAlgError`` branch below (fallback to local_fit) is unreachable and is left in
    place as dead code: gp_fit evaluates the LML at every optimiser step and at the returned theta
    through _kernels_basic_misc._gp_lml_guarded, which turns the LinAlgError of a failed Cholesky
    of K + alpha I into (-inf, 0) as sklearn does (_gpr.py:590-593).  gp_fit's other calls do not
    raise it either: _gp_params does no linear algebra, and scipy's L-BFGS-B, whose own LAPACK
    Cholesky factorisations of its limited-memory matrices (dpotrf in the compiled _lbfgsb)
    report failure through internal status codes, never as LinAlgError (scipy 1.18.1 in the
    comparison venv: neither the _lbfgsb extension nor _lbfgsb_py.py mentions LinAlgError).  A
    failed Cholesky of K at an intermediate trial theta is only an objective of +inf to the
    optimiser; v = -inf reaches this function only when the failure is at the theta the optimiser
    returns, and never as a fallback.  local_fit maps a failure the same way (objective +inf,
    zero gradient)."""
    try:
        kp, v, _ = K.gp_fit(x, y, kn)
        th = np.log([kp["constant"], kp["noise"], kp["length_scale" if kn == "RBF" else "sigma_0"]])
        return th, v
    except np.linalg.LinAlgError:
        th, v, _ = local_fit(x, y, kn, np.zeros(3))
        return th, v


def mse_at(x, y, kn, theta):
    f = K.gp_posterior_mean(x, y, K._gp_params(kn, np.asarray(theta, float)))
    return float(np.mean((np.ravel(y) - f) ** 2))


def resid_identity(x, y, kn, theta):
    """(identity MSE, LOO MSE) at theta: y - f = w' K^-1 y (K includes w' = w + alpha);
    LOO residual r_t = [K^-1 y]_t / [K^-1]_tt (R&W eq. 5.12)."""
    kp = K._gp_params(kn, np.asarray(theta, float))
    Kx, _ = K._gp_kernel(np.ravel(x), np.ravel(x), kp, True)
    Kx[np.diag_indices_from(Kx)] += ALPHA
    L = np.linalg.cholesky(Kx)
    a = sla.cho_solve((L, True), np.ravel(y))
    Ki = sla.cho_solve((L, True), np.eye(Kx.shape[0]))
    wp = kp["noise"] + ALPHA
    return float(wp ** 2 * (a @ a) / a.size), float(np.mean((a / np.diag(Ki)) ** 2))


def proj_grad(theta, g):
    """Max |projected gradient| of the LML (ascent direction) on the box."""
    g = np.array(g, float)
    g[(theta <= LO + 1e-8) & (g < 0)] = 0.0
    g[(theta >= HI - 1e-8) & (g > 0)] = 0.0
    return float(np.max(np.abs(g)))


def multistart(x, y, kn, n, seed):
    rng = np.random.default_rng(seed)
    starts = [np.zeros(3)] + list(rng.uniform(LO, HI, size=(n, 3)))
    best = (None, -np.inf)
    for s in starts:
        th, v, _ = local_fit(x, y, kn, s)
        if v > best[1]:
            best = (th, v)
    return best


# ------------------------------------------------------------------ weight-space BLR (DotProduct)
def _phi(x):
    x = np.ravel(np.asarray(x, float))
    return np.column_stack([np.ones_like(x), x])


def blr_post(x, y, a, w):
    """Posterior mean weights of y = b0 + b1 x + e, prior N(0, diag(a, 1)), noise variance w
    (R&W eq. 2.8: w_bar = A^-1 Phi^T y / w, A = Phi^T Phi / w + Sigma_p^-1)."""
    Phi, y = _phi(x), np.ravel(np.asarray(y, float))
    A = Phi.T @ Phi / w + np.diag([1.0 / a, 1.0])
    m = np.linalg.solve(A, Phi.T @ y / w)
    return m, A


def blr_lml(la, lw, x, y):
    """log N(y | 0, w' I + Phi diag(a,1) Phi^T), w' = w + alpha, by the matrix determinant lemma
    and the stable quadratic form y^T C^-1 y = |y - Phi m|^2 / w' + m^T Sigma_p^-1 m."""
    a, w = np.exp(la), np.exp(lw) + ALPHA
    y = np.ravel(np.asarray(y, float))
    T = y.size
    m, A = blr_post(x, y, a, w)
    r = y - _phi(x) @ m
    quad = r @ r / w + m[0] ** 2 / a + m[1] ** 2
    sign, ld = np.linalg.slogdet(A)
    return float(-0.5 * quad - 0.5 * (T * np.log(w) + np.log(a) + ld) - 0.5 * T * np.log(2 * np.pi))


def blr_ml2(x, y):
    """Type-II ML over (a = c + sigma_0^2, w) on the box a in [1e-5 + 1e-10, 1e5 + 1e10],
    w in [1e-5, 1e5] (sklearn bounds on c, sigma_0, w): 61 x 61 grid, L-BFGS-B, Nelder-Mead."""
    ba = (np.log(1e-5 + 1e-10), np.log(1e5 + 1e10))
    bw = (LO, HI)
    ga, gw = np.linspace(*ba, 61), np.linspace(*bw, 61)
    G = np.array([[blr_lml(p, q, x, y) for q in gw] for p in ga])
    ia, iw = np.unravel_index(np.nanargmax(G), G.shape)
    f = lambda p: -blr_lml(p[0], p[1], x, y)  # noqa: E731
    r = optimize.minimize(f, [ga[ia], gw[iw]], method="L-BFGS-B", bounds=[ba, bw],
                          options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000))
    r2 = optimize.minimize(f, r.x, method="Nelder-Mead", bounds=[ba, bw],
                           options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000, maxfev=40000))
    p = r2.x if r2.fun <= r.fun else r.x
    a, w = np.exp(p)
    m, _ = blr_post(x, y, a, w + ALPHA)
    mse = float(np.mean((np.ravel(y) - _phi(x) @ m) ** 2))
    return mse, dict(a=a, w=w, lml=-min(r.fun, r2.fun))


def ridge_cf(x, y, a, wp):
    """Analytic penalised least squares = BLR posterior mean: minimise
    |y - b0 - b1 x|^2 + wp (b1^2 + b0^2 / a)."""
    x, y = np.ravel(x), np.ravel(y)
    T = x.size
    M = np.array([[T + wp / a, x.sum()], [x.sum(), x @ x + wp]])
    b = np.linalg.solve(M, np.array([y.sum(), x @ y]))
    return float(np.mean((y - b[0] - b[1] * x) ** 2))


def own_oriented(X, kn):
    P = X.shape[0]
    M = np.full((P, P), np.nan)
    TH = np.full((P, P, 3), np.nan)
    for i, j in pairs(P):
        if kn == "DotProduct":
            M[i, j], info = blr_ml2(X[i], X[j])
        else:
            th, _ = scratch_fit(X[i], X[j], kn)
            TH[i, j] = th
            M[i, j] = mse_at(X[i], X[j], kn, th)
    return M, TH


def r2_closed(X):
    T = X.shape[1]
    R = np.corrcoef(X)
    M = (T - 1) / T * (1 - R ** 2)
    np.fill_diagonal(M, np.nan)
    return M, R


# ------------------------------------------------------------------ handlers
def h_scratch(run, X, P):
    kn, T = kname(run), run["T"]
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    extra = {}
    if kn == "DotProduct":
        A_, W_ = [], []
        for i, j in pairs(Pn):
            M[i, j], info = blr_ml2(X[i], X[j])
            A_.append(info["a"]); W_.append(info["w"])
        extra.update(own_a_range=[float(min(A_)), float(max(A_))], own_w_range=[float(min(W_)), float(max(W_))],
                     n_a_at_lower=int(np.sum(np.array(A_) <= 1.0001e-5)))
        rtol, det = 1e-8, f"{SELF}:blr_ml2 (weight-space Bayesian linear regression, type-II ML over (c + sigma_0^2, w), grid + L-BFGS-B + Nelder-Mead) ({VERS})"
    else:
        L_, W_, LML = [], [], []
        for i, j in pairs(Pn):
            th, v = scratch_fit(X[i], X[j], kn)
            M[i, j] = mse_at(X[i], X[j], kn, th)
            L_.append(np.exp(th[2])); W_.append(np.exp(th[1])); LML.append(v)
        extra.update(own_l_range=[float(min(L_)), float(max(L_))], own_w_range=[float(min(W_)), float(max(W_))])
        rtol, det = 1e-6, f"{KER}:gp_fit (own L-BFGS-B, single start theta=0, sklearn box) + gp_posterior_mean ({VERS})"
    if P is not None:
        rel = np.abs(offdiag(P) - offdiag(M)) / np.abs(offdiag(M))
        extra.update(asym_max=asym(P), n_pairs=int(rel.size), n_rel_gt_1em6=int(np.sum(rel > 1e-6)),
                     n_rel_gt_1em3=int(np.sum(rel > 1e-3)), min_offdiag=float(np.min(offdiag(P))),
                     max_offdiag=float(np.max(offdiag(P))))
        if run.get("processes") is not None:
            extra["processes"] = json.dumps(run["processes"])
    return dict(value=M, criterion="tolerance", rtol=rtol, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{det} on {run['path']} data", notes="M[i,j] = f(input x_i -> target x_j)", **extra)


def h_single(run, X, P):
    kn = kname(run)
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    G = np.full((Pn, Pn), np.nan)
    gap = []
    for i, j in pairs(Pn):
        th, v = scratch_fit(X[i], X[j], kn)
        M[i, j] = mse_at(X[i], X[j], kn, th)
        G[i, j], info = blr_ml2(X[i], X[j])
        gap.append(info["lml"] - v)
    extra = dict(lml_global_minus_single_max=float(max(gap)), lml_global_minus_single_min=float(min(gap)),
                 single_vs_global_mse_max_rel=float(np.max(np.abs(offdiag(M) - offdiag(G)) / offdiag(G))))
    if P is not None:
        extra.update(pyspi_vs_global_mse_max_rel=float(np.max(np.abs(offdiag(P) - offdiag(G)) / offdiag(G))))
    return dict(value=M, criterion="tolerance", rtol=1e-6, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{KER}:gp_fit (own L-BFGS-B single start theta=0, sklearn box, default options) + gp_posterior_mean; "
                                 f"global optimum by {SELF}:blr_ml2 reported alongside ({VERS})",
                notes="M[i,j] = f(input x_i -> target x_j) at the own single-start optimum", **extra)


def h_closedform(run, X, P):
    T = run["T"]
    M, R = r2_closed(X)
    rtol = 1e-4 if T <= 64 else 1e-5
    extra = dict(max_abs_r=float(np.max(np.abs(offdiag(R)))))
    if P is not None:
        rel = np.abs(offdiag(P) - offdiag(M)) / offdiag(M)
        extra.update(asym_max=asym(P), max_rel_vs_r2=float(rel.max()))
    return dict(value=M, criterion="tolerance", rtol=rtol, atol=0.0, reference_kind="analytic",
                reference_detail=f"{SELF}:r2_closed ((T-1)/T)(1 - r^2), r = numpy.corrcoef of the {run['path']} rows ({VERS})",
                notes="a-priori bound from the record: 1e-4 relative at T=64, 1e-5 at T>=256 (ridge shrinkage O(s/T))", **extra)


def _theta_from_params(run):
    Pp = load_npz(sibling(run, "params"))
    return Pp


def h_params(run, x, Pp):
    kn = kname(run)
    X = prep(x, run["path"])
    Pn = X.shape[0]
    V = np.full((Pn, Pn), np.nan)
    flag, extra = 0, {}
    if Pp is None:
        return dict(value=V, criterion="optimality", pass_flag=0, reference_kind="independent-impl",
                    reference_detail="probe output missing", notes="")
    gains, pgs, dl, dth, dmse = [], [], [], [], []
    for i, j in pairs(Pn):
        th = Pp[i, j, 1:4]
        v, g = lml(X[i], X[j], kn, th)
        th2, v2, _ = local_fit(X[i], X[j], kn, th, tight=True)
        V[i, j] = v2 - v
        dmse.append(abs(mse_at(X[i], X[j], kn, th2) - Pp[i, j, 0]) / Pp[i, j, 0])
        gains.append(v2 - v); pgs.append(proj_grad(th, g)); dl.append(abs(v - Pp[i, j, 4]))
        dth.append(float(np.max(np.abs(th2 - th))))
    th_all = Pp[..., 1:4][~np.isnan(Pp[..., 0])]
    flag = int(max(gains) <= 1e-3)
    o = offdiag
    extra.update(n_pairs=len(gains), max_lml_gain_polish=float(max(gains)), max_proj_grad=float(max(pgs)),
                 median_proj_grad=float(np.median(pgs)), max_dtheta_polish=float(max(dth)),
                 lml_own_vs_sklearn_max_abs=float(max(dl)),
                 max_rel_mse_change_polish=float(max(dmse)),
                 n_c_at_lower=int(np.sum(th_all[:, 0] <= LO + 1e-6)),
                 n_p3_at_lower=int(np.sum(th_all[:, 2] <= LO + 1e-6)), n_p3_at_upper=int(np.sum(th_all[:, 2] >= HI - 1e-6)),
                 n_w_at_lower=int(np.sum(th_all[:, 1] <= LO + 1e-6)),
                 w_range=[float(np.exp(th_all[:, 1].min())), float(np.exp(th_all[:, 1].max()))],
                 p3_range=[float(np.exp(th_all[:, 2].min())), float(np.exp(th_all[:, 2].max()))],
                 n_restarts_values=sorted({int(v) for v in o(Pp[..., 5])}),
                 nit_range=[int(o(Pp[..., 7]).min()), int(o(Pp[..., 7]).max())],
                 n_stop_pgtol=int(np.sum(o(Pp[..., 9]) == 0)), n_stop_factr=int(np.sum(o(Pp[..., 9]) == 1)),
                 n_stop_other=int(np.sum(o(Pp[..., 9]) == 2)),
                 mse_vs_recorded_refit_max_abs=float(np.max(np.abs(o(Pp[..., 0]) - o(Pp[..., 6])))),
                 lml_vs_recorded_refit_max_abs=float(np.max(np.abs(o(Pp[..., 4]) - o(Pp[..., 10])))),
                 rng_draws_total=int(np.sum(o(Pp[..., 11]))))
    return dict(value=V, criterion="optimality", pass_flag=flag, reference_kind="independent-impl",
                reference_detail=f"{KER}:gp_log_marginal_likelihood at sklearn's theta (probes/gpfit_probes.py:params), "
                                 f"projected gradient + tight L-BFGS-B polish ({SELF}:local_fit tight) ({VERS})",
                notes="value = LML gain of the polish; pass iff gain <= 1e-3 for every pair (local optimum); "
                      "theta = log[c, w, l|sigma_0]; p3 = l (RBF) or sigma_0 (DotProduct)", **extra)


def h_posterior(run, X, P):
    kn = kname(run)
    Pp = _theta_from_params(run)
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    extra = {}
    if Pp is None:
        return dict(value=M, criterion="exact", reference_kind="independent-impl", reference_detail="params missing")
    for i, j in pairs(Pn):
        M[i, j] = mse_at(X[i], X[j], kn, Pp[i, j, 1:4])
    if P is not None:
        extra.update(probe_mse_vs_spi_max_abs=float(np.nanmax(np.abs(Pp[..., 0] - P))),
                     asym_max=asym(P), min_offdiag=float(np.min(offdiag(P))))
    return dict(value=M, criterion="exact", rtol=1e-8, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{KER}:gp_posterior_mean (Cholesky, R&W Alg. 2.1) at sklearn's fitted theta from "
                                 f"{sibling(run, 'params')} ({VERS})",
                notes="isolates the posterior arithmetic and the MSE; theta itself comes from sklearn's optimiser", **extra)


def h_postmean(run, X, P):
    kn = kname(run)
    if P is None:
        return dict(value=np.array(np.nan), criterion="exact", reference_kind="independent-impl",
                    reference_detail="probe output missing")
    th = P[:3]
    x, y = X[0], X[1]
    v, g = lml(x, y, kn, th)
    f = K.gp_posterior_mean(x, y, K._gp_params(kn, th))
    mse = float(np.mean((y - f) ** 2))
    V = np.concatenate([th, [v, mse], f])
    extra = dict(T=int(f.size), theta=[float(t) for t in th], hyper=[float(t) for t in np.exp(th)],
                 lml_sklearn=float(P[3]), lml_own=v, lml_abs_diff=float(abs(v - P[3])),
                 mse_sklearn=float(P[4]), mse_own=mse, mse_abs_diff=float(abs(mse - P[4])),
                 fbar_max_abs_diff=float(np.max(np.abs(f - P[5:]))), fbar_max_abs=float(np.max(np.abs(f))),
                 proj_grad_at_theta=proj_grad(np.asarray(th), g))
    return dict(value=V, criterion="exact", rtol=1e-8, atol=1e-10, reference_kind="independent-impl",
                reference_detail=f"{KER}:gp_log_marginal_likelihood + gp_posterior_mean (Cholesky, R&W Alg. 2.1, eq. 5.8) at "
                                 f"sklearn's kernel_.theta for pair 0 -> 1 ({VERS})",
                notes="vector = [log c, log w, log l|sigma_0 (copied), lml, mse, f_bar(T)]", **extra)


def h_ridgecf(run, X, P):
    Pp = _theta_from_params(run)
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    if Pp is None:
        return dict(value=M, criterion="exact", reference_kind="analytic", reference_detail="params missing")
    av, wv = [], []
    for i, j in pairs(Pn):
        c, w, s0 = np.exp(Pp[i, j, 1:4])
        a, wp = c + s0 ** 2, w + ALPHA
        av.append(a); wv.append(w)
        M[i, j] = ridge_cf(X[i], X[j], a, wp)
    extra = dict(a_range=[float(min(av)), float(max(av))], w_range=[float(min(wv)), float(max(wv))])
    if P is not None:
        extra["asym_max"] = asym(P)
    return dict(value=M, criterion="exact", rtol=1e-8, atol=1e-12, reference_kind="analytic",
                reference_detail=f"{SELF}:ridge_cf closed-form penalised LS (b1 penalty w', b0 penalty w'/(c + sigma_0^2)) "
                                 f"at sklearn's fitted (c, w, sigma_0) from {sibling(run, 'params')} ({VERS})",
                notes="DotProduct GP = Bayesian linear regression (R&W §2.1-2.2): posterior mean = this ridge solution", **extra)


def h_identity(run, X, P):
    kn = kname(run)
    Pp = _theta_from_params(run)
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    LOO = np.full((Pn, Pn), np.nan)
    for i, j in pairs(Pn):
        M[i, j], LOO[i, j] = resid_identity(X[i], X[j], kn, Pp[i, j, 1:4])
    extra = dict(loo_min=float(np.nanmin(LOO)), loo_max=float(np.nanmax(LOO)))
    if P is not None:
        ratio = offdiag(LOO) / offdiag(P)
        k = int(np.argmin(offdiag(P)))
        extra.update(min_insample=float(np.min(offdiag(P))), loo_at_min_insample=float(offdiag(LOO)[k]),
                     max_loo_over_insample=float(ratio.max()), n_insample_lt_1em3=int(np.sum(offdiag(P) < 1e-3)),
                     loo_matrix=json.dumps(np.round(LOO, 6).tolist()))
    return dict(value=M, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="same-lineage",
                reference_detail=f"{SELF}:resid_identity MSE = w'^2 |K^-1 y|^2 / T at sklearn's theta from {sibling(run, 'params')}; "
                                 f"LOO by R&W eq. 5.12 ({VERS})",
                notes="same-lineage because theta comes from sklearn; documents interpolation (in-sample vs LOO)", **extra)


def h_multistart(run, X, P):
    kn, T = kname(run), run["T"]
    Pp = _theta_from_params(run)
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    n = 20
    gains, rows = [], []
    loo_p, loo_b = [], []
    interp_formula = []
    for k, (i, j) in enumerate(pairs(Pn)):
        thb, vb = multistart(X[i], X[j], kn, n, MS_SEED + k)
        thp = Pp[i, j, 1:4]
        vp, _ = lml(X[i], X[j], kn, thp)
        M[i, j] = mse_at(X[i], X[j], kn, thb)
        gains.append(vb - vp)
        lp, lb = resid_identity(X[i], X[j], kn, thp)[1], resid_identity(X[i], X[j], kn, thb)[1]
        loo_p.append(lp); loo_b.append(lb)
        rows.append((i, j, float(Pp[i, j, 0]), float(M[i, j]), float(vb - vp), float(np.exp(thb[1])), float(np.exp(thb[2])), lp, lb))
        if kn == "RBF":
            dmin = float(np.min(np.diff(np.sort(X[i]))[np.diff(np.sort(X[i])) > 0]))
            if np.exp(thb[2]) <= 0.1 * dmin and np.all(np.diff(np.sort(X[i])) > 0):
                wq = np.exp(thb[1]) + ALPHA
                form = (wq / (1 + wq)) ** 2 * float(np.mean(X[j] ** 2))
                interp_formula.append(abs(M[i, j] - form) / form)
    gains = np.array(gains)
    better = gains > 1e-3
    extra = dict(n_pairs=int(gains.size), n_starts=n + 1, n_better_lml=int(better.sum()), max_lml_gain=float(gains.max()),
                 min_lml_gain=float(gains.min()))
    if P is not None:
        dv = np.abs(offdiag(M) - offdiag(P))
        interp = np.array([r[3] < 1e-6 and r[5] <= 2e-5 for r in rows])
        extra.update(n_value_change_gt_1em3=int(np.sum(dv > 1e-3)), max_value_change=float(dv.max()),
                     n_best_interpolating=int(interp.sum()),
                 n_interp_formula_pairs=len(interp_formula),
                 interp_formula_max_rel=float(max(interp_formula)) if interp_formula else float("nan"),
                     n_best_loo_gt_pyspi_loo=int(np.sum(np.array(loo_b) > np.array(loo_p) + 1e-9)),
                     examples=json.dumps([dict(i=r[0], j=r[1], pyspi=round(r[2], 6), best=float(f"{r[3]:.4g}"),
                                               dlml=round(r[4], 3), w_best=float(f"{r[5]:.3g}"), l_best=float(f"{r[6]:.3g}"),
                                               loo_pyspi=round(r[7], 4), loo_best=round(r[8], 4))
                                          for r in rows if r[4] > 1e-3][:6]))
    return dict(value=M, criterion="optimality", pass_flag=int(not better.any()), reference_kind="independent-impl",
                reference_detail=f"{SELF}:multistart (theta=0 + {n} log-uniform starts in the sklearn box, seed {MS_SEED}+pair; "
                                 f"L-BFGS-B on {KER}:gp_log_marginal_likelihood) vs LML at sklearn's theta from {sibling(run, 'params')} ({VERS})",
                notes="value = in-sample MSE at the best multi-start theta; pass iff no start beats pyspi's fit by > 1e-3 in LML", **extra)


CAUSE = {"anm": [(0, 1), (2, 3)], "nonlinear_dependence": [(0, 1)]}


def h_direction(run, X, P):
    kn, ds = kname(run), run["dataset"]
    M, _ = own_oriented(X, kn)
    Pn = X.shape[0]
    extra, flag = {}, 0
    if P is None:
        return dict(value=M, criterion="qualitative", pass_flag=0, reference_kind="independent-impl", reference_detail="")
    idx = [(i, j) for i, j in pairs(Pn) if abs(np.log(M[i, j]) - np.log(M[j, i])) > np.log(1.01)]
    agree = [abs(np.log(P[i, j]) - np.log(M[i, j])) < abs(np.log(P[i, j]) - np.log(M[j, i])) for i, j in idx]
    procs = run.get("processes") or list(range(Pn))
    cp = [(procs.index(c), procs.index(e)) for c, e in CAUSE.get(ds, []) if c in procs and e in procs]
    mech = [bool(P[c, e] < P[e, c]) for c, e in cp]
    extra.update(asym_max=asym(P), n_distinguishable_pairs=len(idx),
                 orientation_agreement=float(np.mean(agree)) if agree else float("nan"),
                 pyspi_pairs=json.dumps({f"{i},{j}": float(f"{P[i, j]:.6g}") for i, j in pairs(Pn)}),
                 own_pairs=json.dumps({f"{i},{j}": float(f"{M[i, j]:.6g}") for i, j in pairs(Pn)}),
                 mechanism=json.dumps({f"{c}->{e}": [float(f"{P[c, e]:.6g}"), float(f"{P[e, c]:.6g}")] for c, e in cp}),
                 mechanism_ok=int(all(mech)) if mech else -1)
    if kn == "DotProduct" and run["path"] != "kernel":
        flag = int(asym(P) <= 1e-12)
        rule = "pass iff max|A - A^T| <= 1e-12 (record expectation: no direction information under z-scoring)"
    elif cp:
        flag = int(all(mech) and (all(agree) if agree else True))
        rule = "pass iff A[cause,effect] < A[effect,cause] for every known pair and A[i,j] is closer (log) to own f(x_i->x_j) than f(x_j->x_i) on every pair where those differ by > 1%"
    else:
        flag = int(len(idx) > 0 and all(agree))
        rule = "pass iff A[i,j] is closer (log) to own f(x_i->x_j) than f(x_j->x_i) on every pair where those differ by > 1%"
    det = f"{SELF}:blr_ml2" if kn == "DotProduct" else f"{KER}:gp_fit + gp_posterior_mean"
    return dict(value=M, criterion="qualitative", pass_flag=flag, reference_kind="independent-impl",
                reference_detail=f"{det} oriented fits on {run['path']} data; {rule} ({VERS})",
                notes="value = own M[i,j] = f(x_i -> x_j)", **extra)


def h_affine(run, X, P):
    kn = kname(run)
    ref_rid = (f"misc.gpfit.DotProduct|bivariate_gaussian|256|pipeline|closedform" if kn == "DotProduct"
               else "misc.gpfit.RBF|bivariate_gaussian|256|pipeline|posterior")
    V = load_npz(ref_rid)
    if V is None:
        V = np.full((X.shape[0],) * 2, np.nan)
    xb = prep(make_data.load("bivariate_gaussian", run["T"])["x"], "pipeline")
    return dict(value=V, criterion="tolerance", rtol=1e-8, atol=1e-12, reference_kind="same-lineage",
                reference_detail=f"pyspi's own {ref_rid} output (z-scoring removes scale/offset); pyspi vs pyspi",
                notes="scaled_offset = scale * bivariate_gaussian + offset (same draw)",
                zscored_rows_max_abs_diff=float(np.max(np.abs(xb - X))))


def h_gaussexp(run, X, P):
    T = run["T"]
    M, R = r2_closed(X)
    rtol = 0.1 if T <= 256 else 0.05
    extra = {}
    if P is not None:
        rel = np.abs(offdiag(P) - offdiag(M)) / offdiag(M)
        extra.update(max_rel=float(rel.max()), median_rel=float(np.median(rel)), n_lt_0p01=int(np.sum(offdiag(P) < 0.01)),
                     asym_max=asym(P), signed_rel_min=float(np.min((offdiag(P) - offdiag(M)) / offdiag(M))),
                     signed_rel_max=float(np.max((offdiag(P) - offdiag(M)) / offdiag(M))))
    return dict(value=M, criterion="tolerance", rtol=rtol, atol=0.0, reference_kind="analytic",
                reference_detail=f"{SELF}:r2_closed Gaussian expectation ((T-1)/T)(1 - r^2) (best predictor of jointly Gaussian data is linear) ({VERS})",
                notes="a-priori rtol from the record: 0.1 at T=256, 0.05 at T=1024; values < 0.01 flag interpolating fits", **extra)


def h_vsridge(run, X, P):
    T, ds = run["T"], run["dataset"]
    R = np.corrcoef(X)
    RG = (T - 1) / T * (1.0 - R ** 2 * (T * T - 1.0) / (T * T))
    np.fill_diagonal(RG, np.nan)
    rid_r = f"misc.lmfit.Ridge|{ds}|{T}|pipeline|closedform"
    Rp = load_npz(rid_r)
    extra = dict(ridge_run_id=rid_r)
    if Rp is not None:
        pr = run.get("processes") or list(range(Rp.shape[0]))
        Rp = Rp[np.ix_(pr, pr)]
        extra["ridge_pyspi_vs_closed_form_max_abs"] = float(np.nanmax(np.abs(Rp - RG)))
    flag = 0
    if P is not None:
        ex = offdiag(P) - offdiag(RG)
        flag = int(np.all(ex <= 1e-12))
        extra.update(n_pairs=int(ex.size), n_rbf_le_ridge=int(np.sum(ex <= 1e-12)), excess_max=float(ex.max()),
                     excess_min=float(ex.min()), rel_excess_max=float(np.max(ex / offdiag(RG))))
    return dict(value=RG, criterion="qualitative", pass_flag=flag, reference_kind="analytic",
                reference_detail=f"ridge (alpha=1) closed form ((T-1)/T)(1 - r^2 (T^2-1)/T^2) (refs/lmfit.py:ridge_closed_form, "
                                 f"validated in out/results/lmfit.json {rid_r}); pass iff A_RBF <= A_Ridge + 1e-12 on every pair ({VERS})",
                notes="value = Ridge MSE matrix on the same z-scored pair", **extra)


def h_determinism(run, x, Pp):
    flag, extra = 0, {}
    if Pp is not None:
        d = [float(np.nanmax(np.abs(Pp[k] - Pp[0]))) for k in range(1, Pp.shape[0])]
        flag = int(max(d) == 0.0)
        extra["max_abs_vs_first"] = d
        other = load_npz(sibling(run, "params"))
        if other is not None:
            extra["vs_params_probe_max_abs"] = float(np.nanmax(np.abs(other[..., 0] - Pp[0])))
            flag = int(flag and extra["vs_params_probe_max_abs"] == 0.0)
    return dict(value=np.array(0.0), criterion="qualitative", pass_flag=flag, reference_kind="same-lineage",
                reference_detail="pyspi vs pyspi self-consistency (probes/gpfit_probes.py:determinism): two GPModel.multivariate calls, "
                                 "random_state 0 and 12345 refits, and a call after reseeding numpy/random; pass iff bit-identical "
                                 "(and identical to the params probe's MSE column, a separate run of the same pyspi_outputs process, if that run exists); "
                                 "cross-process reproducibility is the crossproc tag",
                notes="random_state is unused when n_restarts_optimizer = 0 (_gpr.py:251, 322-333)", **extra)


def h_warnings(run, x, Pp):
    o = offdiag
    n_fit, n_bound, n_esc = int(np.sum(o(Pp[..., 0]))), int(np.sum(o(Pp[..., 1]))), int(np.sum(o(Pp[..., 2])))
    return dict(value=np.array(0.0), criterion="qualitative", pass_flag=int(n_bound > 0 and n_esc == 0),
                reference_kind="same-lineage",
                reference_detail="probes/gpfit_probes.py:warnings_count (sklearn's own warnings, counted with simplefilter('always'))",
                notes="pass = candidate confirmed: sklearn emits bound warnings and none escape GPModel.bivariate (misc.py:143-145)",
                n_pairs=int(o(Pp[..., 0]).size), n_warnings_sklearn=n_fit, n_bound_warnings=n_bound,
                n_pairs_with_bound_warning=int(np.sum(o(Pp[..., 1]) > 0)), n_warnings_escaping_pyspi=n_esc)


def h_scipy118(run, X, P):
    from sklearn.gaussian_process import GaussianProcessRegressor, kernels
    import sklearn
    kn = kname(run)
    tmpl = kernels.ConstantKernel() + kernels.WhiteKernel()
    tmpl += getattr(kernels, kn)()
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    for i, j in pairs(Pn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel=tmpl).fit(X[i][:, None], X[j])
        M[i, j] = float(np.mean((gp.predict(X[i][:, None]) - X[j]) ** 2))
    extra = {}
    if P is not None:
        rel = np.abs(offdiag(P) - offdiag(M)) / np.abs(offdiag(M))
        extra.update(n_pairs=int(rel.size), n_bit_identical=int(np.sum(offdiag(P) == offdiag(M))),
                     max_rel=float(rel.max()), n_rel_gt_1em3=int(np.sum(rel > 1e-3)))
    return dict(value=M, criterion="tolerance", rtol=1e-6, atol=1e-12, reference_kind="same-lineage",
                reference_detail=f"scikit-learn=={sklearn.__version__} GaussianProcessRegressor under scipy=={scipy.__version__} "
                                 f"(comparison venv) vs pyspi under the project venv's SciPy 1.17.1",
                notes="environment sensitivity of the single-start optimum, not a correctness reference", **extra)


def _default_sibling(run):
    """pyspi's default (plain SPI) output for the same dataset/T/path, from a sibling run."""
    for tag in ("posterior", "closedform", "scratch", "scipy118"):
        rid = sibling(run, tag)
        V = load_npz(rid)
        if V is not None and V.ndim == 2:
            return rid, V
    return None, None


def h_crossproc(run, x, Pp):
    rid_s, S = _default_sibling(run)
    Pn = x.shape[0]
    if Pp is None or S is None:
        return dict(value=np.full((5, Pn, Pn), np.nan), criterion="qualitative", pass_flag=0,
                    reference_kind="same-lineage", reference_detail="probe output or sibling run missing")
    modes = ["pipeline", "pipeline_repeat", "scipy_first", "pyspi_first", "env1"]
    od = ~np.eye(Pn, dtype=bool)
    so = S[od]
    extra = dict(stored_run_id=rid_s)
    rels, nbit = {}, {}
    for k, m in enumerate(modes):
        a = Pp[k][od]
        rels[m] = float(np.max(np.abs(a - so) / np.abs(so)))
        nbit[m] = int(np.sum(a == so))
        extra[f"{m}_threads_numpy_scipy"] = [float(Pp[k, 0, 0]), float(Pp[k, 1, 1])]
    extra.update(n_pairs=int(od.sum()), max_rel_vs_stored=rels, n_bit_identical_vs_stored=nbit,
                 pipeline_vs_repeat_n_bit_identical=int(np.sum(Pp[0][od] == Pp[1][od])))
    flag = int(nbit["pipeline"] == od.sum() and nbit["pipeline_repeat"] == od.sum() and max(rels.values()) <= 1e-6)
    V = np.broadcast_to(S, Pp.shape).copy()
    for k in range(Pp.shape[0]):
        V[k][~od] = Pp[k][~od]          # bookkeeping diagonal copied (thread counts, not values)
    return dict(value=V, criterion="qualitative", pass_flag=flag, reference_kind="same-lineage",
                reference_detail=f"pyspi vs pyspi: probes/gpfit_probes.py:crossproc (five fresh project-venv processes) vs the "
                                 f"stored {rid_s} computed inside audit/checks/pyspi_outputs.py",
                notes="pass iff both 'pipeline'-order processes are bit-identical to the stored output and every import mode is "
                      "within rtol 1e-6 (same basin); diagonal = [numpy OpenBLAS threads, SciPy OpenBLAS threads, pyspi before SciPy]",
                **extra)


def h_fixedhp(run, X, P):
    kn, T = kname(run), run["T"]
    Pn = X.shape[0]
    M = np.full((Pn, Pn), np.nan)
    for i, j in pairs(Pn):
        M[i, j] = mse_at(X[i], X[j], kn, np.zeros(3))
    extra = {}
    rid_d, D = _default_sibling(run)
    if D is not None:
        pr = run.get("processes")
        d = np.abs(offdiag(M) - offdiag(D))
        extra.update(default_run_id=rid_d, fixed_minus_default_max_abs=float(d.max()),
                     fixed_minus_default_max_rel=float(np.max(d / np.abs(offdiag(D)))),
                     fixed_range=[float(offdiag(M).min()), float(offdiag(M).max())],
                     default_range=[float(offdiag(D).min()), float(offdiag(D).max())],
                     entries_01_10_fixed=[float(M[0, 1]), float(M[1, 0])],
                     entries_01_10_default=[float(D[0, 1]), float(D[1, 0])])
    if kn == "DotProduct":
        R = np.corrcoef(X)
        RG = (T - 1) / T * (1.0 - R ** 2 * (T * T - 1.0) / (T * T))
        extra["fixed_vs_ridge_alpha1_closed_form_max_abs"] = float(np.nanmax(np.abs(offdiag(M) - offdiag(RG))))
    return dict(value=M, criterion="exact", rtol=1e-8, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{KER}:gp_posterior_mean at theta = 0 (documented values c = w = 1, "
                                 f"{'l' if kn == 'RBF' else 'sigma_0'} = 1 held fixed) vs sklearn optimizer=None ({VERS})",
                notes="answers open question 1 (what the documented, fixed hyperparameters would give)", **extra)


# --- C*RBF + W (amplitude) model, own numpy, theta = log[C, l, w] (sklearn order of ConstantKernel()*RBF() + WhiteKernel())
def amp_lml(x, y, theta):
    C, l, w = np.exp(theta)
    x, y = np.ravel(x), np.ravel(y)
    d2 = (x[:, None] - x[None, :]) ** 2
    R = np.exp(-0.5 * d2 / l ** 2)
    Kx = C * R + (w + ALPHA) * np.eye(x.size)
    try:
        L = np.linalg.cholesky(Kx)
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros(3)
    a = sla.cho_solve((L, True), y)
    v = -0.5 * y @ a - np.sum(np.log(np.diag(L))) - 0.5 * x.size * np.log(2 * np.pi)
    W = np.outer(a, a) - sla.cho_solve((L, True), np.eye(x.size))
    g = 0.5 * np.array([np.sum(W * (C * R)), np.sum(W * (C * R * d2 / l ** 2)), w * np.trace(W)])
    return float(v), g


def amp_fit(x, y, theta0, fixed=()):
    """L-BFGS-B on -amp_lml from theta0 in the [1e-5, 1e5] box; indices in `fixed` stay at theta0."""
    theta0 = np.asarray(theta0, float)
    free = [k for k in range(3) if k not in fixed]

    def f(t):
        th = theta0.copy(); th[free] = t
        v, g = amp_lml(x, y, th)
        return (np.inf, np.zeros(len(free))) if not np.isfinite(v) else (-v, -g[free])
    r = optimize.minimize(f, theta0[free], jac=True, method="L-BFGS-B", bounds=[(LO, HI)] * len(free))
    th = theta0.copy(); th[free] = r.x
    return th, -float(r.fun)


def amp_mse(x, y, theta):
    C, l, w = np.exp(theta)
    x, y = np.ravel(x), np.ravel(y)
    R = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / l ** 2)
    a = np.linalg.solve(C * R + (w + ALPHA) * np.eye(x.size), y)
    return float(np.mean((y - C * R @ a) ** 2))


def h_amplitude(run, X, P):
    T = run["T"]
    Pn = X.shape[0]
    tie = np.full((Pn, Pn), np.nan)
    rows = []
    for k, (i, j) in enumerate(pairs(Pn)):
        x, y = X[i], X[j]
        # amplitude model at l = 1e-5 (RBF Gram ~ I): the two ends of the C + w ridge.
        # interpolating end: w = 1e-5, C free; white-noise end: C = 1e-5, w free
        th_i, L_int = amp_fit(x, y, [0.0, LO, LO], fixed=(1, 2))
        th_n, L_ns = amp_fit(x, y, [LO, LO, 0.0], fixed=(0, 1))
        th_f, L_flat = amp_fit(x, y, [LO, 0.0, 0.0], fixed=(0,))   # no signal with l free (flat constant)
        tie[i, j] = L_int - L_ns
        rng = np.random.default_rng(MS_SEED + k)
        best = (None, -np.inf)
        for s0 in [np.zeros(3)] + list(rng.uniform(LO, HI, size=(20, 3))):
            th, v = amp_fit(x, y, s0)
            if v > best[1]:
                best = (th, v)
        # pyspi's C + W + RBF (theta = log[c, w, l]): interpolating (w = l = 1e-5) vs flat (l = 1e5)
        _, Lp_int, _ = local_fit_fixed(x, y, [0.0, LO, LO], fixed=(1, 2))
        _, Lp_flat, _ = local_fit_fixed(x, y, [0.0, 0.0, HI], fixed=(2,))
        rows.append(dict(i=i, j=j, tie=L_int - L_ns, interp_minus_flat_nosignal=L_int - L_flat,
                         mse_int=amp_mse(x, y, th_i), mse_ns=amp_mse(x, y, th_n), mse_flat=amp_mse(x, y, th_f),
                         own_best_lml=best[1], own_best_mse=amp_mse(x, y, best[0]), own_best_l=float(np.exp(best[0][1])),
                         own_best_gain_over_tie=best[1] - max(L_int, L_ns), pyspi_gap=Lp_int - Lp_flat,
                         sk_mse=float(P[i, j, 0]) if P is not None else np.nan,
                         sk_lml=float(P[i, j, 4]) if P is not None else np.nan))
    ties = np.array([r["tie"] for r in rows])
    flag = int(np.all(np.abs(ties) <= 1e-3))
    sk = np.array([r["sk_mse"] for r in rows])
    extra = dict(n_pairs=len(rows), max_abs_tie=float(np.max(np.abs(ties))),
                 mse_interp_config_range=[float(min(r["mse_int"] for r in rows)), float(max(r["mse_int"] for r in rows))],
                 mse_nosignal_config_range=[float(min(r["mse_ns"] for r in rows)), float(max(r["mse_ns"] for r in rows))],
                 amp_interp_minus_flat_nosignal_range=[float(min(r["interp_minus_flat_nosignal"] for r in rows)),
                                                       float(max(r["interp_minus_flat_nosignal"] for r in rows))],
                 pyspi_model_interp_minus_flat_range=[float(min(r["pyspi_gap"] for r in rows)), float(max(r["pyspi_gap"] for r in rows))],
                 half_log_T=float(0.5 * np.log(T)),
                 own_best_n_interpolating=int(sum(r["own_best_mse"] < 1e-6 for r in rows)),
                 own_best_n_gain_gt_1em3_over_tie=int(sum(r["own_best_gain_over_tie"] > 1e-3 for r in rows)),
                 own_best_mse_range=[float(min(r["own_best_mse"] for r in rows)), float(max(r["own_best_mse"] for r in rows))],
                 sklearn_n_interpolating=int(np.sum(sk < 1e-6)),
                 sklearn_mse_range=[float(np.nanmin(sk)), float(np.nanmax(sk))] if np.isfinite(sk).any() else [],
                 sklearn_lml_minus_own_best_range=[float(min(r["sk_lml"] - r["own_best_lml"] for r in rows)),
                                                   float(max(r["sk_lml"] - r["own_best_lml"] for r in rows))],
                 rows=json.dumps([{k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in r.items()} for r in rows]))
    return dict(value=tie, criterion="qualitative", pass_flag=flag, reference_kind="independent-impl",
                reference_detail=f"{SELF}:amp_lml/amp_fit (own LML and gradient for C*RBF + W, R&W eqs. 2.31, 5.8-5.9; no sklearn) "
                                 f"and local_fit_fixed on {KER}:gp_log_marginal_likelihood for pyspi's C + W + RBF ({VERS})",
                notes="value = LML(interpolating: l = w = 1e-5, C free) - LML(white noise: l = C = 1e-5, w free) under C*RBF + W; pass iff |value| <= 1e-3 "
                      "on every pair (the amplitude turns the interpolating mode into a tie with white noise, it does not remove it)",
                **extra)


def local_fit_fixed(x, y, theta0, fixed=()):
    """local_fit (pyspi's kernel) with the indices in `fixed` held at theta0."""
    theta0 = np.asarray(theta0, float)
    free = [k for k in range(3) if k not in fixed]

    def f(t):
        th = theta0.copy(); th[free] = t
        v, g = lml(x, y, "RBF", th)
        return (np.inf, np.zeros(len(free))) if not np.isfinite(v) else (-v, -np.asarray(g)[free])
    r = optimize.minimize(f, theta0[free], jac=True, method="L-BFGS-B", bounds=[(LO, HI)] * len(free))
    th = theta0.copy(); th[free] = r.x
    return th, -float(r.fun), r


HANDLERS = {"scratch": h_scratch, "closedform": h_closedform, "params": h_params, "posterior": h_posterior,
            "ridgecf": h_ridgecf, "postmean": h_postmean, "single": h_single, "identity": h_identity, "multistart": h_multistart, "direction": h_direction,
            "affine": h_affine, "gaussexp": h_gaussexp, "vsridge": h_vsridge, "determinism": h_determinism,
            "warnings": h_warnings, "scipy118": h_scipy118, "crossproc": h_crossproc, "fixedhp": h_fixedhp,
            "amplitude": h_amplitude}
RAW_TAGS = {"params", "determinism", "warnings", "crossproc"}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    runs = json.loads(MANIFEST.read_text())["runs"]
    # params runs first: posterior/identity/multistart/ridgecf read their probe outputs
    n = 0
    for run in runs:
        rid = run["run_id"]
        if only and only not in rid:
            continue
        tag = rid.split("|")[-1]
        x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
        if run.get("processes") is not None:
            x = x[list(run["processes"])]
        P = load_npz(rid)
        if tag in RAW_TAGS:
            if P is None:
                print(f"skip {rid}: no pyspi probe output yet")
                continue
            out = HANDLERS[tag](run, x, P)
        else:
            out = HANDLERS[tag](run, prep(x, run["path"]), P)
        write(rid, **out)
        n += 1
        print(f"ok {rid}", flush=True)
    print(f"refs/gpfit.py: wrote {n} reference files")


if __name__ == "__main__":
    main()
