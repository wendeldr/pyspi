#!/usr/bin/env python3
"""References for the sgc family: spectral.sgc.nonparametric and spectral.sgc.parametric_order-{None,1,20}.

Runs in the COMPARISON venv (no pyspi, no spectral_connectivity, no nitime).  For every run of
``audit/checks/manifest/sgc.json`` it writes ``out/ref/<run_id>.npz`` in the compare.py contract.
The run tag (fifth ``|`` field of the run_id; ``tag:band`` where band is mean|max-full|low|high)
selects the check.

Estimators re-implemented here (numpy/scipy only):

NONPARAMETRIC (Dhamala, Rangarajan & Ding 2008; Geweke 1982): ``_kernels_spectral.multitaper_fft``
(constant detrend, DPSS NW=3, 5 tapers with eigenvalue > 0.9, n_fft = T) -> taper-mean two-sided CSM
-> per pair ``wilson_factorise`` (tol 1e-8, 60 iterations, library plus-operator) -> Sigma = psi0 psi0^T,
H = psi psi0^-1 -> ``granger_nonparametric`` with S_kk the multitaper auto-spectrum and values <= 0 set
to NaN; grid = first (n_fft+1)//2 bins k/T.  [f, i, j] = GC j -> i.

PARAMETRIC (Geweke 1982 from a VAR fitted by Yule-Walker): per pair (i < j) the non-demeaned
autocovariances r(k)[a, b] = mean_t z_a(t+k) z_b(t) over the N-k overlapping samples (nitime
utils.py:2170-2176), the block Yule-Walker system solved directly with numpy.linalg.solve (NOT the
LWR recursion nitime uses; same system, different algorithm), convention x_t + sum a_k x_{t-k} = e_t,
Sigma = r(0) + sum a_k r(k)^T; order None: replica of nitime fit_model (granger.py:36-69): lags
1..max_order-1 (order = lag-1), BIC = 2 ln det Sigma + 2 p^2 m ln(Ntot)/Ntot with p = 2, Ntot = 2T,
break at the first increase, ValueError when the loop never breaks.  Transfer function
A(w) = I + sum a_k e^{-i w k}, H = A^-1 on w_k = pi k / 513 (scipy.signal.freqz worN=513 grid),
Geweke ratio form f_{j->i} = ln(S_ii / (S_ii - (Sigma_jj - Sigma_ij^2/Sigma_ii)|H_ij|^2)) with
S = H Sigma H^H (no clipping, ratio inside the log as nitime autoregressive.py:499/509), labels
linspace(0, 0.5, 513).

pyspi layer: fmin <= 0 -> 1e-5; keep fmin <= label <= fmax; nanmean/nanmax per cell; cells with
> 50% NaN bins -> NaN; A[i, j] = summary of GC j -> i (spectral.py:417-435).

Tags:
impl        full from-scratch estimator on the run's preprocessing path vs pyspi identifier; tolerance.
exact       (np) library Wilson+Geweke on an EXACT VAR spectrum (probe exact_lib) vs the analytic
            Geweke per-bin values from the true (A_k, Sigma) (var_transfer_function); tolerance 1e-6.
libcsm      (np) library CSM (probe lib_csm) vs ks.cross_spectral_matrix; tolerance 1e-9.
wilsonlib   (np) per-bin pyspi cache vs ks Wilson+Geweke applied to the LIBRARY CSM of the libcsm run.
bins        (np) per-bin pyspi cache vs from-scratch per-bin; extra keys: Wilson reconstruction error,
            NaN-bin fractions with zero padding (n_fft 2T, 4T) and more tapers (NW 10, 20).
dir         qualitative: code-derived convention row = target: A[target, source] > A[source, target] on every
            true link; the count under the documented rows = sources convention (glossary-of-terms.md:27) is reported.
truth       statistical vs the analytic population band statistic of the true model on pyspi's grid;
            se from a Monte Carlo of the from-scratch estimator (fresh draws of the generating model),
            familywise-adjusted: se = sd * z_(1-0.05/(2m))/2, m = judged cells.
mcexp       statistical vs the Monte Carlo mean of the from-scratch estimator (same adjustment).
null        statistical vs 0 (iid / contemporaneous-only data have zero lagged GC).
proj        (order-1) statistical vs the population VAR(1) projection of the VAR(2) truth
            (A* = Gamma(1) Gamma(0)^-1, Sigma* = Gamma(0) - A* Gamma(1)^T).
ols         statistical vs the band statistic of the same Geweke formula on a statsmodels OLS VAR fit
            (trend 'c') at the order nitime uses (order None: the replica's order per pair).
err         qualitative error path from probe outcome codes.
sweep       qualitative: failures over every dataset at T plus EEG-like constructed inputs are on
            degenerate inputs predicted by the replica, and reach the caller as TypeError (parametric)
            or AttributeError (nonparametric).
affine/scale  pyspi with normalise=False on x*[2,0.5]+[3,-5] (affine) or x*[2,0.5] (scale) vs the
            same from-scratch estimator on the UNTRANSFORMED raw x: Geweke GC is invariant under
            per-channel affine maps, so an invariant estimator must reproduce it; tolerance 1e-6.
conv        qualitative consistency: Monte Carlo bias of the mean-full statistic vs the population value
            at T = 256, 1024, 4096 must shrink at least 4-fold (var2_bivariate; iid_gaussian null).
dup         (np) qualitative: shared_envelope at T=1024, where the duplicated pair does not raise.
grid        nitime labels/evaluation points vs linspace(0,.5,513) / k/1026; exact.
yw          nitime (coef, ecov) per pair (probe nitime_model) vs the direct block Yule-Walker solve.
geweke      per-bin pyspi cache vs the Geweke formula applied to nitime's own (coef, ecov).
orders      nitime's chosen order per pair vs the replica of its BIC rule; exact.
maxorder    fit_model(x0, x1, max_order=m), m = 2..9: replica order or -1 (ValueError); exact.
deg         (order-20, T=64) per-bin cache vs from-scratch per-bin; extra keys count negative/NaN bins.
indef       (order-20, T=64) qualitative: a pair has negative or NaN bins iff its LWR noise
            covariance is indefinite (det <= 0 or a negative diagonal).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.signal  # noqa: E402
import scipy.stats  # noqa: E402

import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "sgc"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / FAMILY
MC_DIR = EXCH / "mc_cache"
SELF = "audit/checks/refs/sgc.py"
KS = "audit/checks/refs/_kernels_spectral.py"
R_MC = 200
N_GRID = 513
BANDS = {"full": (0.0, 0.5), "low": (0.0, 0.25), "high": (0.25, 0.5)}
FMIN_STR = {"0": 0.0, "0-25": 0.25, "1e-05": 1e-5, "0-5": 0.5}
SHORT = ["mean-full", "mean-low", "mean-high", "max-full", "max-low", "max-high"]
VERS = f"numpy=={np.__version__}, scipy=={scipy.__version__}"

warnings.simplefilter("ignore", RuntimeWarning)


# ============================================================================ helpers
def unit_kind(unit_id):
    if unit_id.endswith("nonparametric"):
        return ("np", None)
    o = unit_id.rsplit("order-", 1)[1]
    return ("par", None if o == "None" else int(o))


def band_of(identifier):
    m = re.search(r"_(mean|max)_fs-1_fmin-(.+?)_fmax-(.+?)(?:_order-|$)", identifier)
    stat, fmin, fmax = m.group(1), FMIN_STR[m.group(2)], FMIN_STR[m.group(3)]
    if fmin <= 0:
        fmin = 1e-5          # spectral.py:343-345
    return stat, fmin, fmax


def short_of(identifier):
    stat, fmin, fmax = band_of(identifier)
    return f"{stat}-{'low' if fmax == 0.25 else 'high' if fmin == 0.25 else 'full'}"


def short_band(short):
    stat, b = short.split("-")
    fmin, fmax = BANDS[b]
    return stat, (1e-5 if fmin <= 0 else fmin), fmax


def prep(x, path):
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return x
    if path == "pipeline-detrend":
        x = scipy.signal.detrend(x, axis=1, type="linear")
    elif path != "pipeline":
        raise ValueError(path)
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def reduce(G, labels, stat, fmin, fmax, nan_threshold=0.5):
    """pyspi multivariate reducer (spectral.py:417-435)."""
    m = (labels >= fmin) & (labels <= fmax)
    sel = G[m]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = (np.nanmean if stat == "mean" else np.nanmax)(sel, axis=0)
    out = np.array(out, dtype=np.float64, copy=True)
    pct = np.isnan(sel).mean(axis=0)
    np.fill_diagonal(pct, 0.0)
    out[pct > nan_threshold] = np.nan
    np.fill_diagonal(out, np.nan)
    return out


def all_bands(G, labels):
    return np.stack([reduce(G, labels, *short_band(s)) for s in SHORT])


def load_pyspi(rid):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def rid_with(run, tag, dataset=None, T=None, path=None):
    return "|".join([run["unit_id"], dataset or run["dataset"], str(T or run["T"]), path or run["path"], tag])


def write(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    payload = {k: (np.asarray(v) if not isinstance(v, str) else np.asarray(v)) for k, v in kw.items()}
    np.savez(OUT_REF / f"{rid}.npz", **payload)


def write_tol(rid, value, kind, detail, rtol=1e-6, atol=1e-8, notes="", **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), criterion="tolerance", rtol=rtol, atol=atol,
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


def write_exact(rid, value, kind, detail, rtol=1e-10, atol=1e-12, notes="", **extra):
    write(rid, value=np.asarray(value, dtype=np.float64), criterion="exact", rtol=rtol, atol=atol,
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


def write_qual(rid, ok, detail, kind="analytic", notes="", **extra):
    write(rid, value=np.asarray(np.nan), criterion="qualitative", pass_flag=int(bool(ok)),
          reference_kind=kind, reference_detail=detail, notes=notes, **extra)


def fw_se(sd, value):
    """Familywise (Bonferroni over judged cells, two-sided 5%) se so that 2 se = z * sd."""
    m = int(np.isfinite(value).sum())
    z = scipy.stats.norm.ppf(1 - 0.05 / (2 * max(m, 1)))
    return sd * z / 2.0, m, z


def write_stat(rid, value, sd, detail, S, kind="analytic", notes="", **extra):
    se, m, z = fw_se(np.asarray(sd, dtype=float), np.asarray(value, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        zz = np.abs(S - value) / sd
    write(rid, value=np.asarray(value, dtype=np.float64), se=np.asarray(se, dtype=np.float64),
          criterion="statistical", reference_kind=kind, reference_detail=detail, notes=notes,
          familywise_cells=m, familywise_z=float(z), mc_replicates=R_MC,
          pyspi_minus_value_maxabs=float(np.nanmax(np.abs(S - value))),
          max_abs_z_unadjusted=float(np.nanmax(np.where(np.isfinite(zz), zz, np.nan))) if np.isfinite(zz).any() else 0.0,
          **extra)


def cells(M, prefix):
    """Extra scalar keys for the off-diagonal cells of a small matrix."""
    out = {}
    P = M.shape[0]
    for i in range(P):
        for j in range(P):
            if i != j:
                out[f"{prefix}{i}{j}"] = float(M[i, j])
    return out


# ============================================================================ nonparametric
def np_perbin(z, NW=3.0, n_fft=None, info=False):
    """From-scratch nonparametric spectral GC (n_f, P, P), [f, i, j] = GC j -> i, on the library grid."""
    X, fr = ks.multitaper_fft(z, NW=NW, K=None, n_fft=n_fft)
    S = ks.cross_spectral_matrix(X)
    P = z.shape[0]
    nf = (S.shape[0] + 1) // 2
    out = np.full((nf, P, P), np.nan)
    infos = {}
    for a in range(P):
        for b in range(a + 1, P):
            idx = np.array([a, b])
            sub = S[:, idx[:, None], idx[None, :]]
            H, Sg, inf = ks.wilson_factorise(sub)
            g = ks.granger_nonparametric(ks.nonneg(sub), ks.nonneg(H), Sg, nan_nonpositive=True)
            out[:, a, b], out[:, b, a] = g[:, 0, 1], g[:, 1, 0]
            infos[(a, b)] = inf
    return (ks.nonneg(fr), out, infos) if info else (ks.nonneg(fr), out)


# ============================================================================ parametric
def autocov(z, nlags):
    """r[k][a, b] = mean_{t} z_a(t+k) z_b(t) over the N-k overlapping samples, k = 0..nlags-1 (no demeaning)."""
    N = z.shape[1]
    return np.stack([z[:, k:] @ z[:, :N - k].T / (N - k) for k in range(nlags)])


def block_yw(r, p):
    """Solve the block Yule-Walker system directly: r(k) = sum_i A_i r(k-i), k=1..p, r(-m) = r(m)^T.
    Returns nitime-convention a (p, n, n) = -A and Sigma = r(0) - sum A_i r(i)^T."""
    n = r.shape[1]
    if p == 0:
        return np.zeros((0, n, n)), r[0].copy()
    R = lambda m: r[m] if m >= 0 else r[-m].T  # noqa: E731
    G = np.block([[R(k - i) for k in range(1, p + 1)] for i in range(1, p + 1)])   # (np, np)
    Rhs = np.hstack([r[k] for k in range(1, p + 1)])                                 # (n, np)
    B = np.linalg.solve(G.T, Rhs.T).T                                                # B G = Rhs
    A = np.stack([B[:, i * n:(i + 1) * n] for i in range(p)])
    Sigma = r[0] - sum(A[i] @ r[i + 1].T for i in range(p))
    return -A, Sigma


def bic_nitime(Sigma, m, Ntot, p=2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * np.log(np.linalg.det(Sigma)) + (2 * p ** 2 * m * np.log(Ntot)) / Ntot


def fit_pair(z2, order, max_order=50, Ntot=None, return_trace=False):
    """Replica of nitime fit_model (granger.py:17-71) with the direct YW solve."""
    T = z2.shape[1]
    if order is not None:
        a, S = block_yw(autocov(z2, order + 1), order)
        return order, a, S
    Ntot = 2 * T if Ntot is None else Ntot
    c_old, best, trace = np.inf, None, []
    for lag in range(1, max_order):
        a, S = block_yw(autocov(z2, lag), lag - 1)
        c = bic_nitime(S, lag - 1, Ntot)
        trace.append(float(c))
        if c > c_old:
            break
        c_old, best = c, (lag - 1, a, S)
    else:
        raise ValueError(f"Model estimation order did not converge at max_order = {max_order}")
    return (best, trace) if return_trace else best


def freq_grid():
    """Evaluation points of scipy.signal.freqz(worN=513, whole=False): w_k = pi k/513 -> f = k/1026;
    labels nitime get_freqs(1, 1024) = linspace(0, 0.5, 513)."""
    k = np.arange(N_GRID)
    return k / (2.0 * N_GRID), np.linspace(0.0, 0.5, N_GRID)


def transfer(a, f):
    n = a.shape[1]
    Af = np.broadcast_to(np.eye(n, dtype=complex), (f.size, n, n)).copy()
    for k in range(a.shape[0]):
        Af += a[k][None] * np.exp(-2j * np.pi * f * (k + 1))[:, None, None]
    return np.linalg.inv(Af)


def geweke_ratio(H, Sg):
    """(n_f, 2, 2) [f, i, j] = f_{j -> i} = ln(S_ii / (S_ii - (Sg_jj - Sg_ij^2/Sg_ii)|H_ij|^2)), nitime's
    ratio form (no clipping), S = H Sg H^H."""
    S = H @ Sg @ np.conj(np.swapaxes(H, -1, -2))
    p = S[:, [0, 1], [0, 1]].real
    out = np.full((H.shape[0], 2, 2), np.nan)
    for i, j in ((0, 1), (1, 0)):
        gam = Sg[j, j] - Sg[i, j] ** 2 / Sg[i, i]
        intr = p[:, i] - gam * np.abs(H[:, i, j]) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, i, j] = np.log(p[:, i] / intr)
    return out


def par_perbin(z, order, max_order=50, f=None, fitter=None):
    """From-scratch parametric GC per bin (513, P, P) [f, i, j] = GC j -> i, plus fits per pair."""
    f = freq_grid()[0] if f is None else f
    P = z.shape[0]
    out = np.full((f.size, P, P), np.nan)
    fits = {}
    for a in range(P):
        for b in range(a + 1, P):
            o, coef, Sg = (fitter or fit_pair)(z[[a, b]], order, max_order)
            g = geweke_ratio(transfer(coef, f), Sg)
            out[:, a, b], out[:, b, a] = g[:, 0, 1], g[:, 1, 0]
            fits[(a, b)] = (o, coef, Sg)
    return out, fits


def ols_fitter(z2, order, max_order=50):
    """statsmodels OLS VAR (trend 'c') at the replica's order (order None) or the fixed order."""
    from statsmodels.tsa.api import VAR
    if order is None:
        order = fit_pair(z2, None, max_order)[0]
    if order == 0:
        return 0, np.zeros((0, 2, 2)), np.cov(z2)
    res = VAR(z2.T).fit(order, trend="c")
    return order, -np.asarray(res.coefs), np.asarray(res.sigma_u)


# ============================================================================ truth models
def true_model(ds_name, T):
    ds = make_data.load(ds_name, T)
    if ds_name in ("var2_bivariate", "var2_chain"):
        return [ds["A1"], ds["A2"]], ds["Sigma"]
    if ds_name == "gauss_ar1_pair":
        return [ds["A"]], ds["Sigma_E"]
    return None, None


def analytic_perbin(A_list, Sigma, f):
    """Exact Geweke GC per frequency from the true VAR (bivariate system), [f, i, j] = j -> i."""
    H, _, _ = ks.var_transfer_function(A_list, f, Sigma)
    g = geweke_ratio(H, np.asarray(Sigma, dtype=float))
    return g


def kolmogorov_F(A_list, Sigma, n=1 << 16):
    """Time-domain Geweke F_{j->i} = ln(sigma^2_restricted / Sigma_ii), restricted one-step prediction
    variance from the Kolmogorov-Szego formula exp(int_{-1/2}^{1/2} ln S_ii(f) df)."""
    f = np.fft.fftfreq(n)
    _, _, S = ks.var_transfer_function(A_list, f, Sigma)
    F = np.full((2, 2), np.nan)
    for i, j in ((0, 1), (1, 0)):
        F[i, j] = np.mean(np.log(S[:, i, i].real)) - np.log(Sigma[i, i])
    return F


def integral_f(A_list, Sigma, n=1 << 16):
    """int_{-1/2}^{1/2} f_{j->i}(f) df on a uniform grid."""
    f = np.fft.fftfreq(n)
    return np.nanmean(analytic_perbin(A_list, Sigma, f), axis=0)


def np_grid(T):
    return ks.nonneg(np.fft.fftfreq(T))


def simulate(ds_name, T, P, rng):
    if ds_name in ("var2_bivariate", "gauss_ar1_pair"):
        A_list, Sigma = true_model(ds_name, T)
        return make_data.simulate_var([np.asarray(a) for a in A_list], np.asarray(Sigma), T, rng)
    if ds_name == "iid_gaussian":
        return rng.standard_normal((P, T))
    if ds_name == "bivariate_gaussian":
        z = rng.standard_normal((4, T))
        x = np.empty_like(z)
        for k, rho in enumerate(make_data.D2_RHO):
            x[2 * k] = z[2 * k]
            x[2 * k + 1] = rho * z[2 * k] + np.sqrt(1 - rho ** 2) * z[2 * k + 1]
        return x
    raise ValueError(ds_name)


def estimator_bands(kind, order, z):
    """All six band statistics (6, P, P) of the from-scratch estimator on preprocessed z."""
    if kind == "np":
        f, G = np_perbin(z)
        return all_bands(G, f)
    G, _ = par_perbin(z, order)
    return all_bands(G, freq_grid()[1])


def mc(kind, order, ds_name, T, P, path, R=R_MC):
    """Monte Carlo of the from-scratch estimator on fresh draws of the generating model; cached."""
    key = f"{kind}_{order}_{ds_name}_{T}_{P}_{path}_{R}"
    MC_DIR.mkdir(parents=True, exist_ok=True)
    fn = MC_DIR / f"{key}.npz"
    if fn.is_file():
        with np.load(fn) as zz:
            return zz["draws"]
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(R):
        x = simulate(ds_name, T, P, rng)[:P]
        try:
            draws.append(estimator_bands(kind, order, prep(x, path)))
        except (ValueError, np.linalg.LinAlgError):   # replica failure on a draw: recorded as NaN
            draws.append(np.full((len(SHORT), P, P), np.nan))
    draws = np.stack(draws)
    np.savez(fn, draws=draws)
    return draws


# ============================================================================ checks
def chk_impl(run, kind, order, x, S):
    z = prep(x, run["path"])
    stat, fmin, fmax = band_of(run["identifier"])
    if kind == "np":
        f, G, infos = np_perbin(z, info=True)
        V = reduce(G, f, stat, fmin, fmax)
        rec = max(i["recon_err"] for i in infos.values())
        conv = all(i["converged"] for i in infos.values())
        write_tol(run["run_id"], V, "independent-impl",
                  f"{SELF}:np_perbin+reduce ({KS}: multitaper_fft NW=3 K=5, cross_spectral_matrix, "
                  f"wilson_factorise tol 1e-8 x60, granger_nonparametric NaN<=0); pyspi band rule "
                  f"[{fmin:g}, {fmax:g}] {stat}; {VERS}",
                  notes="from-scratch nonparametric pipeline (no spectral_connectivity)",
                  wilson_recon_err_max=rec, wilson_all_converged=int(conv),
                  nan_bin_frac_offdiag=float(np.isnan(G[(f >= fmin) & (f <= fmax)][:, ~np.eye(z.shape[0], dtype=bool)]).mean()),
                  n_bins=int(((f >= fmin) & (f <= fmax)).sum()))
        return
    G, fits = par_perbin(z, order)
    f_eval, labels = freq_grid()
    V = reduce(G, labels, stat, fmin, fmax)
    ords = sorted({int(v[0]) for v in fits.values()})
    neg = int(np.nansum(G < 0))
    write_tol(run["run_id"], V, "independent-impl",
              f"{SELF}:par_perbin+reduce (non-demeaned N-k autocovariances, direct block Yule-Walker "
              f"solve, {'nitime BIC replica' if order is None else f'order {order}'}, Geweke ratio on "
              f"w=pi k/513, labels k/1024); band [{fmin:g}, {fmax:g}] {stat}; {VERS}",
              notes="from-scratch parametric pipeline (no nitime)", orders_used=str(ords),
              negative_bins=neg, nan_bin_frac_offdiag=float(np.isnan(G).sum() - G.shape[0] * z.shape[0])
              / max(1, G.size - G.shape[0] * z.shape[0]))


def chk_exact(run, kind, order, x, S):
    A_list, Sigma = true_model(run["dataset"], run["T"])
    f = np_grid(run["T"])
    g = analytic_perbin(A_list, Sigma, f)
    V = g.copy()
    zero = np.abs(V) < 1e-12
    V[V <= 0] = np.nan             # library rule connectivity.py:1179 applied to the analytic values
    lib_nan_on_zero = int(np.isnan(S[zero]).sum()) if zero.any() else 0
    # bins whose analytic value is exactly 0 (no causal path) are NaN in the reference; record what the
    # library returns there (NaN by its <= 0 rule, or round-off-level positives)
    S0 = S[zero]
    write_tol(run["run_id"], V, "analytic",
              f"{SELF}:analytic_perbin (true A_k, Sigma of {run['dataset']} via {KS}:var_transfer_function; "
              f"Geweke 1982 rotated-noise formula) on k/T, k < T/2; values <= 0 -> NaN as the library",
              rtol=1e-6, atol=1e-6,
              notes="library pairwise_spectral_granger_prediction fed the exact spectrum through P pseudo-tapers",
              analytic_zero_bins=int(zero.sum()), library_nan_on_zero_bins=lib_nan_on_zero,
              library_max_on_zero_bins=float(np.nanmax(S0)) if np.isfinite(S0).any() else float("nan"),
              analytic_max=float(np.nanmax(g)))


def chk_libcsm(run, kind, order, x, S):
    z = prep(x, run["path"])
    X, _ = ks.multitaper_fft(z, NW=3.0)
    C = ks.cross_spectral_matrix(X)
    write_tol(run["run_id"], np.stack([C.real, C.imag]), "independent-impl",
              f"{KS}:multitaper_fft+cross_spectral_matrix (two-sided taper mean); {VERS}", rtol=1e-9, atol=1e-12)


def chk_wilsonlib(run, kind, order, x, S):
    L = load_pyspi(rid_with(run, "libcsm"))
    C = L[0] + 1j * L[1]
    P = C.shape[-1]
    nf = (C.shape[0] + 1) // 2
    out = np.full((nf, P, P), np.nan)
    rec = []
    for a in range(P):
        for b in range(a + 1, P):
            idx = np.array([a, b])
            sub = C[:, idx[:, None], idx[None, :]]
            H, Sg, inf = ks.wilson_factorise(sub)
            g = ks.granger_nonparametric(ks.nonneg(sub), ks.nonneg(H), Sg, nan_nonpositive=True)
            out[:, a, b], out[:, b, a] = g[:, 0, 1], g[:, 1, 0]
            rec.append(inf["recon_err"])
    stat_ref = {s: reduce(out, np_grid(C.shape[0]), *short_band(s)) for s in ("mean-full", "max-full")}
    stat_py = {s: reduce(S, np_grid(C.shape[0]), *short_band(s)) for s in ("mean-full", "max-full")}
    write_tol(run["run_id"], out, "independent-impl",
              f"{KS}:wilson_factorise+granger_nonparametric on the LIBRARY CSM of run {rid_with(run, 'libcsm')}",
              rtol=1e-6, atol=1e-8,
              notes="isolates Wilson factorisation + Geweke formula + NaN rule from the spectral estimate",
              wilson_recon_err_max=float(max(rec)),
              band_mean_full_maxabs=float(np.nanmax(np.abs(stat_ref["mean-full"] - stat_py["mean-full"]))),
              band_max_full_maxabs=float(np.nanmax(np.abs(stat_ref["max-full"] - stat_py["max-full"]))))


def chk_bins(run, kind, order, x, S):
    z = prep(x, run["path"])
    f, G, infos = np_perbin(z, info=True)
    off = ~np.eye(z.shape[0], dtype=bool)
    kept = (f >= 1e-5) & (f <= 0.5)

    def nanfrac(GG, ff):
        k = (ff >= 1e-5) & (ff <= 0.5)
        return float(np.isnan(GG[k][:, off]).mean())

    extra = {"nanfrac_nw3": nanfrac(G, f),
             "recon_err_nw3": max(i["recon_err"] for i in infos.values()),
             "converged_nw3": int(all(i["converged"] for i in infos.values()))}
    T = z.shape[1]
    for lab, kw in (("pad2", dict(n_fft=2 * T)), ("pad4", dict(n_fft=4 * T)), ("nw10", dict(NW=10.0)),
                    ("nw20", dict(NW=20.0))):
        ff, GG, inf2 = np_perbin(z, info=True, **kw)
        extra[f"nanfrac_{lab}"] = nanfrac(GG, ff)
        extra[f"recon_err_{lab}"] = max(i["recon_err"] for i in inf2.values())
        extra[f"converged_{lab}"] = int(all(i["converged"] for i in inf2.values()))
        extra[f"meanfull_offdiag_{lab}"] = float(np.nanmean(reduce(GG, ff, "mean", 1e-5, 0.5)))
    extra["meanfull_offdiag_nw3"] = float(np.nanmean(reduce(G, f, "mean", 1e-5, 0.5)))
    pct = np.isnan(G[kept]).mean(axis=0)
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            if i != j and pct[i, j] > 0:
                extra[f"nanpct_{i}{j}"] = float(pct[i, j])
    write_tol(run["run_id"], G, "independent-impl",
              f"{SELF}:np_perbin per-bin (n_f, P, P) vs pyspi cache F[0]; {VERS}", rtol=1e-6, atol=1e-8,
              notes="extra keys: NaN-bin fraction / Wilson recon error with zero padding (pad2/pad4) and "
                    "more tapers (nw10 K=19, nw20 K=39)", **extra)


TRUE_LINKS = {   # (source, target): documented convention requires A[source, target] > A[target, source]
    "var2_bivariate": [(0, 1)],
    "gauss_ar1_pair": [(1, 0)],          # A_true[0,1] = 0.3 (1 -> 0) is the stronger link; analytic GC below
    "delayed_noise": [(i, j) for i in range(4) for j in range(i + 1, 4)],   # smaller delay leads
}


def chk_dir(run, kind, order, x, S):
    """Empirical meaning of A[i, j] on ground-truth links.  Code-derived convention (trace stage): A[i, j] = GC j -> i
    (row = target).  pass_flag: A[target, source] > A[source, target] on EVERY true link.  The documented convention
    (rows = sources, GitBook glossary-of-terms.md:27) would require the opposite; its count is reported."""
    links = TRUE_LINKS[run["dataset"]]
    doc = [bool(S[s, t] > S[t, s]) for s, t in links]
    tra = [bool(S[t, s] > S[s, t]) for s, t in links]
    tie = [bool(S[t, s] == S[s, t]) for s, t in links]
    extra = cells(S, "A")
    if run["dataset"] == "gauss_ar1_pair":
        A_list, Sigma = true_model("gauss_ar1_pair", run["T"])
        Fk = kolmogorov_F(A_list, Sigma)
        extra.update(analytic_F_1to0=float(Fk[0, 1]), analytic_F_0to1=float(Fk[1, 0]))
    write_qual(run["run_id"], all(tra),
               f"ground truth links (source -> target) {links}; code-derived convention row = target requires "
               "A[target, source] > A[source, target] on every link (the documented rows = sources convention, "
               "glossary-of-terms.md:27, would require the opposite)",
               notes=f"row = target holds on {sum(tra)}/{len(links)} links; documented rows = sources holds on "
                     f"{sum(doc)}/{len(links)}; ties {sum(tie)}",
               links_documented=int(sum(doc)), links_transposed=int(sum(tra)), links_tied=int(sum(tie)),
               n_links=len(links), **extra)


def _mc_for(run, kind, order):
    P = run["_P"]
    return mc(kind, order, run["dataset"], run["T"], P, run["path"])


def chk_truth(run, kind, order, x, S):
    stat, fmin, fmax = band_of(run["identifier"])
    A_list, Sigma = true_model(run["dataset"], run["T"])
    if kind == "np":
        f = np_grid(run["T"])
        g = analytic_perbin(A_list, Sigma, f)
        V = reduce(g, f, stat, fmin, fmax)
        grid = "k/T (library grid, DC and Nyquist excluded)"
    else:
        f_eval, labels = freq_grid()
        g = analytic_perbin(A_list, Sigma, f_eval)
        V = reduce(g, labels, stat, fmin, fmax)
        V_lab = reduce(analytic_perbin(A_list, Sigma, labels), labels, stat, fmin, fmax)
        grid = "w = pi k/513 selected by labels k/1024"
    D = _mc_for(run, kind, order)
    si = SHORT.index(short_of(run["identifier"]))
    sd = np.nanstd(D[:, si], axis=0, ddof=1)
    mcm = np.nanmean(D[:, si], axis=0)
    Fk = kolmogorov_F(A_list, Sigma)
    If = integral_f(A_list, Sigma)
    extra = dict(mc_mean_minus_pop_maxabs=float(np.nanmax(np.abs(mcm - V))),
                 **cells(V, "pop"), **cells(mcm, "mcmean"), **cells(sd, "mcsd"), **cells(S, "pyspi"),
                 analytic_F_kolmogorov_10=float(Fk[1, 0]), analytic_F_kolmogorov_01=float(Fk[0, 1]),
                 integral_f_10=float(If[1, 0]), integral_f_01=float(If[0, 1]))
    if kind != "np":
        extra["pop_label_grid_maxabs_shift"] = float(np.nanmax(np.abs(V_lab - V)))
    write_stat(run["run_id"], V, sd,
               f"population Geweke GC of the true {run['dataset']} model ({KS}:var_transfer_function, "
               f"{SELF}:geweke_ratio) on {grid}, band [{fmin:g}, {fmax:g}] {stat}; se from {R_MC} MC draws of "
               f"{SELF}:estimator_bands", S, **extra)


def chk_mcexp(run, kind, order, x, S):
    stat, fmin, fmax = band_of(run["identifier"])
    D = _mc_for(run, kind, order)
    si = SHORT.index(short_of(run["identifier"]))
    V = np.nanmean(D[:, si], axis=0)
    sd = np.nanstd(D[:, si], axis=0, ddof=1)
    write_stat(run["run_id"], V, sd,
               f"Monte Carlo expectation of the from-scratch estimator ({SELF}:estimator_bands, {R_MC} fresh draws "
               f"of {run['dataset']} at T={run['T']}); checks pyspi behaves as that estimator", S,
               kind="independent-impl", **cells(V, "mcmean"), **cells(sd, "mcsd"))


def chk_null(run, kind, order, x, S):
    stat, fmin, fmax = band_of(run["identifier"])
    D = _mc_for(run, kind, order)
    V = np.zeros_like(S)
    np.fill_diagonal(V, np.nan)
    si = SHORT.index(short_of(run["identifier"]))
    sd = np.nanstd(D[:, si], axis=0, ddof=1)
    mcm = np.nanmean(D[:, si], axis=0)
    extra = dict(mc_mean_offdiag=float(np.nanmean(mcm)), pyspi_mean_offdiag=float(np.nanmean(S)),
                 mc_sd_offdiag=float(np.nanmean(sd)), pyspi_exact_zero_cells=int(np.sum(S == 0)))
    if kind == "np" and run["T"] == 1024:
        z = prep(x, run["path"])
        for nw in (10.0, 40.0):
            f, G = np_perbin(z, NW=nw)
            extra[f"from_scratch_nw{int(nw)}_meanfull_offdiag"] = float(np.nanmean(reduce(G, f, "mean", 1e-5, 0.5)))
    write_stat(run["run_id"], V, sd,
               f"analytic null: lagged GC of {run['dataset']} is 0 at every frequency; se from {R_MC} MC draws of "
               f"{SELF}:estimator_bands", S, **extra)


def chk_proj(run, kind, order, x, S):
    stat, fmin, fmax = band_of(run["identifier"])
    A_list, Sigma = true_model(run["dataset"], run["T"])
    G = make_data.var_autocov([np.asarray(a) for a in A_list], np.asarray(Sigma), 1)
    A1 = G[1] @ np.linalg.inv(G[0])
    Sg = G[0] - A1 @ G[1].T
    f_eval, labels = freq_grid()
    V = reduce(geweke_ratio(transfer(-A1[None], f_eval), Sg), labels, stat, fmin, fmax)
    V2 = reduce(analytic_perbin(A_list, Sigma, f_eval), labels, stat, fmin, fmax)
    D = _mc_for(run, kind, order)
    si = SHORT.index(run["run_id"].split(":")[-1])
    sd = np.nanstd(D[:, si], axis=0, ddof=1)
    write_stat(run["run_id"], V, sd,
               "population VAR(1) projection of the VAR(2) truth (A* = Gamma(1) Gamma(0)^-1, Sigma* = Gamma(0) - "
               f"A* Gamma(1)^T from make_data.var_autocov) through {SELF}:geweke_ratio; se from {R_MC} MC draws", S,
               **cells(V, "proj"), **cells(V2, "truevar2"))


def chk_ols(run, kind, order, x, S):
    import statsmodels
    stat, fmin, fmax = band_of(run["identifier"])
    z = prep(x, run["path"])
    f_eval, labels = freq_grid()
    Go, fo = par_perbin(z, order, fitter=ols_fitter)
    Gl, fl = par_perbin(z, order)
    V = reduce(Go, labels, stat, fmin, fmax)
    D = _mc_for(run, kind, order)
    sd = np.nanstd(D[:, SHORT.index(short_of(run["identifier"]))], axis=0, ddof=1)
    coef_diff = max(float(np.abs(fo[k][1] - fl[k][1]).max()) if fo[k][1].size else 0.0 for k in fo)
    # divisor-N (biased, positive-definite) Yule-Walker for comparison
    def ywN(z2, o, mo):
        if o is None:
            o = fit_pair(z2, None, mo)[0]
        N = z2.shape[1]
        r = np.stack([z2[:, k:] @ z2[:, :N - k].T / N for k in range(o + 1)])
        a, Sg = block_yw(r, o)
        return o, a, Sg
    Gn, _ = par_perbin(z, order, fitter=ywN)
    Vn = reduce(Gn, labels, stat, fmin, fmax)
    write_stat(run["run_id"], V, sd,
               f"Geweke GC ({SELF}:geweke_ratio) of a statsmodels=={statsmodels.__version__} VAR OLS fit "
               f"(trend 'c') at {'the replica order' if order is None else f'order {order}'}, same grid/band; "
               f"se = MC sd of the LWR estimator ({R_MC} draws)", S, kind="external-pkg",
               lwr_minus_ols_maxabs=float(np.nanmax(np.abs(S - V))),
               lwr_minus_ols_perbin_maxabs=float(np.nanmax(np.abs(Gl - Go))),
               lwr_minus_ols_coef_maxabs=coef_diff,
               ywN_minus_lwr_maxabs=float(np.nanmax(np.abs(Vn - S))),
               orders_used=str(sorted({int(v[0]) for v in fo.values()})),
               **cells(V, "ols"), **cells(S, "lwr"), **cells(sd, "mcsd"))


def chk_err(run, kind, order, x, S):
    code_direct, nan_catch, code_del, fi, fj, nreg, nan_calc = S
    if kind == "np":
        exp = dict(direct=1, delegate=1)
        ok = code_direct == 1 and code_del == 1 and nan_calc == 1.0
        rule = ("AttributeError (minimum_phase_decomposition.py:48 xp.linalg.linalg) from the library and from "
                "the direct multivariate() call; real Calculator table all NaN")
    else:
        exp = dict(direct=2, delegate=3)
        ok = code_direct == 2 and code_del == 3 and nan_calc == 1.0 and nan_catch == 1.0
        rule = ("delegate (nitime fit_model) raises ValueError/LinAlgError; direct multivariate() raises TypeError "
                "from warnings.warn(err) at spectral.py:439; real Calculator table all NaN (calculator.py:336-337); "
                "all-NaN return only under catch_warnings(record=True)+simplefilter('always')")
    side = EXCH / (re.sub(r"[^A-Za-z0-9_.-]+", "_", run["run_id"]) + ".json")
    msg = json.loads(side.read_text()) if side.is_file() else {}
    bic = {}
    if kind == "par" and order is None and fi >= 0:
        z2 = prep(x, run["path"])[[int(fi), int(fj)]]
        tr = np.array([bic_nitime(block_yw(autocov(z2, lag), lag - 1)[1], lag - 1, 2 * z2.shape[1])
                       for lag in range(1, 50)])
        bic = dict(replica_bic_first_nan_order=int(np.argmax(np.isnan(tr))) if np.isnan(tr).any() else -1,
                   replica_bic_n_nan=int(np.isnan(tr).sum()),
                   replica_bic_orders0to3=str(np.round(tr[:4], 4).tolist()))
    write_qual(run["run_id"], ok, rule,
               notes=f"direct={msg.get('direct')}; delegate={msg.get('delegate')}; "
                     f"catch_always={msg.get('catch_always_warnings', msg.get('catch_always_exception'))}",
               code_direct=int(code_direct), code_delegate=int(code_del), nanfrac_catch_always=float(nan_catch),
               nanfrac_calculator=float(nan_calc), fail_pair=f"({int(fi)},{int(fj)})", n_regex_filters=int(nreg),
               expected_direct=exp["direct"], expected_delegate=exp["delegate"], **bic)


def _constructed_cases(T):
    base = make_data.load("iid_gaussian", T)["x"][:3].copy()
    flat = base.copy(); flat[2] = 0.0
    dup = base.copy(); dup[2] = base[0]
    scaled = base.copy(); scaled[2] = -3.0 * base[0] + 7.0
    near = base.copy(); near[2] = base[0] + 1e-9 * np.random.default_rng(5).standard_normal(T)
    near6 = base.copy(); near6[2] = base[0] + 1e-6 * np.random.default_rng(6).standard_normal(T)
    return [("eeg:flat_channel", flat), ("eeg:duplicate_channel", dup), ("eeg:scaled_copy", scaled),
            ("eeg:near_duplicate_1e-9", near), ("eeg:near_duplicate_1e-6", near6)]


def _predict(kind, order, xx):
    """'ok' | 'data' (zero-variance channel: Data rejects) | 'fail' (replica raises) |
    'rounding' (a pair has |corr| > 1 - 1e-10: the outcome depends on round-off)."""
    sd = xx.std(axis=1)
    if np.any(sd == 0):
        return "data"
    z = prep(xx, "pipeline")
    C = np.corrcoef(z)
    near = np.abs(C[np.triu_indices(z.shape[0], 1)]).max() > 1 - 1e-10 if z.shape[0] > 1 else False
    try:
        if kind == "np":
            np_perbin(z)
        else:
            par_perbin(z, order)
    except (ValueError, np.linalg.LinAlgError):
        return "rounding" if near else "fail"
    return "rounding" if near else "ok"


def chk_sweep(run, kind, order, x, S):
    side = EXCH / (re.sub(r"[^A-Za-z0-9_.-]+", "_", run["run_id"]) + ".json")
    names = json.loads(side.read_text())["order"]
    man = json.loads((CHECKS / "data" / "MANIFEST.json").read_text())
    ds_names = sorted({f["name"] for f in man["files"] if int(f["T"]) == int(run["T"])})
    cases = [(nm, make_data.load(nm, run["T"])["x"]) for nm in ds_names] + _constructed_cases(run["T"])
    assert [c[0] for c in cases] == names, "case order differs from the probe"
    fail_code = 1 if kind == "np" else 2
    bad, obs_fail, pred_fail, rounding = [], [], [], []
    for (nm, xx), row in zip(cases, S):
        code = int(row[0])
        pr = _predict(kind, order, xx)
        if code != 0:
            obs_fail.append(nm)
        if pr == "fail":
            pred_fail.append(nm)
        if pr == "rounding":
            rounding.append(nm)
        if pr == "data":
            ok = code == 5
        elif pr == "fail":
            ok = code == fail_code
        elif pr == "rounding":
            ok = code in (0, fail_code)
        else:
            ok = code == 0
        if not ok:
            bad.append(f"{nm}: observed code {code}, predicted {pr}")
    write_qual(run["run_id"], not bad,
               f"failure set predicted by the from-scratch replica ({SELF}:_predict): replica raises -> pyspi raises "
               f"{'AttributeError' if kind == 'np' else 'TypeError'}; zero-variance channel -> Data() rejects; "
               "|corr| > 1-1e-10 -> round-off dependent (either outcome accepted)",
               notes="; ".join(bad) if bad else "all cases as predicted",
               observed_failures=", ".join(obs_fail), predicted_failures=", ".join(pred_fail),
               rounding_dependent=", ".join(rounding), n_cases=len(cases))


def chk_affine(run, kind, order, x, S):
    """Invariance of the estimator under a per-channel map y = D x (+ c): pyspi(y) with normalise=False must
    equal the same estimator applied to the untransformed raw x (kernel path)."""
    tag = run["run_id"].split("|")[-1]
    y = np.array(x, dtype=float, copy=True)
    y = y * np.array([2.0, 0.5])[:, None] + (np.array([3.0, -5.0])[:, None] if tag == "affine" else 0.0)
    stat, fmin, fmax = band_of(run["identifier"])
    if kind == "np":
        f, G = np_perbin(x)
        V = reduce(G, f, stat, fmin, fmax)
        f2, G2 = np_perbin(y)
        Vrep = reduce(G2, f2, stat, fmin, fmax)
        f3, G3 = np_perbin(x - x.mean(axis=1, keepdims=True))
        Vdm = reduce(G3, f3, stat, fmin, fmax)
    else:
        lab = freq_grid()[1]
        V = reduce(par_perbin(x, order)[0], lab, stat, fmin, fmax)
        Vrep = reduce(par_perbin(y, order)[0], lab, stat, fmin, fmax)
        Vdm = reduce(par_perbin(x - x.mean(axis=1, keepdims=True), order)[0], lab, stat, fmin, fmax)
    write_tol(run["run_id"], V, "analytic",
              "invariance of Geweke GC under per-channel affine maps (Geweke 1982): the estimator on the "
              f"transformed data must equal the same estimator on the untransformed raw data; {tag} = "
              f"x*[2,0.5]{'+[3,-5]' if tag == 'affine' else ''}; normalise=False (kernel path); "
              f"reference = {SELF}:{'np_perbin' if kind == 'np' else 'par_perbin'} on raw x",
              rtol=1e-6, atol=1e-8,
              notes="replica = from-scratch on the transformed data without demeaning (nitime's autocovariances); "
                    "demeaned = from-scratch on the demeaned untransformed data",
              replica_minus_pyspi_maxabs=float(np.nanmax(np.abs(Vrep - S))),
              raw_minus_demeaned_maxabs=float(np.nanmax(np.abs(V - Vdm))),
              data_mean_abs_max=float(np.abs(x.mean(axis=1)).max()),
              **cells(V, "invariant"), **cells(S, "pyspi"))


def chk_conv(run, kind, order, x, S):
    """Consistency: Monte Carlo bias |E[estimate] - population| of the mean-full statistic at T = 256, 1024,
    4096 must shrink at least 4-fold from T=256 to T=4096 (O(T^-1/2) or faster)."""
    ds = run["dataset"]
    P = run["_P"]
    Ts = (256, 1024, 4096)
    if ds == "iid_gaussian":
        pop = np.zeros((P, P))
        np.fill_diagonal(pop, np.nan)
        popproj = pop
    else:
        A_list, Sigma = true_model(ds, 4096)
        f_eval, lab = (np_grid(4096), np_grid(4096)) if kind == "np" else freq_grid()
        popT = {}
        for T in Ts:
            ff, ll = (np_grid(T), np_grid(T)) if kind == "np" else (f_eval, lab)
            popT[T] = reduce(analytic_perbin(A_list, Sigma, ff), ll, "mean", 1e-5, 0.5)
        if kind == "par" and order == 1:
            G = make_data.var_autocov([np.asarray(a) for a in A_list], np.asarray(Sigma), 1)
            A1 = G[1] @ np.linalg.inv(G[0])
            popproj = reduce(geweke_ratio(transfer(-A1[None], f_eval), G[0] - A1 @ G[1].T), lab, "mean", 1e-5, 0.5)
        else:
            popproj = None
    extra, bias = {}, {}
    for T in Ts:
        D = mc(kind, order, ds, T, P, run["path"])[:, 0]
        m, sd = np.nanmean(D, axis=0), np.nanstd(D, axis=0, ddof=1)
        p_ = pop if ds == "iid_gaussian" else popT[T]
        bias[T] = float(np.nanmax(np.abs(m - p_)))
        extra[f"bias_T{T}"] = bias[T]
        extra[f"sd_T{T}"] = float(np.nanmax(sd))
        if popproj is not None and ds != "iid_gaussian":
            extra[f"bias_vs_var1_projection_T{T}"] = float(np.nanmax(np.abs(m - popproj)))
    ok = bias[4096] <= bias[256] / 4.0
    write_qual(run["run_id"], ok,
               f"consistency of the mean-full statistic on {ds}: max-cell |MC mean - population| at T=4096 must be "
               f"<= 1/4 of that at T=256 ({R_MC} MC draws of {SELF}:estimator_bands per T)",
               notes=f"bias T=256 {bias[256]:.4g}, T=1024 {bias[1024]:.4g}, T=4096 {bias[4096]:.4g}",
               bias_ratio_256_over_4096=float(bias[256] / bias[4096]) if bias[4096] > 0 else float("inf"), **extra)


def chk_grid(run, kind, order, x, S):
    f_eval, labels = freq_grid()
    write_exact(run["run_id"], np.stack([labels, f_eval]), "analytic",
                "nitime utils.py:1207-1211 get_freqs(1, 1024) = linspace(0, 0.5, 513); "
                "scipy.signal.freqz(worN=513, whole=False) w = pi k/513 -> f = k/1026",
                label_minus_point_maxabs=float(np.max(labels - f_eval)))


def _nitime_rows(fits):
    rows = []
    for (i, j) in sorted(fits):
        o, coef, Sg = fits[(i, j)]
        r = np.full(7 + 4 * 48, np.nan)
        r[:3] = (i, j, o)
        r[3:7] = np.asarray(Sg).ravel()
        c = np.asarray(coef).ravel()
        r[7:7 + c.size] = c
        rows.append(r)
    return np.array(rows)


def chk_yw(run, kind, order, x, S):
    z = prep(x, run["path"])
    _, fits = par_perbin(z, order)
    V = _nitime_rows(fits)
    dets = [float(np.linalg.det(v[2])) for v in fits.values()]
    write_tol(run["run_id"], V, "independent-impl",
              f"{SELF}:fit_pair (direct numpy.linalg.solve of the block Yule-Walker system on non-demeaned "
              f"N-k autocovariances; order rule = nitime replica) vs nitime lwr_recursion; {VERS}",
              rtol=1e-7, atol=1e-10, notes="rows [i, j, order, ecov(4), coef(order*4)]",
              min_det_ecov=float(min(dets)), n_indefinite=int(sum(d <= 0 for d in dets)))


def chk_geweke(run, kind, order, x, S):
    M = load_pyspi(rid_with(run, "yw"))
    f_eval, _ = freq_grid()
    P = int(np.nanmax(M[:, 1])) + 1
    out = np.full((N_GRID, P, P), np.nan)
    for r in M:
        i, j, o = int(r[0]), int(r[1]), int(r[2])
        Sg = r[3:7].reshape(2, 2)
        coef = r[7:7 + 4 * o].reshape(o, 2, 2)
        g = geweke_ratio(transfer(coef, f_eval), Sg)
        out[:, i, j], out[:, j, i] = g[:, 0, 1], g[:, 1, 0]
    write_tol(run["run_id"], out, "independent-impl",
              f"{SELF}:transfer+geweke_ratio on nitime's own fitted (coef, ecov) of run {rid_with(run, 'yw')}, "
              "w = pi k/513; [f, a, b] = GC b -> a", rtol=1e-9, atol=1e-12,
              notes="isolates nitime transfer_function_xy/granger_causality_xy + pyspi's transpose from the VAR fit")


def chk_orders(run, kind, order, x, S):
    z = prep(x, run["path"])
    P = z.shape[0]
    V = np.full((P, P), np.nan)
    Vt = np.full((P, P), np.nan)
    Vsm = np.full((P, P), np.nan)
    first_min_vs_global = []
    from statsmodels.tsa.api import VAR
    for a in range(P):
        for b in range(a + 1, P):
            try:
                (o, _, _), trace = fit_pair(z[[a, b]], None, return_trace=True)
            except ValueError:
                o, trace = -1, []
            V[a, b] = V[b, a] = o
            try:
                Vt[a, b] = Vt[b, a] = fit_pair(z[[a, b]], None, Ntot=z.shape[1])[0]
            except ValueError:
                Vt[a, b] = Vt[b, a] = -1
            full = []
            for lag in range(1, 50):          # nitime's own criterion over every fitted order 0..48, no break
                aa, SS = block_yw(autocov(z[[a, b]], lag), lag - 1)
                full.append(bic_nitime(SS, lag - 1, 2 * z.shape[1]))
            full = np.array(full)
            first_min_vs_global.append((int(o), int(np.nanargmin(full)) if np.isfinite(full).any() else -1))
            pm = min(48, z.shape[1] // 10)
            Vsm[a, b] = Vsm[b, a] = VAR(z[[a, b]].T).select_order(maxlags=pm, trend="c").bic
    write_exact(run["run_id"], V, "independent-impl",
                f"{SELF}:fit_pair replica of nitime granger.py:36-69 (lags 1..49, order = lag-1, BIC = 2 ln det + "
                "8 m ln(2T)/(2T), break at first increase) with the direct YW solve",
                notes="extra: same rule with Ntot = T (nitime docstring utils.py:2357-2358), standard Schwarz "
                      "(statsmodels select_order, global minimum, OLS, common sample), and per pair (first-stop "
                      "order, global minimum of nitime's own criterion over orders 0..48)",
                **cells(Vt, "ntotT"), **cells(Vsm, "sm_bic"),
                first_stop_vs_global_min_nitime_bic=str(first_min_vs_global))


def chk_maxorder(run, kind, order, x, S):
    z = prep(x, run["path"])
    out = []
    for m in range(2, 10):
        try:
            out.append(fit_pair(z[[0, 1]], None, max_order=m)[0])
        except ValueError:
            out.append(-1)
    write_exact(run["run_id"], np.array(out, dtype=float), "independent-impl",
                f"{SELF}:fit_pair replica, max_order m = 2..9: lags 1..m-1 fitted (orders 0..m-2), returnable "
                "orders 0..m-3 (the last fitted order is never returned: no break -> ValueError -> -1)")


def chk_deg(run, kind, order, x, S):
    z = prep(x, run["path"])
    G, fits = par_perbin(z, order)
    off = ~np.eye(z.shape[0], dtype=bool)
    write_tol(run["run_id"], G, "independent-impl",
              f"{SELF}:par_perbin per-bin (513, P, P) vs pyspi cache F[0] at T={run['T']} order {order}",
              rtol=1e-6, atol=1e-8,
              pyspi_negative_bin_frac=float(np.nanmean(S[:, off] < 0)), pyspi_nan_bin_frac=float(np.isnan(S[:, off]).mean()),
              ref_negative_bin_frac=float(np.nanmean(G[:, off] < 0)), ref_nan_bin_frac=float(np.isnan(G[:, off]).mean()),
              n_pairs_indefinite=int(sum(np.linalg.det(v[2]) <= 0 or np.any(np.diag(v[2]) < 0) for v in fits.values())),
              n_pairs=len(fits))


def chk_indef(run, kind, order, x, S):
    C = load_pyspi(rid_with(run, "deg"))
    bad, rows = [], []
    for r in S:
        i, j = int(r[0]), int(r[1])
        Sg = r[3:7].reshape(2, 2)
        indef = bool(np.linalg.det(Sg) <= 0 or np.any(np.diag(Sg) < 0))
        v = np.concatenate([C[:, i, j], C[:, j, i]])
        anom = bool(np.any(v < 0) or np.any(np.isnan(v)))
        rows.append(f"({i},{j}) det={np.linalg.det(Sg):.3g} diag=({Sg[0, 0]:.3g},{Sg[1, 1]:.3g}) "
                    f"neg={int(np.sum(v < 0))} nan={int(np.isnan(v).sum())}")
        if indef != anom:
            bad.append(f"({i},{j})")
    write_qual(run["run_id"], not bad,
               "negative or NaN per-bin GC occurs in a pair iff nitime's LWR noise covariance is indefinite "
               "(Geweke ratio: intr < 0 needs Sigma_jj - Sigma_ij^2/Sigma_ii < 0, i.e. det < 0 given Sigma_ii > 0)",
               notes="; ".join(rows)[:3000], mismatched_pairs=", ".join(bad),
               n_indefinite=int(sum("det=-" in r for r in rows)), n_pairs=len(rows))


def chk_dup(run, kind, order, x, S):
    """shared_envelope: x2 is bit-identical to x0.  Records what pyspi returns for the duplicated pair when the
    library's Cholesky initialisation does NOT fail (round-off), and whether that factor is numerically singular."""
    z = prep(x, run["path"])
    X, _ = ks.multitaper_fft(z[[0, 2]], NW=3.0)
    C = ks.cross_spectral_matrix(X)
    G0 = np.fft.ifft(C, axis=0)[0].real
    ev = np.linalg.eigvalsh(G0)
    finite = bool(np.isfinite(S[0, 2]) and np.isfinite(S[2, 0]))
    write_qual(run["run_id"], finite and ev.min() <= 1e-12 * ev.max(),
               "pyspi returns finite GC for a channel and its exact copy although the lag-0 covariance the Wilson "
               "start factorises is numerically singular (min/max eigenvalue <= 1e-12); the copy pair has no "
               "defined GC (both directions should be 0 or undefined)",
               notes=f"A[0,2] = {S[0, 2]:.6g}, A[2,0] = {S[2, 0]:.6g}; lag-0 eigenvalues {ev.tolist()}",
               lag0_eig_min=float(ev.min()), lag0_eig_max=float(ev.max()),
               **cells(S, "A"))


NTOT_R = 200


def _ntot_draws(T, seed, R=NTOT_R, burn=300):
    """Byte-identical copy of probes/sgc_probes.py:_ntot_draws (bivariate VAR(2), weak second lag)."""
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.4, 0.0], [0.3, 0.3]])
    for _ in range(R):
        a2 = rng.uniform(0.02, 0.25)
        A2 = a2 * np.array([[1.0, 0.0], [-1.0, 1.0]])
        e = rng.standard_normal((T + burn, 2))
        x = np.zeros((T + burn, 2))
        for t in range(2, T + burn):
            x[t] = A1 @ x[t - 1] + A2 @ x[t - 2] + e[t]
        yield a2, x[burn:].T.copy()


def chk_ntot(run, kind, order, x, S):
    """Borderline-order draws: does Ntot = 2T (granger.py:38) instead of Ntot = T (nitime's own docstring,
    utils.py:2357-2358) change the order nitime selects, and with it pyspi's value?  value = (R, 6) in the
    probe's layout from the from-scratch side: [a2, replica order with Ntot = 2T, sum(z), replica A[1,0],
    replica A[0,1], sum(raw)], exact against nitime's chosen order and pyspi's matrix; the Ntot = T orders,
    standard Schwarz (statsmodels global minimum) and the value change at the T-rule order go to extra keys."""
    from statsmodels.tsa.api import VAR
    mean_full = short_band("mean-full")
    V2, VT, Vsm, a2s, chk_raw, chk_z, dA, rep_vs_pyspi, rows = [], [], [], [], [], [], [], [], []
    for r, (a2, xr) in enumerate(_ntot_draws(run["T"], run["seed"])):
        z = prep(xr, run["path"])
        a2s.append(a2)
        chk_raw.append(abs(xr.sum() - S[r, 5]))
        chk_z.append(abs(z.sum() - S[r, 2]))
        T = z.shape[1]
        try:
            o2 = fit_pair(z, None)[0]
        except ValueError:
            o2 = -1
        try:
            oT = fit_pair(z, None, Ntot=T)[0]
        except ValueError:
            oT = -1
        V2.append(o2)
        VT.append(oT)
        row = [a2, o2, float(z.sum()), np.nan, np.nan, float(xr.sum())]
        Vsm.append(VAR(z.T).select_order(maxlags=min(48, T // 10), trend="c").bic)
        if o2 >= 0:
            G2, _ = par_perbin(z, o2)
            A2m = reduce(G2, freq_grid()[1], *mean_full)
            row[3], row[4] = A2m[1, 0], A2m[0, 1]
            rep_vs_pyspi.append(max(abs(A2m[1, 0] - S[r, 3]), abs(A2m[0, 1] - S[r, 4])))
            if oT >= 0 and oT != o2:
                GT, _ = par_perbin(z, oT)
                ATm = reduce(GT, freq_grid()[1], *mean_full)
                dA.append((r, a2, o2, oT, float(A2m[1, 0]), float(ATm[1, 0]), float(A2m[0, 1]), float(ATm[0, 1])))
        rows.append(row)
    V2, VT, Vsm = np.array(V2, float), np.array(VT, float), np.array(Vsm, float)
    ok = V2 >= 0
    differ = ok & (VT >= 0) & (VT != V2)
    rel = [abs(t[5] - t[4]) / abs(t[4]) for t in dA if t[4] != 0]
    write_exact(run["run_id"], np.array(rows, dtype=float), "independent-impl",
                f"{SELF}:fit_pair replica of nitime granger.py:36-69 (Ntot = 2T) on {NTOT_R} VAR(2) draws "
                f"(_ntot_draws, A1 = [[0.4,0],[0.3,0.3]], A2 = a2[[1,0],[-1,1]], a2 ~ U(0.02,0.25), T = {run['T']}, "
                f"seed {run['seed']}); {VERS}",
                notes="extra: n_order_differs_ntotT = draws where the same rule with Ntot = T picks another order; "
                      "dA_examples = (draw, a2, order 2T, order T, A[1,0] at 2T, A[1,0] at T, A[0,1] at 2T, A[0,1] at T) "
                      "from the from-scratch estimator, mean over the full band",
                n_draws=int(V2.size), n_valueerror=int((~ok).sum()),
                n_order_differs_ntotT=int(differ.sum()),
                n_ntotT_lower=int((differ & (VT < V2)).sum()), n_ntotT_higher=int((differ & (VT > V2)).sum()),
                n_match_sm_bic_ntot2T=int((ok & (V2 == Vsm)).sum()), n_match_sm_bic_ntotT=int((ok & (VT == Vsm)).sum()),
                orders_2T_counts=str({int(k): int(v) for k, v in zip(*np.unique(V2, return_counts=True))}),
                orders_T_counts=str({int(k): int(v) for k, v in zip(*np.unique(VT, return_counts=True))}),
                orders_smbic_counts=str({int(k): int(v) for k, v in zip(*np.unique(Vsm, return_counts=True))}),
                max_abs_A10_change_at_ntotT=float(max((abs(t[5] - t[4]) for t in dA), default=0.0)),
                median_rel_A10_change_at_ntotT=float(np.median(rel)) if rel else float("nan"),
                replica_vs_pyspi_max_abs=float(max(rep_vs_pyspi, default=float("nan"))),
                draw_checksum_max_abs_raw=float(max(chk_raw)), draw_checksum_max_abs_z=float(max(chk_z)),
                dA_examples=str([tuple(round(v, 5) if isinstance(v, float) else v for v in t) for t in dA[:5]]))


CHECKS_BY_TAG = {
    "impl": chk_impl, "exact": chk_exact, "libcsm": chk_libcsm, "wilsonlib": chk_wilsonlib, "bins": chk_bins,
    "dir": chk_dir, "truth": chk_truth, "mcexp": chk_mcexp, "null": chk_null, "proj": chk_proj, "ols": chk_ols,
    "err": chk_err, "sweep": chk_sweep, "conv": chk_conv, "dup": chk_dup, "affine": chk_affine, "scale": chk_affine, "grid": chk_grid,
    "yw": chk_yw, "geweke": chk_geweke, "orders": chk_orders, "maxorder": chk_maxorder, "deg": chk_deg,
    "indef": chk_indef, "ntot": chk_ntot,
}


def main():
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    only = set(sys.argv[1:])
    n_ok = n_skip = n_err = 0
    for run in man["runs"]:
        rid = run["run_id"]
        if only and not any(o in rid for o in only):
            continue
        if not (OUT_PY / f"{rid}.npz").is_file():
            print(f"SKIP {rid}: no pyspi output", file=sys.stderr)
            n_skip += 1
            continue
        tag = rid.split("|")[4].split(":")[0]
        kind, order = unit_kind(run["unit_id"])
        x = make_data.load(run["dataset"], run["T"])["x"]
        if run.get("processes") is not None:
            x = x[list(run["processes"])]
        run = dict(run, _P=x.shape[0])
        try:
            CHECKS_BY_TAG[tag](run, kind, order, x, load_pyspi(rid))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
            n_err += 1
    print(f"sgc refs: {n_ok} written, {n_skip} skipped (no pyspi output), {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
