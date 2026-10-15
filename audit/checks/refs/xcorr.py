"""References for the xcorr family (COMPARISON venv; never imports pyspi and never uses scipy.signal.correlate:
the xcorr kernels xcorr_full and pyspi_xcorr_replica use np.dot / np.convolve only; the shared kernel module
_kernels_basic_misc.py does import scipy.signal.hilbert at module level, line 25, which these references do not call).

Units: basic.xcorr.{max,mean}_sig-{True,False}, each with the plain and the -sq identifier
(pyspi/statistics/basic.py:75-133, CrossCorrelation).  For every run in
audit/checks/manifest/xcorr.json this script writes out/ref/<run_id>.npz (contract in
audit/checks/compare.py).  The run tag (last '|' field, with an optional '-sq' suffix) selects the
reference:

  replica        _kernels_basic_misc.pyspi_xcorr_replica (np.convolve direct sum, not scipy.signal.correlate)
                 applied in pyspi's orientation (x = higher-index channel, y = lower-index channel,
                 pyspi/base.py:85-94 + basic.py:109-113 cache) on series preprocessed here
                 (own z-score ddof=1 / own least-squares linear detrend).  criterion exact.
                 Extra keys: the T/(T-1) fact at lag 0 (xcorr_full ddof=1 vs replica, pair (1,0)),
                 thresholds, window lags, centre lag, kept-window sizes.
  canon          canon_steps step d0: Brockwell & Davis sample CCF rho_hat(h) (xcorr_full with ddof=1,
                 which equals the B&D divisor-n estimator exactly; no T/(T-1) factor) over the
                 documented window |h| <= T//4.  sig-True: every lag in that window with
                 |rho_hat| > 1.96/sqrt(T), not necessarily contiguous (sig_filter; empty -> 0);
                 sig-False: the whole window, no filter.  Then max/mean of rho_hat or rho_hat^2.
                 The contiguous block around lag 0 is a separate step of the same run (c_doc,
                 reported as gap_c_doc; for sig-False c_doc = d0).
                 criterion tolerance (rtol 1e-8, atol 1e-10).  Extra keys d0..d4 decompose the gap
                 step by step (factor, window, threshold, pyspi cut rule).
  perm-replica   reversed channel order (manifest processes); replica on the reversed array (exact).
                 Extra keys: max|A_rev(back) - A_fwd| for pyspi and for the replica.
  perm-selfcons  SAME-LINEAGE: pyspi(reversed order) vs pyspi(forward order) mapped into the
                 reversed index space (criterion exact; order dependence shows as failure).
  selfcopy       shared_envelope processes [0, 2] (x2 bit-identical to x0): max units -> T/(T-1)
                 (squared: (T/(T-1))^2) analytically when lag 0 is in the kept window; mean units ->
                 T/(T-1) x mean of the B&D autocorrelation of x0 over pyspi's window/rule.
  lag-cached     peak lag of pyspi's cached r (probe) vs analytic d_hi - d_lo (delayed_noise).
  lag-fresh      peak lag of r from bivariate(i, j) evaluated first (probe) vs analytic d_i - d_j.
  cached-r       pyspi's cached r rows (probe) vs T/(T-1) x xcorr_full(x_hi, x_lo, ddof=1) on lags
                 -(T//4-1)..T//4 (tolerance atol 1e-12); extra keys: lag-0 ratio, shifted-window
                 mismatch.
  null           probe exceedance counts on iid_gaussian vs the analytic null expectation
                 (rho_hat(h) ~ N(0, (T-|h|)/T^2) for independent white series) (statistical).
  errsweep       probe status codes vs the codes predicted here from the replica (qualitative).
  pop*           max of pyspi's r vs the population peak cross-correlation over the window (from the
                 dataset's truth arrays), se from Bartlett's formula (statistical).
  zerosum        mean_sig-False vs -(sum of T/(T-1) rho_hat outside the window)/(2 T//4) (tolerance).
  endlag         constructed pair y = x0 delayed by T//4 (probe endlag_pair) in both channel orders:
                 [A_fwd, A_rev] vs the replica (exact); shows the end-lag order dependence of max_sig-False.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
from _kernels_basic_misc import pyspi_xcorr_replica, xcorr_full  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT = CHECKS / "out" / "ref"
PYOUT = CHECKS / "out" / "pyspi"
SELF = "audit/checks/refs/xcorr.py"
KERN = "audit/checks/refs/_kernels_basic_misc.py"


# ------------------------------------------------------------------ preprocessing (own code)
def zscore1(x):
    x = np.asarray(x, float)
    m = x.mean(axis=-1, keepdims=True)
    return (x - m) / x.std(axis=-1, ddof=1, keepdims=True)


def detrend_lin(x):
    """Least-squares removal of a + b t per row (same estimand as scipy.signal.detrend 'linear')."""
    x = np.asarray(x, float)
    T = x.shape[-1]
    t = np.arange(T, dtype=float)
    D = np.column_stack([np.ones(T), t])
    coef, *_ = np.linalg.lstsq(D, x.T, rcond=None)
    return x - (D @ coef).T


def prep(x, path):
    if path == "kernel":
        return np.asarray(x, float)
    if path == "pipeline":
        return zscore1(x)
    if path == "pipeline-detrend":
        return zscore1(detrend_lin(x))
    raise ValueError(path)


def parse_ident(ident):
    return dict(sq=ident.startswith("xcorr-sq"), stat="max" if "_max_" in ident else "mean",
                sig=ident.endswith("sig-True"))


# ------------------------------------------------------------------ pyspi-literal statistic
class PyspiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def literal_stat(xh, xl, sq, stat, sig):
    """pyspi's statistic for one pair in its orientation, through the kernel replica."""
    rep = pyspi_xcorr_replica(xh, xl)
    w = rep["r_window"]
    if sig:
        if "sig_error" in rep:
            raise PyspiError(1, "IndexError")
        N = rep["centre_index"]
        w = w[N - rep["fzr"]:N + rep["fzf"]]
    v = w ** 2 if sq else w
    if v.size == 0:
        if stat == "max":
            raise PyspiError(2, "ValueError (empty window)")
        return float("nan")
    return float(v.max() if stat == "max" else v.mean())


def literal_matrix(X, sq, stat, sig):
    P = X.shape[0]
    A = np.full((P, P), np.nan)
    for a in range(P):
        for b in range(a + 1, P):
            A[a, b] = A[b, a] = literal_stat(X[b], X[a], sq, stat, sig)
    return A


# ------------------------------------------------------------------ canonical (B&D) and steps
def bd_ccf(xh, xl):
    """lags (-(T-1)..T-1) and rho_hat(h) = gamma_hat_{hi,lo}(h)/sqrt(gamma_hat_hh(0) gamma_hat_ll(0)),
    gamma_hat(h) = T^-1 sum (x_{t+h}-xbar)(y_t-ybar): xcorr_full with ddof=1 is algebraically the same
    ((T-1) s1_x s1_y = T s0_x s0_y)."""
    return xcorr_full(xh, xl, ddof=1)


def window_vals(lags, r, T, window):
    L = T // 4
    if window == "doc":
        m = np.abs(lags) <= L
    else:  # pyspi: -(L-1) .. +L (basic.py:107)
        m = (lags >= -(L - 1)) & (lags <= L)
    return lags[m], r[m]




def pyspi_rule(r):
    N = len(r) // 2
    thr = 1.96 / np.sqrt(N)
    a = np.where(np.abs(r[N:]) <= thr)[0]
    b = np.where(np.abs(r[:N]) <= thr)[0]
    if a.size == 0 or b.size == 0:
        return None
    return r[N - int(b[-1]):N + int(a[0])]


def summ(v, sq, stat, empty=0.0):
    v = v ** 2 if sq else v
    if v.size == 0:
        return empty
    return float(v.max() if stat == "max" else v.mean())


def sig_filter(r, thr):
    """Documented reading: keep only the lags with |r| > thr (not necessarily contiguous)."""
    return r[np.abs(r) > thr]


def contig_at(r, i0, thr):
    """Contiguous run of |r| > thr containing index i0 (empty if r[i0] is insignificant)."""
    if abs(r[i0]) <= thr:
        return r[:0]
    lo = hi = i0
    while hi + 1 < r.size and abs(r[hi + 1]) > thr:
        hi += 1
    while lo > 0 and abs(r[lo - 1]) > thr:
        lo -= 1
    return r[lo:hi + 1]


STEP_KEYS = ("d0", "d1", "d2", "d3", "d3c", "d3c1", "d4", "c_doc")


def canon_steps(xh, xl, T, sq, stat, sig):
    """Step values (empty selection -> 0 for the documented/intermediate readings):
    d0   canonical/documented: B&D rho_hat, window |h| <= T//4, sig: lags with |rho| > 1.96/sqrt(T)
         (supplement 'only use the statistically significant values'), no factor;
    d1   + factor T/(T-1);
    d2   + pyspi window -(T//4-1)..T//4 (basic.py:107);
    d3   + implemented threshold 1.96/sqrt(T//4) (basic.py:118-119), still a significance filter;
    d3c  contiguous significant block around lag 0 (code comment 'truncate at first ... zero'), code thr;
    d3c1 contiguous block around the element at lag +1 (= pyspi's centre N), code thr, cut centred on lag 0;
    d4   pyspi's literal rule r[N-fzr:N+fzf] (mirrored left cut; NaN if pyspi would raise);
    c_doc contiguous block around lag 0 with the documented threshold on the documented window."""
    lags, rho = bd_ccf(xh, xl)
    f = T / (T - 1)
    L = T // 4
    thr_doc, thr_code = 1.96 / math.sqrt(T), 1.96 / math.sqrt(L)
    out = {}
    lw, rw = window_vals(lags, rho, T, "doc")
    lp, rp = window_vals(lags, f * rho, T, "pyspi")
    if not sig:
        out["d0"] = summ(rw, sq, stat)
        out["d1"] = summ(f * rw, sq, stat)
        for k in ("d2", "d3", "d3c", "d3c1", "d4"):
            out[k] = summ(rp, sq, stat)
        out["c_doc"] = out["d0"]
        return out
    out["d0"] = summ(sig_filter(rw, thr_doc), sq, stat)
    out["d1"] = summ(sig_filter(f * rw, thr_doc), sq, stat)
    out["d2"] = summ(sig_filter(rp, thr_doc), sq, stat)
    out["d3"] = summ(sig_filter(rp, thr_code), sq, stat)
    i0 = int(np.where(lp == 0)[0][0])
    out["d3c"] = summ(contig_at(rp, i0, thr_code), sq, stat)
    out["d3c1"] = summ(contig_at(rp, i0 + 1, thr_code), sq, stat)
    k = pyspi_rule(rp)
    out["d4"] = float("nan") if k is None else summ(k, sq, stat, empty=float("nan"))
    out["c_doc"] = summ(contig_at(rw, int(np.where(lw == 0)[0][0]), thr_doc), sq, stat)
    return out


# ------------------------------------------------------------------ population CCF (truth arrays)
def population_gammas(ds_name, d, procs, K):
    """Gamma(k) = E[x_{t+k} x_t^T] for k = 0..K (selected processes)."""
    if ds_name == "bivariate_gaussian":
        C = np.asarray(d["cov_true"])
        G = [C] + [np.zeros_like(C)] * K
    elif ds_name == "var2_bivariate":
        A1, A2, Sig = (np.asarray(d[k]) for k in ("A1", "A2", "Sigma"))
        P = A1.shape[0]
        F = np.block([[A1, A2], [np.eye(P), np.zeros((P, P))]])
        Q = np.zeros((2 * P, 2 * P))
        Q[:P, :P] = Sig
        S = solve_discrete_lyapunov(F, Q)
        G, Fk = [], np.eye(2 * P)
        for _ in range(K + 1):
            G.append((Fk @ S)[:P, :P])
            Fk = F @ Fk
    elif ds_name == "gauss_ar1_pair":
        A, SX = np.asarray(d["A"]), np.asarray(d["Sigma_X"])
        G, Ak = [], np.eye(A.shape[0])
        for _ in range(K + 1):
            G.append(Ak @ SX)
            Ak = A @ Ak
    else:
        raise ValueError(ds_name)
    G = np.array(G)
    if procs is not None:
        G = G[:, procs][:, :, procs]
    return G


def bartlett_var(rho11, rho22, rho12, h, T):
    """Bartlett's large-sample variance of rho_hat_12(h) (Brockwell & Davis, cross-correlation
    asymptotics for a bivariate linear process); rho12(k) = corr(X1_{t+k}, X2_t) as a dict-like
    callable over integer k, rho11/rho22 even."""
    M = 400
    s = 0.0
    r12h = rho12(h)
    for j in range(-M, M + 1):
        s += (rho11(j) * rho22(j) + rho12(j + h) * rho12(-(j - h))
              - 2 * r12h * (rho11(j) * rho12(j + h) + rho22(j) * rho12(-(j - h)))
              + r12h ** 2 * (rho12(j) ** 2 + 0.5 * rho11(j) ** 2 + 0.5 * rho22(j) ** 2))
    return s / T


# ------------------------------------------------------------------ helpers
def save(rid, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for k, v in kw.items():
        arrs[k] = np.asarray(v)
    np.savez(OUT / f"{rid}.npz", **arrs)


def load_pyspi(rid):
    p = PYOUT / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], float)


def split_tag(tag):
    sq = tag.endswith("-sq")
    return (tag[:-3] if sq else tag), sq


def max_offdiag_asym(A):
    m = ~np.isnan(A)
    return float(np.nanmax(np.abs(A - A.T)[m])) if m.any() else float("nan")


# ------------------------------------------------------------------ per-run references
def ref_replica(run, X, raw, ident):
    p = parse_ident(ident)
    T = X.shape[1]
    A = literal_matrix(X, **p)
    rep = pyspi_xcorr_replica(X[1], X[0])
    lags, rfull = xcorr_full(X[1], X[0], ddof=1)
    r0_full = float(rfull[T - 1])
    r0_rep = rep["r_lag0_replica"]
    extra = dict(lag0_xcorr_full_ddof1=r0_full, lag0_replica=r0_rep, lag0_ratio=r0_rep / r0_full,
                 T_over_Tm1=T / (T - 1), lag0_ratio_minus_factor=r0_rep / r0_full - T / (T - 1),
                 window_lag_min=int(rep["lags_window"][0]), window_lag_max=int(rep["lags_window"][-1]),
                 window_len=int(rep["lags_window"].size), centre_lag=int(rep["centre_lag"]),
                 thr_code=float(rep["sig_thr"]), thr_doc=float(1.96 / math.sqrt(T)),
                 all_finite=int(np.all(np.isfinite(A[~np.eye(X.shape[0], dtype=bool)]))),
                 max_abs_offdiag=float(np.nanmax(np.abs(A))))
    if p["sig"]:
        P = X.shape[0]
        kept, lag0_in, kept_sig = [], [], []
        for a in range(P):
            for b in range(a + 1, P):
                rp = pyspi_xcorr_replica(X[b], X[a])
                kl = rp["kept_lags"]
                kept.append(kl.size)
                lag0_in.append(int(0 in kl))
                N = rp["centre_index"]
                kw = rp["r_window"][N - rp["fzr"]:N + rp["fzf"]]
                kept_sig.append(float(np.mean(np.abs(kw) > rp["sig_thr"])) if kw.size else float("nan"))
        extra.update(kept_len_min=int(min(kept)), kept_len_max=int(max(kept)),
                     pairs_lag0_kept=int(sum(lag0_in)), n_pairs=len(kept),
                     kept_frac_significant_min=float(np.nanmin(kept_sig)),
                     kept_frac_significant_max=float(np.nanmax(kept_sig)))
        if run["dataset"] == "delayed_noise":
            rp = pyspi_xcorr_replica(X[3], X[0])
            extra.update(pair03_fzr=int(rp["fzr"]), pair03_fzf=int(rp["fzf"]),
                         pair03_kept_lag_min=int(rp["kept_lags"][0]) if rp["kept_lags"].size else 0,
                         pair03_kept_lag_max=int(rp["kept_lags"][-1]) if rp["kept_lags"].size else 0,
                         pair03_r_lag1=float(rp["r_window"][rp["centre_index"]]),
                         pair03_r_lag5=float(rp["r_window"][rp["centre_index"] + 4]))
    if run["dataset"] == "scaled_offset":
        Xz = zscore1(raw)
        B = np.full_like(A, np.nan)
        for a in range(4):
            for b in range(a + 1, 4):
                lags, rho = bd_ccf(Xz[b], Xz[a])
                _, rw = window_vals(lags, rho, T, "pyspi")
                B[a, b] = B[b, a] = summ(rw, p["sq"], p["stat"])
        extra.update(canon_centred_A23=float(B[2, 3]), raw_literal_A23=float(A[2, 3]),
                     canon_centred_A13=float(B[1, 3]), raw_literal_A13=float(A[1, 3]),
                     max_abs_raw_minus_centred=float(np.nanmax(np.abs(A - B))))
    return dict(value=A, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{KERN}:pyspi_xcorr_replica (np.convolve direct sum, not scipy.signal.correlate) via "
                                 f"{SELF}:literal_matrix; preprocessing by {SELF}:prep",
                notes="pyspi orientation x=x_hi, y=x_lo; lag0 fields are pair (hi=1, lo=0): pyspi r(0) / "
                      "xcorr_full(ddof=1) r(0) vs T/(T-1)", **extra)


def ref_canon(run, X, raw, ident):
    p = parse_ident(ident)
    T, P = X.shape[1], X.shape[0]
    S = load_pyspi(run["run_id"])
    D = {k: np.full((P, P), np.nan) for k in STEP_KEYS}
    Xc = zscore1(raw) if run["path"] == "kernel" else X   # canonical rho_hat is centred anyway
    for a in range(P):
        for b in range(a + 1, P):
            st = canon_steps(Xc[b], Xc[a], T, **p)
            for k in D:
                D[k][a, b] = D[k][b, a] = st[k]
    extra = {}
    if S is not None:
        for k in D:
            m = ~np.isnan(S) & ~np.isnan(D[k])
            extra[f"gap_{k}"] = float(np.max(np.abs(S[m] - D[k][m]))) if m.any() else float("nan")
        m = ~np.isnan(S)
        extra["max_rel_gap_d0"] = float(np.max(np.abs(S[m] - D["d0"][m]) / np.maximum(np.abs(D["d0"][m]), 1e-300)))
    return dict(value=D["d0"], criterion="tolerance", rtol=1e-8, atol=1e-10, reference_kind="analytic",
                reference_detail=f"{SELF}:canon_steps; Brockwell & Davis sample CCF (divisor T, centred) via "
                                 f"{KERN}:xcorr_full(ddof=1); documented window |h|<=T//4; sig: documented rule "
                                 "keep lags with |rho|>1.96/sqrt(T) (empty -> 0)",
                notes="gap_K = max|pyspi - step K|: d0 canonical/documented, d1 x T/(T-1), d2 + pyspi window "
                      "-(T//4-1)..T//4, d3 + threshold 1.96/sqrt(T//4) (filter), d3c contiguous about lag 0, "
                      "d3c1 contiguous about lag +1, d4 pyspi literal mirrored cut; c_doc contiguous about lag 0 "
                      "with the documented threshold/window",
                **extra)


def ref_perm_replica(run, X, raw, ident):
    p = parse_ident(ident)
    A = literal_matrix(X, **p)              # reversed index space (X is already reversed)
    perm = np.asarray(run["processes"])
    inv = np.argsort(perm)
    Xf = X[inv]
    Af = literal_matrix(Xf, **p)            # forward order
    extra = dict(replica_order_gap=float(np.nanmax(np.abs(A[np.ix_(inv, inv)] - Af))))
    base, sq = split_tag(run["run_id"].split("|")[-1])
    fwd = "|".join(run["run_id"].split("|")[:-1] + ["replica" + ("-sq" if sq else "")])
    Sf, Sr = load_pyspi(fwd), load_pyspi(run["run_id"])
    if Sf is not None and Sr is not None:
        G = Sr[np.ix_(inv, inv)]
        extra["pyspi_order_gap"] = float(np.nanmax(np.abs(G - Sf)))
        if run["dataset"] == "delayed_noise":
            extra.update(pyspi_fwd_A03=float(Sf[0, 3]), pyspi_rev_A03=float(G[0, 3]))
    return dict(value=A, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{KERN}:pyspi_xcorr_replica on the reversed channel order",
                notes="pyspi on reversed order vs replica on reversed order; order gaps compare with the "
                      "forward-order run " + fwd, **extra)


def ref_perm_selfcons(run, X, raw, ident):
    perm = np.asarray(run["processes"])
    base, sq = split_tag(run["run_id"].split("|")[-1])
    fwd = "|".join(run["run_id"].split("|")[:-1] + ["replica" + ("-sq" if sq else "")])
    Sf = load_pyspi(fwd)
    V = Sf[np.ix_(perm, perm)]
    return dict(value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="same-lineage",
                reference_detail=f"pyspi itself: forward-order output {fwd} permuted into the reversed index space",
                notes="same-lineage self-consistency: an undirected SPI should not depend on channel order")


def ref_selfcopy(run, X, raw, ident):
    p = parse_ident(ident)
    T = X.shape[1]
    f = T / (T - 1)
    assert np.array_equal(raw[0], raw[1])
    lags, rho = bd_ccf(X[1], X[0])
    lp, rp = window_vals(lags, f * rho, T, "pyspi")
    if p["sig"]:
        kept = pyspi_rule(rp)
    else:
        kept = rp
    lag0_in = bool(kept is not None and np.any(np.isclose(kept, f, rtol=0, atol=1e-12)))
    if p["stat"] == "max" and lag0_in:
        v = f ** 2 if p["sq"] else f
        kind, det = "analytic", "T/(T-1) (squared: (T/(T-1))^2): rho_hat(0)=1 for identical series, scaled"
    else:
        v = summ(kept, p["sq"], p["stat"], empty=float("nan"))
        kind, det = "analytic", "T/(T-1) x B&D autocorrelation of x0 over pyspi's window/rule"
    A = np.array([[np.nan, v], [v, np.nan]])
    rep = pyspi_xcorr_replica(X[1], X[0])
    return dict(value=A, criterion="tolerance", rtol=0.0, atol=1e-12, reference_kind=kind,
                reference_detail=f"{SELF}:ref_selfcopy: {det}",
                notes="shared_envelope processes [0,2], x2 bit-identical to x0; canonical lag-0 value 1",
                T_over_Tm1=f, lag0_replica_minus_factor=rep["r_lag0_replica"] - f, lag0_in_window=int(lag0_in),
                exceeds_one=int(v > 1))


def _delays(run):
    d = np.asarray(make_data.load(run["dataset"], run["T"])["delay"], float)
    if run.get("processes") is not None:
        d = d[list(run["processes"])]
    return d


def ref_lag_cached(run, X, raw, ident):
    d = _delays(run)
    P = d.size
    V = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                V[i, j] = d[max(i, j)] - d[min(i, j)]
    sign = np.sign(np.nan_to_num(V))
    return dict(value=V, criterion="direction", expected_sign=sign, reference_kind="analytic",
                reference_detail="delayed_noise construction x_k(t)=x0(t-d_k)+noise, d=(0,1,3,5): "
                                 "correlate(x_hi, x_lo) peaks at +(d_hi - d_lo)",
                notes="cache used by multivariate is correlate(x_hi, x_lo) for both (i,j) and (j,i): "
                      "positive lag = LOWER-index channel leads")


def ref_lag_fresh(run, X, raw, ident):
    d = _delays(run)
    V = d[:, None] - d[None, :]
    V = V.astype(float)
    np.fill_diagonal(V, np.nan)
    return dict(value=V, criterion="direction", expected_sign=np.sign(np.nan_to_num(V)),
                reference_kind="analytic",
                reference_detail="delayed_noise construction: correlate(x_i, x_j) peaks at lag d_i - d_j",
                notes="bivariate(i,j) evaluated first: positive peak lag means x_j leads x_i (x_i is x_j delayed)")


def ref_cached_r(run, X, raw, ident):
    T, P = X.shape[1], X.shape[0]
    L = T // 4
    f = T / (T - 1)
    rows, rows_shift, ratios = [], [], []
    for a in range(P):
        for b in range(a + 1, P):
            lags, rho = bd_ccf(X[b], X[a])
            m = (lags >= -(L - 1)) & (lags <= L)
            ms = (lags >= -L) & (lags <= L - 1)
            rows.append(f * rho[m])
            rows_shift.append(f * rho[ms])
    V = np.vstack(rows)
    S = load_pyspi(run["run_id"])
    extra = {}
    if S is not None:
        i0 = L - 1
        big = np.abs(V) > 1e-3
        extra.update(lag0_pyspi_pair01=float(S[0, i0]), lag0_bd_pair01=float(V[0, i0] / f),
                     lag0_ratio_pair01=float(S[0, i0] / (V[0, i0] / f)), T_over_Tm1=f,
                     lag0_ratio_minus_factor=float(S[0, i0] / (V[0, i0] / f) - f),
                     max_ratio_dev_all_lags=float(np.max(np.abs(S[big] / (V[big] / f) - f))),
                     shifted_window_max_abs=float(np.max(np.abs(S - np.vstack(rows_shift)))))
    return dict(value=V, criterion="tolerance", rtol=0.0, atol=1e-12, reference_kind="analytic",
                reference_detail=f"T/(T-1) x Brockwell & Davis rho_hat via {KERN}:xcorr_full(x_hi, x_lo, ddof=1) "
                                 "on lags -(T//4-1)..T//4",
                notes="rows = pairs (0,1),(0,2),...; lag of column c is c-(T//4-1); shifted_window_max_abs = "
                      "mismatch if the window were -T//4..T//4-1", **extra)


def ref_null(run, X, raw, ident):
    T, P = X.shape[1], X.shape[0]
    L = T // 4
    f = T / (T - 1)
    h = np.arange(-(L - 1), L + 1)
    sd = f * np.sqrt((T - np.abs(h)) / T ** 2)
    npairs = P * (P - 1) // 2
    out = []
    for thr in (1.96 / math.sqrt(T), 1.96 / math.sqrt(L)):
        pr = 2 * norm.sf(thr / sd)
        out.append((npairs * pr.sum(), math.sqrt(npairs * (pr * (1 - pr)).sum())))
    V = np.array([npairs * h.size, out[0][0], out[1][0]])
    se = np.array([0.0, out[0][1], max(out[1][1], 0.5)])
    S = load_pyspi(run["run_id"])
    extra = dict(nominal_frac_doc=0.05, nominal_frac_code=float(2 * norm.sf(1.96 * math.sqrt(T / L))),
                 expected_frac_doc=float(out[0][0] / V[0]), expected_frac_code=float(out[1][0] / V[0]),
                 thr_doc=1.96 / math.sqrt(T), thr_code=1.96 / math.sqrt(L))
    if S is not None:
        extra.update(observed_frac_doc=float(S[1] / S[0]), observed_frac_code=float(S[2] / S[0]),
                     observed_n_doc=int(S[1]), observed_n_code=int(S[2]))
    return dict(value=V, se=se, criterion="statistical", reference_kind="analytic",
                reference_detail="null: independent white series, rho_hat(h) ~ N(0,(T-|h|)/T^2) (Bartlett), "
                                 "pyspi r = T/(T-1) rho_hat; expected exceedance counts over all pairs and window lags",
                notes="value = [n_lags, E n(|r|>1.96/sqrt(T)), E n(|r|>1.96/sqrt(T//4))]; se from binomial "
                      "variance (floor 0.5 count for the code threshold)", **extra)


def predict_status(X, ident):
    p = parse_ident(ident)
    P = X.shape[0]
    M = np.zeros((P, P))
    info = dict(n_fail_pos_side=0, n_fail_neg_side=0)
    for a in range(P):
        for b in range(a + 1, P):
            try:
                v = literal_stat(X[b], X[a], **p)
                c = 0 if np.isfinite(v) else 4
            except PyspiError as e:
                c = e.code
                if c == 1:
                    rp = pyspi_xcorr_replica(X[b], X[a])
                    w, N, thr = rp["r_window"], rp["centre_index"], rp["sig_thr"]
                    info["n_fail_pos_side"] += int(not np.any(np.abs(w[N:]) <= thr))
                    info["n_fail_neg_side"] += int(not np.any(np.abs(w[:N]) <= thr))
            M[a, b] = M[b, a] = c
    diag, nonfinite = 0, False
    for j in range(P):                       # Directed.multivariate order (pyspi/base.py:91-93)
        for i in range(P):
            if i == j:
                continue
            c = M[min(i, j), max(i, j)]
            if c in (1, 2, 3):
                diag = int(c)
                break
            nonfinite |= c == 4
        if diag:
            break
    if not diag and nonfinite:
        diag = 4
    np.fill_diagonal(M, diag)
    return M, info


def ref_errsweep(run, X, raw, ident):
    M, info = predict_status(X, ident)
    S = load_pyspi(run["run_id"])
    ok = S is not None and S.shape == M.shape and np.array_equal(S, M)
    P = X.shape[0]
    off = ~np.eye(P, dtype=bool)
    n_err = int(np.sum(M[off] == 1) // 2)
    T = X.shape[1]
    note = (f"predicted multivariate status {int(M[0, 0])} (0 ok, 1 IndexError, 2 ValueError, 4 NaN); "
            f"{n_err} of {P * (P - 1) // 2} pairs IndexError (no |r| <= 1.96/sqrt(T//4)={1.96 / math.sqrt(T // 4):.4f} "
            f"on the positive half in {info['n_fail_pos_side']} pairs, on the negative half in "
            f"{info['n_fail_neg_side']}); probe status matrix {'equals' if ok else 'DIFFERS from'} the prediction")
    return dict(value=M, criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
                reference_detail=f"{SELF}:predict_status via {KERN}:pyspi_xcorr_replica (own preprocessing)",
                notes=note, predicted_multivariate_status=int(M[0, 0]),
                observed_multivariate_status=int(S[0, 0]) if S is not None else -1,
                n_pairs_indexerror=n_err, n_pairs=P * (P - 1) // 2, **info)


def ref_pop(run, X, raw, ident):
    p = parse_ident(ident)
    T = X.shape[1]
    L = T // 4
    d = make_data.load(run["dataset"], T)
    procs = run.get("processes")
    K = 450
    G = population_gammas(run["dataset"], d, procs, K)
    sd = np.sqrt(np.diag(G[0]))

    def rho(i, j):
        def f(k):
            if abs(k) > K:
                return 0.0
            g = G[k][i, j] if k >= 0 else G[-k][j, i]
            return float(g / (sd[i] * sd[j]))
        return f
    hi, lo = 1, 0
    r12 = rho(hi, lo)
    lags = np.arange(-(L - 1), L + 1)
    vals = np.array([r12(int(k)) for k in lags])
    kstar = int(lags[np.argmax(vals)])
    pk = float(vals.max())
    var = bartlett_var(rho(hi, hi), rho(lo, lo), r12, kstar, T)
    v = pk ** 2 if p["sq"] else pk
    A = np.array([[np.nan, v], [v, np.nan]])
    se = np.full((2, 2), math.sqrt(var) * (2 * abs(pk) if p["sq"] else 1.0))
    S = load_pyspi(run["run_id"])
    extra = dict(pop_peak=pk, pop_peak_lag=kstar, bartlett_se=math.sqrt(var), T_over_Tm1=T / (T - 1))
    if S is not None:
        extra.update(pyspi_value=float(S[0, 1]), z=float((S[0, 1] - v) / (math.sqrt(var))))
    return dict(value=A, se=se, criterion="statistical", reference_kind="analytic",
                reference_detail=f"{SELF}:ref_pop: population CCF from truth arrays (Gamma(k)=E[x_(t+k) x_t^T]), "
                                 "peak over pyspi's window, Bartlett variance at the peak lag",
                notes="pyspi orientation (x_hi, x_lo); the T/(T-1) factor is not applied (canonical target)",
                **extra)


def ref_zerosum(run, X, raw, ident):
    T, P = X.shape[1], X.shape[0]
    L = T // 4
    f = T / (T - 1)
    V = np.full((P, P), np.nan)
    lag0_contrib, tot = [], []
    for a in range(P):
        for b in range(a + 1, P):
            lags, rho = bd_ccf(X[b], X[a])
            m = (lags >= -(L - 1)) & (lags <= L)
            V[a, b] = V[b, a] = -f * rho[~m].sum() / (2 * L)
            lag0_contrib.append(f * rho[T - 1] / (2 * L))
            tot.append(abs(rho.sum()))
    return dict(value=V, criterion="tolerance", rtol=0.0, atol=1e-12, reference_kind="analytic",
                reference_detail="identity: sum over all 2T-1 lags of centred cross-products = (sum x)(sum y) = 0, so "
                                 "window mean = -(T/(T-1)) sum_{outside} rho_hat / (2 T//4)",
                notes="lag0_contrib = r(0)/(2L) per pair", max_abs_value=float(np.nanmax(np.abs(V))),
                max_abs_lag0_contrib=float(np.max(np.abs(lag0_contrib))), max_abs_total_sum=float(max(tot)))


def crafted_r(T, case):
    """Same formula as probes/xcorr_probes.py:crafted_r (written independently here)."""
    L = T // 4
    lags = np.arange(-(L - 1), L + 1)
    if case == "A":
        return lags, np.where(np.abs(lags) <= 2, 0.9, 0.0)
    return lags, np.where((lags >= -(L - 2)) & (lags <= 0), 0.9, 0.0)


def ref_crafted(run, X, raw, ident):
    p = parse_ident(ident)
    T = X.shape[1]
    vals, extra = [], {}
    for case in ("A", "B"):
        lags, r = crafted_r(T, case)
        k = pyspi_rule(r)
        if k is None:
            v, c = float("nan"), 1
        elif k.size == 0:
            v, c = (float("nan"), 2) if p["stat"] == "max" else (float("nan"), 4)
        else:
            v, c = summ(k, p["sq"], p["stat"]), 0
        vals += [v, c]
        N = len(r) // 2
        if k is not None:
            b = np.where(np.abs(r[:N]) <= 1.96 / np.sqrt(N))[0][-1]
            a = np.where(np.abs(r[N:]) <= 1.96 / np.sqrt(N))[0][0]
            kl = lags[N - b:N + a]
            extra[f"case{case}_kept_lag_min"] = int(kl[0]) if kl.size else 0
            extra[f"case{case}_kept_lag_max"] = int(kl[-1]) if kl.size else 0
            extra[f"case{case}_kept_len"] = int(kl.size)
        extra[f"case{case}_documented"] = summ(sig_filter(r, 1.96 / math.sqrt(T)), p["sq"], p["stat"], empty=0.0)
    return dict(value=np.array(vals), criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="analytic",
                reference_detail=f"{SELF}:ref_crafted: pyspi's rule basic.py:116-120 evaluated by hand on crafted r",
                notes="[value_A, status_A, value_B, status_B]; A: r=0.9 at |lag|<=2 else 0; B: r=0.9 on lags "
                      "-(T//4-2)..0 else 0 (fzr=fzf=0, empty window); status 0 ok, 2 ValueError, 4 NaN", **extra)


def ref_endlag(run, X, raw, ident):
    """Constructed pair with the CCF peak on the window end lag (same construction as
    probes/xcorr_probes.py:endlag_pair, written independently here): x0 = raw[0], y = np.roll(x0, L),
    L = T//4; forward order [x0, y] (x_hi = y, peak at lag +L, inside -(L-1)..L) and reversed order
    [y, x0] (x_hi = x0, peak at lag -L, outside).  Value [A_fwd, A_rev] from the replica (exact)."""
    p = parse_ident(ident)
    x0 = np.asarray(raw[0], float)
    T = x0.size
    L = T // 4
    y = np.roll(x0, L)
    vals, canon, peak = [], [], []
    for arr in (np.vstack([x0, y]), np.vstack([y, x0])):
        Z = prep(arr, run["path"])
        vals.append(literal_stat(Z[1], Z[0], **p))
        lags, rho = bd_ccf(Z[1], Z[0])
        _, rw = window_vals(lags, rho, T, "doc")
        canon.append(summ(rw, p["sq"], p["stat"]))
        peak.append(int(lags[np.argmax(rho)]))
    return dict(value=np.array(vals), criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{KERN}:pyspi_xcorr_replica via {SELF}:literal_stat on the constructed pair "
                                 f"({SELF}:ref_endlag)",
                notes="[A_fwd, A_rev]; y = x0 delayed by T//4 (circular roll); fwd peak at lag +T//4 (inside "
                      "pyspi's window), rev peak at -T//4 (outside); canon_* = Brockwell & Davis max over the "
                      "documented symmetric window |h| <= T//4 (order-invariant)",
                order_gap=float(vals[0] - vals[1]), canon_fwd=float(canon[0]), canon_rev=float(canon[1]),
                canon_order_gap=float(canon[0] - canon[1]), peak_lag_fwd=peak[0], peak_lag_rev=peak[1],
                window_end_lag=L)


DISPATCH = {"crafted": ref_crafted, "replica": ref_replica, "canon": ref_canon, "perm-replica": ref_perm_replica,
            "perm-selfcons": ref_perm_selfcons, "selfcopy": ref_selfcopy, "lag-cached": ref_lag_cached,
            "lag-fresh": ref_lag_fresh, "cached-r": ref_cached_r, "null": ref_null, "errsweep": ref_errsweep,
            "zerosum": ref_zerosum, "endlag": ref_endlag, "pop": ref_pop, "pop23": ref_pop, "pop01": ref_pop}


def main():
    m = json.loads((CHECKS / "manifest" / "xcorr.json").read_text())
    n = 0
    for run in m["runs"]:
        tag, _ = split_tag(run["run_id"].split("|")[-1])
        d = make_data.load(run["dataset"], run["T"])
        raw = np.asarray(d["x"], float)
        if run.get("processes") is not None:
            raw = raw[list(run["processes"])]
        X = prep(raw, run["path"])
        out = DISPATCH[tag](run, X, raw, run["identifier"])
        save(run["run_id"], **out)
        n += 1
    print(f"xcorr refs: {n} written")


if __name__ == "__main__":
    main()
