"""Reference kernels: basic statistics (covariance family, cross-correlation,
rank correlations, pairwise distances) and miscellaneous pyspi SPIs
(cointegration, 1-D linear models, envelope correlation, IDS, GP regression).

Independent re-implementations from the cited papers/definitions using only
numpy/scipy primitives.  Library sources (sklearn, statsmodels, scipy.signal,
mne_connectivity, pyspi/lib/ids) were READ to mirror conventions (cited as
file:line, paths relative to .venv/lib/python3.12/site-packages unless they
start with pyspi/) and are never imported.

Array convention for every multivariate function in this module: X has shape
(P, T) = (processes, observations), the pyspi/MANIFEST "ps" order.  NOTE this
is the TRANSPOSE of sklearn's (n_samples, n_features).

Run ``python _kernels_basic_misc.py --selftest``.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import linalg as sla
from scipy import optimize
from scipy.signal import hilbert

EPS = np.finfo(np.float64).eps


# ============================================================ covariance family
def empirical_cov(X, ddof=0):
    """Sample covariance S = (X - mean)(X - mean)^T / (T - ddof); X (P, T) -> (P, P).

    ddof=0 mirrors sklearn empirical_covariance (np.cov(X.T, bias=1),
    sklearn/covariance/_empirical_covariance.py:107)."""
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=1, keepdims=True)
    return Xc @ Xc.T / (X.shape[1] - ddof)


def precision(S):
    """Moore-Penrose inverse of a symmetric matrix by eigendecomposition, i.e.
    the scipy.linalg.pinvh algorithm sklearn uses for ``precision_``
    (sklearn/covariance/_empirical_covariance.py:213): eigenvalues with
    |w| <= max|w| * P * eps are discarded (pinvh default rtol)."""
    w, V = np.linalg.eigh(np.asarray(S, float))
    keep = np.abs(w) > np.abs(w).max() * S.shape[0] * EPS
    return (V[:, keep] / w[keep]) @ V[:, keep].T


def shrunk_cov(X, shrinkage=0.1):
    """(1 - s) S + s mu I with mu = tr(S)/P, S the ddof=0 covariance; X (P, T).
    Target mu*I as sklearn shrunk_covariance (_shrunk_covariance.py:151-154)."""
    S = empirical_cov(X)
    mu = np.trace(S) / S.shape[0]
    return (1 - shrinkage) * S + shrinkage * mu * np.eye(S.shape[0]), shrinkage


def ledoit_wolf(X):
    """Ledoit & Wolf (2004) J. Multivariate Anal. 88:365, eq. 14 (Lemma 3.2-3.4):
    S* = (b^2/d^2) m I + (a^2/d^2) S with <A,B> = tr(A B^T)/P, m = <S, I>,
    d^2 = ||S - m I||^2, bbar^2 = T^-2 sum_k ||x_k x_k^T - S||^2,
    b^2 = min(bbar^2, d^2), a^2 = d^2 - b^2.  Shrinkage = b^2/d^2.
    The bbar^2 sum is evaluated literally per sample (no expansion).
    X (P, T), centred internally; returns (cov (P,P), shrinkage).
    Same estimator as sklearn ledoit_wolf_shrinkage (_shrunk_covariance.py:364-399),
    whose shrinkage is 0 when b^2 == 0 (mirrored)."""
    X = np.asarray(X, float)
    P, T = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    S = Xc @ Xc.T / T
    m = np.trace(S) / P
    d2 = np.sum((S - m * np.eye(P)) ** 2) / P
    outer = np.einsum("it,jt->tij", Xc, Xc)
    bbar2 = np.sum((outer - S) ** 2) / P / T**2
    b2 = min(bbar2, d2)
    s = 0.0 if b2 == 0 else b2 / d2
    return s * m * np.eye(P) + (1 - s) * S, s


def oas(X, paper=True):
    """Oracle Approximating Shrinkage, Chen, Wiesel, Eldar & Hero (2010) IEEE TSP
    58:5016, eq. 23:  rho = min( ((1 - 2/p) tr(S^2) + tr^2(S)) /
    ((n + 1 - 2/p)(tr(S^2) - tr^2(S)/p)), 1 ), Sigma = (1-rho) S + rho tr(S)/p I.
    paper=False reproduces sklearn's variant that drops the 2/p terms
    (_shrunk_covariance.py:64-85; rho=1 when the denominator is 0), including its
    single-feature special case rho = 0.0 for P == 1 (_shrunk_covariance.py:55-59;
    the covariance is the same for any rho when P == 1).  paper=True with P == 1
    gives a 0/0 denominator and keeps the rho=1 rule (cov unchanged either way).
    X (P, T) with n = T, p = P; returns (cov, rho)."""
    S = empirical_cov(X)
    p, n = S.shape[0], X.shape[1]
    if p == 1 and not paper:
        return S, 0.0
    trS2, tr2S = np.sum(S * S), np.trace(S) ** 2
    c = 2.0 / p if paper else 0.0
    den = (n + 1 - c) * (trS2 - tr2S / p)
    rho = 1.0 if den == 0 else min(((1 - c) * trS2 + tr2S) / den, 1.0)
    return (1 - rho) * S + rho * np.trace(S) / p * np.eye(p), rho


def glasso_kkt_residual(S, Theta, alpha):
    """Max violation of the graphical-lasso optimality (KKT) conditions,
    Friedman, Hastie & Tibshirani (2008) Biostatistics 9:432 eq. 2 / Banerjee 2008:
    S - W + alpha * Gamma = 0, W = Theta^-1, Gamma_ij = sign(Theta_ij) if
    Theta_ij != 0 else any value in [-1, 1].  The diagonal is NOT penalised
    (sklearn convention, _graph_lasso.py:52 and docstring :329), so the diagonal
    condition is S_ii = W_ii.  Returns max over entries of
    |S - W + alpha sign(Theta)| (nonzero off-diagonal), max(|S - W| - alpha, 0)
    (zero off-diagonal) and |S - W| (diagonal)."""
    S, Theta = np.asarray(S, float), np.asarray(Theta, float)
    G = S - np.linalg.inv(Theta)
    off = ~np.eye(S.shape[0], dtype=bool)
    nz = off & (Theta != 0)
    z = off & (Theta == 0)
    v = np.zeros_like(G)
    v[nz] = np.abs(G[nz] + alpha * np.sign(Theta[nz]))
    v[z] = np.maximum(np.abs(G[z]) - alpha, 0.0)
    v[~off] = np.abs(G[~off])
    return float(v.max())


# ============================================================ cross-correlation
def xcorr_full(x, y, ddof=1):
    """Normalised cross-correlation for every lag tau = -(T-1)..(T-1):
        r(tau) = sum_t (x[t+tau] - xbar)(y[t] - ybar) / ((T - ddof) s_x s_y),
    s with the same ddof (so r(0) is exactly Pearson's r).  Fixed (T - ddof)
    denominator at every lag (biased / no overlap correction).
    SIGN CONVENTION (same as scipy.signal.correlate(x, y, 'full'),
    scipy/signal/_signaltools.py:168 and correlation_lags :383): a peak at tau > 0
    means x[t + tau] ~ y[t], i.e. Y LEADS X (x is y delayed by tau).
    Returns (lags (2T-1,), r (2T-1,))."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    T = x.size
    xc, yc = x - x.mean(), y - y.mean()
    lags = np.arange(-(T - 1), T)
    r = np.array([np.dot(xc[max(k, 0):T + min(k, 0)], yc[max(-k, 0):T - max(k, 0)]) for k in lags])
    return lags, r / ((T - ddof) * xc.std(ddof=ddof) * yc.std(ddof=ddof))


def pyspi_xcorr_replica(x, y):
    """Line-by-line replica of pyspi CrossCorrelation (pyspi/statistics/basic.py:103-120)
    on the series AS PYSPI SEES THEM (pass z-scored ddof=1 data, pyspi/data.py:195):
      r = correlate(x, y, 'full') / x.std() / y.std() / (T-1)   (std ddof=0, no demeaning)
      r = r[T - T//4 : T + T//4]   -> index i <-> lag i - T//4 + 1 (window centre = lag +1)
      sigonly: N = len(r)//2 (= T//4), thr = 1.96/sqrt(N) (N, not T),
               fzf = first i in r[N:] with |r| <= thr, fzr = LAST i in r[:N] with |r| <= thr,
               kept = r[N - fzr : N + fzf]   (start uses N - fzr, not fzr + 1).
    Returns a dict with the window, its lags, the sigonly fields (or 'sig_error' when
    np.where(...)[0][0] would raise IndexError in pyspi), max/mean statistics, and an
    'intended' variant (thr = 1.96/sqrt(T), contiguous block around lag 0 bounded by the
    first |r| <= thr on each side) so the offset/threshold hypotheses can be quantified.

    EMPTY KEPT WINDOW (fzr = fzf = 0): pyspi's ``np.max`` raises ValueError (zero-size
    array) and its ``np.mean`` returns NaN (pyspi/statistics/basic.py:122-131).  This
    replica does not raise, so the other fields stay available: it returns
    ``max_sig = mean_sig = NaN`` and adds ``max_sig_error`` naming pyspi's ValueError.
    Callers that need pyspi's outcome must test for that key (refs/xcorr.py:literal_stat
    raises its own status codes instead).

    The 'intended' variant is NOT a documented reading of the SPI.  It always includes
    lag 0 (even when |r(0)| <= thr), extends over the full lag range -(T-1)..T-1 rather
    than pyspi's +-T//4 window, and uses pyspi's scaled values (r(0) = Pearson r x T/(T-1)
    on ddof=1 z-scored input).  The audit's reading of the documentation is the ``canon``
    reference of refs/xcorr.py (canon_steps step d0): the Brockwell & Davis sample CCF over
    |h| <= T//4; for sig=True every lag in that window with |rho_hat| > 1.96/sqrt(T), not
    necessarily contiguous (empty -> 0); for sig=False the whole window, unfiltered.  The
    contiguous block around lag 0 is a separate step of the same run (c_doc, reported as
    ``gap_c_doc``)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    T = x.size
    full = np.convolve(x, y[::-1], mode="full") / x.std() / y.std() / (T - 1)
    full_lags = np.arange(-(T - 1), T)
    w = full[T - T // 4:T + T // 4]
    lags = full_lags[T - T // 4:T + T // 4]
    N = len(w) // 2
    out = dict(r_window=w, lags_window=lags, centre_index=N, centre_lag=int(lags[N]),
               r_lag0_replica=float(full[T - 1]), max_nosig=float(w.max()), mean_nosig=float(w.mean()))
    thr = 1.96 / np.sqrt(N)
    out["sig_thr"] = thr
    a = np.where(np.abs(w[N:]) <= thr)[0]
    b = np.where(np.abs(w[:N]) <= thr)[0]
    if a.size == 0 or b.size == 0:
        out["sig_error"] = "IndexError: no |r| <= thr on one side"
    else:
        fzf, fzr = int(a[0]), int(b[-1])
        kept = w[N - fzr:N + fzf]
        out.update(fzf=fzf, fzr=fzr, kept_lags=lags[N - fzr:N + fzf],
                   max_sig=float(kept.max()) if kept.size else np.nan,
                   mean_sig=float(kept.mean()) if kept.size else np.nan)
        if not kept.size:
            out["max_sig_error"] = "ValueError: empty kept window (pyspi np.max raises; np.mean gives NaN)"
    thr_i = 1.96 / np.sqrt(T)
    lo = hi = T - 1                       # index of lag 0 in `full`
    while hi + 1 < full.size and abs(full[hi + 1]) > thr_i: hi += 1
    while lo > 0 and abs(full[lo - 1]) > thr_i: lo -= 1
    seg = full[lo:hi + 1]
    out.update(intended_thr=thr_i, intended_lags=full_lags[lo:hi + 1],
               intended_max=float(seg.max()), intended_mean=float(seg.mean()))
    return out


# ============================================================ rank correlations
def _rank_average(a):
    """1-based ranks with ties given their average rank."""
    a = np.asarray(a, float)
    s = np.sort(a)   # a tie block occupies 0-based positions lo..hi-1 -> mean 1-based rank (lo+1+hi)/2
    return (np.searchsorted(s, a, "left") + 1 + np.searchsorted(s, a, "right")) / 2.0


def spearman(x, y):
    """Spearman (1904) rho: Pearson correlation of average ranks (ties averaged)."""
    rx, ry = _rank_average(x), _rank_average(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def kendall_tau_b(x, y):
    """Kendall (1945) Biometrika 33:239 tie-corrected tau_b by O(n^2) pair count:
    tau_b = (n_c - n_d) / sqrt((n0 - n1)(n0 - n2)), n0 = n(n-1)/2, n1 / n2 = pairs
    tied in x / y.  Pairs tied in x or y count as neither concordant nor discordant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    i, j = np.triu_indices(n, 1)
    sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    s = np.sum(sx * sy)
    n0 = n * (n - 1) / 2
    n1, n2 = np.sum(sx == 0), np.sum(sy == 0)
    return float(s / np.sqrt((n0 - n1) * (n0 - n2)))


def _kendall_tau_b_knight(x, y):
    """Knight (1966) JASA 61:436 O(n log n) tau_b (merge-sort inversion count),
    independent second computation used only by the selftest."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    o = np.lexsort((y, x))
    xs, ys = x[o], list(y[o])

    ties = lambda c: float(np.sum(c * (c - 1) / 2))
    n1, n2 = (ties(np.unique(v, return_counts=True)[1]) for v in (xs, ys))
    n3 = ties(np.unique(np.stack([xs, np.asarray(ys)]), axis=1, return_counts=True)[1])

    def sort_count(v):
        if len(v) <= 1:
            return v, 0
        a, ca = sort_count(v[:len(v) // 2])
        b, cb = sort_count(v[len(v) // 2:])
        out, k, i, j = [], ca + cb, 0, 0
        while i < len(a) and j < len(b):
            if b[j] < a[i]:
                out.append(b[j]); k += len(a) - i; j += 1
            else:
                out.append(a[i]); i += 1
        return out + a[i:] + b[j:], k

    _, swaps = sort_count(ys)
    n0 = n * (n - 1) / 2
    return float((n0 - n1 - n2 + n3 - 2 * swaps) / np.sqrt((n0 - n1) * (n0 - n2)))


# ============================================================ pairwise distances
def pdist_metric(X, metric):
    """(P, P) distances between the rows of X (P, T) (each process's whole time series
    is one vector, as pyspi PairwiseDistance: pairwise_distances(data (P,T)),
    pyspi/statistics/distance.py:40).  Definitions (scipy.spatial.distance docs):
    euclidean sqrt(sum (u-v)^2); cityblock sum |u-v|; cosine 1 - u.v/(|u||v|);
    chebyshev max |u-v|; canberra sum |u-v|/(|u|+|v|) with 0/0 terms = 0;
    braycurtis sum |u-v| / sum |u+v|.

    braycurtis with a zero denominator (u = -v) returns +inf, or NaN when u = v = 0,
    without a RuntimeWarning.  cosine is evaluated literally as 1 - u.v/(|u||v|), which
    cancels for near-identical rows: relative error 5.4e-2 on the pdist check's rows
    z(x0 + eps x1) at eps = 1e-7 (audit/notes/pdist.md, distance.pdist.cosine).
    refs/pdist.py:stable_metric is the cancellation-free form |u/|u| - v/|v||^2 / 2."""
    X = np.asarray(X, float)
    U, V = X[:, None, :], X[None, :, :]
    D = U - V
    if metric == "euclidean":
        return np.sqrt(np.sum(D**2, -1))
    if metric == "cityblock":
        return np.sum(np.abs(D), -1)
    if metric == "chebyshev":
        return np.max(np.abs(D), -1)
    if metric == "cosine":
        nrm = np.linalg.norm(X, axis=1)
        return 1.0 - (X @ X.T) / np.outer(nrm, nrm)
    if metric == "canberra":
        den = np.abs(U) + np.abs(V)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(den > 0, np.abs(D) / den, 0.0)
        return np.sum(q, -1)
    if metric == "braycurtis":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sum(np.abs(D), -1) / np.sum(np.abs(U + V), -1)
    raise ValueError(metric)


def zscored_distance_identities(x, y, ddof=1):
    """Helper identities for z-scored (ddof) data with Pearson r:
    cosine = 1 - r and euclidean = sqrt(2 (T - ddof)(1 - r)).  Returns
    (cosine_pred, euclid_pred, cosine_kernel, euclid_kernel)."""
    T = len(x)
    z = np.stack([(v - np.mean(v)) / np.std(v, ddof=ddof) for v in (x, y)])
    r = float(np.corrcoef(z)[0, 1])
    return (1 - r, np.sqrt(2 * (T - ddof) * (1 - r)),
            pdist_metric(z, "cosine")[0, 1], pdist_metric(z, "euclidean")[0, 1])


# ============================================================ cointegration
def _sm_detrend(y, order):
    """statsmodels coint_johansen's inner detrend (vecm.py:650-653): OLS residual on
    np.vander(np.linspace(-1, 1, n), order+1); order -1 = untouched."""
    if order == -1:
        return y
    V = np.vander(np.linspace(-1, 1, len(y)), order + 1)
    return y - V @ np.linalg.lstsq(V, y, rcond=None)[0]


def johansen(y, det_order=0, k_ar_diff=1):
    """Johansen (1991) Econometrica 59:1551 reduced-rank regression test.
    y (P, T).  VECM dy_t = Pi y_{t-1} + sum_{i=1..k} G_i dy_{t-i} + det + e_t.
    R0 / R1 = residuals of dy_t / y_{t-1} on the lagged differences Z; S_ij = R_i^T R_j / n;
    lambda = eig(S11^-1 S10 S00^-1 S01) sorted descending;
    trace(r) = -n sum_{i>r} log(1 - lambda_i); max_eig(r) = -n log(1 - lambda_{r+1}),
    n = T - k - 1 rows.  Reported at r = 0 as pyspi does (pyspi/statistics/misc.py:85-86, index [0]).
    Deterministic terms placed EXACTLY as statsmodels coint_johansen (vecm.py:665-684):
      det_order >= 0: the LEVELS are first detrended with a degree-det_order polynomial
        (det_order 0 = demeaned, 1 = linear trend removed by OLS on linspace(-1,1)), then
        dy, Z and the lagged levels are each demeaned (f = 0);
      det_order = -1: no detrending at all (f = -1).
    statsmodels pairs dy_t with the level y_{t-k} (vecm.py:681-682, Johansen 1988 form);
    y_{t-1} - y_{t-k} lies in span(Z), so both forms give identical eigenvalues; here y_{t-1}.
    R urca::ca.jo second opinion: ecdet='none' (unrestricted constant) ~ det_order=0;
    ecdet='const' (constant restricted to the cointegration space) has NO statsmodels
    equivalent; det_order=1 (pre-detrended levels, demeaned regressions) is not identical
    to ecdet='trend' (trend restricted to the cointegration space, estimated in-sample).
    ca.jo's K is the VAR order in levels = k_ar_diff + 1.
    Known divergence: for k_ar_diff = 0 statsmodels' lx = endog[:nobs - 0][1:] pairs dy_t with
    the CONTEMPORANEOUS level y_t (vecm.py:681-682), not y_{t-1}; this kernel keeps y_{t-1}
    (pyspi configs use k_ar_diff 1 and 10 only, pyspi/config.yaml:1305-1325).
    Statistics as vecm.py:715-718 with n = rkt.shape[0].
    Returns dict(eig, trace_stat (P,), max_eig_stat (P,), nobs)."""
    y = np.asarray(y, float).T
    T, P = y.shape
    f = 0 if det_order > -1 else -1
    y = _sm_detrend(y, det_order)
    dy = np.diff(y, axis=0)
    k = k_ar_diff
    rows = np.arange(k, T - 1)
    Z = np.column_stack([dy[rows - i] for i in range(1, k + 1)]) if k > 0 else np.zeros((rows.size, 0))
    d0, lv = dy[rows], y[rows]
    if f == 0:
        Z = Z - Z.mean(0) if Z.size else Z
        d0, lv = d0 - d0.mean(0), lv - lv.mean(0)

    def res(a):
        return a if Z.size == 0 else a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]

    R0, R1 = res(d0), res(lv)
    n = R0.shape[0]
    S00, S01, S11 = R0.T @ R0 / n, R0.T @ R1 / n, R1.T @ R1 / n
    lam = np.linalg.eigvals(np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01)))
    lam = np.sort(lam.real)[::-1]
    trace = np.array([-n * np.sum(np.log(1 - lam[r:])) for r in range(P)])
    return dict(eig=lam, trace_stat=trace, max_eig_stat=-n * np.log(1 - lam), nobs=n)


def _ols(yv, Xm):
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    e = yv - Xm @ beta
    return beta, e, int(np.linalg.matrix_rank(Xm))


def engle_granger_tstat(y0, y1, trend="c", maxlag=10, autolag="aic", return_lag=False):
    """Engle & Granger (1987) Econometrica 55:251 two-step test statistic, as statsmodels
    coint (stattools/_stattools.py:2680-2717): OLS of y0 on [y1, deterministic terms]
    (trend 'n' | 'c' | 'ct', add_trend appended); if R^2 >= 1 - 100 sqrt(eps) return -inf
    (:2690-2706); else ADF on the residuals with NO deterministic term (:2691-2697):
      d e_t = g e_{t-1} + sum_{i=1..L} c_i d e_{t-i} + u_t, t-statistic of g.
    autolag rule mirrored from adfuller (_stattools.py:387-436) and _autolag (:147-152): every
    L in 0..maxlag is fitted on the SAME sample (rows lost to maxlag lags), L* minimises
    aic = -2 llf + 2k, llf = -n/2 (log 2 pi + log(SSR/n) + 1), k = rank of the design
    (bic: log(n) k; regression/linear_model.py:1074, :2168-2172), ties -> smallest L (tuple
    min); then re-fitted with L* lags on its own longer sample (:420-436).  autolag=None
    uses L = maxlag.  maxlag > len//2 - 1 raises, as adfuller (:380).
    return_lag=True returns (t, L) with the lag L used in the final regression; the
    R^2 early exit then returns (-inf, None).

    SCOPE: L and t follow statsmodels only when the ADF lag design has full rank.  On a
    rank-deficient design (the audit's shared_envelope pairs among channels 0/1/2 other
    than the -inf pair 0/2: rank 7 of 11, SSR 1e-22..1e-27 for every L >= 6) rounding
    decides the IC minimum, so L can differ from adfuller's usedlag (kernel 9-10 against
    5-7) and t from coint's.  t also uses inv(Xa'Xa) where statsmodels OLS uses pinv
    (regression/linear_model.py:383-388), so on such a design t can be NaN (negative
    inv(Xa'Xa)[0,0], with a RuntimeWarning) or inv can raise LinAlgError.  Over 6192
    audit cases (trend n/c/ct x aic/bic, maxlag 10, raw and z-scored, T <= 1024; statsmodels
    0.15.0) 136 had a different L, 136 a different t (NaN counted as different) and 8
    raised, all on shared_envelope.  The coint check reads shared_envelope only
    qualitatively (refs/coint.py degen branch, pyspi values), so no check uses these."""
    y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
    T = y0.size
    det = {"n": [], "c": [np.ones(T)], "ct": [np.ones(T), np.arange(1.0, T + 1)]}[trend]
    Xm = np.column_stack([y1] + det)
    _, e, _ = _ols(y0, Xm)
    tss = np.sum((y0 - y0.mean()) ** 2) if trend != "n" else np.sum(y0**2)
    if 1 - np.sum(e**2) / tss >= 1 - 100 * np.sqrt(EPS):
        return (-np.inf, None) if return_lag else -np.inf
    if maxlag > T // 2 - 1:
        raise ValueError("maxlag too large")
    de = np.diff(e)

    def design(L):
        idx = np.arange(L, de.size)
        return de[idx], np.column_stack([e[idx]] + [de[idx - i] for i in range(1, L + 1)])

    if autolag:
        dep_full, Xfull = design(maxlag)
        best = None
        for L in range(maxlag + 1):
            _, u, k = _ols(dep_full, Xfull[:, :L + 1])
            nn = dep_full.size
            llf = -nn / 2 * (np.log(2 * np.pi) + np.log(u @ u / nn) + 1)
            ic = -2 * llf + (2 * k if autolag == "aic" else np.log(nn) * k)
            if best is None or (ic, L) < best:
                best = (ic, L)
        L = best[1]
    else:
        L = maxlag
    dep, Xa = design(L)
    beta, u, k = _ols(dep, Xa)
    s2 = u @ u / (dep.size - k)
    t = float(beta[0] / np.sqrt(s2 * np.linalg.inv(Xa.T @ Xa)[0, 0]))
    return (t, L) if return_lag else t


# ============================================================ 1-D linear models (MSE)
def _centre(x, y):
    x, y = np.ravel(np.asarray(x, float)), np.ravel(np.asarray(y, float))
    return x, y, x - x.mean(), y - y.mean()


def _mse_from_w(x, y, w):
    """sklearn fit_intercept=True: intercept = ybar - w xbar; MSE on training data
    (pyspi LinearModel, pyspi/statistics/misc.py:118-126, sklearn.metrics.mean_squared_error)."""
    return float(np.mean((y - (y.mean() + w * (x - x.mean()))) ** 2))


def ols_mse(x, y):
    """Least-squares y ~ a + w x; w = sum(xc yc)/sum(xc^2); training MSE."""
    x, y, xc, yc = _centre(x, y)
    return _mse_from_w(x, y, xc @ yc / (xc @ xc))


def ridge_mse(x, y, alpha=1.0):
    """Ridge (Hoerl & Kennard 1970), sklearn objective ||y - Xw||^2 + alpha ||w||^2 with
    a free intercept (sklearn/linear_model/_ridge.py:1027): w = sum(xc yc)/(sum(xc^2) + alpha)."""
    x, y, xc, yc = _centre(x, y)
    return _mse_from_w(x, y, xc @ yc / (xc @ xc + alpha))


def _soft(z, t):
    return np.sign(z) * max(abs(z) - t, 0.0)


def elasticnet_1d_mse(x, y, alpha=1.0, l1_ratio=0.5):
    """Elastic net (Zou & Hastie 2005), sklearn objective (_coordinate_descent.py:501-503)
    (1/2n)||y - Xw||^2 + alpha rho |w| + 0.5 alpha (1 - rho) w^2, free intercept.
    One regressor -> exact minimiser w = S(sum(xc yc)/n, alpha rho) / (sum(xc^2)/n + alpha(1-rho)),
    S = soft threshold.

    This is the exact minimiser, NOT always what sklearn returns.  sklearn's coordinate
    descent first checks the duality gap at w = 0 and returns w = 0 (n_iter_ 0) when
    gap <= tol ||y_c||^2, tol = 1e-4 (sklearn/linear_model/_cd_fast.pyx:391-400).  With
    s = sum(xc yc)/n and a = alpha rho that gap is 0.5 ||y_c||^2 (1 - a/|s|)^2 for |s| > a,
    so sklearn returns w = 0 in the band a < |s| <= a / (1 - sqrt(2 tol)), where the exact
    minimiser is not 0.  On ddof=1 z-scored data with the defaults (alpha 1, l1_ratio 0.5)
    the band is 0.5T/(T-1) < |r| <= 0.5T/((T-1)(1 - sqrt(2e-4))); there this function's MSE
    is up to about 4.8e-3 below sklearn's (T-1)/T (audit/notes/lmfit.md).  Outside the band
    the first update lands on the minimiser and the next gap check stops there.
    refs/lmfit.py:enet_sklearn_tol models the exit (exact on the lmfit check runs; a
    synthetic r sweep put sklearn's switch at the modelled bound, with 1 of 1604 boundary
    points misclassified; audit/notes/lmfit.md)."""
    x, y, xc, yc = _centre(x, y)
    n = x.size
    return _mse_from_w(x, y, _soft(xc @ yc / n, alpha * l1_ratio) / (xc @ xc / n + alpha * (1 - l1_ratio)))


def lasso_1d_mse(x, y, alpha=1.0):
    """Lasso (Tibshirani 1996), sklearn objective (1/2n)||y - Xw||^2 + alpha |w|
    (_coordinate_descent.py:1334): w = S(sum(xc yc)/n, alpha)/(sum(xc^2)/n).
    On ddof=1 z-scored data sum(xc yc)/n = r(T-1)/T, so w = 0 and MSE = (T-1)/T
    whenever |r|(T-1)/T <= alpha (always, for the default alpha = 1).
    Matches sklearn only where sklearn's initial duality-gap exit and the exact minimiser
    coincide, i.e. outside the band alpha < |s| <= alpha / (1 - sqrt(2e-4)) (see
    elasticnet_1d_mse).  On ddof=1 z-scored data at alpha = 1, |s| < alpha for every pair,
    so both give w = 0."""
    return elasticnet_1d_mse(x, y, alpha, 1.0)


def bayesian_ridge_mse(x, y, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                       lambda_1=1e-6, lambda_2=1e-6, return_params=False):
    """Bayesian ridge by evidence maximisation (MacKay 1992 Neural Comput. 4:415;
    Tipping 2001 JMLR 1:211), mirroring sklearn BayesianRidge.fit (_bayes.py:270-349):
    centre x, y; init alpha = 1/(var(y) + eps), lambda = 1; loop: m = (X^T X + lambda/alpha I)^-1 X^T y,
    gamma = sum alpha e_i/(lambda + alpha e_i) (e = eig X^T X),
    lambda <- (gamma + 2 l1)/(|m|^2 + 2 l2), alpha <- (n - gamma + 2 a1)/(SSE + 2 a2),
    stop when iter > 0 and sum|m_old - m| < tol (checked AFTER the update, :335);
    the returned coefficient is recomputed with the final alpha, lambda (:347).
    x may be (n,) or (n, p) with n > p."""
    X = np.asarray(x, float).reshape(len(y), -1)
    y = np.ravel(np.asarray(y, float))
    Xc, yc = X - X.mean(0), y - y.mean()
    n, p = Xc.shape
    alpha, lam = 1.0 / (y.var() + EPS), 1.0
    XtX, Xty = Xc.T @ Xc, Xc.T @ yc
    ev = np.linalg.eigvalsh(XtX)

    def coef(a, l):
        m = np.linalg.solve(XtX + (l / a) * np.eye(p), Xty)
        return m, np.sum((yc - Xc @ m) ** 2)

    m_old = None
    for it in range(max_iter):
        m, sse = coef(alpha, lam)
        gamma = np.sum(alpha * ev / (lam + alpha * ev))
        lam = (gamma + 2 * lambda_1) / (m @ m + 2 * lambda_2)
        alpha = (n - gamma + 2 * alpha_1) / (sse + 2 * alpha_2)
        if it != 0 and np.sum(np.abs(m_old - m)) < tol:
            break
        m_old = m.copy()
    m, _ = coef(alpha, lam)
    mse = float(np.mean((yc - Xc @ m) ** 2))
    return (mse, dict(coef=m, alpha=alpha, lambda_=lam, n_iter=it + 1)) if return_params else mse


# ============================================================ envelope correlation
def _next_5smooth(n):
    """Smallest 2^a 3^b 5^c >= n (mne.filter.next_fast_len, mne/filter.py:42-48)."""
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def envelope_correlation(X, orth=False, log=False, absolute=False, mne_order=False):
    """Amplitude-envelope correlation, Hipp et al. (2012) Nat. Neurosci. 15:884.
    X (P, T) real -> (P, P).  Mirrors mne_connectivity.envelope_correlation:
    analytic a = hilbert(x, N=next_fast_len(T))[:T] (envelope.py:139-140), 5-smooth length;
    E = |a|, or log(|a|^2) when log (:146-151).
    orth=False: C_ij = Pearson(E_i, E_j); `absolute` is IGNORED (only used under orth, :184-188).
    orth=True ('pairwise'): for row i, column j: O_ij = |Im(a_i conj(a_j)/|a_j|)| (the part of
    a_i orthogonal to a_j, ABSOLUTE value taken, :168-169), log(O^2) when log (:173-175),
    C_ij = Pearson(O_ij, E_j) (:181-183; zero-variance series give 0 by std := 1, :158/:178;
    the diagonal is forced to 0 via O_ii := 1, :172); then |C| if absolute and C <- (C + C^T)/2.
    Evaluation order: by default Im(a_i conj(a_j)) = Im a_i Re a_j - Re a_i Im a_j is formed
    first and then divided by |a_j|, which is EXACTLY 0 when a_i == a_j bitwise (zero-lag copy ->
    zero variance -> C = 0, the definitional answer).  mne_order=True instead evaluates
    a_i * (conj(a_j)/|a_j|) as mne does (:147-148, :168), whose ~eps*|a| rounding residue is
    scale-free correlated with the envelope (gives C ~ 0.3 for a bit-identical copy).  Under
    log, a row containing log(0) is treated as zero variance (C = 0) instead of NaN."""
    X = np.asarray(X, float)
    P, T = X.shape
    a = hilbert(X, N=_next_5smooth(T), axis=-1)[:, :T]
    mag = np.abs(a)
    E = np.log(mag**2) if log else mag

    def zc(v):
        v = v - v.mean(-1, keepdims=True)
        s = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(s == 0, 1.0, s)

    Ez = zc(E)
    if not orth:
        return Ez @ Ez.T
    C = np.empty((P, P))
    for i in range(P):
        if mne_order:
            O = np.abs((a[i][None, :] * (a.conj() / mag)).imag)
        else:
            O = np.abs(a[i].imag[None, :] * a.real - a[i].real[None, :] * a.imag) / mag
        O[i] = 1.0
        if log:
            with np.errstate(divide="ignore"):
                O = np.log(O**2)
            O[~np.isfinite(O).all(-1)] = 0.0   # log(0): orthogonal part identically 0 -> C = 0
        C[i] = np.sum(zc(O) * Ez, axis=-1)
    if absolute:
        C = np.abs(C)
    return (C + C.T) / 2


# ============================================================ interdependence score
def ids(X, terms=6, bandwidth=0.5, pnorm="max"):
    """Interdependence score, Radhakrishnan et al. (2025) PNAS: features
    phi_k(y) = exp(-B y^2) y^k / sqrt(k!), k = 0..terms-1, per process; C = |corr| between
    every pair of transformed columns; IDS_ij = max_{k,l} C[(k,i),(l,j)] (pnorm='max'),
    mean_{k,l} (pnorm=1) or sqrt(mean C^2) (pnorm=2).  Non-finite correlations -> 0.
    Mirrors pyspi/lib/ids/numpy_dependence.py:16-23 (transform), :77-93 (corrcoef,
    nan_to_num, reductions).  X (P, T) as given (pyspi feeds ddof=1 z-scored data,
    pyspi/statistics/misc.py:197-198) -> (P, P)."""
    X = np.asarray(X, float)
    P, T = X.shape
    F = np.concatenate([np.exp(-bandwidth * X**2) * X**k / math.sqrt(math.factorial(k))
                        for k in range(terms)], axis=0)          # row k*P + i
    Fc = F - F.mean(1, keepdims=True)
    nrm = np.linalg.norm(Fc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (Fc @ Fc.T) / np.outer(nrm, nrm)
    C = np.abs(np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)).reshape(terms, P, terms, P)
    if pnorm == "max":
        return C.max(axis=(0, 2))
    if pnorm == 2:
        return np.sqrt((C**2).mean(axis=(0, 2)))
    if pnorm == 1:
        return C.mean(axis=(0, 2))
    raise ValueError(pnorm)


# ============================================================ Gaussian processes
# Every gp_* function here uses the ADDITIVE kernel C + White + {RBF | DotProduct} that
# pyspi's GPModel builds (pyspi/statistics/misc.py:136-137).  That is not sklearn's default
# C() * RBF() (sklearn/gaussian_process/_gpr.py:249, used by cdt's ANM) and not C * RBF + White;
# for the product kernel use _kernels_distance.gp_posterior_mean / gp_fit_hyperparameters.
def _gp_kernel(x1, x2, kp, train):
    """Constant + White + {RBF | DotProduct} (a sum, not C * RBF), sklearn parameterisation
    (kernels.py:1187 ConstantKernel c, :1325 WhiteKernel w, :1562 RBF exp(-d^2/(2 l^2)),
    :2197 DotProduct sigma_0^2 + x.x').  White contributes w I only on the training
    covariance (Y=None), and ZERO in a cross-covariance even at identical points (:1405-1419).
    Returns K and dK/dtheta stacked on the last axis, theta = log[c, w, l or sigma_0]
    (sklearn's theta order for ConstantKernel()+WhiteKernel()+K(), pyspi/statistics/misc.py:136-137)."""
    c, w = kp["constant"], kp["noise"]
    if kp["kernel"] == "RBF":
        l = kp["length_scale"]
        d2 = (x1[:, None] - x2[None, :]) ** 2
        Kb = np.exp(-0.5 * d2 / l**2)
        dKb = Kb * d2 / l**2
    elif kp["kernel"] == "DotProduct":
        s0 = kp["sigma_0"]
        Kb = np.outer(x1, x2) + s0**2
        dKb = np.full_like(Kb, 2 * s0**2)
    else:
        raise ValueError(kp["kernel"])
    W = w * np.eye(x1.size) if train else np.zeros_like(Kb)
    K = c + W + Kb
    return K, np.stack([np.full_like(Kb, c), W, dKb], axis=-1)


def gp_posterior_mean(x, y, kernel_params, x_star=None, alpha=1e-10):
    """GP posterior mean (Rasmussen & Williams 2006, eq. 2.25 / Alg. 2.1):
    m(x*) = K(x*, X) (K(X,X) + alpha I)^-1 y, zero prior mean, y NOT normalised
    (sklearn GaussianProcessRegressor defaults alpha=1e-10, normalize_y=False,
    _gpr.py:215-218; jitter :350; K_trans = kernel(x*, X_train) :446 excludes the White term).
    x, y (n,); x_star defaults to x (pyspi predicts on the training inputs, pyspi/statistics/misc.py:146)."""
    x, y = np.ravel(np.asarray(x, float)), np.ravel(np.asarray(y, float))
    xs = x if x_star is None else np.ravel(np.asarray(x_star, float))
    K, _ = _gp_kernel(x, x, kernel_params, True)
    K[np.diag_indices_from(K)] += alpha
    Ks, _ = _gp_kernel(xs, x, kernel_params, False)
    L = np.linalg.cholesky(K)
    return Ks @ sla.cho_solve((L, True), y)


def gp_log_marginal_likelihood(x, y, kernel_params, alpha=1e-10, eval_gradient=True):
    """log p(y|X,theta) = -1/2 y^T K^-1 y - 1/2 log|K| - n/2 log 2 pi  (R&W eq. 5.8) and
    its gradient 1/2 tr((a a^T - K^-1) dK/dtheta_j), a = K^-1 y (R&W eq. 5.9), with respect
    to theta = log hyperparameters [c, w, l | sigma_0] (sklearn's space, _gpr.py:589-650).
    The value needs only a = cho_solve(L, y); the explicit K^-1 = cho_solve(L, I) is formed
    only for the gradient, as sklearn does (_gpr.py:630-633).  eval_gradient=False returns
    the value alone (float) and skips K^-1; the default returns (value, gradient).
    A failed Cholesky raises LinAlgError here; gp_fit maps it to -inf as sklearn does."""
    x, y = np.ravel(np.asarray(x, float)), np.ravel(np.asarray(y, float))
    K, dK = _gp_kernel(x, x, kernel_params, True)
    K[np.diag_indices_from(K)] += alpha
    L = np.linalg.cholesky(K)
    a = sla.cho_solve((L, True), y)
    lml = -0.5 * y @ a - np.sum(np.log(np.diag(L))) - 0.5 * x.size * np.log(2 * np.pi)
    if not eval_gradient:
        return float(lml)
    inner = np.outer(a, a) - sla.cho_solve((L, True), np.eye(x.size))
    return float(lml), 0.5 * np.einsum("ij,jik->k", inner, dK)


def _gp_params(kernel, theta):
    e = np.exp(theta)
    kp = dict(kernel=kernel, constant=e[0], noise=e[1])
    kp["length_scale" if kernel == "RBF" else "sigma_0"] = e[2]
    return kp


def _gp_lml_guarded(x, y, kp):
    """(lml, grad), or (-inf, 0) when the Cholesky of K + alpha I fails, as sklearn's
    log_marginal_likelihood (sklearn/gaussian_process/_gpr.py:590-593)."""
    try:
        return gp_log_marginal_likelihood(x, y, kp)
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros(3)


def gp_fit(x, y, kernel="RBF", bounds=(1e-5, 1e5)):
    """Type-II ML hyperparameters by L-BFGS-B on -lml from theta = log(1) with sklearn's
    default bounds and no restarts (_gpr.py:217, :659; kernels.py default values 1.0,
    bounds (1e-5, 1e5)).  Additive kernel C + White + K (see the section note), not C * RBF.
    A failed Cholesky gives lml = -inf and a zero gradient (objective +inf) as in sklearn
    (_gpr.py:590-593) instead of raising; if the optimiser stops at such a point the
    returned lml is -inf.  Returns (kernel_params, lml, grad).  Local optimum only."""
    f = lambda t: tuple(-v for v in _gp_lml_guarded(x, y, _gp_params(kernel, t)))
    lb = [(np.log(bounds[0]), np.log(bounds[1]))] * 3
    r = optimize.minimize(f, np.zeros(3), jac=True, method="L-BFGS-B", bounds=lb)
    kp = _gp_params(kernel, r.x)
    lml, g = _gp_lml_guarded(x, y, kp)
    return kp, lml, g


# ============================================================ selftest
def selftest():
    sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
    import make_data as md
    res = []

    def check(label, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")

    z1 = lambda v: (v - v.mean()) / v.std(ddof=1)
    off4 = ~np.eye(4, dtype=bool)

    # (1) Ledoit-Wolf / OAS on bivariate_gaussian T=256
    X = md.load("bivariate_gaussian", 256)["x"]
    S = empirical_cov(X)
    mu = np.trace(S) / 4
    for name, (C, s) in (("ledoit_wolf", ledoit_wolf(X)), ("oas", oas(X)), ("oas[sklearn]", oas(X, False))):
        between = (np.all(np.abs(C[off4]) <= np.abs(S[off4])) and np.all(np.sign(C[off4]) == np.sign(S[off4]))
                   and np.all((np.diag(C) - np.diag(S)) * (np.diag(C) - mu) <= 1e-15))
        check(f"1 {name} shrinkage in [0,1], between S and mu*I", 0 <= s <= 1 and between, f"shrinkage={s:.5f}")
    x1 = X[:1]
    (C1p, r1p), (C1s, r1s) = oas(x1), oas(x1, False)
    check("1 oas P=1: sklearn rho=0, paper rho=1, cov = var either way",
          r1s == 0.0 and r1p == 1.0 and abs(C1s[0, 0] - x1.var()) < 1e-15 and abs(C1p[0, 0] - x1.var()) < 1e-15,
          f"rho sklearn={r1s}, paper={r1p}")
    Xc = X - X.mean(1, keepdims=True)
    bb = (np.sum(np.sum(Xc**2, 0) ** 2) / 256 - np.sum(S**2)) / (4 * 256)   # expanded bbar^2
    d2 = np.sum((S - mu * np.eye(4)) ** 2) / 4
    check("1 ledoit_wolf literal = expanded bbar^2", abs(ledoit_wolf(X)[1] - min(bb, d2) / d2) < 1e-12,
          f"{min(bb, d2) / d2:.12f}")
    check("1 precision(S) @ S = I, shrunk_cov keeps trace", np.abs(precision(S) @ S - np.eye(4)).max() < 1e-10
          and abs(np.trace(shrunk_cov(X)[0]) - np.trace(S)) < 1e-12, "tol 1e-10 / 1e-12")

    # (2) glasso KKT on sparse_precision
    sp = md.load("sparse_precision", 4096)
    r0 = glasso_kkt_residual(sp["cov_true"], sp["precision_true"], 0.0)
    r1 = glasso_kkt_residual(sp["cov_true"], sp["precision_true"], 0.1)
    check("2 KKT(true Sigma, true Theta, alpha=0) ~ 0; alpha=0.1 -> 0.1", r0 < 1e-12 and abs(r1 - 0.1) < 1e-12,
          f"{r0:.1e}, {r1:.6f}")
    re = glasso_kkt_residual(empirical_cov(sp["x"]), sp["precision_true"], 1e-3)
    tol = 5 * np.sqrt(2 / 4096) * sp["cov_true"].diagonal().max()   # 5 x sampling sd of S_ij
    check("2 KKT(empirical S T=4096, true Theta, alpha=1e-3) at sampling level", re < tol, f"{re:.4f} (tol {tol:.4f})")
    S2 = np.array([[1.0, 0.5], [0.5, 2.0]])
    W2 = np.array([[1.0, 0.3], [0.3, 2.0]])   # p=2 glasso solution: W12 = soft(S12, alpha)
    check("2 KKT(2x2 closed-form glasso, alpha=0.2)", glasso_kkt_residual(S2, np.linalg.inv(W2), 0.2) < 1e-12)

    # (3) cross-correlation on delayed_noise T=1024 (x_k = x0 delayed by d_k)
    dn = md.load("delayed_noise", 1024)
    for k in (1, 2, 3):
        d = int(dn["delay"][k])
        lags, r = xcorr_full(dn["x"][k], dn["x"][0])
        lags2, r2 = xcorr_full(dn["x"][0], dn["x"][k])
        check(f"3 xcorr_full(x{k}, x0) peak at tau=+{d} (y=x0 leads)", lags[r.argmax()] == d and lags2[r2.argmax()] == -d,
              f"argmax {lags[r.argmax()]}, swapped {lags2[r2.argmax()]}, r_peak={r.max():.3f}")
    a, b = z1(dn["x"][0]), z1(dn["x"][1])
    rep, T = pyspi_xcorr_replica(a, b), a.size
    check("3 pyspi window centre sits at lag +1 (not 0)", rep["centre_lag"] == 1 and rep["lags_window"][0] == -T // 4 + 1,
          f"centre index {rep['centre_index']} -> lag {rep['centre_lag']}; window [{rep['lags_window'][0]}, "
          f"{rep['lags_window'][-1]}]")
    r_0 = xcorr_full(a, b)[1][T - 1]
    check("3 pyspi lag-0 value = r T/(T-1) on ddof=1 z-scored data",
          abs(rep["r_lag0_replica"] - r_0 * T / (T - 1)) < 1e-12, f"{rep['r_lag0_replica']:.6f} vs r {r_0:.6f}")
    print(f"      info replica(x0, x1): sig thr {rep['sig_thr']:.4f} vs 1.96/sqrt(T) {rep['intended_thr']:.4f}; "
          f"fzr={rep['fzr']}, fzf={rep['fzf']}, kept lags [{rep['kept_lags'][0]}, {rep['kept_lags'][-1]}] vs "
          f"intended [{rep['intended_lags'][0]}, {rep['intended_lags'][-1]}]; mean_sig {rep['mean_sig']:.4f} "
          f"vs intended {rep['intended_mean']:.4f}")

    # (4) rank correlations
    rng, worst = np.random.default_rng(4), 0.0
    for n in (5, 12, 40):
        for _ in range(20):
            u, v = rng.integers(0, 4, n).astype(float), rng.integers(0, 5, n).astype(float)
            if np.ptp(u) > 0 and np.ptp(v) > 0:
                worst = max(worst, abs(kendall_tau_b(u, v) - _kendall_tau_b_knight(u, v)))
        u, v = rng.normal(size=n), rng.normal(size=n)
        worst = max(worst, abs(kendall_tau_b(u, v) - _kendall_tau_b_knight(u, v)))
    check("4 kendall_tau_b O(n^2) = Knight O(n log n), with ties", worst < 1e-12, f"max diff {worst:.1e}")
    # x=[1,2,2,3], y=[1,3,2,2]: nc=3, nd=1, n0=6, one x-tie, one y-tie -> (3-1)/sqrt(5*5)
    check("4 kendall_tau_b hand example with ties = 0.4",
          abs(kendall_tau_b([1, 2, 2, 3.0], [1, 3, 2, 2.0]) - 0.4) < 1e-15)
    check("4 spearman monotone = 1; average ranks for ties",
          abs(spearman(np.arange(9.0), np.exp(np.arange(9.0))) - 1) < 1e-15
          and np.allclose(_rank_average([3, 1, 3, 2]), [3.5, 1, 3.5, 2]))

    # (5) pairwise distances
    bg = md.load("bivariate_gaussian", 1024)["x"]
    cp, ep, ck, ek = zscored_distance_identities(bg[2], bg[3])
    check("5 cosine = 1 - r and euclidean = sqrt(2(T-1)(1-r)) on z-scored", abs(cp - ck) < 1e-10
          and abs(ep - ek) < 1e-10, f"cos {ck:.6f}, euc {ek:.6f} (tol 1e-10)")
    hv = dict(euclidean=np.sqrt(5), cityblock=3, chebyshev=2, canberra=4 / 3, braycurtis=3 / 11,
              cosine=1 - 11 / np.sqrt(14 * 13))
    err = max(abs(pdist_metric(np.array([[1, 2, 3.0], [2, 0, 3.0]]), m)[0, 1] - v) for m, v in hv.items())
    check("5 six metrics on hand example [1,2,3] vs [2,0,3]", err < 1e-15, f"max err {err:.1e}")

    # (6) cointegration on cointegrated T=1024: pair (0,1) cointegrated, (2,3) independent walks
    ci = md.load("cointegrated", 1024)["x"]
    for k in (1, 10):
        jc, ji = johansen(ci[[0, 1]], 0, k), johansen(ci[[2, 3]], 0, k)
        check(f"6 johansen r=0 trace (det_order=0, k_ar_diff={k}) coint >> indep",
              jc["trace_stat"][0] > 50 and jc["trace_stat"][0] > 5 * ji["trace_stat"][0],
              f"coint trace {jc['trace_stat'][0]:.2f} / max_eig {jc['max_eig_stat'][0]:.2f}; indep trace "
              f"{ji['trace_stat'][0]:.2f} / max_eig {ji['max_eig_stat'][0]:.2f}")
    tc, ti = engle_granger_tstat(ci[0], ci[1]), engle_granger_tstat(ci[2], ci[3])
    check("6 engle_granger t (trend c, aic, maxlag 10) strongly negative for coint pair",
          tc < -10 and ti > tc / 3, f"coint {tc:.3f}, indep {ti:.3f}")
    tl, lag = engle_granger_tstat(ci[0], ci[1], return_lag=True)
    check("6 engle_granger return_lag=True: same t, lag in 0..maxlag; R^2 exit -> (-inf, None)",
          tl == tc and 0 <= lag <= 10 and engle_granger_tstat(ci[0], 2 * ci[0] + 1, return_lag=True) == (-np.inf, None),
          f"t {tl:.3f}, lag {lag}")

    # (7) 1-D linear models on ddof=1 z-scored data
    for T in (64, 1024):
        g = md.load("bivariate_gaussian", T)["x"]
        u, v = z1(g[2]), z1(g[3])
        q = abs(np.corrcoef(u, v)[0, 1]) * (T - 1) / T
        check(f"7 lasso_1d_mse T={T} = (T-1)/T since |r|(T-1)/T={q:.3f} <= alpha=1",
              abs(lasso_1d_mse(u, v) - (T - 1) / T) < 1e-12, f"{lasso_1d_mse(u, v):.6f}")
    u, v = z1(bg[2]), z1(bg[3])
    ols = ols_mse(u, v)
    check("7 OLS = (T-1)/T (1-r^2); ridge(alpha->0) = OLS; EN(l1_ratio=1) = lasso",
          abs(ols - 1023 / 1024 * (1 - np.corrcoef(u, v)[0, 1] ** 2)) < 1e-12 and abs(ridge_mse(u, v, 1e-12) - ols) < 1e-12
          and abs(elasticnet_1d_mse(u, v, 0.3, 1.0) - lasso_1d_mse(u, v, 0.3)) < 1e-15,
          f"ols {ols:.6f}, ridge {ridge_mse(u, v):.6f}, EN {elasticnet_1d_mse(u, v):.6f}")
    mse, prm = bayesian_ridge_mse(u, v, return_params=True)
    xc, yc = u - u.mean(), v - v.mean()
    fixed = abs(prm["coef"][0] - xc @ yc / (xc @ xc + prm["lambda_"] / prm["alpha"]))
    check("7 bayesian_ridge: OLS <= MSE <= var(y); coef = ridge(lambda/alpha); converged",
          ols <= mse <= np.var(v) and fixed < 1e-12 and prm["n_iter"] < 300,
          f"mse {mse:.6f}, n_iter {prm['n_iter']}, lambda/alpha {prm['lambda_'] / prm['alpha']:.4g}")

    # (8) envelope correlation on shared_envelope T=4096 (x2 bit-identical to x0)
    se = md.load("shared_envelope", 4096)["x"]
    C, Co = envelope_correlation(se), envelope_correlation(se, orth=True)
    Cm, Col = envelope_correlation(se, orth=True, mne_order=True), envelope_correlation(se, orth=True, log=True)
    check("8 shared pair (0,1) high, independent (0,3) ~ 0", C[0, 1] > 0.8 and abs(C[0, 3]) < 0.1,
          f"C01={C[0, 1]:.3f}, C02={C[0, 2]:.3f}, C03={C[0, 3]:.3f}")
    check("8 orthogonalised zero-lag copy (0,2) = 0 (also under log)", abs(Co[0, 2]) < 1e-12 and abs(Col[0, 2]) < 1e-12,
          f"Co02={Co[0, 2]:.1e} (mne evaluation order: {Cm[0, 2]:.3f}); Co01={Co[0, 1]:.3f} (constant pi/3 "
          f"lag survives orth); Co03={Co[0, 3]:.3f}")
    check("8 outputs symmetric", np.allclose(C, C.T) and np.allclose(Co, Co.T))

    # (9) IDS on nonlinear_dependence T=1024 (x1 = x0^2 + noise)
    Z = np.stack([z1(v) for v in md.load("nonlinear_dependence", 1024)["x"]])
    I = ids(Z)
    Ii = ids(np.stack([z1(v) for v in md.load("iid_gaussian", 1024)["x"]]))
    check("9 ids(x, x) = 1", abs(ids(np.stack([Z[0], Z[0]]))[0, 1] - 1) < 1e-12)
    check("9 ids(x0, x0^2) clearly > 0 and > iid pair", I[0, 1] > 0.5 and I[0, 1] > 3 * Ii[0, 1],
          f"ids {I[0, 1]:.3f} (Pearson {np.corrcoef(Z[0], Z[1])[0, 1]:.3f}), circle {I[2, 3]:.3f}, iid {Ii[0, 1]:.3f}")

    # (10) GP on anm T=64: eq. 5.9 gradient vs finite differences; projected gradient ~0 at the fit
    an = md.load("anm", 64)["x"]
    gx, gy = z1(an[0]), z1(an[1])
    for kern in ("RBF", "DotProduct"):
        th = np.log([1.3, 0.4, 0.7])
        gr = gp_log_marginal_likelihood(gx, gy, _gp_params(kern, th))[1]
        fd = np.array([(gp_log_marginal_likelihood(gx, gy, _gp_params(kern, th + h))[0]
                        - gp_log_marginal_likelihood(gx, gy, _gp_params(kern, th - h))[0]) / 2e-6 for h in np.eye(3) * 1e-6])
        kp, lml, g = gp_fit(gx, gy, kern)
        tt = np.log([kp["constant"], kp["noise"], kp.get("length_scale", kp.get("sigma_0"))])
        g = np.where((tt <= np.log(1e-5) + 1e-8) & (g < 0) | (tt >= np.log(1e5) - 1e-8) & (g > 0), 0.0, g)
        m = gp_posterior_mean(gx, gy, kp)
        check(f"10 GP {kern}: gradient vs FD; projected gradient at fit", np.abs(gr - fd).max() < 1e-5
              and np.abs(g).max() < 1e-2 and np.isfinite(m).all(),
              f"FD diff {np.abs(gr - fd).max():.1e}; lml {lml:.3f}, theta {np.round(np.exp(tt), 5)}, "
              f"|proj grad| {np.abs(g).max():.1e}, mse {np.mean((m - gy) ** 2):.4f}")
    kp_t = _gp_params("RBF", np.log([1.3, 0.4, 0.7]))
    xb = np.linspace(-3, 3, 400) * 50
    kp_b = _gp_params("DotProduct", np.log([1e5, 1e-5, 1e5]))       # K + alpha I not numerically PD
    try:
        gp_log_marginal_likelihood(xb, np.sin(xb), kp_b)
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    check("10 GP eval_gradient=False = value; failed Cholesky raises in lml, -inf via the gp_fit guard",
          gp_log_marginal_likelihood(gx, gy, kp_t, eval_gradient=False) == gp_log_marginal_likelihood(gx, gy, kp_t)[0]
          and raised and _gp_lml_guarded(xb, np.sin(xb), kp_b)[0] == -np.inf, f"raised {raised}")
    n_fail = res.count(False)
    print(f"{len(res) - n_fail}/{len(res)} PASS")
    return n_fail == 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
