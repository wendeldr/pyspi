"""Reference kernels: distance-similarity and causal-inference SPIs (pyspi calculation audit).

Independent numpy/scipy implementations written from the cited papers. Library conventions are
mirrored only where a docstring says so, with the library file:line (paths relative to
``.venv/lib/python3.12/site-packages/``). This module never imports hyppo, tslearn, cdt or pyEDM.
dcor, dtaidistance and POT are used only as labelled second opinions inside ``selftest``.
Run ``python _kernels_distance.py --selftest``. Equation, definition and algorithm numbers come from memory,
except Kravtsova et al. (from audit/notes/gwtau.md), and were not re-checked against the PDFs.

Shapes: 1-D series are ``(n,)``. Multivariate samples are ``(n, p)``. Datasets are ``(n_series, T)``.

There is no MGC kernel in this module. The hhg_mgc and lagged_dcorr_mgc family references carry their own
(refs/hhg_mgc.py:local_corr / mgc_from_dist; refs/lagged_dcorr_mgc.py:mgc_local_corr / mgc_stat).
"""
import sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import digamma


def _as2d(x):
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _pdist(x):
    """Exact Euclidean distance matrix, (n, n), computed by direct differencing."""
    x = _as2d(x)
    return np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))


def _sk_sqdist(x):
    """Squared distances with sklearn's float64 arithmetic: -2 X X^T + |x_i|^2 + |x_j|^2, clipped at 0,
    zero diagonal (sklearn/metrics/pairwise.py:409-421). Used only by the hyppo/cdt replicas."""
    x = _as2d(x)
    xx = np.einsum("ij,ij->i", x, x)[:, None]
    d = -2 * (x @ x.T); d += xx; d += xx.T
    np.maximum(d, 0, out=d); np.fill_diagonal(d, 0)
    return d


# ----------------------------------------------------------------- distance correlation
def _double_centre(a):
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def _u_centre(a):
    n = a.shape[0]
    u = a - a.sum(0, keepdims=True) / (n - 2) - a.sum(1, keepdims=True) / (n - 2) + a.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(u, 0.0)
    return u


def dcorr_from_dist(dx, dy, biased=False):
    """Squared distance correlation from (n, n) distance matrices. Returns (dcor2, dcov2_xy, dvar2_x, dvar2_y).
    biased=True: V-statistic (Szekely, Rizzo & Bakirov 2007, Ann. Stat. 35, Sec. 2): A = double-centred a_ij,
        dCov_n^2 = (1/n^2) sum A_ij B_ij, dCor_n^2 = dCov^2(x,y) / sqrt(dVar^2 x dVar^2 y).
    biased=False: bias-corrected dCor^2 (Szekely & Rizzo 2014, Ann. Stat. 42, Sec. 3, U-centring): A~_ij = a_ij
        - a_i./(n-2) - a_.j/(n-2) + a_../((n-1)(n-2)), A~_ii = 0, (A~.B~) = sum A~B~ / (n(n-3)); it can be negative.
    dcor2 is 0 when a variance is <= 0 and is formed from the unscaled sums, as hyppo's _dcorr (only_dcov=False,
    dcorr.py:402, 432-445). biased=False, n = 3: A~ is identically 0 (A~_12 = a12 - (a12+a13) - (a12+a23)
    + (a12+a13+a23)), so dcor2 = 0 and the covariances are NaN; hyppo returns a rounding-noise ratio there
    (scratchpad probe kernels/probe_edges_fix.py, which audit/checks/archive_probes.py packs into
    audit/checks/probes/scratch_probes_2026-10-15.tar.xz under the same relative path). biased=False with n < 3 raises
    ValueError.
    """
    n = dx.shape[0]
    if biased:
        A, B, s = _double_centre(dx), _double_centre(dy), 1.0 / n ** 2
    else:
        if n < 3:   # U-centring divides by n - 2
            raise ValueError(f"bias-corrected dCor needs n >= 3 samples, got {n}")
        A, B, s = _u_centre(dx), _u_centre(dy), (1.0 / (n * (n - 3)) if n > 3 else np.nan)
    sxy, sxx, syy = np.sum(A * B), np.sum(A * A), np.sum(B * B)
    d2 = 0.0 if (sxx <= 0 or syy <= 0 or (not biased and n == 3)) else sxy / np.sqrt(sxx * syy)
    return d2, s * sxy, s * sxx, s * syy


def hyppo_dcorr_from_dist(dx, dy, biased=False):
    """What hyppo's Dcorr.statistic returns for precomputed distances (mirrored, not imported).
    hyppo/independence/dcorr.py:417-447 (_dcorr): 0 if varx<=0, vary<=0 or (biased and cov<=0), else cov/sqrt(varx*vary),
    square-rooted when biased. So hyppo returns dCor (NOT squared) for bias=True and the signed bias-corrected dCor^2
    for bias=False. Centring: dcorr.py:262-281. The non-fast path is always used (is_fast is set only in .test(), :237)."""
    d2 = dcorr_from_dist(dx, dy, biased)[0]          # already 0 when a variance is <= 0
    if biased and d2 <= 0:
        return 0.0
    return float(np.sqrt(d2)) if biased else float(d2)


def dcorr(x, y, biased=False, sklearn_arith=False):
    """Distance correlation of x (n,)|(n,p) and y (n,)|(n,q), Euclidean distances (Szekely, Rizzo & Bakirov 2007
    biased V-statistic; Szekely & Rizzo 2014 U-centred). Returns dict(dcor2 = dCor^2, dcor = sqrt(max(dcor2, 0)),
    hyppo = the value pyspi stores: dCor if biased else the signed dCor^2, see hyppo_dcorr_from_dist).
    sklearn_arith=True builds distances with sklearn's dot-product formula, as hyppo does (hyppo/tools/common.py:497-499
    -> pairwise_distances); the difference is at rounding level."""
    if sklearn_arith:
        dx, dy = np.sqrt(_sk_sqdist(x)), np.sqrt(_sk_sqdist(y))
    else:
        dx, dy = _pdist(x), _pdist(y)
    d2 = dcorr_from_dist(dx, dy, biased)[0]
    return {"dcor2": float(d2), "dcor": float(np.sqrt(max(d2, 0.0))),
            "hyppo": hyppo_dcorr_from_dist(dx, dy, biased)}


# ------------------------------------------------------------------------------- HSIC
def median_gamma(x, sklearn_arith=False):
    """Gaussian-kernel gamma by the median heuristic, as hyppo computes it (hyppo/tools/common.py:321-333):
    med = median of the off-diagonal Euclidean distances of x (both triangles; the stride trick at common.py:326-330
    visits every off-diagonal entry once), med = 1 if it is 0, gamma = 1/(2 med^2). hyppo derives gamma from x ONLY
    and reuses it for the y kernel (common.py:342-343)."""
    d = np.sqrt(_sk_sqdist(x)) if sklearn_arith else _pdist(x)
    med = np.median(d[~np.eye(d.shape[0], dtype=bool)])
    med = med if med else 1.0
    return 1.0 / (2 * med ** 2)


def hsic_biased(x, y, gamma=None):
    """Biased empirical HSIC with Gaussian kernels, HSIC_b = tr(K H L H) / n^2: the 1/n^2 form of Gretton, Fukumizu,
    Teo, Song, Schoelkopf & Smola 2008 (NIPS 20, 'A kernel statistical test of independence'). Gretton et al. 2005
    (ALT) eq. 9 scales by (n-1)^-2 instead; the factor cancels in 'normalised'. H = I - 11^T/n, K_ij = exp(-gamma |x_i - x_j|^2).
    gamma=None uses median_gamma(x) for BOTH kernels, mirroring hyppo (common.py:321-343); a pair (gamma_x, gamma_y) is allowed.
    Returns dict(hsic, hsic_xx, hsic_yy, normalised = hsic / sqrt(hsic_xx * hsic_yy), gamma_x, gamma_y). normalised is 0
    when hsic_xx * hsic_yy <= 0 (a constant series), as hyppo returns 0 for a zero variance (dcorr.py:438-439)."""
    gx, gy = (median_gamma(x),) * 2 if gamma is None else (np.broadcast_to(gamma, 2))
    K, L = np.exp(-gx * _pdist(x) ** 2), np.exp(-gy * _pdist(y) ** 2)
    n = K.shape[0]
    Kc, Lc = _double_centre(K), _double_centre(L)              # H K H, H L H
    h = lambda a, b: float(np.sum(a * b) / n ** 2)             # tr(HKH HLH) = sum (HKH)*(HLH)
    hxy, hxx, hyy = h(Kc, Lc), h(Kc, Kc), h(Lc, Lc)
    nrm = hxy / np.sqrt(hxx * hyy) if hxx * hyy > 0 else 0.0
    return {"hsic": hxy, "hsic_xx": hxx, "hsic_yy": hyy, "normalised": nrm,
            "gamma_x": float(gx), "gamma_y": float(gy)}


def hyppo_hsic_replica(x, y, biased=False):
    """Re-creates hyppo's Hsic.statistic, which pyspi's hsic / hsic_biased call (hyppo/independence/hsic.py:117-122):
    Gaussian K, L with the x-derived median gamma (common.py:321-343, sklearn arithmetic), D_x = 1 - K/max(K),
    D_y = 1 - L/max(L), then Dcorr(compute_distance=None, bias). So it is dCor (biased) or the signed U-dCor^2 of the
    kernel-induced distances (hyppo_dcorr_from_dist), NOT the Gretton tr(KHLH)/n^2."""
    g = median_gamma(x, sklearn_arith=True)
    K, L = np.exp(-g * _sk_sqdist(x)), np.exp(-g * _sk_sqdist(y))
    return hyppo_dcorr_from_dist(1 - K / np.max(K), 1 - L / np.max(L), biased)


# -------------------------------------------------------------------------------- HHG
def hhg(x, y):
    """Heller, Heller & Gorfine 2013 (Biometrika 100, 503-510) sum-of-chi-square statistic T = sum_{i != j} S(i,j),
    S(i,j) = (n-2) (A12 A21 - A11 A22)^2 / (A1. A2. A.1 A.2), A = 2x2 table over k not in {i,j} of
    [d(x_i,x_k) <= d(x_i,x_j)] x [d(y_i,y_k) <= d(y_i,y_j)]. A zero margin contributes 0 (hyppo/independence/hhg.py:293-295).
    Swapping x and y transposes every table, so T(x,y) = T(y,x): pyspi labels hhg 'Directed' but it is symmetric.
    Cost O(n^3); fine for n <= ~300."""
    dx, dy = _pdist(x), _pdist(y)
    n, tot = dx.shape[0], 0.0
    for i in range(n):
        a = (dx[i][None, :] <= dx[i][:, None]).astype(float)    # a[j, k] = d(x_i,x_k) <= d(x_i,x_j)
        b = (dy[i][None, :] <= dy[i][:, None]).astype(float)
        a[:, i] = b[:, i] = 0.0                               # drop k = i
        a[np.arange(n), np.arange(n)] = b[np.arange(n), np.arange(n)] = 0.0   # drop k = j
        t11 = (a * b).sum(1); t12 = (a * (1 - b)).sum(1); t21 = ((1 - a) * b).sum(1)
        t22 = (n - 2) - t11 - t12 - t21                       # k = i, j are in none of the four cells
        den = (t11 + t12) * (t21 + t22) * (t11 + t21) * (t12 + t22)
        num = (n - 2) * (t12 * t21 - t11 * t22) ** 2
        s = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        s[i] = 0.0
        tot += s.sum()
    return float(tot)


# ------------------------------------------------------------------------------ DcorrX
def dcorrx_mehta(x, y, max_lag, biased=False):
    """Cross-distance correlation (Mehta et al. 2019, arXiv:1907.02088): DcorrX = sum_{j=0}^{M} ((n-j)/n) Dcorr(x[j:n], y[0:n-j]),
    with Dcorr = U-centred dCor^2 (hyppo's default bias=False) on distances between the SERIES SAMPLES of each slice,
    so D_x^{(j)} = D_x[j:, j:]. Lag j pairs x_{t+j} with y_t (y leads); in pyspi x = process i, y = process j.
    Distances are exact differences (_pdist); pyspi (hyppo) uses sklearn's dot-product arithmetic (see _sk_sqdist),
    so even the lag-0 term, which is the same quantity in both, differs by rounding: <= 6.2e-16 on the lagged_dcorr_mgc
    check data, 5.0e-11 on the tied logistic_maps (audit/notes/lagged_dcorr_mgc.md).
    Returns (stat, list of weighted per-lag terms, opt_lag = argmax of the terms)."""
    dx, dy = _pdist(x), _pdist(y)
    n, terms = dx.shape[0], []
    for j in range(max_lag + 1):
        v = hyppo_dcorr_from_dist(dx[j:, j:], dy[:n - j, :n - j], biased)
        terms.append((n - j) * v / n)
    return float(np.sum(terms)), terms, int(np.argmax(terms))


def hyppo_dcorrx_replica(x, y, max_lag, biased=False):
    """What hyppo's DcorrX.statistic computes (hyppo/time_series/_utils.py:85-108), mirrored:
    lag 0 is Dcorr on the raw series (line 93). For j >= 1, the SLICES OF THE DISTANCE MATRICES (lines 99-101)
    go to a Dcorr whose compute_distance is still 'euclidean' (line 92). Each row of D_x[j:, j:] is treated
    as a point in R^{n-j}, and Euclidean distances are taken again (distance-of-distance-matrix). Weights are
    (n-j)/n (line 102). stat = sum, opt_lag = argmax (lines 105-106). sklearn arithmetic is used throughout.
    """
    dx, dy = np.sqrt(_sk_sqdist(x)), np.sqrt(_sk_sqdist(y))
    n = dx.shape[0]
    terms = [hyppo_dcorr_from_dist(dx, dy, biased)]
    for j in range(1, max_lag + 1):
        sx, sy = dx[j:n, j:n], dy[0:n - j, 0:n - j]
        v = hyppo_dcorr_from_dist(np.sqrt(_sk_sqdist(sx)), np.sqrt(_sk_sqdist(sy)), biased)
        terms.append((n - j) * v / n)
    return float(np.sum(terms)), terms, int(np.argmax(terms))


# --------------------------------------------------------------------- DTW family masks
def sakoe_chiba_mask(sz1, sz2, radius=1):
    """Sakoe-Chiba band, mirroring tslearn/metrics/_masks.py:15-29 (for equal lengths, |i-j| <= radius)."""
    mask = np.zeros((sz1, sz2), bool)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in range(sz2):
            mask[max(0, i - radius):min(sz1, i + width) + 1, i] = True
    else:
        width = sz2 - sz1 + radius
        for i in range(sz1):
            mask[i, max(0, i - radius):min(sz2, i + width) + 1] = True
    return mask


def itakura_mask(sz1, sz2, max_slope=2.0):
    """Itakura parallelogram (Itakura 1975), mirroring tslearn/metrics/_masks.py:91-140, including the
    2-decimal rounding of the bounds before ceil/floor."""
    min_slope = 1 / float(max_slope)
    max_slope = max_slope * float(sz1) / float(sz2)
    min_slope = min_slope * float(sz1) / float(sz2)
    k = np.arange(sz2)
    lb = np.ceil(np.maximum(np.round(min_slope * k, 2), np.round((sz1 - 1) - max_slope * (sz2 - 1) + max_slope * k, 2)))
    ub = np.floor(np.minimum(np.round(max_slope * k, 2), np.round((sz1 - 1) - min_slope * (sz2 - 1) + min_slope * k, 2)) + 1)
    mask = np.zeros((sz1, sz2), bool)
    for i in range(sz2):
        mask[int(lb[i]):int(ub[i]), i] = True
    if not (mask.any(1).all() and mask.any(0).all()):
        raise ValueError("itakura_max_slope constraint is infeasible for these sizes")
    return mask


def _mask(n, m, constraint, radius, slope):
    """tslearn/metrics/_masks.py:318-350: None -> full mask; 'sakoe_chiba' -> radius (default 1);
    'itakura' -> max slope (default 2)."""
    if constraint is None:
        return np.ones((n, m), bool)
    if constraint == "sakoe_chiba":
        return sakoe_chiba_mask(n, m, radius)
    if constraint == "itakura":
        return itakura_mask(n, m, slope)
    raise ValueError(constraint)


def _diag(n, m, d):
    """1-based cells (i, j) of anti-diagonal i + j = d in an (n+1, m+1) DP table."""
    i = np.arange(max(1, d - m), min(n, d - 1) + 1)
    return i, d - i


def _sqcost(x, y):
    x, y = _as2d(x), _as2d(y)
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


# ------------------------------------------------------------------------------ DTW
def dtw_acc(x, y, constraint=None, sakoe_chiba_radius=1, itakura_max_slope=2.0):
    """Accumulated cost R (n+1, m+1), with R[0,0] = 0, inf boundaries and inf in masked cells.
    R[i,j] = c(i,j) + min(R[i-1,j], R[i,j-1], R[i-1,j-1]), with squared local cost (tslearn/metrics/_dtw.py:325-344).
    Each anti-diagonal is filled in one vectorised step, which applies the same arithmetic as the cell loop."""
    C = _sqcost(x, y)
    n, m = C.shape
    M = _mask(n, m, constraint, sakoe_chiba_radius, itakura_max_slope)
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    for d in range(2, n + m + 1):
        i, j = _diag(n, m, d)
        best = np.minimum(np.minimum(R[i - 1, j], R[i, j - 1]), R[i - 1, j - 1])
        R[i, j] = np.where(M[i - 1, j - 1], C[i - 1, j - 1] + best, np.inf)
    return R


def dtw_dp(x, y, constraint=None, sakoe_chiba_radius=1, itakura_max_slope=2.0, squared_cost=True):
    """Dynamic time warping by O(nm) dynamic programming (Sakoe & Chiba 1978, IEEE TASSP 26, 43-49).
    squared_cost=True (tslearn form, pyspi's dtw*): sqrt(min over paths of sum (x_i - y_j)^2), steps (1,0), (0,1), (1,1)
        of weight 1 (tslearn/metrics/_dtw.py:383-400), masks of tslearn/metrics/_masks.py; radius r = dtaidistance window r+1.
    squared_cost=False: the 1978 symmetric form with P = 0, g(i,j) = min(g(i-1,j) + d, g(i-1,j-1) + 2d, g(i,j-1) + d),
        d = |x_i - y_j|, D = g(n,m)/(n+m), same masks."""
    if squared_cost:
        return float(np.sqrt(dtw_acc(x, y, constraint, sakoe_chiba_radius, itakura_max_slope)[-1, -1]))
    C = np.sqrt(_sqcost(x, y))
    n, m = C.shape
    M = _mask(n, m, constraint, sakoe_chiba_radius, itakura_max_slope)
    g = np.full((n + 1, m + 1), np.inf)
    g[0, 0] = 0.0
    for d in range(2, n + m + 1):
        i, j = _diag(n, m, d)
        c = C[i - 1, j - 1]
        v = np.minimum(np.minimum(g[i - 1, j] + c, g[i, j - 1] + c), g[i - 1, j - 1] + 2 * c)
        g[i, j] = np.where(M[i - 1, j - 1], v, np.inf)
    return float(g[n, m] / (n + m))


def dtw_path(x, y):
    """Optimal unconstrained path. Backtracking mirrors tslearn/metrics/utils.py:20-46: from (n-1, m-1),
    take argmin over [diag, up (i-1), left (j-1)], with the first index winning ties.
    Returns (dist, list of (i, j))."""
    A = dtw_acc(x, y)[1:, 1:]
    i, j = A.shape[0] - 1, A.shape[1] - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin([A[i - 1, j - 1], A[i - 1, j], A[i, j - 1]]))
            i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
        path.append((i, j))
    return float(np.sqrt(A[-1, -1])), path[::-1]


# ----------------------------------------------------------------------------- LCSS
def lcss_dp(x, y, eps=1.0, constraint=None, sakoe_chiba_radius=1, itakura_max_slope=2.0, strict=False):
    """LCSS similarity (Vlachos, Gunopulos & Kollios 2002, ICDE) with tslearn's conventions: a match is ||x_i - y_j|| <= eps
    (strict=True gives Vlachos' < eps), a masked cell holds 0, score L(n,m)/min(n,m) (tslearn/metrics/dtw_variants.py:2301-2310
    generic path, 2343-2345). For the Sakoe-Chiba band, 0 in masked cells equals Vlachos' delta-window rule.
    NOTE: the numpy path pyspi runs (dtw_variants.py:2259) gates cells with numpy.isfinite(bool mask), always True, so in
    tslearn 0.9.0 pyspi's lcss_constraint-* equal unconstrained lcss (audit/notes/dtw_family.md): use constraint=None."""
    C = np.sqrt(_sqcost(x, y))
    n, m = C.shape
    M = _mask(n, m, constraint, sakoe_chiba_radius, itakura_max_slope)
    L = np.zeros((n + 1, m + 1))
    for d in range(2, n + m + 1):
        i, j = _diag(n, m, d)
        hit = C[i - 1, j - 1] < eps if strict else C[i - 1, j - 1] <= eps
        v = np.where(hit, 1 + L[i - 1, j - 1], np.maximum(L[i, j - 1], L[i - 1, j]))
        L[i, j] = np.where(M[i - 1, j - 1], v, 0.0)
    return float(L[n, m] / min(n, m))


# ------------------------------------------------------------------------ soft-DTW
def _softmin3(a, b, c, gamma):
    """-gamma log(sum exp(-v/gamma)), with the max-shift order of tslearn/metrics/soft_dtw_fast.py:37-48."""
    a, b, c = a / -gamma, b / -gamma, c / -gamma
    mx = np.maximum(np.maximum(a, b), c)
    return -gamma * (np.log(np.exp(a - mx) + np.exp(b - mx) + np.exp(c - mx)) + mx)


def soft_dtw_acc(x, y, gamma=1.0):
    """Soft-DTW table R (n+1, m+1) (Cuturi & Blondel 2017, ICML, Algorithm 1):
    R[i,j] = D[i,j] + softmin_gamma(R[i-1,j], R[i-1,j-1], R[i,j-1]), D = squared Euclidean,
    R[0,0] = 0, other boundaries = +inf (tslearn uses DBL_MAX, soft_dtw_fast.py:115-118; same result).
    Returns (R, D)."""
    D = _sqcost(x, y)
    n, m = D.shape
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    with np.errstate(invalid="ignore"):
        for d in range(2, n + m + 1):
            i, j = _diag(n, m, d)
            R[i, j] = D[i - 1, j - 1] + _softmin3(R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma)
    return R, D


def soft_dtw_dp(x, y, gamma=1.0):
    """Soft-DTW value R[n,m] (Cuturi & Blondel 2017, Alg. 1); -> dtw^2 as gamma -> 0 (tslearn returns dtw^2 exactly at
    gamma == 0, softdtw_variants.py:627-628). For gamma > 0 softmin < min, so soft_dtw(x, x) < 0 (not a divergence), and
    dtw^2 - gamma*log(3)*(n+m-1) <= soft_dtw <= dtw^2. pyspi's softdtw* SPIs ignore the global_constraint
    (distance.py:205-208), so the constraint variants equal softdtw."""
    return float(soft_dtw_acc(x, y, gamma)[0][-1, -1])


def soft_dtw_grad_D(R, D, gamma):
    """E = d soft_dtw / d D (n, m), from the backward recursion of Cuturi & Blondel 2017, Algorithm 2."""
    n, m = D.shape
    Dp = np.zeros((n + 2, m + 2)); Dp[1:-1, 1:-1] = D
    Rp = np.full((n + 2, m + 2), -np.inf); Rp[1:-1, 1:-1] = R[1:, 1:]; Rp[-1, -1] = R[n, m]
    E = np.zeros((n + 2, m + 2)); E[-1, -1] = 1.0
    for d in range(n + m, 1, -1):
        i, j = _diag(n, m, d)
        a = np.exp((Rp[i + 1, j] - Rp[i, j] - Dp[i + 1, j]) / gamma)
        b = np.exp((Rp[i, j + 1] - Rp[i, j] - Dp[i, j + 1]) / gamma)
        c = np.exp((Rp[i + 1, j + 1] - Rp[i, j] - Dp[i + 1, j + 1]) / gamma)
        E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    return E[1:-1, 1:-1]


def soft_dtw_barycenter_objective(z, X, gamma=1.0, weights=None):
    """Objective and gradient of the soft-DTW barycenter (Cuturi & Blondel 2017, Sec. 4, barycenters):
    f(z) = sum_k w_k sdtw_gamma(z, X_k), grad_i = sum_k w_k sum_j E^k_ij 2 (z_i - X_kj).
    The weights are uniform ones, as in tslearn/barycenters/softdtw.py:19-35 and utils._set_weights.
    Use it as an optimality check: the gradient at pyspi's softdtw barycenter should be small. z (T,), X (K, T)."""
    z = np.asarray(z, float)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    f, g = 0.0, np.zeros_like(z)
    for wk, xk in zip(w, X):
        R, D = soft_dtw_acc(z, xk, gamma)
        E = soft_dtw_grad_D(R, D, gamma)
        f += wk * R[-1, -1]
        g += wk * 2 * (E.sum(1) * z - E @ np.asarray(xk, float))
    return float(f), g


# ---------------------------------------------------------------------- barycenters
def euclidean_barycenter(X):
    """Euclidean barycenter = pointwise mean over series (tslearn/barycenters/euclidean.py:44). X (K, T) -> (T,)."""
    return np.asarray(X, float).mean(0)


def dba(X, max_iter=30, tol=1e-5):
    """DTW barycenter averaging (Petitjean, Ketterlin & Gancarski 2011, Pattern Recogn. 44, 678-693) as pyspi calls it
    (tslearn dtw_barycenter_averaging, n_init=1, tslearn/barycenters/dba.py:700-737): init = pointwise mean (dba.py:23-25);
    each iteration aligns every series to the barycenter (dtw_path), cost = mean_k dtw_k^2 (uniform weights,
    dba.py:229-305), then sets each point to the mean of its associated points (dba.py:400-434); stop when
    |cost_prev - cost| < tol or the cost rises, returning the just-updated barycenter either way (dba.py:727-737).
    X (K, T). Returns (barycenter (T,), per-iteration costs of the pre-update barycenter)."""
    X = np.asarray(X, float)
    z = X.mean(0)
    costs, prev = [], np.inf
    for _ in range(max_iter):
        num, cnt, cost = np.zeros_like(z), np.zeros_like(z), 0.0
        for xk in X:
            dist, path = dtw_path(z, xk)
            cost += dist ** 2
            p = np.array(path)
            np.add.at(num, p[:, 0], xk[p[:, 1]])
            np.add.at(cnt, p[:, 0], 1.0)
        cost /= len(X)
        costs.append(cost)
        z = num / cnt
        if abs(prev - cost) < tol or prev < cost:
            break
        prev = cost
    return z, costs


# ---------------------------------------------------------------------------- GWtau
def wasserstein2_1d(a, b):
    """Exact W2 between the empirical uniform distributions of a (N,) and b (M,):
    sqrt of the integral over u in (0,1) of (F_a^{-1}(u) - F_b^{-1}(u))^2. This is computed on the merged
    quantile breakpoints. It matches Kravtsova et al. 2023 eq. 11 (N = M) and eq. 12 (lambda_ij overlaps)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    u = np.union1d(np.arange(len(a) + 1) / len(a), np.arange(len(b) + 1) / len(b))
    mid, du = 0.5 * (u[1:] + u[:-1]), np.diff(u)
    ia = np.minimum((mid * len(a)).astype(int), len(a) - 1)
    ib = np.minimum((mid * len(b)).astype(int), len(b) - 1)
    return float(np.sqrt(np.sum(du * (a[ia] - b[ib]) ** 2)))


def root_arc_lengths(x, include_root=False):
    """Root distances on the trajectory f(t) = (t, x_t), t = 0..n-1:
    d(r, x_k) = sum_{t<k} sqrt(1 + (x_{t+1} - x_t)^2) (Kravtsova, McGee & Dawes 2023, Bull. Math. Biol. 85:77, eq. 8).
    include_root=False drops the root's own 0, as pyspi distance.py:273-277 and the authors' vec_geo_dist.m do (n-1 values)."""
    x = np.asarray(x, float).ravel()
    L = np.cumsum(np.sqrt(1.0 + np.diff(x) ** 2))
    return np.concatenate([[0.0], L]) if include_root else L


def gwtau(x, y, include_root=False):
    """GW-tau with p = 2 (Kravtsova et al. 2023, eqs. 5, 10; Prop. 1a): W2 between the root-distance distributions of
    the two series; pyspi docs: 'distances from the start of each time series to every point ... compared using the
    Wasserstein distance' (spi-descriptions/distance-similarity.md:164-180; no supplement.txt entry in section 1.2).
    pyspi (distance.py:279-315) sorts both vectors DESCENDING and takes sqrt(mean((a-b)^2)): for equal lengths the same
    monotone coupling, so equal to this W2. pyspi drops the root (include_root=False); the paper's uniform mass on all n
    points (include_root=True) changes the value, for equal lengths, by exactly sqrt((n-1)/n)."""
    return wasserstein2_1d(root_arc_lengths(x, include_root), root_arc_lengths(y, include_root))


# ------------------------------------------------------------------------------ CCM
def _pearson0(a, b):
    """Pearson rho, set to 0 when either variance is 0 (cppEDM Common.cc behaviour, per audit/notes/ccm.md)."""
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def simplex_predictions(x, y, E, tau=-1, Tp=0, lib_indices=None):
    """Simplex cross-map predictions of y from the delay embedding of x (Sugihara & May 1990, Nature 344; Sugihara et al.
    2012, Science 338, supplement). Returns (rows (m,), y_hat (m,), tied (m,) bool).
    Embedding row t = (x_t, x_{t+tau}, ..., x_{t+(E-1)tau}) (tau < 0: past lags); rows need a full embedding and a valid
    t+Tp. Every valid row is predicted from the library lib_indices (default: all valid rows), t itself excluded.
    Euclidean distances, summed coordinate by coordinate as cppEDM Distance() (EDM_Neighbors.cc L453-468). The E+1
    nearest (stable argsort) get u_i = exp(-d_i/d_1) (if d_1 = 0, u_i = [d_i == 0]), floored at 1e-6 (Simplex.cc L44-80);
    y_hat = sum u_i y(t_i + Tp) / sum u_i.
    tied[a]: the (E+1)-th and (E+2)-th distances are equal, so the neighbour SET is ambiguous. pyEDM's CCM does not expand
    ties (the expansion at EDM_Neighbors.cc L282 is gated on method == Simplex; CCM runs as Method::CCM, API.cc L659): on
    tied rows it keeps whichever E+1 rows libstdc++'s unstable std::sort (EDM_Neighbors.cc L226) puts first, which is
    implementation-defined. Compare with pyEDM only where no row is tied (scratchpad probe kernels/probe_ccm_stdsort.py:
    an introsort emulation reproduces pyEDM to 1e-13 on logistic_maps, both directions; audit/checks/archive_probes.py
    packs the probe into audit/checks/probes/scratch_probes_2026-10-15.tar.xz under the same relative path)."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    n, lag, k = len(x), -tau, E + 1
    rows = np.arange((E - 1) * lag, n)
    rows = rows[(rows + Tp >= 0) & (rows + Tp < n)]
    emb = np.stack([x[rows - j * lag] for j in range(E)], 1)
    lib = rows if lib_indices is None else np.intersect1d(rows, np.asarray(lib_indices))
    Lemb, ylib = emb[np.searchsorted(rows, lib)], y[lib + Tp]
    pred, tied = np.empty(len(rows)), np.zeros(len(rows), bool)
    for a, t in enumerate(rows):
        ss = np.zeros(len(lib))
        for j in range(E):
            ss += (Lemb[:, j] - emb[a, j]) ** 2
        d = np.sqrt(ss)
        d[lib == t] = np.inf
        o = np.argsort(d, kind="stable")
        tied[a] = len(o) > k and d[o[k - 1]] == d[o[k]]
        dk = d[o[:k]]
        w = np.exp(-dk / dk[0]) if dk[0] > 0 else (dk == 0).astype(float)
        w = np.maximum(w, 1e-6)
        pred[a] = np.sum(w * ylib[o[:k]]) / np.sum(w)
    return rows + Tp, pred, tied


def simplex_crossmap(x, y, E, tau=-1, Tp=0, lib_indices=None, return_tied=False):
    """Cross-map skill rho(y, y_hat | M_x) = Pearson rho over simplex_predictions (Sugihara et al. 2012).
    Direction (Sugihara): high skill of 'x xmap y' is evidence that y CAUSES x. This equals pyEDM's column
    'x:y' (columns=x, target=y). pyEDM's defaults are E, Tp=0, tau=-1, knn=E+1, exclusionRadius=0
    (pyEDM/CoreEDM.py:363-385). x, y (n,). Returns rho, or (rho, number of tied prediction rows) with
    return_tied=True; a non-zero count means pyEDM's value depends on std::sort order (see simplex_predictions)."""
    t, pred, tied = simplex_predictions(x, y, E, tau, Tp, lib_indices)
    rho = _pearson0(np.asarray(y, float).ravel()[t], pred)
    return (rho, int(tied.sum())) if return_tied else rho


def ccm_skill_curve(x, y, E, lib_sizes, n_samples=100, seed=0, tau=-1, random=True):
    """CCM curve (Sugihara 2012, Fig. 3): mean simplex_crossmap(x, y) over n_samples libraries per size L. random=True
    draws L valid rows without replacement (pyEDM's default); numpy's RNG cannot reproduce pyEDM's
    std::default_random_engine(seed) draws, so only curve statistics are comparable. random=False uses the first L
    valid rows (contiguous). Returns (len(lib_sizes),)."""
    rng = np.random.default_rng(seed)
    valid = np.arange((E - 1) * (-tau), len(np.ravel(x)))
    out = []
    for L in lib_sizes:
        if not random:
            out.append(simplex_crossmap(x, y, E, tau, 0, valid[:L]))
            continue
        out.append(np.mean([simplex_crossmap(x, y, E, tau, 0, rng.choice(valid, L, replace=False))
                            for _ in range(n_samples)]))
    return np.array(out)


# -------------------------------------------------------------------- causal (cdt)
def _sqdiff_scaled(x, l):
    """(x_i/l - x_j/l)^2, (n, n), exact differencing of pre-scaled inputs: sklearn RBF's arithmetic, pdist(X / length_scale,
    'sqeuclidean') for k(X) and cdist for k(X, X_train) (sklearn/gaussian_process/kernels.py:1561, 1569), NOT pairwise.py's
    dot-product form. In the ANM regime (l ~ 1e-4, residual ~ 1e-10 y) rounding-level changes in K reach the z-scored residual."""
    z = np.asarray(x, float).ravel() / l
    return (z[:, None] - z[None, :]) ** 2


def _rbf(x, c, l):
    """c * exp(-0.5 * (x_i/l - x_j/l)^2), (n, n): sklearn ConstantKernel(c) * RBF(l) (kernels.py:1561-1564)."""
    return c * np.exp(-0.5 * _sqdiff_scaled(x, l))


def gp_posterior_mean(x, y, constant=1.0, length_scale=1.0, alpha=1e-10):
    """GP regression posterior mean at the training inputs (Rasmussen & Williams 2006, Alg. 2.1), with zero prior
    mean (normalize_y=False), kernel k = constant * exp(-|x-x'|^2 / (2 length_scale^2)) and jitter alpha.
    These mirror sklearn GaussianProcessRegressor: K + alpha I, lower Cholesky, alpha_ = cho_solve, mean = K alpha_
    (sklearn/gaussian_process/_gpr.py:350-363, 446-447), with the RBF arithmetic of _sqdiff_scaled.
    x, y (n,). Returns the prediction (n,)."""
    K = _rbf(x, constant, length_scale)
    cf = cho_factor(K + alpha * np.eye(len(K)), lower=True)
    return K @ cho_solve(cf, np.asarray(y, float).ravel())


def gp_fit_hyperparameters(x, y, alpha=1e-10):
    """Maximise the log marginal likelihood over log(constant), log(length_scale) with L-BFGS-B,
    starting at (1, 1) with bounds [1e-5, 1e5]. This mirrors sklearn's default C()*RBF() and
    n_restarts_optimizer=0 (_gpr.py:216-217, 249; kernels.py:1236, 1511). Analytic gradient (Rasmussen & Williams 2006, Sec. 5.4.1).
    An independent optimiser, so expect agreement with sklearn's fit only to optimiser tolerance."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()

    def nlml(th):
        c, l = np.exp(th)
        S = _sqdiff_scaled(x, l)                               # (x_i - x_j)^2 / l^2, sklearn arithmetic
        K0 = np.exp(-0.5 * S)
        cf = cho_factor(c * K0 + alpha * np.eye(len(x)), lower=True)
        a = cho_solve(cf, y)
        Ki = cho_solve(cf, np.eye(len(x)))
        W = np.outer(a, a) - Ki
        g = [-0.5 * np.sum(W * (c * K0)), -0.5 * np.sum(W * (c * K0 * S))]
        f = 0.5 * y @ a + np.log(np.diag(cf[0])).sum() + 0.5 * len(x) * np.log(2 * np.pi)
        return f, np.array(g)
    r = minimize(nlml, np.zeros(2), jac=True, method="L-BFGS-B", bounds=[(np.log(1e-5), np.log(1e5))] * 2)
    return tuple(np.exp(r.x))


def cdt_fast_hsic_gamma(X, Y, maxpnt=200):
    """cdt's FastHsicTestGamma statistic, mirrored from cdt/causality/pairwise/ANM.py:53-103:
    keep floor(arange(0, m, (m-1)/(maxpnt-1))) evenly spaced rows when m > maxpnt (ANM.py:81-86);
    rbf_dot bandwidth deg = sqrt(0.5 * median(upper-triangle squared distances > 0)) (ANM.py:61-63);
    K = exp(-H/(2 deg^2)); statistic = (1/m) sum(HKH * HLH), i.e. 1/m scaling, not 1/m^2 (ANM.py:99).
    A non-finite value is set to 0."""
    X, Y = np.asarray(X, float).ravel(), np.asarray(Y, float).ravel()
    m = len(X)
    if m > maxpnt:
        idx = np.floor(np.arange(0, m, float(m - 1) / (maxpnt - 1))).astype(int)
        X, Y, m = X[idx], Y[idx], len(idx)

    def rbf_dot(v):
        G = (v * v)[:, None]
        H = G + G.T - 2.0 * np.outer(v, v)
        dists = (H - np.tril(H)).ravel()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
        return np.exp(-H / 2.0 / deg ** 2)
    Kc, Lc = _double_centre(rbf_dot(X)), _double_centre(rbf_dot(Y))
    s = (1.0 / m) * np.sum(Kc.T * Lc)
    return float(s) if np.isfinite(s) else 0.0


def anm_replica(x, y, constant=None, length_scale=None, alpha=1e-10):
    """ANM fit score for x -> y (Hoyer et al. 2009, NIPS): GP-regress y = f(x) + n, then measure dependence between the
    residual and x (lower = x -> y more plausible). As pyspi runs it (pyspi/statistics/causal.py:18-29, patched anm_score)
    the z-scored inputs are not rescaled; residual = y_hat - y; score = normalized_hsic(residual, x) (ANM.py:106-111:
    z-score each with np.std ddof=0, then cdt_fast_hsic_gamma). constant/length_scale fix the GP (e.g. to a known sklearn
    fit); None fits them with gp_fit_hyperparameters, since sklearn OPTIMISES the default C()*RBF() (_gpr.py:249).
    With alpha=1e-10 the fit usually reaches length_scale ~ 1e-5..1e-4 (interpolation): the residual is ~1e-10 y, the
    z-scored score measures dependence between y and x (audit/notes/cdt_pairwise.md:65), and the RBF arithmetic matters
    (see _sqdiff_scaled).
    ENVIRONMENT: at sklearn's fitted (constant, length_scale) the replica follows pyspi's arithmetic: <= 1.6e-15 on 8 of
    9 cdt_pairwise replica-theta runs. The ninth (nonlinear_dependence T=256, pair (3, 1)) differs by 7.5e-8 in the
    comparison venv (scipy 1.18.1) and by 6.6e-16 when the same replica runs in the project venv (scipy 1.17.1): LAPACK
    roundoff amplified by the estimator, not a replica defect. The fit is also BLAS-thread sensitive:
    multi-threaded OpenBLAS moved sklearn's log theta by up to 5.6e-7 and the score by ~1e-6, so reproduce with
    OPENBLAS_NUM_THREADS=1 (audit/notes/cdt_pairwise.md, "Thread and environment sensitivity of anm")."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if constant is None or length_scale is None:
        constant, length_scale = gp_fit_hyperparameters(x, y, alpha)
    r = gp_posterior_mean(x, y, constant, length_scale, alpha) - y
    zs = lambda v: (v - v.mean()) / v.std()
    return cdt_fast_hsic_gamma(zs(r), zs(x))


def _cds_discretize(v, ffactor, maxdev, norm=True):
    """cdt CDS.py:56-80 for numerical data: if v has more unique values than the discretised range
    (2*ffactor*maxdev+1 values once it is long enough), z-score it, re-standardise on |z| < maxdev, round(z*ffactor)
    (half-to-even) and clip to +-ffactor*maxdev. With norm=False it only rounds and clips."""
    nu, nv = len(np.unique(v)), 2 * ffactor * maxdev + 1
    if norm and not nu > nv:                 # CDS.py:61, 70 reduce to: more unique values than nv
        return v
    if norm:
        v = (v - v.mean()) / v.std()
        vf = v[np.abs(v) < maxdev]
        v = (v - vf.mean()) / vf.std()
    return np.clip(np.round(v * ffactor), -ffactor * maxdev, ffactor * maxdev)


def cds_ref(x, y, ffactor=2, maxdev=3, minc=12):
    """Conditional distribution similarity x -> y (Fonollosa 2016, arXiv:1601.06680, 'CDS'), mirroring cdt CDS.cds_score
    (CDS.py:143-195) for continuous y: discretise x; for each x-bin with more than minc points, centre y on the bin mean,
    scale by the GLOBAL std(y) (ddof 0), discretise into the 13 bins -6..6 and normalise to a pmf. Score = std of all
    entries of (pmf matrix - column mean); low (similar conditionals) favours x -> y. cdt's predict_proba =
    cds(y,x) - cds(x,y), so positive means x -> y."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    xd = _cds_discretize(x, ffactor, maxdev)
    vals = np.arange(-ffactor * maxdev, ffactor * maxdev + 1)
    if not len(np.unique(y)) > len(vals):
        raise NotImplementedError("discrete-y branch of CDS.py:178-184 not mirrored")
    pyx = []
    for a in np.unique(xd):
        sel = xd == a
        if sel.sum() > minc:
            yx = _cds_discretize((y[sel] - y[sel].mean()) / y.std(), ffactor, maxdev, norm=False)
            p = np.array([(yx == v).sum() for v in vals], float)
            pyx.append(p / p.sum())
    if not pyx:
        return 0.0
    pyx = np.array(pyx)
    return float(np.std(pyx - pyx.mean(0)))


def reci_ref(x, y):
    """RECI fit score for x -> y (Bloebaum et al. 2018, AISTATS): min-max scale both variables to [0, 1],
    fit y by OLS on [1, x^3] and return the MSE. Mirrors cdt RECI.b_fit_score (RECI.py:104-118: degree-3 polynomial
    features with the x and x^2 columns zeroed). Infer x -> y when reci(x, y) < reci(y, x).
    A constant variable scales to 0, as sklearn minmax_scale does (zero range -> 1, preprocessing/_data.py:537-539)."""
    s = lambda v: (v - v.min()) / ((v.max() - v.min()) or 1.0)
    x, y = s(np.asarray(x, float).ravel()), s(np.asarray(y, float).ravel())
    Z = np.column_stack([np.ones_like(x), x ** 3])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.mean((Z @ beta - y) ** 2))


def _spacing_entropy(v):
    """Spacing estimator used by cdt (IGCI.py:42-56): sum of log|delta| over non-zero sorted spacings,
    divided by (n-1), plus psi(n) - psi(1)."""
    d = np.diff(np.sort(v))
    d = d[d != 0]
    return float(np.sum(np.log(np.abs(d))) / (len(v) - 1) + digamma(len(v)) - digamma(1))


def igci_ref(x, y):
    """IGCI, entropy estimator with a Gaussian reference measure (Janzing et al. 2012, Artif. Intell. 182-183,
    with Daniusis et al. 2010). Standardise each variable (ddof 0, as cdt's StandardScaler,
    IGCI.py:142), then C_{x->y} = H(y) - H(x), and x -> y is inferred when C_{x->y} < 0.
    Returns the antisymmetric score H(x) - H(y) = -C_{x->y}, which is cdt's predict_proba (IGCI.py:141, 148-151) and pyspi's
    igci value. Positive means x -> y."""
    z = lambda v: (v - v.mean()) / v.std()
    return _spacing_entropy(z(np.asarray(x, float).ravel())) - _spacing_entropy(z(np.asarray(y, float).ravel()))


# -------------------------------------------------------------------------- selftest
def selftest():
    import warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
    import make_data, dcor, ot                               # dcor, ot, dtaidistance: labelled 2nd opinions
    from dtaidistance import dtw as tdtw
    res = []

    def check(label, ok, detail):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")

    rng = np.random.default_rng(7)
    # (1) dcorr
    nd = make_data.load("nonlinear_dependence", 4096)["x"]
    indep = [(0, 2), (0, 3), (1, 2), (1, 3)]
    d01 = dcorr(nd[0], nd[1])
    din = [dcorr(nd[a], nd[b])["dcor2"] for a, b in indep]
    check("dcorr U-dCor x^2 pair > 0.3 (dCor^2 ~0.23)", d01["dcor"] > 0.3, f"dCor {d01['dcor']:.4f}, dCor^2 {d01['dcor2']:.4f}")
    check("dcorr U-dCor^2 independent pairs < 0.05", max(np.abs(din)) < 0.05, f"max |.| = {max(np.abs(din)):.2e}")
    b01 = dcorr(nd[0], nd[1], biased=True)
    check("dcorr biased dCor x^2 pair > 0.3", b01["dcor"] > 0.3, f"{b01['dcor']:.4f}")
    e1 = abs(b01["dcor2"] - dcor.distance_correlation_sqr(nd[0], nd[1]))
    e2 = abs(d01["dcor2"] - dcor.u_distance_correlation_sqr(nd[0], nd[1]))
    check("dcorr vs dcor package (V and U, T=4096) [2nd opinion]", max(e1, e2) < 1e-10, f"{e1:.1e}, {e2:.1e}")
    check("hyppo convention: biased value is sqrt(dcor2)", abs(b01["hyppo"] - np.sqrt(b01["dcor2"])) < 1e-15, "ok")
    # (2) HSIC
    h01 = hsic_biased(nd[0], nd[1])
    hin = [hsic_biased(nd[a], nd[b])["normalised"] for a, b in indep]
    check("hsic normalised x^2 pair > 0.3", h01["normalised"] > 0.3, f"{h01['normalised']:.4f}")
    check("hsic normalised independent pairs < 0.05", max(hin) < 0.05, f"max = {max(hin):.2e}")
    rep = [hyppo_hsic_replica(nd[0], nd[1]), max(abs(hyppo_hsic_replica(nd[a], nd[b])) for a, b in indep)]
    check("hyppo_hsic_replica: x^2 > 0.3, independent < 0.05", rep[0] > 0.3 and rep[1] < 0.05, f"{rep[0]:.4f}, {rep[1]:.2e}")
    a = rng.standard_normal(64)
    check("hsic analytic: normalised hsic(x, x) = 1", abs(hsic_biased(a, a)["normalised"] - 1) < 1e-12, "ok")
    # (3) HHG
    s = make_data.load("nonlinear_dependence", 256)["x"][:, :128]
    t1, t2, t0 = hhg(s[0], s[1]), hhg(s[1], s[0]), hhg(s[0], s[2])
    check("hhg symmetric in (x, y)", abs(t1 - t2) < 1e-9 * t1, f"{t1:.6g} vs {t2:.6g}")
    check("hhg dependent >> independent (T=128)", t1 > 0 and t1 > 3 * t0, f"{t1:.4g} vs {t0:.4g}")
    # DcorrX
    dn = make_data.load("delayed_noise", 256)["x"]
    sm, terms, lag = dcorrx_mehta(dn[1], dn[0], 3)
    sh = hyppo_dcorrx_replica(dn[1], dn[0], 3)[0]
    check("dcorrx_mehta finds the delay-1 lag (x1 lags x0)", lag == 1 and terms[1] > 0.5,
          f"opt_lag={lag}, term={terms[1]:.3f}; hyppo replica stat {sh:.4f} vs Mehta {sm:.4f}")
    # (4) DTW vs dtaidistance
    u, v = dn[0], dn[3]
    e_unc = abs(dtw_dp(u, v) - tdtw.distance(u, v, use_c=False))
    e_sc = abs(dtw_dp(u, v, "sakoe_chiba", 1) - tdtw.distance(u, v, window=2, use_c=False))
    e_sc0 = abs(dtw_dp(u, v, "sakoe_chiba", 0) - np.linalg.norm(u - v))
    check("dtw_dp vs dtaidistance unconstrained, T=256 [2nd opinion]", e_unc < 1e-10, f"{e_unc:.1e}")
    check("dtw_dp sakoe_chiba r=1 vs dtaidistance window=2 [2nd opinion]", e_sc < 1e-10, f"{e_sc:.1e}")
    it = dtw_dp(u, v, "itakura")
    check("dtw: r=0 band = Euclidean; unconstrained <= itakura", e_sc0 < 1e-10 and dtw_dp(u, v) <= it, f"{e_sc0:.1e}")
    check("itakura mask: tslearn docstring example (_masks.py:181-186)",
          np.array_equal(itakura_mask(6, 6, 3).astype(int).sum(0), [1, 3, 4, 4, 3, 1]), "max_slope=3 column counts")
    check("abs-cost 1978 form: x vs shifted copy", abs(dtw_dp([0, 1, 2, 3], [0, 0, 1, 2, 3], squared_cost=False)) < 1e-15, "0")
    # (5) LCSS and soft-DTW
    check("lcss(x, x) = 1; tslearn doc examples 1.0 and 2/3", lcss_dp(u, u) == 1.0 == lcss_dp(u, u, constraint="itakura")
          and lcss_dp([1, 2, 3], [1., 2., 2., 4., 7.]) == 1.0 and abs(lcss_dp([1, 2, 3], [-2., 5., 7.], 3) - 2 / 3) < 1e-15, "ok")
    sx = soft_dtw_dp(u[:64], u[:64])
    check("soft_dtw(x, x) < 0 for gamma=1 (not a divergence)", sx < 0, f"{sx:.4f}")
    g = 1e-3
    s_g, d2 = soft_dtw_dp(u[:64], v[:64], g), dtw_dp(u[:64], v[:64]) ** 2
    check("soft_dtw gamma->0 within analytic bound of dtw^2", d2 - g * np.log(3) * 127 <= s_g <= d2 + 1e-12,
          f"{s_g:.6f} vs {d2:.6f}")
    z0 = rng.standard_normal(24); Xs = rng.standard_normal((2, 24))
    f0, g0 = soft_dtw_barycenter_objective(z0, Xs)
    fd = np.array([(soft_dtw_barycenter_objective(z0 + 1e-6 * e, Xs)[0] - soft_dtw_barycenter_objective(z0 - 1e-6 * e, Xs)[0]) / 2e-6
                   for e in np.eye(24)])
    check("soft-DTW barycenter gradient vs finite differences", np.max(np.abs(fd - g0)) < 1e-5 * max(1, np.abs(g0).max()),
          f"max err {np.max(np.abs(fd - g0)):.1e}")
    # (6) DBA
    z, costs = dba(dn[[0, 3], :128])
    check("dba cost non-increasing", all(np.diff(costs) <= 1e-12), f"{len(costs)} iters, {costs[0]:.3f} -> {costs[-1]:.3f}")
    check("euclidean_barycenter = mean", np.allclose(euclidean_barycenter(dn[:2]), dn[:2].mean(0)), "ok")
    # GWtau
    gx, gy = rng.standard_normal(50), rng.standard_normal(80)
    ra, rb = root_arc_lengths(gx), root_arc_lengths(gy)
    e = abs(wasserstein2_1d(ra, rb) - np.sqrt(ot.wasserstein_1d(ra, rb, p=2)))
    check("W2 unequal sizes vs ot.wasserstein_1d(p=2) [2nd opinion]", e < 1e-10, f"{e:.1e}")
    r1, r2 = gwtau(u, v), gwtau(u, v, include_root=True)
    desc = np.sqrt(np.mean((np.sort(root_arc_lengths(u))[::-1] - np.sort(root_arc_lengths(v))[::-1]) ** 2))
    check("gwtau = pyspi descending-sort RMS; root factor sqrt((n-1)/n)",
          abs(r1 - desc) < 1e-12 and abs(r2 - r1 * np.sqrt(255 / 256)) < 1e-12, f"{r1:.5f}")
    # (7) CCM on tie-free data: Sugihara 2012 eq. 1 with a CHAOTIC driver (r_y = 3.7), y -> x only (beta_xy = 0.1)
    cx, cy = np.empty(1300), np.empty(1300); cx[0], cy[0] = 0.4, 0.2
    for t in range(1299):
        cx[t + 1] = cx[t] * (3.8 - 3.8 * cx[t] - 0.1 * cy[t]); cy[t + 1] = cy[t] * (3.7 - 3.7 * cy[t])
    cx, cy = cx[300:], cy[300:]
    (xy, nxy), (yx, nyx) = simplex_crossmap(cx, cy, 2, return_tied=True), simplex_crossmap(cy, cx, 2, return_tied=True)
    check("ccm chaotic maps: x xmap y (y causes x) exceeds y xmap x by > 0.3, T=1000, E=2, no tied rows",
          xy - yx > 0.3 and nxy == nyx == 0, f"{xy:.3f} vs {yx:.3f}, tied rows {nxy}, {nyx}")
    curve = ccm_skill_curve(cx, cy, 2, [20, 100, 500], n_samples=10)
    check("ccm skill rises with library size (chaotic maps)", curve[-1] > curve[0], np.array2string(curve, precision=3))
    lm = make_data.load("logistic_maps", 256)["x"]          # period-4 driver: 4 distinct values of y
    nt = [simplex_crossmap(lm[0], lm[1], 2, return_tied=True)[1], simplex_crossmap(lm[1], lm[0], 2, return_tied=True)[1]]
    check("ccm tie flag: logistic_maps 'y xmap x' tied on every row, 'x xmap y' on none", nt == [0, 255], f"{nt}")
    lc = make_data.load("logistic_maps_chaotic", 256)["x"]  # x (r=3.8) -> y (r=3.5) only; both aperiodic
    (cxy, ncxy), (cyx, ncyx) = simplex_crossmap(lc[0], lc[1], 2, return_tied=True), simplex_crossmap(lc[1], lc[0], 2, return_tied=True)
    check("ccm direction on logistic_maps_chaotic (x causes y): 'y xmap x' exceeds 'x xmap y' by > 0.3, no tied rows",
          cyx - cxy > 0.3 and ncxy == ncyx == 0, f"{cyx:.3f} vs {cxy:.3f}, tied rows {ncyx}, {ncxy}")
    ls = make_data.load("logistic_maps_sugihara", 256)["x"]  # Fig. 1 caption: x -> y 0.1 stronger than y -> x 0.02
    sxy, syx = simplex_crossmap(ls[0], ls[1], 2), simplex_crossmap(ls[1], ls[0], 2)
    check("ccm direction on logistic_maps_sugihara: 'y xmap x' exceeds 'x xmap y' by > 0.2", syx - sxy > 0.2, f"{syx:.3f} vs {sxy:.3f}")
    pr = simplex_predictions([0., 1., 3., 7.], [0., 1., 3., 7.], 1)[1]   # E=1, 2 neighbours, hand-computed
    w = lambda d1, d2, a, b: (np.exp(-1) * a + np.exp(-d2 / d1) * b) / (np.exp(-1) + np.exp(-d2 / d1))
    check("simplex weights exp(-d/d1), hand case", np.allclose(pr, [w(1, 3, 1, 3), w(1, 2, 0, 3), w(2, 3, 1, 0), w(4, 6, 3, 1)], 1e-14, 0), "ok")
    # (8) causal direction on the anm dataset (x0 -> x1 cubic, x2 -> x3 tanh)
    an = make_data.load("anm", 1024)["x"]
    zs = lambda q: (q - q.mean()) / q.std(ddof=1)
    for c, ef in [(0, 1), (2, 3)]:
        X, Y = zs(an[c]), zs(an[ef])
        r_f, r_b = reci_ref(X, Y), reci_ref(Y, X)
        ig, cd = igci_ref(X, Y), cds_ref(Y, X) - cds_ref(X, Y)
        print(f"INFO  pair {c}->{ef}: reci fwd {r_f:.4f} bwd {r_b:.4f}; igci {ig:+.4f}; cds(y,x)-cds(x,y) {cd:+.4f}")
        check(f"cds picks x{c} -> x{ef}", cd > 0, f"{cd:+.4f}")
        if c == 0:
            check("reci picks x0 -> x1 (cubic)", r_f < r_b, f"{r_f:.4f} < {r_b:.4f}")
    print("INFO  method limits (not kernel failures): igci (Gaussian ref) is wrong-signed on the noisy cubic pair;"
          " reci is a near-tie on the tanh pair")
    check("igci antisymmetric", abs(igci_ref(an[0], an[1]) + igci_ref(an[1], an[0])) < 1e-12, "exact")
    xu = rng.uniform(-1, 1, 5000)
    check("igci noiseless y = x^3, x uniform: positive (x -> y)", igci_ref(xu, xu ** 3) > 0, f"{igci_ref(xu, xu ** 3):+.4f}")
    hn = _spacing_entropy(rng.standard_normal(200000))      # estimator sd ~ sqrt(pi^2/6/n) ~ 0.003
    check("spacing entropy of N(0,1) ~ 0.5 log(2 pi e), n=2e5", abs(hn - 0.5 * np.log(2 * np.pi * np.e)) < 0.015, f"{hn:.4f}")
    Xa, Ya = zs(an[0][:300]), zs(an[1][:300])
    af, ab = anm_replica(Xa, Ya, 1.0, 0.5, 0.1), anm_replica(Ya, Xa, 1.0, 0.5, 0.1)
    check("anm Hoyer form (fixed GP l=0.5, alpha=0.1, T=300) lower in causal direction", af < ab, f"{af:.4f} < {ab:.4f}")
    th = gp_fit_hyperparameters(Xa, Ya)
    from scipy.spatial.distance import pdist, squareform     # scipy primitive, as sklearn RBF uses (kernels.py:1561)
    Ksk = th[0] * np.exp(-0.5 * squareform(pdist(Xa.reshape(-1, 1) / th[1], "sqeuclidean")))
    check("_rbf arithmetic = pdist(x/l) at the fitted theta (bitwise)", np.array_equal(_rbf(Xa, *th), Ksk), "max diff 0")
    print(f"INFO  fitted GP (sklearn-default bounds, alpha=1e-10): constant={th[0]:.4g}, length_scale={th[1]:.3g}")
    p, q = rng.standard_normal((2, 150)); q = q + p ** 2
    sq = lambda w: np.median(_pdist(w)[np.triu_indices(150, 1)] ** 2)
    e = abs(cdt_fast_hsic_gamma(p, q) - 150 * hsic_biased(p, q, gamma=(1 / sq(p), 1 / sq(q)))["hsic"])
    check("cdt hsic replica = m * Gretton HSIC_b at deg^2 = median/2", e < 1e-12, f"{e:.1e}")
    # (9) degenerate inputs
    try:
        dcorr(a[:2], a[2:4]); bad = True
    except ValueError:
        bad = False
    d3 = dcorr(a[:3], a[3:6])
    check("dcorr: U-form n=3 gives 0 (A~ = 0 analytically), n=2 raises", not bad and d3["hyppo"] == 0.0, f"n=3 {d3['hyppo']}")
    check("hsic normalised = 0 for a constant series", hsic_biased(np.ones(30), a[:30])["normalised"] == 0.0, "0")
    yc = (a[:30] - a[:30].min()) / np.ptp(a[:30])
    check("reci constant x -> MSE = var(minmax y) (OLS on [1, 0])", abs(reci_ref(np.ones(30), a[:30]) - yc.var()) < 1e-14, "ok")
    print(f"{sum(res)}/{len(res)} checks passed")
    return all(res)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
