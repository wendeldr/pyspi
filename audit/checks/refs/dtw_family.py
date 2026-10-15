"""dtw_family references (COMPARISON venv): distance.dtw.constraint-{None,itakura,sakoe_chiba},
distance.lcss.constraint-{None,itakura,sakoe_chiba}, distance.softdtw (identifiers softdtw,
softdtw_constraint-itakura, softdtw_constraint-sakoe-chiba).

pyspi (pyspi/statistics/distance.py:174-208) calls tslearn 0.9.0 dtw / lcss / soft_dtw on rows i, j and passes
only global_constraint (soft_dtw: nothing), so radius = 1 (sakoe_chiba), max slope = 2.0 (itakura), eps = 1.0
(lcss), gamma = 1.0 (soft_dtw) are tslearn defaults (_masks.py:339-348, dtw_variants.py:2383, softdtw_variants.py:530).
Undirected.multivariate (base.py:132-138) computes both orders and copies the upper triangle into the lower one.

This script never imports tslearn or pyspi.  Independent references:
  _kernels_distance.dtw_dp / lcss_dp / soft_dtw_dp (anti-diagonal numpy DPs; masks from its sakoe_chiba_mask /
  itakura_mask, which mirror tslearn's mask code and are checked here cell by cell against an exact-integer
  parallelogram / band built in this file), own batched DPs for the Monte Carlo nulls (checked against the kernels
  at start-up), own masked soft-DTW, own Itakura local-slope DP; dtaidistance==2.5.1 (external package, C and
  pure-python paths) for unconstrained DTW and the Sakoe-Chiba band (window = radius + 1).

Run tags (5th field of the run_id; soft-DTW run_ids carry the identifier as a 6th field):
  dp            every ORDERED pair computed separately (both orientations) on the rows pyspi receives (raw / own
                ddof=1 z-score / own LS detrend + z-score), exact rtol 1e-10 atol 1e-12 (soft-DTW: tolerance
                rtol 1e-10 atol 1e-10). LCSS constraint units: the reference is the MASKED LCSS (the documented
                quantity); soft-DTW constraint identifiers: unconstrained soft-DTW (what the code computes)
  unmasked      LCSS constraint units: exact vs UNconstrained LCSS
  masked        soft-DTW constraint identifiers: tolerance vs MASKED soft-DTW (documented quantity; KNOWN bug 3)
  ident         DTW/LCSS/softdtw identifier: probe rows (x0, x0, x1) z-scored: f(x, x) on the off-diagonal, exact
                (lcss_constraint-*: vs the UNconstrained LCSS, what the numpy path computes);
                soft-DTW constraint identifiers: SAME-LINEAGE bit identity (rtol 0, atol 0) with pyspi softdtw
  same          LCSS constraint units: SAME-LINEAGE bit identity (rtol 0, atol 0) with pyspi lcss
  unmirrored    probe Directed.multivariate (no mirroring; tslearn computes each order) vs dp reference; extra keys
                give max|B - B.T| of pyspi's unmirrored matrix and of the reference
  scaleinv      scaled_offset (pipeline) vs reference on the z-scored bivariate_gaussian draw (independent-impl)
  scaleinv-self scaled_offset (pipeline) vs pyspi's own bivariate_gaussian pipeline matrix (same-lineage)
  scale-raw     scaled_offset on the kernel path (no z-score) vs reference on the raw rows; extra keys compare
                with the z-scored value (scale sensitivity when normalisation is off)
  trend         trending (pipeline) vs reference on z-scored trending rows; extra key: change vs the untrended draw
  detrend       trending (pipeline-detrend) vs reference on z-score(LS-detrend(rows))
  canon         DTW: tolerance (rtol 1e-8) vs the Sakoe-Chiba 1978 symmetric form (|a-b| cost, diagonal weight 2,
                divided by n+m); expected to differ (convention); extra keys hold both values and the rank agreement
  bounds        DTW optimality: ||x-y||_2 >= dtw >= sqrt(max(sum_i min_j c_ij, sum_j min_i c_ij)) and
                dtw_None <= dtw_itakura, dtw_None <= dtw_sakoe_chiba (pyspi matrices of the dp runs)
  delay         DTW qualitative: delayed_noise ratios dtw(z0, zk)/||z0 - zk|| for delays 1, 3, 5
  dtai          DTW external-pkg: dtaidistance dtw.distance (window None / 2), tolerance rtol 1e-10
  mask          probe tslearn compute_mask vs the exact-integer parallelogram / band (exact)
  local-slope   DTW itakura qualitative: pyspi <= DTW with a symmetric local slope rule (no two consecutive (0,1) and
                no two consecutive (1,0) moves; slopes 1/2..2; not Itakura 1975's asymmetric rule) inside the same
                parallelogram
  null          statistical: iid_gaussian pipeline vs Monte Carlo mean for independent z-scored Gaussian pairs;
                se = MC sd * z_{1-0.025/m} / 2 (Bonferroni over the m distinct pairs)
  null-masked   LCSS constraint units: same null but for the MASKED LCSS (documented quantity)
  null-neg      qualitative: rows (x0, -x0, x1, x2): sanity ordering f(x, -x) less similar than f(x, independent)
  range         LCSS qualitative: bivariate_gaussian rho 0.8 pair scores above rho 0.3 pair
  strict        LCSS exact vs the strict '< eps' rule of Vlachos et al.
  torch         LCSS constraint units: tslearn torch backend (probe) vs masked reference (exact)
  div           soft-DTW qualitative: raw value vs the soft-DTW divergence computed from scratch
  gamma-<g>     soft-DTW: tslearn soft_dtw(gamma=g) (probe) vs soft_dtw_dp(gamma=g), tolerance rtol 1e-9
  gbound-<g>    soft-DTW optimality: dtw^2 - g log(3) (n+m-1) <= soft_dtw_g <= dtw^2 (hard dtw from the kernel)
  near / offset soft-DTW: near-identical rows / +1e4 offset on the kernel path vs direct-difference reference
  rawcall       spi.bivariate(x0, x1), (x1, x0) on raw arrays vs f(zscore(first), raw second) (exact)

Writes out/ref/<run_id>.npz per the compare.py contract.
"""
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np
from scipy import stats

import make_data
import _kernels_distance as K
from dtaidistance import dtw as dtai_dtw
import dtaidistance

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
SELF = "audit/checks/refs/dtw_family.py"
KMOD = "audit/checks/refs/_kernels_distance.py"
MC_SEED = 20261015
MC_N = 400


# ------------------------------------------------------------------ primitives
def zscore(X):
    """ddof=1 z-score per row (pyspi/data.py:195)."""
    X = np.asarray(X, float)
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)


def detrend(X):
    """Least-squares removal of a + b t per row (what scipy.signal.detrend type='linear' does, data.py:187)."""
    X = np.asarray(X, float)
    T = X.shape[1]
    A = np.column_stack([np.ones(T), np.arange(T, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)
    return X - (A @ coef).T


def preprocess(X, path):
    if path == "kernel":
        return np.asarray(X, float)
    if path == "pipeline":
        return zscore(X)
    if path == "pipeline-detrend":
        return zscore(detrend(X))
    raise ValueError(path)


def derive(kind, x):
    """Mirror of probes/dtw_family_probes.py derive()."""
    x = np.asarray(x, dtype=np.float64)
    if kind == "null-neg":
        return np.vstack([x[0], -x[0], x[1], x[2]])
    if kind == "near":
        return np.vstack([x[0], x[0] + 1e-3 * x[1], x[0] + 1e-6 * x[1]])
    if kind == "offset":
        return np.vstack([x[0] + 1e4, x[1] + 1e4, x[2] + 1e4])
    if kind == "ident":
        return np.vstack([x[0], x[0], x[1]])
    raise ValueError(kind)


def exact_itakura(T):
    """Itakura parallelogram for equal lengths, slope 2, exact integer arithmetic (0-based (i, j)):
    j/2 <= i <= 2j and (T-1-j)/2 <= T-1-i <= 2(T-1-j), i.e. 2i >= j, i <= 2j, i >= 2j-(T-1), 2i <= (T-1)+j."""
    i, j = np.indices((T, T))
    return (2 * i >= j) & (i <= 2 * j) & (i >= 2 * j - (T - 1)) & (2 * i <= (T - 1) + j)


def exact_band(T, r=1):
    i, j = np.indices((T, T))
    return np.abs(i - j) <= r


def own_mask(T, con):
    if con is None:
        return np.ones((T, T), bool)
    return exact_itakura(T) if con == "itakura" else exact_band(T)


# ------------------------------------------------------------------ own DPs
def _cells(n, m, d):
    i = np.arange(max(1, d - m), min(n, d - 1) + 1)
    return i, d - i


def abs1_dtw(x, y, M):
    """|a-b| cost, weight-1 steps, unnormalised (the 'abs-cost unweighted' DP of the trace note)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, m = len(x), len(y)
    C = np.abs(x[:, None] - y[None, :])
    R = np.full((n + 1, m + 1), np.inf); R[0, 0] = 0.0
    for d in range(2, n + m + 1):
        i, j = _cells(n, m, d)
        v = C[i - 1, j - 1] + np.minimum(np.minimum(R[i - 1, j], R[i, j - 1]), R[i - 1, j - 1])
        R[i, j] = np.where(M[i - 1, j - 1], v, np.inf)
    return float(R[n, m])


def local_slope_dtw(x, y, M):
    """sqrt(min sum (x_i-y_j)^2) over paths with steps (1,0), (0,1), (1,1), no two consecutive (0,1) moves and no
    two consecutive (1,0) moves, restricted to the admissible cells M.  States: last move H (0,1), V (1,0), D (1,1) or
    start.  This is ONE symmetric local rule: local slopes 1/2..2, the slope range of the Sakoe-Chiba 1978 P = 1
    pattern, except that an H-V / V-H corner is allowed here and steps carry weight 1.  It is not Itakura 1975's
    asymmetric rule (test index advances 1 per step, reference index 0/1/2 with no two consecutive 0 advances), which
    is a different DP; the gaps reported by the 'local-slope' runs are specific to the rule implemented here."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, m = len(x), len(y)
    C = (x[:, None] - y[None, :]) ** 2
    inf = np.inf
    H = np.full((n + 1, m + 1), inf); V = H.copy(); D = H.copy()
    D[1, 1] = C[0, 0] if M[0, 0] else inf      # start cell: no previous move, treated as D
    for d in range(3, n + m + 1):
        i, j = _cells(n, m, d)
        c = C[i - 1, j - 1]
        h = c + np.minimum(V[i, j - 1], D[i, j - 1])
        v = c + np.minimum(H[i - 1, j], D[i - 1, j])
        dd = c + np.minimum(np.minimum(H[i - 1, j - 1], V[i - 1, j - 1]), D[i - 1, j - 1])
        ok = M[i - 1, j - 1]
        H[i, j] = np.where(ok, h, inf); V[i, j] = np.where(ok, v, inf); D[i, j] = np.where(ok, dd, inf)
    return float(np.sqrt(min(H[n, m], V[n, m], D[n, m])))


def _softmin(a, b, c, g):
    """-g log(e^{-a/g} + e^{-b/g} + e^{-c/g}) via logsumexp; +inf when all three are +inf."""
    S = np.stack([a, b, c])
    allinf = np.all(np.isinf(S), axis=0)
    with np.errstate(invalid="ignore"):
        out = -g * np.logaddexp(np.logaddexp(-S[0] / g, -S[1] / g), -S[2] / g)
    return np.where(allinf, np.inf, out)


def soft_dtw_masked(x, y, M, gamma=1.0):
    """Soft-DTW (Cuturi & Blondel 2017, Alg. 1) restricted to admissible cells M (masked cells +inf): the quantity the
    documentation names softdtw_constraint-*; with M all True it equals soft_dtw_dp."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, m = len(x), len(y)
    D = (x[:, None] - y[None, :]) ** 2
    R = np.full((n + 1, m + 1), np.inf); R[0, 0] = 0.0
    for d in range(2, n + m + 1):
        i, j = _cells(n, m, d)
        v = D[i - 1, j - 1] + _softmin(R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma)
        R[i, j] = np.where(M[i - 1, j - 1], v, np.inf)
    return float(R[n, m])


def batch_dp(X, Y, kind, M, gamma=1.0, eps=1.0):
    """Batched DP over B pairs (rows of X, Y; equal length T) keeping two anti-diagonals.
    kind: 'dtw' -> sqrt(R[T,T]); 'lcss' -> L[T,T]/T (masked cells 0, <= eps); 'sdtw' -> R[T,T] (gamma)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    B, n = X.shape
    m = Y.shape[1]
    fill = 0.0 if kind == "lcss" else np.inf
    p2 = np.full((B, n + 1), fill)          # diagonal d-2, indexed by i
    p1 = np.full((B, n + 1), fill)          # diagonal d-1
    if kind != "lcss":
        p2[:, 0] = 0.0                       # R[0,0] = 0 (diagonal 0)
    for d in range(2, n + m + 1):
        i, j = _cells(n, m, d)
        cur = np.full((B, n + 1), fill)
        diff = X[:, i - 1] - Y[:, j - 1]
        ok = M[i - 1, j - 1]
        if kind == "dtw":
            v = diff ** 2 + np.minimum(np.minimum(p1[:, i - 1], p1[:, i]), p2[:, i - 1])
            cur[:, i] = np.where(ok, v, np.inf)
        elif kind == "sdtw":
            v = diff ** 2 + _softmin(p1[:, i - 1], p2[:, i - 1], p1[:, i], gamma)
            cur[:, i] = np.where(ok, v, np.inf)
        else:
            hit = np.abs(diff) <= eps
            v = np.where(hit, 1 + p2[:, i - 1], np.maximum(p1[:, i], p1[:, i - 1]))
            cur[:, i] = np.where(ok, v, 0.0)
        p2, p1 = p1, cur
    out = p1[:, n]
    if kind == "dtw":
        return np.sqrt(out)
    if kind == "lcss":
        return out / min(n, m)
    return out


# ------------------------------------------------------------------ pair functions (cached)
_ROWS = {}


def rows_key(key, X):
    _ROWS[key] = np.asarray(X, float)
    return key


def pair_fn(fname, con, x, y, extra=None):
    if fname == "dtw":
        return K.dtw_dp(x, y, con)
    if fname == "canon":
        return K.dtw_dp(x, y, con, squared_cost=False)
    if fname == "abs1":
        return abs1_dtw(x, y, own_mask(len(x), con))
    if fname == "local":
        return local_slope_dtw(x, y, own_mask(len(x), con))
    if fname == "lcss":
        return K.lcss_dp(x, y, 1.0, con)
    if fname == "lcss_strict":
        return K.lcss_dp(x, y, 1.0, con, strict=True)
    if fname == "sdtw":
        return K.soft_dtw_dp(x, y, 1.0 if extra is None else extra)
    if fname == "sdtw_masked":
        return soft_dtw_masked(x, y, own_mask(len(x), con), 1.0)
    if fname == "dtai":
        w = None if con is None else 2          # dtaidistance window w admits |i-j| < w; tslearn radius 1 -> w = 2
        return dtai_dtw.distance(np.ascontiguousarray(x), np.ascontiguousarray(y), window=w, use_c=True)
    if fname == "dtai_py":
        w = None if con is None else 2
        return dtai_dtw.distance(np.ascontiguousarray(x), np.ascontiguousarray(y), window=w, use_c=False)
    raise ValueError(fname)


@functools.lru_cache(maxsize=None)
def pair_cached(fname, con, key, i, j, extra=None):
    X = _ROWS[key]
    return pair_fn(fname, con, X[i], X[j], extra)


def matrix(fname, con, key, extra=None):
    """Every ordered pair (i, j), i != j, computed separately (no mirroring); NaN diagonal."""
    P = _ROWS[key].shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = pair_cached(fname, con, key, i, j, extra)
    return A


# ------------------------------------------------------------------ io helpers
def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_REF / f"{rid}.npz", **{k: np.asarray(v) for k, v in kw.items()})


def load_pyspi(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], float)


def load_raw(ds, T, processes=None):
    x = np.asarray(make_data.load(ds, T)["x"], float)
    return x if processes is None else x[list(processes)]


def max_asym(A):
    if A is None:
        return np.nan
    d = np.abs(A - A.T)
    return float(np.nanmax(d)) if np.isfinite(d).any() else np.nan


def upper(A):
    return A[np.triu_indices(A.shape[0], 1)]


def tag_of(run):
    return run["run_id"].split("|")[4]


def ident_of(run):
    p = run["run_id"].split("|")
    return p[5] if len(p) > 5 else run.get("identifier")


def family_of(unit):
    return "dtw" if ".dtw." in unit else ("lcss" if ".lcss." in unit else "sdtw")


def con_of(run):
    """Constraint the CODE path could apply (from the unit id / identifier)."""
    s = run["unit_id"] + "|" + (ident_of(run) or "")
    if "itakura" in s:
        return "itakura"
    if "sakoe" in s:
        return "sakoe_chiba"
    return None


def sibling(run, unit=None, ds=None, T=None, path=None, tag="dp", suffix=None):
    parts = [unit or run["unit_id"], ds or run["dataset"], str(T or run["T"]), path or run["path"], tag]
    if suffix:
        parts.append(suffix)
    return "|".join(parts)


def zcrit(m):
    return float(stats.norm.ppf(1 - 0.025 / m))


# ------------------------------------------------------------------ Monte Carlo nulls
@functools.lru_cache(maxsize=None)
def mc_null(kind, con, T, n=MC_N):
    rng = np.random.default_rng([MC_SEED, T, {"dtw": 1, "lcss": 2, "sdtw": 3}[kind], {None: 0, "itakura": 1, "sakoe_chiba": 2}[con]])
    M = own_mask(T, con)
    vals = []
    for start in range(0, n, 100):
        b = min(100, n - start)
        X = zscore(rng.standard_normal((b, T)))
        Y = zscore(rng.standard_normal((b, T)))
        vals.append(batch_dp(X, Y, kind, M))
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std(ddof=1)), v


def startup_selftest():
    """Batched DPs and own masked soft-DTW vs the kernels on two short pairs (assert-level)."""
    rng = np.random.default_rng(7)
    X, Y = rng.standard_normal((2, 50)), rng.standard_normal((2, 50))
    for con in (None, "itakura", "sakoe_chiba"):
        M = own_mask(50, con)
        assert np.array_equal(M, K._mask(50, 50, con, 1, 2.0)), con
        b = batch_dp(X, Y, "dtw", M)
        assert np.allclose(b, [K.dtw_dp(X[k], Y[k], con) for k in range(2)], rtol=1e-13, atol=0), con
        b = batch_dp(X, Y, "lcss", M)
        assert np.array_equal(b, [K.lcss_dp(X[k], Y[k], 1.0, con) for k in range(2)]), con
        b = batch_dp(X, Y, "sdtw", M)
        assert np.allclose(b, [soft_dtw_masked(X[k], Y[k], M) for k in range(2)], rtol=1e-12, atol=1e-12), con
    s1, s2 = soft_dtw_masked(X[0], Y[0], own_mask(50, None)), K.soft_dtw_dp(X[0], Y[0])
    assert abs(s1 - s2) <= 1e-11 * abs(s2), (s1, s2)


# ------------------------------------------------------------------ handlers
def detail(fname, con):
    d = {"dtw": f"{KMOD}:dtw_dp(constraint={con!r}, radius 1, slope 2.0; squared cost, sqrt)",
         "lcss": f"{KMOD}:lcss_dp(eps=1.0, constraint={con!r}, <= rule, masked cells 0, / min(n,m))",
         "sdtw": f"{KMOD}:soft_dtw_dp(gamma=1.0) (Cuturi & Blondel 2017 Alg. 1)",
         "sdtw_masked": f"{SELF}:soft_dtw_masked(constraint={con!r}, gamma=1.0)"}
    return d[fname]


def handle(run):
    rid, unit, tag, path = run["run_id"], run["unit_id"], tag_of(run), run["path"]
    T, ds = run["T"], run["dataset"]
    fam = family_of(unit)
    con = con_of(run)
    S = load_pyspi(rid)
    base = dict(pyspi_max_asym=max_asym(S) if S is not None and S.ndim == 2 and S.shape[0] == S.shape[1] and S.shape[0] < 50 else np.nan)
    fname = {"dtw": "dtw", "lcss": "lcss", "sdtw": "sdtw"}[fam]
    exact = dict(criterion="exact", rtol=1e-10, atol=1e-12)
    if fam == "sdtw":
        exact = dict(criterion="tolerance", rtol=1e-10, atol=1e-10)
    # the constraint the reference applies for the plain 'dp'-type checks
    ref_con = con
    if fam == "sdtw":
        ref_con = None                                  # tslearn soft_dtw has no constraint argument
    raw = load_raw(ds, T, run.get("processes"))

    if tag in ("dp", "unmirrored", "unmasked"):
        rc = None if tag == "unmasked" else ref_con
        if fam == "lcss" and tag == "unmirrored":
            rc = None                                   # symmetry measurement on what the numpy path computes
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix(fname, rc, key)
        notes = f"{fname} reference, constraint {rc!r}, rows = {path} preprocessing of {ds} T={T}; every ordered pair computed separately"
        if fam == "lcss" and tag == "dp" and con is not None:
            Vu = matrix("lcss", None, key)
            n_diff = int(np.sum(np.abs(V - Vu) > 0))
            base.update(masked_vs_unmasked_n_differ=n_diff, masked_vs_unmasked_max=float(np.nanmax(np.abs(V - Vu))),
                        pyspi_eq_unmasked=int(S is not None and np.array_equal(S, Vu, equal_nan=True)))
            notes += "; MASKED LCSS = the documented constrained quantity; a fail here means the constraint is not applied"
        if fam == "dtw" and con == "itakura":
            base.update(kernel_mask_eq_exact=int(np.array_equal(K.itakura_mask(raw.shape[1], raw.shape[1]), exact_itakura(raw.shape[1]))))
        if fam == "dtw" and con == "sakoe_chiba":
            base.update(kernel_mask_eq_exact=int(np.array_equal(K.sakoe_chiba_mask(raw.shape[1], raw.shape[1]), exact_band(raw.shape[1]))))
        base.update(ref_max_asym=max_asym(V), ref_01=float(V[0, 1]), pyspi_01=float(S[0, 1]) if S is not None else np.nan)
        save(rid, value=V, reference_kind="independent-impl", reference_detail=f"{SELF}:matrix -> " + detail(fname, rc),
             notes=notes, **exact, **base)
        return

    if tag == "masked":                                 # soft-DTW constraint identifiers vs documented masked soft-DTW
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix("sdtw_masked", con, key)
        Vu = matrix("sdtw", None, key)
        save(rid, value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
             reference_detail=f"{SELF}:soft_dtw_masked (Cuturi & Blondel 2017 Alg. 1 restricted to the {con} mask)",
             notes="documented quantity (supplement 1.2.9: 'includes the same global constraints as DTW'); fail = constraint ignored, KNOWN_ISSUES bug 3",
             masked_minus_unconstrained_min=float(np.nanmin(V - Vu)), masked_minus_unconstrained_max=float(np.nanmax(V - Vu)), **base)
        return

    if tag == "ident" and fam == "sdtw" and ident_of(run) != "softdtw":
        P = load_pyspi(sibling(run, tag="dp", suffix="softdtw"))
        save(rid, value=P, criterion="exact", rtol=0.0, atol=0.0, reference_kind="same-lineage",
             reference_detail=f"pyspi softdtw matrix ({sibling(run, tag='dp', suffix='softdtw')})",
             notes="bit identity of the constraint identifier with the unconstrained identifier (KNOWN_ISSUES bug 3)",
             bit_identical=int(S is not None and P is not None and np.array_equal(S, P, equal_nan=True)), **base)
        return

    if tag == "ident":
        Xd = preprocess(derive("ident", raw), path)
        key = rows_key((ds, T, path, "ident"), Xd)
        rc = None if fam == "lcss" else ref_con          # lcss_constraint-*: compare with what the numpy path computes
        V = matrix(fname, rc, key)
        save(rid, value=V, **(exact if fam != "dtw" else dict(criterion="exact", rtol=1e-10, atol=1e-12)),
             reference_kind="independent-impl" if fam != "dtw" else "analytic",
             reference_detail=f"{SELF}:matrix on rows (x0, x0, x1) -> " + detail(fname, rc),
             notes="A[0,1] = f(x, x): dtw 0, lcss 1, soft-DTW its (negative) self value",
             pyspi_self=float(S[0, 1]), ref_self=float(V[0, 1]), **base)
        return

    if tag == "same":
        P = load_pyspi(sibling(run, unit="distance.lcss.constraint-None"))
        save(rid, value=P, criterion="exact", rtol=0.0, atol=0.0, reference_kind="same-lineage",
             reference_detail=f"pyspi lcss matrix ({sibling(run, unit='distance.lcss.constraint-None')})",
             notes="bit identity with unconstrained lcss (ignored mask, tslearn dtw_variants.py:2259)",
             bit_identical=int(S is not None and P is not None and np.array_equal(S, P, equal_nan=True)), **base)
        return

    if tag in ("scaleinv", "scaleinv-self"):
        if tag == "scaleinv-self":
            sib = sibling(run, ds="bivariate_gaussian", tag="dp", suffix="softdtw" if fam == "sdtw" else None)
            P = load_pyspi(sib)
            save(rid, value=P, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="same-lineage",
                 reference_detail=f"pyspi matrix on bivariate_gaussian ({sib})",
                 notes="z-scoring removes per-process scale and offset (scaled_offset = scale*D2 + offset)", **base)
            return
        rc = None if fam == "lcss" else ref_con
        key = rows_key(("bivariate_gaussian", T, "pipeline", "rows"), zscore(load_raw("bivariate_gaussian", T)))
        V = matrix(fname, rc, key)
        save(rid, value=V, **exact, reference_kind="independent-impl",
             reference_detail=f"{SELF}:matrix on z-scored bivariate_gaussian -> " + detail(fname, rc),
             notes="scaled_offset through the pipeline vs the reference on the z-scored unscaled draw", **base)
        return

    if tag == "scale-raw":
        rc = None if fam == "lcss" else ref_con
        key = rows_key((ds, T, "kernel", "rows"), raw)
        V = matrix(fname, rc, key)
        kz = rows_key((ds, T, "pipeline", "rows"), zscore(raw))
        Z = matrix(fname, rc, kz)
        base.update(ratio_raw_over_z_min=float(np.nanmin(V / Z)), ratio_raw_over_z_max=float(np.nanmax(V / Z)),
                    raw_01=float(V[0, 1]), z_01=float(Z[0, 1]), raw_23=float(V[2, 3]), z_23=float(Z[2, 3]))
        save(rid, value=V, **exact, reference_kind="independent-impl",
             reference_detail=f"{SELF}:matrix on raw scaled_offset rows -> " + detail(fname, rc),
             notes="kernel path (normalise=False): the SPI sees raw units; extra keys compare with the z-scored value", **base)
        return

    if tag in ("trend", "detrend"):
        rc = None if fam == "lcss" else ref_con
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix(fname, rc, key)
        kb = rows_key(("bivariate_gaussian", T, "pipeline", "rows"), zscore(load_raw("bivariate_gaussian", T)))
        Bm = matrix(fname, rc, kb)
        base.update(max_abs_change_vs_untrended=float(np.nanmax(np.abs(V - Bm))),
                    max_rel_change_vs_untrended=float(np.nanmax(np.abs(V - Bm) / np.abs(Bm))))
        save(rid, value=V, **exact, reference_kind="independent-impl",
             reference_detail=f"{SELF}:matrix on {path} rows of trending -> " + detail(fname, rc),
             notes="trending = D2 + slope*t; extra keys: change against the z-scored untrended draw", **base)
        return

    if tag == "canon":
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        Cn = matrix("canon", con, key)
        A1 = matrix("abs1", con, key)
        Sq = matrix("dtw", con, key)
        u = upper(S)
        rho = float(stats.spearmanr(u, upper(Cn)).statistic) if len(u) > 2 else np.nan
        save(rid, value=Cn, criterion="tolerance", rtol=1e-8, atol=1e-12, reference_kind="analytic",
             reference_detail=f"{KMOD}:dtw_dp(squared_cost=False): Sakoe & Chiba 1978 symmetric form, P=0, g/(n+m), |a-b| cost",
             notes="expected to differ: tslearn returns sqrt(min sum (a-b)^2) with weight-1 steps and no normalisation",
             canon_01=float(Cn[0, 1]), abs1_01=float(A1[0, 1]), sqrt_sq_ref_01=float(Sq[0, 1]),
             pyspi_01=float(S[0, 1]), spearman_pyspi_vs_canon=rho,
             ratio_pyspi_over_canon_min=float(np.nanmin(S / Cn)), ratio_pyspi_over_canon_max=float(np.nanmax(S / Cn)), **base)
        return

    if tag == "bounds":
        Xk = preprocess(raw, path)
        P = Xk.shape[0]
        E = np.full((P, P), np.nan); LB = E.copy()
        for i in range(P):
            for j in range(P):
                if i != j:
                    c = (Xk[i][:, None] - Xk[j][None, :]) ** 2
                    E[i, j] = np.linalg.norm(Xk[i] - Xk[j])
                    LB[i, j] = np.sqrt(max(c.min(1).sum(), c.min(0).sum()))
        off = ~np.eye(P, dtype=bool)
        up_ok = bool(np.all(S[off] <= E[off] * (1 + 1e-12)))
        lo_ok = bool(np.all(S[off] >= LB[off] * (1 - 1e-12)))
        order_ok = True
        ords = {}
        for other in ("None", "itakura", "sakoe_chiba"):
            O = load_pyspi(sibling(run, unit=f"distance.dtw.constraint-{other}", tag="dp"))
            ords[other] = O
        if con is None:
            for other in ("itakura", "sakoe_chiba"):
                order_ok &= bool(np.all(S[off] <= ords[other][off] * (1 + 1e-12)))
        else:
            order_ok &= bool(np.all(ords["None"][off] <= S[off] * (1 + 1e-12)))
        save(rid, value=E, criterion="optimality", pass_flag=int(up_ok and lo_ok and order_ok), reference_kind="analytic",
             reference_detail="identity path admissible => dtw <= ||x-y||_2; every row/column visited => dtw >= sqrt(max(sum_i min_j c, sum_j min_i c)); nested admissible sets => dtw_None <= dtw_constrained",
             notes="pass = all three bound families hold on every ordered pair",
             upper_ok=int(up_ok), lower_ok=int(lo_ok), order_ok=int(order_ok),
             ratio_dtw_over_euclid_min=float(np.nanmin(S / E)), ratio_dtw_over_euclid_max=float(np.nanmax(S / E)),
             ratio_dtw_over_lb_min=float(np.nanmin(S / LB)), **base)
        return

    if tag == "delay":
        Z = preprocess(raw, path)
        r = [float(S[0, k] / np.linalg.norm(Z[0] - Z[k])) for k in (1, 2, 3)]
        if con == "sakoe_chiba":
            ok = r[0] < 0.25 and r[1] > 0.5 and r[2] > 0.5
            expect = "radius 1: delay 1 absorbed (ratio < 0.25), delays 3 and 5 not (ratio > 0.5)"
        else:
            ok = all(v < 0.25 for v in r)
            expect = "all delays (1, 3, 5) absorbed by warping (ratio < 0.25)"
        save(rid, value=np.asarray(r), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
             reference_detail="delayed_noise truth: x_k = x0 delayed by (0,1,3,5) + 0.1 noise; ratio dtw/||z0 - zk||",
             notes="expectation: " + expect, ratio_d1=r[0], ratio_d3=r[1], ratio_d5=r[2], **base)
        return

    if tag == "dtai":
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix("dtai", con, key)
        extra = {}
        if T <= 256:
            Vp = matrix("dtai_py", con, key)
            extra["dtai_c_vs_python_max_abs"] = float(np.nanmax(np.abs(V - Vp)))
        X = _ROWS[key]
        w1 = dtai_dtw.distance(X[0], X[1], window=1, use_c=True)
        extra.update(dtai_window1_minus_euclid=float(w1 - np.linalg.norm(X[0] - X[1])),
                     dtai_window3_01=float(dtai_dtw.distance(X[0], X[1], window=3, use_c=True)))
        save(rid, value=V, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="external-pkg",
             reference_detail=f"dtaidistance=={dtaidistance.__version__} dtw.distance(use_c=True, window={'None' if con is None else 2}) (window w admits |i-j| < w)",
             notes="tslearn radius r <-> dtaidistance window r+1", **extra, **base)
        return

    if tag == "mask":
        E = exact_itakura(T) if con == "itakura" else exact_band(T)
        save(rid, value=E.astype(float), criterion="exact", rtol=0.0, atol=0.0, reference_kind="independent-impl",
             reference_detail=f"{SELF}:{'exact_itakura (integer inequalities)' if con == 'itakura' else 'exact_band |i-j| <= 1'}",
             notes="tslearn compute_mask(T, T) vs exact-arithmetic mask, cell by cell",
             n_cells_ref=int(E.sum()), frac_cells=float(E.mean()), n_cells_tslearn=int(S.sum()),
             n_cells_differ=int((S.astype(bool) != E).sum()))
        return

    if tag == "local-slope":
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix("local", con, key)
        off = ~np.eye(V.shape[0], dtype=bool)
        ok = bool(np.all(np.isfinite(V[off])) and np.all(S[off] <= V[off] * (1 + 1e-12)))
        save(rid, value=V, criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
             reference_detail=f"{SELF}:local_slope_dtw (symmetric rule: no two consecutive (0,1) / (1,0) moves, slopes 1/2..2; not Itakura 1975 asymmetric rule) inside the slope-2 parallelogram",
             notes="pyspi (global parallelogram only) <= local-slope-constrained DTW on every pair; extra keys give the gap",
             rel_gap_min=float(np.nanmin(V[off] / S[off] - 1)), rel_gap_max=float(np.nanmax(V[off] / S[off] - 1)),
             rel_gap_median=float(np.nanmedian(V[off] / S[off] - 1)), **base)
        return

    if tag in ("null", "null-masked"):
        kind = fam
        mc_con = con if (fam == "dtw" or tag == "null-masked") else None
        mu, sd, mcv = mc_null(kind, mc_con, T)
        P = S.shape[0]
        m = P * (P - 1) // 2
        V = np.full_like(S, mu); np.fill_diagonal(V, np.nan)
        se = np.full_like(S, sd * zcrit(m) / 2.0)
        u = upper(S)
        save(rid, value=V, criterion="statistical", se=se, reference_kind="analytic",
             reference_detail=f"{SELF}:mc_null({kind}, constraint={mc_con!r}, T={T}, N={MC_N}) independent z-scored N(0,1) pairs; batched DP checked against the kernels",
             notes=f"pass if every pair within z_(1-0.025/{m}) MC sd of the MC mean",
             mc_mean=mu, mc_sd=sd, mc_frac_zero=float(np.mean(mcv == 0)), mc_q01=float(np.quantile(mcv, 0.01)), mc_q99=float(np.quantile(mcv, 0.99)), pyspi_mean=float(u.mean()), pyspi_min=float(u.min()), pyspi_max=float(u.max()), **base)
        return

    if tag == "null-neg":
        Xd = preprocess(derive("null-neg", raw), path)
        key = rows_key((ds, T, path, "null-neg"), Xd)
        rc = None if fam == "lcss" else ref_con
        V = matrix(fname, rc, key)
        neg = float(S[0, 1]); ind = float(np.mean([S[0, 2], S[0, 3], S[1, 2], S[1, 3]]))
        if fam == "lcss":
            ok = neg < min(S[0, 2], S[0, 3])
            sense = "similarity: lcss(x, -x) < lcss(x, independent)"
        else:
            ok = neg > max(S[0, 2], S[0, 3])
            sense = "distance: f(x, -x) > f(x, independent)"
        e_neg = float(np.linalg.norm(Xd[0] - Xd[1])); e_ind = float(np.mean([np.linalg.norm(Xd[a] - Xd[b]) for a, b in ((0, 2), (0, 3), (1, 2), (1, 3))]))
        save(rid, value=V, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
             reference_detail=f"sanity ordering ({sense}); rows (x0, -x0, x1, x2) z-scored; value = {fname} reference matrix",
             notes="pass = the anti-correlated pair is ranked less similar than the independent pairs; rel_gap = f(x,-x)/mean f(indep) - 1",
             f_neg=neg, f_indep_mean=ind, rel_gap=float(neg / ind - 1), euclid_rel_gap=float(e_neg / e_ind - 1),
             ref_vs_pyspi_max_abs=float(np.nanmax(np.abs(V - S))), **base)
        return

    if tag == "range":
        mu, sd, _ = mc_null("lcss", None, T)
        ok = S[2, 3] > S[0, 1]
        save(rid, value=np.asarray([S[0, 1], S[2, 3]]), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
             reference_detail="bivariate_gaussian truth rho = (0.3, 0.8) for pairs (0,1), (2,3); null MC mean for comparison",
             notes="pass = rho 0.8 pair scores above rho 0.3 pair; extra keys: values in null-SD units",
             lcss_rho03=float(S[0, 1]), lcss_rho08=float(S[2, 3]), null_mean=mu, null_sd=sd,
             z_rho03=float((S[0, 1] - mu) / sd), z_rho08=float((S[2, 3] - mu) / sd), **base)
        return

    if tag == "strict":
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix("lcss_strict", None, key)
        save(rid, value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
             reference_detail=f"{KMOD}:lcss_dp(strict=True): Vlachos et al. 2002 '< eps' match rule",
             notes="<= vs < differs only on exact ties |a-b| == eps", **base)
        return

    if tag == "torch":
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix("lcss", con, key)
        Vu = matrix("lcss", None, key)
        save(rid, value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
             reference_detail=f"{KMOD}:lcss_dp(constraint={con!r}) masked reference; the compared value is tslearn's torch backend",
             notes="checks that tslearn's torch branch (dtw_variants.py:2306, 'if mask[...]') honours the mask",
             torch_eq_unmasked=int(np.array_equal(S, Vu, equal_nan=True)), **base)
        return

    if tag == "div":
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        X = _ROWS[key]
        P = X.shape[0]
        self_v = np.array([K.soft_dtw_dp(X[i], X[i]) for i in range(P)])
        Dv = S - 0.5 * (self_v[:, None] + self_v[None, :])
        np.fill_diagonal(Dv, np.nan)
        u = upper(S)
        frac_neg = float(np.mean(u < 0))
        rho = float(stats.spearmanr(u, upper(Dv)).statistic) if len(u) > 2 else np.nan
        save(rid, value=Dv, criterion="qualitative", pass_flag=int(frac_neg >= 0.5 and np.all(upper(Dv) >= -1e-9)), reference_kind="analytic",
             reference_detail="soft-DTW divergence D = sdtw(x,y) - (sdtw(x,x) + sdtw(y,y))/2 (Blondel, Mensch & Vert 2021) from soft_dtw_dp",
             notes="pass = most raw values negative (not a distance) while the divergence is >= 0",
             frac_raw_negative=frac_neg, raw_min=float(u.min()), raw_max=float(u.max()), self_mean=float(self_v.mean()),
             div_min=float(np.nanmin(upper(Dv))), div_max=float(np.nanmax(upper(Dv))), spearman_raw_vs_div=rho, **base)
        return

    if tag.startswith("gamma-") or tag.startswith("gbound-"):
        g = float(tag.split("-", 1)[1])
        key = rows_key((ds, T, path, "rows"), preprocess(raw, path))
        V = matrix("sdtw", None, key, extra=g)
        if tag.startswith("gamma-"):
            save(rid, value=V, criterion="tolerance", rtol=1e-9, atol=1e-9, reference_kind="independent-impl",
                 reference_detail=f"{KMOD}:soft_dtw_dp(gamma={g})", notes="tslearn soft_dtw(gamma=g) via probe", **base)
            return
        H = matrix("dtw", None, key) ** 2
        n = T
        lo = H - g * np.log(3) * (2 * n - 1)
        off = ~np.eye(H.shape[0], dtype=bool)
        ok = bool(np.all(S[off] <= H[off] * (1 + 1e-12) + 1e-12) and np.all(S[off] >= lo[off] - 1e-9))
        save(rid, value=H, criterion="optimality", pass_flag=int(ok), reference_kind="analytic",
             reference_detail="min - g log 3 <= softmin_g <= min per cell => dtw^2 - g log(3)(n+m-1) <= sdtw_g <= dtw^2",
             notes="tslearn soft_dtw(gamma=g) inside the analytic band around the hard DTW^2",
             max_rel_to_dtw2=float(np.nanmax(np.abs(S[off] - H[off]) / H[off])), band_width=float(g * np.log(3) * (2 * n - 1)), **base)
        return

    if tag in ("near", "offset"):
        Xd = preprocess(derive(tag, raw), path)
        key = rows_key((ds, T, path, tag), Xd)
        V = matrix("sdtw", None, key)
        save(rid, value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
             reference_detail=f"{KMOD}:soft_dtw_dp (direct (a-b)^2 cost)",
             notes="tslearn builds the cost with sklearn's |a|^2+|b|^2-2ab expansion (softdtw_variants.py:1208-1216)",
             max_rel=float(np.nanmax(np.abs(S - V) / np.abs(V))), max_abs=float(np.nanmax(np.abs(S - V))), **base)
        return

    if tag == "rawcall":
        rc = None if fam == "lcss" else ref_con
        x0, x1 = raw[0], raw[1]
        z = lambda v: (v - v.mean()) / v.std(ddof=1)
        a = pair_fn(fname, rc, z(x0), x1)
        b = pair_fn(fname, rc, z(x1), x0)
        zz = pair_fn(fname, rc, z(x0), z(x1))
        save(rid, value=np.asarray([a, b]), **exact, reference_kind="independent-impl",
             reference_detail=f"{SELF}: f(zscore(first), raw second) -> " + detail(fname, rc),
             notes="two-array API: parse_bivariate z-scores only the first array (pyspi/base.py:33-36, data.py:230-234); see preproc.two_array_api",
             both_zscored=zz, raw_call_01=float(S[0]), raw_call_10=float(S[1]))
        return

    raise ValueError(f"unhandled tag {tag!r} in {rid}")


def main():
    startup_selftest()
    m = json.loads((CHECKS / "manifest" / "dtw_family.json").read_text())
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in m["runs"]:
        if only and not any(o in run["run_id"] for o in only):
            continue
        try:
            handle(run)
            n_ok += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"REF ERROR {run['run_id']}: {exc!r}", file=sys.stderr)
            n_err += 1
    print(f"dtw_family refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
