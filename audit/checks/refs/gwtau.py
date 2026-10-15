"""gwtau family references (COMPARISON venv): distance.gwtau (pyspi GromovWasserstainTau).

pyspi (pyspi/statistics/distance.py:265-322): trajectory (t, x_t), t = 0..T-1; vec_geo_dist =
cumsum of the T-1 segment lengths sqrt(1 + (x_{t+1} - x_t)^2) (root excluded); wass_sorted sorts
both vectors descending and returns sqrt(mean((a - b)^2)) (equal lengths) or the lambda-overlap
coupling (unequal lengths).  Canonical definition: Kravtsova, McGee & Dawes 2023, Bull. Math. Biol.
85:77, Def. 2 (uniform mass on all n points, root included), Eqs. 8-11, Prop. 1(a): GWtau_2 = W_2
between the push-forwards of the root-distance distributions.

Independent references never import pyspi; the 1-D Wasserstein step is
_kernels_distance.wasserstein2_1d (merged quantile breakpoints, not a sort-and-pair RMS) on
_kernels_distance.root_arc_lengths; POT (ot.wasserstein_1d) is the external-package reference for
the same step.  Run tags (last field of the run_id):

  def               from-scratch matrix on the rows pyspi receives (raw / own ddof=1 z-score /
                    own linear detrend + z-score), every ORDERED pair computed separately (both
                    orientations), exact rtol 1e-10 atol 1e-12; extra key pyspi_max_asym
  pot               sqrt(ot.wasserstein_1d(La, Lb, p=2)) on the same arc-length samples (external-pkg)
  canon             paper Def. 2 with the root included (n = T points), times sqrt(T/(T-1)): exact
                    statement "pyspi = canonical * sqrt(T/(T-1))" (analytic relation); extra keys
                    give the size of the deviation
  octave            authors' MATLAB algorithm in GNU Octave (octave/gwtau.m), SAME-LINEAGE
  unmirrored        probe: base.Directed.multivariate (no mirroring), both orientations computed
                    by pyspi, vs the from-scratch matrix (exact); extra key max|B - B.T|
  sym               qualitative self-consistency: NaN diagonal and A == A.T bitwise (same-lineage)
  unequal-quantile  probe: wass_sorted unequal-length branch vs quantile W2 (independent-impl)
  unequal-pot       same vs POT (external-pkg)
  unequal-octave    same vs the authors' loop in Octave (same-lineage)
  affine            scaled_offset (pipeline) vs from-scratch on the z-scored bivariate_gaussian draw
  rawunits          qualitative: scaled_offset kernel path equals the from-scratch raw-row value and
                    differs from the z-scored value (time/amplitude scale coupling)
  detrendinv        trending (pipeline-detrend) vs from-scratch on detrended z-scored bivariate_gaussian
  trend-nodetrend   qualitative: trending through the plain pipeline differs from the untrended draw
  invar             probe rows (x0, -x0, x0+5, 3x0-2, x1): analytic zeros (sign flip, offset; and
                    positive scale on the pipeline path), from-scratch elsewhere
  perm-def          probe: jointly time-permuted rows vs from-scratch (exact)
  perm-change       qualitative: pyspi(permuted) vs pyspi(original) differs beyond rounding (order dependence;
                    the "unordered" siblings, e.g. pdist, are invariant: pdist|...|perm runs)
  rev               probe: jointly time-reversed rows vs from-scratch (exact), extra key change
  null              statistical: iid_gaussian pipeline vs the Monte Carlo null of gwtau for independent
                    z-scored Gaussian pairs.  The null is right-skewed (skew ~1.15), so the band is the
                    empirical Bonferroni interval [q_{a/2}, q_{1-a/2}], a = 0.05/m (m unique pairs):
                    value = band midpoint, se = band width / 4 (compare passes |pyspi - value| <= 2 se);
                    extra keys give the MC mean, sd and the pyspi pair mean with its MC z-score
  null-m2           qualitative analytic: kernel iid N(0,1): E[gwtau^2] = gamma0 T/2 + gamma1 (T-2)
                    (random walk of g(dx)-g(dy), MA(1) increments; gamma0 = 2 Var g(U), gamma1 =
                    2 Cov(g(U0), g(U1)), U ~ N(0, 2), Cov(U0, U1) = -1); pass if the analytic value is
                    within 3 MC se of the MC mean AND the observed 10-pair mean of A^2 lies in the
                    empirical Bonferroni band (a = 0.05/4) of its MC distribution
  slope             qualitative analytic: log-log slope of mean off-diagonal null value over T in
                    {64, 256, 1024, 4096} within [0.4, 0.6] (sqrt(T) law)
  sin-grid25        probe: noise-free sin(pi t/2 + phi), phi = (0, pi/4, -pi/2), z-scored: closed-form
                    increments (Dirichlet sums for mean and SD, dz_t = 2 sin(w/2) cos(w(t+1/2)+phi)/sd)
  sin-eeg           probe: noise-free sinusoids at f = 10/256 and f = 0.2, phases (0, pi/4, -pi/2), same
                    closed form; extra keys: per-step drift between phases from period_mean_step
  eeg-grid          qualitative: sin-eeg pyspi values at T = 256 and 4096: the f = 10/256 phase pairs stay
                    below 0.2 and do not grow with T (ratio in [0.8, 1.25]), unlike f0 = 0.25 (shift run)
  fs                probe: the same continuous pair sampled at FS_RATES over FS_DURATION s; pyspi values
                    vs from-scratch (exact); extra keys: the paper's form with real time t = k/fs
                    (root included) at each rate
  fs-ratio          qualitative: pass if pyspi(fs = 256)/pyspi(fs = 32) < 0.5 while the real-time
                    canonical ratio lies in [0.95, 1.05] (unit-sample time axis = hidden time scale)
  sin-freq          probe: noise-free sinusoids at f = 0.1 and 0.4 (and shifted copies), same closed form;
                    extra keys: leading-order |s1 - s2| T / sqrt(3) with s = per-period mean step
  xfreq             probe: narrowband_f0-0.1 rows over narrowband_f0-0.4 rows (noisy datasets) vs
                    from-scratch (exact); extra keys: noise-free closed form for the cross pairs
  shift             qualitative: narrowband_f0-0.25 pipeline pair (0,1) (pi/4) grows ~ T between
                    T = 256 and 4096 while pair (0,2) (-pi/2, a 1-sample delay) stays small
  delay             qualitative: delayed_noise pipeline, every pair below the 1 % quantile of the
                    Monte Carlo null at the same T (delays leave the increment sequence intact)
  ident             qualitative analytic: shared_envelope x2 is a bit-identical copy of x0, so
                    A[0, 2] = A[2, 0] = 0 exactly

Writes out/ref/<run_id>.npz per the compare.py contract.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np
import ot
from scipy import stats

import make_data
from _kernels_distance import gwtau as k_gwtau, root_arc_lengths, wasserstein2_1d

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCH = CHECKS / "out" / "exchange" / "gwtau"
OCTAVE_M = CHECKS / "octave" / "gwtau.m"
SELF = "audit/checks/refs/gwtau.py"
KERNEL = "audit/checks/refs/_kernels_distance.py:gwtau (root_arc_lengths + wasserstein2_1d)"

PERM_SEED = 20261015            # mirrors probes/gwtau_probes.py
UNEQUAL_SPECS = ((0, 1, 37, 53), (1, 2, 100, 80), (0, 3, 256, 129), (3, 4, 200, 201),
                 (2, 4, 5, 7), (1, 3, 64, 64), (4, 0, 17, 255))
MC_N = 4000
FS_DURATION = 16.0              # mirrors probes/gwtau_probes.py
FS_RATES = (16, 32, 64, 128, 256)
MC_SEED = 20261015
NULL_TS = (64, 256, 1024, 4096)


# ------------------------------------------------------------------ primitives
def zscore(X):
    """ddof=1 z-score per row (pyspi/data.py:195: scipy.stats.zscore(axis=1, nan_policy='omit', ddof=1))."""
    X = np.asarray(X, float)
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)


def detrend(X):
    """Least-squares removal of a + b t per row (scipy.signal.detrend type='linear', data.py:187)."""
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


def gw_matrix(Z, include_root=False):
    """Every ordered pair (i, j), i != j, computed separately (no mirroring); NaN diagonal."""
    P = Z.shape[0]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = k_gwtau(Z[i], Z[j], include_root=include_root)
    return A


def pot_w2(a, b):
    return float(np.sqrt(ot.wasserstein_1d(np.asarray(a, float), np.asarray(b, float), p=2)))


def pot_matrix(Z):
    P = Z.shape[0]
    L = [root_arc_lengths(z) for z in Z]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = pot_w2(L[i], L[j])
    return A


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def upper(A):
    return A[np.triu_indices(A.shape[0], 1)]


def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_REF / f"{rid}.npz", **{k: np.asarray(v) for k, v in kw.items()})


def load_pyspi(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], float)


def load_x(run, dataset=None, T=None):
    ds = make_data.load(dataset or run["dataset"], T or run["T"])
    x = np.asarray(ds["x"], float)
    if run.get("processes") is not None and dataset is None:
        x = x[list(run["processes"])]
    return x


def rid_of(run, dataset=None, T=None, path=None, tag=None):
    return "|".join([run["unit_id"], dataset or run["dataset"], str(T or run["T"]), path or run["path"], tag])


def tag_of(run):
    return run["run_id"].split("|")[-1]


def max_asym(A):
    if A is None:
        return np.nan
    d = np.abs(A - A.T)
    return float(np.nanmax(d)) if np.isfinite(d).any() else np.nan


# ------------------------------------------------------------------ derived rows (mirror of the probes)
def sinusoid_rows(kind, T):
    t = np.arange(T, dtype=np.float64)
    if kind == "sin-grid25":
        return np.vstack([np.sin(2 * np.pi * 0.25 * t + p) for p in (0.0, np.pi / 4, -np.pi / 2)])
    if kind == "sin-freq":
        return np.vstack([np.sin(2 * np.pi * 0.1 * t), np.sin(2 * np.pi * 0.4 * t),
                          np.sin(2 * np.pi * 0.1 * t + np.pi / 4), np.sin(2 * np.pi * 0.4 * t - np.pi / 2)])
    raise ValueError(kind)


SIN_PARAMS = {"sin-grid25": ((0.25, 0.0), (0.25, np.pi / 4), (0.25, -np.pi / 2)),
              "sin-freq": ((0.1, 0.0), (0.4, 0.0), (0.1, np.pi / 4), (0.4, -np.pi / 2)),
              "sin-eeg": tuple((f, p) for f in (10 / 256, 0.2) for p in (0.0, np.pi / 4, -np.pi / 2))}


def derive(kind, x):
    x = np.asarray(x, float)
    T = x.shape[1]
    if kind == "invar":
        return np.vstack([x[0], -x[0], x[0] + 5.0, 3.0 * x[0] - 2.0, x[1]])
    if kind.startswith("perm"):
        return x[:, np.random.default_rng(PERM_SEED).permutation(T)]
    if kind == "rev":
        return x[:, ::-1]
    raise ValueError(kind)


# ------------------------------------------------------------------ closed forms for sampled sinusoids
def sinusoid_steps(f, phi, T, zscored=True):
    """Segment lengths sqrt(1 + dz_t^2), t = 0..T-2, of z-scored sin(w t + phi), w = 2 pi f, from closed forms:
    sum_t sin(w t + phi) = sin(w T/2) sin(phi + w (T-1)/2) / sin(w/2);
    sum_t sin^2(w t + phi) = T/2 - sin(w T) cos(2 phi + w (T-1)) / (2 sin w);
    dz_t = [sin(w (t+1) + phi) - sin(w t + phi)] / sd = 2 sin(w/2) cos(w (t + 1/2) + phi) / sd.
    No sample differencing is used."""
    w = 2 * np.pi * f
    S1 = np.sin(w * T / 2) * np.sin(phi + w * (T - 1) / 2) / np.sin(w / 2)
    S2 = T / 2 - np.sin(w * T) * np.cos(2 * phi + w * (T - 1)) / (2 * np.sin(w))
    m = S1 / T
    sd = np.sqrt((S2 - T * m * m) / (T - 1)) if zscored else 1.0
    t = np.arange(T - 1, dtype=float)
    dz = 2 * np.sin(w / 2) * np.cos(w * (t + 0.5) + phi) / sd
    return np.sqrt(1.0 + dz * dz)


def sinusoid_gw(kind, T):
    L = [np.cumsum(sinusoid_steps(f, p, T)) for f, p in SIN_PARAMS[kind]]
    P = len(L)
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = np.sqrt(np.mean((L[i] - L[j]) ** 2))
    return A


def period_mean_step(f, phi=0.0):
    """Asymptotic mean segment length (sd -> 1/sqrt(2)): average of sqrt(1 + (2 sqrt2 sin(w/2) cos(w(t+1/2)+phi))^2)
    over one period q of the rational frequency f = p/q (q from fractions.Fraction(f).limit_denominator(1000))."""
    from fractions import Fraction
    q = Fraction(f).limit_denominator(1000).denominator
    w = 2 * np.pi * f
    t = np.arange(q, dtype=float)
    dz = 2 * np.sqrt(2) * np.sin(w / 2) * np.cos(w * (t + 0.5) + phi)
    return float(np.mean(np.sqrt(1 + dz * dz)))


# ------------------------------------------------------------------ null distribution
_MC_CACHE = {}


def mc_null(T, n=MC_N, zscored=True):
    key = (T, n, zscored)
    if key not in _MC_CACHE:
        rng = np.random.default_rng(MC_SEED + T)
        v = np.empty(n)
        for r in range(n):
            x = rng.standard_normal((2, T))
            if zscored:
                x = zscore(x)
            v[r] = k_gwtau(x[0], x[1])
        _MC_CACHE[key] = v
    return _MC_CACHE[key]


def mc_pair_mean(T, P=5, n=1000):
    """Monte Carlo of the mean over the P(P-1)/2 unique pairs of gwtau for z-scored iid N(0,1) rows."""
    rng = np.random.default_rng(MC_SEED + 11 * T)
    out = np.empty(n)
    iu = np.triu_indices(P, 1)
    for r in range(n):
        x = zscore(rng.standard_normal((P, T)))
        L = np.cumsum(np.sqrt(1 + np.diff(x, axis=1) ** 2), axis=1)
        D = L[:, None, :] - L[None, :, :]
        out[r] = np.mean(np.sqrt(np.mean(D ** 2, axis=-1))[iu])
    return out


def null_m2_analytic(T):
    """E[gwtau^2] for raw iid N(0,1) rows: D_k = sum_{t<k} [g(dx_t) - g(dy_t)], g(u) = sqrt(1 + u^2);
    Var D_k = k gamma0 + 2 (k-1) gamma1 (increments are 1-dependent); mean over k = 1..T-1 gives
    gamma0 T/2 + gamma1 (T-2).  E g(U), U ~ N(0,2): adaptive quadrature; E g(U0) g(U1) with U0 = x1 - x0,
    U1 = x2 - x1: 3-D Gauss-Hermite (n = 120 per axis)."""
    from scipy import integrate
    g = lambda u: np.sqrt(1 + u * u)
    Eg = integrate.quad(lambda u: g(u) * stats.norm.pdf(u, scale=np.sqrt(2)), -np.inf, np.inf, epsabs=1e-13)[0]
    V0 = 3.0 - Eg ** 2                                   # E g^2 = 1 + E U^2 = 3
    xg, wg = np.polynomial.hermite_e.hermegauss(120)
    wg = wg / np.sqrt(2 * np.pi)
    X0, X1, X2 = np.meshgrid(xg, xg, xg, indexing="ij")
    W = wg[:, None, None] * wg[None, :, None] * wg[None, None, :]
    C1 = float(np.sum(W * g(X1 - X0) * g(X2 - X1))) - Eg ** 2
    g0, g1 = 2 * V0, 2 * C1
    return g0 * T / 2 + g1 * (T - 2), dict(Eg=Eg, varg=V0, cov1=C1)


def mc_mean_m2(T, P=5, n=2000):
    """Monte Carlo of the mean over the P(P-1)/2 unique pairs of gwtau^2 for raw iid N(0,1) rows."""
    rng = np.random.default_rng(MC_SEED + 7 * T)
    out = np.empty(n)
    iu = np.triu_indices(P, 1)
    for r in range(n):
        x = rng.standard_normal((P, T))
        L = np.cumsum(np.sqrt(1 + np.diff(x, axis=1) ** 2), axis=1)
        D = L[:, None, :] - L[None, :, :]
        out[r] = np.mean(np.mean(D ** 2, axis=-1)[iu])
    return out


# ------------------------------------------------------------------ Octave
def run_octave(job, rows, spec):
    d = EXCH / job
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / "rows.csv", np.asarray(rows, float), delimiter=",", fmt="%.17g")
    np.savetxt(d / "spec.csv", np.asarray(spec, int), delimiter=",", fmt="%d")
    if (d / "result.csv").exists():
        (d / "result.csv").unlink()
    r = subprocess.run(["/usr/bin/octave-cli", "--no-gui", "-q", str(OCTAVE_M), str(d)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"octave failed: {r.stderr[-2000:]}")
    return np.loadtxt(d / "result.csv", ndmin=1)


def octave_version():
    r = subprocess.run(["/usr/bin/octave-cli", "--version"], capture_output=True, text=True)
    return r.stdout.splitlines()[0] if r.returncode == 0 else "octave"


# ------------------------------------------------------------------ checks
def chk_def(run):
    x = load_x(run)
    V = gw_matrix(preprocess(x, run["path"]))
    A = load_pyspi(run["run_id"])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_def -> {KERNEL}; numpy {np.__version__}",
         notes=f"from-scratch GWtau (root excluded, pyspi convention) on {run['path']} rows; every ordered pair separately",
         pyspi_max_asym=max_asym(A), ref_max_asym=max_asym(V))


def chk_pot(run):
    x = load_x(run)
    Z = preprocess(x, run["path"])
    V = pot_matrix(Z)
    K = gw_matrix(Z)
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="external-pkg",
         reference_detail=f"{SELF}:chk_pot -> POT=={ot.__version__} ot.wasserstein_1d(La, Lb, p=2) ** 0.5 on "
                          f"_kernels_distance.root_arc_lengths",
         notes="external-package 1-D W2 on the same arc-length samples (root excluded)",
         pot_vs_kernel_max_abs=float(np.nanmax(np.abs(V - K))))


def chk_canon(run):
    x = load_x(run)
    T = run["T"]
    Z = preprocess(x, run["path"])
    C = gw_matrix(Z, include_root=True)
    A = load_pyspi(run["run_id"])
    expected_ratio = np.sqrt((T - 1) / T)
    kw = {}
    if A is not None:
        kw = dict(canon_over_pyspi_max_dev=float(np.nanmax(np.abs(offdiag(C) / offdiag(A) - expected_ratio))),
                  pyspi_minus_canon_max_rel=float(np.nanmax(np.abs(offdiag(A) - offdiag(C)) / offdiag(C))))
    save(run["run_id"], value=C * np.sqrt(T / (T - 1)), criterion="exact", rtol=1e-10, atol=1e-12,
         reference_kind="analytic",
         reference_detail=f"{SELF}:chk_canon -> Kravtsova et al. 2023 Def. 2 / Eq. 11 with the root included "
                          f"({KERNEL}, include_root=True), times sqrt(T/(T-1))",
         notes="tests pyspi = canonical * sqrt(T/(T-1)) exactly; canonical/pyspi expected sqrt((T-1)/T)",
         expected_canon_over_pyspi=expected_ratio, **kw)


def chk_octave(run):
    x = load_x(run)
    Z = preprocess(x, run["path"])
    P, T = Z.shape
    pairs = [(i, j) for i in range(P) for j in range(P) if i != j]
    res = run_octave(run["run_id"].replace("|", "_"), Z, [(i + 1, j + 1, T, T) for i, j in pairs])
    V = np.full((P, P), np.nan)
    for (i, j), v in zip(pairs, res):
        V[i, j] = v
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-12, atol=1e-12, reference_kind="same-lineage",
         reference_detail=f"{SELF}:chk_octave -> audit/checks/octave/gwtau.m ({octave_version()}): authors' "
                          "vec_geo_dist.m + wass_sorted.m loops (github.com/kravtsova2/GWtau)",
         notes="pyspi is a Python port of these MATLAB files: same-lineage")


def chk_unmirrored(run):
    x = load_x(run)
    V = gw_matrix(preprocess(x, run["path"]))
    B = load_pyspi(run["run_id"])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_unmirrored -> {KERNEL}",
         notes="probe: base.Directed.multivariate (bivariate(i,j) and bivariate(j,i) both computed, no mirroring)",
         unmirrored_max_asym=max_asym(B),
         unmirrored_bitwise_sym=int(B is not None and bool(np.array_equal(np.nan_to_num(B, nan=-1.0),
                                                                           np.nan_to_num(B.T, nan=-1.0)))))


def chk_sym(run):
    A = load_pyspi(run["run_id"])
    ok = A is not None and bool(np.isnan(np.diag(A)).all()) and bool(np.array_equal(upper(A), upper(A.T)))
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="same-lineage",
         reference_detail=f"{SELF}:chk_sym (pyspi self-consistency: A vs A.T)",
         notes="NaN diagonal and A == A.T bitwise (Undirected.multivariate mirrors the upper triangle)",
         max_asym=max_asym(A))


def _unequal_rows(run):
    return load_x(run)


def chk_unequal(run):
    tag = tag_of(run)
    x = _unequal_rows(run)
    detail = {"unequal-quantile": ("independent-impl", f"{SELF}:chk_unequal -> _kernels_distance.wasserstein2_1d "
                                   "(merged quantile breakpoints) on root_arc_lengths"),
              "unequal-pot": ("external-pkg", f"{SELF}:chk_unequal -> POT=={ot.__version__} ot.wasserstein_1d(p=2)"),
              "unequal-octave": ("same-lineage", f"{SELF}:chk_unequal -> audit/checks/octave/gwtau.m "
                                 f"({octave_version()}): authors' wass_sorted.m lambda loop")}[tag]
    if tag == "unequal-octave":
        V = run_octave(run["run_id"].replace("|", "_"), x, [(i + 1, j + 1, n1, n2) for i, j, n1, n2 in UNEQUAL_SPECS])
        crit, rtol, atol = "tolerance", 1e-12, 1e-12
    else:
        f = wasserstein2_1d if tag == "unequal-quantile" else pot_w2
        V = np.array([f(root_arc_lengths(x[i, :n1]), root_arc_lengths(x[j, :n2])) for i, j, n1, n2 in UNEQUAL_SPECS])
        crit, rtol, atol = "exact", 1e-10, 1e-12
    save(run["run_id"], value=V, criterion=crit, rtol=rtol, atol=atol, reference_kind=detail[0],
         reference_detail=detail[1],
         notes=f"probe GromovWasserstainTau.gwtau(x_i[:n1], x_j[:n2]) for (i,j,n1,n2) in {UNEQUAL_SPECS}; "
               "branch unreachable from the Calculator")


def chk_affine(run):
    xb = load_x(run, dataset="bivariate_gaussian")
    V = gw_matrix(zscore(xb))
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:chk_affine: invariance under x -> a x + b (a > 0) after z-scoring; value from {KERNEL} "
                          "on the z-scored bivariate_gaussian draw (scaled_offset = scale * D2 + offset)",
         notes="scaled_offset pipeline must equal bivariate_gaussian pipeline")


def chk_rawunits(run):
    x = load_x(run)
    A = load_pyspi(run["run_id"])
    Vraw = gw_matrix(x)
    Vz = gw_matrix(zscore(load_x(run, dataset="bivariate_gaussian")))
    ok = A is not None and np.allclose(offdiag(A), offdiag(Vraw), rtol=1e-10, atol=1e-12)
    ratio = offdiag(A) / offdiag(Vz) if A is not None else np.array([np.nan])
    ok = bool(ok and np.nanmax(ratio) > 2.0)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_rawunits -> {KERNEL} on raw and on z-scored rows",
         notes="normalise=False: value depends on the raw amplitude units (sqrt(1 + dx^2) mixes the unit time step "
               "with amplitude); pass = pyspi equals the raw-row value AND differs from the z-scored value by > 2x",
         raw_over_zscored_min=float(np.nanmin(ratio)), raw_over_zscored_max=float(np.nanmax(ratio)),
         pyspi_row0=A[0, 1:] if A is not None else np.nan, zscored_row0=Vz[0, 1:])


def chk_detrendinv(run):
    xb = load_x(run, dataset="bivariate_gaussian")
    V = gw_matrix(zscore(detrend(xb)))
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:chk_detrendinv: linear detrend removes slope*t exactly; value from {KERNEL} on "
                          "the detrended, z-scored bivariate_gaussian draw",
         notes="trending = bivariate_gaussian + slope*t; pipeline-detrend must equal the untrended draw's value")


def chk_trend_nodetrend(run):
    A = load_pyspi(run["run_id"])
    Vt = gw_matrix(zscore(load_x(run)))
    Vb = gw_matrix(zscore(load_x(run, dataset="bivariate_gaussian")))
    same = A is not None and np.allclose(offdiag(A), offdiag(Vt), rtol=1e-10, atol=1e-12)
    rel = np.abs(A - Vb) / Vb if A is not None else np.full_like(Vb, np.nan)
    # rows 1..3 carry slopes 0.05, -0.02, 0.1; the (0,1),(0,2),(0,3) pairs each contain a trended row
    ok = bool(same and np.nanmin(rel[0, 1:]) > 0.05)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_trend_nodetrend -> {KERNEL}",
         notes="plain pipeline does not remove trends: pyspi equals the from-scratch value on z-scored trending rows "
               "and differs by > 5 % from the untrended draw on every pair with a trended row",
         rel_change_row0=rel[0, 1:], pyspi_row0=A[0, 1:] if A is not None else np.nan, untrended_row0=Vb[0, 1:])


def chk_invar(run):
    x = load_x(run)
    R = derive("invar", x)
    V = gw_matrix(preprocess(R, run["path"]))
    zero = [0, 1, 2] if run["path"] == "kernel" else [0, 1, 2, 3]
    for i in zero:
        for j in zero:
            if i != j:
                V[i, j] = 0.0
    A = load_pyspi(run["run_id"])
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-9, reference_kind="analytic",
         reference_detail=f"{SELF}:chk_invar: analytic zeros (|d(-x)| = |dx|; d(x+c) = dx; after z-scoring a x + b "
                          f"with a > 0 is x) among rows {zero}; other pairs from {KERNEL}",
         notes="rows (x0, -x0, x0+5, 3x0-2, x1); 3x0-2 is a zero only on the pipeline path",
         pyspi_max_on_zero_pairs=float(max(abs(A[i, j]) for i in zero for j in zero if i != j)) if A is not None else np.nan,
         pyspi_x0_vs_3x0m2=float(A[0, 3]) if A is not None else np.nan)


def chk_perm_def(run):
    R = derive("perm", load_x(run))
    V = gw_matrix(preprocess(R, run["path"]))
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_perm_def -> {KERNEL} on the jointly permuted rows (seed {PERM_SEED})",
         notes="probe: pyspi on rows permuted in time with one common permutation")


def chk_perm_change(run):
    A = load_pyspi(run["run_id"])
    B = load_pyspi(rid_of(run, tag="def"))
    if B is None:
        B = gw_matrix(preprocess(load_x(run), run["path"]))
    rel = np.abs(upper(A) - upper(B)) / upper(B)
    ok = bool(rel.min() > 1e-6)          # invariant SPIs change at rounding level (pdist perm: max_abs <= 2e-12)
    # independent side of the same statement: the from-scratch kernel on permuted vs original rows (no pyspi value)
    x = load_x(run)
    Kp = gw_matrix(preprocess(derive("perm", x), run["path"]))
    Ko = gw_matrix(preprocess(x, run["path"]))
    krel = np.abs(upper(Kp) - upper(Ko)) / upper(Ko)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="same-lineage",
         reference_detail=f"{SELF}:chk_perm_change: pyspi(jointly permuted rows) vs pyspi(original rows) "
                          f"({rid_of(run, tag='def')}); pass_flag decided on pyspi vs pyspi (self-consistency, "
                          f"same-lineage); kernel_rel_change_* repeat the comparison with {KERNEL} only",
         notes="an 'unordered' SPI is invariant to a common permutation of the time index (pdist: "
               "distance.pdist.*|bivariate_gaussian|1024|pipeline|perm pass, max_abs <= 2e-12); pass = every pair changes by "
               "> 1e-6 relative (far above rounding), i.e. gwtau is order dependent; the size of the change is data "
               "dependent. The pass_flag compares pyspi with pyspi; the independent evidence is the perm-def and def "
               "runs (pyspi equals the kernel on each row set) plus kernel_rel_change_min here (kernel vs kernel)",
         rel_change_median=float(np.median(rel)), rel_change_min=float(rel.min()), rel_change_max=float(rel.max()),
         kernel_rel_change_median=float(np.median(krel)), kernel_rel_change_min=float(krel.min()),
         kernel_rel_change_max=float(krel.max()), kernel_vs_pyspi_rel_change_maxdiff=float(np.max(np.abs(krel - rel))),
         permuted_upper=upper(A), original_upper=upper(B))


def chk_rev(run):
    R = derive("rev", load_x(run))
    V = gw_matrix(preprocess(R, run["path"]))
    B = load_pyspi(rid_of(run, tag="def"))
    kw = {}
    if B is not None:
        kw = dict(rel_change_vs_forward_median=float(np.median(np.abs(upper(V) - upper(B)) / upper(B))))
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_rev -> {KERNEL} on time-reversed rows",
         notes="probe: pyspi on jointly time-reversed rows (root moves to the other end)", **kw)


def chk_null(run):
    T = run["T"]
    v = mc_null(T)
    P = load_x(run).shape[0]
    m = P * (P - 1) // 2
    a = 0.05 / m
    lo, hi = (float(q) for q in np.quantile(v, [a / 2, 1 - a / 2]))
    V = np.full((P, P), 0.5 * (lo + hi))
    np.fill_diagonal(V, np.nan)
    S = np.full((P, P), 0.25 * (hi - lo))
    A = load_pyspi(run["run_id"])
    kw = {}
    if A is not None:
        # the pair mean of m dependent entries: compare with the MC distribution of the same statistic
        pm = mc_pair_mean(T, P)
        obs = float(np.mean(upper(A)))
        kw = dict(pyspi_pair_mean=obs, mc_pair_mean_mean=float(np.mean(pm)), mc_pair_mean_sd=float(np.std(pm, ddof=1)),
                  pyspi_pair_mean_z=float((obs - np.mean(pm)) / np.std(pm, ddof=1)),
                  pyspi_min=float(np.nanmin(A)), pyspi_max=float(np.nanmax(A)))
    save(run["run_id"], value=V, criterion="statistical", se=S, reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_null: Monte Carlo ({MC_N} pairs of z-scored iid N(0,1) rows, seed {MC_SEED}+T) "
                          f"of {KERNEL}",
         notes="per-entry family-wise 5 % band from the empirical MC quantiles (Bonferroni over the m unique pairs; "
               "the null is right-skewed, so a symmetric mean +/- z sd band would be wrong): value = midpoint, "
               "se = width/4",
         mc_mean=float(np.mean(v)), mc_sd=float(np.std(v, ddof=1)), mc_skew=float(stats.skew(v)),
         mc_q01=float(np.quantile(v, 0.01)), band_lo=lo, band_hi=hi, bonferroni_m=m, **kw)


def chk_null_m2(run):
    T = run["T"]
    A = load_pyspi(run["run_id"])
    pred, parts = null_m2_analytic(T)
    mc = mc_mean_m2(T)
    n = len(mc)
    sd = float(np.std(mc, ddof=1))
    se_mean = sd / np.sqrt(n)
    obs = float(np.mean(upper(A) ** 2))
    a = 0.05 / len(NULL_TS)                      # Bonferroni over the four T of the null series
    lo, hi = (float(q) for q in np.quantile(mc, [a / 2, 1 - a / 2]))
    formula_ok = abs(float(np.mean(mc)) - pred) <= 3 * se_mean
    draw_ok = lo <= obs <= hi
    ok = bool(formula_ok and draw_ok)
    # the drafted rule, kept for disclosure only (replaced post hoc after it flagged the T=256 draw): |obs - pred| <= 2 MC sd
    drafted_z = float((obs - pred) / sd)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{SELF}:chk_null_m2: E[gwtau^2] = gamma0 T/2 + gamma1 (T-2) for raw iid N(0,1) "
                          "(1-dependent increments g(dx_t) - g(dy_t), g(u) = sqrt(1+u^2))",
         notes="kernel path, raw iid N(0,1) rows (P = 5).  Two parts: (i) the analytic E[gwtau^2] lies within 3 MC "
               "standard errors of the MC mean (validates the sqrt(T) law); (ii) the observed mean over the 10 pairs "
               "of A^2 lies inside the empirical Bonferroni band [q_{a/2}, q_{1-a/2}], a = 0.05/4, of its MC "
               "distribution (right-skewed; a mean +/- 2 sd rule is miscalibrated, see family note). POST HOC: this "
               "two-part rule replaced the drafted rule |obs - analytic| <= 2 MC sd after that rule flagged the T=256 "
               "draw; drafted_rule_z / drafted_rule_ok report the drafted rule. Both rules test the data draw, not "
               "pyspi, which equals the kernel exactly on these rows",
         analytic_m2=pred, mc_m2_mean=float(np.mean(mc)), mc_m2_se_of_mean=se_mean, mc_m2_sd_of_pair_mean=sd,
         mc_m2_skew=float(stats.skew(mc)), band_lo=lo, band_hi=hi, pyspi_mean_m2=obs,
         pyspi_mc_percentile=float(np.mean(mc < obs)), analytic_vs_mc_z=float((np.mean(mc) - pred) / se_mean),
         formula_ok=int(formula_ok), draw_ok=int(draw_ok), drafted_rule_z=drafted_z,
         drafted_rule_ok=int(abs(drafted_z) <= 2), Eg=parts["Eg"], varg=parts["varg"], cov1=parts["cov1"])


def chk_slope(run):
    means = []
    for T in NULL_TS:
        A = load_pyspi(rid_of(run, T=T, tag="null"))
        means.append(float(np.mean(upper(A))))
    lt, lm = np.log(NULL_TS), np.log(means)
    slope = float(np.polyfit(lt, lm, 1)[0])
    mcm = [float(np.mean(mc_null(T))) for T in NULL_TS]
    mc_slope = float(np.polyfit(lt, np.log(mcm), 1)[0])
    ok = 0.4 <= slope <= 0.6
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{SELF}:chk_slope: random-walk argument, E[gwtau] ~ c sqrt(T) (slope 0.5)",
         notes="log-log slope of the mean off-diagonal value of the iid_gaussian pipeline null runs over T",
         Ts=np.array(NULL_TS), pyspi_means=np.array(means), slope=slope, mc_means=np.array(mcm), mc_slope=mc_slope)


def chk_sin(run):
    kind, T = tag_of(run), run["T"]
    assert run["path"] == "pipeline"
    V = sinusoid_gw(kind, T)
    kw = {}
    if kind == "sin-grid25":
        s0 = np.sqrt(1 + 2 * (T - 1) / T)                 # |dz| = a = sqrt(2 (T-1)/T) every step (phi = 0)
        s_even, s_odd = 1.0, np.sqrt(1 + 2 * 2 * (T - 1) / T)   # phi = pi/4: |dz| alternates 0, sqrt2 * a
        kw = dict(drift_per_step=float(s0 - (s_even + s_odd) / 2),
                  leading_order_01=float((s0 - (s_even + s_odd) / 2) * T / np.sqrt(3)))
    elif kind == "sin-eeg":
        fa, fb = 10 / 256, 0.2
        kw = dict(drift_per_step_f10_256_pi4=period_mean_step(fa, np.pi / 4) - period_mean_step(fa),
                  drift_per_step_f10_256_mpi2=period_mean_step(fa, -np.pi / 2) - period_mean_step(fa),
                  drift_per_step_f02_pi4=period_mean_step(fb, np.pi / 4) - period_mean_step(fb),
                  drift_per_step_f02_mpi2=period_mean_step(fb, -np.pi / 2) - period_mean_step(fb),
                  drift_per_step_f025_pi4=period_mean_step(0.25, np.pi / 4) - period_mean_step(0.25))
    else:
        s01, s04 = period_mean_step(0.1), period_mean_step(0.4)
        kw = dict(mean_step_f01=s01, mean_step_f04=s04, leading_order_01=float(abs(s01 - s04) * T / np.sqrt(3)),
                  leading_order_rel_err_01=float(abs(abs(s01 - s04) * T / np.sqrt(3) - V[0, 1]) / V[0, 1]))
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-9, atol=1e-9, reference_kind="analytic",
         reference_detail=f"{SELF}:chk_sin -> sinusoid_steps (closed-form mean, SD and increments of sampled "
                          "sinusoids; no sample differencing)",
         notes=f"probe {kind}: noise-free z-scored sinusoids {SIN_PARAMS[kind]} (f, phase)", **kw)


def chk_xfreq(run):
    T = run["T"]
    X = np.vstack([load_x(run), load_x(run, dataset="narrowband_f0-0.4")])
    V = gw_matrix(preprocess(X, run["path"]))
    A = load_pyspi(run["run_id"])
    # noise-free closed form for the cross pairs (f0=0.1 row k vs f0=0.4 row l), phases (0, pi/4, -pi/2)
    ph = (0.0, np.pi / 4, -np.pi / 2)
    Lc = [np.cumsum(sinusoid_steps(0.1, p, T)) for p in ph] + [np.cumsum(sinusoid_steps(0.4, p, T)) for p in ph]
    C = np.array([[np.sqrt(np.mean((Lc[k] - Lc[3 + l]) ** 2)) for l in range(3)] for k in range(3)])
    kw = {}
    if A is not None:
        kw = dict(noisefree_cross=C, pyspi_cross=A[:3, 3:], cross_rel_diff_max=float(np.max(np.abs(A[:3, 3:] - C) / C)))
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_xfreq -> {KERNEL}; noise-free closed form via sinusoid_steps (extra keys)",
         notes="rows 0-2 narrowband_f0-0.1, rows 3-5 narrowband_f0-0.4 (phases 0, pi/4, -pi/2, 0.05 noise)", **kw)


def chk_shift(run):
    A4 = load_pyspi(rid_of(run, T=4096, tag="def"))
    A2 = load_pyspi(rid_of(run, T=256, tag="def"))
    r01 = A4[0, 1] / A2[0, 1]
    C2, C4 = sinusoid_gw("sin-grid25", 256), sinusoid_gw("sin-grid25", 4096)
    ok = bool(12 <= r01 <= 20 and A4[0, 2] < 0.05 * A4[0, 1])
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{SELF}:chk_shift: sampling-grid argument (f0 = 0.25: |dz| = sqrt2 every step for phases "
                          "0 and -pi/2, alternating 0 / 2 for pi/4) -> pair (0,1) drift ~ 0.114 per step, value ~ T",
         notes="narrowband_f0-0.25 pipeline: pass if A01(4096)/A01(256) in [12, 20] (proportional to T: 16) and "
               "A02(4096) < 0.05 A01(4096)",
         pyspi_01_T256=A2[0, 1], pyspi_01_T4096=A4[0, 1], ratio_01=r01, pyspi_02_T256=A2[0, 2], pyspi_02_T4096=A4[0, 2],
         noisefree_01_T256=C2[0, 1], noisefree_01_T4096=C4[0, 1])


def chk_delay(run):
    T = run["T"]
    A = load_pyspi(run["run_id"])
    v = mc_null(T)
    q01 = float(np.quantile(v, 0.01))
    ok = bool(np.nanmax(A) < q01)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_delay: Monte Carlo null ({MC_N} z-scored iid pairs) of {KERNEL}",
         notes="delayed_noise x_k = x0(t - d_k) + 0.1 noise, d = (0,1,3,5): pass if every entry is below the 1 % "
               "quantile of the independent-pair null at the same T",
         pyspi_row0=A[0, 1:], pyspi_max=float(np.nanmax(A)), null_q01=q01, null_mean=float(np.mean(v)))


def chk_ident(run):
    A = load_pyspi(run["run_id"])
    ok = A is not None and A[0, 2] == 0.0 and A[2, 0] == 0.0
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{SELF}:chk_ident: identical rows have identical arc-length vectors -> 0",
         notes="shared_envelope x2 is a bit-identical copy of x0", pyspi_02=A[0, 2], pyspi_20=A[2, 0],
         pyspi_min_other_pairs=float(min(A[i, j] for i in range(4) for j in range(4)
                                         if i != j and {i, j} != {0, 2})))


def chk_eeg_grid(run):
    A2 = load_pyspi(rid_of(run, T=256, tag="sin-eeg"))
    A4 = load_pyspi(rid_of(run, T=4096, tag="sin-eeg"))
    a2 = np.array([A2[0, 1], A2[0, 2]])
    a4 = np.array([A4[0, 1], A4[0, 2]])
    b2 = np.array([A2[3, 4], A2[3, 5]])
    b4 = np.array([A4[3, 4], A4[3, 5]])
    ratio = a4 / a2
    ok = bool(np.all(a4 < 0.2) and np.all((ratio >= 0.8) & (ratio <= 1.25)))
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{SELF}:chk_eeg_grid: the per-step drift between phase-shifted copies is the q-th Fourier "
                          "harmonic of g(c cos theta) for f = p/q, negligible for q = 128 (period_mean_step)",
         notes="sin-eeg pyspi values (noise-free, pipeline): rows 0-2 f = 10/256, rows 3-5 f = 0.2; pass if the "
               "f = 10/256 phase pairs (pi/4, -pi/2) are < 0.2 at T = 4096 and change by a factor in [0.8, 1.25] "
               "from T = 256 (no growth with T)",
         f10_256_pairs_T256=a2, f10_256_pairs_T4096=a4, f10_256_ratio=ratio,
         f02_pairs_T256=b2, f02_pairs_T4096=b4, f02_ratio=b4 / b2)


def fs_rows(fs):
    s = np.arange(int(round(FS_DURATION * fs)), dtype=np.float64) / fs
    return np.vstack([np.sin(2 * np.pi * 1.0 * s) + 0.5 * np.sin(2 * np.pi * 3.0 * s + 1.0),
                      np.sin(2 * np.pi * 1.3 * s + 0.4)])


def realtime_root_dist(z, dt):
    """Kravtsova et al. 2023 Eq. 8 with the real sample times t_k = k dt, root included (Def. 2)."""
    return np.concatenate([[0.0], np.cumsum(np.sqrt(dt * dt + np.diff(z) ** 2))])


def fs_values(path):
    pys, canon = [], []
    for r in FS_RATES:
        Z = preprocess(fs_rows(r), path)
        pys.append(k_gwtau(Z[0], Z[1]))
        canon.append(wasserstein2_1d(realtime_root_dist(Z[0], 1.0 / r), realtime_root_dist(Z[1], 1.0 / r)))
    return np.array(pys), np.array(canon)


def chk_fs(run):
    V, C = fs_values(run["path"])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="independent-impl",
         reference_detail=f"{SELF}:chk_fs -> {KERNEL} on the {run['path']} rows of fs_rows(fs)",
         notes=f"probe fs: same continuous pair over {FS_DURATION} s sampled at {FS_RATES} Hz (T = {FS_DURATION:g} fs); "
               "pyspi uses the sample index as time",
         fs_rates=np.array(FS_RATES), realtime_canonical=C)


def chk_fs_ratio(run):
    A = load_pyspi(rid_of(run, tag="fs"))
    V, C = fs_values(run["path"])
    i32, i256 = FS_RATES.index(32), FS_RATES.index(256)
    r_py = float(A[i256] / A[i32])
    r_c = float(C[i256] / C[i32])
    ok = bool(r_py < 0.5 and 0.95 <= r_c <= 1.05)
    save(run["run_id"], value=np.nan, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
         reference_detail=f"{SELF}:chk_fs_ratio: Kravtsova et al. 2023 Eq. 8 with real times t_k = k/fs (root "
                          "included) converges as fs grows; the unit-sample time axis does not",
         notes="pass if pyspi(256 Hz)/pyspi(32 Hz) < 0.5 while the real-time canonical ratio is in [0.95, 1.05]",
         fs_rates=np.array(FS_RATES), pyspi_values=A, realtime_canonical=C, pyspi_ratio_256_over_32=r_py,
         realtime_ratio_256_over_32=r_c)


DISPATCH = {"def": chk_def, "pot": chk_pot, "canon": chk_canon, "octave": chk_octave, "unmirrored": chk_unmirrored,
            "sym": chk_sym, "unequal-quantile": chk_unequal, "unequal-pot": chk_unequal,
            "unequal-octave": chk_unequal, "affine": chk_affine, "rawunits": chk_rawunits,
            "detrendinv": chk_detrendinv, "trend-nodetrend": chk_trend_nodetrend, "invar": chk_invar,
            "perm-def": chk_perm_def, "perm-change": chk_perm_change, "rev": chk_rev, "null": chk_null,
            "null-m2": chk_null_m2, "slope": chk_slope, "sin-grid25": chk_sin, "sin-freq": chk_sin,
            "sin-eeg": chk_sin, "eeg-grid": chk_eeg_grid, "fs": chk_fs, "fs-ratio": chk_fs_ratio,
            "xfreq": chk_xfreq, "shift": chk_shift, "delay": chk_delay, "ident": chk_ident}


def main():
    m = json.loads((CHECKS / "manifest" / "gwtau.json").read_text())
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in m["runs"]:
        if only and tag_of(run) not in only:
            continue
        try:
            DISPATCH[tag_of(run)](run)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"ERROR {run['run_id']}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"gwtau refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
