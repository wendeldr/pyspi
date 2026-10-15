#!/usr/bin/env python3
"""References for the spectral_coherence family: spectral.cohmag, spectral.phase, spectral.icoh.

Runs in the COMPARISON venv (no pyspi, no spectral_connectivity).  For every run of
``audit/checks/manifest/spectral_coherence.json`` it writes ``out/ref/<run_id>.npz`` in the
compare.py contract.  The run tag (fifth ``|`` field of the run_id, before the colon) selects
the check:

impl        independent re-implementation (numpy FFT + scipy DPSS via ``_kernels_spectral``):
            constant detrend, 5 DPSS tapers NW=3 (concentration > 0.9), n_fft =
            5-smooth next_fast_len (scipy.fftpack rule), unweighted taper mean
            S_ij = mean_k X_i conj(X_j), C = S_ij / sqrt(S_ii S_jj), first (n_fft+1)//2 bins,
            closed band fmin <= f <= fmax, nanmean/nanmax; measure |C|^2, angle C, |Im C|.
            Pipeline paths z-score (ddof=1) after an optional linear detrend, done here.
            Extra keys quantify the alternatives (strict band, DC excluded, unsquared /
            circular / signed forms) and pyspi's symmetry.
invariance  pyspi(scaled_offset) == pyspi(bivariate_gaussian) (same draw; scale/offset
            invariance of coherency after the constant detrend), exact; same-lineage
            (pyspi vs pyspi).
perm        pyspi(x[p]) == pyspi(x)[p][:, p], exact; same-lineage (pyspi vs pyspi).
identical   shared_envelope x2 is bit-identical to x0: C_02 = 1 -> cohmag 1, phase 0, icoh 0.
f0bin       single FFT bin f = 0.25 (custom band fmin = fmax = 0.25, probe) on the
            narrowband sinusoids: analytic C_ij(f0) = exp(i (phi_i - phi_j)); tolerance
            = 4 x the Monte Carlo RMS deviation of the estimator from the analytic value.
null        iid_gaussian: Monte Carlo (fresh iid draws, this reference) mean and sd of the
            band statistic; statistical, se = sd x z_(1-0.05/(2m))/2 (familywise over the
            m distinct entries judged).  Per-bin analytic null of |C|^2: Beta(1, K-1), mean 1/K.
pop         canonical population value: var2_bivariate from the analytic S(f) = H Sigma H^H
            at the FFT grid; bivariate_gaussian C = rho (zero lag) on the coupled pairs;
            statistical with the Monte Carlo sd of the estimator.
mcexp       Monte Carlo expectation of the estimator itself (same generator, fresh draws).
eeg         the 11 retained EEG band identifiers (Hz/2048) after detrend + z-score.
emptymax    qualitative: statistic=max on a band with no FFT bin must raise ValueError.
nfftN       T not 5-smooth (x[:, :N]): zero padding to the scipy.fftpack next_fast_len.
gridN       pyspi's own frequency vector vs nonneg(fftfreq(next_fast_len_235(N))), exact.
mne         external package mne-connectivity (eigenvalue-weighted tapers): pyspi must sit
            much closer to the library form than to the canonical form (qualitative).
dcpi        phase: A[i,j] = -A[j,i] + 2 pi [Re S_ij(0) < 0] / n_bins (DC term), exact;
            same-lineage (value built from pyspi's own A.T).
circular    phase: circular band mean arg sum_f C/|C| (canonical angle average), tolerance.
delay       phase: analytic arithmetic band mean of wrap(2 pi f (d_j - d_i)) (delayed_noise).
direction   phase: sign of A[i,j] where the analytic phase does not wrap in the band.
constV      one channel replaced by the constant V (zero power): NaN row/column expected.
signed      icoh: Nolte 2004 signed Im C band mean, tolerance.
unsquared   cohmag: band mean of |C| (unsquared coherence), tolerance.
"""
from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.fft  # noqa: E402
import scipy.signal  # noqa: E402
import scipy.stats  # noqa: E402

import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "spectral_coherence"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
SELF = "audit/checks/refs/spectral_coherence.py"
K_TAPERS = 5
N_MC = 400
MC_SEED = 20261014
BANDS = {"full": (0.0, 0.5), "low": (0.0, 0.25), "high": (0.25, 0.5)}
EEG_BANDS = [
    (1 / 2048, 4 / 2048), (1 / 2048, 70 / 2048), (1 / 2048, 250 / 2048), (4 / 2048, 8 / 2048),
    (8 / 2048, 12 / 2048), (12 / 2048, 30 / 2048), (30 / 2048, 70 / 2048), (70 / 2048, 150 / 2048),
    (70 / 2048, 250 / 2048), (150 / 2048, 250 / 2048), (0.0, 0.5),
]
D6_DELAYS = np.array([0, 1, 3, 5])
D7_PHASE = np.array([0.0, np.pi / 4, -np.pi / 2])

warnings.simplefilter("ignore", RuntimeWarning)


# ----------------------------------------------------------------------------- core reference
def preprocess(x, path):
    """kernel: untouched; pipeline: z-score ddof=1; pipeline-detrend: linear detrend then z-score
    (the order of pyspi/data.py:184-195, written here with numpy/scipy primitives)."""
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return x
    if path == "pipeline-detrend":
        x = scipy.signal.detrend(x, axis=1, type="linear")
    elif path != "pipeline":
        raise ValueError(path)
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def coherency_ref(x, n_fft=None):
    """Per-bin complex coherency on the retained non-negative grid: (f, C[f, i, j])."""
    X, fr = ks.multitaper_fft(x, NW=3.0, K=None, n_fft=n_fft)
    S = ks.cross_spectral_matrix(X, nonnegative=True)
    return ks.nonneg(fr), ks.coherency(S), S


def measure(C, unit, form="library"):
    if unit == "cohmag":
        return np.abs(C) ** (2 if form == "library" else 1)
    if unit == "phase":
        return np.angle(C)
    if unit == "icoh":
        return np.abs(C.imag) if form == "library" else C.imag
    raise ValueError(unit)


def reduce(v, f, fmin, fmax, stat, inclusive=True, drop_dc=False):
    m = ks.band_mask(f, fmin, fmax, inclusive=inclusive)
    if drop_dc:
        m &= f > 0
    sel = v[m]
    P = v.shape[-1]
    if sel.shape[0] == 0:
        if stat == "max":
            raise ValueError("zero-size array (empty band)")
        out = np.full((P, P), np.nan)
    else:
        out = (np.nanmean if stat == "mean" else np.nanmax)(sel, axis=0)
    out = np.array(out, dtype=np.float64, copy=True)
    np.fill_diagonal(out, np.nan)
    return out


def circular_mean(v_angle, f, fmin, fmax):
    m = ks.band_mask(f, fmin, fmax)
    out = np.angle(np.exp(1j * v_angle[m]).sum(axis=0))
    np.fill_diagonal(out, np.nan)
    return out


def offdiag(A):
    return A[~np.eye(A.shape[0], dtype=bool)]


def c_fw(m):
    """se multiplier so that the compare rule |d| <= 2 se is a familywise 5 % test over m entries."""
    return float(scipy.stats.norm.ppf(1 - 0.05 / (2 * max(int(m), 1))) / 2.0)


# ----------------------------------------------------------------------------- data generators
def gen_iid(T, rng):
    return rng.standard_normal((5, T))


def gen_bivariate(T, rng):
    z = rng.standard_normal((4, T))
    x = np.empty_like(z)
    for k, rho in enumerate((0.3, 0.8)):
        x[2 * k] = z[2 * k]
        x[2 * k + 1] = rho * z[2 * k] + np.sqrt(1 - rho ** 2) * z[2 * k + 1]
    return x


def gen_var2(T, rng, burn=1000):
    A1 = np.array([[0.5, 0.0], [0.5, 0.4]])
    A2 = np.array([[-0.3, 0.0], [-0.2, -0.2]])
    L = np.linalg.cholesky(np.diag([1.0, 0.8]))
    n = T + burn
    e = rng.standard_normal((n, 2)) @ L.T
    x = np.zeros((n, 2))
    for t in range(2, n):
        x[t] = e[t] + A1 @ x[t - 1] + A2 @ x[t - 2]
    return np.ascontiguousarray(x[burn:].T)


def gen_delayed(T, rng):
    m = int(D6_DELAYS.max())
    base = rng.standard_normal(T + m)
    x = np.empty((4, T))
    x[0] = base[m:]
    for k, d in enumerate(D6_DELAYS[1:], start=1):
        x[k] = base[m - d:m - d + T] + 0.1 * rng.standard_normal(T)
    return x


def gen_narrowband(f0):
    def g(T, rng):
        t = np.arange(T, dtype=np.float64)
        x = np.stack([np.sin(2 * np.pi * f0 * t + ph) for ph in D7_PHASE])
        return x + 0.05 * rng.standard_normal(x.shape)
    return g


GENERATORS = {"iid_gaussian": gen_iid, "bivariate_gaussian": gen_bivariate, "var2_bivariate": gen_var2,
              "delayed_noise": gen_delayed, "narrowband_f0-0.25": gen_narrowband(0.25)}


@lru_cache(maxsize=None)
def mc(dataset, T):
    """Monte Carlo of THIS reference on fresh draws of the dataset's construction.
    Returns {(unit, form, stat, band): array (N_MC, P, P)} for the config bands."""
    rng = np.random.default_rng(MC_SEED + 7919 * T + sum(map(ord, dataset)))
    gen = GENERATORS[dataset]
    out = {}
    for r in range(N_MC):
        f, C, _ = coherency_ref(gen(T, rng))
        for unit in ("cohmag", "phase", "icoh"):
            forms = ("library", "canonical") if unit != "phase" else ("library",)
            for form in forms:
                v = measure(C, unit, form)
                for stat in ("mean", "max"):
                    for band, (lo, hi) in BANDS.items():
                        out.setdefault((unit, form, stat, band), []).append(reduce(v, f, lo, hi, stat))
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- helpers
def load_x(run):
    x = np.asarray(make_data.load(run["dataset"], run["T"])["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    return x


def pyspi_npz(rid):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return {k: z[k] for k in z.files}


def parse(run):
    parts = run["run_id"].split("|")
    tag = parts[4]
    check, _, sb = tag.partition(":")
    stat, _, band = sb.partition("-")
    return check, stat, band, run["unit_id"].split(".")[1]


def impl_value(x, unit, stat, band, path="kernel", form="library", n_fft=None, inclusive=True, drop_dc=False):
    f, C, _ = coherency_ref(preprocess(x, path), n_fft=n_fft)
    lo, hi = BANDS[band]
    return reduce(measure(C, unit, form), f, lo, hi, stat, inclusive=inclusive, drop_dc=drop_dc)


def maxabs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) & np.isnan(b))
    if not m.any():
        return 0.0
    d = np.abs(a[m] - b[m])
    return float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")


def save(rid, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in kw.items()}
    np.savez(OUT_REF / f"{rid}.npz", **arrays)


def sym_keys(A, unit):
    return dict(pyspi_sym_maxabs=float(np.nanmax(np.abs(A - A.T))),
                pyspi_antisym_maxabs=float(np.nanmax(np.abs(A + A.T))))


DETAIL_IMPL = (f"{SELF}:impl_value (numpy.fft + scipy.signal.windows.dpss via _kernels_spectral."
               "multitaper_fft/cross_spectral_matrix/coherency; scipy " + scipy.__version__ +
               ", numpy " + np.__version__ + "); Thomson 1982 multitaper, unweighted taper mean")


# ----------------------------------------------------------------------------- checks
def do_impl(run):
    check, stat, band, unit = parse(run)
    x = load_x(run)
    path = run["path"]
    V = impl_value(x, unit, stat, band, path)
    A = pyspi_npz(run["run_id"])
    S = A["value"]
    extra = sym_keys(S, unit)
    xin = preprocess(x, path)
    extra["input_maxabs_vs_pyspi_input"] = maxabs(xin, A["input"])
    f, C, Sm = coherency_ref(xin)
    lo, hi = BANDS[band]
    extra["n_bins"] = int(ks.band_mask(f, lo, hi).sum())
    try:
        extra["alt_strict_band_maxabs"] = maxabs(S, impl_value(x, unit, stat, band, path, inclusive=False))
    except ValueError:
        extra["alt_strict_band_maxabs"] = float("nan")
    if lo == 0.0:
        extra["alt_no_dc_maxabs"] = maxabs(S, impl_value(x, unit, stat, band, path, drop_dc=True))
    if unit == "cohmag":
        extra["alt_unsquared_maxabs"] = maxabs(S, impl_value(x, unit, stat, band, path, form="canonical"))
    if unit == "icoh":
        extra["alt_signed_maxabs"] = maxabs(S, impl_value(x, unit, stat, band, path, form="canonical"))
        extra["dc_abs_imag_coherency_max"] = float(np.nanmax(np.abs(C[0].imag)))
    if unit == "phase" and stat == "mean":
        extra["alt_circular_maxabs"] = maxabs(S, circular_mean(np.angle(C), f, lo, hi))
        extra["n_pairs_negative_real_dc"] = int(np.triu(Sm[0].real < 0, 1).sum())
    if run["dataset"] == "trending":
        # distance from the same draw without the trend (bivariate_gaussian, z-scored)
        xb = np.asarray(make_data.load("bivariate_gaussian", run["T"])["x"])
        U = impl_value(xb, unit, stat, band, "pipeline")
        extra["maxabs_vs_untrended_draw"] = maxabs(S, U)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-10,
         reference_kind="independent-impl", reference_detail=DETAIL_IMPL,
         notes=f"{unit} {stat} band {band} ({lo}, {hi}) closed; path {path}", **extra)


def do_invariance(run):
    check, stat, band, unit = parse(run)
    rid_b = f"{run['unit_id']}|bivariate_gaussian|{run['T']}|kernel|impl:{stat}-{band}"
    Vb = pyspi_npz(rid_b)["value"]
    S = pyspi_npz(run["run_id"])["value"]
    ref = impl_value(load_x(run), unit, stat, band)
    save(run["run_id"], value=Vb, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="same-lineage",
         reference_detail=f"{SELF}:do_invariance: pyspi-vs-pyspi self-consistency; coherency is invariant to "
                          f"per-channel scale and offset (offset removed by the constant detrend); value = pyspi run {rid_b}",
         notes="scaled_offset x = s * bivariate_gaussian + o (same draw), kernel path (no z-score)",
         independent_ref_maxabs=maxabs(S, ref))


def do_perm(run):
    check, stat, band, unit = parse(run)
    p = list(run["processes"])
    rid0 = f"{run['unit_id']}|delayed_noise|{run['T']}|kernel|impl:{stat}-{band}"
    A0 = pyspi_npz(rid0)["value"]
    save(run["run_id"], value=A0[np.ix_(p, p)], criterion="exact", rtol=1e-10, atol=1e-12,
         reference_kind="same-lineage",
         reference_detail=f"{SELF}:do_perm: pyspi-vs-pyspi self-consistency, permutation equivariance A(X[p]) = A(X)[p][:, p]; A(X) = pyspi run {rid0}",
         notes=f"processes {p}")


def do_identical(run):
    check, stat, band, unit = parse(run)
    val = {"cohmag": 1.0, "phase": 0.0, "icoh": 0.0}[unit]
    V = np.array([[np.nan, val], [val, np.nan]])
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:do_identical: x2 == x0 bit for bit (make_data D14) => S_02 = S_00 real, C_02 = 1",
         notes="shared_envelope processes [0, 2], pipeline path")


def f0_analytic(unit):
    d = D7_PHASE[:, None] - D7_PHASE[None, :]
    C = np.exp(1j * d)
    v = {"cohmag": np.abs(C) ** 2, "phase": np.angle(C), "icoh": np.abs(C.imag)}[unit]
    v = np.array(v, dtype=float)
    np.fill_diagonal(v, np.nan)
    return v


def do_f0bin(run):
    check, stat, band, unit = parse(run)
    T = run["T"]
    Va = f0_analytic(unit)
    rng = np.random.default_rng(MC_SEED + T)
    gen = gen_narrowband(0.25)
    devs = []
    for _ in range(200):
        f, C, _ = coherency_ref(gen(T, rng))
        devs.append(reduce(measure(C, unit), f, 0.25, 0.25, "mean") - Va)
    rms = np.sqrt(np.nanmean(np.array(devs) ** 2, axis=0))
    atol = 4.0 * float(np.nanmax(rms))
    x = load_x(run)
    f, C, _ = coherency_ref(x)
    ref_same = reduce(measure(C, unit), f, 0.25, 0.25, "mean")
    S = pyspi_npz(run["run_id"])["value"]
    save(run["run_id"], value=Va, criterion="tolerance", rtol=0.0, atol=atol, reference_kind="analytic",
         reference_detail=f"{SELF}:do_f0bin: x_k = sin(2 pi 0.25 t + phi_k) + 0.05 noise, phi = (0, pi/4, -pi/2) "
                          "(make_data D7) => C_ij(f0) = exp(i(phi_i - phi_j)); atol = 4 x Monte Carlo RMS "
                          "deviation of the estimator from it (200 fresh noise draws)",
         notes="pyspi custom band fmin = fmax = 0.25 selects exactly the f0 bin (closed band); "
               "a strict band would be empty",
         mc_rms_max=float(np.nanmax(rms)), independent_ref_maxabs=maxabs(S, ref_same),
         analytic_offdiag=offdiag(Va))


def do_null(run):
    check, stat, band, unit = parse(run)
    T = run["T"]
    M = mc("iid_gaussian", T)[(unit, "library", stat, band)]
    P = M.shape[1]
    vals = np.array([offdiag(m) for m in M])
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    m_entries = P * (P - 1) if (unit == "phase" and stat == "max") else P * (P - 1) // 2
    V = np.full((P, P), mu)
    np.fill_diagonal(V, np.nan)
    se = np.full((P, P), sd * c_fw(m_entries))
    S = pyspi_npz(run["run_id"])["value"]
    pair_avg = vals.mean(axis=1)
    z_avg = (float(np.nanmean(offdiag(S))) - pair_avg.mean()) / pair_avg.std(ddof=1)
    extra = dict(mc_mean=mu, mc_sd=sd, n_mc=N_MC, familywise_m=m_entries,
                 pyspi_offdiag_mean=float(np.nanmean(offdiag(S))), z_pair_average=float(z_avg),
                 independent_ref_maxabs=maxabs(S, impl_value(load_x(run), unit, stat, band)))
    if unit == "cohmag":
        extra["analytic_per_bin_null_mean_1_over_K"] = 1.0 / K_TAPERS
    save(run["run_id"], value=V, se=se, criterion="statistical", reference_kind="independent-impl",
         reference_detail=f"{SELF}:mc('iid_gaussian', T) Monte Carlo ({N_MC} fresh iid N(0,1) draws) of the "
                          "independent reference; per-bin |C|^2 null Beta(1, K-1), mean 1/K (K = 5)",
         notes=f"null level {unit} {stat} {band}; se = MC sd x z(1-0.05/(2m))/2, m = {m_entries}",
         **extra)


def population(run, unit, stat, band, form="library"):
    """Canonical population band value on the same FFT grid and band rule; returns (V, judged mask)."""
    T = run["T"]
    n_fft = ks.next_fast_len_235(T)
    f = ks.nonneg(np.fft.fftfreq(n_fft))
    if run["dataset"] == "var2_bivariate":
        d = make_data.load("var2_bivariate", T)
        _, _, St = ks.var_transfer_function([d["A1"], d["A2"]], f, d["Sigma"])
        Ct = ks.coherency(St)
        judged = ~np.eye(2, dtype=bool)
    else:
        Ct = np.zeros((f.size, 4, 4), dtype=complex)
        for (i, j), rho in zip(((0, 1), (2, 3)), (0.3, 0.8)):
            Ct[:, i, j] = Ct[:, j, i] = rho
        judged = np.zeros((4, 4), dtype=bool)
        for (i, j) in ((0, 1), (2, 3)):
            judged[i, j] = judged[j, i] = True
        if unit != "phase":
            judged = ~np.eye(4, dtype=bool)   # |C|^2 = 0 and Im C = 0 on the uncoupled pairs are defined
    lo, hi = BANDS[band]
    V = reduce(measure(Ct, unit, form), f, lo, hi, stat)
    V[~judged & ~np.eye(V.shape[0], dtype=bool)] = 0.0
    return V, judged, f, Ct


def stored_S_true_check(T):
    d = make_data.load("var2_bivariate", T)
    _, _, S = ks.var_transfer_function([d["A1"], d["A2"]], d["freqs"], d["Sigma"])
    return float(np.abs(S - d["S_true"]).max())


def do_pop_or_mcexp(run):
    check, stat, band, unit = parse(run)
    ds, T = run["dataset"], run["T"]
    M = mc(ds, T)[(unit, "library", stat, band)]
    mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0, ddof=1)
    Vp, judged, f, Ct = population(run, unit, stat, band)
    m_entries = int(np.triu(judged, 1).sum()) * (2 if unit == "phase" else 1)
    se = np.where(judged, sd * c_fw(m_entries), np.inf)
    np.fill_diagonal(se, np.nan)
    S = pyspi_npz(run["run_id"])["value"]
    extra = dict(n_mc=N_MC, familywise_m=m_entries, mc_sd_max=float(np.nanmax(np.where(judged, sd, np.nan))),
                 population_offdiag=offdiag(Vp)[:4], mc_mean_offdiag=offdiag(mu)[:4], pyspi_offdiag=offdiag(S)[:4],
                 bias_mc_minus_population_max=maxabs(np.where(judged, mu, np.nan), np.where(judged, Vp, np.nan)),
                 independent_ref_maxabs=maxabs(S, impl_value(load_x(run), unit, stat, band)),
                 mc_quantile_of_pyspi_offdiag=offdiag((M <= S[None]).mean(axis=0))[:4])
    if ds == "var2_bivariate":
        extra["S_true_rebuilt_vs_stored_maxabs"] = stored_S_true_check(T)
    if unit == "cohmag":
        extra["carter_bias_prediction_offdiag"] = offdiag(Vp + (1 - Vp) ** 2 / K_TAPERS)[:4]
    if unit == "phase":
        lo, hi = BANDS[band]
        extra["population_circular_mean_offdiag"] = offdiag(circular_mean(np.angle(Ct), f, lo, hi))[:4]
    if unit == "icoh":
        Vs, _, _, _ = population(run, unit, stat, band, form="canonical")
        extra["population_signed_offdiag"] = offdiag(Vs)[:4]
    if check == "pop":
        save(run["run_id"], value=Vp, se=se, criterion="statistical", reference_kind="analytic",
             reference_detail=(f"{SELF}:population: " + ("S(f) = H Sigma H^H (VAR(2) A1, A2, Sigma of make_data D5; "
                               "_kernels_spectral.var_transfer_function) on the FFT grid" if ds == "var2_bivariate"
                               else "C = rho (0.3, 0.8) at every f for the coupled pairs, 0 otherwise (make_data D2)")
                               + "; canonical quantity, same bins and band rule; se from the estimator's Monte Carlo sd"),
             notes=f"canonical population vs pyspi ({unit} {stat} {band}, T={T}); entries not judged have se=inf",
             **extra)
    else:
        save(run["run_id"], value=mu, se=se, criterion="statistical", reference_kind="independent-impl",
             reference_detail=f"{SELF}:mc('{ds}', {T}): Monte Carlo mean of the independent estimator "
                              f"({N_MC} fresh draws of the make_data construction)",
             notes=f"expectation of the estimator as coded ({unit} {stat} {band}, T={T})", **extra)


def do_eeg(run):
    check, stat, band, unit = parse(run)
    x = load_x(run)
    xin = preprocess(x, run["path"])
    f, C, _ = coherency_ref(xin)
    v = measure(C, unit)
    V = np.stack([reduce(v, f, lo, hi, "mean") for lo, hi in EEG_BANDS])
    try:
        Vs = np.stack([reduce(v, f, lo, hi, "mean", inclusive=False) for lo, hi in EEG_BANDS])
    except ValueError:
        Vs = np.full_like(V, np.nan)
    A = pyspi_npz(run["run_id"])
    nb = [int(ks.band_mask(f, lo, hi).sum()) for lo, hi in EEG_BANDS]
    nbs = [int(ks.band_mask(f, lo, hi, inclusive=False).sum()) for lo, hi in EEG_BANDS]
    T = x.shape[1]
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
         reference_detail=DETAIL_IMPL + "; EEG band edges Hz/2048 from pyspi/workflow/retained185.yaml:227-247",
         notes="stack of the 11 retained EEG identifiers in units.yaml order; detrend + z-score",
         n_bins_inclusive=nb[:10], n_bins_strict=nbs[:10], n_bins_full=nb[10],
         alt_strict_band_maxabs=maxabs(A["value"], Vs),
         empty_bands=[i for i, n in enumerate(nb) if n == 0],
         half_bandwidth_W_hz_at_2048=3.0 / ks.next_fast_len_235(T) * 2048.0,
         fft_bin_spacing_hz_at_2048=2048.0 / ks.next_fast_len_235(T))


def do_emptymax(run):
    check, stat, band, unit = parse(run)
    code, n_bins_py = pyspi_npz(run["run_id"])["value"]
    f = ks.nonneg(np.fft.fftfreq(ks.next_fast_len_235(run["T"])))
    kw = run["probe"]["kwargs"]
    n_ref = int(ks.band_mask(f, kw["fmin"], kw["fmax"]).sum())
    ok = (code == 1.0) and n_ref == 0 and int(n_bins_py) == 0
    save(run["run_id"], value=np.array([1.0, 0.0]), criterion="qualitative", pass_flag=int(ok),
         reference_kind="analytic",
         reference_detail=f"{SELF}:do_emptymax: grid k/{ks.next_fast_len_235(run['T'])} has no bin in "
                          "[1/2048, 4/2048]; np.nanmax over an empty axis raises ValueError",
         notes="pass = pyspi raised ValueError('zero-size array ...') for statistic=max on an empty band",
         pyspi_code=float(code), ref_n_bins=n_ref)


def do_nfft(run):
    check, stat, band, unit = parse(run)
    T_sub = int(run["probe"]["kwargs"]["T_sub"])
    x = load_x(run)[:, :T_sub]
    V = impl_value(x, unit, stat, band)
    S = pyspi_npz(run["run_id"])["value"]
    alt = impl_value(x, unit, stat, band, n_fft=scipy.fft.next_fast_len(T_sub))
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
         reference_detail=DETAIL_IMPL + "; n_fft = 5-smooth next_fast_len (scipy.fftpack rule)",
         notes=f"T = {T_sub}: n_fft = {ks.next_fast_len_235(T_sub)} (scipy.fft.next_fast_len gives "
               f"{scipy.fft.next_fast_len(T_sub)})",
         n_fft=ks.next_fast_len_235(T_sub), alt_scipy_fft_rule_maxabs=maxabs(S, alt))


def do_grid(run):
    check, stat, band, unit = parse(run)
    T_sub = int(run["probe"]["kwargs"]["T_sub"])
    n_fft = ks.next_fast_len_235(T_sub)
    f = ks.nonneg(np.fft.fftfreq(n_fft))
    save(run["run_id"], value=f, criterion="exact", rtol=0.0, atol=1e-15, reference_kind="independent-impl",
         reference_detail=f"{SELF}:do_grid: nonneg(numpy.fft.fftfreq(next_fast_len_235(T)))",
         notes=f"T = {T_sub}: n_fft = {n_fft}, {f.size} bins, DC included, top bin {f[-1]:.6f} (fs/2 excluded)",
         n_bins=int(f.size), top_bin=float(f[-1]), documented_bins=125, documented_f0=4.0 / T_sub)


def do_mne(run):
    import mne
    import mne_connectivity
    from mne_connectivity import spectral_connectivity_epochs
    check, stat, band, unit = parse(run)
    x = load_x(run)
    T = x.shape[1]
    xd = x - x.mean(axis=1, keepdims=True)
    P = x.shape[0]
    res = {}
    for meth in ("coh", "imcoh", "cohy"):
        con = spectral_connectivity_epochs(xd[None], method=meth, mode="multitaper", sfreq=1.0, fmin=0.0, fmax=0.5,
                                           mt_bandwidth=6.0 / T, mt_adaptive=False, mt_low_bias=True,
                                           faverage=False, verbose=False)
        d = con.get_data(output="dense")          # [a, b] (a > b) holds the pair; upper triangle 0
        fm = np.asarray(con.freqs)
        full = np.zeros((fm.size, P, P), dtype=d.dtype)
        for a in range(P):
            for b in range(a):
                full[:, a, b] = d[a, b]
                full[:, b, a] = np.conj(d[a, b]) if meth == "cohy" else (-d[a, b] if meth == "imcoh" else d[a, b])
        res[meth] = (fm, full)
    fm = res["coh"][0]
    f = ks.nonneg(np.fft.fftfreq(ks.next_fast_len_235(T)))
    keep = np.isin(np.round(fm * T), np.round(f * T))     # drop mne's Nyquist bin
    lo, hi = BANDS[band]

    def bm(v):
        return reduce(v[keep], fm[keep], lo, hi, "mean")

    if unit == "cohmag":
        lib, can = bm(res["coh"][1] ** 2), bm(res["coh"][1])
        lib_name, can_name = "mean(mne coh^2)", "mean(mne coh) (unsquared)"
    elif unit == "icoh":
        lib, can = bm(np.abs(res["imcoh"][1])), bm(res["imcoh"][1])
        lib_name, can_name = "mean|mne imcoh|", "mean(mne imcoh) (signed)"
    else:
        ang = np.angle(res["cohy"][1][keep])
        lib = bm(np.angle(res["cohy"][1]))
        can = circular_mean(ang, fm[keep], lo, hi)
        lib_name, can_name = "arithmetic mean of angle(mne cohy)", "circular mean of angle(mne cohy)"
    S = pyspi_npz(run["run_id"])["value"]
    d_lib, d_can = maxabs(S, lib), maxabs(S, can)
    ok = d_lib < 0.25 * d_can
    save(run["run_id"], value=lib, criterion="qualitative", pass_flag=int(ok), reference_kind="external-pkg",
         reference_detail=f"mne-connectivity=={mne_connectivity.__version__} spectral_connectivity_epochs "
                          f"(mode multitaper, mt_bandwidth 6/T, adaptive False, low_bias True; mne=={mne.__version__}); "
                          "mne weights tapers by their eigenvalues (sqrt(eig) per coefficient), the library uses unit weights",
         notes=f"pass = max|pyspi - {lib_name}| < 0.25 x max|pyspi - {can_name}|",
         maxabs_vs_library_form=d_lib, maxabs_vs_canonical_form=d_can, n_bins=int(keep.sum()))


def do_dcpi(run):
    check, stat, band, unit = parse(run)
    S = pyspi_npz(run["run_id"])["value"]
    x = load_x(run)
    f, C, Sm = coherency_ref(x)
    lo, hi = BANDS[band]
    n = int(ks.band_mask(f, lo, hi).sum())
    neg = (Sm[0].real < 0).astype(float) if lo == 0.0 else np.zeros(Sm.shape[1:])
    V = -S.T + 2 * np.pi * neg / n
    np.fill_diagonal(V, np.nan)
    save(run["run_id"], value=V, criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="same-lineage",
         reference_detail=f"{SELF}:do_dcpi: pyspi-vs-pyspi identity (value built from -A.T of the same pyspi run "
                          "plus an analytic DC term); angle(conj z) = -angle(z) except on the negative real axis, where "
                          "numpy.angle returns +pi for a +0 imaginary part in both orientations; the DC cross-"
                          "spectrum of real input is real => A[i,j] + A[j,i] = 2 pi [Re S_ij(0) < 0] / n_bins",
         notes=f"value = -A.T(pyspi) + 2 pi neg_dc / {n}; neg_dc from the independent reference",
         n_bins=n, n_pairs_negative_real_dc=int(np.triu(neg, 1).sum()),
         pyspi_antisym_maxabs=float(np.nanmax(np.abs(S + S.T))))


def do_circular(run):
    check, stat, band, unit = parse(run)
    x = load_x(run)
    f, C, _ = coherency_ref(x)
    lo, hi = BANDS[band]
    V = circular_mean(np.angle(C), f, lo, hi)
    S = pyspi_npz(run["run_id"])["value"]
    extra = dict(arith_ref_maxabs=maxabs(S, reduce(np.angle(C), f, lo, hi, "mean")))
    if run["dataset"] == "delayed_noise":
        extra["pyspi_A30"] = float(S[3, 0]); extra["circular_A30"] = float(V[3, 0])
        extra["pyspi_A20"] = float(S[2, 0]); extra["circular_A20"] = float(V[2, 0])
    else:
        Vp, _, fp, Ct = population(run, unit, "mean", band)
        extra["population_arith_A01"] = float(Vp[0, 1])
        extra["population_circular_A01"] = float(circular_mean(np.angle(Ct), fp, lo, hi)[0, 1])
        extra["pyspi_A01"] = float(S[0, 1]); extra["circular_A01"] = float(V[0, 1])
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
         reference_detail=f"{SELF}:circular_mean: arg sum_f exp(i angle C(f)) over the same bins "
                          "(circular mean direction; Mardia & Jupp 2000, section not re-checked)",
         notes="canonical angle average vs pyspi's arithmetic mean of principal angles (REVIEW M4)", **extra)


def delay_analytic(f, lo, hi):
    d = D6_DELAYS
    m = ks.band_mask(f, lo, hi)
    ph = 2 * np.pi * f[m][:, None, None] * (d[None, None, :] - d[None, :, None])   # A[i,j]: phi_i - phi_j
    V = np.angle(np.exp(1j * ph)).mean(axis=0)
    np.fill_diagonal(V, np.nan)
    return V


def do_delay(run):
    check, stat, band, unit = parse(run)
    T = run["T"]
    f = ks.nonneg(np.fft.fftfreq(ks.next_fast_len_235(T)))
    lo, hi = BANDS[band]
    V = delay_analytic(f, lo, hi)
    M = mc("delayed_noise", T)[("phase", "library", "mean", band)]
    mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0, ddof=1)
    S = pyspi_npz(run["run_id"])["value"]
    if check == "delay":
        se = sd * c_fw(6)
        save(run["run_id"], value=V, se=se, criterion="statistical", reference_kind="analytic",
             reference_detail=f"{SELF}:delay_analytic: x_k(t) = x0(t - d_k) => angle S_ij(f) = 2 pi f (d_j - d_i), "
                              "principal value, arithmetic band mean over the same bins; se = Monte Carlo sd x "
                              "z(1-0.05/12)/2 (6 distinct pairs)",
             notes="i leads j (d_i < d_j) gives positive values where the phase does not wrap",
             mc_minus_analytic_maxabs=maxabs(mu, V), mc_sd_max=float(np.nanmax(sd)),
             pyspi_A01=float(S[0, 1]), pyspi_A10=float(S[1, 0]), analytic_A01=float(V[0, 1]),
             pyspi_A03=float(S[0, 3]), analytic_A03=float(V[0, 3]), mc_mean_A03=float(mu[0, 3]))
    else:
        dd = D6_DELAYS[None, :] - D6_DELAYS[:, None]
        nowrap = np.abs(dd) * hi <= 0.5
        sign = np.where(nowrap, np.sign(dd), 0).astype(float)
        np.fill_diagonal(sign, 0)
        save(run["run_id"], value=V, expected_sign=sign, criterion="direction", reference_kind="analytic",
             reference_detail=f"{SELF}:do_delay: sign of 2 pi f (d_j - d_i) where |d_j - d_i| fmax <= 1/2 (no wrap)",
             notes="expected_sign +1 at [i, j] when x_i leads x_j (x_j is x_i delayed), 0 = not judged (wraps)",
             n_judged=int((sign != 0).sum()))


def do_const(run):
    check, stat, band, unit = parse(run)
    kw = run["probe"]["kwargs"]
    x = load_x(run)
    x[int(kw["channel"])] = float(kw["value"])
    f, C, Sm = coherency_ref(x)
    V = reduce(measure(C, unit), f, *BANDS[band], stat)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
         reference_detail=DETAIL_IMPL + "; zero normaliser -> NaN (_kernels_spectral.coherency)",
         notes=f"channel {kw['channel']} = {kw['value']} (constant): expected NaN row/column, other pairs unchanged",
         ref_power_constant_channel_max=float(np.abs(Sm[:, kw['channel'], kw['channel']]).max()))


def do_signed(run):
    check, stat, band, unit = parse(run)
    x = load_x(run)
    f, C, _ = coherency_ref(x)
    lo, hi = BANDS[band]
    V = reduce(C.imag, f, lo, hi, "mean")
    S = pyspi_npz(run["run_id"])["value"]
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
         reference_detail=f"{SELF}:do_signed: Nolte et al. 2004 signed ImCoh = Im C_ij, band mean, same bins",
         notes="canonical (signed, antisymmetric) vs pyspi |Im C|", pyspi_A01=float(S[0, 1]), pyspi_A10=float(S[1, 0]),
         signed_A01=float(V[0, 1]), signed_A10=float(V[1, 0]), pyspi_A03=float(S[0, 3]), signed_A03=float(V[0, 3]),
         abs_of_signed_mean_A03=float(abs(V[0, 3])))


def do_unsquared(run):
    check, stat, band, unit = parse(run)
    x = load_x(run)
    V = impl_value(x, unit, stat, band, form="canonical")
    S = pyspi_npz(run["run_id"])["value"]
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-10, atol=1e-10, reference_kind="independent-impl",
         reference_detail=f"{SELF}:do_unsquared: band mean of |C| (coherence coefficient, Bastos & Schoffelen 2016)",
         notes="canonical unsquared |C| vs pyspi; rho = (0.3, 0.8) on pairs (0,1), (2,3)",
         pyspi_A01=float(S[0, 1]), unsquared_A01=float(V[0, 1]), pyspi_A23=float(S[2, 3]),
         unsquared_A23=float(V[2, 3]), squared_ref_A01=float(impl_value(x, unit, stat, band)[0, 1]))


HANDLERS = {"impl": do_impl, "invariance": do_invariance, "perm": do_perm, "identical": do_identical,
            "f0bin": do_f0bin, "null": do_null, "pop": do_pop_or_mcexp, "mcexp": do_pop_or_mcexp,
            "eeg": do_eeg, "emptymax": do_emptymax, "mne": do_mne, "dcpi": do_dcpi, "circular": do_circular,
            "delay": do_delay, "direction": do_delay, "signed": do_signed, "unsquared": do_unsquared}


def main():
    runs = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())["runs"]
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in runs:
        rid = run["run_id"]
        check = parse(run)[0]
        if only and check not in only:
            continue
        key = next((p for p in ("nfft", "grid", "const") if check.startswith(p)), check)
        fn = {"nfft": do_nfft, "grid": do_grid, "const": do_const}.get(key) or HANDLERS[key]
        try:
            fn(run)
            n_ok += 1
        except Exception as exc:   # report and continue; a missing reference shows up in compare
            n_err += 1
            print(f"ERROR {rid}: {exc!r}", file=sys.stderr)
    print(f"{FAMILY} references: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
