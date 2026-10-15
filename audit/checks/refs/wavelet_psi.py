#!/usr/bin/env python3
"""References for the wavelet_psi family (unit ``wavelet.psi``).

Runs in the COMPARISON venv.  The replica and every analytic reference use
numpy/scipy only (no mne, no mne_connectivity); the single ``sl`` run imports
mne_connectivity in this venv to show what the library returns and is labelled
same-lineage.  For every run of ``manifest/wavelet_psi.json`` it writes
``out/ref/<run_id>.npz`` in the contract of ``compare.py``.  The run's tag (the
last ``|`` field, ``tag:variant``) selects the check; variants are
mean|max x full|low|high = the six config identifiers.

What pyspi computes (trace, notes/wavelet_psi.md):
  W_p(f, t)   Morlet CWT, f = linspace(0.2, 0.5, 10) restricted to f >= 5/T
              (mne_connectivity default fmin, spectral/epochs.py:97-113, 141-153),
              n_cycles = 7, zero-mean, 'same' centring of the full linear
              convolution (mne tfr.py:322-400, _centered tfr.py:4035-4043).
  C_ij(f, t)  W_i conj(W_j) / (|W_i| |W_j|)   (one epoch; |C| = 1)
  psi_ij(t)   Im sum_k conj(C(f_k, t)) C(f_{k+1}, t)  (adjacent pairs, all kept f)
  A[i, j]     stat(psi_ij(t) for t in freq_id), freq_id = indices k of the kept
              frequencies with fmin <= f_k <= fmax; computed for i > j and
              mirrored, NaN diagonal (wavelet.py:219-237).

Tags
  rep      independent replica of the above: closed-form Morlet wavelets
           (_kernels_spectral.morlet_wavelet), DIRECT time-domain np.convolve
           (full) centred as mne does, single-epoch coherency, adjacent-pair
           PSI (_kernels_spectral.psi_adjacent), pyspi's time-index selection,
           mirror.  tolerance rtol 1e-8, atol 1e-10.
  doc      the documented quantity (supplement 1.5.4): mean or max over ALL
           time points of the per-time PSI restricted to the wavelet
           frequencies inside [fmin, fmax], in pyspi's orientation.  Expected
           to FAIL (axis selection, KNOWN_ISSUES).  Extra keys: the same over
           the cone-of-influence-free window and the differences.
  dir      direction against the truth in pyspi's orientation A[i, j] =
           A[j, i] = PSI(max(i, j) -> min(i, j)): delayed_noise and
           logistic_maps_chaotic analytic (higher index lags: expected -1);
           var2_* from the population Nolte PSI (Nolte 2008 eq. 3) on the
           wavelet grid in the band, coherency from make_data.var_spectral_matrix.
           Extra keys: central_window_agree = the same statistic over the
           same number of samples centred at T//2 (edge control), and the
           sign agreement with the truth of the INTENDED quantity
           (psi_adjacent on the time-averaged wavelet coherency from
           _kernels_spectral.cwt_morlet, n_cycles 7, whole record and cone-free
           window) and of the documented all-time / cone-free per-time means.
  null     iid_gaussian: Monte Carlo null of pyspi's statistic (own FFT
           kernel, 4000 draws of two z-scored iid channels per T).  qualitative:
           pass iff every pair lies inside the empirical two-sided Bonferroni
           (0.05 / pairs) quantile interval of the MC null.
           Extra keys: mc_sd of pyspi's statistic and of the all-time mean.
  tsel     probe adds 100 to the cached PSI at one time index at a time; the
           reference predicts max |dA| from the replica (tolerance).
  xsel     probe replaces x[:, t0:] by 10 x N(0,1) (seed 1000 + t0); the
           reference recomputes the same perturbations with the replica
           (tolerance) and reports the dependence horizon.
  libinfo  [shape of the dense PSI, freqs_computed] exact.
  rawpsi   library PSI(t) (N, N, T) vs the replica's raw PSI (upper triangle
           and diagonal 0 as in mne's dense output), tolerance; extra keys:
           sum-of-sines identity and Nyquist phase from the replica.
  cohmod   |C| of the library coherency = 1 (exact, analytic).
  nyq      Im C(0.5, t) / |C(0.5, t)| = 0 (analytic, tolerance atol 1e-9).
  perm     probe returns A(X[::-1]) mapped back; reference = replica on the
           reversed input (tolerance); extra key max |A_perm + A| against
           pyspi's own plain run (rep run of the same identifier).
  short    probe: six identifiers on x[:, :20]; replica with the frequency
           drop, NaN where the selection is empty (tolerance, NaN mask).
  sl       mne_connectivity.phase_slope_index in the comparison venv plus
           pyspi's selection (same-lineage, tolerance 1e-12).
  winperm  probe reverses x[:, a:b] (pipeline path; channel mean/SD unchanged)
           and returns [six identifiers on x, on the perturbed x].  tail
           window (a = 37 > max selected t 9 + f=0.2 wavelet half-length 27):
           reference = replica of the UNPERTURBED input for both halves
           (prediction: the stored value does not change; tolerance).  head
           (t < 10) and mid (10 <= t < 37) windows: qualitative, pass iff every
           identifier changes by > 1e-6 AND pyspi equals the replica on both
           inputs within 1e-9.  Extra keys: the change of the documented
           all-time quantity under the same perturbation.
  int      the INTENDED quantity: Nolte eq. 3 adjacent-pair PSI
           (_kernels_spectral.psi_adjacent) on the time-averaged coherency of
           an own Morlet CWT (_kernels_spectral.cwt_morlet, 7 cycles) at the
           wavelet frequencies inside the band, pyspi orientation (expected to
           FAIL on value; extra keys give the sign agreement of the intended
           quantity and of pyspi with the truth).
  nullmc   probe: pyspi on 300 iid N(0,1) pairs per T -> [mean, SD] of A[1,0]
           per identifier; reference: own-kernel Monte Carlo (4000 draws) of
           the same statistic, statistical; se is scaled so that the 2-se rule
           is a two-sided Bonferroni test at family-wise 0.05 over the 36
           numbers of the three nullmc runs (declared before running).
  tailpert probe: six identifiers on x and on three perturbations of the
           tail x[:, 37:] (offset +0.5 on every channel; x[0, 37:] * 3; tail
           rescaled by 3 about its own mean, which keeps the channel mean and
           changes only the SD).  Reference: the replica on the same four
           inputs (tolerance).  Extra keys: the change per perturbation.  The
           kernel path predicts 0 for all three; the pipeline path predicts a
           change whenever the channel MEAN moves (the z-score subtracts it, and
           the zero-mean Morlet transform of a constant is not zero inside the
           zero-padded edge window) and none for the SD-only perturbation (a
           positive rescaling leaves the coherency phase unchanged); with
           detrend the fitted line also carries the tail.
  welch    proposed check 4 for delayed_noise: canonical Nolte eq. 3 sign from
           the analytic population spectrum of a pure delay (S_ij =
           s_i s_j exp(-2 pi i f (d_i - d_j)), noise 0.1 on x1..x3) on the
           Welch grid k/64 inside the band (0 and 0.5 excluded);
           value = numpy Welch estimate (periodic Hann, L = 64, 50% overlap,
           numpy.fft.rfft only) of the same PSI; direction criterion with
           expected_sign from the analytic population PSI.  Extra keys:
           Welch-vs-analytic sign agreement, analytic lower triangle.
  popband  var2_*: population Nolte PSI (make_data.var_spectral_matrix) over
           the band definitions [0, 0.5], [0, 0.25], [0.25, 0.5], [0.2, 0.5],
           [0.2, 0.25] on the grid k/64 (0 and 0.5 excluded) and over the
           wavelet grid in [0.2, 0.5] and [0.2, 0.25]; qualitative: pass iff
           every lower-triangle sign is the same in all seven band/grid
           definitions (the band choice cannot explain the direction misses).
  fscache  probe: mean (fmin 0, fmax 0.5) PSI with fs = 1, then fs = 2 on the
           same Data, then fs = 2 on a fresh Data.  qualitative: pass iff the
           shared-Data fs = 2 value equals the fs = 1 value bitwise (cache
           reuse), the fresh fs = 2 value differs by > 1e-6 and equals the
           fs = 2 replica (wavelets at f / fs cycles per sample, default
           fmin 5 fs / T) within 1e-9.
  base     scaled_offset pipeline vs the replica on its D2 base draw
           (bivariate_gaussian) pipeline: z-scoring removes scale and offset.

Paths: kernel = raw x; pipeline = scipy.stats.zscore(ddof=1); pipeline-detrend
= scipy.signal.detrend (linear) then zscore(ddof=1) (pyspi/data.py:184-195).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np  # noqa: E402
import scipy.signal  # noqa: E402
import scipy.stats  # noqa: E402

import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

FAMILY = "wavelet_psi"
CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCHANGE = CHECKS / "out" / "exchange" / FAMILY
SELF = "refs/wavelet_psi.py"
FREQS = np.linspace(0.2, 0.5, 10)
N_CYCLES = 7.0
VARIANTS = {
    "mean-full": ("mean", 0.0, 0.5), "mean-low": ("mean", 0.0, 0.25), "mean-high": ("mean", 0.25, 0.5),
    "max-full": ("max", 0.0, 0.5), "max-low": ("max", 0.0, 0.25), "max-high": ("max", 0.25, 0.5),
}
ORDER = list(VARIANTS)
REP_TOL = dict(rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------------ helpers
def preprocess(x, path):
    x = np.asarray(x, dtype=np.float64)
    if path == "kernel":
        return x.copy()
    if path == "pipeline-detrend":
        x = scipy.signal.detrend(x, axis=1)
    return scipy.stats.zscore(x, axis=1, ddof=1)


def save(run_id, **kw):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_REF / f"{run_id}.npz", **{k: np.asarray(v) for k, v in kw.items()})


def load_pyspi(run_id):
    with np.load(OUT_PY / f"{run_id}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def load_exchange(run_id):
    p = EXCHANGE / (run_id.replace("|", "__").replace("/", "_") + ".json")
    return json.loads(p.read_text()) if p.is_file() else {}


def freqs_kept(T):
    """mne_connectivity default fmin = 5 cycles / duration = 5/T at sfreq 1."""
    return FREQS[FREQS >= 5.0 / T]


def half_len(f):
    return (ks.morlet_wavelet(f, N_CYCLES).size - 1) // 2


# --------------------------------------------------------------- replica
def cwt_direct(x, freqs, sfreq=1.0):
    """Direct time-domain linear convolution with the closed-form Morlet wavelet,
    centre T samples of the full convolution (start (T + L - 1 - T) // 2)."""
    x = np.atleast_2d(x)
    P, T = x.shape
    out = np.empty((P, len(freqs), T), dtype=complex)
    for k, f in enumerate(freqs):
        W = ks.morlet_wavelet(f, N_CYCLES, sfreq)
        for p in range(P):
            full = np.convolve(x[p], W)
            s = (full.size - T) // 2
            out[p, k] = full[s:s + T]
    return out


def coherency_single(W):
    """C (n_f, P, P, T) = W_i conj(W_j) / (|W_i| |W_j|) for one epoch."""
    U = W / np.abs(W)
    C = U[:, None] * np.conj(U[None, :])            # (P, P, n_f, T)
    return np.moveaxis(C, 2, 0)


def raw_psi(W, freqs, fmin=-np.inf, fmax=np.inf, inclusive=False):
    """psi_ij(t) (P, P, T) = Im sum adjacent conj(C(f_k)) C(f_k+1); Psi_ij > 0: i leads."""
    return ks.psi_adjacent(coherency_single(W), freqs, fmin, fmax, inclusive=inclusive)


def orient(Aij):
    """pyspi orientation: lower triangle (seed i > target j) mirrored, NaN diagonal."""
    A = np.array(Aij, dtype=float)
    iu = np.triu_indices(A.shape[0], 1)
    A[iu] = A.T[iu]
    np.fill_diagonal(A, np.nan)
    return A


def select(raw, freqs, variant):
    """pyspi selection: time indices = indices of kept freqs in [fmin, fmax]."""
    st, fmin, fmax = VARIANTS[variant]
    sel = np.nonzero((freqs >= fmin) & (freqs <= fmax))[0]
    P = raw.shape[0]
    if sel.size == 0:                              # mean -> NaN, max -> ValueError -> all-NaN
        return np.full((P, P), np.nan), sel
    v = raw[..., sel]
    A = v.mean(axis=-1) if st == "mean" else v.max(axis=-1)
    return orient(A), sel


def replica(x, path, variant):
    xp = preprocess(x, path)
    fk = freqs_kept(xp.shape[1])
    raw = raw_psi(cwt_direct(xp, fk), fk)
    return select(raw, fk, variant)[0]


# ------------------------------------------------- documented / intended
def band_freqs(T, variant):
    _, fmin, fmax = VARIANTS[variant]
    fk = freqs_kept(T)
    return fk[(fk >= fmin) & (fk <= fmax)]


def documented(xp, variant, window="all"):
    """Mean/max over time of the per-time PSI restricted to the band's wavelet
    frequencies (supplement 1.5.4 read literally), pyspi orientation."""
    st = VARIANTS[variant][0]
    T = xp.shape[1]
    fb = band_freqs(T, variant)
    if fb.size < 2:
        return np.full((xp.shape[0],) * 2, np.nan)
    raw = raw_psi(cwt_direct(xp, fb), fb)
    if window == "coi":
        h = half_len(fb.min())
        raw = raw[..., h:T - h]
    A = raw.mean(axis=-1) if st == "mean" else raw.max(axis=-1)
    return orient(A)


def intended(xp, variant, window="all"):
    """Nolte eq. 3 on the time-averaged (smoothed) wavelet coherency: the
    canonical PSI with expectations replaced by the time average of the
    Morlet cross-spectrum (_kernels_spectral.cwt_morlet, n_cycles 7)."""
    T = xp.shape[1]
    fb = band_freqs(T, variant)
    if fb.size < 2:
        return np.full((xp.shape[0],) * 2, np.nan)
    W = ks.cwt_morlet(xp, fb, N_CYCLES)            # (P, n_f, T)
    if window == "coi":
        h = half_len(fb.min())
        W = W[..., h:T - h]
    S = np.einsum("ift,jft->fij", W, np.conj(W)) / W.shape[-1]
    C = ks.coherency(S)
    return orient(ks.psi_adjacent(C, fb, -np.inf, np.inf))


def central_window(xp, variant):
    """Control for the edge question: pyspi's statistic over the SAME number of
    time samples as freq_id, but centred in the record (t = T//2 ...), far from
    the zero-padded edges; full 10-frequency PSI as pyspi."""
    T = xp.shape[1]
    fk = freqs_kept(T)
    raw = raw_psi(cwt_direct(xp, fk), fk)
    sel = select(raw, fk, variant)[1]
    c = T // 2
    v = raw[..., c:c + sel.size]
    return orient(v.mean(axis=-1) if VARIANTS[variant][0] == "mean" else v.max(axis=-1))


def truth_sign(ds, d, T, variant):
    """Expected sign in pyspi orientation; also returns a description."""
    P = d["x"].shape[0]
    if ds in ("delayed_noise", "logistic_maps_chaotic"):
        # x_k lags x_0 by d_k (increasing) / x0 drives x1: the higher index lags,
        # raw PSI(i -> j) for i > j < 0
        E = -np.ones((P, P))
        np.fill_diagonal(E, 0)
        if ds == "delayed_noise":
            return E, "analytic: x_k = x0 delayed by d_k = (0,1,3,5), the higher index lags"
        return E, ("coupling direction x0 -> x1 (CONVENTIONS D8); for this nonlinear map the PSI sign is an "
                   "expectation supported by the intended quantity, not derived analytically")
    fb = band_freqs(T, variant)
    S = make_data.var_spectral_matrix([d["A1"], d["A2"]], d["Sigma"], fb)
    Psi = ks.psi_adjacent(ks.coherency(S), fb, -np.inf, np.inf)
    E = np.sign(orient(Psi))
    np.fill_diagonal(E, 0)
    return E, f"population Nolte eq. 3 on make_data.var_spectral_matrix at wavelet freqs {np.round(fb, 4).tolist()}"


def agree(A, E):
    m = (E != 0) & ~np.isnan(A)
    return float(np.mean(np.sign(A[m]) == np.sign(E[m]))) if m.any() else float("nan")


# ------------------------------------------------------------------ handlers
def h_rep(run, variant):
    d = make_data.load(run["dataset"], run["T"])
    V = replica(d["x"], run["path"], variant)
    extra = {}
    if run["dataset"] == "trending":
        base = make_data.load("bivariate_gaussian", run["T"])["x"]
        Vb = replica(base, run["path"], variant)
        extra["max_abs_vs_base_draw_same_path"] = float(np.nanmax(np.abs(V - Vb)))
        other = "pipeline" if run["path"] == "pipeline-detrend" else "pipeline-detrend"
        extra["max_abs_vs_other_path"] = float(np.nanmax(np.abs(V - replica(d["x"], other, variant))))
    save(run["run_id"], value=V, criterion="tolerance", **REP_TOL, reference_kind="independent-impl",
         reference_detail=f"{SELF}:replica (closed-form Morlet n_cycles 7 via _kernels_spectral.morlet_wavelet, "
                          "direct np.convolve full + mne 'same' centring, single-epoch coherency, "
                          "_kernels_spectral.psi_adjacent, pyspi time-index selection, lower->upper mirror)",
         notes="independent replica of what pyspi executes (confirms the trace, not the definition)", **extra)


def h_doc(run, variant):
    d = make_data.load(run["dataset"], run["T"])
    xp = preprocess(d["x"], run["path"])
    V = documented(xp, variant, "all")
    Vc = documented(xp, variant, "coi")
    R = replica(d["x"], run["path"], variant)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-8, atol=1e-10, reference_kind="analytic",
         reference_detail=f"{SELF}:documented (supplement 1.5.4: mean/max PSI across all time points, band-limited "
                          "frequencies = wavelet freqs in [fmin, fmax], adjacent pairs, pyspi orientation)",
         notes="expected to FAIL: pyspi averages time samples t in freq_id instead (KNOWN: wavelet PSI axis selection)",
         doc_coi_max_abs_diff=float(np.nanmax(np.abs(R - Vc))),
         doc_all_max_abs_diff=float(np.nanmax(np.abs(R - V))),
         doc_all_vs_coi_max_abs=float(np.nanmax(np.abs(V - Vc))),
         sign_agree_pyspi_doc_all=agree(R, np.nan_to_num(np.sign(V))),
         n_band_freqs=int(band_freqs(run["T"], variant).size))


def h_dir(run, variant):
    ds, T = run["dataset"], run["T"]
    d = make_data.load(ds, T)
    xp = preprocess(d["x"], run["path"])
    E, desc = truth_sign(ds, d, T, variant)
    R = replica(d["x"], run["path"], variant)
    ia, ic = intended(xp, variant, "all"), intended(xp, variant, "coi")
    da, dc = documented(xp, variant, "all"), documented(xp, variant, "coi")
    st = VARIANTS[variant][0]
    cw = central_window(xp, variant)
    save(run["run_id"], value=R, criterion="direction", expected_sign=E, reference_kind="analytic",
         reference_detail=f"{SELF}:truth_sign ({desc}); pyspi orientation A[i,j]=A[j,i]=PSI(max->min)",
         notes="pass iff sign(pyspi) equals the true lead/lag sign on every off-diagonal entry",
         intended_whole_agree=agree(ia, E), intended_coi_agree=agree(ic, E),
         documented_alltime_agree=agree(da, E), documented_coi_agree=agree(dc, E),
         replica_agree=agree(R, E), central_window_agree=agree(cw, E),
         intended_whole_lower=np.round(ia[np.tril_indices(ia.shape[0], -1)], 4),
         statistic=st)


_MC = {}
_MC_CENTRAL = {}
N_MC = 4000


def mc_null(T, n_rep=N_MC):
    """Monte Carlo null of pyspi's statistic for one pair (raw psi[1, 0]):
    returns {variant: samples} and the all-time-mean samples."""
    if T in _MC:
        return _MC[T]
    rng = np.random.default_rng(20261015 + T)
    fk = freqs_kept(T)
    stats = {v: np.empty(n_rep) for v in ORDER}
    central = {v: np.empty(n_rep) for v in ORDER}
    allmean = np.empty(n_rep)
    sels = {v: np.nonzero((fk >= VARIANTS[v][1]) & (fk <= VARIANTS[v][2]))[0] for v in ORDER}
    for r in range(n_rep):
        x = scipy.stats.zscore(rng.standard_normal((2, T)), axis=1, ddof=1)
        raw = raw_psi(ks.cwt_morlet(x, fk, N_CYCLES), fk)[1, 0]
        allmean[r] = raw.mean()
        for v in ORDER:
            sv = raw[sels[v]]
            stats[v][r] = sv.mean() if VARIANTS[v][0] == "mean" else sv.max()
            cv = raw[T // 2:T // 2 + sels[v].size]
            central[v][r] = cv.mean() if VARIANTS[v][0] == "mean" else cv.max()
    _MC[T] = (stats, allmean)
    _MC_CENTRAL[T] = central
    return _MC[T]


def h_null(run, variant):
    T = run["T"]
    A = load_pyspi(run["run_id"])
    stats, allmean = mc_null(T)
    s = stats[variant]
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    P = A.shape[0]
    npairs = P * (P - 1) // 2
    alpha = 0.05 / npairs                                   # Bonferroni over pairs, two-sided
    lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2])
    lower = A[np.tril_indices(P, -1)]
    inside = (lower >= lo) & (lower <= hi)
    z = np.abs(lower - mu) / sd
    pemp = [float(2 * min(np.mean(s <= v), np.mean(s >= v))) for v in lower]
    V = np.full_like(A, mu)
    np.fill_diagonal(V, np.nan)
    save(run["run_id"], value=V, criterion="qualitative", pass_flag=int(inside.all()), reference_kind="analytic",
         reference_detail=f"{SELF}:mc_null ({N_MC} draws of 2 iid N(0,1) channels, z-scored, own FFT kernel "
                          "_kernels_spectral.cwt_morlet + psi_adjacent + pyspi selection)",
         notes="pass iff every pair lies inside the empirical two-sided Bonferroni (0.05 / pairs) quantile "
               "interval of the MC null (distribution-free; the max statistic is skewed, so a normal z rule is "
               "reported but not used)",
         mc_mean=mu, mc_sd=sd, mc_q_lo=float(lo), mc_q_hi=float(hi),
         mc_sd_alltime_mean=float(allmean.std(ddof=1)), mc_mean_alltime_mean=float(allmean.mean()),
         sd_ratio_stat_over_alltime=sd / float(allmean.std(ddof=1)), max_z=float(z.max()),
         min_p_empirical=float(min(pemp)), alpha_bonferroni=alpha, pyspi_lower=np.round(lower, 4),
         replica_max_abs_diff=float(np.nanmax(np.abs(A - replica(make_data.load(run["dataset"], T)["x"],
                                                                   run["path"], variant)))))


def _raw_replica(x, path):
    xp = preprocess(x, path)
    fk = freqs_kept(xp.shape[1])
    return raw_psi(cwt_direct(xp, fk), fk), fk


def h_tsel(run):
    d = make_data.load(run["dataset"], run["T"])
    raw, fk = _raw_replica(d["x"], run["path"])
    P, T = raw.shape[0], raw.shape[-1]
    lower = np.tril(np.ones((P, P), dtype=bool), -1)
    off = ~np.eye(P, dtype=bool)
    base = [select(raw, fk, v)[0] for v in ORDER]
    V = np.zeros((len(ORDER), T))
    for t in range(T):
        r = raw.copy()
        r[lower, t] += 100.0
        for k, v in enumerate(ORDER):
            V[k, t] = np.nanmax(np.abs(select(r, fk, v)[0][off] - base[k][off]))
    sels = {v: select(raw, fk, v)[1].tolist() for v in ORDER}
    S = load_pyspi(run["run_id"])
    changed = {v: np.nonzero(S[k] > 1e-9)[0].tolist() for k, v in enumerate(ORDER)}
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-8, atol=1e-9, reference_kind="analytic",
         reference_detail=f"{SELF}:h_tsel (replica raw PSI, +100 at one time index, pyspi selection)",
         notes="stored value changes iff the perturbed TIME index is in freq_id (indices of wavelet frequencies)",
         selected_time_indices=json.dumps(sels), pyspi_changed_time_indices=json.dumps(changed),
         n_time_indices=T, pyspi_max_change_outside_selection=float(max(
             (S[k][np.setdiff1d(np.arange(T), sels[v])].max() if T > len(sels[v]) else 0.0)
             for k, v in enumerate(ORDER))))


def h_xsel(run):
    d = make_data.load(run["dataset"], run["T"])
    x = np.asarray(d["x"], dtype=np.float64)
    path = run["path"]
    off = ~np.eye(x.shape[0], dtype=bool)
    base = [replica(x, path, v) for v in ORDER]
    n = 64
    V = np.zeros((len(ORDER), n))
    for t0 in range(n):
        rng = np.random.default_rng(1000 + t0)
        xp = x.copy()
        xp[:, t0:] = 10.0 * rng.standard_normal((x.shape[0], x.shape[1] - t0))
        raw, fk = _raw_replica(xp, path)
        for k, v in enumerate(ORDER):
            V[k, t0] = np.nanmax(np.abs(select(raw, fk, v)[0][off] - base[k][off]))
    S = load_pyspi(run["run_id"])
    fk = freqs_kept(run["T"])
    h = half_len(fk.min())
    bound = {v: int(select(np.zeros((2, 2, run["T"])), fk, v)[1].max() + h) for v in ORDER}
    horizon = {v: int(np.nonzero(S[k] > 1e-9)[0].max()) for k, v in enumerate(ORDER)}
    beyond = max(float(S[k, bound[v] + 1:].max()) for k, v in enumerate(ORDER))
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-8, atol=1e-10, reference_kind="analytic",
         reference_detail=f"{SELF}:h_xsel (replica on the same perturbed inputs x[:, t0:] <- 10 N(0,1), seed 1000+t0)",
         notes="last t0 that changes the stored value = max(selected t) + half-length of the f=0.2 wavelet (27)",
         horizon_bound=json.dumps(bound), pyspi_horizon=json.dumps(horizon),
         pyspi_max_change_beyond_bound=beyond, wavelet_half_length_f02=h)


def h_libinfo(run):
    T = run["T"]
    d = make_data.load(run["dataset"], T)
    P = d["x"].shape[0]
    V = np.r_[[P, P, 1, T], freqs_kept(T)].astype(float)
    save(run["run_id"], value=V, criterion="exact", rtol=1e-12, atol=1e-14, reference_kind="analytic",
         reference_detail=f"{SELF}:h_libinfo (dense PSI shape (N, N, 1 band, T); freqs_computed = linspace(0.2,0.5,10) >= 5/T)",
         notes="the last axis of the cached PSI is time (length T); there is a single band axis of length 1")


def h_rawpsi(run):
    d = make_data.load(run["dataset"], run["T"])
    xp = preprocess(d["x"], run["path"])
    fk = freqs_kept(xp.shape[1])
    W = cwt_direct(xp, fk)
    raw = raw_psi(W, fk)
    P = raw.shape[0]
    V = np.zeros_like(raw)
    il = np.tril_indices(P, -1)
    V[il] = raw[il]
    dphi = np.angle(W[:, None]) - np.angle(W[None, :])              # (P, P, n_f, T)
    sines = np.sin(np.diff(dphi, axis=2)).sum(axis=2)
    C = coherency_single(W)
    k5 = int(np.argmin(np.abs(fk - 0.5)))
    nyq = np.mod(dphi[:, :, k5][il], np.pi)
    nyq = np.minimum(nyq, np.pi - nyq)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-8, atol=1e-10, reference_kind="independent-impl",
         reference_detail=f"{SELF}:h_rawpsi (replica raw PSI(t); mne dense output has 0 upper triangle and diagonal)",
         notes="library step only: per-time adjacent-pair PSI of the single-epoch coherency",
         sum_of_sines_max_abs=float(np.abs(sines[il] - raw[il]).max()),
         own_abs_C_minus_1=float(np.abs(np.abs(C) - 1).max()),
         own_nyquist_dphi_mod_pi_max=float(nyq.max()),
         own_max_abs_Im_W05=float(np.abs(np.imag(ks.morlet_wavelet(0.5, N_CYCLES))).max()))


def h_cohmod(run):
    S = load_pyspi(run["run_id"])
    save(run["run_id"], value=np.ones_like(S), criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="analytic",
         reference_detail=f"{SELF}:h_cohmod (one epoch: |W_i conj W_j| / (|W_i||W_j|) = 1)",
         notes="no coherence weighting is possible: |C(f,t)| = 1 for every pair, frequency and time")


def h_nyq(run):
    S = load_pyspi(run["run_id"])
    save(run["run_id"], value=np.zeros_like(S), criterion="tolerance", rtol=0.0, atol=1e-9, reference_kind="analytic",
         reference_detail=f"{SELF}:h_nyq (f = 0.5 at sfreq 1: Morlet exp(i pi t) real, so C(0.5, t) is real)",
         notes="dphi(0.5, t) in {0, pi}: the last adjacent increment is not a phase slope",
         own_max_abs_Im_W05=float(np.abs(np.imag(ks.morlet_wavelet(0.5, N_CYCLES))).max()))


def h_perm(run, variant):
    d = make_data.load(run["dataset"], run["T"])
    x = np.asarray(d["x"], dtype=np.float64)
    p = np.arange(x.shape[0])[::-1]
    inv = np.argsort(p)
    V = replica(x[p], run["path"], variant)[np.ix_(inv, inv)]
    S = load_pyspi(run["run_id"])
    stem = f"wavelet.psi|{run['dataset']}|{run['T']}|{run['path']}|"
    for tag in ("rep", "null", "dir"):
        if (OUT_PY / f"{stem}{tag}:{variant}.npz").is_file():
            break
    plain = load_pyspi(f"{stem}{tag}:{variant}")
    save(run["run_id"], value=V, criterion="tolerance", **REP_TOL, reference_kind="independent-impl",
         reference_detail=f"{SELF}:replica on X[::-1], mapped back",
         notes="the stored sign depends on channel order: mean variants A_rev = -A exactly",
         max_abs_perm_plus_plain=float(np.nanmax(np.abs(S + plain))),
         max_abs_perm_minus_plain=float(np.nanmax(np.abs(S - plain))),
         plain_10=float(plain[1, 0]), perm_10=float(S[1, 0]), plain_run_tag=tag)


def h_short(run):
    d = make_data.load(run["dataset"], run["T"])
    x = np.asarray(d["x"], dtype=np.float64)[:, :20]
    V = np.stack([replica(x, run["path"], v) for v in ORDER])
    ex = load_exchange(run["run_id"])
    errs = {k: v for k, v in ex.items() if k != "freqs_computed"}
    save(run["run_id"], value=V, criterion="tolerance", **REP_TOL, reference_kind="independent-impl",
         reference_detail=f"{SELF}:replica on x[:, :20] with freqs >= 5/20 (8 of 10 kept), empty selection -> all-NaN",
         notes="T < 25 edge case: low band selects no time index; mean -> NaN (RuntimeWarning), max raises ValueError",
         freqs_kept_ref=np.round(freqs_kept(20), 6), freqs_computed_pyspi=json.dumps(ex.get("freqs_computed")),
         errors=json.dumps(errs))


def h_sl(run, variant):
    import mne_connectivity as mnec
    d = make_data.load(run["dataset"], run["T"])
    xp = preprocess(d["x"], run["path"])
    con = mnec.phase_slope_index(data=xp[None], mode="cwt_morlet", sfreq=1, mt_adaptive=True,
                                 cwt_freqs=FREQS, verbose="WARNING")
    psi = np.real(con.get_data(output="dense"))[:, :, 0, :]
    fc = np.asarray(con.attrs["freqs_computed"][0])
    V = select(psi, fc, variant)[0]
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-12, atol=1e-12, reference_kind="same-lineage",
         reference_detail=f"mne_connectivity=={mnec.__version__} phase_slope_index (comparison venv) + {SELF}:select",
         notes="same-lineage: shows what the library returns; not independent evidence")


def h_base(run, variant):
    base = make_data.load("bivariate_gaussian", run["T"])["x"]
    V = replica(base, run["path"], variant)
    save(run["run_id"], value=V, criterion="tolerance", **REP_TOL, reference_kind="analytic",
         reference_detail=f"{SELF}:replica on the D2 base draw (bivariate_gaussian), same path",
         notes="z-scoring removes the per-channel scale and offset of scaled_offset")


WIN_HORIZON = 37          # max selected t (9) + half-length of the f = 0.2 wavelet (27) + 1


def _rev(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    b = x.shape[1] if b is None else int(b)
    xp = x.copy()
    xp[:, a:b] = x[:, a:b][:, ::-1]
    return xp


def h_winperm(run):
    d = make_data.load(run["dataset"], run["T"])
    x = np.asarray(d["x"], dtype=np.float64)
    a, b = run["probe"]["window"]
    xp = _rev(x, a, b)
    path = run["path"]
    R0 = np.stack([replica(x, path, v) for v in ORDER])
    R1 = np.stack([replica(xp, path, v) for v in ORDER])
    S = load_pyspi(run["run_id"])
    off = ~np.eye(x.shape[0], dtype=bool)
    pys_change = np.array([np.nanmax(np.abs(S[1, k][off] - S[0, k][off])) for k in range(len(ORDER))])
    rep_change = np.array([np.nanmax(np.abs(R1[k][off] - R0[k][off])) for k in range(len(ORDER))])
    x0p, x1p = preprocess(x, path), preprocess(xp, path)
    doc_change = np.array([np.nanmax(np.abs(documented(x1p, v)[off] - documented(x0p, v)[off])) for v in ORDER])
    rep_diff = float(np.nanmax(np.abs(S - np.stack([R0, R1]))))
    extra = dict(window=json.dumps([a, b]), pyspi_change_per_identifier=np.round(pys_change, 12),
                 replica_change_per_identifier=np.round(rep_change, 12),
                 documented_alltime_change_per_identifier=np.round(doc_change, 6),
                 pyspi_vs_replica_max_abs=rep_diff, horizon=WIN_HORIZON, identifier_order=json.dumps(ORDER))
    detail = f"{SELF}:h_winperm (replica on x and on x with x[:, a:b] reversed; pyspi time-index selection)"
    if a >= WIN_HORIZON:
        save(run["run_id"], value=np.stack([R0, R0]), criterion="tolerance", rtol=1e-8, atol=1e-10,
             reference_kind="analytic", reference_detail=detail,
             notes="prediction from the axis selection: input samples t >= 37 cannot reach the stored value, so the "
                   "perturbed output equals the unperturbed one", **extra)
    else:
        ok = bool((pys_change > 1e-6).all() and rep_diff <= 1e-9)
        save(run["run_id"], value=np.stack([R0, R1]), criterion="qualitative", pass_flag=int(ok),
             reference_kind="analytic", reference_detail=detail,
             notes="pass iff every identifier's stored value changes (> 1e-6) and pyspi equals the replica on both "
                   "inputs within 1e-9", **extra)


def h_int(run, variant):
    ds, T = run["dataset"], run["T"]
    d = make_data.load(ds, T)
    xp = preprocess(d["x"], run["path"])
    E, desc = truth_sign(ds, d, T, variant)
    V = intended(xp, variant, "all")
    Vc = intended(xp, variant, "coi")
    S = load_pyspi(run["run_id"])
    il = np.tril_indices(V.shape[0], -1)
    save(run["run_id"], value=V, criterion="tolerance", rtol=1e-8, atol=1e-10, reference_kind="independent-impl",
         reference_detail=f"{SELF}:intended (_kernels_spectral.cwt_morlet n_cycles 7 at wavelet freqs "
                          f"{np.round(band_freqs(T, variant), 4).tolist()}, time-averaged cross-spectrum, "
                          "coherency, _kernels_spectral.psi_adjacent = Nolte 2008 eq. 3; pyspi orientation)",
         notes="expected to FAIL: pyspi averages per-time unit-modulus PSI over t in freq_id instead; sign convention "
               "Psi(i -> j) > 0 = i leads, stored at A[i,j] = A[j,i] for i > j",
         intended_lower=np.round(V[il], 4), intended_coi_lower=np.round(Vc[il], 4), pyspi_lower=np.round(S[il], 4),
         truth_lower=E[il], truth=desc, intended_truth_agree=agree(V, E), intended_coi_truth_agree=agree(Vc, E),
         pyspi_truth_agree=agree(S, E), pyspi_intended_sign_agree=agree(S, np.nan_to_num(np.sign(V))))


NULLMC_FAMILY_N = 36       # 3 T x 6 identifiers x [mean, SD]


def h_nullmc(run):
    T = run["T"]
    n = int(run["probe"]["n"])
    stats, allmean = mc_null(T)
    S = load_pyspi(run["run_id"])
    V = np.empty((len(ORDER), 2))
    SE = np.empty((len(ORDER), 2))
    rng = np.random.default_rng(99)
    for k, v in enumerate(ORDER):
        s = stats[v]
        mu, sd = s.mean(), s.std(ddof=1)
        boot = np.array([rng.choice(s, n, replace=True).std(ddof=1) for _ in range(2000)])
        V[k] = (mu, sd)
        SE[k] = (np.sqrt(sd ** 2 / n + sd ** 2 / s.size), np.sqrt(boot.var(ddof=1) + boot.var(ddof=1) * n / s.size))
    zc = float(scipy.stats.norm.ppf(1 - 0.05 / (2 * NULLMC_FAMILY_N)))
    sd_all = float(allmean.std(ddof=1))
    save(run["run_id"], value=V, criterion="statistical", se=SE * zc / 2.0, reference_kind="analytic",
         reference_detail=f"{SELF}:mc_null ({N_MC} own-kernel draws of two z-scored iid N(0,1) channels; FFT Morlet "
                          "_kernels_spectral.cwt_morlet + psi_adjacent + pyspi selection); SE of the SD by bootstrap",
         notes=f"pyspi {n} iid draws: [mean, SD] of A[1,0] per identifier vs the MC null; se scaled by z_crit/2 = "
               f"{zc / 2:.3f} (two-sided Bonferroni, family-wise 0.05 over {NULLMC_FAMILY_N} numbers)",
         identifier_order=json.dumps(ORDER), pyspi_sd_mean_full=float(S[0, 1]), mc_sd_mean_full=float(V[0, 1]),
         mc_sd_alltime_mean=sd_all, sd_ratio_pyspi_mean_full_over_alltime=float(S[0, 1]) / sd_all,
         pyspi_mean_max_full=float(S[3, 0]), z_crit=zc,
         mc_sd_central_window_mean_full=float(_MC_CENTRAL[T]["mean-full"].std(ddof=1)),
         mc_sd_central_window_mean_low=float(_MC_CENTRAL[T]["mean-low"].std(ddof=1)),
         mc_sd_mean_low=float(V[1, 1]))


TAIL0 = 37
TAIL_LABELS = ["offset_all_+0.5", "scale_ch0_x3", "sd_only_x3_about_tail_mean"]


def _tail_variants(x):
    x = np.asarray(x, dtype=np.float64)
    a = x.copy()
    a[:, TAIL0:] += 0.5
    b = x.copy()
    b[0, TAIL0:] *= 3.0
    c = x.copy()
    m = x[:, TAIL0:].mean(axis=1, keepdims=True)
    c[:, TAIL0:] = m + 3.0 * (x[:, TAIL0:] - m)
    return [x, a, b, c]


def h_tailpert(run):
    d = make_data.load(run["dataset"], run["T"])
    xs = _tail_variants(d["x"])
    path = run["path"]
    R = np.stack([np.stack([replica(v, path, w) for w in ORDER]) for v in xs])
    S = load_pyspi(run["run_id"])
    off = ~np.eye(R.shape[-1], dtype=bool)

    def change(Z, k):
        return float(max(np.nanmax(np.abs(Z[k, i][off] - Z[0, i][off])) for i in range(len(ORDER))))
    extra = {}
    for k, lab in enumerate(TAIL_LABELS, start=1):
        extra[f"pyspi_change_{lab}"] = change(S, k)
        extra[f"replica_change_{lab}"] = change(R, k)
    mean_shift = [float(np.abs(v.mean(axis=1) - xs[0].mean(axis=1)).max()) for v in xs[1:]]
    sd_ratio = [float(np.abs(v.std(axis=1, ddof=1) / xs[0].std(axis=1, ddof=1) - 1).max()) for v in xs[1:]]
    save(run["run_id"], value=R, criterion="tolerance", **REP_TOL, reference_kind="independent-impl",
         reference_detail=f"{SELF}:h_tailpert (replica on x and on three perturbations of x[:, 37:])",
         notes="samples t >= 37 reach the stored value only through preprocessing statistics: the channel mean "
               "(z-score) and, with detrend, the fitted line; the SD is a positive rescaling that PSI ignores; on "
               "the kernel path they cannot reach it",
         channel_mean_shift_max=np.round(mean_shift, 6), channel_sd_rel_change_max=np.round(sd_ratio, 6),
         perturbation_order=json.dumps(TAIL_LABELS), identifier_order=json.dumps(ORDER), **extra)


WELCH_L = 64


def _welch_grid():
    f = np.arange(WELCH_L // 2 + 1) / WELCH_L
    return f


def _open_band(f, fmin, fmax):
    return (f >= fmin) & (f <= fmax) & (f > 0) & (f < 0.5)


def welch_coherency(xp, L=WELCH_L):
    P, T = xp.shape
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)      # periodic Hann
    starts = range(0, T - L + 1, L // 2)
    X = np.stack([np.fft.rfft(xp[:, s:s + L] * win, axis=1) for s in starts])   # (nseg, P, nf)
    S = np.einsum("spf,sqf->fpq", X, np.conj(X)) / X.shape[0]
    return ks.coherency(S), len(starts)


def delay_population_coherency(f):
    d = np.asarray(make_data.D6_DELAYS if hasattr(make_data, "D6_DELAYS") else (0, 1, 3, 5), dtype=float)
    noise = float(getattr(make_data, "D6_NOISE", 0.1))
    var = np.r_[1.0, np.full(d.size - 1, 1.0 + noise ** 2)]
    S = np.exp(-2j * np.pi * f[:, None, None] * (d[None, :, None] - d[None, None, :]))
    S = S.astype(complex)
    for i in range(d.size):
        S[:, i, i] = var[i]
    return ks.coherency(S)


def _psi_band(C, f, fmin, fmax):
    m = _open_band(f, fmin, fmax)
    return ks.psi_adjacent(C[m], f[m], -np.inf, np.inf)


def h_welch(run, variant):
    T = run["T"]
    d = make_data.load(run["dataset"], T)
    xp = preprocess(d["x"], run["path"])
    _, fmin, fmax = VARIANTS[variant]
    f = _welch_grid()
    Cw, nseg = welch_coherency(xp)
    Vw = orient(_psi_band(Cw, f, fmin, fmax))
    Va = orient(_psi_band(delay_population_coherency(f), f, fmin, fmax))
    E = np.sign(Va)
    np.fill_diagonal(E, 0)
    S = load_pyspi(run["run_id"])
    il = np.tril_indices(Vw.shape[0], -1)
    save(run["run_id"], value=Vw, criterion="direction", expected_sign=E, reference_kind="analytic",
         reference_detail=f"{SELF}:h_welch (Nolte 2008 eq. 3 on the analytic pure-delay coherency, grid k/{WELCH_L} in "
                          f"[{fmin}, {fmax}] without 0 and 0.5; value = numpy Welch estimate, periodic Hann, "
                          f"L = {WELCH_L}, 50% overlap, {nseg} segments)",
         notes="pass iff sign(pyspi) equals the canonical (analytic population) PSI sign on every entry; "
               "pyspi orientation A[i,j] = A[j,i] = PSI(max -> min)",
         analytic_lower=np.round(Va[il], 4), welch_lower=np.round(Vw[il], 4), pyspi_lower=np.round(S[il], 4),
         welch_analytic_agree=agree(Vw, E), pyspi_welch_agree=agree(S, np.nan_to_num(np.sign(Vw))),
         pyspi_analytic_agree=agree(S, E), n_band_bins=int(_open_band(f, fmin, fmax).sum()), n_segments=nseg)


POP_BANDS = [("grid64_0_0.5", 0.0, 0.5), ("grid64_0_0.25", 0.0, 0.25), ("grid64_0.25_0.5", 0.25, 0.5),
             ("grid64_0.2_0.5", 0.2, 0.5), ("grid64_0.2_0.25", 0.2, 0.25)]


def h_popband(run, variant):
    ds, T = run["dataset"], run["T"]
    d = make_data.load(ds, T)
    f = _welch_grid()
    S_true = make_data.var_spectral_matrix([d["A1"], d["A2"]], d["Sigma"], f)
    C = ks.coherency(S_true)
    il = np.tril_indices(np.asarray(d["Sigma"]).shape[0], -1)
    vals, signs = {}, []
    for lab, lo, hi in POP_BANDS:
        Psi = orient(_psi_band(C, f, lo, hi))
        vals[f"pop_lower_{lab}"] = np.round(Psi[il], 4)
        signs.append(np.sign(Psi[il]))
    for lab, v in (("wavelet_0.2_0.5", "mean-full"), ("wavelet_0.2_0.25", "mean-low")):
        fb = band_freqs(T, v)
        Sw = make_data.var_spectral_matrix([d["A1"], d["A2"]], d["Sigma"], fb)
        Psi = orient(ks.psi_adjacent(ks.coherency(Sw), fb, -np.inf, np.inf))
        vals[f"pop_lower_{lab}"] = np.round(Psi[il], 4)
        signs.append(np.sign(Psi[il]))
    signs = np.stack(signs)
    same = bool((signs == signs[0]).all() and (signs != 0).all())
    E, desc = truth_sign(ds, d, T, variant)
    S = load_pyspi(run["run_id"])
    V = np.full_like(S, np.nan)
    V[il] = signs[0]
    V = orient(V)
    save(run["run_id"], value=V, criterion="qualitative", pass_flag=int(same and bool((signs[0] == E[il]).all())),
         reference_kind="analytic",
         reference_detail=f"{SELF}:h_popband (Nolte 2008 eq. 3 on make_data.var_spectral_matrix over seven band/grid "
                          "definitions; pyspi orientation)",
         notes="pass iff the population PSI sign of every lower-triangle pair is the same in all seven band/grid "
               "definitions and equals the truth_sign used by the dir runs",
         n_definitions=int(signs.shape[0]), pyspi_lower=np.round(S[il], 4), pyspi_truth_agree=agree(S, E), **vals)


def replica_fs(x, path, variant, fs):
    xp = preprocess(x, path)
    T = xp.shape[1]
    fk = FREQS[FREQS >= 5.0 * fs / T]
    raw = raw_psi(cwt_direct(xp, fk, sfreq=fs), fk)
    return select(raw, fk, variant)[0]


def h_fscache(run):
    d = make_data.load(run["dataset"], run["T"])
    S = load_pyspi(run["run_id"])
    A1, A2s, A2f = S
    R1 = replica(d["x"], run["path"], "mean-full")
    R2 = replica_fs(d["x"], run["path"], "mean-full", 2.0)
    off = ~np.eye(A1.shape[0], dtype=bool)
    same = bool(np.array_equal(A1, A2s, equal_nan=True))
    diff = float(np.nanmax(np.abs(A2f[off] - A1[off])))
    rep2 = float(np.nanmax(np.abs(A2f[off] - R2[off])))
    rep1 = float(np.nanmax(np.abs(A1[off] - R1[off])))
    ok = same and diff > 1e-6 and rep2 <= 1e-9 and rep1 <= 1e-9
    save(run["run_id"], value=np.stack([R1, R1, R2]), criterion="qualitative", pass_flag=int(ok),
         reference_kind="analytic",
         reference_detail=f"{SELF}:h_fscache (code reading wavelet.py:201-206: data.mne_psi is not keyed by fs; "
                          "replica_fs = wavelets at f / fs cycles per sample, default fmin 5 fs / T)",
         notes="pass iff the fs = 2 SPI on the shared Data returns the fs = 1 value bitwise, a fresh fs = 2 Data "
               "differs by > 1e-6, and both fresh values equal the replica within 1e-9",
         shared_fs2_equals_fs1=int(same), fresh_fs2_minus_fs1=diff, fresh_fs2_vs_replica_fs2=rep2,
         fs1_vs_replica=rep1)


def handle(run):
    tag = run["run_id"].split("|")[-1]
    name, _, variant = tag.partition(":")
    if name == "perm":
        variant = {v: k for k, v in {
            "mean-full": "psi_wavelet_mean_fs-1_fmin-0_fmax-0-5_mean",
            "max-full": "psi_wavelet_max_fs-1_fmin-0_fmax-0-5_max"}.items()}[run["probe"]["identifier"]]
        return h_perm(run, variant)
    fn = {"rep": h_rep, "doc": h_doc, "dir": h_dir, "null": h_null, "sl": h_sl, "base": h_base}.get(name)
    if fn is not None:
        return fn(run, variant)
    if name == "int":
        return h_int(run, variant)
    if name in ("welch", "popband"):
        return {"welch": h_welch, "popband": h_popband}[name](run, variant)
    return {"tsel": h_tsel, "xsel": h_xsel, "libinfo": h_libinfo, "rawpsi": h_rawpsi, "cohmod": h_cohmod,
            "nyq": h_nyq, "short": h_short, "winperm": h_winperm, "nullmc": h_nullmc,
            "tailpert": h_tailpert, "fscache": h_fscache}[name.split("-")[0]](run)


def main():
    m = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_err = 0
    for run in m["runs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                handle(run)
            n_ok += 1
        except Exception as exc:
            n_err += 1
            import traceback
            print(f"REF ERROR {run['run_id']}: {exc!r}\n{traceback.format_exc()[-1500:]}", file=sys.stderr)
    print(f"refs/{FAMILY}.py: {n_ok} references written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
