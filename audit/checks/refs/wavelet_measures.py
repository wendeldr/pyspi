#!/usr/bin/env python3
"""References for the wavelet_measures family (COMPARISON venv).

Units: the nine extra wavelet classes of ``pyspi/statistics/wavelet.py``
(cohmag, phase, icoh, plv, ppc, pli, dswpli = identifier dspli, wspli, dwspli)
and the infrastructure unit ``wavelet.morlet_cwt``.  No reference imports mne
or mne_connectivity.  numpy / scipy and ``_kernels_spectral`` (closed-form
Morlet ``morlet_wavelet`` and FFT CWT ``cwt_morlet``, neither imports mne) are
the building blocks.  The ``port`` reference replays mne's arithmetic
(scipy.fft FFT convolution, accumulators) and is labelled same-lineage.

What pyspi computes (trace, notes/wavelet_measures.md): one epoch, Morlet
wavelets with n_cycles = f/7 on f = linspace(0.2, 0.5, 125) -> every wavelet
is one sample W = sqrt(2) (+- 1 ulp), X(f, t) = sqrt(2) x(t) + FFT round-off,
csd = X_hi conj(X_lo) for lower-triangle connections, single-epoch estimator
per (f, t), np.nanmean / np.nanmax over (f in band, t), lower -> upper mirror,
NaN diagonal.

Tags (last ``|`` field of the run_id, ``tag[:variant]``)
  six       constant units: the six identifiers equal the analytic constant
            (cohmag, plv 1; dwspli 0; ppc, dspli -inf: exact; icoh 0 within
            atol 1e-12).
  ident     phase / pli / wspli: identity evaluated on the library CWT exported
            by the PROJECT-venv probe (same-lineage self-consistency: the
            reference only copies the probe's identity slice) (phase: mean = pi (n_+pi - n_-pi)/N, max = pi;
            pli: mean = 1 - frac(Im csd == 0), max = 1; wspli == pli).
  rep       plain identifier.  constant units: analytic constant.  phase / pli /
            wspli: ``port`` (same-lineage replay of mne's FFT arithmetic with
            scipy.fft on pyspi's stored input), exact; extra keys give the same
            replay with numpy.fft (a different FFT implementation).
  scratch   from-scratch pipeline with DIRECT convolution (np.convolve, no FFT):
            closed-form Morlet, single-epoch estimator formulas, pyspi's band
            rule, statistic and mirror.  exact/tolerance; ppc qualitative (both
            non-finite).
  canon     canonical estimator with a proper Morlet CWT (n_cycles = 7,
            _kernels_spectral.cwt_morlet): time-averaged single-trial cohmag,
            phase, icoh, plv, pli, wspli (Lachaux 2002 / Stam 2007 / Vinck 2011
            with the average over time); epoched (K = 8) ppc, dspli, dwspli
            (Vinck 2010, 2011).  tolerance atol 0.05: an ILLUSTRATION expected to
            disagree.
  canonep   pli / wspli: epoched (K = 8) PLI / wPLI across epochs per (f, t), n_cycles = 7;
            illustration at atol 0.05, expected to disagree.
  canonwin  cohmag / plv: sliding-window (n_co = 6 cycles, Lachaux 2002) single-trial
            coherence / PLV, n_cycles = 7; illustration at atol 0.05, expected to disagree.
  perm      [A(x), A(x reversed) mapped back]: constants (analytic); icoh [A, -A];
            pli, wspli [A, A]; phase [A, -A + 2 pi n(Im csd == 0, Re csd < 0)/N]
            (pyspi vs pyspi: same-lineage).
  roundoff  [A(x), A(z-scored again)] expected [A, A] within 1e-12 (fails if the
            value depends on the FFT round-off pattern; pyspi vs pyspi:
            same-lineage).
  pts       ppc / dspli per-point census: qualitative.
  ratio     ppc: max |Im csd| / |Re csd| < 1.05e-8 (the +inf threshold): qualitative.
  pinf      ppc: sign rule x/0 on the residue |acc|^2 - 1 of the synthetic csd.
  dsplisyn  dspli: [NaN, -inf] analytic.
  wavelets, cwtx, cwtdirect, mtadapt, freqs, short, orient7, orient1,
  fscache   infrastructure (see the functions).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.fft  # noqa: E402

import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

FAMILY = "wavelet_measures"
CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCHANGE = CHECKS / "out" / "exchange" / FAMILY
SELF = "refs/wavelet_measures.py"
FREQS = np.linspace(0.2, 0.5, 125)
IDS = [("mean", 0, 0.5), ("mean", 0, 0.25), ("mean", 0.25, 0.5),
       ("max", 0, 0.5), ("max", 0, 0.25), ("max", 0.25, 0.5)]
VARIANTS = {"mean-full": 0, "mean-low": 1, "mean-high": 2, "max-full": 3, "max-low": 4, "max-high": 5}
MEAS = {"wavelet.cohmag": "coh", "wavelet.phase": "phase", "wavelet.icoh": "imcoh", "wavelet.plv": "plv",
        "wavelet.ppc": "ppc", "wavelet.pli": "pli", "wavelet.dswpli": "pli2_unbiased",
        "wavelet.wspli": "wpli", "wavelet.dwspli": "wpli2_debiased"}
CONST = {"wavelet.cohmag": 1.0, "wavelet.plv": 1.0, "wavelet.dwspli": 0.0, "wavelet.ppc": -np.inf,
         "wavelet.dswpli": -np.inf, "wavelet.icoh": 0.0}
SQ2 = np.sqrt(2.0)


# ------------------------------------------------------------------ helpers
def load_py(rid):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=float), np.asarray(z["input"], dtype=float)


def prep(x, path):
    """Preprocessing of a PROBE run's stored input (pyspi_outputs stores the raw x for probe runs):
    kernel = raw; pipeline = zscore(ddof=1); pipeline-detrend = linear detrend then zscore
    (pyspi/data.py:184-195)."""
    import scipy.signal
    import scipy.stats
    x = np.asarray(x, dtype=float)
    if path == "kernel":
        return x
    if path == "pipeline-detrend":
        x = scipy.signal.detrend(x, axis=1)
    return scipy.stats.zscore(x, axis=1, ddof=1)


def write(rid, value, criterion, kind, detail, notes="", **extra):
    OUT_REF.mkdir(parents=True, exist_ok=True)
    payload = dict(value=np.asarray(value, dtype=float), criterion=criterion, reference_kind=kind,
                   reference_detail=detail, notes=notes)
    for k, v in extra.items():
        payload[k] = np.asarray(v)
    np.savez(OUT_REF / f"{rid}.npz", **payload)


def const_matrix(P, c):
    A = np.full((P, P), c, dtype=float)
    np.fill_diagonal(A, np.nan)
    return A


def mirror(L):
    A = np.array(L, dtype=float, copy=True)
    ui = np.triu_indices(A.shape[0], 1)
    A[ui] = A.T[ui]
    np.fill_diagonal(A, np.nan)
    return A


def sym_maxabs(A):
    """max |A - A.T| over off-diagonal entries; equal entries (incl. equal infinities) count 0."""
    A = np.asarray(A, dtype=float)
    B = A.T
    same = (A == B) | (np.isnan(A) & np.isnan(B))
    with np.errstate(invalid="ignore"):
        d = np.abs(A - B)
    d[same] = 0.0
    np.fill_diagonal(d, 0.0)
    return float(np.nanmax(d)) if d.size else 0.0


def band(k):
    _, fmin, fmax = IDS[k]
    return (FREQS >= fmin) & (FREQS <= fmax)


def statfn(k):
    return np.nanmean if IDS[k][0] == "mean" else np.nanmax


def variant_of(rid):
    tag = rid.split("|")[-1]
    return VARIANTS[tag.split(":")[1]] if ":" in tag and tag.split(":")[1] in VARIANTS else 0


def agg(per_point, k):
    """pyspi aggregation of a per-pair (n_f, n_t) array: band mask then nanmean/nanmax over (f, t)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sel = per_point[band(k)]
        if sel.size == 0:
            return np.nan
        return float(statfn(k)(sel))


# ------------------------------------------------------------------ CWTs
def direct_cwt(x):
    """From-scratch CWT: closed-form Morlet (ks.morlet_wavelet, n_cycles f/7) and DIRECT
    convolution np.convolve(mode='same') of the real and imaginary parts (no FFT)."""
    x = np.atleast_2d(x)
    out = np.empty((x.shape[0], FREQS.size, x.shape[1]), dtype=complex)
    for k, f in enumerate(FREQS):
        W = ks.morlet_wavelet(f, f / 7.0)
        for p in range(x.shape[0]):
            out[p, k] = np.convolve(x[p], W.real, mode="same") + 1j * np.convolve(x[p], W.imag, mode="same")
    return out


def port_cwt(x, fftmod):
    """Replay of mne _cwt_gen (tfr.py:322-401) with ``fftmod`` (scipy.fft = what mne uses,
    or numpy.fft): nfft = 5-smooth >= T + max|W| - 1 (tfr.py:303-319), ifft(fft(x) fft(W)),
    centred."""
    x = np.atleast_2d(x)
    T = x.shape[1]
    Ws = [ks.morlet_wavelet(f, f / 7.0) for f in FREQS]
    nfft = ks.next_fast_len_235(T + max(w.size for w in Ws) - 1)
    fW = np.empty((len(Ws), nfft), dtype=np.complex128)
    for i, W in enumerate(Ws):
        fW[i] = fftmod.fft(W, nfft)
    out = np.empty((x.shape[0], len(Ws), T), dtype=complex)
    for p in range(x.shape[0]):
        fx = fftmod.fft(x[p], nfft)
        for i, W in enumerate(Ws):
            ret = fftmod.ifft(fx * fW[i])[: T + W.size - 1]
            s = (ret.size - T) // 2
            out[p, i] = ret[s:s + T]
    return out


# ------------------------------------------------------------------ single-epoch estimators
def single_epoch(meas, Xi, Xj, accumulate=True):
    """Per-(f, t) single-epoch value of ``meas`` for csd = Xi conj(Xj).  ``accumulate``
    replays mne-connectivity's accumulators (zeros + csd, epochs_bivariate.py:55-345);
    otherwise the textbook N = 1 formula."""
    csd = Xi * Xj.conjugate()
    psd_i, psd_j = (Xi * Xi.conj()).real, (Xj * Xj.conj()).real
    with np.errstate(divide="ignore", invalid="ignore"):
        if meas in ("coh", "phase", "imcoh"):
            acc = csd
            if accumulate:
                acc = np.zeros(csd.shape, dtype=np.complex128)
                acc += csd
            c = (acc / 1) / np.sqrt(psd_i * psd_j)
            if not accumulate:
                c = c + 0j   # principal value: arg of a negative real is +pi (signed zeros normalised)
            return {"coh": np.abs(c), "phase": np.angle(c), "imcoh": c.imag}[meas]
        if meas == "plv":
            return np.abs(csd / np.abs(csd))
        if meas == "ppc":
            d = np.abs(csd)
            u = np.where(d == 0, 0, csd / np.where(d == 0, 1, d))
            return np.real((u * np.conj(u) - 1) / (1 * (1 - 1.0)))
        im = csd.imag
        if meas == "pli":
            return np.abs(np.sign(im))
        if meas == "pli2_unbiased":
            return (1 * np.sign(im) ** 2 - 1) / (1 - 1)
        if meas == "wpli":
            den = np.abs(im)
            return np.where(den == 0, 0.0, np.abs(im) / np.where(den == 0, 1, den))
        if meas == "wpli2_debiased":
            den = np.abs(im) ** 2 - im ** 2
            return np.where(den == 0, 0.0, (im ** 2 - im ** 2) / np.where(den == 0, 1, den))
    raise ValueError(meas)


def pipeline_value(meas, X, k, dense=False):
    """pyspi aggregation (wavelet.py:80-89) for one identifier from per-channel CWT X (P, nf, T).
    ``dense`` builds mne's dense (P, P, nf, T) tensor (upper triangle 0) and applies the
    statistic to it exactly as pyspi does."""
    P = X.shape[0]
    if dense:
        conn = np.zeros((P, P, FREQS.size, X.shape[2]))
        for i in range(P):
            for j in range(i):
                conn[i, j] = single_epoch(meas, X[i], X[j])
        fid = np.where(band(k))[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            A = statfn(k)(conn[..., fid, :], axis=(2, 3))
        return mirror(A)
    L = np.zeros((P, P))
    for i in range(P):
        for j in range(i):
            L[i, j] = agg(single_epoch(meas, X[i], X[j], accumulate=False), k)
    return mirror(L)


# ------------------------------------------------------------------ canonical (n_cycles = 7)
def canonical(meas, z, k, K=8):
    """Canonical estimator on a proper Morlet CWT (n_cycles = 7).  Time-averaged single-trial
    forms for coh / phase / imcoh / plv / pli / wpli (average over t BEFORE normalising or
    taking the magnitude); epoched (K non-overlapping epochs) forms for ppc, pli2_unbiased,
    wpli2_debiased (Vinck 2010 eq. for PPC; Vinck 2011 debiased estimators), then the band
    statistic over (f, t).  Returns the mirrored (P, P) matrix and the per-frequency values of
    pair (1, 0)."""
    P, T = z.shape
    per_f10 = None
    L = np.zeros((P, P))
    if meas in ("ppc", "pli2_unbiased", "wpli2_debiased"):
        n = T // K
        Xe = np.stack([ks.cwt_morlet(z[:, e * n:(e + 1) * n], FREQS, 7.0) for e in range(K)])  # (K,P,nf,n)
        for i in range(P):
            for j in range(i):
                S = Xe[:, i] * Xe[:, j].conj()
                with np.errstate(divide="ignore", invalid="ignore"):
                    if meas == "ppc":
                        u = S / np.abs(S)
                        v = (np.abs(u.sum(0)) ** 2 - K) / (K * (K - 1))
                    elif meas == "pli2_unbiased":
                        v = (K * np.sign(S.imag).mean(0) ** 2 - 1) / (K - 1)
                    else:
                        im = S.imag
                        v = (im.sum(0) ** 2 - (im ** 2).sum(0)) / (np.abs(im).sum(0) ** 2 - (im ** 2).sum(0))
                L[i, j] = agg(v, k)
                if (i, j) == (1, 0):
                    per_f10 = v.mean(-1)
        return mirror(L), per_f10
    X = ks.cwt_morlet(z, FREQS, 7.0)  # (P, nf, T)
    for i in range(P):
        for j in range(i):
            S = X[i] * X[j].conj()
            Sii, Sjj = (np.abs(X[i]) ** 2).mean(-1), (np.abs(X[j]) ** 2).mean(-1)
            Sm = S.mean(-1)
            if meas == "coh":
                v = np.abs(Sm) / np.sqrt(Sii * Sjj)
            elif meas == "phase":
                v = np.angle(Sm)
            elif meas == "imcoh":
                v = Sm.imag / np.sqrt(Sii * Sjj)
            elif meas == "plv":
                v = np.abs((S / np.abs(S)).mean(-1))
            elif meas == "pli":
                v = np.abs(np.sign(S.imag).mean(-1))
            elif meas == "wpli":
                v = np.abs(S.imag.mean(-1)) / np.abs(S.imag).mean(-1)
            else:
                raise ValueError(meas)
            L[i, j] = agg(v[:, None], k)
            if (i, j) == (1, 0):
                per_f10 = v
    return mirror(L), per_f10



def canonical_epoched_lag(meas, z, k, K=8):
    """Epoched (K non-overlapping epochs) PLI / wPLI on a Morlet CWT with n_cycles = 7: per (f, t)
    PLI = |mean_k sign(Im S_k)| (Stam 2007 over trials), wPLI = |sum_k Im S_k| / sum_k |Im S_k|
    (Vinck 2011 eq. for wPLI over trials), then pyspi's band statistic over (f, t)."""
    P, T = z.shape
    n = T // K
    Xe = np.stack([ks.cwt_morlet(z[:, e * n:(e + 1) * n], FREQS, 7.0) for e in range(K)])  # (K,P,nf,n)
    L = np.zeros((P, P))
    per_f10 = None
    for i in range(P):
        for j in range(i):
            im = (Xe[:, i] * Xe[:, j].conj()).imag
            with np.errstate(divide="ignore", invalid="ignore"):
                if meas == "pli":
                    v = np.abs(np.sign(im).mean(0))
                elif meas == "wpli":
                    v = np.abs(im.sum(0)) / np.abs(im).sum(0)
                else:
                    raise ValueError(meas)
            L[i, j] = agg(v, k)
            if (i, j) == (1, 0):
                per_f10 = np.nanmean(v, -1)
    return mirror(L), per_f10


def canonical_window(meas, z, k, n_co=6.0):
    """Sliding-window single-trial coherence / PLV (Lachaux et al. 2002): Morlet CWT with n_cycles = 7,
    then at each frequency a centred boxcar of w = round(n_co / f) samples over the cross- and
    auto-spectra (coh: |<S_xy>_w| / sqrt(<S_xx>_w <S_yy>_w)) or over the unit phasors (plv:
    |<S_xy/|S_xy|>_w|); positions where the window does not fit are NaN; pyspi's band statistic
    over (f, t)."""
    P, T = z.shape
    X = ks.cwt_morlet(z, FREQS, 7.0)
    L = np.zeros((P, P))
    per_f10 = None

    def smooth(a, w):
        out = np.full(a.shape, np.nan, dtype=a.dtype if np.iscomplexobj(a) else float)
        c = np.convolve(a, np.ones(w) / w, mode="valid")
        s = (w - 1) // 2
        out[s:s + c.size] = c
        return out

    for i in range(P):
        for j in range(i):
            v = np.full((FREQS.size, T), np.nan)
            for q, f in enumerate(FREQS):
                w = int(round(n_co / f))
                S = X[i, q] * X[j, q].conj()
                if meas == "coh":
                    sxy = smooth(S, w)
                    sxx = smooth(np.abs(X[i, q]) ** 2, w)
                    syy = smooth(np.abs(X[j, q]) ** 2, w)
                    v[q] = np.abs(sxy) / np.sqrt(sxx * syy)
                elif meas == "plv":
                    v[q] = np.abs(smooth(S / np.abs(S), w))
                else:
                    raise ValueError(meas)
            L[i, j] = agg(v, k)
            if (i, j) == (1, 0):
                per_f10 = np.nanmean(v, -1)
    return mirror(L), per_f10


# ------------------------------------------------------------------ tag handlers
def h_six(run, rid):
    S, _ = load_py(rid)
    P = S.shape[-1]
    c = CONST[run["unit_id"]]
    V = np.stack([const_matrix(P, c)] * 6)
    sym = max(sym_maxabs(A) for A in S)
    off = S[:, ~np.eye(P, dtype=bool)]
    extra = dict(sym_maxabs=sym, n_nonfinite=int((~np.isfinite(off)).sum()), n_offdiag=int(off.size),
                 n_nan=int(np.isnan(off).sum()), n_neginf=int((off == -np.inf).sum()),
                 n_posinf=int((off == np.inf).sum()))
    if run["unit_id"] == "wavelet.icoh":
        extra["max_abs_value"] = float(np.nanmax(np.abs(off)))
        write(rid, V, "tolerance", "analytic",
              "analytic: real one-sample wavelet -> Im csd is FFT round-off; icoh = 0 within atol 1e-12",
              rtol=0.0, atol=1e-12, **extra)
        return
    write(rid, V, "exact", "analytic",
          f"analytic: one epoch (and one-sample wavelet) -> {MEAS[run['unit_id']]} = {c} at every (f, t); "
          "six identifiers identical", **extra)


def h_ident(run, rid):
    S, _ = load_py(rid)                    # (3, 6, P, P)
    unit = run["unit_id"]
    V = np.stack([S[1], S[1], S[2]])
    P = S.shape[-1]
    off = ~np.eye(P, dtype=bool)
    extra = dict(sym_maxabs=max(sym_maxabs(A) for A in S[0]))
    means = S[0][:3][:, off]
    extra["band_spread_mean_ids"] = float(np.nanmax(means.max(0) - means.min(0)))
    if unit == "wavelet.phase":
        extra.update(max_angle_dev=float(np.nanmax(S[2][0][off])),
                     frac_angles_not_exact_in_set_mean=float(np.nanmean(S[2][1][off])),
                     frac_imzero_re_neg_mean=float(np.nanmean(S[2][2][off])),
                     frac_raw_csd_negzero_mean=float(np.nanmean(S[2][3][off])),
                     frac_coherency_negzero_max=float(np.nanmax(S[2][4][off])),
                     max_ids=np.unique(np.round(S[0][3:][:, off], 15)).tolist()[:8].__repr__())
        write(rid, V, "tolerance", "same-lineage",
              "self-consistency: identity slice computed by the PROJECT-venv probe ident from mne's own CWT and csd "
              "(this function only copies it): angle of (0 + csd)/sqrt(psd psd) lies "
              "within round-off of {0, +pi, -pi}; mean = pi (n_+pi - n_-pi)/N, max = pi if any angle near +pi",
              notes="slices: [pyspi six, identity six, aux (copied)]", rtol=0.0, atol=1e-12, **extra)
    elif unit == "wavelet.pli":
        extra.update(frac_imzero_full_mean=float(np.nanmean(S[2][5][off])),
                     frac_imzero_full_max=float(np.nanmax(S[2][5][off])),
                     pyspi_mean_full_min=float(np.nanmin(S[0][0][off])),
                     pyspi_mean_full_max=float(np.nanmax(S[0][0][off])))
        write(rid, V, "exact", "same-lineage",
              "self-consistency: identity slice computed by the PROJECT-venv probe ident from mne's own CWT and csd "
              "(this function only copies it): pli mean = 1 - frac(Im csd == 0), "
              "max = 1 if any Im csd != 0", notes="slices: [pyspi six, identity six, aux (copied)]", **extra)
    else:  # wspli
        extra["wspli_equals_pli_bitwise"] = int(np.array_equal(S[0], S[1], equal_nan=True))
        write(rid, V, "exact", "same-lineage",
              "pyspi vs pyspi: one epoch -> wpli = |Im|/|Im| (0 if Im == 0) = |sign Im| = pli; wspli six vs pli six "
              "on the same Data", notes="slices: [pyspi wspli six, pyspi pli six, pli six]", **extra)


def h_rep(run, rid):
    S, xin = load_py(rid)
    unit = run["unit_id"]
    P = S.shape[0]
    k = variant_of(rid)
    if unit in CONST:
        tol = dict(rtol=0.0, atol=1e-12) if unit == "wavelet.icoh" else {}
        write(rid, const_matrix(P, CONST[unit]), "tolerance" if tol else "exact", "analytic",
              f"analytic constant {CONST[unit]} (one epoch, one-sample wavelet)", sym_maxabs=sym_maxabs(S),
              **tol)
        return
    meas = MEAS[unit]
    Xs = port_cwt(xin, scipy.fft)
    V = pipeline_value(meas, Xs, k, dense=True)
    Xn = port_cwt(xin, np.fft)
    Vn = pipeline_value(meas, Xn, k, dense=True)
    m = ~np.eye(P, dtype=bool)
    zs, zn = Xs.imag == 0, Xn.imag == 0
    write(rid, V, "exact", "same-lineage",
          f"{SELF}:port_cwt+pipeline_value (replay of mne tfr._cwt_gen with scipy.fft=={scipy.__version__}, "
          "mne-connectivity accumulators, pyspi aggregation) on pyspi's stored input",
          sym_maxabs=sym_maxabs(S), numpy_fft_maxabs_vs_pyspi=float(np.nanmax(np.abs(Vn[m] - S[m]))),
          scipy_vs_numpy_imzero_pattern_mismatch=float(np.mean(zs != zn)),
          numpy_fft_value_10=float(Vn[1, 0]))


def h_scratch(run, rid):
    S, xin = load_py(rid)
    unit = run["unit_id"]
    meas = MEAS[unit]
    P = S.shape[0]
    k = variant_of(rid)
    X = direct_cwt(xin)
    V = pipeline_value(meas, X, k)
    m = ~np.eye(P, dtype=bool)
    detail = (f"{SELF}:direct_cwt+pipeline_value (closed-form Morlet n_cycles f/7, np.convolve 'same', "
              "single-epoch textbook formulas, pyspi band/statistic/mirror; no FFT, no mne)")
    extra = dict(sym_maxabs=sym_maxabs(S), scratch_value_10=float(V[1, 0]), pyspi_value_10=float(S[1, 0]),
                 scratch_im_exact_zero=int(np.all(X.imag == 0)))
    if unit == "wavelet.ppc":
        pp = single_epoch("ppc", X[1], X[0], accumulate=False)
        extra.update(scratch_pts_neginf_10=int(np.sum(pp == -np.inf)), scratch_pts_nan_10=int(np.isnan(pp).sum()),
                     scratch_pts_posinf_10=int(np.sum(pp == np.inf)))
    if unit == "wavelet.ppc":
        ok = bool((~np.isfinite(S[m])).all() and (~np.isfinite(V[m])).all())
        write(rid, V, "qualitative", "independent-impl", detail, pass_flag=int(ok),
              notes="pass iff both pipelines are non-finite everywhere off the diagonal; the from-scratch "
                    "CWT has Im exactly 0, yet numpy's complex/real division csd/|csd| returns 1 - 2^-53 at "
                    "some points (residue < 0 -> -inf), elsewhere 0/0 = NaN", **extra)
    elif unit == "wavelet.icoh":
        write(rid, V, "tolerance", "independent-impl", detail, rtol=0.0, atol=1e-12, **extra)
    else:
        write(rid, V, "exact", "independent-impl", detail, **extra)


def h_canon(run, rid, form_tag="canon"):
    S, xin = load_py(rid)
    unit = run["unit_id"]
    k = variant_of(rid)
    if form_tag == "canonep":
        V, pf = canonical_epoched_lag(MEAS[unit], xin, k)
    elif form_tag == "canonwin":
        V, pf = canonical_window(MEAS[unit], xin, k)
    else:
        V, pf = canonical(MEAS[unit], xin, k)
    extra = dict(sym_maxabs=sym_maxabs(S), canon_value_10=float(V[1, 0]), pyspi_value_10=float(S[1, 0]),
                 canon_f025_10=float(pf[np.argmin(np.abs(FREQS - 0.25))]),
                 canon_min_offdiag=float(np.nanmin(V)), canon_max_offdiag=float(np.nanmax(V)))
    if unit == "wavelet.cohmag" and run["dataset"].startswith("narrowband"):
        T = xin.shape[1]
        docgrid = np.linspace(4.0 / T, 0.5, 125)
        Xd = ks.cwt_morlet(xin[:2], docgrid, 7.0)
        S10 = (Xd[1] * Xd[0].conj()).mean(-1)
        c = np.abs(S10) / np.sqrt((np.abs(Xd[1]) ** 2).mean(-1) * (np.abs(Xd[0]) ** 2).mean(-1))
        extra.update(doc_grid_coh10_max=float(c.max()), doc_grid_argmax_f=float(docgrid[np.argmax(c)]),
                     pyspi_grid_coh10_max=float(pf.max()))
    if form_tag == "canonep":
        form = "epoched K=8 (T/8 samples per epoch) PLI/wPLI across epochs per (f, t)"
    elif form_tag == "canonwin":
        form = "sliding window of n_co = 6 cycles (Lachaux 2002), NaN where the window does not fit"
    else:
        form = ("epoched K=8 (T/8 samples per epoch)" if MEAS[unit] in ("ppc", "pli2_unbiased", "wpli2_debiased")
                else "time-averaged single trial")
    write(rid, V, "tolerance", "independent-impl",
          f"{SELF}:canonical ({form}; Morlet n_cycles=7 via _kernels_spectral.cwt_morlet; pyspi band grid and "
          "statistic; hi,lo orientation mirrored)",
          notes="illustration of what the measure would be with a real time-frequency decomposition; "
                "expected to disagree", rtol=0.0, atol=0.05, **extra)


def h_perm(run, rid):
    S, _ = load_py(rid)
    unit = run["unit_id"]
    A, Ar = S[0], S[1]
    P = A.shape[0]
    m = ~np.eye(P, dtype=bool)
    with np.errstate(invalid="ignore"):
        extra = dict(sym_maxabs=sym_maxabs(A), sym_maxabs_rev=sym_maxabs(Ar),
                     max_abs_rev_minus_fwd=float(np.nanmax(np.where(A[m] == Ar[m], 0, np.abs(Ar[m] - A[m])))),
                     max_abs_rev_plus_fwd=float(np.nanmax(np.where(A[m] == -Ar[m], 0, np.abs(Ar[m] + A[m])))))
    if unit in CONST and unit != "wavelet.icoh":
        V = np.stack([const_matrix(P, CONST[unit])] * 2)
        write(rid, V, "exact", "analytic", "analytic constant in both channel orders", **extra)
    elif unit == "wavelet.icoh":
        write(rid, np.stack([A, -A]), "exact", "same-lineage",
              "pyspi vs pyspi (reversed channel order): reversing the channels conjugates csd (conn[hi,lo] = X_hi conj X_lo), so the "
              "round-off icoh changes sign exactly", **extra)
    elif unit in ("wavelet.pli", "wavelet.wspli"):
        write(rid, np.stack([A, A]), "exact", "same-lineage",
              "pyspi vs pyspi (reversed channel order): |sign Im| and |Im|/|Im| are invariant under conjugation", **extra)
    else:  # phase
        C = S[2]
        write(rid, np.stack([A, -A + C, C]), "tolerance", "same-lineage",
              "pyspi vs pyspi (reversed channel order), offset C from the PROJECT-venv probe on mne's csd: "
              "conjugation negates every angle except exact-zero Im points with Re < 0, which the "
              "+0.0 accumulator maps to +pi in both orders: A_rev = -A + 2 pi n(Im=0, Re<0)/N",
              rtol=0.0, atol=1e-12, offset_10=float(C[1, 0]), **extra)


def h_roundoff(run, rid):
    S, _ = load_py(rid)
    A = S[0]
    P = A.shape[0]
    m = ~np.eye(P, dtype=bool)
    write(rid, np.stack([A, A, S[2]]), "tolerance", "same-lineage",
          "pyspi vs pyspi on re-z-scored input: a function of the data changes by O(1e-15) when the input changes by ~1e-16",
          notes="slices: [A(z), A(zscore(z)), max|zscore(z) - z| (copied)]", rtol=0.0, atol=1e-12,
          max_abs_change=float(np.nanmax(np.abs(S[1][m] - A[m]))), max_input_change=float(S[2][0, 0]),
          value_10=float(A[1, 0]), value_10_rezscored=float(S[1][1, 0]))


def h_pts(run, rid):
    S, _ = load_py(rid)
    P = S.shape[-1]
    m = np.tril(np.ones((P, P), dtype=bool), -1)
    ninf, nnan, npinf, nfin = (S[q][m] for q in range(4))
    if run["unit_id"] == "wavelet.ppc":
        ok = bool((npinf == 0).all() and (nfin == 0).all() and (ninf > 0).all())
        notes = "pass iff every pair: no +inf, no finite point, at least one -inf"
    else:
        ok = bool((npinf == 0).all() and (nfin == 0).all() and (ninf > 0).all())
        notes = "pass iff every pair: only -inf (Im == 0) and NaN (Im != 0) points, at least one -inf"
    write(rid, S, "qualitative", "analytic", "analytic: N = 1 -> x/0 per point", notes=notes, pass_flag=int(ok),
          neginf_10=int(S[0][1, 0]), nan_10=int(S[1][1, 0]), posinf_total=int(npinf.sum()),
          finite_total=int(nfin.sum()), neginf_frac_mean=float((ninf / (ninf + nnan + npinf + nfin)).mean()))


def h_ratio(run, rid):
    S, _ = load_py(rid)
    mx = float(np.nanmax(S))
    write(rid, S, "qualitative", "analytic",
          "analytic threshold: |acc|^2 - 1 > 0 needs Im(acc)^2 >= 2^-53, i.e. |Im csd|/|Re csd| >~ 1.05e-8",
          pass_flag=int(mx < 1.05e-8), max_ratio=mx)


def h_pinf(run, rid):
    S, _ = load_py(rid)
    r = np.linspace(0.0, 3e-8, 61)
    V = np.empty_like(S)
    first_pos, bitwise_at_pos = [], []
    for q, re in enumerate((1.0, 0.37, 2.9e-7)):
        csd = re + 1j * re * r
        d = np.abs(csd)
        u = csd / d
        res = (u * np.conj(u)).real - 1.0
        V[q] = np.where(res > 0, np.inf, np.where(res < 0, -np.inf, np.nan))
        pos = res > 0
        first_pos.append(float(r[pos].min()) if pos.any() else np.nan)
        bitwise_at_pos.append(int(np.sum(pos & (d == np.abs(csd.real)))))
    write(rid, V, "exact", "analytic",
          "analytic sign rule x/0 = sign(x) inf, 0/0 = NaN on the residue |csd/|csd||^2 - 1 (N = 1)",
          first_ratio_posinf=first_pos, n_posinf_with_bitwise_abs_equal=bitwise_at_pos)


def h_dsplisyn(run, rid):
    write(rid, np.array([np.nan, -np.inf]), "exact", "analytic",
          "analytic: N = 1 -> (sign^2 - 1)/0: 0/0 = NaN where Im != 0, -1/0 = -inf where Im == 0; nanmean is "
          "NaN without an exact zero and -inf with one")


def h_wavelets(run, rid):
    S, _ = load_py(rid)
    V = np.column_stack([np.ones(125), np.full(125, SQ2), np.zeros(125), np.full(125, 1 / (14 * np.pi))])
    ulp = np.spacing(SQ2)
    canon_len = [ks.morlet_wavelet(f, 7.0).size for f in FREQS]
    W7 = ks.morlet_wavelet(0.25, 7.0)
    H = np.abs(np.fft.fft(W7, 4096))
    fr = np.fft.fftfreq(4096)
    half = fr[(H >= H.max() / 2) & (fr > 0)]
    W1 = ks.morlet_wavelet(0.25, 0.25 / 7.0)
    H1 = np.abs(np.fft.fft(W1, 4096))
    write(rid, V, "exact", "analytic",
          "analytic: sigma_t = (f/7)/(2 pi f) = 1/(14 pi) = 0.0227 < 1/5 sample -> t = [0], W = (1-c)/(sqrt(.5)|1-c|) "
          "= sqrt(2) (mne tfr.py:159-178)",
          n_distinct_W=int(np.unique(S[:, 1]).size), max_W_dev_ulps=float(np.max(np.abs(S[:, 1] - SQ2)) / ulp),
          canon7_len_min=int(min(canon_len)), canon7_len_max=int(max(canon_len)),
          canon7_f025_halfpower_band=[float(half.min()), float(half.max())],
          pipeline_wavelet_response_flatness=float(H1.max() - H1.min()))


def h_cwtx(run, rid):
    S, xin = load_py(rid)
    V = np.stack([np.broadcast_to(SQ2 * xin[:, None, :], S[0].shape), np.zeros(S[1].shape)])
    write(rid, V, "tolerance", "analytic", "analytic: X(f, t) = sqrt(2) x(t) for every f (one-sample wavelet)",
          rtol=1e-12, atol=1e-12,
          max_abs_re_minus_sqrt2x=float(np.abs(S[0] - V[0]).max()), max_abs_im=float(np.abs(S[1]).max()),
          frac_im_exact_zero=float(np.mean(S[1] == 0)),
          max_spread_over_freqs=float((S[0].max(axis=1) - S[0].min(axis=1)).max()),
          max_abs_x=float(np.abs(xin).max()))


def h_cwtdirect(run, rid):
    S, xin = load_py(rid)
    W0 = np.array([ks.morlet_wavelet(f, f / 7.0)[0].real for f in FREQS])
    V = np.stack([xin[:, None, :] * W0[None, :, None], np.zeros(S[1].shape)])
    write(rid, V, "exact", "analytic",
          "analytic: direct convolution with a length-1 kernel is the product x(t) * W0(f) (W0 from the closed form)",
          notes="rtol = atol = 0: bitwise", rtol=0.0, atol=0.0,
          bitwise_equal_x_times_W0=int(np.array_equal(S, V)),
          bitwise_equal_x_times_sqrt2=int(np.array_equal(S[0], np.broadcast_to(SQ2 * xin[:, None, :], S[0].shape))),
          n_W0_not_sqrt2=int(np.sum(W0 != SQ2)))


def h_mtadapt(run, rid):
    S, _ = load_py(rid)
    write(rid, np.zeros_like(S), "exact", "same-lineage",
          "library vs library: spectral_connectivity_epochs(mt_adaptive=True) vs False for the nine measures; "
          "expected 0 (epochs.py:337-341 use mt_adaptive only in the multitaper branch)",
          notes="columns: max |diff|, count not bitwise equal")


def h_freqs(run, rid):
    S, _ = load_py(rid)
    T = run["T"]
    doc = np.linspace(4.0 / T, 0.5, 125)
    write(rid, FREQS, "exact", "analytic", "analytic: pyspi grid linspace(0.2, 0.5, 125) for every T (wavelet.py:54)",
          n_low_band=int(np.sum((S >= 0) & (S <= 0.25))), n_high_band=int(np.sum((S >= 0.25) & (S <= 0.5))),
          documented_first_bin=float(4.0 / T), documented_n_low_band=int(np.sum(doc <= 0.25)))


def h_short(run, rid):
    S, _ = load_py(rid)
    n = int(run["probe"]["n"])
    P = S.shape[-1]
    fmin_lib = 5.0 / n
    kept = FREQS[FREQS >= fmin_lib]
    V = []
    for k in range(6):
        _, fmin, fmax = IDS[k]
        nk = np.sum((kept >= fmin) & (kept <= fmax))
        V.append(const_matrix(P, 1.0 if nk > 0 else np.nan))
    info = {}
    f = EXCHANGE / (rid.replace("|", "__").replace("/", "_") + ".json")
    if f.is_file():
        info = json.loads(f.read_text())
    write(rid, np.stack(V), "exact", "analytic",
          "analytic: mne-connectivity keeps cwt freqs >= fmin = 5/T (epochs.py:141-153); an empty band gives NaN "
          "(nanmean) or an exception (nanmax, Calculator stores NaN)",
          n_kept_expected=int(kept.size), n_kept_cached=str(info.get("n_freqs_cached")),
          errors=json.dumps(info.get("errors"))[:300])


def h_orient7(run, rid):
    S, xin = load_py(rid)
    xin = prep(xin, run["path"])
    P = xin.shape[0]
    X = ks.cwt_morlet(xin, FREQS, 7.0)
    M = np.zeros((P, P, FREQS.size), dtype=complex)
    for i in range(P):
        for j in range(i):
            csd = X[i] * X[j].conj()
            M[i, j] = (csd / np.sqrt((np.abs(X[i]) ** 2) * (np.abs(X[j]) ** 2))).mean(-1)
    V = np.stack([M.real, M.imag])
    ang = np.angle(S[0] + 1j * S[1])
    k25 = np.argmin(np.abs(FREQS - 0.25))
    extra = dict(lib_angle_10_f025=float(ang[1, 0, k25]), upper_triangle_max_abs=float(np.abs(S[:, 0, 1:]).max()))
    if run["dataset"] == "delayed_noise":
        k02 = 0
        extra.update(truth_angle_10_f025=float(-2 * np.pi * FREQS[k25] * 1),
                     lib_angle_10_f02=float(ang[1, 0, k02]), truth_angle_10_f02=float(-2 * np.pi * FREQS[k02] * 1))
    else:
        # D7: x_k = sin(2 pi f0 t + phase_k), phase = (0, +pi/4, -pi/2); conn[hi, lo] = X_hi conj X_lo
        extra.update(truth_angle_10_f025=float(np.pi / 4), lib_angle_20_f025=float(ang[2, 0, k25]),
                     truth_angle_20_f025=float(-np.pi / 2))
    write(rid, V, "tolerance", "independent-impl",
          f"{SELF}:h_orient7 (_kernels_spectral.cwt_morlet n_cycles 7; per-point X_i conj X_j/(|X_i||X_j|) for i > j, "
          "time mean; upper triangle and diagonal 0 as mne's dense output)", rtol=1e-6, atol=1e-9, **extra)


def h_orient1(run, rid):
    S, xin = load_py(rid)
    xin = prep(xin, run["path"])
    P = xin.shape[0]
    L = np.zeros((P, P))
    Mx = np.zeros((P, P))
    for i in range(P):
        for j in range(i):
            L[i, j] = np.mean(np.sign(xin[i] * xin[j]))
            Mx[i, j] = 1.0
    write(rid, np.stack([L, Mx]), "tolerance", "analytic",
          "analytic: cached cohy conn[hi, lo](f, t) = sign(z_hi z_lo) + i round-off for every f, |conn| = 1; "
          "upper triangle and diagonal 0 before pyspi's mirror", rtol=0.0, atol=1e-12)


def h_fscache(run, rid):
    S, _ = load_py(rid)
    A1, A10s, A10f = S
    same = bool(np.array_equal(A1, A10s, equal_nan=True))
    m = ~np.eye(A1.shape[0], dtype=bool)
    diff = float(np.nanmax(np.abs(A10f[m] - A1[m])))
    write(rid, S, "qualitative", "analytic",
          "code reading wavelet.py:48-74: cache key data.mne[measure] ignores fs; fs=10 gives sigma_t*fs = 0.227 "
          "samples -> 3-sample wavelets, so a fresh fs=10 result differs",
          notes="pass iff the fs=10 SPI on the shared Data returns the fs=1 result bitwise and a fresh fs=10 "
                "Data differs by > 1e-6", pass_flag=int(same and diff > 1e-6),
          shared_equals_fs1=int(same), fresh_fs10_minus_fs1=diff)


HANDLERS = dict(six=h_six, ident=h_ident, rep=h_rep, scratch=h_scratch, canon=h_canon,
                canonep=lambda run, rid: h_canon(run, rid, "canonep"),
                canonwin=lambda run, rid: h_canon(run, rid, "canonwin"), perm=h_perm,
                roundoff=h_roundoff, pts=h_pts, ratio=h_ratio, pinf=h_pinf, dsplisyn=h_dsplisyn,
                wavelets=h_wavelets, cwtx=h_cwtx, cwtdirect=h_cwtdirect, mtadapt=h_mtadapt, freqs=h_freqs,
                short=h_short, orient7=h_orient7, orient1=h_orient1, fscache=h_fscache)


def main():
    man = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())
    n_ok = n_err = 0
    for run in man["runs"]:
        rid = run["run_id"]
        if not (OUT_PY / f"{rid}.npz").is_file():
            print(f"skip (no pyspi output) {rid}")
            continue
        tag = rid.split("|")[-1].split(":")[0]
        try:
            HANDLERS[tag](run, rid)
            n_ok += 1
        except Exception as exc:  # report, keep going
            import traceback
            traceback.print_exc()
            print(f"ERROR {rid}: {exc!r}")
            n_err += 1
    print(f"{FAMILY}: {n_ok} references written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
