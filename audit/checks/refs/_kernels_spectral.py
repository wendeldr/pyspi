#!/usr/bin/env python3
"""Reference kernels: multitaper spectra and spectral connectivity (pyspi audit).

Independent numpy/scipy implementations used to judge pyspi's spectral SPIs,
which delegate to ``spectral_connectivity`` 1.1.2 (multitaper family) and
``mne`` / ``mne_connectivity`` (wavelet family).  Neither library is imported
here; their conventions are mirrored where stated, citing the library file and
line (paths relative to ``.venv/lib/python3.12/site-packages/``; ``SC/`` means
``spectral_connectivity/``, ``mpd`` = ``SC/minimum_phase_decomposition.py``).

Conventions used throughout
---------------------------
* Data ``x`` is "ps" ordered, shape (P, T) (as ``make_data``), fs = 1 so
  frequencies are in cycles/sample.
* Fourier coefficients ``coeffs`` have shape (K, n_fft, P) (tapers first, as
  SC/transforms.py:243 after dropping the window/trial axes); any extra leading
  axes (e.g. segments) are treated as further observations.
* ``X(f) = sum_t w(t) x(t) exp(-2 pi i f t)`` (forward FFT), tapers have unit
  energy, so ``S(f)`` estimates the TWO-SIDED density with
  ``integral_{-1/2}^{1/2} S df = Gamma(0)``; this equals make_data's
  ``S_true = H Sigma H^H`` (same sign convention).
* ``S[f, i, j] = E[X_i(f) conj(X_j(f))]`` (SC/connectivity.py:196-207 with
  ``_complex_inner_product`` = a @ b^H, SC/connectivity.py:1187-1190).
* Directed quantities use ``M[i, j]`` = from source j to target i (row =
  target), like make_data's ``A[i, j]`` and the library's H.
* Two-sided arrays follow the FFT bin order of ``np.fft.fftfreq(n_fft)``; the
  library keeps the first ``(n_fft + 1) // 2`` bins (DC kept, Nyquist dropped
  for even n_fft; SC/connectivity.py:73-89) -- see ``nonneg``.

Run ``python _kernels_spectral.py --selftest``.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.fft
import scipy.signal.windows

CHECKS_DIR = "/media/dan/Data/git/pyspi/audit/checks"


# ------------------------------------------------------------------ grid helpers
def next_fast_len_235(n: int) -> int:
    """Smallest 5-smooth integer >= n (2^a 3^b 5^c).

    Mirrors ``scipy.fftpack.next_fast_len`` which SC uses for ``n_fft``
    (SC/transforms.py:24-29 import, :181-185 use).  NB ``scipy.fft.next_fast_len``
    is 11-smooth for complex input and can differ (97 -> 98 vs 100).
    """
    m = max(int(n), 1)
    while True:
        r = m
        for q in (2, 3, 5):
            while r % q == 0:
                r //= q
        if r == 1:
            return m
        m += 1


def nonneg(a, axis=0):
    """First ``(n + 1) // 2`` entries along ``axis`` (SC/connectivity.py:73-89)."""
    a = np.asarray(a)
    return np.take(a, np.arange((a.shape[axis] + 1) // 2), axis=axis)


def band_mask(freqs, fmin, fmax, inclusive=True):
    """Boolean frequency mask.  ``inclusive`` = ``fmin <= f <= fmax`` as pyspi's
    multivariate reducer (pyspi/statistics/spectral.py:88); ``inclusive=False``
    = strict, as pyspi's bivariate reducer (spectral.py:132) and ``library_bandpass``."""
    f = np.asarray(freqs)
    return (f >= fmin) & (f <= fmax) if inclusive else (f > fmin) & (f < fmax)


def library_bandpass(data, freqs, fmin, fmax, axis=0):
    """SC ``_bandpass``: STRICT ``fmin < f < fmax`` (SC/connectivity.py:1313-1335).
    Returns (data restricted along ``axis``, restricted freqs)."""
    m = band_mask(freqs, fmin, fmax, inclusive=False)
    return np.take(data, np.nonzero(m)[0], axis=axis), np.asarray(freqs)[m]


def library_frequency_step(freqs, frequency_resolution):
    """SC ``_get_independent_frequency_step`` = ceil(resolution / df)
    (SC/connectivity.py:1338-1357); pyspi passes resolution (fmax-fmin)/50
    (pyspi/statistics/spectral.py:296-301)."""
    return int(np.ceil(frequency_resolution / (freqs[1] - freqs[0])))


# --------------------------------------------------------------------- multitaper
def dpss_tapers(T, NW=3.0, K=None, eig_min=0.9, fs=1.0, low_bias=True):
    """Slepian (DPSS) tapers, Thomson 1982 / Slepian 1978, via
    ``scipy.signal.windows.dpss(T, NW, Kmax=K, return_ratios=True)`` (unit L2 norm).

    Mirrors: K default ``floor(2 NW - 1)`` (SC/transforms.py:162-163); keep tapers
    with concentration eigenvalue > ``eig_min`` = 0.9, falling back to the single
    best taper if none qualify (SC/transforms.py:668-673); scale by sqrt(fs)
    (SC/transforms.py:387).  Taper signs agree with SC's ``_fix_taper_sign``
    (probe: max |diff| < 1e-13 for T = 64..1024, NW=3, K=5).

    VALID FOR T >= 9 (NW=3, K=5: identical sets vs SC for T = 9..1099, 2048, 4096).
    At T = 7, 8 SC's inverse iteration divides by zero (SC/transforms.py:423); one
    taper/eigenvalue is NaN (index 2 at T=7, index 3 at T=8) and is dropped (4
    kept); this kernel keeps all 5
    genuine tapers (T=8 ratio 0.9993 = SC's Percival-Walden estimate, :676-697).

    Returns tapers (K', T) and eigenvalues (K',).
    """
    K = int(np.floor(2 * NW - 1)) if K is None else int(K)
    w, lam = scipy.signal.windows.dpss(int(T), NW, Kmax=K, return_ratios=True)
    w, lam = np.atleast_2d(w), np.atleast_1d(lam)
    if low_bias:
        keep = lam > eig_min
        if not keep.any():
            keep = np.arange(lam.size) == np.argmax(lam)
        w, lam = w[keep], lam[keep]
    return w * np.sqrt(fs), lam


def multitaper_fft(x, NW=3.0, K=None, n_fft=None, fs=1.0, detrend=True, eig_min=0.9):
    """Per-taper Fourier coefficients (Thomson 1982 multitaper estimator).

    ``X_k(f) = fft(w_k(t) (x(t) - mean x), n_fft) / fs``.  Mirrors SC: constant
    detrend per signal (SC/transforms.py:79, :253-254), n_fft =
    ``next_fast_len_235(T)`` (zero padding when T is not 5-smooth;
    SC/transforms.py:181-185), division by fs (SC/transforms.py:354).

    Parameters: x (P, T) or (T,).  Returns coeffs (K', n_fft, P) complex and the
    two-sided grid ``freqs = fftfreq(n_fft, 1/fs)`` (SC/transforms.py:188-190).
    The library reports only ``nonneg(freqs)`` = first (n_fft+1)//2 bins: DC
    included; for even n_fft the Nyquist bin (index n_fft/2, labelled -fs/2) is
    excluded, so the top retained frequency is fs/2 - fs/n_fft.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    P, T = x.shape
    if detrend:
        x = x - x.mean(axis=1, keepdims=True)
    n_fft = next_fast_len_235(T) if n_fft is None else int(n_fft)
    w, _ = dpss_tapers(T, NW, K, eig_min=eig_min, fs=fs)
    X = np.fft.fft(w[:, None, :] * x[None, :, :], n=n_fft, axis=-1) / fs   # (K, P, n_fft)
    return np.ascontiguousarray(X.transpose(0, 2, 1)), np.fft.fftfreq(n_fft, 1.0 / fs)


def segment_coeffs(x, n_segments, NW=3.0, K=None):
    """Coefficients for ``n_segments`` non-overlapping segments, shape
    (n_segments, K', n_fft, P); passing them to the per-observation measures takes
    the expectation over segments x tapers (SC 'trials_tapers',
    SC/connectivity.py:45-53, 145-150)."""
    x = np.atleast_2d(x)
    L = x.shape[1] // n_segments
    return np.stack([multitaper_fft(x[:, s * L:(s + 1) * L], NW, K)[0] for s in range(n_segments)])


def _obs_axes(c):
    return tuple(range(c.ndim - 2))


def _per_obs_csm(coeffs, matmul=False):
    """X_i conj(X_j) per observation: (..., n_fft, P, P).

    ``matmul=False`` (default) forms the product elementwise.  ``matmul=True`` forms it
    with ``np.matmul`` on (P, 1) x (1, P) blocks, the operation SC uses
    (``_cross_spectral_matrix`` SC/connectivity.py:196-207 via ``_complex_inner_product``
    :1187-1190).  The two agree to rounding.  They differ in kind only where the exact
    Im S is 0: on the diagonal and for bit-identical channels the matmul returns Im S
    exactly 0, as SC does, while the elementwise product leaves rounding noise of order
    1e-16..1e-15 (numpy 2.5.4 in both venvs; scratchpad probe
    kernel_fix/probes/pli_matmul_vs_sc.py, which audit/checks/archive_probes.py packs into
    audit/checks/probes/scratch_probes_2026-10-15.tar.xz under the same relative path,
    and the ``tie_*`` keys of out/results/spectral_phase_sync.json for the elementwise
    noise).  Only the sign- and Im-based phase-lag measures below are sensitive to that
    noise."""
    c = np.asarray(coeffs)
    if matmul:
        return np.matmul(c[..., :, None], np.conj(c[..., None, :]))
    return c[..., :, None] * np.conj(c[..., None, :])


def cross_spectral_matrix(coeffs, nonnegative=False):
    """``S[f, i, j] = mean_obs X_i(f) conj(X_j(f))`` (Thomson 1982; Percival &
    Walden 1993 ch. 7).  Conjugation order mirrors SC/connectivity.py:196-207 and
    :1187-1190 (``a @ b^H``), so ``angle S_ij = phi_i - phi_j``.  Mean over all
    leading (taper/segment) axes = SC 'trials_tapers'.  Returns (n_fft, P, P),
    or (n_fft+1)//2 bins when ``nonnegative``."""
    S = _per_obs_csm(coeffs).mean(axis=_obs_axes(np.asarray(coeffs)))
    return nonneg(S, 0) if nonnegative else S


def power(S):
    """Auto-spectra = real diagonal of S, shape (n_f, P) (SC/connectivity.py:189-193)."""
    return np.real(np.diagonal(S, axis1=-2, axis2=-1))


# ------------------------------------------------------------ undirected measures
def coherency(S, nan_diagonal=False):
    """Complex coherency ``C_ij = S_ij / sqrt(S_ii S_jj)`` (Nolte et al. 2004).
    Zero normaliser -> NaN; SC sets the diagonal to NaN (SC/connectivity.py:295-313)
    -- ``nan_diagonal=True`` mirrors that."""
    p = power(S)
    norm = np.sqrt(p[..., :, None] * p[..., None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = S / np.where(norm == 0, np.nan, norm)
    if nan_diagonal:
        i = np.arange(S.shape[-1])
        C[..., i, i] = np.nan
    return C


def coherence_magnitude(S, squared=True):
    """Magnitude-squared coherence ``|C|^2`` (library form, SC/connectivity.py:326-343);
    ``squared=False`` gives the modulus |C| (canonical 'coherence' in many texts)."""
    a = np.abs(coherency(S))
    return a ** 2 if squared else a


def coherence_phase(S):
    """``angle(C_ij) = phi_i - phi_j`` in (-pi, pi] (SC/connectivity.py:315-324)."""
    return np.angle(coherency(S))


def imaginary_coherence(S, signed=False):
    """Imaginary coherency ``Im C_ij`` (Nolte et al. 2004).  Library returns
    ``|Im S_ij| / sqrt(S_ii S_jj)`` (SC/connectivity.py:346-372); ``signed=True``
    gives Nolte's signed ``Im C_ij`` (antisymmetric)."""
    v = np.imag(coherency(S))
    return v if signed else np.abs(v)


def phase_locking_value(coeffs, modulus=True):
    """PLV = ``|E_obs[ S^(k)_ij / |S^(k)_ij| ]|`` (Lachaux et al. 1999, with the
    expectation over tapers, as SC/connectivity.py:511-540).  For Lachaux's
    expectation over trials/segments pass ``segment_coeffs`` output (expectation
    then runs over segments x tapers); for single-taper segments use K=1.
    Returns (n_fft, P, P) real (complex mean if ``modulus=False``)."""
    c = np.asarray(coeffs)
    s = _per_obs_csm(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (s / np.abs(s)).mean(axis=_obs_axes(c))
    return np.abs(m) if modulus else m


def phase_lag_index(coeffs, signed=True, matmul=False):
    """PLI = ``E_obs[sign Im S^(k)_ij]`` (Stam et al. 2007).  Library is signed
    (SC/connectivity.py:542-575); Stam's PLI is ``|.|`` (``signed=False``).

    SCOPE (this function, weighted_phase_lag_index, debiased_squared_pli and
    debiased_squared_wpli).  With the default ``matmul=False`` the per-observation S is
    formed elementwise (``_per_obs_csm``), so on the diagonal and for bit-identical
    channels Im S is rounding noise where SC has exactly 0.  The sign of that noise is
    arbitrary, so those entries can differ from SC by O(1): 1.25 for dsPLI on the
    bit-identical pair (0, 2) of shared_envelope T=1024 (``ref_no_tie_rule_max_abs_diff_pair02``
    of run ``spectral.dspli|shared_envelope|1024|kernel|perbin``).  All other entries follow
    SC's formula.  ``matmul=True`` forms S as SC does and reproduces its exact zeros, hence
    SC's values at those entries: diagonal PLI = 0, wPLI = 0 (denominator floor),
    dsPLI = -1/(N-1), dswPLI = NaN (0/0).  Fed the same kernel coefficients, SC and
    ``matmul=True`` agree exactly for PLI, wPLI and dsPLI and to 2.1e-14 for dswPLI, NaN
    pattern and diagonal included (shared_envelope T=256/1024, delayed_noise, iid_gaussian,
    var2_bivariate, narrowband_f0-0.25; scratchpad kernel_fix/probes/pli_matmul_vs_sc.py).
    Neither setting masks the diagonal; pyspi overwrites it with NaN
    (pyspi/calculator.py:332).  refs/spectral_phase_sync.py:csm instead zeroes Im S where
    ``|Im S| <= 1e-12 max_k |S|``; on those datasets (T = 64..1024, raw and z-scored) that
    rule zeroes exactly the entries where the matmul gives 0 (scratchpad probe
    kernel_fix/probes/tie_pattern_allbins.py).  audit/checks/archive_probes.py packs both
    scratchpad probes cited here into audit/checks/probes/scratch_probes_2026-10-15.tar.xz
    under the same relative paths."""
    c = np.asarray(coeffs)
    v = np.sign(np.imag(_per_obs_csm(c, matmul))).mean(axis=_obs_axes(c))
    return v if signed else np.abs(v)


def weighted_phase_lag_index(coeffs, signed=True, matmul=False):
    """wPLI = ``E[Im S] / E[|Im S|]`` (Vinck 2011, wPLI definition; Vinck's is the
    modulus ``|E[Im S]| / E[|Im S|]``).  Library: signed numerator and
    denominator < eps replaced by 1 (SC/connectivity.py:577-602), mirrored.
    ``matmul``: see the SCOPE note of ``phase_lag_index``."""
    c = np.asarray(coeffs)
    im = np.imag(_per_obs_csm(c, matmul))
    num, den = im.mean(axis=_obs_axes(c)), np.abs(im).mean(axis=_obs_axes(c))
    den = np.where(den < np.finfo(float).eps, 1.0, den)
    return num / den if signed else np.abs(num) / den


def _n_obs(c):
    return int(np.prod(np.asarray(c).shape[:-2]))


def debiased_squared_pli(coeffs, matmul=False):
    """Debiased PLI^2 ``(N PLI^2 - 1) / (N - 1)``, N = number of observations
    (tapers) (Vinck 2011, debiased estimator; formula as SC/connectivity.py:604-625).
    ``matmul``: see the SCOPE note of ``phase_lag_index``."""
    N = _n_obs(coeffs)
    return (N * phase_lag_index(coeffs, matmul=matmul) ** 2 - 1.0) / (N - 1.0)


def debiased_squared_wpli(coeffs, matmul=False):
    """Debiased wPLI^2 (Vinck 2011): ``sum_{j!=k} Im_j Im_k / sum_{j!=k} |Im_j||Im_k|``
    = ``((sum Im)^2 - sum Im^2) / ((sum |Im|)^2 - sum Im^2)``; zero denominator
    -> NaN, as SC/connectivity.py:627-670.  ``matmul``: see the SCOPE note of
    ``phase_lag_index``."""
    c = np.asarray(coeffs)
    im, ax = np.imag(_per_obs_csm(c, matmul)), _obs_axes(c)
    s, s2, sa = im.sum(axis=ax), (im ** 2).sum(axis=ax), np.abs(im).sum(axis=ax)
    den = sa ** 2 - s2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s ** 2 - s2) / np.where(den == 0, np.nan, den)


def pairwise_phase_consistency(coeffs):
    """PPC = ``2/(N(N-1)) sum_{j<k} cos(theta_j - theta_k)`` = ``(|sum_k e^{i theta_k}|^2
    - N) / (N^2 - N)`` (Vinck et al. 2010, definition and closed form), N = number of
    tapers (x segments), as SC/connectivity.py:673-696."""
    N = _n_obs(coeffs)
    z = phase_locking_value(coeffs, modulus=False) * N
    return (np.abs(z) ** 2 - N) / (N ** 2 - N)


# ----------------------------------------------------------------- phase slope
def psi_adjacent(C, freqs, fmin, fmax, inclusive=False, step=1):
    """Phase slope index, Nolte 2008 eq. 3: ``Psi_ij = Im sum_{f in F} conj(C_ij(f))
    C_ij(f + df)`` summed over ADJACENT frequency pairs inside the band (``step``
    subsamples the band first, like SC's independent-frequency step).  C is
    (n_f, P, P) on grid ``freqs``.  Sign: with ``S_ij = X_i conj X_j``, if
    ``x_j(t) = x_i(t - d)`` then ``angle C_ij = +2 pi f d`` and ``Psi_ij > 0``:
    positive Psi_ij means i leads (drives) j, as in Nolte 2008.  Antisymmetric."""
    m = band_mask(freqs, fmin, fmax, inclusive)
    Cb = np.asarray(C)[m][::step]
    return np.imag((np.conj(Cb[:-1]) * Cb[1:]).sum(axis=0))


def psi_all_pairs(C, freqs, fmin, fmax, step=1):
    """What SC actually computes: ``Im sum_{a<b} conj(C(f_a)) C(f_b)`` over ALL
    unordered pairs of the strictly band-passed, step-subsampled frequencies
    (SC/connectivity.py:1061-1104, ``_inner_combination`` :1107-1113, strict
    ``_bandpass`` :1313-1335).  This is not Nolte's adjacent-pair sum: distant
    pairs contribute ``sin(2 pi d (f_b - f_a))`` terms that change sign once
    ``d (f_b - f_a) > 1/2``.  Computed in O(n) by prefix sums (algebraically
    identical to the pair loop).

    Raises ValueError where SC fails instead of returning an empty-sum 0: < 2
    bins after band-pass + step (SC IndexError in ``_inner_combination``) and
    step < 1, e.g. fmin == fmax (SC ZeroDivisionError, :1095-1101).  With pyspi's
    band (0, 0.5) the first case needs T <= 4 (at most one strictly interior bin,
    e.g. f = 0.25 at T = 4)."""
    if int(step) < 1:
        raise ValueError(f"psi_all_pairs: frequency step {step} < 1 (SC raises ZeroDivisionError)")
    Cb, _ = library_bandpass(C, freqs, fmin, fmax, axis=0)
    Cb = Cb[::int(step)]
    if Cb.shape[0] < 2:
        raise ValueError(f"psi_all_pairs: {Cb.shape[0]} bin(s) in ({fmin}, {fmax}) after step {step}; "
                         "SC raises IndexError")
    prefix = np.cumsum(Cb, axis=0) - Cb                      # sum_{a<b} C_a
    return np.imag((np.conj(prefix) * Cb).sum(axis=0))


def group_delay_fit(phase, freqs, mask=None):
    """Gotman 1983 group delay: ordinary least squares of the unwrapped coherency
    phase on frequency, ``delay = slope / (2 pi)`` (SC/connectivity.py:960-971
    takes the same slope/(2 pi); its significance masking via coherence_bias and
    'largest independent group', :946-959 and :1432, is deliberately NOT replicated).  For
    ``phase = angle C_ij`` a positive delay means i leads j by ``delay`` samples
    (SC stores +slope at [i, j], i<j, and -slope at [j, i]).

    SCOPE: a reference for SC ONLY where coherence is significant over the whole
    band (mask keeps every bin; delayed_noise T=1024 truncated to its first 1021
    samples: SC [1.010 2.989 5.000], kernel [1.007 2.990 4.998]).  Elsewhere they
    legitimately differ (var2 T=256: SC -0.472, kernel 1.309) or SC is all-NaN
    (iid_gaussian T=1024 truncated to 300 samples, var2 T=4096; make_data.load
    only serves T in {64, 256, 1024, 4096}, so slice [:, :n] for other lengths);
    such differences are NOT evidence of a pyspi error.

    ORDER OF OPERATIONS: with a ``mask`` this function drops the masked bins FIRST and
    unwraps only the kept phases; SC unwraps over every band-passed bin and masks
    afterwards (``np.ma.masked_array(xp.unwrap(xp.angle(...)), mask=~is_significant)``,
    SC/connectivity.py:953-959).  For a contiguous run of kept bins the two unwrapped
    phases differ by a constant multiple of 2 pi (same slope, different intercept); across
    a gap of masked bins they can differ by multiples of 2 pi and so give a different slope.

    phase: (n_f,) or (n_f, ...); returns dict delay, slope, intercept, r."""
    f = np.asarray(freqs, dtype=float)
    ph = np.asarray(phase)
    if mask is not None:
        f, ph = f[mask], ph[mask]
    y = np.unwrap(ph, axis=0).reshape(f.size, -1)
    fc, yc = f - f.mean(), y - y.mean(axis=0)
    slope = (fc @ yc) / (fc @ fc)
    r = (fc @ yc) / np.sqrt((fc @ fc) * (yc ** 2).sum(axis=0))
    shp = ph.shape[1:]
    return {"delay": (slope / (2 * np.pi)).reshape(shp), "slope": slope.reshape(shp),
            "intercept": (y.mean(axis=0) - slope * f.mean()).reshape(shp), "r": r.reshape(shp)}


# ------------------------------------------------- spectral factorisation & VAR
def wilson_factorise(S, tol=1e-8, max_iter=60, halve_offdiag=True):
    """Wilson 1972 spectral factorisation as used by Dhamala, Rangarajan & Ding
    2008 (NeuroImage 41:354, appendix): find causal psi(f) = sum_{k>=0} psi_k
    e^{-2 pi i f k} with ``S = psi psi^H`` by ``psi <- psi [g]_+``,
    ``g = psi^{-1} S psi^{-H} + I``.  Then ``Sigma = psi_0 psi_0^T`` and
    ``H = psi psi_0^{-1}`` (Dhamala 2008 appendix; SC/connectivity.py:1116-1170).

    S: TWO-SIDED (n_fft, P, P) on the fftfreq grid.  Mirrored from mpd: initial
    psi = Cholesky(Re ifft(S)[0])^T (mpd:44-47), causal part = ifft lags
    0..(n+1)//2-1 kept (mpd:98), convergence when max|psi_new - psi_old| < tol
    (mpd:102-125), 60 iterations max (mpd:161-162).  Lag-0 plus-operator:
    ``halve_offdiag=True`` = mpd:88-95 (triu(0.5 g_0), halves upper off-diagonals
    too); ``False`` = Wilson's split (strict upper + half diagonal) with
    ``[g0]_+ + [g0]_+^H = g0``.  Both have the same fixed point (g = 2I).

    Like mpd, H and Sigma are returned even if unconverged, and ``converged`` is
    only the step-size test (can be True while psi psi^H != S).  Check BOTH
    ``converged`` and ``recon_err`` (max|psi psi^H - S|/max|S|; 2e-3..8e-2 on
    multitaper CSMs from grid truncation) before using DTF/PDC/gPDC/dDTF/DC/
    Granger built on it: narrowband_f0-0.25 gives 0.4-0.6 (unusable, though H
    matches SC to ~1e-8).  A singular lag-0 covariance raises LinAlgError only when the
    Cholesky of the rounded lag-0 matrix fails.  A constant channel (zero after the default
    detrend) raised at every T tried; a duplicated channel depends on rounding: on
    z-scored shared_envelope (ch 2 == ch 0) pair 0/2 and all 4 channels raise at T = 256
    and 4096, and the raw pair 0/2 raises at T = 1024.  Where SC's Cholesky fails, SC
    dies with AttributeError instead (mpd:48 uses ``xp.linalg.linalg``, gone in numpy 2).
    On z-scored shared_envelope T = 1024 the Cholesky succeeds, as SC's start does there
    (audit/notes/sgc.md), and the factorisation runs 60 iterations unconverged (recon_err
    0.21 for pair 0/2, 0.49 for all 4; multitaper_fft + cross_spectral_matrix defaults).

    START DEPENDENCE: on a CSM that is not exactly factorisable (any finite-sample
    multitaper estimate) the step-converged psi depends on the Cholesky start, so H and
    Sigma are not equivariant under channel rescaling or reordering (per-bin H moved by up
    to O(1) at isolated bins under rescaling; swapping the channel order moves per-bin DTF
    by 5.3e-3 on var2_bivariate; audit/notes/spectral_directed.md).  SC uses the same start
    and stopping rule and shows the same start dependence (audit/notes/sgc.md: under channel
    scaling the from-scratch value converges to the same scale-dependent fixed point at
    tolerance 1e-14).  On the exact VAR spectrum S_true the fixed point is the true factor
    (selftest (3): max|H - H_true| <= 4.6e-15).

    Returns H (n_fft, P, P), Sigma (P, P), info {n_iter, converged, recon_err}.
    """
    S = np.asarray(S, dtype=np.complex128)
    n, P, _ = S.shape
    I = np.eye(P)
    psi = np.broadcast_to(np.linalg.cholesky(np.fft.ifft(S, axis=0)[0].real).T, S.shape)
    psi = psi.astype(np.complex128)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        pinv = np.linalg.inv(psi)
        g = pinv @ S @ np.conj(np.swapaxes(pinv, -1, -2)) + I
        gk = np.fft.ifft(g, axis=0)
        g0 = gk[0]
        gk[0] = np.triu(0.5 * g0) if halve_offdiag else np.triu(g0, 1) + 0.5 * np.diag(np.diag(g0))
        gk[(n + 1) // 2:] = 0
        new = psi @ np.fft.fft(gk, axis=0)
        err = np.abs(new - psi).max()
        psi = new
        if err < tol:
            converged = True
            break
    A0 = np.fft.ifft(psi, axis=0)[0].real
    H = psi @ np.linalg.inv(A0)
    Sigma = A0 @ A0.T
    recon = np.abs(psi @ np.conj(np.swapaxes(psi, -1, -2)) - S).max() / np.abs(S).max()
    return H, Sigma, {"n_iter": it, "converged": converged, "recon_err": float(recon)}


def var_transfer_function(A_list, freqs, Sigma=None):
    """Analytic VAR(p) frequency response (Luetkepohl 2005 sec. 2.1; Geweke 1982):
    ``A(f) = I - sum_k A_k exp(-2 pi i f k)``, ``H(f) = A(f)^{-1}``,
    ``S(f) = H Sigma H^H`` (two-sided density, same convention as make_data's
    ``var_spectral_matrix`` / stored S_true).  Returns H, A_f (n_f, P, P) and S
    (None if Sigma is None)."""
    f = np.asarray(freqs, dtype=float)
    P = np.asarray(A_list[0]).shape[0]
    Af = np.broadcast_to(np.eye(P, dtype=complex), (f.size, P, P)).copy()
    for k, A in enumerate(A_list, start=1):
        Af -= np.asarray(A)[None] * np.exp(-2j * np.pi * f * k)[:, None, None]
    H = np.linalg.inv(Af)
    S = None if Sigma is None else H @ np.asarray(Sigma) @ np.conj(np.swapaxes(H, -1, -2))
    return H, Af, S


def _form(v, form):
    """v is a squared quantity; 'squared' (library) or 'modulus' (canonical)."""
    if form not in ("squared", "modulus"):
        raise ValueError(form)
    return v if form == "squared" else np.sqrt(v)


def dtf(H, form="squared"):
    """Directed transfer function (Kaminski & Blinowska 1991):
    ``DTF_ij^2 = |H_ij|^2 / sum_m |H_im|^2`` (j -> i; rows sum to 1).
    Library = squared (SC/connectivity.py:758-779, inflow :1222-1231)."""
    a2 = np.abs(H) ** 2
    return _form(a2 / a2.sum(axis=-1, keepdims=True), form)


def directed_coherence(H, Sigma, form="squared"):
    """Directed coherence (Baccala et al. 1998): ``gamma_ij = sigma_j H_ij /
    sqrt(sum_m sigma_m^2 |H_im|^2)`` with ``sigma_m^2 = Sigma_mm``; squared form
    sums to 1 over j.  NOT what SC computes -- see
    ``directed_coherence_library_as_coded`` (that one agrees with SC to
    1e-11).  Form checked against the scanned paper (Baccala et al. 1998, Applied
    Signal Processing 5:40, Eq. 21-24 and Remark 6 on p. 43) in the spectral_directed
    check (audit/notes/spectral_directed.md, spectral.dcoh)."""
    s2 = np.real(np.diag(Sigma))
    a2 = s2[None, None, :] * np.abs(H) ** 2
    return _form(a2 / a2.sum(axis=-1, keepdims=True), form)


def directed_coherence_library_as_coded(H, Sigma):
    """Literal transcription of SC's directed_coherence (SC/connectivity.py:781-805):
    ``sqrt(nv) |H|^2 / sqrt(sum_j nv |H_ij|^2)`` where ``_get_noise_variance``
    (SC/connectivity.py:1234-1238) broadcasts ``Sigma_ii`` along the ROW (target)
    axis, so the noise variance cancels: result ``|H_ij|^2 / sqrt(sum_j |H_ij|^2)``
    (neither Baccala's squared nor modulus form; not scale invariant)."""
    nv = np.real(np.diag(Sigma))[None, :, None]
    a2 = np.abs(H) ** 2
    return np.sqrt(nv) * a2 / np.sqrt((nv * a2).sum(axis=-1, keepdims=True))


def pdc(A_f, form="squared"):
    """Partial directed coherence (Baccala & Sameshima 2001):
    ``PDC_ij = |A_ij| / sqrt(sum_m |A_mj|^2)`` (j -> i; columns sum to 1 when
    squared) with ``A(f) = I - sum A_k e^{-2 pi i f k}`` or ``inv(H)``.  Library =
    squared (SC/connectivity.py:807-844, outflow :1241-1250)."""
    a2 = np.abs(A_f) ** 2
    return _form(a2 / a2.sum(axis=-2, keepdims=True), form)


def gpdc(A_f, Sigma, form="squared"):
    """Generalised PDC (Baccala, Sameshima & Takahashi 2007):
    ``|A_ij| / sigma_i / sqrt(sum_m |A_mj|^2 / sigma_m^2)``.  Library = squared,
    same noise weighting (SC/connectivity.py:846-875)."""
    s2 = np.real(np.diag(Sigma))[None, :, None]
    a2 = np.abs(A_f) ** 2 / s2
    return _form(a2 / a2.sum(axis=-2, keepdims=True), form)


def partial_coherence(S, form="squared"):
    """Partial coherence from ``G = S^{-1}``: ``|G_ij|^2 / (G_ii G_jj)`` (Dahlhaus 2000)."""
    G = np.linalg.inv(S)
    d = np.real(np.diagonal(G, axis1=-2, axis2=-1))
    return _form(np.abs(G) ** 2 / (d[..., :, None] * d[..., None, :]), form)


def ddtf(H, S, form="squared"):
    """Direct DTF (Korzeniewska et al. 2003): ``dDTF_ij^2 = ffDTF_ij^2
    chi_ij^2`` with full-frequency DTF ``ffDTF_ij^2 = |H_ij|^2 / sum_f sum_m
    |H_im(f)|^2`` (sum over the frequencies supplied; SC supplies the
    (n_fft+1)//2 non-negative bins) and partial coherence chi^2 from S.
    SC instead multiplies by sqrt(PDC) -- see ``ddtf_library_as_coded``.
    UNVERIFIED: Korzeniewska 2003 not retrieved; only "ffDTF x partial
    coherence" is supported by secondary sources (ffDTF normalisation unchecked).
    The library-as-coded variant is the one that agrees with SC (to 1e-11)."""
    return _form(ddtf_ff(H) * partial_coherence(S), form)


def ddtf_ff(H):
    """Squared full-frequency DTF (Korzeniewska et al. 2003)."""
    a2 = np.abs(H) ** 2
    return a2 / a2.sum(axis=(-1, -3), keepdims=True)


def ddtf_library_as_coded(H):
    """SC's direct_directed_transfer_function (SC/connectivity.py:877-900):
    ``|ffDTF_ij| * sqrt(PDC_ij^2)`` with PDC from ``inv(H)`` (modulus form, PDC in
    place of Korzeniewska's partial coherence)."""
    return np.sqrt(ddtf_ff(H)) * pdc(np.linalg.inv(H), "modulus")


def granger_nonparametric(S, H, Sigma, nan_nonpositive=False):
    """Pairwise Geweke spectral Granger causality (Geweke 1982;
    nonparametric via Wilson, Dhamala et al. 2008): for target i, source j
    ``I_{j->i}(f) = ln( S_ii / (S_ii - (Sigma_jj - Sigma_ij^2 / Sigma_ii) |H_ij|^2) )``.
    Valid when H, Sigma come from the BIVARIATE (i, j) model.  Mirrors
    SC/connectivity.py:1173-1180 and :1193-1211; SC maps values <= 0 to NaN
    (``nan_nonpositive=True``).  Returns (n_f, P, P), [i, j] = j -> i, NaN diagonal.

    LOG FORM: ``ln S_ii - ln intr`` with S_ii taken from the supplied S and ``intr == 0``
    replaced by eps, exactly SC's ``_estimate_predictive_power`` (SC/connectivity.py:
    1173-1180).  nitime, which pyspi's parametric SGC calls, instead forms both terms from
    the fitted model, S_ii = auto + cross with intr = the auto term sigma |H_hat|^2 >= 0,
    and takes ``ln(S_ii / intr)`` (nitime/algorithms/autoregressive.py:495-499, :505-509).
    With S = H Sigma H^H the two forms agree wherever intr > 0 and differ where intr == 0
    (finite here, +inf in nitime).  With a supplied S that is not H Sigma H^H (a multitaper
    CSM with an inexact Wilson factor) intr can be negative, and this function returns NaN
    there.  refs/sgc.py:geweke_ratio is the nitime-style form (S from H and Sigma, ratio,
    no clipping)."""
    Sg = np.real(np.asarray(Sigma))
    v = np.diag(Sg)
    R = v[None, :] - Sg ** 2 / v[:, None]
    p = power(S)
    intr = p[..., :, None] - R[None] * np.abs(H) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        gc = np.log(p[..., :, None]) - np.log(np.where(intr == 0, np.finfo(float).eps, intr))
    if nan_nonpositive:
        gc[gc <= 0] = np.nan
    i = np.arange(S.shape[-1])
    gc[..., i, i] = np.nan
    return gc


def pairwise_spectral_granger(S_twosided, **kw):
    """Library-style loop (SC/connectivity.py:698-748): Wilson-factorise each
    2x2 sub-CSM, keep non-negative bins, apply ``granger_nonparametric``."""
    S = np.asarray(S_twosided)
    P = S.shape[-1]
    out = np.full(((S.shape[0] + 1) // 2, P, P), np.nan)
    for a in range(P):
        for b in range(a + 1, P):
            idx = np.array([a, b])
            sub = S[:, idx[:, None], idx[None, :]]
            H, Sg, _ = wilson_factorise(sub)
            g = granger_nonparametric(nonneg(sub), nonneg(H), Sg, **kw)
            out[:, a, b], out[:, b, a] = g[:, 0, 1], g[:, 1, 0]
    return out


# ------------------------------------------------------------------- wavelets
def morlet_wavelet(f, n_cycles, sfreq=1.0, zero_mean=True):
    """Complex Morlet wavelet (Tallon-Baudry 1997), mirroring
    ``mne.time_frequency.tfr.morlet`` (mne/time_frequency/tfr.py:138-178):
    ``sigma_t = n_cycles / (2 pi f)``, ``t = arange(0, 5 sigma_t, 1/sfreq)``
    mirrored about 0 (length ``2 ceil(5 sigma_t sfreq) - 1``), oscillation
    ``exp(2 pi i f t)`` minus ``exp(-2 (pi f sigma_t)^2)`` if zero_mean, Gaussian
    envelope, scaled to norm sqrt(2).  pyspi's generic wavelet SPIs (base class ``mne``)
    pass n_cycles = f/7 (pyspi/statistics/wavelet.py:54-55) -> sigma_t = 1/(14 pi) < 1
    sample.  PhaseSlopeIndex (wavelet.py:193-227) passes no n_cycles, so it gets
    mne_connectivity.phase_slope_index's default n_cycles = 7 (mne_connectivity/effective.py:37)."""
    sigma_t = n_cycles / (2.0 * np.pi * f)
    t = np.arange(0.0, 5.0 * sigma_t, 1.0 / sfreq)
    t = np.r_[-t[::-1], t[1:]]
    osc = np.exp(2j * np.pi * f * t)
    if zero_mean:
        osc = osc - np.exp(-2 * (np.pi * f * sigma_t) ** 2)
    W = osc * np.exp(-t ** 2 / (2.0 * sigma_t ** 2))
    return W / (np.sqrt(0.5) * np.linalg.norm(W))


def cwt_morlet(x, freqs, n_cycles, sfreq=1.0, zero_mean=True):
    """Continuous wavelet transform by FFT convolution, 'same' centring, as
    ``mne.time_frequency.tfr._cwt_gen`` (tfr.py:322-402; nfft >= T + W - 1,
    tfr.py:303-317; centring ``_centered`` tfr.py:4035-4043).
    x (P, T) -> (P, n_freqs, T) complex."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    T = x.shape[1]
    nc = np.broadcast_to(np.asarray(n_cycles, dtype=float), np.shape(freqs))
    Ws = [morlet_wavelet(f, c, sfreq, zero_mean) for f, c in zip(freqs, nc)]
    nfft = scipy.fft.next_fast_len(T + max(w.size for w in Ws) - 1)
    fx = np.fft.fft(x, nfft, axis=-1)
    out = np.empty((x.shape[0], len(Ws), T), dtype=complex)
    for k, W in enumerate(Ws):
        full = np.fft.ifft(fx * np.fft.fft(W, nfft), axis=-1)[:, :T + W.size - 1]
        s = (full.shape[1] - T) // 2
        out[:, k] = full[:, s:s + T]
    return out


# ===================================================================== selftest
def selftest() -> bool:
    sys.path.insert(0, CHECKS_DIR)
    import make_data as md

    res = []

    def check(label, ok, detail):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")

    rng = np.random.default_rng(20261014)
    # ---- grid / tapers
    try:
        import scipy.fftpack as fp
        ns = list(range(1, 300)) + [1021, 4097, 5000]
        bad = [n for n in ns if fp.next_fast_len(n) != next_fast_len_235(n)]
        check("next_fast_len_235 == scipy.fftpack.next_fast_len", not bad, f"{len(ns)} n, mismatches {bad[:5]}")
    except ImportError:
        check("next_fast_len_235 spot values", [next_fast_len_235(n) for n in (97, 1021)] == [100, 1024], "fftpack absent")
    w, lam = dpss_tapers(256, 3, 5)
    w2, _ = dpss_tapers(256, 3, 5, eig_min=0.95)
    check("dpss unit energy, low-bias rule", np.allclose((w ** 2).sum(1), 1, atol=1e-12) and w.shape[0] == 5
          and w2.shape[0] == 4, f"eig {np.round(lam, 4)}, K kept at 0.9/0.95: {w.shape[0]}/{w2.shape[0]}")
    pw = []
    for T_ in (8, 9, 64, 256):          # Percival & Walden 1993 p.390 concentration, as SC computes it
        wt, lt = dpss_tapers(T_, 3, 5, low_bias=False)
        r = np.array([[v[:T_ - k] @ v[k:] for k in range(T_)] for v in wt])
        filt = 4 * (3 / T_) * np.sinc(2 * (3 / T_) * np.arange(T_))
        filt[0] = 2 * 3 / T_
        pw.append(np.abs(r @ filt - lt).max())
    check("dpss ratios = P&W autocorrelation eigenvalues; T=8 keeps 5 (SC: 4, NaN taper)",
          max(pw) < 1e-10 and dpss_tapers(8, 3, 5)[0].shape[0] == 5,
          f"max |ratio - P&W| {max(pw):.1e} over T=8,9,64,256 (tol 1e-10)")
    x = rng.standard_normal((2, 300))
    c1, fr = multitaper_fft(x, 3, 1)
    xd = x - x.mean(1, keepdims=True)
    lhs = cross_spectral_matrix(c1).mean(0)
    w1 = dpss_tapers(300, 3, 1)[0][0]
    rhs = np.einsum("t,it,jt->ij", w1 ** 2, xd, xd).reshape(-1)
    check("Parseval scaling mean_f S = sum_t w^2 x_i x_j", np.allclose(lhs.reshape(-1).real, rhs, atol=1e-12)
          and c1.shape == (1, 300, 2) and len(nonneg(fr)) == 150 and np.isclose(fr[149], 149 / 300) and fr[150] < 0,
          f"max diff {np.abs(lhs.reshape(-1).real - rhs).max():.1e}; n_fft=300, nonneg bins 150, top f {fr[149]:.4f}")

    # ---- (1) coherency of phase-shifted sinusoids
    d = md.load("narrowband_f0-0.25", 4096)
    f0, ph = float(d["f0"]), d["phase_shift"]
    t = np.arange(4096.0)
    clean = np.stack([np.sin(2 * np.pi * f0 * t + p) for p in ph])
    for lab, data, tol in (("analytic noise-free", clean, 1e-6), ("dataset (noise 0.05)", d["x"], 5e-3)):
        co, fr = multitaper_fft(data)
        k0 = int(np.argmin(np.abs(fr - f0)))
        C = coherency(cross_spectral_matrix(co))[k0]
        err = max(abs(C[k, 0] - np.exp(1j * ph[k])) for k in range(3))
        coh = min(coherence_magnitude(cross_spectral_matrix(co))[k0][k, 0] for k in (1, 2))
        check(f"(1) C_k0(f0) = exp(i phi_k), |C|^2 = 1, {lab}", err < tol and 1 - coh < tol,
              f"max |C - e^(i phi)| {err:.2e}, 1-|C|^2 {1 - coh:.2e} (tol {tol:g}); bin f={fr[k0]}")
    # phase measures on perfectly phase-locked coefficients (analytic)
    K, nf, phi = 5, 64, 0.7
    X0 = rng.standard_normal((K, nf)) + 1j * rng.standard_normal((K, nf))
    co = np.stack([X0, X0 * rng.uniform(0.2, 3, (K, nf)) * np.exp(-1j * phi)], axis=-1)
    vals = {"PLV": phase_locking_value(co)[:, 0, 1], "PLI": phase_lag_index(co)[:, 0, 1],
            "wPLI": weighted_phase_lag_index(co)[:, 0, 1], "PPC": pairwise_phase_consistency(co)[:, 0, 1],
            "dPLI2": debiased_squared_pli(co)[:, 0, 1], "dwPLI2": debiased_squared_wpli(co)[:, 0, 1],
            "phase": coherence_phase(cross_spectral_matrix(co))[:, 0, 1] / phi,
            "imcoh/|C|": imaginary_coherence(cross_spectral_matrix(co), signed=True)[:, 0, 1]
            / (coherence_magnitude(cross_spectral_matrix(co), squared=False)[:, 0, 1] * np.sin(phi))}
    err = max(np.abs(v - 1).max() for v in vals.values())
    check("phase measures = 1 for locked coeffs (angle S_01 = +phi)", err < 1e-12, f"max |v-1| {err:.1e} over {list(vals)}")
    # matmul=True: exact Im S = 0 on the diagonal and for a bit-identical channel (SC's values there);
    # elsewhere equal to the elementwise default to rounding
    cm = np.concatenate([co, co[..., :1]], axis=-1)                # channel 2 = bit-identical copy of 0
    fns = (phase_lag_index, weighted_phase_lag_index, debiased_squared_pli, debiased_squared_wpli)
    tied = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], bool)
    mm = [fn(cm, matmul=True) for fn in fns]
    exp_tied = (0.0, 0.0, -1.0 / (K - 1), np.nan)
    ok_t = all(np.array_equal(v[:, tied], np.full(v[:, tied].shape, e), equal_nan=True) for v, e in zip(mm, exp_tied))
    e_off = max(np.abs(v[:, ~tied] - fn(cm)[:, ~tied]).max() for v, fn in zip(mm, fns))
    check("matmul=True: tied entries PLI 0, wPLI 0, dsPLI -1/(K-1), dswPLI NaN; untied = default", ok_t and e_off < 1e-12,
          f"tied entries as SC {ok_t}; untied max |matmul - default| {e_off:.1e}")
    # independent coefficients: debiased measures unbiased, identities vs brute force
    K, nf = 5, 20000
    co = rng.standard_normal((K, nf, 2)) + 1j * rng.standard_normal((K, nf, 2))
    ppc, plv = pairwise_phase_consistency(co)[:, 0, 1], phase_locking_value(co)[:, 0, 1]
    dpli, dwpli = debiased_squared_pli(co)[:, 0, 1], debiased_squared_wpli(co)[:, 0, 1]
    z = [abs(v.mean()) / (v.std() / np.sqrt(nf)) for v in (ppc, dpli, dwpli[np.isfinite(dwpli)])]
    check("PPC, dPLI^2, dwPLI^2 unbiased (mean 0) on independent coeffs", max(z) < 4,
          f"|mean|/SE = {np.round(z, 2)} (tol 4); mean PLV^2 {np.mean(plv ** 2):.4f} vs 1/K=0.2")
    th = np.angle(co[..., 0] * np.conj(co[..., 1]))[:, :50]
    im = np.imag(co[..., 0] * np.conj(co[..., 1]))[:, :50]
    jk = [(j, k) for j in range(K) for k in range(K) if j != k]
    ppc_bf = np.array([np.mean([np.cos(th[j, f] - th[k, f]) for j, k in jk]) for f in range(50)])
    dw_bf = np.array([sum(im[j, f] * im[k, f] for j, k in jk) / sum(abs(im[j, f] * im[k, f]) for j, k in jk)
                      for f in range(50)])
    e = max(np.abs(ppc_bf - ppc[:50]).max(), np.abs(dw_bf - dwpli[:50]).max())
    check("PPC / dwPLI^2 closed forms = pairwise definitions (Vinck 2010/2011)", e < 1e-12, f"max diff {e:.1e}")

    # ---- (2) multitaper CSM vs S_true (var2_bivariate, T=4096)
    d = md.load("var2_bivariate", 4096)
    A_list, Sig = [d["A1"], d["A2"]], d["Sigma"]
    _, _, S_chk = var_transfer_function(A_list, d["freqs"], Sig)
    check("var_transfer_function reproduces stored S_true", np.abs(S_chk - d["S_true"]).max() < 1e-12,
          f"max diff {np.abs(S_chk - d['S_true']).max():.1e}")
    co, fr = multitaper_fft(d["x"])
    Sh = cross_spectral_matrix(co)
    H_true, Af_true, S_an = var_transfer_function(A_list, fr, Sig)
    errs = []
    for lo in np.arange(0, 0.5, 0.1):
        m = band_mask(fr, lo, lo + 0.1, inclusive=False)
        a, b = Sh[m].mean(0), S_an[m].mean(0)
        errs.append(np.linalg.norm(a - b) / np.linalg.norm(b))
    check("(2) band-averaged multitaper CSM vs analytic S (5 bands of 0.1)", max(errs) < 0.1,
          f"rel Frobenius err {np.round(errs, 3)} (tol 0.1)")

    # ---- (3) Wilson factorisation of the exact two-sided S_true
    St = d["S_true"]                                              # f = k/1024, k = 0..512
    S2 = np.concatenate([St, np.conj(St[-2:0:-1])], axis=0)       # S(-f) = conj S(f)
    f2 = np.fft.fftfreq(1024)
    _, _, S2_an = var_transfer_function(A_list, f2, Sig)
    check("two-sided S_true assembly matches analytic grid", np.abs(S2 - S2_an).max() < 1e-12,
          f"max diff {np.abs(S2 - S2_an).max():.1e}")
    H_tr2 = var_transfer_function(A_list, f2)[0]
    for hv in (True, False):
        H, Sg, info = wilson_factorise(S2, halve_offdiag=hv)
        rec = np.abs(H @ Sg @ np.conj(np.swapaxes(H, -1, -2)) - S2).max() / np.abs(S2).max()
        sig_err = np.abs(np.diag(Sg) - np.diag(Sig)).max() / np.diag(Sig).min()
        herr = np.abs(H - H_tr2).max()
        check(f"(3) Wilson on S_true ({'library triu(g0/2)' if hv else 'Wilson split'})",
              rec < 1e-6 and sig_err < 0.05 and info["converged"],
              f"recon {rec:.1e} (tol 1e-6), Sigma rel err {sig_err:.1e} (tol 0.05), max|H-H_true| {herr:.1e}, "
              f"iters {info['n_iter']}, Sigma_offdiag {Sg[0, 1]:.1e}")
    # non-convergence / bad reconstruction must be visible in info (narrowband), singular lag-0 raises
    nb = []
    for T_ in (1024, 4096):
        co_nb, _ = multitaper_fft(md.load("narrowband_f0-0.25", T_)["x"])
        nb.append(wilson_factorise(cross_spectral_matrix(co_nb))[2])
    try:
        wilson_factorise(cross_spectral_matrix(multitaper_fft(np.ones((2, 256)))[0]))
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    check("Wilson info flags narrowband failure; constant series raises LinAlgError",
          nb[0]["converged"] and nb[0]["recon_err"] > 0.1 and not nb[1]["converged"] and raised,
          f"T=1024 converged {nb[0]['converged']} recon {nb[0]['recon_err']:.2f} (>0.1); T=4096 converged "
          f"{nb[1]['converged']} after {nb[1]['n_iter']}; LinAlgError {raised}")
    # Kolmogorov-Szego: log det Sigma = int log det S df; a K-taper CSM is complex Wishart(K)/K, so
    # E log det Sigma_hat = log det Sigma + sum_{i<P} digamma(K - i) - P log K (bias, not an error)
    from scipy.special import digamma
    _, Sgw, info = wilson_factorise(Sh)
    pred = digamma(co.shape[0]) + digamma(co.shape[0] - 1) - 2 * np.log(co.shape[0])
    obs = np.log(np.linalg.det(Sgw) / np.linalg.det(Sig))
    check("Wilson on multitaper CSM (T=4096): log det Sigma bias = Wishart prediction", abs(obs - pred) < 0.05,
          f"log det ratio {obs:.3f} vs predicted {pred:.3f} (tol 0.05); Sigma diag {np.round(np.diag(Sgw), 3)} "
          f"vs {np.diag(Sig)} (K=5 tapers bias Sigma low), iters {info['n_iter']}")
    # Granger: nonparametric (Wilson on S_true) vs analytic Geweke from true H, Sigma
    gc_np = pairwise_spectral_granger(S2)
    gc_an = granger_nonparametric(nonneg(S2), nonneg(H_tr2), Sig)
    e = np.nanmax(np.abs(gc_np - gc_an))
    check("Granger: Wilson path = analytic Geweke; x->y > 0, y->x = 0", e < 1e-6 and gc_an[:, 1, 0].min() > 0
          and np.abs(gc_an[:, 0, 1]).max() < 1e-12,
          f"max diff {e:.1e}; min GC x->y {gc_an[:, 1, 0].min():.3f}, max|GC y->x| {np.abs(gc_an[:, 0, 1]).max():.1e}")

    # ---- (4) DTF / PDC family from the true A
    Hn, Afn, Sn = H_true[: len(fr) // 2], Af_true[: len(fr) // 2], S_an[: len(fr) // 2]
    P_, D_ = pdc(Afn), dtf(Hn)
    check("(4) PDC(x->y)=PDC[1,0] > 0, PDC(y->x)=PDC[0,1] = 0", P_[:, 1, 0].min() > 0 and P_[:, 0, 1].max() == 0,
          f"min PDC[1,0] {P_[:, 1, 0].min():.3f}, max PDC[0,1] {P_[:, 0, 1].max():.1e}")
    e1 = np.abs(D_ - P_)[:, ~np.eye(2, dtype=bool)].max()
    e2 = max(np.abs(D_.sum(-1) - 1).max(), np.abs(P_.sum(-2) - 1).max())
    e3 = np.abs(pdc(np.linalg.inv(Hn)) - P_).max()
    check("2x2 identity DTF_ij = PDC_ij (i != j); row/column normalisation", max(e1, e2, e3) < 1e-12,
          f"|DTF-PDC| {e1:.1e}, norm err {e2:.1e}, PDC(inv H) vs PDC(A) {e3:.1e}")
    c = 2.7
    e4 = max(np.abs(gpdc(Afn, c * np.eye(2)) - P_).max(), np.abs(directed_coherence(Hn, c * np.eye(2)) - D_).max())
    G = np.conj(np.swapaxes(Afn, -1, -2)) @ np.linalg.inv(Sig) @ Afn
    e5 = np.abs(np.linalg.inv(Sn) - G).max() / np.abs(G).max()
    check("gPDC(cI)=PDC, DC(cI)=DTF, S^-1 = A^H Sigma^-1 A", max(e4, e5) < 1e-10, f"max diff {e4:.1e}, {e5:.1e}")
    dd = ddtf(Hn, Sn)
    ddl = ddtf_library_as_coded(Hn)
    check("dDTF[0,1] = 0 (canonical and library-as-coded); forms differ", dd[:, 0, 1].max() < 1e-20
          and ddl[:, 0, 1].max() < 1e-12 and np.abs(np.sqrt(dd) - ddl)[:, 1, 0].max() > 1e-4,
          f"max |sqrt(canon) - lib| on [1,0] {np.abs(np.sqrt(dd) - ddl)[:, 1, 0].max():.1e}")
    dcl = directed_coherence_library_as_coded(Hn, Sig)
    check("DC library-as-coded independent of Sigma (noise variance cancels)",
          np.abs(dcl - directed_coherence_library_as_coded(Hn, np.diag([5.0, 0.1]))).max() < 1e-12,
          f"max |DC_lib - DC canonical sq| {np.abs(dcl - directed_coherence(Hn, Sig)).max():.3f}")

    # ---- (5) PSI and (7) group delay on delayed_noise
    d = md.load("delayed_noise", 4096)
    co, fr = multitaper_fft(d["x"])
    Cn, fn = nonneg(coherency(cross_spectral_matrix(co))), nonneg(fr)
    pa = psi_adjacent(Cn, fn, 0.0, 0.5)
    pp = psi_all_pairs(Cn, fn, 0.0, 0.5)
    step = library_frequency_step(fn, 0.5 / 50)
    pps = psi_all_pairs(Cn, fn, 0.0, 0.5, step=step)
    ok = all(pa[0, k] > 0 and pa[k, 0] < 0 for k in (1, 2, 3)) and np.allclose(pa, -pa.T)
    diff = [abs(pp[0, k] - pa[0, k]) / abs(pa[0, k]) for k in (1, 2, 3)]
    check("(5) psi_adjacent[0,k] > 0 (x0 leads), antisymmetric; all-pairs differs", ok and min(diff) > 0.01,
          f"adjacent {np.round(pa[0, 1:], 3)}, all-pairs {np.round(pp[0, 1:], 1)}, "
          f"all-pairs step={step} {np.round(pps[0, 1:], 3)}")
    fa = np.arange(1, 60) / 128
    for dl in (1, 3, 5):
        Cd = np.exp(2j * np.pi * fa * dl)[:, None, None] * np.ones((1, 2, 2))
        ea = psi_adjacent(Cd, fa, 0, 1)[0, 0] - 58 * np.sin(2 * np.pi * dl / 128)
        bf = sum(np.sin(2 * np.pi * dl * (fa[b] - fa[a])) for a in range(59) for b in range(a + 1, 59))
        eb = psi_all_pairs(Cd, fa, 0, 1)[0, 0] - bf
        if abs(ea) > 1e-10 or abs(eb) > 1e-9:
            break
    check("PSI closed forms on pure-delay coherency (d=1,3,5)", abs(ea) < 1e-10 and abs(eb) < 1e-9,
          f"adjacent err {ea:.1e}, all-pairs vs brute-force pair loop err {eb:.1e}")
    fg, errs_ = np.arange(64) / 128, []
    Cg = np.ones((64, 2, 2), complex)
    for band, stp in (((0.2, 0.21), 1), ((0.3, 0.3), library_frequency_step(fg, 0.0)), ((0.1, 0.2), 20)):
        try:
            psi_all_pairs(Cg, fg, *band, step=stp)
        except ValueError:
            errs_.append(band)
    check("psi_all_pairs raises on <2 bins / step 0 (SC IndexError / ZeroDivisionError)", len(errs_) == 3,
          f"raised for {errs_}; normal band ok: {psi_all_pairs(Cg, fg, 0.1, 0.2).shape == (2, 2)}")
    gd = group_delay_fit(np.angle(np.exp(2j * np.pi * fa * 5.3)), fa)
    check("group_delay_fit on wrapped pure-delay phase: delay 5.3, r = 1", abs(gd["delay"] - 5.3) < 1e-10
          and abs(gd["r"] - 1) < 1e-12, f"delay {float(gd['delay']):.12f}, r {float(gd['r']):.12f}")
    m = band_mask(fn, 0, 0.5, inclusive=False)
    est = [float(group_delay_fit(np.angle(Cn[:, 0, k]), fn, m)["delay"]) for k in (1, 2, 3)]
    check("(7) group_delay_fit recovers delays 1,3,5 within 0.2", np.abs(np.array(est) - d["delay"][1:]).max() < 0.2,
          f"estimated {np.round(est, 4)}")

    # ---- (6) Morlet wavelet lengths and the degenerate pyspi setting
    fq = np.linspace(0.2, 0.5, 125)
    lens = [morlet_wavelet(f, f / 7.0).size for f in fq]
    check("(6) morlet n_cycles=f/7 has length 1 for all 125 freqs", set(lens) == {1}, f"lengths {set(lens)}")
    l7 = [morlet_wavelet(f, 7.0).size for f in fq]
    exp7 = [2 * int(np.ceil(5 * 7.0 / (2 * np.pi * f))) - 1 for f in fq]
    check("morlet n_cycles=7 lengths = 2 ceil(5 sigma_t) - 1", l7 == exp7 and l7[0] == 55 and l7[-1] == 23,
          f"f=0.2 -> {l7[0]}, f=0.5 -> {l7[-1]}, norms sqrt(2): "
          f"{np.allclose([np.linalg.norm(morlet_wavelet(f, 7.0)) for f in fq], np.sqrt(2))}")
    xs = d["x"][:1, :512]
    cw = cwt_morlet(xs, fq[:5], fq[:5] / 7)
    W7 = morlet_wavelet(0.25, 7.0)
    ref = np.convolve(xs[0], W7)[(W7.size - 1) // 2:(W7.size - 1) // 2 + 512]
    e = np.abs(cwt_morlet(xs, [0.25], 7.0)[0, 0] - ref).max()
    check("cwt: f/7 wavelets give sqrt(2)*x at every freq; n_cycles=7 = np.convolve 'same'",
          np.allclose(cw, np.sqrt(2) * xs[:, None, :], atol=1e-12) and e < 1e-10, f"conv err {e:.1e}")

    n_fail = res.count(False)
    print(f"selftest: {len(res) - n_fail}/{len(res)} passed")
    return n_fail == 0


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
