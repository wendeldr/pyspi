#!/usr/bin/env python3
"""References for the spectral_phase_sync family (COMPARISON venv; no pyspi, no spectral_connectivity).

Units: spectral.plv, spectral.pli, spectral.wpli, spectral.dspli, spectral.dswpli, spectral.ppc.
pyspi delegates all six to spectral_connectivity 1.1.2 (Multitaper + Connectivity); nothing here
imports that package.  Building blocks:

* ``_kernels_spectral.multitaper_fft`` (scipy.signal.windows.dpss + numpy.fft; mean removal,
  NW = 3, K = 5 low-bias tapers, n_fft = 5-smooth next_fast_len), restricted to the first
  (n_fft + 1) // 2 bins (DC kept, Nyquist dropped) as the library does.
* per-taper cross-spectra ``S_ij,k = X_i,k conj(X_j,k)`` formed ELEMENTWISE, followed by the
  sign-tie rule: ``Im S_ij,k`` is set to exactly 0 where ``|Im S_ij,k| <= 1e-12 max_k |S_ij,k|``.
  Reason (measured here, keys ``tie_*`` in results.json): the library forms S with a matmul
  (connectivity.py:1187-1190) and gets Im S exactly 0 on the diagonal and for bit-identical
  channels, whereas the elementwise product leaves rounding noise of order 1e-16..1e-15 there.
  At DC the elementwise Im S is already exactly 0.0 (the DC coefficient of real input is real),
  so the rule changes nothing at DC.
* the six estimators written out per bin from the papers, expectation over the K tapers:
  PLV = |mean_k S/|S||  (Lachaux et al. 1999);
  PLI = mean_k sign Im S (signed, as the library; Stam et al. 2007 take |.|);
  wPLI = mean_k Im S / mean_k |Im S| with the library's absolute-eps floor (denominator < eps -> 1)
         (Vinck et al. 2011 take |numerator|);
  dsPLI = (K PLI^2 - 1)/(K - 1) (Vinck et al. 2011 closed form) and, for the canonical check, the
          pairwise form 1/(K(K-1)) sum_{j != k} sign Im S_j sign Im S_k by explicit double loop;
  dswPLI = sum_{j != k} Im_j Im_k / sum_{j != k} |Im_j||Im_k| (0/0 -> NaN), closed and double loop;
  PPC = 2/(K(K-1)) sum_{j<k} cos(theta_j - theta_k) (Vinck et al. 2010), closed and double loop.
* external package: mne_connectivity 0.9.0 ``spectral_connectivity_epochs`` on an
  ``mne.time_frequency.EpochsSpectrumArray`` whose "epochs" are the 5 taper coefficient sets
  (no taper dimension -> mode 'fourier', unit weights), so mne's epoch expectation runs over the
  tapers exactly as the library's ``trials_tapers`` expectation does.  mne returns the UNSIGNED
  PLI/wPLI (epochs_bivariate.py:163-185, 231-268) and maps a zero dswPLI denominator to 0
  (:300-306) where the library returns NaN.
* analytic nulls: E|mean of K iid unit phasors| = (1/K) int_0^inf (1 - J0(t)^K)/t^2 dt
  (= 0.401632 for K = 5; asymptotic sqrt(pi/(4K)) = 0.396333); E = 0 for the sign/imaginary
  measures and PPC under independence.
* population model (var2_bivariate): the expectation of each K = 5 estimator when the 5 taper
  coefficient pairs are iid circular complex Gaussian with covariance S_true(f) (make_data's
  analytic VAR spectrum), by Monte Carlo; the sampling se of the band mean from re-simulating the
  VAR (make_data.simulate_var) 200 times and applying the reference estimator.
* same-lineage (pyspi self-consistency, no independent reference): the symmetry runs ``sym``
  (value = +-A.T of pyspi's own matrix), normalisation invariance ``inv`` (value = pyspi's
  kernel-path run), pairwise separability ``sep*`` (value = pyspi's P-channel sub-block) and
  ``detrend-effect`` (two pyspi runs).  The max-variant ``sym`` runs of PLI/wPLI are the
  exception: their pass flag tests A[j,i] = -min_f X_ij(f) against the independent per-bin
  reference and are labelled independent-impl.

Writes audit/checks/out/ref/<run_id>.npz for every run of manifest/spectral_phase_sync.json.
"""
import json
import sys
import warnings
from pathlib import Path

CHECKS = "/media/dan/Data/git/pyspi/audit/checks"
sys.path.insert(0, CHECKS)
sys.path.insert(0, CHECKS + "/refs")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.signal  # noqa: E402
from scipy import integrate, special  # noqa: E402

import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

FAMILY = "spectral_phase_sync"
OUT_REF = Path(CHECKS) / "out" / "ref"
OUT_PY = Path(CHECKS) / "out" / "pyspi"
SELF = "refs/spectral_phase_sync.py"
K = 5
TIE_REL = 1e-12
REP_TOL = 1e-10
MEASURE = {"spectral.plv": "plv", "spectral.pli": "pli", "spectral.wpli": "wpli",
           "spectral.dspli": "dspli", "spectral.dswpli": "dswpli", "spectral.ppc": "ppc"}
ANTISYM = ("pli", "wpli")
BANDS = {"fmin-0_fmax-0-5": (0.0, 0.5), "fmin-0_fmax-0-25": (0.0, 0.25), "fmin-0-25_fmax-0-5": (0.25, 0.5)}
MNE_METHOD = {"plv": "plv", "pli": "pli", "wpli": "wpli", "dspli": "pli2_unbiased",
              "dswpli": "wpli2_debiased", "ppc": "ppc"}


# ------------------------------------------------------------------ helpers
def parse_ident(ident):
    _, rest = ident.split("_multitaper_")
    stat, _, band = rest.split("_", 2)
    return stat, BANDS[band]


def load_x(ds, T, processes=None):
    x = np.asarray(make_data.load(ds, T)["x"], dtype=np.float64)
    return x if processes is None else x[list(processes)]


def preprocess(x, path):
    """pyspi Data: kernel = untouched; pipeline = zscore ddof=1; pipeline-detrend = linear detrend
    then zscore (pyspi/data.py:184-195)."""
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "pipeline-detrend":
        x = scipy.signal.detrend(x, axis=1)
    if path in ("pipeline", "pipeline-detrend"):
        x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)
    return x


def coeffs(x):
    """Independent multitaper coefficients on the library's retained grid: (K, nb, P), freqs (nb,)."""
    c, f = ks.multitaper_fft(x, NW=3.0)
    assert c.shape[0] == K, c.shape
    return ks.nonneg(c, 1), ks.nonneg(f)


def csm(c, tie=True):
    """Per-taper elementwise cross-spectra (K, nb, P, P) with the sign-tie rule; returns S, stats."""
    S = c[:, :, :, None] * np.conj(c[:, :, None, :])
    P = c.shape[2]
    im = np.abs(S.imag)
    off = ~np.eye(P, dtype=bool)
    stats = {"tie_dc_offdiag_max_abs_im": float(im[:, 0][:, off].max()) if P > 1 else 0.0,
             "tie_diag_max_abs_im": float(im[:, :, np.arange(P), np.arange(P)].max())}
    if tie:
        thr = TIE_REL * np.abs(S).max(axis=0, keepdims=True)
        z = im <= thr
        stats["tie_zeroed_offdiag_nonDC"] = int(z[:, 1:][:, :, off].sum())
        S = np.where(z, S.real + 0j, S)
    return S, stats


def perbin_measure(S, m):
    with np.errstate(invalid="ignore", divide="ignore"):
        im = S.imag
        if m == "plv":
            v = np.abs((S / np.abs(S)).mean(axis=0))
        elif m == "pli":
            v = np.sign(im).mean(axis=0)
        elif m == "wpli":
            num, den = im.mean(axis=0), np.abs(im).mean(axis=0)
            den = np.where(den < np.finfo(float).eps, 1.0, den)
            v = num / den
        elif m == "dspli":
            p = np.sign(im).mean(axis=0)
            v = (K * p ** 2 - 1.0) / (K - 1.0)
        elif m == "dswpli":
            s, s2, sa = im.sum(0), (im ** 2).sum(0), np.abs(im).sum(0)
            den = sa ** 2 - s2
            v = (s ** 2 - s2) / np.where(den == 0, np.nan, den)
        elif m == "ppc":
            z = (S / np.abs(S)).sum(axis=0)
            v = (np.abs(z) ** 2 - K) / (K ** 2 - K)
        else:
            raise ValueError(m)
    v = np.array(v, dtype=np.float64)
    i = np.arange(v.shape[1])
    v[:, i, i] = np.nan
    return v


def pairwise_forms(S, m):
    """Canonical pairwise definitions by explicit double loops over tapers j != k."""
    nb, P = S.shape[1], S.shape[2]
    num = np.zeros((nb, P, P))
    den = np.zeros((nb, P, P))
    for j in range(K):
        for k in range(K):
            if j == k:
                continue
            if m == "dspli":
                num += np.sign(S[j].imag) * np.sign(S[k].imag)
            elif m == "dswpli":
                num += S[j].imag * S[k].imag
                den += np.abs(S[j].imag) * np.abs(S[k].imag)
            elif m == "ppc":
                if j < k:
                    num += np.cos(np.angle(S[j]) - np.angle(S[k]))
    with np.errstate(invalid="ignore", divide="ignore"):
        if m == "dspli":
            v = num / (K * (K - 1))
        elif m == "dswpli":
            v = num / np.where(den == 0, np.nan, den)
        else:
            v = 2.0 * num / (K * (K - 1))
    i = np.arange(P)
    v[:, i, i] = np.nan
    return v


def band_reduce(pb, f, stat, fmin, fmax):
    mask = (f >= fmin) & (f <= fmax)
    fn = np.nanmean if stat == "mean" else np.nanmax
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = fn(pb[mask], axis=0)
    v = np.array(v, dtype=np.float64)
    np.fill_diagonal(v, np.nan)
    return v


_CACHE = {}


def ref_perbin(ds, T, path, m, processes=None, tie=True):
    key = (ds, T, path, tuple(processes) if processes else None, tie)
    if key not in _CACHE:
        x = preprocess(load_x(ds, T, processes), path)
        c, f = coeffs(x)
        S, st = csm(c, tie=tie)
        _CACHE[key] = (S, f, st, c)
    S, f, st, c = _CACHE[key]
    return perbin_measure(S, m), f, st


def pyspi_value(rid):
    p = OUT_PY / f"{rid}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return np.asarray(z["value"], dtype=np.float64)


def sym_stats(A):
    if A is None or A.ndim != 2:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"pyspi_max_abs_A_minus_AT": float(np.nanmax(np.abs(A - A.T))),
                "pyspi_max_abs_A_plus_AT": float(np.nanmax(np.abs(A + A.T))),
                "pyspi_min": float(np.nanmin(A)), "pyspi_max": float(np.nanmax(A))}


def save(rid, value, criterion, kind, detail, notes="", **kw):
    d = {"value": np.asarray(value, dtype=np.float64), "criterion": np.array(criterion),
         "reference_kind": np.array(kind), "reference_detail": np.array(detail), "notes": np.array(notes)}
    for k, v in kw.items():
        if v is None:
            continue
        d[k] = np.asarray(v)
    np.savez(OUT_REF / f"{rid}.npz", **d)


# ------------------------------------------------------------------ mne bridge
def mne_perbin(c, f, m, faverage_bands=None):
    import mne
    from mne.time_frequency import EpochsSpectrumArray
    from mne_connectivity import spectral_connectivity_epochs
    mne.set_log_level("ERROR")
    P = c.shape[2]
    data = np.transpose(c, (0, 2, 1))     # (epochs = tapers, channels, freqs)
    info = mne.create_info([f"c{i}" for i in range(P)], sfreq=1.0, ch_types="eeg")
    spec = EpochsSpectrumArray(data, info, freqs=f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if faverage_bands is None:
            con = spectral_connectivity_epochs(spec, method=MNE_METHOD[m], fmin=0.0, fmax=0.5, verbose=False)
            d = np.moveaxis(con.get_data(output="dense"), -1, 0)        # (nb, P, P) lower triangle
        else:
            fmins = tuple(b[0] for b in faverage_bands)
            fmaxs = tuple(b[1] for b in faverage_bands)
            con = spectral_connectivity_epochs(spec, method=MNE_METHOD[m], fmin=fmins, fmax=fmaxs,
                                               faverage=True, verbose=False)
            d = np.moveaxis(con.get_data(output="dense"), -1, 0)        # (n_bands, P, P)
    d = d + np.swapaxes(d, 1, 2)          # unsigned / symmetric measures: fill the upper triangle
    i = np.arange(P)
    d[:, i, i] = np.nan
    return d


def mne_version():
    import mne
    import mne_connectivity
    return f"mne=={mne.__version__}, mne_connectivity=={mne_connectivity.__version__}"


# ------------------------------------------------------------------ analytic nulls
def expected_abs_mean_phasor(Kn):
    """E|(1/K) sum_k exp(i theta_k)|, theta iid uniform: (1/K) int_0^inf (1 - J0(t)^K)/t^2 dt."""
    fn = lambda t: (1.0 - special.j0(t) ** Kn) / t ** 2  # noqa: E731
    edges = np.concatenate([[0.0], np.arange(1, 4001) * np.pi / 2])
    v = sum(integrate.quad(fn, lo, hi, limit=200, epsabs=1e-15)[0] for lo, hi in zip(edges[:-1], edges[1:]))
    return (v + 1.0 / edges[-1]) / Kn


def null_stat(x, m):
    S, _ = csm(coeffs(preprocess(x, "pipeline"))[0])
    pb = perbin_measure(S, m)
    nb, P = pb.shape[0], pb.shape[1]
    keep = (np.arange(nb) >= 6) & (np.arange(nb) <= nb - 7)
    iu = np.triu_indices(P, 1)
    return float(np.mean(pb[keep][:, iu[0], iu[1]])), pb[keep][:, iu[0], iu[1]]


# ------------------------------------------------------------------ population model
VAR = None


def var_model():
    global VAR
    if VAR is None:
        d = make_data.load("var2_bivariate", 4096)
        VAR = (np.asarray(d["A1"]), np.asarray(d["A2"]), np.asarray(d["Sigma"]))
    return VAR


def independent_taper_expectation(m, freqs, R=2000, seed=7):
    """E[estimator(f)] when the K taper coefficient pairs are iid CN(0, S_true(f))."""
    A1, A2, Sig = var_model()
    Sf = make_data.var_spectral_matrix([A1, A2], Sig, freqs)
    rng = np.random.default_rng(seed)
    out = np.empty((freqs.size, 2, 2))
    for lo in range(0, freqs.size, 128):
        hi = min(lo + 128, freqs.size)
        L = np.linalg.cholesky(Sf[lo:hi])                                    # (b, 2, 2)
        g = (rng.standard_normal((R, K, hi - lo, 2)) + 1j * rng.standard_normal((R, K, hi - lo, 2))) / np.sqrt(2)
        Z = np.einsum("bij,rkbj->rkbi", L, g)                                # (R, K, b, 2)
        S = Z[..., :, None] * np.conj(Z[..., None, :])                     # (R, K, b, 2, 2)
        S = np.swapaxes(S, 0, 1).reshape(K, R * (hi - lo), 2, 2)            # tapers first
        vals = perbin_measure(S, m).reshape(R, hi - lo, 2, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[lo:hi] = np.nanmean(vals, axis=0)
    return out, Sf


def var_sampling(m, stat, fmin, fmax, n_rep=200, seed=11):
    A1, A2, Sig = var_model()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        x = make_data.simulate_var([A1, A2], Sig, 4096, rng)
        c, f = coeffs(x)
        S, _ = csm(c)
        vals.append(band_reduce(perbin_measure(S, m), f, stat, fmin, fmax))
    return np.array(vals)


# ------------------------------------------------------------------ handlers
def h_rep(r, m, kind_note=""):
    stat, (fmin, fmax) = parse_ident(r["identifier"])
    pb, f, st = ref_perbin(r["dataset"], r["T"], r["path"], m)
    val = band_reduce(pb, f, stat, fmin, fmax)
    A = pyspi_value(r["run_id"])
    extra = sym_stats(A)
    if r["dataset"] == "shared_envelope" and A is not None:
        extra["pyspi_A02"] = float(A[0, 2])
        extra["pyspi_A20"] = float(A[2, 0])
        extra["ref_A02"] = float(val[0, 2])
    if r["dataset"] == "trending" and r["path"] == "pipeline":
        other = pyspi_value(r["run_id"].replace("|pipeline|", "|pipeline-detrend|"))
        if A is not None and other is not None:
            extra["detrend_effect_max_abs"] = float(np.nanmax(np.abs(A - other)))
    save(r["run_id"], val, "tolerance", "independent-impl",
         f"{SELF}:h_rep (scipy dpss + numpy fft multitaper, per-taper formulas, inclusive band mask, "
         f"nan{stat}); sign-tie rule {TIE_REL:g} relative",
         "exact replication from independent taper coefficients, both orientations", rtol=REP_TOL, atol=REP_TOL, **extra)


def h_perbin(r, m):
    pb, f, st = ref_perbin(r["dataset"], r["T"], r["path"], m)
    A = pyspi_value(r["run_id"])
    extra = dict(st)
    if A is not None:
        P = A.shape[1]
        off = ~np.eye(P, dtype=bool)
        dc = A[0][off]
        extra["pyspi_dc_unique"] = np.unique(np.round(dc[~np.isnan(dc)], 12)) if np.any(~np.isnan(dc)) else np.array([np.nan])
        extra["pyspi_dc_all_nan"] = bool(np.all(np.isnan(dc)))
        extra["pyspi_perbin_min"] = float(np.nanmin(A))
        if r["dataset"] == "shared_envelope":
            extra["pyspi_pair02_unique"] = np.unique(np.round(A[:, 0, 2][~np.isnan(A[:, 0, 2])], 12))[:16] \
                if np.any(~np.isnan(A[:, 0, 2])) else np.array([np.nan])
            pb_nt, _, _ = ref_perbin(r["dataset"], r["T"], r["path"], m, tie=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                extra["no_tie_rule_max_abs_diff_pair02"] = float(np.nanmax(np.abs(pb_nt[:, 0, 2] - A[:, 0, 2]))) \
                    if np.any(~np.isnan(pb_nt[:, 0, 2]) & ~np.isnan(A[:, 0, 2])) else float("nan")
                extra["no_tie_rule_nan_pattern_equal_pair02"] = bool(np.array_equal(np.isnan(pb_nt[:, 0, 2]), np.isnan(A[:, 0, 2])))
            S, _, _, c = _CACHE[(r["dataset"], r["T"], r["path"], None, True)]
            Sraw = c[:, :, 0] * np.conj(c[:, :, 2])
            extra["tie_identical_pair02_max_abs_im_elementwise"] = float(np.abs(Sraw.imag).max())
    save(r["run_id"], pb, "tolerance", "independent-impl",
         f"{SELF}:h_perbin (per-bin estimator from scipy dpss + numpy fft coefficients)",
         "per-bin replication, diagonal masked", rtol=REP_TOL, atol=REP_TOL, **extra)


def h_mne_perbin(r, m):
    x = preprocess(load_x(r["dataset"], r["T"]), r["path"])
    c, f = coeffs(x)
    d = mne_perbin(c, f, m)
    notes = "mne epochs = the 5 taper coefficient sets (kernel multitaper front end); unsigned forms"
    if m == "dswpli":
        n0 = int(np.sum(d[0] == 0))
        d[0] = np.where(np.isnan(d[0]), np.nan, np.nan)
        notes += f"; DC row set to NaN: mne maps the zero denominator to 0 ({n0} entries), the library to NaN"
    if m in ANTISYM:
        notes += "; pyspi side is |per-bin| (probe perbin_abs)"
    save(r["run_id"], d, "tolerance", "external-pkg",
         f"{mne_version()} spectral_connectivity_epochs(EpochsSpectrumArray of taper coefficients, method="
         f"'{MNE_METHOD[m]}')", notes, rtol=REP_TOL, atol=REP_TOL)


def h_mne_band(r, m):
    stat, band = parse_ident(r["identifier"])
    x = preprocess(load_x(r["dataset"], r["T"]), r["path"])
    c, f = coeffs(x)
    extra = {}
    if m == "dswpli":
        d = mne_perbin(c, f, m)
        d[0] = np.nan
        val = band_reduce(d, f, "mean", *band)
        notes = "mne per-bin wpli2_debiased, DC set NaN (library convention), nanmean over the inclusive band"
    else:
        val = mne_perbin(c, f, m, faverage_bands=[band])[0]
        notes = "mne faverage=True over the inclusive band (epochs.py:230 freqs >= fmin & <= fmax)"
    A = pyspi_value(r["run_id"])
    if A is not None and m in ANTISYM:
        notes += "; canonical band summary mean_f |X(f)| vs pyspi mean_f X(f) (signed)"
        pb, _, _ = ref_perbin(r["dataset"], r["T"], r["path"], m)
        absmean = np.abs(band_reduce(pb, f, "mean", *band))
        extra["abs_of_pyspi_equals_abs_signed_mean_max_diff"] = float(np.nanmax(np.abs(np.abs(A) - absmean)))
        extra["pyspi_A01"] = float(A[0, 1]); extra["mne_A01"] = float(val[0, 1])
        extra["pyspi_A03"] = float(A[0, 3]); extra["mne_A03"] = float(val[0, 3])
        extra["pyspi_offdiag_mean"] = float(np.nanmean(A)); extra["mne_offdiag_mean"] = float(np.nanmean(val))
    save(r["run_id"], val, "tolerance", "external-pkg",
         f"{mne_version()} spectral_connectivity_epochs(method='{MNE_METHOD[m]}', fmin={band[0]}, fmax={band[1]})",
         notes, rtol=REP_TOL, atol=REP_TOL, **extra)


def h_sym(r, m):
    A = pyspi_value(r["run_id"])
    stat, (fmin, fmax) = parse_ident(r["identifier"])
    extra = sym_stats(A)
    if r["dataset"] == "shared_envelope" and A is not None:
        extra["pyspi_A02"] = float(A[0, 2])
    if m in ANTISYM and stat == "max":
        pb, f, _ = ref_perbin(r["dataset"], r["T"], r["path"], m)
        mask = (f >= fmin) & (f <= fmax)
        pred_T = -np.nanmin(pb[mask], axis=0)      # A[j, i] = -min_f X_ij(f)
        np.fill_diagonal(pred_T, np.nan)
        ok_identity = bool(np.allclose(A.T[~np.isnan(A.T)], pred_T[~np.isnan(pred_T)], atol=1e-10, rtol=1e-10))
        asym = extra["pyspi_max_abs_A_plus_AT"] > 1e-12 and extra["pyspi_max_abs_A_minus_AT"] > 1e-12
        # pass_flag compares pyspi's transpose with -min_f of the independent per-bin reference
        save(r["run_id"], A, "qualitative", "independent-impl",
             f"{SELF}:h_sym (identity A[j,i] = -min_f X_ij(f) from the independent per-bin reference)",
             "max of a per-bin antisymmetric quantity: neither symmetric nor antisymmetric in general "
             "(a saturated all-ones matrix is symmetric); pass = the identity holds",
             pass_flag=int(ok_identity), identity_holds=ok_identity, asymmetric=asym, **extra)
        return
    sign = -1.0 if m in ANTISYM else 1.0
    save(r["run_id"], sign * A.T, "exact", "same-lineage",
         f"pyspi self-consistency, {SELF}:h_sym ({'A = -A.T' if sign < 0 else 'A = A.T'} from Im S_ji = -Im S_ij)",
         "value = pyspi's own transpose; property check, not an independent reference", rtol=0.0, atol=0.0, **extra)


def h_inv(r, m):
    other = r["run_id"].replace("|pipeline|inv|", "|kernel|rep|")
    B = pyspi_value(other)
    save(r["run_id"], B, "tolerance", "same-lineage",
         f"pyspi self-consistency, {SELF}:h_inv (positive per-channel scaling and offset invariance; pyspi kernel-path run {other})",
         "Data(normalise=True) vs Data(normalise=False) on scaled_offset", rtol=1e-12, atol=1e-12)


def h_sep(r, m):
    procs = r["processes"]
    full_id = f"{r['unit_id']}|{r['dataset']}|{r['T']}|kernel|rep|{r['run_id'].split('|')[-1]}"
    B = pyspi_value(full_id)
    sub = B[np.ix_(procs, procs)]
    save(r["run_id"], sub, "exact", "same-lineage",
         f"pyspi self-consistency, {SELF}:h_sep (pair (i,j) depends only on channels i and j; P-channel run {full_id})",
         "2-channel subset vs the P-channel matrix entries", rtol=0.0, atol=0.0)


NULL_CACHE = {}


def h_null(r, m):
    T = r["T"]
    n_rep = 300
    rng = np.random.default_rng(1234 + T)
    sims = np.array([null_stat(rng.standard_normal((5, T)), m)[0] for _ in range(n_rep)])
    se = float(sims.std(ddof=1))
    target = expected_abs_mean_phasor(K) if m == "plv" else 0.0
    _, perbin_vals = null_stat(rng.standard_normal((5, T)), m)
    A = pyspi_value(r["run_id"])
    save(r["run_id"], np.array(target), "statistical", "analytic",
         f"{SELF}:h_null (" + ("E|mean of 5 iid unit phasors| = (1/5) int_0^inf (1-J0(t)^5)/t^2 dt"
                               if m == "plv" else "E = 0 under independence (iid signs / symmetric Im)") +
         f"; se = sd of the same summary over {n_rep} fresh iid N(0,1) (5,{T}) draws with the reference estimator)",
         "statistic = mean over upper-triangle pairs and interior bins (6 bins from DC/Nyquist excluded)",
         se=np.array(se), mc_mean=float(sims.mean()), mc_se_of_mean=float(sims.std(ddof=1) / np.sqrt(n_rep)),
         asymptotic_sqrt_pi_over_4K=float(np.sqrt(np.pi / (4 * K))), analytic_target=float(target),
         z_pyspi=float((float(A) - target) / se) if A is not None else float("nan"))


def h_sat(r, m):
    T = r["T"]
    out = np.empty((30, 2))
    for s in range(30):
        z = np.random.default_rng(10000 + s).standard_normal((5, T))
        c, f = coeffs(preprocess(z, "pipeline"))
        S, _ = csm(c)
        A = band_reduce(perbin_measure(S, m), f, "max", 0.0, 0.5)
        off = ~np.eye(5, dtype=bool)
        out[s] = (np.mean(A[off] == 1.0), np.mean(A[off]))
    nb = (ks.next_fast_len_235(T) + 1) // 2
    p1 = {"pli": 1 / 32, "wpli": 1 / 32, "dspli": 1 / 16, "dswpli": 1 / 16}.get(m, 0.0)
    save(r["run_id"], out, "tolerance", "independent-impl",
         f"{SELF}:h_sat (same 30 seeds, reference estimator, full-band nanmax)",
         "columns: fraction of off-diagonal entries == 1.0, mean off-diagonal max", rtol=REP_TOL, atol=REP_TOL,
         ref_fraction_eq1_mean=float(out[:, 0].mean()), ref_fraction_eq1_min=float(out[:, 0].min()),
         ref_mean_max=float(out[:, 1].mean()), n_bins=nb,
         independent_bin_prob_eq1=float(1 - (1 - p1) ** nb) if p1 else 0.0,
         independent_bin_prob_eq1_nonDC=float(1 - (1 - p1) ** (nb - 1)) if p1 else 0.0)


SCALES = (1.0, 1e-6, 1e-8, 1e-10)


def h_scale(r, m, floor_reproduced):
    x = load_x(r["dataset"], r["T"])
    vals = []
    for s in SCALES:
        xs = x * s if floor_reproduced else x
        c, f = coeffs(xs)
        S, _ = csm(c)
        vals.append(band_reduce(perbin_measure(S, m), f, "mean", 0.0, 0.5))
    A = pyspi_value(r["run_id"])
    extra = {}
    if A is not None:
        extra = {f"pyspi_A01_scale_{s:g}": float(A[i, 0, 1]) for i, s in enumerate(SCALES)}
    if floor_reproduced:
        detail = f"{SELF}:h_scale (reference on x*s with the library absolute-eps floor reproduced)"
        notes = "reproduces the implemented scale dependence"
    else:
        detail = f"{SELF}:h_scale (canonical scale invariance: the s = 1 reference value for every s)"
        notes = "phase measures are invariant to a common positive scale; a failure is a deviation"
    save(r["run_id"], np.stack(vals), "tolerance", "analytic" if not floor_reproduced else "independent-impl",
         detail, notes, rtol=1e-9, atol=1e-12, **extra)


def h_pop(r, m):
    stat, (fmin, fmax) = parse_ident(r["identifier"])
    f = np.arange(2048) / 4096.0
    E, Sf = independent_taper_expectation(m, f)
    target = band_reduce(E, f, stat, fmin, fmax)
    sims = var_sampling(m, stat, fmin, fmax)
    se = sims.std(axis=0, ddof=1)
    np.fill_diagonal(se, np.nan)
    Esim = np.nanmean(sims, axis=0)
    A = pyspi_value(r["run_id"])
    C = Sf[:, 0, 1] / np.sqrt(Sf[:, 0, 0].real * Sf[:, 1, 1].real)
    rho = np.abs(C)
    pop_plv = (np.pi / 4) * rho * special.hyp2f1(0.5, 0.5, 2.0, rho ** 2)    # K -> infinity
    extra = {"target_01": float(target[0, 1]), "se_01": float(se[0, 1]), "sim_mean_01": float(Esim[0, 1]),
             "taper_dependence_bias_01": float(Esim[0, 1] - target[0, 1]),
             "taper_dependence_bias_z": float((Esim[0, 1] - target[0, 1]) / (se[0, 1] / np.sqrt(sims.shape[0]))),
             "band_mean_abs_coherency": float(rho.mean()),
             "population_plv_closed_form_band_mean": float(pop_plv.mean()),
             "population_plv_squared_band_mean": float((pop_plv ** 2).mean())}
    if A is not None:
        extra["pyspi_01"] = float(A[0, 1]); extra["pyspi_10"] = float(A[1, 0])
    save(r["run_id"], target, "statistical", "analytic",
         f"{SELF}:h_pop (E of the K=5 estimator under iid CN(0, S_true(f)) taper coefficients, 2000 MC draws "
         f"per bin; se = sd over 200 re-simulated VAR(2) series, reference estimator)",
         "population (independent-taper model) check on var2_bivariate T=4096", se=se, **extra)


def _nb_bins(ds, T):
    f0 = float(ds.split("-")[1])
    n = ks.next_fast_len_235(T)
    m0 = int(round(f0 * n))
    return f0, m0, n


NB_PHASE = (0.0, np.pi / 4, -np.pi / 2)


def h_nb_sign(r, m):
    A = pyspi_value(r["run_id"])
    f0, m0, n = _nb_bins(r["dataset"], r["T"])
    nb = (n + 1) // 2
    P = 3
    es = np.zeros((nb, P, P))
    for i in range(P):
        for j in range(P):
            if i != j:
                es[m0 - 1:m0 + 2, i, j] = np.sign(np.sin(NB_PHASE[i] - NB_PHASE[j]))
    pb, _, _ = ref_perbin(r["dataset"], r["T"], "kernel", m)
    extra = {}
    if A is not None:
        extra = {"pyspi_f0_01": float(A[m0, 0, 1]), "pyspi_f0_02": float(A[m0, 0, 2]), "pyspi_f0_12": float(A[m0, 1, 2]),
                 "pyspi_f0_10": float(A[m0, 1, 0]), "f0_bin_freq": m0 / n}
    save(r["run_id"], pb, "direction", "analytic",
         f"{SELF}:h_nb_sign (relative phase phi_i - phi_j at f0 with phase_shift (0, +pi/4, -pi/2): "
         "(0,1) -pi/4 -> -1, (0,2) +pi/2 -> +1, (1,2) +3pi/4 -> +1, transposes negated; bins m0-1..m0+1)",
         "value = independent per-bin reference; criterion uses expected_sign only", expected_sign=es, **extra)


def h_nb_lock(r, m):
    """Constant relative phase at f0 -> per-bin value ~1 where every taper carries the sinusoid.
    Re-derived after the first run: when f0 lies exactly on the grid (f0 * n_fft integer, here
    f0 = 0.25), the odd DPSS tapers (k = 1, 3) are antisymmetric, so their gain at zero frequency
    offset is sum_t w_k(t) = 0 and at the f0 bin they carry only noise and the negative-frequency
    image; the value at that single bin is then not ~1.  The check is applied at the neighbouring
    bins m0 +- 1 for on-grid f0 (all five tapers have non-zero gain there) and at m0 otherwise."""
    A = pyspi_value(r["run_id"])
    f0, m0, n = _nb_bins(r["dataset"], r["T"])
    on_grid = abs(f0 * n - round(f0 * n)) < 1e-9
    thr = {"plv": 0.99, "ppc": 0.98, "dspli": 0.98, "dswpli": 0.98}[m]
    off = ~np.eye(3, dtype=bool)
    bins = [m0 - 1, m0 + 1] if on_grid else [m0]
    v = np.concatenate([A[b][off] for b in bins])
    w, _ = ks.dpss_tapers(r["T"])
    odd_gain = float(np.abs(w[[1, 3]].sum(axis=1)).max() / np.abs(w[0].sum()))
    ok = bool(np.all(v >= thr)) and (not on_grid or odd_gain < 1e-9)
    save(r["run_id"], A, "qualitative", "analytic",
         f"{SELF}:h_nb_lock (constant relative phase at f0 -> per-bin value ~1; threshold {thr}; on-grid f0 "
         "checked at m0 +- 1 because the odd DPSS tapers have zero gain at the exact line frequency)",
         f"f0 bin {m0}/{n}, on_grid={on_grid}", pass_flag=int(ok), pyspi_min_checked=float(v.min()),
         threshold=thr, on_grid=on_grid, pyspi_min_at_m0=float(A[m0][off].min()),
         pyspi_max_at_m0=float(A[m0][off].max()), odd_taper_relative_dc_gain=odd_gain)


def _delay_sign(f, dd):
    return np.sign(np.sin(2 * np.pi * f * dd))


def h_dir(r, m):
    stat, (fmin, fmax) = parse_ident(r["identifier"])
    d = np.array([0, 1, 3, 5])
    T = r["T"]
    n = ks.next_fast_len_235(T)
    f = np.arange((n + 1) // 2) / n
    mask = (f >= fmin) & (f <= fmax)
    es = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                mm = _delay_sign(f[mask], d[j] - d[i]).mean()
                es[i, j] = np.sign(mm) if abs(mm) > 0.1 else 0.0
    pb, ff, _ = ref_perbin(r["dataset"], T, r["path"], m)
    A = pyspi_value(r["run_id"])
    extra = {}
    if A is not None:
        for k in (1, 2, 3):
            extra[f"pyspi_A0{k}"] = float(A[0, k])
            extra[f"mean_abs_perbin_0{k}"] = float(np.mean(np.abs(pb[mask, 0, k])))
    save(r["run_id"], band_reduce(pb, ff, stat, fmin, fmax), "direction", "analytic",
         f"{SELF}:h_dir (x_k = x0 delayed by d_k = (0,1,3,5): expected sign of the band mean = sign of "
         "mean_f sign(sin(2 pi f (d_j - d_i))) where |that mean| > 0.1, else unchecked)",
         "A[i,j] > 0 <=> i phase-leads j", expected_sign=es, **extra)


def h_dir_perbin(r, m):
    d = np.array([0, 1, 3, 5])
    T = r["T"]
    n = ks.next_fast_len_235(T)
    f = np.arange((n + 1) // 2) / n
    es = np.zeros((f.size, 4, 4))
    full = np.zeros((f.size, 4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                s = np.sin(2 * np.pi * f * (d[j] - d[i]))
                full[:, i, j] = np.sign(s)
                es[:, i, j] = np.where(np.abs(s) > 0.3, np.sign(s), 0.0)
    es[0] = 0
    full[0] = 0
    A = pyspi_value(r["run_id"])
    extra = {}
    if A is not None:
        msk = full != 0
        extra["agreement_all_nonDC_bins"] = float(np.mean(np.sign(A[msk]) == full[msk]))
        for k in (1, 2, 3):
            mk = full[:, 0, k] != 0
            extra[f"agreement_0{k}"] = float(np.mean(np.sign(A[mk, 0, k]) == full[mk, 0, k]))
    pb, _, _ = ref_perbin(r["dataset"], T, "kernel", m)
    save(r["run_id"], pb, "direction", "analytic",
         f"{SELF}:h_dir_perbin (per-bin sign(sin(2 pi f (d_j - d_i))) where |sin| > 0.3, DC excluded)",
         "A[i,j](f) > 0 <=> i phase-leads j at f", expected_sign=es, **extra)


def h_canon_pairwise(r, m):
    _ = ref_perbin(r["dataset"], r["T"], "kernel", m)
    S = _CACHE[(r["dataset"], r["T"], "kernel", None, True)][0]
    v = pairwise_forms(S, m)
    A = pyspi_value(r["run_id"])
    extra = {}
    if A is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dif = A - v
            extra["max_abs_diff_DC"] = float(np.nanmax(np.abs(dif[0]))) if np.any(~np.isnan(dif[0])) else float("nan")
            nd = dif[1:].copy()
            if r["dataset"] == "shared_envelope":
                extra["pair02_pyspi_unique"] = np.unique(A[:, 0, 2][~np.isnan(A[:, 0, 2])])[:16]
                extra["pair02_pairwise_unique"] = np.unique(v[:, 0, 2][~np.isnan(v[:, 0, 2])])[:16]
                nd[:, 0, 2] = np.nan; nd[:, 2, 0] = np.nan
            extra["max_abs_diff_nonDC_nonidentical"] = float(np.nanmax(np.abs(nd)))
    save(r["run_id"], v, "tolerance", "independent-impl",
         f"{SELF}:pairwise_forms (canonical pairwise definition, explicit double loop over tapers j != k)",
         "Vinck 2010/2011 pairwise estimator over the 5 tapers", rtol=REP_TOL, atol=REP_TOL, **extra)


def h_grid(r, m):
    n = ks.next_fast_len_235(r["T"])
    f = np.fft.fftfreq(n)[: (n + 1) // 2]
    lo = (f >= 0) & (f <= 0.25)
    hi = (f >= 0.25) & (f <= 0.5)
    save(r["run_id"], f, "exact", "analytic",
         f"{SELF}:h_grid (fftfreq(next_fast_len(T))[:(n+1)//2]: DC kept, Nyquist dropped)",
         "documented: 125 uniform bins on [4/T, fs/2]", rtol=0.0, atol=0.0,
         n_bins=int(f.size), first_bin=float(f[0]), last_bin=float(f[-1]), documented_n_bins=125,
         documented_f0=4.0 / r["T"], bins_in_both_bands=int(np.sum(lo & hi)),
         n_low=int(lo.sum()), n_high=int(hi.sum()))


def h_eeg(r, m):
    A = pyspi_value(r["run_id"])
    import importlib.util
    spec = importlib.util.spec_from_file_location("p", CHECKS + "/probes/spectral_phase_sync_probes.py")
    # the probe module imports nothing outside numpy at module level, so its edge tables can be read here
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    counts = np.zeros((2, 2, 10), dtype=int)
    for li, L in enumerate((512, 1024)):
        n = ks.next_fast_len_235(L)
        f = np.arange((n + 1) // 2) / n
        for vi in range(2):
            for b in range(10):
                lo, hi = (p.EEG_K[b][0] / 2048, p.EEG_K[b][1] / 2048) if vi == 0 else p.EEG_ROUNDED[b]
                counts[li, vi, b] = int(np.sum((f >= lo) & (f <= hi)))
    pred_nan = counts == 0
    ok = bool(np.array_equal(np.isnan(A), pred_nan))
    save(r["run_id"], A, "qualitative", "analytic",
         f"{SELF}:h_eeg (inclusive bin counts on the library grid; empty band -> nanmean of empty slice -> NaN)",
         "rows: L = 512, 1024; exact k/2048 edges vs .3g-rounded identifier edges",
         pass_flag=int(ok), counts_L512_exact=counts[0, 0], counts_L512_rounded=counts[0, 1],
         counts_L1024_exact=counts[1, 0], counts_L1024_rounded=counts[1, 1],
         pyspi_nan_L512_rounded=np.isnan(A[0, 1]).astype(int))


def main():
    OUT_REF.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(CHECKS) / "manifest" / f"{FAMILY}.json").read_text())
    n = 0
    errors = 0
    for r in man["runs"]:
        m = MEASURE[r["unit_id"]]
        tag = r["run_id"].split("|")[4]
        try:
            if tag == "rep":
                h_rep(r, m)
            elif tag == "detrend-effect":
                A = pyspi_value(r["run_id"])
                extra = {}
                for b in BANDS:
                    for st in ("mean", "max"):
                        ident = f"{r['run_id'].split('|')[0].split('.')[1]}_multitaper_{st}_fs-1_{b}"
                        a = pyspi_value(f"{r['unit_id']}|trending|256|pipeline|rep|{ident.split('_multitaper_')[1]}")
                        bb = pyspi_value(f"{r['unit_id']}|trending|256|pipeline-detrend|rep|{ident.split('_multitaper_')[1]}")
                        extra[f"max_abs_diff_{st}_{b}"] = float(np.nanmax(np.abs(a - bb)))
                save(r["run_id"], A, "qualitative", "same-lineage",
                     f"pyspi self-consistency, {SELF}:main detrend-effect (pyspi pipeline vs pipeline-detrend runs on trending)",
                     "the dependency removes only the channel mean; pass = values change by > 1e-3 in some identifier",
                     pass_flag=int(max(extra.values()) > 1e-3), **extra)
            elif tag == "perbin":
                h_perbin(r, m)
            elif tag == "mne-perbin":
                h_mne_perbin(r, m)
            elif tag == "mne-band":
                h_mne_band(r, m)
            elif tag == "sym":
                h_sym(r, m)
            elif tag == "inv":
                h_inv(r, m)
            elif tag.startswith("sep"):
                h_sep(r, m)
            elif tag == "null":
                h_null(r, m)
            elif tag == "sat":
                h_sat(r, m)
            elif tag == "scale-canon":
                h_scale(r, m, floor_reproduced=False)
            elif tag == "scale-floor":
                h_scale(r, m, floor_reproduced=True)
            elif tag == "pop":
                h_pop(r, m)
            elif tag == "nb-sign":
                h_nb_sign(r, m)
            elif tag == "nb-lock":
                h_nb_lock(r, m)
            elif tag == "dir":
                h_dir(r, m)
            elif tag == "dir-perbin":
                h_dir_perbin(r, m)
            elif tag == "canon-pairwise":
                h_canon_pairwise(r, m)
            elif tag == "grid":
                h_grid(r, m)
            elif tag == "eeg-bands":
                h_eeg(r, m)
            else:
                raise ValueError(f"unknown tag {tag}")
            n += 1
        except Exception as exc:  # report and continue; a missing npz shows up as 'no reference'
            errors += 1
            import traceback
            print(f"REF ERROR {r['run_id']}: {exc!r}\n{traceback.format_exc()[-1500:]}", file=sys.stderr)
    print(f"{FAMILY}: wrote {n} reference files, {errors} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
